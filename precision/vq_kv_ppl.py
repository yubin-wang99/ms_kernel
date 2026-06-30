"""INT-base vector-VQ residual on KV — PPL gate.

Strategic survivor from vq_residual.md: vector VQ is the best fractional-band residual quantizer, but on
WEIGHT it is double-capped (E2M3 wall per-bit; caveat-C2 ~2x compute on FP-native tensorcores). Both caps
lift on (INT4/INT6 base) x (KV scope): the per-element VQ LUT folds into the INT software-dequant as a
non-uniform remap (C2 escaped), and KV has no cheap E2M3 wall. This file runs the PPL gate there.

Setup mirrors kv_ladder_step1_ppl.py: SDPA is patched to quantize K/V on the fly.
  K: rotate head_dim D by H128 (per-channel outliers), INT-n MX D-block + vector-VQ residual, fold back.
  V: transpose tokens to last, INT-n MX T-block + vector-VQ residual along T.
Codebook: in-distribution per (layer, K/V), learned on the FIRST window and cached (an upper bound on a
real calibrated/global table; C3 deferred). Gate: ΔPPL <= 3% vs BF16 5.6877.

Run: HF_HUB_OFFLINE=1 MSAQ_MODEL=NousResearch/Meta-Llama-3.1-8B CUDA_VISIBLE_DEVICES=0 \
        .venv/bin/python precision/vq_kv_ppl.py > precision/vq_kv_ppl_llama31_8b.txt 2>&1
"""
import os, torch
import torch.nn.functional as F
from em_sharing import _base_encode, BASES
from mxfp6_verify import _fp6_grid

DEV = "cuda"; MAXLEN, STRIDE, MAX_WINDOWS = 2048, 1024, 30
BLOCK = 32


def hadamard(n):
    H = torch.ones(1, 1)
    while H.shape[0] < n:
        H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
    return H
HD = hadamard(128).to(DEV).float()


# ---- vector k-means (fit on a sample, cache codebook; assign chunked) ----
def vq_fit(X, K, iters=6, cap=50_000):
    N = X.shape[0]
    fit = X[torch.randint(0, N, (cap,), device=X.device)] if N > cap else X
    c = fit[torch.randint(0, fit.shape[0], (K,), device=X.device)].clone()
    for _ in range(iters):
        a = (-2 * fit @ c.t() + (c * c).sum(1)[None]).argmin(1)
        sums = torch.zeros_like(c).index_add_(0, a, fit)
        cnts = torch.zeros(K, device=X.device).index_add_(0, a, torch.ones_like(a, dtype=fit.dtype))
        nc = sums / cnts[:, None].clamp(min=1)
        c = torch.where(cnts[:, None] > 0, nc, c)
    return c


def vq_assign(X, c):
    out = torch.empty_like(X)
    for i in range(0, X.shape[0], 2_000_000):
        xb = X[i:i + 2_000_000]
        out[i:i + 2_000_000] = c[(-2 * xb @ c.t() + (c * c).sum(1)[None]).argmin(1)]
    return out


_CB = {}                                                  # codebook cache, reset per config
def base_vq_nd(x, base, g, K, key):
    """INT/FP MX base along last axis (padded to mult of 32) + vector-VQ residual (group g, codebook K)
    on the per-block-scaled residual (ulp=1 for INT). K=0 -> pure base (anchor)."""
    *lead, L = x.shape
    pad = (-L) % BLOCK
    xp = F.pad(x, (0, pad)) if pad else x
    xf = xp.reshape(-1, BLOCK).float()
    spec = BASES[base]
    s, snap = _base_encode(xf, spec)
    y = xf / s; code = snap(y)
    if K:
        # ulp-normalized residual (C6). INT: ulp=1; FP: local grid step (KV decode is GEMV/software
        # dequant for ALL bases, so the per-element VQ LUT folds for FP too — C2 escaped on KV).
        if spec["kind"] == "int":
            rn = (y - code)
            ulp = 1.0
        else:
            grid = _fp6_grid(spec["eb"], spec["mb"]); steps = torch.diff(grid)
            idx = torch.bucketize(code.abs(), grid, right=False).clamp(0, grid.numel() - 2)
            ulp = steps[idx]
            rn = (y - code) / ulp
        Xg = rn.reshape(-1, g)
        if key not in _CB:
            _CB[key] = vq_fit(Xg, K)
        corr = vq_assign(Xg, _CB[key]).reshape(code.shape)
        code = code + corr * ulp
    out = (code * s).reshape(*lead, L + pad)
    return (out[..., :L] if pad else out).to(x.dtype)


def q_K(k, base, g, K, layer):
    kr = k.float() @ HD
    q = base_vq_nd(kr, base, g, K, ("k", layer))
    return ((q @ HD.t()) / 128.0).to(k.dtype)


def q_V(v, base, g, K, layer):
    vq = base_vq_nd(v.transpose(-1, -2), base, g, K, ("v", layer))
    return vq.transpose(-1, -2).to(v.dtype)


# ---- SDPA patch (tracks layer index via call counter mod n_layer) ----
_real = F.scaled_dot_product_attention
_C = {"on": False, "fn": None, "i": 0, "nl": 32}
def _patch(q, k, v, *a, **kw):
    if _C["on"]:
        L = _C["i"] % _C["nl"]; _C["i"] += 1
        k = _C["fn"](k, "k", L); v = _C["fn"](v, "v", L)
    return _real(q, k, v, *a, **kw)
F.scaled_dot_product_attention = _patch


@torch.no_grad()
def ppl(model, ids):
    seq = ids.size(1); nll, ntok, prev, n = 0.0, 0, 0, 0
    for b in range(0, seq, STRIDE):
        e = min(b + MAXLEN, seq); trg = e - prev
        inp = ids[:, b:e].to(DEV); tgt = inp.clone(); tgt[:, :-trg] = -100
        nll += model(inp, labels=tgt).loss.double().item() * trg; ntok += trg; prev = e; n += 1
        if n >= MAX_WINDOWS or e == seq: break
    return torch.exp(torch.tensor(nll / ntok)).item()


def bits(base, g, K):
    n = (1 + BASES[base]["eb"] + BASES[base]["mb"]) if BASES[base]["kind"] == "fp" else BASES[base]["n"]
    return n + 8.0 / BLOCK + ((K.bit_length() - 1) / g if K else 0.0)


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from datasets import load_dataset
    MODEL = os.environ.get("MSAQ_MODEL", "meta-llama/Llama-3.1-8B-Instruct")
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16,
                                                 attn_implementation="sdpa").to(DEV).eval()
    _C["nl"] = model.config.num_hidden_layers
    try: _wt = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    except Exception: _wt = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    ids = tok("\n\n".join(t for t in _wt["text"] if t.strip()), return_tensors="pt").input_ids
    bf = ppl(model, ids)
    print(f"wikitext-2: {ids.size(1):,} tok | BF16 PPL = {bf:.4f} | {MAX_WINDOWS}w | "
          f"scope=KV (INT-base vector-VQ residual)\n", flush=True)

    def run(base, g, K):
        _CB.clear(); _C.update(on=True, fn=lambda t, w, L: (q_K if w == "k" else q_V)(t, base, g, K, L),
                               i=0)
        p = ppl(model, ids); _C["on"] = False
        return p, (p / bf - 1) * 100

    # (label, base, g, K)  K=0 -> anchor (pure base). INT = asked-for survivor; FP+VQ = the KV-correct
    # variant (KV decode is GEMV/software-dequant for all bases, so FP also folds the VQ LUT).
    CFGS = [
        ("FP4 native",        "FP4",  0, 0),
        ("INT4 zero",         "INT4", 0, 0),
        ("INT4+VQ g8/K16",    "INT4", 8, 16),
        ("INT4+VQ g8/K256",   "INT4", 8, 256),
        ("INT4+VQ g4/K256",   "INT4", 4, 256),
        ("FP4+VQ g8/K16",     "FP4",  8, 16),
        ("FP4+VQ g8/K256",    "FP4",  8, 256),
        ("FP4+VQ g4/K256",    "FP4",  4, 256),
        ("FP6 native",        "FP6",  0, 0),
        ("INT6 zero",         "INT6", 0, 0),
        ("INT6+VQ g8/K256",   "INT6", 8, 256),
        ("FP6+VQ g8/K16",     "FP6",  8, 16),
    ]
    print(f"{'config':>18} {'b/elem':>7} | {'PPL':>9} {'dPPL%':>8}  gate(<=3%)", flush=True)
    for nm, base, g, K in CFGS:
        p, d = run(base, g, K)
        gate = "PASS" if d <= 3.0 else ""
        print(f"{nm:>18} {bits(base, g, K):>7.3f} | {p:>9.4f} {d:>+7.2f}%  {gate}", flush=True)


if __name__ == "__main__":
    main()
