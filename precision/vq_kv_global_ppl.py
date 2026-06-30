"""FP4 + vector-VQ residual on KV — generalization gate with a FIXED (calibration-learned) codebook.

vq_kv_results.md showed FP4+VQ passes the KV PPL gate and beats mantissa-share iso-bit, but the VQ
codebook was learned IN-DISTRIBUTION on the eval data (an upper bound; caveat C3 + calibration asymmetry).
This file closes that: learn the codebook on wikitext-2 TRAIN (disjoint), apply FIXED to wikitext-2 TEST.

Three regimes per config:
  indist   — fit on the eval tensor (per layer, cached on first eval window)  [the prior upper bound]
  perlayer — fixed per-(layer,K/V) codebook learned on calibration            [data generalization]
  global   — fixed single codebook per K/V, shared across ALL layers, calib   [does one table suffice? C3]

Run: HF_HUB_OFFLINE=1 MSAQ_MODEL=NousResearch/Meta-Llama-3.1-8B CUDA_VISIBLE_DEVICES=2 \
        .venv/bin/python precision/vq_kv_global_ppl.py > precision/vq_kv_global_llama31_8b.txt 2>&1
"""
import os, torch
import torch.nn.functional as F
from em_sharing import _base_encode, BASES
from mxfp6_verify import _fp6_grid
from vq_kv_ppl import hadamard, vq_fit, vq_assign, HD

DEV = "cuda"; MAXLEN, STRIDE = 2048, 1024
EVAL_WINDOWS, CALIB_WINDOWS = 30, 8
BLOCK = 32
BASE = "FP4"


def fp_residual_nd(x):
    """FP base residual along last axis (padded to mult of 32). Returns parts to reconstruct +
    per-element normalized residual rn (=(y-code)/ulp)."""
    *lead, L = x.shape
    pad = (-L) % BLOCK
    xp = F.pad(x, (0, pad)) if pad else x
    xf = xp.reshape(-1, BLOCK).float()
    spec = BASES[BASE]
    s, snap = _base_encode(xf, spec)
    y = xf / s; code = snap(y)
    grid = _fp6_grid(spec["eb"], spec["mb"]); steps = torch.diff(grid)
    idx = torch.bucketize(code.abs(), grid, right=False).clamp(0, grid.numel() - 2)
    ulp = steps[idx]
    rn = (y - code) / ulp
    return (lead, L, pad, s, code, ulp), rn


def reconstruct(parts, rn, x_dtype):
    lead, L, pad, s, code, ulp = parts
    out = ((code + rn * ulp) * s).reshape(*lead, L + pad)
    return (out[..., :L] if pad else out).to(x_dtype)


# ---- collection (calibration) ----
BUF = {}                                                   # (kv, layer, g) -> [groups, g] sample
CAP = 40_000
def collect(rn, kv, layer, gs):
    for g in gs:
        Xg = rn.reshape(-1, g)
        n = Xg.shape[0]
        take = min(n, CAP)
        sel = Xg[torch.randint(0, n, (take,), device=Xg.device)]
        key = (kv, layer, g)
        if key in BUF:
            cur = BUF[key]
            merged = torch.cat([cur, sel], 0)
            if merged.shape[0] > CAP:
                merged = merged[torch.randperm(merged.shape[0], device=merged.device)[:CAP]]
            BUF[key] = merged
        else:
            BUF[key] = sel


# ---- SDPA patch (collect | apply) ----
_real = F.scaled_dot_product_attention
_C = {"mode": "off", "i": 0, "nl": 32, "g": 8, "K": 0, "cb": None, "indist": {}}
def _kv_quant(t, kv, layer):
    src = t.float()
    if kv == "k":
        z = src @ HD
        parts, rn = fp_residual_nd(z)
    else:
        z = src.transpose(-1, -2)
        parts, rn = fp_residual_nd(z)
    g, K = _C["g"], _C["K"]
    if _C["mode"] == "collect":
        collect(rn, kv, layer, (4, 8))
        return t                                            # calib forward unquantized
    if K:
        Xg = rn.reshape(-1, g)
        if _C["mode"] == "global":
            c = _C["cb"][kv]
        elif _C["mode"] == "perlayer":
            c = _C["cb"][(kv, layer)]
        else:                                               # indist: fit on eval tensor, cache per layer
            ck = (kv, layer)
            if ck not in _C["indist"]:
                _C["indist"][ck] = vq_fit(Xg, K)
            c = _C["indist"][ck]
        rn = vq_assign(Xg, c).reshape(rn.shape)
    zq = reconstruct(parts, rn, src.dtype)
    if kv == "k":
        return ((zq.float() @ HD.t()) / 128.0).to(t.dtype)
    return zq.transpose(-1, -2).to(t.dtype)


def _patch(q, k, v, *a, **kw):
    if _C["mode"] != "off":
        L = _C["i"] % _C["nl"]; _C["i"] += 1
        k = _kv_quant(k, "k", L); v = _kv_quant(v, "v", L)
    return _real(q, k, v, *a, **kw)
F.scaled_dot_product_attention = _patch


@torch.no_grad()
def ppl(model, ids, nwin):
    seq = ids.size(1); nll, ntok, prev, n = 0.0, 0, 0, 0
    for b in range(0, seq, STRIDE):
        e = min(b + MAXLEN, seq); trg = e - prev
        inp = ids[:, b:e].to(DEV); tgt = inp.clone(); tgt[:, :-trg] = -100
        nll += model(inp, labels=tgt).loss.double().item() * trg; ntok += trg; prev = e; n += 1
        if n >= nwin or e == seq: break
    return torch.exp(torch.tensor(nll / ntok)).item()


def bits(g, K):
    spec = BASES[BASE]; n = 1 + spec["eb"] + spec["mb"]
    return n + 8.0 / BLOCK + ((K.bit_length() - 1) / g if K else 0.0)


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from datasets import load_dataset
    MODEL = os.environ.get("MSAQ_MODEL", "meta-llama/Llama-3.1-8B-Instruct")
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16,
                                                 attn_implementation="sdpa").to(DEV).eval()
    _C["nl"] = model.config.num_hidden_layers
    def _ids(split):
        try: d = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
        except Exception: d = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split=split)
        return tok("\n\n".join(t for t in d["text"] if t.strip()), return_tensors="pt").input_ids
    test_ids, train_ids = _ids("test"), _ids("train")

    _C["mode"] = "off"
    bf = ppl(model, test_ids, EVAL_WINDOWS)
    print(f"wikitext-2 TEST: {test_ids.size(1):,} tok | BF16 PPL = {bf:.4f} | {EVAL_WINDOWS}w | "
          f"FP4-base vector-VQ residual; codebook calib on TRAIN ({CALIB_WINDOWS}w)\n", flush=True)

    # ---- calibration: collect FP4 residual groups on TRAIN, fit fixed codebooks ----
    _C.update(mode="collect", i=0)
    ppl(model, train_ids, CALIB_WINDOWS)
    _C["mode"] = "off"
    print(f"calibration collected: {len(BUF)} (kv,layer,g) buffers, "
          f"~{BUF[('k',0,8)].shape[0]} groups each\n", flush=True)

    def build_cb(g, K):
        per = {(kv, L): vq_fit(BUF[(kv, L, g)], K) for kv in ("k", "v") for L in range(_C["nl"])}
        glob = {}
        for kv in ("k", "v"):
            pool = torch.cat([BUF[(kv, L, g)] for L in range(_C["nl"])], 0)
            pool = pool[torch.randperm(pool.shape[0], device=pool.device)[:120_000]]
            glob[kv] = vq_fit(pool, K)
        return per, glob

    CFGS = [(8, 16), (8, 256), (4, 256)]
    print(f"{'config':>14} {'b/elem':>7} | {'indist':>16} {'perlayer':>16} {'global':>16}", flush=True)
    print(f"{'':>14} {'':>7} | {'PPL  dPPL%':>16} {'PPL  dPPL%':>16} {'PPL  dPPL%':>16}", flush=True)
    for g, K in CFGS:
        per, glob = build_cb(g, K)
        res = {}
        for mode, cb in (("indist", None), ("perlayer", per), ("global", glob)):
            _C.update(mode=mode, i=0, g=g, K=K, cb=cb, indist={})
            p = ppl(model, test_ids, EVAL_WINDOWS); _C["mode"] = "off"
            res[mode] = (p, (p / bf - 1) * 100)
        def cell(m): return f"{res[m][0]:.4f} {res[m][1]:+.2f}%"
        print(f"{f'FP4+VQ g{g}/K{K}':>14} {bits(g, K):>7.3f} | "
              f"{cell('indist'):>16} {cell('perlayer'):>16} {cell('global'):>16}", flush=True)


if __name__ == "__main__":
    main()
