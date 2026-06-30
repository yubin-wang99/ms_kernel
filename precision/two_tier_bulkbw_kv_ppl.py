"""Sweep the RESIDUAL-SCALE bit-width. two-tier's per-group shared residual rides its own power-of-2
scale; so far that scale was a full E8M0 (8 exponent bits). But the residual is bounded (it's a
correction), so a SMALLER exponent field (E4M0, E2M0) should suffice and save bits. Sweep
(u, mg, bulk_bw) = [2,3,4] x [2,4,8,16,32] x [2,4,8] on KV (rotation + MX+, DC residual) to find the
LOWEST bits/elem that stays within 3% PPL of BF16.

Per group of mg: one E{bulk_bw}M0 scale + one u-bit signed shared int (anchored at k_top=maxexp,
window [k_top-2^bulk_bw+1, k_top]). bits = (1+eb+mb) + 8/32[base scale] + 5/32[MX+] + (bulk_bw+u)/mg.
bulk_bw=8 reproduces the old E8M0 scheme.
Run:
    HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 MSAQ_MODEL=NousResearch/Meta-Llama-3.1-8B \
        CUDA_VISIBLE_DEVICES=0 .venv/bin/python precision/two_tier_bulkbw_kv_ppl.py \
        > precision/two_tier_bulkbw_kv_llama31_8b.txt 2>&1
Selftest: .venv/bin/python precision/two_tier_bulkbw_kv_ppl.py --selftest
"""
import os, sys, torch
import torch.nn.functional as F
from mxfp6_verify import _fp6_grid
from msaq_mxfp8_ppl import BLOCK
from two_tier_ppl import _snap_grid
from two_tier_mxplus_ppl import _mxplus_snap

DEV = "cuda"; MAXLEN, STRIDE, MAX_WINDOWS = 2048, 1024, 30


def quant_bw(x, eb, mb, u, mg, bulk_bw, mxplus=True, Hblk=None, efb_iters=2):
    """two-tier with a per-group E{bulk_bw}M0 residual scale. x=[...,K], block=32 / mg.
    Hblk=[G,32,32] -> A-weighted (§4) per-group optimum (uses gs x gs diagonal sub-blocks); None -> DC."""
    *lead, K = x.shape
    assert K % BLOCK == 0 and BLOCK % mg == 0
    G = K // BLOCK; nsg = BLOCK // mg
    grid = _fp6_grid(eb, mb); maxval = grid[-1].item()
    maxexp = ((1 << eb) - 1) - ((1 << (eb - 1)) - 1)
    xf = x.to(torch.float32).reshape(-1, G, BLOCK)
    absmax = xf.abs().amax(-1, keepdim=True).clamp(min=1e-30)
    s = torch.exp2(torch.floor(torch.log2(absmax)) - float(maxexp))
    y = xf / s

    def snap(t):
        if mxplus: return _mxplus_snap(t, eb, mb, maxexp, grid, maxval)
        return torch.sign(t) * _snap_grid(t.abs().clamp(max=maxval), grid)

    base = snap(y)
    if u > 0:
        if Hblk is not None:
            H = Hblk.to(torch.float32).reshape(G, nsg, mg, nsg, mg)
            i = torch.arange(nsg, device=x.device)
            Hd = H[:, i, :, i, :].permute(1, 0, 2, 3).contiguous()    # [G,nsg,mg,mg]
            denom = Hd.sum((-1, -2)).clamp(min=1e-30); Hsum = Hd.sum(-2)
        qmax = (1 << (u - 1)) - 1
        k_top = float(maxexp); k_lo = k_top - float((1 << bulk_bw) - 1)
        for it in range(max(1, efb_iters + 1)):
            r = (y - base).reshape(-1, G, nsg, mg)
            r_cont = r.mean(-1) if Hblk is None else torch.einsum("gpk,bgpk->bgp", Hsum, r) / denom
            kstar = torch.floor(torch.log2((r_cont.abs() / max(qmax, 1)).clamp(min=1e-30)))
            d = torch.exp2(kstar.clamp(k_lo, k_top))                 # E{bulk_bw}M0
            si = torch.round(r_cont / d).clamp(-qmax - 1, qmax)
            shared = (si * d).unsqueeze(-1).expand(-1, G, nsg, mg).reshape(-1, G, BLOCK)
            if it == efb_iters: break
            base = snap(y - shared)
        base = base + shared
    return (base * s).reshape(x.shape).to(x.dtype)


def bits(eb, mb, u, mg, bulk_bw, mxplus=True):
    b = (1 + eb + mb) + 8.0 / BLOCK + (5.0 / BLOCK if mxplus else 0.0)
    return b + (bulk_bw + u) / mg if u else b


# ---- KV harness (K rotated H128 + D-block, V T-block) ----
def hadamard(n):
    H = torch.ones(1, 1)
    while H.shape[0] < n:
        H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
    return H
HD = hadamard(128).to(DEV).to(torch.float32)

def q_K(k, cfg): return ((quant_bw(k.float() @ HD, *cfg) @ HD.t()) / 128.0).to(k.dtype)
def q_V(v, cfg): return quant_bw(v.transpose(-1, -2).float(), *cfg).transpose(-1, -2).to(v.dtype)

_real = F.scaled_dot_product_attention
_C = {"on": False, "cfg": None}
def _patch(q, k, v, *a, **kw):
    if _C["on"]:
        k = q_K(k, _C["cfg"]); v = q_V(v, _C["cfg"])
    return _real(q, k, v, *a, **kw)
F.scaled_dot_product_attention = _patch


def selftest():
    torch.manual_seed(0); dev = DEV
    W = torch.randn(256, 4096, device=dev) * 0.02
    def qsnr(x, xq):
        e = (x - xq).double().pow(2).mean(); return 10 * torch.log10(x.double().pow(2).mean() / e.clamp(min=1e-45)).item()
    print("E2M1+u3 mg32 — bulk_bw effect (QSNR vs bits):")
    for bw in (8, 4, 2):
        q = quant_bw(W, 2, 1, 3, 32, bw)
        print(f"  bulk_bw={bw} ({bits(2,1,3,32,bw):.3f}b): {qsnr(W,q):.2f} dB")
    print("bulk_bw=8 should ~= old E8M0; smaller bw trades range for bits.")


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from datasets import load_dataset
    MODEL = os.environ.get("MSAQ_MODEL", "meta-llama/Llama-3.1-8B-Instruct")
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, attn_implementation="sdpa").to(DEV).eval()
    try: _wt = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    except Exception: _wt = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    ids = tok("\n\n".join(t for t in _wt["text"] if t.strip()), return_tensors="pt").input_ids

    @torch.no_grad()
    def ppl():
        seq = ids.size(1); nll, ntok, prev, n = 0.0, 0, 0, 0
        for b in range(0, seq, STRIDE):
            e = min(b + MAXLEN, seq); trg = e - prev
            inp = ids[:, b:e].to(DEV); t = inp.clone(); t[:, :-trg] = -100
            nll += model(inp, labels=t).loss.double().item() * trg; ntok += trg; prev = e; n += 1
            if n >= MAX_WINDOWS or e == seq: break
        return torch.exp(torch.tensor(nll / ntok)).item()

    bf = ppl()
    print(f"wikitext-2: {ids.size(1):,} tok | BF16 PPL = {bf:.4f} | KV (rot+MX+), bulk_bw sweep\n", flush=True)
    rows = []
    # extended grid: u1 added, MX+ on/off (push below 4.5b)
    UU = [int(x) for x in os.environ.get("US", "1,2,3,4").split(",")]
    MGS = [int(x) for x in os.environ.get("MGS", "32,16,8").split(",")]
    BWS = [int(x) for x in os.environ.get("BWS", "2,4").split(",")]
    print(f"{'mx+':>3} {'u':>2} {'mg':>3} {'bw':>3} | {'b/elem':>7} | {'dPPL%':>7} {'<3%':>4}", flush=True)
    for mxp in (True, False):
        for u in UU:
            for mg in MGS:
                for bw in BWS:
                    cfg = (2, 1, u, mg, bw, mxp)
                    _C.update(on=True, cfg=cfg); p = ppl(); _C["on"] = False
                    d = (p / bf - 1) * 100; b = bits(*cfg)
                    rows.append((b, d, u, mg, bw, mxp))
                    print(f"{('Y' if mxp else 'n'):>3} {u:>2} {mg:>3} {bw:>3} | {b:>7.3f} | {d:>+6.2f}% {'OK' if d <= 3 else '':>4}", flush=True)
    ok = sorted([r for r in rows if r[1] <= 3.0])
    print("\n=== within 3%, sorted by bits (lowest first) ===", flush=True)
    for b, d, u, mg, bw, mxp in ok[:10]:
        print(f"  {b:.3f} b/elem : u{u}/mg{mg}/bw{bw}/{'MX+' if mxp else 'noMX+'}  {d:+.2f}%", flush=True)
    if ok:
        b, d, u, mg, bw, mxp = ok[0]
        print(f"\nLOWEST bits within 3%: {b:.3f} b/elem  (u{u}/mg{mg}/bulk_bw{bw}/{'MX+' if mxp else 'noMX+'}, {d:+.2f}%)", flush=True)


if __name__ == "__main__":
    if "--selftest" in sys.argv: selftest()
    else: main()
