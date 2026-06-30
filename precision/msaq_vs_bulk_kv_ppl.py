"""Head-to-head at matched bits: the deployed MXINT8-MSAQ (INT8 base + u-bit shared on the FIXED block
E8M0 scale) vs our bulk-scaled two-tier (MXFP base + u-bit shared on an ADAPTIVE per-group E{bw}M0
scale). Same logic (base + additive shared); the only conceptual difference is the residual's scale.
Which is more accurate at the same bits/elem? KV scope (the deployed MXINT8 / S3 vpack scope), K
rotated H128 + D-block, V T-block, applied identically to both schemes.

  MXINT8-MSAQ.efb : msaq_mxint8_efb(x,u,mg)      bits = (8-u) + u/mg + 8/32
  two-tier bulk   : quant_bw(x,2,1,u,mg,bw,MX±)  bits = 4 + 8/32 + [5/32 if MX+] + (bw+u)/mg
Run:
    HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 MSAQ_MODEL=NousResearch/Meta-Llama-3.1-8B \
        CUDA_VISIBLE_DEVICES=0 .venv/bin/python precision/msaq_vs_bulk_kv_ppl.py \
        > precision/msaq_vs_bulk_kv_llama31_8b.txt 2>&1
"""
import os, torch
import torch.nn.functional as F
from msaq_mxfp8_ppl import msaq_mxint8_efb, bits_mxint8, BLOCK
from two_tier_bulkbw_kv_ppl import quant_bw, bits as bits_bulk

DEV = "cuda"; MAXLEN, STRIDE, MW = 2048, 1024, 30


def hadamard(n):
    H = torch.ones(1, 1)
    while H.shape[0] < n:
        H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
    return H
HD = hadamard(128).to(DEV).to(torch.float32)


def msaq_int(x): return msaq_mxint8_efb(x, msaq_int.u, msaq_int.mg).to(x.dtype)   # set attrs per cfg

def make_int(u, mg):
    def kf(k): return ((msaq_mxint8_efb((k.float() @ HD), u, mg) @ HD.t()) / 128.0).to(k.dtype)
    def vf(v): return msaq_mxint8_efb(v.float().transpose(-1, -2), u, mg).transpose(-1, -2).to(v.dtype)
    return kf, vf

def make_bulk(u, mg, bw, mxp):
    cfg = (2, 1, u, mg, bw, mxp)
    def kf(k): return ((quant_bw(k.float() @ HD, *cfg) @ HD.t()) / 128.0).to(k.dtype)
    def vf(v): return quant_bw(v.float().transpose(-1, -2), *cfg).transpose(-1, -2).to(v.dtype)
    return kf, vf

_real = F.scaled_dot_product_attention
_C = {"on": False, "kf": None, "vf": None}
def _patch(q, k, v, *a, **kw):
    if _C["on"]: k = _C["kf"](k); v = _C["vf"](v)
    return _real(q, k, v, *a, **kw)
F.scaled_dot_product_attention = _patch


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
            if n >= MW or e == seq: break
        return torch.exp(torch.tensor(nll / ntok)).item()

    bf = ppl()
    print(f"wikitext-2: {ids.size(1):,} tok | BF16 PPL = {bf:.4f} | KV (rot+D / T-block) | MXINT8-MSAQ vs bulk\n", flush=True)

    def run(kf, vf):
        _C.update(on=True, kf=kf, vf=vf); p = ppl(); _C["on"] = False; return (p / bf - 1) * 100

    pts = []   # (bits, dPPL, scheme, label)
    print("--- MXINT8-MSAQ.efb (INT8 base + block-scale shared) ---", flush=True)
    for u in (2, 3, 4):
        for mg in (32, 16, 8, 4, 2):
            kf, vf = make_int(u, mg); d = run(kf, vf); b = bits_mxint8(u, mg)
            pts.append((b, d, "INT8-MSAQ", f"u{u}/mg{mg}"))
            print(f"  u{u}/mg{mg:<2} {b:>6.3f}b {d:>+6.2f}% {'OK' if d<=3 else ''}", flush=True)
    print("--- two-tier bulk (MXFP base + bulk-scale shared) ---", flush=True)
    for mxp in (False, True):
        for u in (1, 2, 3):
            for mg in (32, 16, 8):
                for bw in (2, 4):
                    kf, vf = make_bulk(u, mg, bw, mxp); d = run(kf, vf); b = bits_bulk(2, 1, u, mg, bw, mxp)
                    tag = "bulk+MX+" if mxp else "bulk"
                    pts.append((b, d, tag, f"u{u}/mg{mg}/bw{bw}"))
                    print(f"  {tag:>8} u{u}/mg{mg}/bw{bw} {b:>6.3f}b {d:>+6.2f}% {'OK' if d<=3 else ''}", flush=True)

    print("\n=== Pareto frontier (lowest dPPL at each bit budget; scheme tagged) ===", flush=True)
    best = sorted(pts)
    front = []
    for b, d, sch, lab in best:
        if not front or d < front[-1][1] - 1e-9:
            front.append((b, d, sch, lab))
    for b, d, sch, lab in front:
        print(f"  {b:>6.3f}b  {d:>+6.2f}%  {sch:>9}  {lab}", flush=True)
    print("\n=== matched-bit verdict (within ±0.06b bins) ===", flush=True)
    import math
    bins = {}
    for b, d, sch, lab in pts:
        key = round(b * 8) / 8
        bins.setdefault(key, []).append((d, sch, lab, b))
    for key in sorted(bins):
        winner = min(bins[key])
        schemes = {s for _, s, _, _ in bins[key]}
        if len(schemes) > 1:   # only bins where both schemes compete
            line = "  ".join(f"{s}:{min(d for d,ss,_,_ in bins[key] if ss==s):+.2f}%" for s in sorted(schemes))
            print(f"  ~{key:.3f}b | {line}  -> {winner[1]}", flush=True)


if __name__ == "__main__":
    main()
