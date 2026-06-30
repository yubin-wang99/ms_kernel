"""K/V asymmetric format: so far K and V got the same rung, but they differ (K needs rotation + MX+
helps it more; V is T-blocked and MX+ helps it less). Sweep K-rung x V-rung independently over
{L=two-tier 4.406b, M=MXINT8-MSAQ 5.344b, H=E2M3 6.25b} (uniform across layers). If an asymmetric
(K!=V) combo sits below the symmetric points on the avg-bytes-vs-PPL frontier, K and V should be
allocated separately. avg b/elem = (B_K + B_V)/2.
Run:
    HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 MSAQ_MODEL=NousResearch/Meta-Llama-3.1-8B \
        CUDA_VISIBLE_DEVICES=0 .venv/bin/python precision/kv_asym_ppl.py \
        > precision/kv_asym_llama31_8b.txt 2>&1
"""
import os, torch
import torch.nn.functional as F
from msaq_mxfp8_ppl import msaq_mxint8_efb, bits_mxint8
from two_tier_bulkbw_kv_ppl import quant_bw, bits as bits_bulk

DEV = "cuda"; MAXLEN, STRIDE, MW = 2048, 1024, 30


def hadamard(n):
    H = torch.ones(1, 1)
    while H.shape[0] < n:
        H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
    return H
HD = hadamard(128).to(DEV).to(torch.float32)

# K quantizers (rotate + D-block); V quantizers (T-block). L/M/H rungs.
KQ = {"L": lambda k: ((quant_bw(k.float() @ HD, 2, 1, 0, 32, 2, True) @ HD.t()) / 128.0).to(k.dtype),
      "M": lambda k: ((msaq_mxint8_efb(k.float() @ HD, 3, 32) @ HD.t()) / 128.0).to(k.dtype),
      "H": lambda k: ((quant_bw(k.float() @ HD, 2, 3, 0, 32, 2, False) @ HD.t()) / 128.0).to(k.dtype)}
VQ = {"L": lambda v: quant_bw(v.float().transpose(-1, -2), 2, 1, 0, 32, 2, True).transpose(-1, -2).to(v.dtype),
      "M": lambda v: msaq_mxint8_efb(v.float().transpose(-1, -2), 3, 32).transpose(-1, -2).to(v.dtype),
      "H": lambda v: quant_bw(v.float().transpose(-1, -2), 2, 3, 0, 32, 2, False).transpose(-1, -2).to(v.dtype)}
RB = {"L": bits_bulk(2, 1, 0, 32, 2, True), "M": bits_mxint8(3, 32), "H": bits_bulk(2, 3, 0, 32, 2, False)}

_real = F.scaled_dot_product_attention
_C = {"on": False, "k": "H", "v": "H"}
def _patch(q, k, v, *a, **kw):
    if _C["on"]: k = KQ[_C["k"]](k); v = VQ[_C["v"]](v)
    return _real(q, k, v, *a, **kw)
F.scaled_dot_product_attention = _patch


@torch.no_grad()
def ppl(model, ids):
    seq = ids.size(1); nll, ntok, prev, n = 0.0, 0, 0, 0
    for b in range(0, seq, STRIDE):
        e = min(b + MAXLEN, seq); trg = e - prev
        inp = ids[:, b:e].to(DEV); t = inp.clone(); t[:, :-trg] = -100
        nll += model(inp, labels=t).loss.double().item() * trg; ntok += trg; prev = e; n += 1
        if n >= MW or e == seq: break
    return torch.exp(torch.tensor(nll / ntok)).item()


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from datasets import load_dataset
    MODEL = os.environ.get("MSAQ_MODEL", "meta-llama/Llama-3.1-8B-Instruct")
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, attn_implementation="sdpa").to(DEV).eval()
    try: _wt = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    except Exception: _wt = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    ids = tok("\n\n".join(t for t in _wt["text"] if t.strip()), return_tensors="pt").input_ids
    bf = ppl(model, ids)
    print(f"wikitext-2: {ids.size(1):,} tok | BF16 PPL = {bf:.4f} | K/V asymmetric | "
          f"L={RB['L']:.3f} M={RB['M']:.3f} H={RB['H']:.3f} b\n", flush=True)
    rows = []
    print(f"{'K':>2} {'V':>2} | {'avg b':>6} | {'dPPL%':>7}", flush=True)
    for kr in ("L", "M", "H"):
        for vr in ("L", "M", "H"):
            _C.update(on=True, k=kr, v=vr); p = ppl(model, ids); _C["on"] = False
            avg = (RB[kr] + RB[vr]) / 2; d = (p / bf - 1) * 100
            rows.append((avg, d, kr, vr, RB[kr], RB[vr]))
            print(f"{kr:>2} {vr:>2} | {avg:>6.3f} | {d:>+6.2f}%", flush=True)
    print("\n=== sorted by avg bytes; * = on the Pareto frontier ===", flush=True)
    best = sorted(rows); front = []
    for avg, d, kr, vr, bk, bv in best:
        par = not front or d < front[-1] - 1e-9
        if par: front.append(d)
        print(f"  {avg:.3f}b  {d:>+6.2f}%  K={kr}({bk:.2f}) V={vr}({bv:.2f}) {'*' if par else ''}"
              f"{'  <- ASYMMETRIC' if kr != vr and par else ''}", flush=True)


if __name__ == "__main__":
    main()
