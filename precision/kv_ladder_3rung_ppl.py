"""KV ladder — 3-rung HYBRID allocation, using the regime-split finding (msaq_vs_bulk): two-tier
(MXFP4 base) owns low bits, MXINT8-MSAQ (INT8 base) owns mid bits, E2M3 owns high. A per-layer
allocation that can pick the regime-appropriate FORMAT per layer should beat the 2-rung (two-tier-low
+ E2M3-high) curve, because sensitive-but-not-extreme layers can use the INT8 MID rung instead of
jumping all the way to E2M3.

  LOW  = two-tier MX+ E2M1, no residual   (4.406b, +2.39% uniform)
  MID  = MXINT8-MSAQ.efb u3/mg32          (5.344b, +1.41%)
  HIGH = E2M3 native                       (6.250b, +0.30%)
All on K (rot+D-block) / V (T-block). Probe each layer's cost at LOW and at MID (rest HIGH); assign
rungs greedily by cost-per-bit-saved under a byte budget; validate joint PPL; compare to 2-rung.
Run:
    HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 MSAQ_MODEL=NousResearch/Meta-Llama-3.1-8B \
        CUDA_VISIBLE_DEVICES=0 .venv/bin/python precision/kv_ladder_3rung_ppl.py \
        > precision/kv_ladder_3rung_llama31_8b.txt 2>&1
"""
import os, torch
import torch.nn.functional as F
from msaq_mxfp8_ppl import msaq_mxint8_efb, BLOCK
from two_tier_bulkbw_kv_ppl import quant_bw, bits as bits_bulk
from msaq_mxfp8_ppl import bits_mxint8

DEV = "cuda"; MAXLEN, STRIDE = 2048, 1024
PROBE_WIN = int(os.environ.get("PROBE_WIN", "12")); FULL_WIN = int(os.environ.get("FULL_WIN", "30"))


def hadamard(n):
    H = torch.ones(1, 1)
    while H.shape[0] < n:
        H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
    return H
HD = hadamard(128).to(DEV).to(torch.float32)

# --- the three rungs as (kf, vf) quantizers, bits ---
def _bulk_K(k): return ((quant_bw(k.float() @ HD, 2, 1, 0, 32, 2, True) @ HD.t()) / 128.0).to(k.dtype)
def _bulk_V(v): return quant_bw(v.float().transpose(-1, -2), 2, 1, 0, 32, 2, True).transpose(-1, -2).to(v.dtype)
def _int_K(k):  return ((msaq_mxint8_efb(k.float() @ HD, 3, 32) @ HD.t()) / 128.0).to(k.dtype)
def _int_V(v):  return msaq_mxint8_efb(v.float().transpose(-1, -2), 3, 32).transpose(-1, -2).to(v.dtype)
def _e2m3_K(k): return ((quant_bw(k.float() @ HD, 2, 3, 0, 32, 2, False) @ HD.t()) / 128.0).to(k.dtype)
def _e2m3_V(v): return quant_bw(v.float().transpose(-1, -2), 2, 3, 0, 32, 2, False).transpose(-1, -2).to(v.dtype)

RUNG = {"L": (_bulk_K, _bulk_V, bits_bulk(2, 1, 0, 32, 2, True)),   # 4.406
        "M": (_int_K, _int_V, bits_mxint8(3, 32)),                  # 5.344
        "H": (_e2m3_K, _e2m3_V, bits_bulk(2, 3, 0, 32, 2, False))}  # 6.250
B = {r: RUNG[r][2] for r in RUNG}

_real = F.scaled_dot_product_attention
_CUR = {"l": -1}; ASSIGN = {}
def _patch(q, k, v, *a, **kw):
    r = ASSIGN.get(_CUR["l"], "H")
    kf, vf, _ = RUNG[r]
    if ASSIGN: k = kf(k); v = vf(v)
    return _real(q, k, v, *a, **kw)
F.scaled_dot_product_attention = _patch


@torch.no_grad()
def ppl(model, ids, nwin):
    seq = ids.size(1); nll, ntok, prev, n = 0.0, 0, 0, 0
    for b in range(0, seq, STRIDE):
        e = min(b + MAXLEN, seq); trg = e - prev
        inp = ids[:, b:e].to(DEV); t = inp.clone(); t[:, :-trg] = -100
        nll += model(inp, labels=t).loss.double().item() * trg; ntok += trg; prev = e; n += 1
        if n >= nwin or e == seq: break
    return torch.exp(torch.tensor(nll / ntok)).item()


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from datasets import load_dataset
    MODEL = os.environ.get("MSAQ_MODEL", "meta-llama/Llama-3.1-8B-Instruct")
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, attn_implementation="sdpa").to(DEV).eval()
    attn = [(int(n.split(".")[2]), m) for n, m in model.named_modules() if n.endswith("self_attn")]
    NL = len(attn)
    for i, m in attn: m.register_forward_pre_hook(lambda mod, a, kw, i=i: _CUR.update(l=i), with_kwargs=True)
    try: _wt = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    except Exception: _wt = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    ids = tok("\n\n".join(t for t in _wt["text"] if t.strip()), return_tensors="pt").input_ids

    global ASSIGN
    bf = ppl(model, ids, FULL_WIN)
    print(f"wikitext-2: {ids.size(1):,} tok | BF16 PPL = {bf:.4f} | 3-rung hybrid | "
          f"L={B['L']:.3f} M={B['M']:.3f} H={B['H']:.3f} b\n", flush=True)

    # cost of putting layer L at LOW or MID (rest HIGH)
    ASSIGN = {i: "H" for i in range(NL)}; base = ppl(model, ids, PROBE_WIN)
    cL, cM = {}, {}
    for L in range(NL):
        ASSIGN = {i: "H" for i in range(NL)}; ASSIGN[L] = "L"; cL[L] = (ppl(model, ids, PROBE_WIN) / base - 1) * 100
        ASSIGN = {i: "H" for i in range(NL)}; ASSIGN[L] = "M"; cM[L] = (ppl(model, ids, PROBE_WIN) / base - 1) * 100
    print("per-layer cost  L(4.406b)  M(5.344b)  [rest H]:", flush=True)
    print("  " + " ".join(f"{L}:{cL[L]:+.2f}/{cM[L]:+.2f}" for L in range(NL)), flush=True)

    # greedy assignment for a target avg bytes: start all-H, downgrade by cost-per-bit-saved.
    # each layer has two downgrade options H->M (save H-M bits, cost cM) and H->L (save H-L, cost cL).
    def assign_for(target):
        a = {i: "H" for i in range(NL)}
        # candidate downgrades (layer, to_rung, dbits, dcost) ; pick lowest cost/bit greedily
        while True:
            avg = sum(B[a[i]] for i in range(NL)) / NL
            if avg <= target: break
            best = None
            for L in range(NL):
                if a[L] == "H":
                    for to, c in (("M", cM[L]), ("L", cL[L])):
                        eff = c / (B["H"] - B[to]);
                        if best is None or eff < best[0]: best = (eff, L, to)
                elif a[L] == "M":
                    eff = (cL[L] - cM[L]) / (B["M"] - B["L"])
                    if best is None or eff < best[0]: best = (eff, L, "L")
            if best is None: break
            _, L, to = best; a[L] = to
        return a

    print(f"\n3-rung curve (FULL {FULL_WIN}w):", flush=True)
    print(f"{'target':>7} {'avg b':>7} | {'PPL':>9} {'dPPL%':>8} | {'#L/#M/#H':>9}", flush=True)
    for target in (6.25, 5.875, 5.5, 5.125, 4.75, 4.406):
        a = assign_for(target); ASSIGN = a
        avg = sum(B[a[i]] for i in range(NL)) / NL
        nL = sum(v == "L" for v in a.values()); nM = sum(v == "M" for v in a.values()); nH = sum(v == "H" for v in a.values())
        p = ppl(model, ids, FULL_WIN)
        print(f"{target:>7.3f} {avg:>7.3f} | {p:>9.4f} {(p/bf-1)*100:>+7.2f}% | {nL}/{nM}/{nH}", flush=True)


if __name__ == "__main__":
    main()
