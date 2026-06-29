"""KV ladder — head-wise allocation (finer than Step-2's per-layer). Does (layer,head) granularity
beat per-layer at matched bytes? 8 KV heads x 32 layers = 256 cells; a full per-cell probe is costly,
so rank cells by a SEPARABLE sensitivity sens(L,h) = layer_sens[L] + head_sens[h]:
  - head_sens[h]: head h cheap across ALL layers, rest E2M3 (8 probes).
  - layer_sens[L]: layer L (all heads) cheap, rest E2M3 (32 probes) — same as Step 2 but re-measured
    here for consistency.
Then greedily cheap the lowest-sens cells, and measure the ACTUAL joint PPL at a few byte points;
compare to the per-layer curve at matched average bytes. cheap = MX+ E2M1+u3 gs32 (4.75b), E2M3 (6.25).
Run:
    HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 MSAQ_MODEL=NousResearch/Meta-Llama-3.1-8B \
        CUDA_VISIBLE_DEVICES=0 .venv/bin/python precision/kv_ladder_headwise_ppl.py \
        > precision/kv_ladder_headwise_llama31_8b.txt 2>&1
"""
import os, torch
import torch.nn.functional as F
from msaq_mxfp8_ppl import BLOCK
from two_tier_gs_sweep_ppl import quant, bits

DEV = "cuda"; MAXLEN, STRIDE = 2048, 1024
PROBE_WIN = int(os.environ.get("PROBE_WIN", "12"))
FULL_WIN = int(os.environ.get("FULL_WIN", "30"))
CHEAP = (2, 1, 3, 32, True); QUAL = (2, 3, 0, 32, False)
B_CHEAP, B_QUAL = bits(2, 1, 3, 32, True), bits(2, 3, 0, 32)


def hadamard(n):
    H = torch.ones(1, 1)
    while H.shape[0] < n:
        H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
    return H
HD = hadamard(128).to(DEV).to(torch.float32)


def quant_nd(x, cfg):
    eb, mb, u, gs, mxp = cfg
    *lead, L = x.shape; pad = (-L) % BLOCK
    xp = F.pad(x, (0, pad)) if pad else x
    q = quant(xp.reshape(-1, L + pad).contiguous().float(), eb, mb, u, gs, Hblk=None, mxplus=mxp)
    q = q.reshape(*lead, L + pad)
    return q[..., :L] if pad else q


def q_K(k, cfg): return ((quant_nd(k.float() @ HD, cfg) @ HD.t()) / 128.0).to(k.dtype)
def q_V(v, cfg): return quant_nd(v.transpose(-1, -2), cfg).transpose(-1, -2).to(v.dtype)

_real = F.scaled_dot_product_attention
_CUR = {"l": -1}
# ASSIGN[(layer,head)] -> cfg. absent => bf16 (we always fill, so absent shouldn't happen in runs)
ASSIGN = {}
def _patch(q, k, v, *a, **kw):
    L = _CUR["l"]
    if L >= 0 and ASSIGN:
        Hkv = k.shape[1]
        kk = k.clone(); vv = v.clone()
        for h in range(Hkv):
            cfg = ASSIGN.get((L, h), QUAL)
            kk[:, h] = q_K(k[:, h], cfg); vv[:, h] = q_V(v[:, h], cfg)
        k, v = kk, vv
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
    NL = len(attn); NH = model.config.num_key_value_heads
    for i, m in attn:
        m.register_forward_pre_hook(lambda mod, a, kw, i=i: _CUR.update(l=i), with_kwargs=True)
    try: _wt = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    except Exception: _wt = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    ids = tok("\n\n".join(t for t in _wt["text"] if t.strip()), return_tensors="pt").input_ids

    global ASSIGN
    bf = ppl(model, ids, FULL_WIN)
    print(f"wikitext-2: {ids.size(1):,} tok | BF16 PPL = {bf:.4f} | NL={NL} NH={NH} | head-wise KV alloc\n", flush=True)

    allq = {(L, h): QUAL for L in range(NL) for h in range(NH)}
    ASSIGN = dict(allq); base = ppl(model, ids, PROBE_WIN)
    # head_sens[h]: head h cheap in ALL layers
    head_sens = {}
    for h in range(NH):
        ASSIGN = dict(allq)
        for L in range(NL): ASSIGN[(L, h)] = CHEAP
        head_sens[h] = (ppl(model, ids, PROBE_WIN) / base - 1) * 100
    print("head_sens (cheap across all layers):", " ".join(f"{h}:{head_sens[h]:+.2f}" for h in range(NH)), flush=True)
    # layer_sens[L]: layer L (all heads) cheap
    layer_sens = {}
    for L in range(NL):
        ASSIGN = dict(allq)
        for h in range(NH): ASSIGN[(L, h)] = CHEAP
        layer_sens[L] = (ppl(model, ids, PROBE_WIN) / base - 1) * 100
    lo = sorted(range(NL), key=lambda L: layer_sens[L])
    print("layer_sens rank(least->most):", " ".join(f"{L}:{layer_sens[L]:+.2f}" for L in lo), flush=True)

    # separable ranking of all 256 cells (per-head sens shares the layer's cost / NH)
    cells = [(L, h) for L in range(NL) for h in range(NH)]
    score = {(L, h): layer_sens[L] / NH + head_sens[h] / NL for (L, h) in cells}  # additive, comparable scale
    cellorder = sorted(cells, key=lambda c: score[c])

    # head-wise allocation vs per-layer at matched avg bytes. Per-layer hits bytes at multiples of
    # NH cheap cells (whole layers); head-wise can hit any. Compare at K_layer in {8,16,24}.
    print(f"\nhead-wise vs per-layer (FULL {FULL_WIN}w):", flush=True)
    print(f"{'avg b/elem':>10} | {'per-layer dPPL':>15} {'head-wise dPPL':>15}", flush=True)
    for Kl in (8, 16, 24):
        ncheap = Kl * NH                                          # same #cheap cells as Kl whole layers
        avgb = (ncheap * B_CHEAP + (NL * NH - ncheap) * B_QUAL) / (NL * NH)
        # per-layer: cheap the Kl least-sensitive WHOLE layers
        ASSIGN = dict(allq)
        for L in lo[:Kl]:
            for h in range(NH): ASSIGN[(L, h)] = CHEAP
        pl = (ppl(model, ids, FULL_WIN) / bf - 1) * 100
        # head-wise: cheap the ncheap lowest-score CELLS
        ASSIGN = dict(allq)
        for c in cellorder[:ncheap]: ASSIGN[c] = CHEAP
        hw = (ppl(model, ids, FULL_WIN) / bf - 1) * 100
        print(f"{avgb:>10.3f} | {pl:>+14.2f}% {hw:>+14.2f}%", flush=True)


if __name__ == "__main__":
    main()
