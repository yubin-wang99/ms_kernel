"""KV ladder — Step 2: per-layer allocation (§7-third gate). Does a MIXED per-layer assignment
(cheap two-tier rung on tolerant layers, E2M3 on sensitive ones) beat a UNIFORM rung at the same
average bytes/token? Two rungs:
    cheap   = MX+ E2M1+u3 gs32  (4.75 b/elem, Step-1 sweet spot, KV-scope +2.71%)
    quality = E2M3 native        (6.25 b/elem, +0.30%)
Method: (1) per-layer sensitivity probe — only layer L cheap, rest E2M3, measure ΔPPL (PROBE_WIN
windows, ranking-only). (2) rank layers least->most sensitive. (3) sweep K = #cheap layers (the K
least-sensitive go cheap); measure the ACTUAL joint PPL (FULL_WIN). avg bytes = (K*4.75+(32-K)*6.25)/32.
Compare the allocation curve to the Step-1 UNIFORM ladder at matched bytes (printed for reference).

Per-layer control: a forward-pre-hook on each self_attn sets the current layer index; the global sdpa
patch reads ALLOC[layer]. K is rotated+D-blocked, V is T-blocked (kv_ladder_design §2).
Run:
    HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 MSAQ_MODEL=NousResearch/Meta-Llama-3.1-8B \
        CUDA_VISIBLE_DEVICES=0 .venv/bin/python precision/kv_ladder_step2_alloc_ppl.py \
        > precision/kv_ladder_step2_llama31_8b.txt 2>&1
"""
import os, torch
import torch.nn.functional as F
from msaq_mxfp8_ppl import BLOCK
from two_tier_gs_sweep_ppl import quant, bits

DEV = "cuda"; MAXLEN, STRIDE = 2048, 1024
PROBE_WIN = int(os.environ.get("PROBE_WIN", "12"))         # ranking-only (fast)
FULL_WIN = int(os.environ.get("FULL_WIN", "30"))           # final allocation evals
_CU = int(os.environ.get("CHEAP_U", "3"))                  # cheap rung u (CHEAP_U=0 -> 4.406b, no residual)
CHEAP = (2, 1, _CU, 32, True)                              # default MX+ E2M1+u3 gs32 -> 4.75b
QUAL = (2, 3, 0, 32, False)                                # E2M3 -> 6.25b
B_CHEAP, B_QUAL = bits(2, 1, _CU, 32, True), bits(2, 3, 0, 32)


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
ALLOC = {}                                                 # layer_idx -> cfg tuple (absent = bf16)
def _patch(q, k, v, *a, **kw):
    cfg = ALLOC.get(_CUR["l"])
    if cfg is not None:
        k = q_K(k, cfg); v = q_V(v, cfg)
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


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from datasets import load_dataset
    MODEL = os.environ.get("MSAQ_MODEL", "meta-llama/Llama-3.1-8B-Instruct")
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16,
                                                 attn_implementation="sdpa").to(DEV).eval()
    # per-layer index hooks
    attn = [(int(n.split(".")[2]), m) for n, m in model.named_modules() if n.endswith("self_attn")]
    NL = len(attn)
    for i, m in attn:
        m.register_forward_pre_hook(lambda mod, args, kwargs, i=i: _CUR.update(l=i), with_kwargs=True)
    try: _wt = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    except Exception: _wt = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    ids = tok("\n\n".join(t for t in _wt["text"] if t.strip()), return_tensors="pt").input_ids

    global ALLOC
    bf = ppl(model, ids, FULL_WIN)
    print(f"wikitext-2: {ids.size(1):,} tok | BF16 PPL = {bf:.4f} | NL={NL} | "
          f"cheap={B_CHEAP:.3f}b E2M3={B_QUAL:.3f}b\n", flush=True)

    # (1) per-layer sensitivity: only layer L cheap, rest E2M3 (ranking, PROBE_WIN)
    ALLOC = {i: QUAL for i in range(NL)}
    base_e2m3 = ppl(model, ids, PROBE_WIN)
    print(f"probe baseline (all E2M3, {PROBE_WIN}w) PPL={base_e2m3:.4f}; per-layer cheap ΔPPL%:", flush=True)
    cost = {}
    for L in range(NL):
        ALLOC = {i: QUAL for i in range(NL)}; ALLOC[L] = CHEAP
        cost[L] = (ppl(model, ids, PROBE_WIN) / base_e2m3 - 1) * 100
    order = sorted(range(NL), key=lambda L: cost[L])        # least sensitive first
    print("  rank(least->most sensitive):", " ".join(f"{L}:{cost[L]:+.2f}" for L in order), flush=True)

    # (2) allocation sweep: K least-sensitive layers -> cheap, rest E2M3 (FULL_WIN, real joint PPL)
    print(f"\nallocation curve (FULL {FULL_WIN}w) — K cheap layers vs uniform:", flush=True)
    print(f"{'K':>3} {'avg b/elem':>10} | {'PPL':>9} {'dPPL%':>8}", flush=True)
    for K in [0, 4, 8, 12, 16, 20, 24, 28, 32]:
        ALLOC = {i: QUAL for i in range(NL)}
        for L in order[:K]: ALLOC[L] = CHEAP
        avgb = (K * B_CHEAP + (NL - K) * B_QUAL) / NL
        p = ppl(model, ids, FULL_WIN)
        print(f"{K:>3} {avgb:>10.3f} | {p:>9.4f} {(p/bf-1)*100:>+7.2f}%", flush=True)


if __name__ == "__main__":
    main()
