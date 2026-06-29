"""two-tier WEIGHT allocation — apply the Step-2 per-layer mixed-precision idea to weights (offline,
so trivially runtime-free). The uniform weight gate was negative (two-tier can't beat the E2M3 wall
at usable accuracy); the question here is whether per-layer MIXING (cheap on tolerant layers, E2M3 on
sensitive) gives a better bits-vs-PPL frontier than any uniform weight rung — i.e. usable accuracy
BELOW 6.25b on average.

Two rungs (same as KV Step 2): cheap = MX+ E2M1+u3 gs32 (4.75b, A-weighted), quality = E2M3 (6.25b).
A layer = its 7 Linears (q/k/v/o/gate/up/down) together. Per-Linear within-group Gram H_g calibrated
on wikitext-2 train. Probe each layer's sensitivity, rank, sweep K cheap layers, measure joint PPL.
Run:
    HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 MSAQ_MODEL=NousResearch/Meta-Llama-3.1-8B \
        CUDA_VISIBLE_DEVICES=0 .venv/bin/python precision/two_tier_weight_alloc_ppl.py \
        > precision/two_tier_weight_alloc_llama31_8b.txt 2>&1
"""
import os, torch
from msaq_mxfp8_ppl import BLOCK
from two_tier_gs_sweep_ppl import quant, bits
from two_tier_aware_ppl import collect_hessian
from two_tier_mxplus_ppl import two_tier_mxplus

DEV = "cuda"; MAXLEN, STRIDE = 2048, 1024
PROBE_WIN = int(os.environ.get("PROBE_WIN", "12"))
FULL_WIN = int(os.environ.get("FULL_WIN", "30"))
B_CHEAP, B_QUAL = bits(2, 1, 3, 32, True), bits(2, 3, 0, 32)        # 4.75, 6.25
KEYS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


def cheap(w, H): return two_tier_mxplus(w, 2, 1, 3, H, efb_iters=2)  # MX+ E2M1+u3 gs32, A-weighted
def qual(w, H):  return quant(w, 2, 3, 0, 32)                        # E2M3 native


def main():
    import torch.nn.functional as F
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from datasets import load_dataset
    MODEL = os.environ.get("MSAQ_MODEL", "meta-llama/Llama-3.1-8B-Instruct")
    tgt = lambda n, m: isinstance(m, torch.nn.Linear) and any(p in n for p in KEYS)

    @torch.no_grad()
    def ppl(model, ids, nwin):
        seq = ids.size(1); nll, ntok, prev, n = 0.0, 0, 0, 0
        for b in range(0, seq, STRIDE):
            e = min(b + MAXLEN, seq); trg = e - prev
            inp = ids[:, b:e].to(DEV); t = inp.clone(); t[:, :-trg] = -100
            nll += model(inp, labels=t).loss.double().item() * trg; ntok += trg; prev = e; n += 1
            if n >= nwin or e == seq: break
        return torch.exp(torch.tensor(nll / ntok)).item()

    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, attn_implementation="sdpa").to(DEV).eval()
    def wt(sp):
        try: return load_dataset("wikitext", "wikitext-2-raw-v1", split=sp)
        except Exception: return load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split=sp)
    test_ids = tok("\n\n".join(t for t in wt("test")["text"] if t.strip()), return_tensors="pt").input_ids
    cal_ids = tok("\n\n".join(t for t in wt("train")["text"] if t.strip()), return_tensors="pt").input_ids
    targets = [(n, m) for n, m in model.named_modules() if tgt(n, m)]
    layer_of = {m: int(n.split(".")[2]) for n, m in targets}
    NL = max(layer_of.values()) + 1
    master = {m: m.weight.detach().to("cpu", copy=True) for _, m in targets}

    bf = ppl(model, test_ids, FULL_WIN)
    print(f"wikitext-2: test {test_ids.size(1):,} tok | BF16 PPL = {bf:.4f} | NL={NL} | "
          f"cheap={B_CHEAP:.3f}b E2M3={B_QUAL:.3f}b | WEIGHT allocation\n", flush=True)
    print("calibrating per-Linear Hessian on TRAIN (16x2048)...", flush=True)
    Hblk = collect_hessian(model, targets, cal_ids, DEV, n_windows=16, win=2048)
    print("done.\n", flush=True)

    def apply(alloc):                                              # alloc: layer -> 'c'/'q'
        torch.cuda.empty_cache()
        for _, m in targets:
            qfn = cheap if alloc.get(layer_of[m]) == "c" else qual
            w = qfn(master[m].to(DEV), Hblk[m]).to(m.weight.dtype)
            m.weight.data.copy_(w); del w
    def restore():
        for _, m in targets: m.weight.data.copy_(master[m].to(DEV))

    # (1) per-layer sensitivity (only layer L cheap, rest E2M3)
    apply({i: "q" for i in range(NL)}); base_q = ppl(model, test_ids, PROBE_WIN)
    print(f"probe baseline (all E2M3, {PROBE_WIN}w) PPL={base_q:.4f}; per-layer cheap ΔPPL%:", flush=True)
    cost = {}
    for L in range(NL):
        a = {i: "q" for i in range(NL)}; a[L] = "c"; apply(a)
        cost[L] = (ppl(model, test_ids, PROBE_WIN) / base_q - 1) * 100
    order = sorted(range(NL), key=lambda L: cost[L])
    print("  rank(least->most sensitive):", " ".join(f"{L}:{cost[L]:+.2f}" for L in order), flush=True)

    # (2) allocation sweep (real joint PPL)
    print(f"\nallocation curve (FULL {FULL_WIN}w) — K cheap layers vs uniform:", flush=True)
    print(f"{'K':>3} {'avg b/elem':>10} | {'PPL':>9} {'dPPL%':>8}", flush=True)
    for K in [0, 4, 8, 12, 16, 20, 24, 28, 32]:
        a = {i: "q" for i in range(NL)}
        for L in order[:K]: a[L] = "c"
        apply(a); p = ppl(model, test_ids, FULL_WIN); restore()
        avgb = (K * B_CHEAP + (NL - K) * B_QUAL) / NL
        print(f"{K:>3} {avgb:>10.3f} | {p:>9.4f} {(p/bf-1)*100:>+7.2f}%", flush=True)


if __name__ == "__main__":
    main()
