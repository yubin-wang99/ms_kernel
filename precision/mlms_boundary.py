"""Multi-level robustness: does hierarchical (multi-level) sharing stay within the 3%
BF16-PPL bar at LOWER bits/elem than single-level can? And is MSAQ-multi >= naive-multi?

Compares, on wikitext-2:
  - single-level robust frontier reference (weight u3/mg8=5.625, act u3/mg4=6.0, kv u3/mg32=5.344)
  - naive-multi  (ssnf bit-plane hierarchical sharing)   at bpe 5.0 / 5.25 / 5.5
  - MSAQ-multi   (our residual successive signed sharing) at the same configs
Run: CUDA_VISIBLE_DEVICES=0 python precision/mlms_boundary.py > precision/mlms.txt 2>&1
"""
import sys, torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from mlms_quant import naive_ml, msaq_ml, bpe
from lightms_qsnr import light_ms                       # single-level reference

import os
MODEL = "meta-llama/Llama-3.1-8B-Instruct"; DEV = "cuda"
MAXLEN, STRIDE = 2048, 1024
MAX_WINDOWS = int(os.environ.get("MLMS_WIN", "30"))
LINEAR_KEYS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
CRIT = 3.0

def is_target(n, m):
    return isinstance(m, torch.nn.Linear) and any(p in n for p in LINEAR_KEYS)

# Two regimes (LSB-first bw/mg):
#  (A) 3 shared bits (5 unshared) at bpe iso to single-level -> tests the HIERARCHY benefit.
#  (B) 4 shared bits (4 unshared) BELOW single-level's ~5.34 floor -> the regime single-level
#      has NO robust option (u4 fails); this is where multi-level must win to matter.
# Per-scope multi-level configs. MSB-fine principle (user): keep bits near MSB LESS shared
# (small mg) and push LSBs coarse (big mg). Tests if the hierarchy beats single-level's floor.
ML_CFGS_BY_SCOPE = {
    "kv": [
        ("[2,2]/[32,2]",     [2, 2],    [32, 2]),     # bpe 5.31  (top-2 fine)
        ("[1,1,2]/[32,32,2]",[1, 1, 2], [32, 32, 2]), # bpe 5.31  (LSBs split, top-2 fine)
        ("[1,2,1]/[32,32,2]",[1, 2, 1], [32, 32, 2]), # bpe 5.31  (top bit fine, mid coarse)
        ("[3,1]/[32,2]",     [3, 1],    [32, 2]),      # bpe 4.84
        ("[2,2]/[32,4]",     [2, 2],    [32, 4]),      # bpe 4.81
    ],
    "activation": [
        ("[2,1]/[8,8]",   [2, 1], [8, 8]),    # bpe 5.625  (iso single u3/mg8, below act floor 6.0)
        ("[2,1]/[16,16]", [2, 1], [16, 16]),  # bpe 5.4375
        ("[2,2]/[32,2]",  [2, 2], [32, 2]),   # bpe 5.31
        ("[2,2]/[16,4]",  [2, 2], [16, 4]),   # bpe 4.875
    ],
    "weight": [
        ("[2,1]/[8,8]",   [2, 1], [8, 8]),    # bpe 5.625  iso single u3/mg8
        ("[2,1]/[32,32]", [2, 1], [32, 32]),  # bpe 5.344
        ("[2,2]/[32,2]",  [2, 2], [32, 2]),   # bpe 5.31
    ],
}
# single-level reference (u, mg) -> bpe = (8-u)+u/mg+0.25
SL_REF = {"weight": (3, 8), "activation": (3, 4), "kv": (3, 32)}

# ---- KV SDPA patch ----------------------------------------------------------
_real_sdpa = F.scaled_dot_product_attention
_KV = {"fn": None, "on": False}
def _kv_sdpa(q, k, v, *a, **kw):
    if _KV["on"]:
        k = _KV["fn"](k).to(k.dtype); v = _KV["fn"](v).to(v.dtype)
    return _real_sdpa(q, k, v, *a, **kw)
F.scaled_dot_product_attention = _kv_sdpa

@torch.no_grad()
def ppl(model, ids):
    seq = ids.size(1); nll, ntok, prev, n = 0.0, 0, 0, 0
    for b in range(0, seq, STRIDE):
        e = min(b + MAXLEN, seq); trg = e - prev
        inp = ids[:, b:e].to(DEV); tgt = inp.clone(); tgt[:, :-trg] = -100
        nll += model(inp, labels=tgt).loss.double().item() * trg; ntok += trg; prev = e; n += 1
        if n >= MAX_WINDOWS or e == seq: break
    return torch.exp(torch.tensor(nll / ntok)).item()

if __name__ == "__main__":
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16,
                                                 attn_implementation="sdpa").to(DEV).eval()
    ids = tok("\n\n".join(t for t in load_dataset("wikitext", "wikitext-2-raw-v1", split="test")["text"]
                          if t.strip()), return_tensors="pt").input_ids
    targets = [(n, m) for n, m in model.named_modules() if is_target(n, m)]
    master = {n: m.weight.detach().to("cpu", copy=True) for n, m in targets}

    bf = ppl(model, ids); print(f"BF16 PPL = {bf:.4f}; criterion within {CRIT}%\n", flush=True)
    pct = lambda p: (p / bf - 1) * 100

    def run_weight(fn):
        for n, m in targets: m.weight.data.copy_(fn(master[n].to(DEV)).to(m.weight.dtype))
        p = ppl(model, ids)
        for n, m in targets: m.weight.data.copy_(master[n].to(DEV))
        return p
    def run_act(fn):
        orig = {}
        for n, m in targets:
            orig[m] = m.forward
            def mk(mod):
                w, b = mod.weight, mod.bias
                return lambda x: F.linear(fn(x).to(w.dtype), w, b)
            m.forward = mk(m)
        p = ppl(model, ids)
        for m, f in orig.items(): m.forward = f
        return p
    def run_kv(fn):
        _KV.update(fn=fn, on=True); p = ppl(model, ids); _KV["on"] = False
        return p
    RUN = {"weight": run_weight, "activation": run_act, "kv": run_kv}

    scopes = sys.argv[1:] or ["weight", "activation", "kv"]
    for scope in scopes:
        runner = RUN[scope]
        print(f"=== {scope} ===", flush=True)
        u, mg = SL_REF[scope]
        b_sl = (8 - u) + u / mg + 0.25
        p = runner(lambda x: light_ms(x, u, mg))
        print(f"  single-level u{u}/mg{mg:>2} bpe={b_sl:.3f} | {pct(p):>+7.2f}% | "
              f"{'OK' if pct(p) <= CRIT else 'FAIL'}  (reference frontier)", flush=True)
        for tag, bw, mg2 in ML_CFGS_BY_SCOPE[scope]:
            b = bpe(bw, mg2)
            pn = runner(lambda x: naive_ml(x, bw, mg2)); pm = runner(lambda x: msaq_ml(x, bw, mg2))
            mark = lambda v: 'OK' if v <= CRIT else 'FAIL'
            print(f"  ML {tag:>11} bpe={b:.3f} | naive {pct(pn):>+7.2f}% {mark(pct(pn)):>4} | "
                  f"MSAQ {pct(pm):>+7.2f}% {mark(pct(pm)):>4} | gain {pct(pn)-pct(pm):>+6.2f}pp", flush=True)
        print(flush=True)
