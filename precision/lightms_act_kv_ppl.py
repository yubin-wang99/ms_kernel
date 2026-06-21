"""light-MS PPL on ACTIVATION and KV scopes (online-quant paths) — wikitext-2.
  act-only : quantize each target Linear's INPUT activation (weight stays BF16).
  kv-only  : quantize K and V along head_dim (block 32) inside attention (weight BF16).
naive-MS / light-MS / MSAQ-signed, vs BF16, criterion within 3%.
Run: CUDA_VISIBLE_DEVICES=0 python precision/lightms_act_kv_ppl.py [act|kv|both]
"""
import sys, torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from lightms_qsnr import naive_ms, light_ms, msaq_signed

MODEL = "meta-llama/Llama-3.1-8B-Instruct"
DEV = "cuda"
MAXLEN, STRIDE, MAX_WINDOWS = 2048, 1024, 30
LINEAR_KEYS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
METHODS = (("naive", naive_ms), ("light", light_ms), ("msaq", msaq_signed))

def is_target(n, m):
    return isinstance(m, torch.nn.Linear) and any(p in n for p in LINEAR_KEYS)

# ---- activation-scope: patch Linear.forward to quantize the input -----------
def patch_act(model, fn, u, mg):
    orig = {}
    for n, m in model.named_modules():
        if is_target(n, m):
            orig[m] = m.forward
            def mk(mod):
                w, b = mod.weight, mod.bias
                def f(x):
                    return F.linear(fn(x, u, mg).to(w.dtype), w, b)
                return f
            m.forward = mk(m)
    return orig

def unpatch_act(orig):
    for m, f in orig.items(): m.forward = f

# ---- KV-scope: wrap SDPA to quantize key & value along head_dim -------------
_real_sdpa = F.scaled_dot_product_attention
_KV = {"fn": None, "u": 0, "mg": 0, "hits": 0}
def _kv_sdpa(query, key, value, *a, **kw):
    if _KV["fn"] is not None:
        _KV["hits"] += 1
        key = _KV["fn"](key, _KV["u"], _KV["mg"]).to(key.dtype)
        value = _KV["fn"](value, _KV["u"], _KV["mg"]).to(value.dtype)
    return _real_sdpa(query, key, value, *a, **kw)

@torch.no_grad()
def ppl(model, ids):
    seq = ids.size(1); nll_sum, ntok, prev_end, n = 0.0, 0, 0, 0
    for begin in range(0, seq, STRIDE):
        end = min(begin + MAXLEN, seq); trg = end - prev_end
        inp = ids[:, begin:end].to(DEV); tgt = inp.clone(); tgt[:, :-trg] = -100
        out = model(inp, labels=tgt)
        nll_sum += out.loss.double().item() * trg; ntok += trg; prev_end = end; n += 1
        if n >= MAX_WINDOWS or end == seq: break
    return torch.exp(torch.tensor(nll_sum / ntok)).item()

if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "both"
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16,
                                                 attn_implementation="sdpa").to(DEV).eval()
    F.scaled_dot_product_attention = _kv_sdpa            # global patch (verified to fire; inactive until _KV['fn'] set)
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    ids = tok("\n\n".join(t for t in ds["text"] if t.strip()), return_tensors="pt").input_ids
    print(f"wikitext-2: {ids.size(1):,} tokens; window={MAXLEN} stride={STRIDE} max_win={MAX_WINDOWS}\n")

    bf = ppl(model, ids); print(f"BF16 PPL = {bf:.4f}")
    # verify KV patch fires
    _KV.update(fn=light_ms, u=4, mg=8, hits=0); _ = ppl(model, ids)
    print(f"[kv patch sanity] sdpa hits with fn set = {_KV['hits']} (should be > 0)"); _KV["fn"] = None

    cfgs = [(2, 2), (3, 4), (4, 2), (4, 8)]
    def report(scope):
        print(f"\n=== {scope}-only PPL (% over BF16 {bf:.3f}, 3% criterion) ===")
        print(f"{'u':>2} {'mg':>3} | {'naive':>8} {'light':>8} {'MSAQ':>8} | within3%(l/m)")
        for u, mg in cfgs:
            r = {}
            for t, fn in METHODS:
                if scope == "act":
                    o = patch_act(model, fn, u, mg); p = ppl(model, ids); unpatch_act(o)
                else:
                    _KV.update(fn=fn, u=u, mg=mg); p = ppl(model, ids); _KV["fn"] = None
                r[t] = p
            pc = lambda p: (p / bf - 1) * 100
            ok = lambda p: "OK" if pc(p) <= 3 else "FAIL"
            print(f"{u:>2} {mg:>3} | {pc(r['naive']):>+7.2f}% {pc(r['light']):>+7.2f}% {pc(r['msaq']):>+7.2f}% | "
                  f"{ok(r['light'])}/{ok(r['msaq'])}")
    if which in ("act", "both"): report("act")
    if which in ("kv", "both"): report("kv")
