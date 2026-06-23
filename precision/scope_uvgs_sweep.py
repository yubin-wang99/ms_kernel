"""Per-scope max-aggressive robust (u, gs): plain MSAQ (block=32, no rotation/two-level).
For S1..S5 find the most aggressive (u,gs) keeping wikitext PPL within 3.5% of BF16.
  S1 weight | S2 weight+act | S3 KV | S4 weight+KV | S5 weight+act+KV
sweep u (S2,S5: {2,3}; else {2,3,4}) x gs {2,4,8,16,32}. gs ascending, break on first FAIL
(aggressiveness is monotonic in gs and u -> prunes the hopeless tail). MAX_WINDOWS bounds runtime.
Run: CUDA_VISIBLE_DEVICES=0 python precision/scope_uvgs_sweep.py [maxwin] > precision/scope_uvgs_sweep.txt 2>&1
"""
import sys, time, torch, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from lightms_qsnr import msaq_signed

MODEL = "meta-llama/Llama-3.1-8B-Instruct"; DEV = "cuda"
MAXLEN, STRIDE = 2048, 1024
MAX_WINDOWS = int(sys.argv[1]) if len(sys.argv) > 1 else 20
TOL = 3.5
LINEAR_KEYS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
GS = [2, 4, 8, 16, 32]
# scope -> (quant_weight, quant_act, quant_kv, u_list)
SCOPES = [("S1 weight",      (True,  False, False, [2, 3, 4])),
          ("S2 weight+act",  (True,  True,  False, [2, 3])),
          ("S3 KV",          (False, False, True,  [2, 3, 4])),
          ("S4 weight+KV",   (True,  False, True,  [2, 3, 4])),
          ("S5 weight+act+KV",(True, True,  True,  [2, 3]))]
Q = msaq_signed

def bits_per_elem(u, gs):                                # MSAQ-s footprint: upper + shared + E8M0 scale
    UB = 32 * (8 - u) // 8; SB = ((32 // gs) * u + 7) // 8
    return (UB + SB + 1) * 8.0 / 32.0

def is_target(n, m):
    return isinstance(m, torch.nn.Linear) and any(p in n for p in LINEAR_KEYS)

def patch_act(targets, u, mg):
    orig = {}
    for _, m in targets:
        orig[m] = m.forward
        def mk(mod):
            w, b = mod.weight, mod.bias
            def f(x): return F.linear(Q(x, u, mg).to(w.dtype), w, b)
            return f
        m.forward = mk(m)
    return orig
def unpatch_act(o):
    for m, f in o.items(): m.forward = f

_real_sdpa = F.scaled_dot_product_attention
_KV = {"on": False, "u": 0, "mg": 0}
def _kv_sdpa(q, k, v, *a, **kw):
    if _KV["on"]:
        k = Q(k, _KV["u"], _KV["mg"]).to(k.dtype); v = Q(v, _KV["u"], _KV["mg"]).to(v.dtype)
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
    master = {m: m.weight.detach().to("cpu", copy=True) for _, m in targets}
    def restore():
        for _, m in targets: m.weight.data.copy_(master[m].to(DEV))
    def quant_w(u, mg):
        for _, m in targets: m.weight.data.copy_(Q(master[m].to(DEV), u, mg).to(m.weight.dtype))

    t0 = time.time()
    bf = ppl(model, ids)
    print(f"wikitext-2: {ids.size(1):,} tok | BF16 PPL = {bf:.4f} | max_win={MAX_WINDOWS} | "
          f"criterion <= +{TOL}% | baseline pass {time.time()-t0:.0f}s\n", flush=True)

    def run(spec, u, gs):
        qw, qa, qkv, _ = spec
        restore()
        if qw: quant_w(u, gs)
        oa = patch_act(targets, u, gs) if qa else None
        if qkv: _KV.update(on=True, u=u, mg=gs)
        p = ppl(model, ids)
        if oa: unpatch_act(oa)
        _KV["on"] = False; restore()
        return (p / bf - 1) * 100

    best = {}
    for name, spec in SCOPES:
        ulist = spec[3]
        print(f"=== {name} ===", flush=True)
        passes = []                                       # (u,gs,pct,bits)
        for u in ulist:
            for gs in GS:
                pct = run(spec, u, gs); ok = pct <= TOL
                tag = "OK" if ok else "FAIL"
                print(f"   u{u}/gs{gs:<2} : {pct:>+6.2f}%  {tag}  ({bits_per_elem(u,gs):.2f} b/elem)", flush=True)
                if ok: passes.append((u, gs, pct, bits_per_elem(u, gs)))
                else: break                               # gs ascending -> larger gs only worse
        if passes:
            mostagg = min(passes, key=lambda r: r[3])     # min bits/elem = most aggressive robust
            best[name] = mostagg
            print(f"   -> MAX-aggressive robust: u{mostagg[0]}/gs{mostagg[1]} "
                  f"({mostagg[2]:+.2f}%, {mostagg[3]:.2f} b/elem)\n", flush=True)
        else:
            best[name] = None; print("   -> none robust\n", flush=True)

    print("="*60 + "\nSUMMARY — most-aggressive robust (u,gs) per scope (<=+3.5% PPL)", flush=True)
    for name, _ in SCOPES:
        r = best[name]
        s = f"u{r[0]}/gs{r[1]}  ({r[2]:+.2f}%, {r[3]:.2f} bits/elem)" if r else "none"
        print(f"  {name:18s}: {s}", flush=True)
    print(f"\ntotal {time.time()-t0:.0f}s", flush=True)
