"""(a) robust-aggressive boundary per scope: sweep light-MS PPL over a (u,mg) grid,
rank by bits-per-element, find the MIN-bits config passing the 3% criterion.
Scopes: weight / activation / kv (online-quant). wikitext-2.
Run: CUDA_VISIBLE_DEVICES=0 python precision/lightms_boundary.py > precision/boundary.txt 2>&1
"""
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from lightms_qsnr import light_ms

MODEL = "meta-llama/Llama-3.1-8B-Instruct"; DEV = "cuda"
MAXLEN, STRIDE, MAX_WINDOWS = 2048, 1024, 30
LINEAR_KEYS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
CRIT = 3.0

def bits_per_elem(u, mg):                         # MSAQ-signed: (8-u) unshared + u/mg shared + 8/32 scale
    return (8 - u) + u / mg + 8.0 / 32.0

def is_target(n, m):
    return isinstance(m, torch.nn.Linear) and any(p in n for p in LINEAR_KEYS)

# ---- KV-scope SDPA patch (verified to fire) ---------------------------------
_real_sdpa = F.scaled_dot_product_attention
_KV = {"u": 0, "mg": 0, "on": False}
def _kv_sdpa(q, k, v, *a, **kw):
    if _KV["on"]:
        k = light_ms(k, _KV["u"], _KV["mg"]).to(k.dtype)
        v = light_ms(v, _KV["u"], _KV["mg"]).to(v.dtype)
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

    bf = ppl(model, ids); print(f"BF16 PPL = {bf:.4f}; criterion = within {CRIT}%\n", flush=True)

    grid = [(2, 2), (3, 4), (3, 8), (3, 16), (3, 32), (4, 2), (4, 4), (4, 8), (4, 16)]

    def run_weight(u, mg):
        for n, m in targets: m.weight.data.copy_(light_ms(master[n].to(DEV), u, mg).to(m.weight.dtype))
        p = ppl(model, ids)
        for n, m in targets: m.weight.data.copy_(master[n].to(DEV))   # restore
        return p
    def run_act(u, mg):
        orig = {}
        for n, m in targets:
            orig[m] = m.forward
            def mk(mod):
                w, b = mod.weight, mod.bias
                return lambda x: F.linear(light_ms(x, u, mg).to(w.dtype), w, b)
            m.forward = mk(m)
        p = ppl(model, ids)
        for m, f in orig.items(): m.forward = f
        return p
    def run_kv(u, mg):
        _KV.update(u=u, mg=mg, on=True); p = ppl(model, ids); _KV["on"] = False
        return p

    for scope, runner in (("weight", run_weight), ("activation", run_act), ("kv", run_kv)):
        rows = []
        for u, mg in grid:
            p = runner(u, mg); pct = (p / bf - 1) * 100
            rows.append((bits_per_elem(u, mg), u, mg, pct))
        rows.sort()                                  # by bits ascending (most aggressive first)
        print(f"=== {scope} (sorted by bits/elem, most aggressive first) ===", flush=True)
        print(f"{'bits':>5} {'u':>2} {'mg':>3} | {'light %':>8} | 3%?", flush=True)
        best = None
        for bits, u, mg, pct in rows:
            ok = pct <= CRIT
            if ok and best is None: best = (bits, u, mg, pct)
            print(f"{bits:>5.3f} {u:>2} {mg:>3} | {pct:>+7.2f}% | {'OK' if ok else 'FAIL'}", flush=True)
        if best:
            print(f"  -> MIN-BITS robust: u{best[1]}/mg{best[2]} = {best[0]:.3f} bits/elem ({best[3]:+.2f}%)\n", flush=True)
        else:
            print("  -> none in grid passes 3%\n", flush=True)
