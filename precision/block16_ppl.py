"""Block-size precision sweep: does MSAQ-signed stay robust at u=4 with a 16-element
block instead of the OCP-standard 32-element block? wikitext-2 PPL across the 4
quantization scopes (weight / weight+activation / KV / weight+KV).

OCP MXINT8 shares one E8M0 scale over 32 elements; a 16-element block gives a FINER
scale (more scales, less averaging of outliers under one scale) at the cost of 2x
scale-byte overhead. Hypothesis: block=16 widens the robust frontier so the aggressive
nibble u=4 config that FAILS at block=32 (e.g. KV u4/mg8 = +5.14%) becomes robust.

Quantizer = msaq_signed (the deployed MSAQ-s format). BLOCK is the module global in
lightms_qsnr, read live by _blocks(), so we monkeypatch it. Criterion: within 3% of BF16.
Run: CUDA_VISIBLE_DEVICES=0 python precision/block16_ppl.py > precision/block16_ppl.txt 2>&1
"""
import torch, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import lightms_qsnr
from lightms_qsnr import msaq_signed

MODEL = "meta-llama/Llama-3.1-8B-Instruct"; DEV = "cuda"
MAXLEN, STRIDE, MAX_WINDOWS = 2048, 1024, 30
LINEAR_KEYS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
BLOCKS = (8,)
CONFIGS = [(4, 8), (4, 4), (4, 2)]                      # u=4 robustness sweep over mg
SCOPES = ("weight", "wa", "kv", "wkv")
Q = msaq_signed

def is_target(n, m):
    return isinstance(m, torch.nn.Linear) and any(p in n for p in LINEAR_KEYS)

# ---- activation scope: quantize each target Linear's INPUT ------------------
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

def unpatch_act(orig):
    for m, f in orig.items(): m.forward = f

# ---- KV scope: quantize K and V along head_dim inside SDPA ------------------
_real_sdpa = F.scaled_dot_product_attention
_KV = {"on": False, "u": 0, "mg": 0}
def _kv_sdpa(q, k, v, *a, **kw):
    if _KV["on"]:
        k = Q(k, _KV["u"], _KV["mg"]).to(k.dtype)
        v = Q(v, _KV["u"], _KV["mg"]).to(v.dtype)
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
    def restore():                                       # all weights back to BF16
        for _, m in targets: m.weight.data.copy_(master[m].to(DEV))
    def quant_w(u, mg):                                  # in-place MSAQ weights (current BLOCK)
        for _, m in targets: m.weight.data.copy_(Q(master[m].to(DEV), u, mg).to(m.weight.dtype))

    bf = ppl(model, ids)
    print(f"wikitext-2: {ids.size(1):,} tokens | BF16 PPL = {bf:.4f} | criterion: within 3%\n", flush=True)

    def run(scope, u, mg):
        restore()
        if scope in ("weight", "wa", "wkv"): quant_w(u, mg)
        oa = patch_act(targets, u, mg) if scope == "wa" else None
        if scope in ("kv", "wkv"): _KV.update(on=True, u=u, mg=mg)
        p = ppl(model, ids)
        if oa: unpatch_act(oa)
        _KV["on"] = False; restore()
        return (p / bf - 1) * 100

    LABEL = {"weight": "weight", "wa": "weight+act", "kv": "KV", "wkv": "weight+KV"}
    for bs in BLOCKS:
        lightms_qsnr.BLOCK = bs
        print(f"================  BLOCK = {bs} elements  ================", flush=True)
        print(f"{'scope':>11} | " + " ".join(f"u{u}/mg{mg:<2}" for u, mg in CONFIGS) + " | within 3%?", flush=True)
        for scope in SCOPES:
            cells, oks = [], []
            for u, mg in CONFIGS:
                pct = run(scope, u, mg); cells.append(f"{pct:>+6.2f}%"); oks.append("OK" if pct <= 3 else "FAIL")
            print(f"{LABEL[scope]:>11} | " + "  ".join(cells) + " | " + " ".join(oks), flush=True)
        print(flush=True)
