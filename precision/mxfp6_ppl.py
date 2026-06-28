"""MXFP6 (HARDWARE-NATIVE) vs MXFP8-MSAQ E3M4 vs MXINT8-MSAQ — wikitext-2 PPL.

Decision experiment for the "reformulate MSAQ to a standard MX element" path. Blackwell tensor
cores natively consume MXFP6 (E3M2 / E2M3, 6-bit element + UE8M0 block scale = 6.25 b/elem) with
NO per-element unpack — but only E4M3/E5M2/E3M2/E2M3/E2M1, never the custom E3M4 + shared mantissa.
So the question is: does the hardware-native MXFP6 match the custom 6.0-bit MSAQ-E3M4 on accuracy?
If yes -> drop MSAQ, ride the tensor cores. If no -> MSAQ's sub-byte squeeze is justified.

Each format quantizes weight / weight+act / KV / weight+KV; table is BF16-relative PPL %.
Plain MXFP{6,8} = msaq_mxfp8(u=0) (no sharing). Run:
    MSAQ_MODEL=NousResearch/Meta-Llama-3.1-8B CUDA_VISIBLE_DEVICES=0 \
        python precision/mxfp6_ppl.py > precision/mxfp6_ppl_llama31_8b.txt 2>&1
"""
import os, torch, torch.nn.functional as F
from msaq_mxfp8_ppl import msaq_mxfp8, msaq_mxint8, BLOCK

DEV = "cuda"
MAXLEN, STRIDE, MAX_WINDOWS = 2048, 1024, 30
LINEAR_KEYS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
SCOPES = ("weight", "wa", "kv", "wkv")
LABEL = {"weight": "weight", "wa": "weight+act", "kv": "KV", "wkv": "weight+KV"}

# (name, bits/elem, quant fn) — the columns. Plain MXFP6/8 are hardware-native (no sharing);
# the two 6.0b MSAQ formats are the custom (CUDA-core) references.
COLUMNS = [
    ("MXFP6-E3M2*", 6.25, lambda x: msaq_mxfp8(x, 0, 1, 3, 2)),   # HW-native
    ("MXFP6-E2M3*", 6.25, lambda x: msaq_mxfp8(x, 0, 1, 2, 3)),   # HW-native
    ("E3M4-MSAQ",   6.00, lambda x: msaq_mxfp8(x, 3, 4, 3, 4, efb_iters=2)),
    ("MXINT8-MSAQ", 6.00, lambda x: msaq_mxint8(x, 3, 4)),
    ("MXFP8-E4M3*", 8.25, lambda x: msaq_mxfp8(x, 0, 1, 4, 3)),   # HW-native anchor (more bits)
]  # * = standard MX element -> rides native block-scaled tensor cores


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from datasets import load_dataset
    MODEL = os.environ.get("MSAQ_MODEL", "meta-llama/Llama-3.1-8B-Instruct")
    _real_sdpa = F.scaled_dot_product_attention
    _KV = {"on": False, "q": None}

    def _kv_sdpa(q, k, v, *a, **kw):
        if _KV["on"]:
            k = _KV["q"](k).to(k.dtype); v = _KV["q"](v).to(v.dtype)
        return _real_sdpa(q, k, v, *a, **kw)
    F.scaled_dot_product_attention = _kv_sdpa

    def is_target(n, m): return isinstance(m, torch.nn.Linear) and any(p in n for p in LINEAR_KEYS)

    @torch.no_grad()
    def ppl(model, ids):
        seq = ids.size(1); nll, ntok, prev, n = 0.0, 0, 0, 0
        for b in range(0, seq, STRIDE):
            e = min(b + MAXLEN, seq); trg = e - prev
            inp = ids[:, b:e].to(DEV); tgt = inp.clone(); tgt[:, :-trg] = -100
            nll += model(inp, labels=tgt).loss.double().item() * trg; ntok += trg; prev = e; n += 1
            if n >= MAX_WINDOWS or e == seq: break
        return torch.exp(torch.tensor(nll / ntok)).item()

    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16,
                                                 attn_implementation="sdpa").to(DEV).eval()
    try:
        _wt = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    except Exception:
        _wt = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    ids = tok("\n\n".join(t for t in _wt["text"] if t.strip()), return_tensors="pt").input_ids
    targets = [(n, m) for n, m in model.named_modules() if is_target(n, m)]
    master = {m: m.weight.detach().to("cpu", copy=True) for _, m in targets}

    def restore():
        for _, m in targets: m.weight.data.copy_(master[m].to(DEV))

    def patch_act(qfn):
        orig = {}
        for _, m in targets:
            orig[m] = m.forward
            def mk(mod):
                w, b = mod.weight, mod.bias
                def f(x): return F.linear(qfn(x).to(x.dtype), w, b)
                return f
            m.forward = mk(m)
        return orig

    bf = ppl(model, ids)
    print(f"wikitext-2: {ids.size(1):,} tokens | BF16 PPL = {bf:.4f} | block={BLOCK} | * = HW-native MX element\n", flush=True)

    def run(scope, qfn):
        restore()
        if scope in ("weight", "wa", "wkv"):
            for _, m in targets: m.weight.data.copy_(qfn(master[m].to(DEV)).to(m.weight.dtype))
        oa = patch_act(qfn) if scope == "wa" else None
        if scope in ("kv", "wkv"): _KV.update(on=True, q=qfn)
        p = ppl(model, ids)
        if oa:
            for m, f in oa.items(): m.forward = f
        _KV["on"] = False; restore()
        return (p / bf - 1) * 100

    hdr = "".join(f"{nm}({b:.2f}b)".rjust(18) for nm, b, _ in COLUMNS)
    print(f"{'scope':>11} |{hdr}", flush=True)
    for scope in SCOPES:
        cells = []
        for nm, b, qfn in COLUMNS:
            pct = run(scope, qfn)
            cells.append(f"{pct:+.2f}% {'OK' if pct <= 3 else 'X'}".rjust(18))
        print(f"{LABEL[scope]:>11} |" + "".join(cells), flush=True)


if __name__ == "__main__":
    main()
