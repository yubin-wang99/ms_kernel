"""MX6/MX9-style TWO-LEVEL scaling on MSAQ: does a per-sub-block microexponent rescue u=4?
block=32 fixed. wikitext-2 PPL across the 4 scopes (weight / weight+act / KV / weight+KV).

MX9/MX6 (Rouhani et al., shared microexponents) = L1 8-bit E8M0 over a block + L2 d2-bit
microexponent over k2=2 elements. The L2 sub-scale gives small sub-blocks a locally finer scale
at d2/k2 bit/elem (vs a full 8-bit scale for a smaller block). We keep MSAQ's u/mg mantissa-sharing
and replace its single per-32 E8M0 scale with two-level: per-32 E8M0 (L1) + per-2 d2-bit microexp (L2).
d2=0 reproduces msaq_signed exactly (cross-check); d2=1 = MX-style; d2=2 = stronger. Within 3% of BF16.
Run: CUDA_VISIBLE_DEVICES=0 python precision/mx_2lvl_ppl.py > precision/mx_2lvl_ppl.txt 2>&1
"""
import torch, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

MODEL = "meta-llama/Llama-3.1-8B-Instruct"; DEV = "cuda"
MAXLEN, STRIDE, MAX_WINDOWS = 2048, 1024, 30
LINEAR_KEYS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
K1, K2 = 32, 2                                           # L1 block, L2 sub-block
D2S = (0, 1, 2)                                          # microexponent bits (0 = msaq_signed baseline)
CONFIGS = [(4, 8), (4, 4), (4, 2)]
SCOPES = ("weight", "wa", "kv", "wkv")

def msaq_2lvl(x, u, mg, d2):
    xf = x.reshape(-1, K1).to(torch.float32)
    nb = xf.shape[0]
    sabs = xf.reshape(nb, K1 // K2, K2).abs().amax(-1).clamp(min=1e-30)     # per-sub-block absmax
    ei = torch.floor(torch.log2(sabs))
    E1 = ei.amax(-1, keepdim=True)                                          # L1: block max exponent (E8M0)
    off = (E1 - ei).clamp(0, (1 << d2) - 1)                                 # L2: d2-bit microexp (steps down)
    s_sub = torch.exp2((E1 - off) - 6.0)                                    # per-sub-block scale (E8M0 -6 headroom)
    s_el = s_sub.unsqueeze(-1).expand(-1, -1, K2).reshape(nb, K1)           # per element
    q_max = (1 << (7 - u)) - 1
    q_un = torch.round(xf / (s_el * float(1 << u))).clamp(-q_max, q_max)
    x_un = q_un * (s_el * float(1 << u))
    s_min, s_max = -(1 << (u - 1)), (1 << (u - 1)) - 1
    r = (xf - x_un) / s_el                                                  # dimensionless residual
    r_avg = r.reshape(nb, K1 // mg, mg).mean(-1, keepdim=True).expand(-1, -1, mg).reshape(nb, K1)
    shared = torch.round(r_avg).clamp(s_min, s_max)
    return (x_un + shared * s_el).reshape(x.shape)

_D2 = {"v": 1}
def Q(x, u, mg): return msaq_2lvl(x, u, mg, _D2["v"])

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
def unpatch_act(orig):
    for m, f in orig.items(): m.forward = f

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

    bf = ppl(model, ids)
    print(f"wikitext-2: {ids.size(1):,} tokens | BF16 PPL = {bf:.4f} | L1=E8M0/32, L2=microexp/2 | within 3%\n", flush=True)

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
    for d2 in D2S:
        _D2["v"] = d2
        tag = "d2=0 (=msaq_signed baseline)" if d2 == 0 else f"d2={d2} (two-level, {d2}-bit microexp/2)"
        print(f"================  {tag}  ================", flush=True)
        print(f"{'scope':>11} | " + " ".join(f"u{u}/mg{mg:<2}" for u, mg in CONFIGS) + " | within 3%?", flush=True)
        for scope in SCOPES:
            cells, oks = [], []
            for u, mg in CONFIGS:
                pct = run(scope, u, mg); cells.append(f"{pct:>+6.2f}%"); oks.append("OK" if pct <= 3 else "FAIL")
            print(f"{LABEL[scope]:>11} | " + "  ".join(cells) + " | " + " ".join(oks), flush=True)
        print(flush=True)
