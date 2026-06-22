"""Scale-factor format sweep: block=32 fixed, swap the per-block scale format
E8M0 (current, power-of-2 only) -> E4M3 / UE5M3 (3 mantissa bits). Does a finer
(mantissa'd) block scale make MSAQ u=4 robust where E8M0 fails? wikitext-2 PPL
across the 4 scopes (weight / weight+act / KV / weight+KV).

E8M0 snaps the block scale to a power of 2 (0 mantissa bits) -> coarse fit to the
block absmax. E4M3 (4exp/3mant) and UE5M3 (5exp/3mant) carry 3 mantissa bits so the
scale can be non-pow2 -> tighter fit -> less quant error. CAVEAT: E4M3's 4-bit
exponent (min normal 2^-6) is too narrow for a raw per-block scale (~2^-10) -> it
would underflow. So E4M3/UE5M3 use the deployed 2-LEVEL scheme (NVFP4-style):
per-tensor E8M0 global × per-block format micro-scale. E8M0 stays single-level (baseline).
fmt=e8m0 reproduces msaq_signed exactly (cross-check). Criterion: within 3% of BF16.
Run: CUDA_VISIBLE_DEVICES=0 python precision/scale_fmt_ppl.py > precision/scale_fmt_ppl.txt 2>&1
"""
import torch, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import lightms_qsnr

MODEL = "meta-llama/Llama-3.1-8B-Instruct"; DEV = "cuda"
MAXLEN, STRIDE, MAX_WINDOWS = 2048, 1024, 30
LINEAR_KEYS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
BLOCK = 32
FMTS = ("e8m0", "e4m3", "ue5m3")
CONFIGS = [(4, 8), (4, 4), (4, 2)]
SCOPES = ("weight", "wa", "kv", "wkv")

def _floatN_floor(s, eb, mb):                            # largest E{eb}M{mb} value <= s
    bias = 2 ** (eb - 1) - 1
    emin = 1 - bias
    sub = 2.0 ** (emin - mb)
    maxexp = (2 ** eb - 1) - bias
    maxval = (2 - 2.0 ** (-mb)) * (2.0 ** maxexp)
    e = torch.floor(torch.log2(s.clamp(min=1e-45))).clamp(min=emin)
    step = torch.exp2(e - mb)
    q = torch.floor(s / step) * step                     # floor to the format grid (no clip)
    q = q.clamp(max=maxval)
    return torch.where(s < sub, torch.zeros_like(q), q)

def msaq_scalefmt(x, u, mg, fmt):                       # MSAQ-signed with chosen scale format
    xf = x.reshape(-1, BLOCK).to(torch.float32)
    q_max = (1 << (7 - u)) - 1
    absmax = xf.abs().amax(-1, keepdim=True).clamp(min=1e-30)
    if fmt == "e8m0":                                    # DEPLOYED baseline (== msaq_signed): /64 headroom
        s_base = torch.exp2(torch.floor(torch.log2(absmax / 64.0)))     # pow2-only, single level
    else:                                                # finer grid -> tightest no-clip fit (best-vs-best)
        eb, mb = (4, 3) if fmt == "e4m3" else (5, 3)
        target = absmax / (q_max * float(1 << u))        # ideal s_base (max elem -> q_max); format-floor = no clip
        gabs = (x.abs().amax().clamp(min=1e-30)) / (q_max * float(1 << u))
        g = torch.exp2(torch.round(torch.log2(gabs)))    # per-tensor pow2 global (brings micro into format range)
        s_base = (g * _floatN_floor(target / g, eb, mb)).clamp(min=1e-30)
    s_un = s_base * float(1 << u)
    q_un = torch.round(xf / s_un).clamp(-q_max, q_max)
    x_un = q_un * s_un
    res = xf - x_un
    s_min, s_max = -(1 << (u - 1)), (1 << (u - 1)) - 1
    res_avg = res.reshape(res.shape[0], -1, mg).mean(-1, keepdim=True).expand(-1, -1, mg).reshape(res.shape)
    shared = torch.round(res_avg / s_base).clamp(s_min, s_max)
    return (x_un + shared * s_base).reshape(x.shape)

_FMT = {"v": "e8m0"}
def Q(x, u, mg): return msaq_scalefmt(x, u, mg, _FMT["v"])

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
    print(f"wikitext-2: {ids.size(1):,} tokens | BF16 PPL = {bf:.4f} | block={BLOCK} | criterion within 3%\n", flush=True)

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
    for fmt in FMTS:
        _FMT["v"] = fmt
        print(f"================  scale format = {fmt.upper()}  (block={BLOCK})  ================", flush=True)
        print(f"{'scope':>11} | " + " ".join(f"u{u}/mg{mg:<2}" for u, mg in CONFIGS) + " | within 3%?", flush=True)
        for scope in SCOPES:
            cells, oks = [], []
            for u, mg in CONFIGS:
                pct = run(scope, u, mg); cells.append(f"{pct:>+6.2f}%"); oks.append("OK" if pct <= 3 else "FAIL")
            print(f"{LABEL[scope]:>11} | " + "  ".join(cells) + " | " + " ".join(oks), flush=True)
        print(flush=True)
