"""MSAQ on MXFP8: apply mantissa-sharing to FLOAT8 elements (E4M3 / E5M2 / E3M4),
not the INT8 of MXINT8. block=32, per-block E8M0 scale (the "MX"). Each element is FP8
(sign + eb exp + mb mantissa); MSAQ keeps (mb-u) mantissa bits per element and SHARES the
low u mantissa bits across mg elements -> per-element exponent (dynamic range) is preserved,
only the fine mantissa is shared. The shared correction is normalized to each element's own
mantissa LSB (2^(e_i-(mb-u))) so it applies correctly across elements with different exponents
(the FP analog of MXINT8-MSAQ, where all elems share one linear scale).

Cross-checks (selftest): u=0 -> plain MXFP8 baseline; mg=1 -> full FP8 (no loss). wikitext-2
PPL across the 4 scopes (weight / weight+act / KV / weight+KV), mirroring scale_fmt_ppl.py.

Run (numeric validation, no model, this repo's .venv):
    CUDA_VISIBLE_DEVICES=0 python precision/msaq_mxfp8_ppl.py --selftest
Run (PPL, needs transformers + Llama-3.1-8B in your precision env):
    CUDA_VISIBLE_DEVICES=0 python precision/msaq_mxfp8_ppl.py > precision/msaq_mxfp8_ppl.txt 2>&1
"""
import sys, torch

BLOCK = 32
# (eb, mb) per format. maxexp/maxval follow scale_fmt_ppl convention (all-ones exp = normal).
FMT = {"e4m3": (4, 3), "e5m2": (5, 2), "e3m4": (3, 4)}
SCOPES = ("weight", "wa", "kv", "wkv")


def _fmt_params(eb, mb):
    bias = 2 ** (eb - 1) - 1
    emin = 1 - bias
    maxexp = (2 ** eb - 1) - bias
    maxval = (2 - 2.0 ** (-mb)) * (2.0 ** maxexp)
    return emin, maxexp, maxval


def _fp_round(y, e, step, maxval):
    """Round y to the FP grid at per-element exponent e (quantum=step), clamp to +-maxval."""
    q = torch.round(y / step) * step
    return q.clamp(-maxval, maxval)


def msaq_mxfp8(x, u, mg, eb, mb):
    """MSAQ-MXFP8: FP8(E{eb}M{mb}) elements with low-u mantissa bits shared over mg.
    u=0 -> plain MXFP8; u<mb keeps (mb-u) per-element mantissa + u shared."""
    assert 0 <= u <= mb, f"u={u} must be in [0,{mb}]"
    emin, maxexp, maxval = _fmt_params(eb, mb)
    xf = x.reshape(-1, BLOCK).to(torch.float32)
    absmax = xf.abs().amax(-1, keepdim=True).clamp(min=1e-30)
    # MX block scale: bring block absmax near the format max (power-of-2 E8M0).
    s_base = torch.exp2(torch.floor(torch.log2(absmax)) - float(maxexp))
    y = xf / s_base
    # per-element exponent (clamped to the format's normal/subnormal range)
    e = torch.floor(torch.log2(y.abs().clamp(min=1e-30))).clamp(min=float(emin), max=float(maxexp))
    mb_up = mb - u
    step_up = torch.exp2(e - float(mb_up))                 # upper-mantissa quantum (2^u coarser than full)
    upper = _fp_round(y, e, step_up, maxval)               # per-element FP8 with (mb-u) mantissa bits
    if u == 0:
        return (upper * s_base).reshape(x.shape)
    res = y - upper
    frac = res / step_up                                   # normalized residual in [-0.5, 0.5)
    frac_avg = frac.reshape(frac.shape[0], -1, mg).mean(-1, keepdim=True).expand(-1, -1, mg).reshape(frac.shape)
    half = 1 << (u - 1)
    shared = (torch.round(frac_avg * float(1 << u)).clamp(-half, half - 1)) / float(1 << u)   # u-bit signed fraction
    rec = upper + shared * step_up
    return (rec * s_base).reshape(x.shape)


def bits_mxfp8(eb, mb, u, mg):
    """Effective bits/element: sign + exp + (mb-u) upper mantissa + u/mg shared + E8M0(8)/BLOCK."""
    return 1 + eb + (mb - u) + u / mg + 8.0 / BLOCK


# ---------------------------------------------------------------------------
# Numeric validation (runs anywhere with torch; no model). QSNR on weight-like data.
# ---------------------------------------------------------------------------
def _qsnr(x, xq):
    e = (x - xq).pow(2).mean()
    return 10.0 * torch.log10(x.pow(2).mean() / e.clamp(min=1e-45)).item()


def msaq_mxint8(x, u, mg):
    """Reference: the deployed MXINT8-MSAQ (== scale_fmt_ppl msaq_scalefmt e8m0). INT8 element,
    upper (8-u)-bit + shared u-bit over mg, E8M0 block scale. bits = (8-u)+u/mg+8/32."""
    xf = x.reshape(-1, BLOCK).to(torch.float32)
    absmax = xf.abs().amax(-1, keepdim=True).clamp(min=1e-30)
    if u == 0:                                              # plain MXINT8 (no sharing)
        s = torch.exp2(torch.ceil(torch.log2(absmax / 127.0)))
        return (torch.round(xf / s).clamp(-127, 127) * s).reshape(x.shape)
    q_max = (1 << (7 - u)) - 1
    s_base = torch.exp2(torch.floor(torch.log2(absmax / 64.0)))
    s_un = s_base * float(1 << u)
    q_un = torch.round(xf / s_un).clamp(-q_max, q_max)
    res = xf - q_un * s_un
    s_min, s_max = -(1 << (u - 1)), (1 << (u - 1)) - 1
    res_avg = res.reshape(res.shape[0], -1, mg).mean(-1, keepdim=True).expand(-1, -1, mg).reshape(res.shape)
    shared = torch.round(res_avg / s_base).clamp(s_min, s_max)
    return (q_un * s_un + shared * s_base).reshape(x.shape)


def bits_mxint8(u, mg):
    return (8 - u) + u / mg + 8.0 / BLOCK


def selftest():
    torch.manual_seed(0)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    # representative tensors: Gaussian weights + a heavier-tailed (Student-t-ish) one for outliers
    W = torch.randn(4096, 4096, device=dev) * 0.02
    Wt = (torch.randn(4096, 4096, device=dev) * 0.02) * (1 + 3 * torch.rand(4096, 1, device=dev) ** 4)
    # Ws: high INTRA-block dynamic range (log-uniform magnitudes span ~2^10 within a 32-block).
    # This is where FP8 (per-element exponent) should beat INT8 (uniform grid starves small elems).
    sgn = torch.sign(torch.randn(4096, 4096, device=dev))
    Ws = sgn * torch.exp2(-10.0 * torch.rand(4096, 4096, device=dev))
    print(f"=== MSAQ-MXFP8 selftest (device={dev}) ===")
    print("cross-checks (should hold exactly):")
    for f, (eb, mb) in FMT.items():
        base = msaq_mxfp8(W, 0, 8, eb, mb)                 # u=0
        mg1 = msaq_mxfp8(W, min(1, mb), 1, eb, mb)         # mg=1 (full FP8)
        full = msaq_mxfp8(W, 0, 1, eb, mb)                 # plain FP8 reference
        d_u0 = (base - msaq_mxfp8(W, 0, 1, eb, mb)).abs().max().item()
        d_mg1 = (mg1 - full).abs().max().item()
        print(f"  {f}: u0==MXFP8 (Δ={d_u0:.1e}) | mg1 QSNR≈full (Δmax={d_mg1:.1e})")
    print("\nQSNR (dB) vs effective bits/elem  [W=Gaussian | Wt=heavy-tail | Ws=hi intra-block range]:")
    print(f"{'fmt':>6} {'cfg':>10} {'bits':>5} | {'QSNR_W':>7} {'QSNR_Wt':>7} {'QSNR_Ws':>7}")
    for f, (eb, mb) in FMT.items():
        cfgs = [(0, 1)] + [(u, mg) for u in range(1, mb + 1) for mg in (2, 4, 8)]
        for (u, mg) in cfgs:
            qW = _qsnr(W, msaq_mxfp8(W, u, mg, eb, mb))
            qWt = _qsnr(Wt, msaq_mxfp8(Wt, u, mg, eb, mb))
            qWs = _qsnr(Ws, msaq_mxfp8(Ws, u, mg, eb, mb))
            tag = "MXFP8" if u == 0 else f"u{u}/mg{mg}"
            print(f"{f:>6} {tag:>10} {bits_mxfp8(eb,mb,u,mg):>5.2f} | {qW:>7.2f} {qWt:>7.2f} {qWs:>7.2f}")
        print()
    print("reference — MXINT8-MSAQ (deployed, INT8 element) at matched bits:")
    print(f"{'fmt':>6} {'cfg':>10} {'bits':>5} | {'QSNR_W':>7} {'QSNR_Wt':>7} {'QSNR_Ws':>7}")
    for (u, mg) in [(0, 1), (2, 8), (2, 4), (3, 8), (3, 4)]:
        qW = _qsnr(W, msaq_mxint8(W, u, mg)); qWt = _qsnr(Wt, msaq_mxint8(Wt, u, mg))
        qWs = _qsnr(Ws, msaq_mxint8(Ws, u, mg))
        tag = "MXINT8" if u == 0 else f"u{u}/mg{mg}"
        print(f"{'int8':>6} {tag:>10} {bits_mxint8(u,mg):>5.2f} | {qW:>7.2f} {qWt:>7.2f} {qWs:>7.2f}")
    print()


# ---------------------------------------------------------------------------
# wikitext-2 PPL across scopes (needs transformers + datasets + the model).
# ---------------------------------------------------------------------------
def run_ppl():
    import os
    import torch.nn.functional as F
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from datasets import load_dataset
    MODEL = os.environ.get("MSAQ_MODEL", "meta-llama/Llama-3.1-8B-Instruct"); DEV = "cuda"
    MAXLEN, STRIDE, MAX_WINDOWS = 2048, 1024, 30
    LINEAR_KEYS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
    # per-format (u,mg) configs (u<=mb)
    CFG = {"e4m3": [(1, 8), (1, 4), (2, 4)], "e5m2": [(1, 8), (1, 4), (2, 4)],
           "e3m4": [(1, 8), (2, 4), (3, 4)]}
    _S = {"eb": 4, "mb": 3, "base": "fp8"}
    def Q(x, u, mg):
        if _S["base"] == "int8": return msaq_mxint8(x, u, mg).to(x.dtype)
        return msaq_mxfp8(x, u, mg, _S["eb"], _S["mb"]).to(x.dtype)

    def is_target(n, m): return isinstance(m, torch.nn.Linear) and any(p in n for p in LINEAR_KEYS)
    def patch_act(targets, u, mg):
        orig = {}
        for _, m in targets:
            orig[m] = m.forward
            def mk(mod):
                w, b = mod.weight, mod.bias
                def f(x): return F.linear(Q(x, u, mg), w, b)
                return f
            m.forward = mk(m)
        return orig
    def unpatch_act(orig):
        for m, f in orig.items(): m.forward = f
    _real_sdpa = F.scaled_dot_product_attention
    _KV = {"on": False, "u": 0, "mg": 0}
    def _kv_sdpa(q, k, v, *a, **kw):
        if _KV["on"]:
            k = Q(k, _KV["u"], _KV["mg"]); v = Q(v, _KV["u"], _KV["mg"])
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

    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16,
                                                 attn_implementation="sdpa").to(DEV).eval()
    try:
        _wt = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    except Exception:                                      # newer hub/datasets: use the parquet mirror
        _wt = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    ids = tok("\n\n".join(t for t in _wt["text"] if t.strip()), return_tensors="pt").input_ids
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
    def sweep(name, cfgs, bitfn):
        print(f"================  {name}  (block={BLOCK})  ================", flush=True)
        print(f"{'scope':>11} | " + " ".join(f"u{u}/mg{mg}({bitfn(u,mg):.2f}b)" for u, mg in cfgs) + " | within 3%?", flush=True)
        for scope in SCOPES:
            cells, oks = [], []
            for u, mg in cfgs:
                pct = run(scope, u, mg); cells.append(f"{pct:>+6.2f}%"); oks.append("OK" if pct <= 3 else "FAIL")
            print(f"{LABEL[scope]:>11} | " + "  ".join(cells) + " | " + " ".join(oks), flush=True)
        print(flush=True)
    for f, (eb, mb) in FMT.items():
        _S["base"], _S["eb"], _S["mb"] = "fp8", eb, mb
        sweep(f"MSAQ-MXFP8 element = {f.upper()}", CFG[f], lambda u, mg, eb=eb, mb=mb: bits_mxfp8(eb, mb, u, mg))
    # MXINT8-MSAQ at bit-matched configs (u1/mg8=7.38b, u2/mg4=6.75b, u3/mg4=6.00b == E3M4's configs)
    _S["base"] = "int8"
    sweep("MSAQ-MXINT8 (reference, INT8 element)", [(1, 8), (2, 4), (3, 4)], bits_mxint8)


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest()
    else:
        run_ppl()
