"""Two-tier MSAQ: native MXFP base + contraction-axis shared u-bit residual (the GEMM-decompose
reformulation). Distinct from msaq_mxfp8 (which shares MANTISSA bits, u<=mb): here the residual is
a SEPARATE additive u-bit term shared over a group of g=32 along the CONTRACTION axis K, so the
correction becomes a 1/32-FLOP GEMM (Y = A*W_hat + A_bar*R_bar) that hides in the base MMA shadow.

    W ~= W_hat + B(R_bar),   W_hat = MXFP_base(W)  (E2M1 / E2M3, native, no per-elem unpack)
    R = W - W_hat,           R_bar[g,n] = Q_u( mean_{k in g} R[k,n] )   (DC residual, §1)

This file is the §7-first gate: iso-bit weight PPL on Llama-3.1-8B. Does E2M1+shared-residual push
the bits-vs-PPL frontier BELOW the native rungs (MXFP4=4.25, E2M3=6.25)? efb = reconstruction-L2
coordinate descent (no calibration; the A-weighted §4 objective is a later refinement).

Selftest (no model, validates the math on GPU):
    .venv/bin/python precision/two_tier_ppl.py --selftest
PPL:
    MSAQ_MODEL=NousResearch/Meta-Llama-3.1-8B CUDA_VISIBLE_DEVICES=0 \
        .venv/bin/python precision/two_tier_ppl.py > precision/two_tier_ppl_llama31_8b.txt 2>&1
"""
import os, sys, torch
from mxfp6_verify import e2m3_clean, _fp6_grid
from msaq_mxfp8_ppl import msaq_mxint8_efb, BLOCK


def _snap_grid(ay, grid):
    """Snap nonneg magnitudes `ay` to the nearest point of sorted magnitude `grid`."""
    idx = torch.bucketize(ay, grid)
    lo = (idx - 1).clamp(0, grid.numel() - 1); hi = idx.clamp(0, grid.numel() - 1)
    gl, gh = grid[lo], grid[hi]
    return torch.where((ay - gl).abs() <= (gh - ay).abs(), gl, gh)


def two_tier(x, eb, mb, u, g=BLOCK, efb_iters=2):
    """Native E{eb}M{mb} MX base + contraction-axis g-grouped additive u-bit shared residual.

    The MX block (BLOCK=32) runs along the last (=contraction K) axis of a [N,K] weight, so the
    residual group g=32 IS the MX block: one signed u-bit value per block, on a per-group E8M0
    residual scale (the honest +8/g-bit "residual exponent" variant of §3). u=0 -> exactly the
    native base (cross-check). efb_iters>0: re-snap the native base against the fixed shared term
    (reconstruction-L2 coordinate descent) -> strictly non-increasing recon L2, base stays native.
    """
    assert g == BLOCK, "residual group must equal the MX block (contraction-axis aligned)"
    grid = _fp6_grid(eb, mb)
    maxval = grid[-1].item()
    maxexp = ((1 << eb) - 1) - ((1 << (eb - 1)) - 1)
    xf = x.reshape(-1, g).to(torch.float32)
    absmax = xf.abs().amax(-1, keepdim=True).clamp(min=1e-30)
    s = torch.exp2(torch.floor(torch.log2(absmax)) - float(maxexp))      # fixed per-block E8M0 (base)
    y = xf / s                                                            # scaled domain (block scale 1)

    def snap_base(t):                                                     # native MXFP snap, sign-mag
        sgn = torch.sign(t); ay = t.abs().clamp(max=maxval)
        return sgn * _snap_grid(ay, grid)

    base = snap_base(y)
    if u == 0:
        return (base * s).reshape(x.shape)
    qmax = (1 << (u - 1)) - 1                                             # signed u-bit grid [-2^(u-1)+1 .. qmax]
    shared = None
    for it in range(max(1, efb_iters + 1)):
        r = (y - base)                                                    # residual, scaled domain
        m = r.mean(-1, keepdim=True)                                      # DC component per group (§1)
        d = torch.exp2(torch.floor(torch.log2(m.abs().clamp(min=1e-30) / max(qmax, 1))))  # per-group E8M0
        shared = torch.round(m / d).clamp(-qmax - 1, qmax) * d            # u-bit signed, broadcast
        if it == efb_iters:
            break
        base = snap_base(y - shared)                                      # efb: re-snap native base
    return ((base + shared) * s).reshape(x.shape)


def bits_two_tier(eb, mb, u, g=BLOCK):
    """sign+exp+mant (native, =1+eb+mb) + E8M0(8)/BLOCK base scale + (u + E8M0(8)) shared / g."""
    return (1 + eb + mb) + 8.0 / BLOCK + (u + 8.0) / g if u else (1 + eb + mb) + 8.0 / BLOCK


# --- columns for the iso-bit frontier (weight). native rungs + two-tier fills between them. ---
COLUMNS = [
    ("MXFP4(E2M1)",   bits_two_tier(2, 1, 0), lambda x: two_tier(x, 2, 1, 0)),   # native FP4 rung (4.25)
    ("E2M1+u2",       bits_two_tier(2, 1, 2), lambda x: two_tier(x, 2, 1, 2)),   # fractional fill
    ("E2M1+u4",       bits_two_tier(2, 1, 4), lambda x: two_tier(x, 2, 1, 4)),   # the target (E2M1-rescue)
    ("MXINT8-MSAQ.efb", 6.00,                 lambda x: msaq_mxint8_efb(x, 3, 4, 2)),  # deployed ref
    ("E2M3(native)",  bits_two_tier(2, 3, 0), lambda x: two_tier(x, 2, 3, 0)),   # native FP6 rung (6.25)
    ("E2M3+u4",       bits_two_tier(2, 3, 4), lambda x: two_tier(x, 2, 3, 4)),   # low-risk fill
]


def _qsnr(x, xq):
    e = (x - xq).double().pow(2).mean()
    return 10.0 * torch.log10(x.double().pow(2).mean() / e.clamp(min=1e-45)).item()


def selftest():
    torch.manual_seed(0)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    W = torch.randn(4096, 4096, device=dev) * 0.02
    Wt = (torch.randn(4096, 4096, device=dev) * 0.02) * (1 + 3 * torch.rand(4096, 1, device=dev) ** 4)
    print(f"=== two_tier selftest (device={dev}) ===")
    # cross-check: u=0 == native e2m3_clean base
    for (eb, mb) in [(2, 1), (2, 3)]:
        a = two_tier(W, eb, mb, 0)
        b = e2m3_clean(W, eb, mb)[0]
        print(f"  E2M{mb} u0 == native base: dmax={ (a-b).abs().max().item():.1e}")
    # efb monotonicity: recon MSE non-increasing in efb_iters
    print("\n  efb recon-MSE monotone (E2M1+u4, should be non-increasing):")
    for it in range(4):
        mse = (W - two_tier(W, 2, 1, 4, efb_iters=it)).pow(2).mean().item()
        print(f"    iters={it}: MSE={mse:.6e}")
    # QSNR vs bits frontier
    print(f"\n  {'cfg':>16} {'bits':>6} | {'QSNR_W':>8} {'QSNR_Wt':>8}")
    for nm, b, qfn in COLUMNS:
        print(f"  {nm:>16} {b:>6.3f} | {_qsnr(W, qfn(W)):>8.2f} {_qsnr(Wt, qfn(Wt)):>8.2f}")


# ----------------------------------------------------------------------------- PPL (mxfp6_ppl skeleton)
def run_ppl():
    import torch.nn.functional as F
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from datasets import load_dataset
    DEV = "cuda"; MAXLEN, STRIDE, MAX_WINDOWS = 2048, 1024, 30
    LINEAR_KEYS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
    MODEL = os.environ.get("MSAQ_MODEL", "meta-llama/Llama-3.1-8B-Instruct")

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

    bf = ppl(model, ids)
    print(f"wikitext-2: {ids.size(1):,} tokens | BF16 PPL = {bf:.4f} | block={BLOCK} | weight-only frontier\n", flush=True)
    print(f"{'cfg':>16} {'bits':>6} | {'PPL':>9} {'dPPL%':>8}", flush=True)
    for nm, b, qfn in COLUMNS:
        restore()
        for _, m in targets: m.weight.data.copy_(qfn(master[m].to(DEV)).to(m.weight.dtype))
        p = ppl(model, ids); restore()
        print(f"{nm:>16} {b:>6.3f} | {p:>9.4f} {(p/bf-1)*100:>+7.2f}%", flush=True)


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest()
    else:
        run_ppl()
