"""Audit the MXFP6-E2M3 > MSAQ finding for confounds (per-tensor scale, impl bug, bit unfairness).

Checks, on REAL Llama-3.1-8B weights + synthetic:
  (A) independent clean E2M3 (enumerated FP6 grid) == msaq_mxfp8(u=0,2,3)  -> rules out impl bug
  (B) every format uses the SAME per-32-block E8M0 (power-of-2) scale       -> rules out per-tensor leak
  (C) MXINT6 (clean, 6.25b) as a neutral 6.25b baseline                     -> is E2M3 anomalous, or is
                                                                                6.25b-per-block just good?
  (D) bit-matched: give MSAQ MORE bits (E3M4 6.5b) — does it catch E2M3 6.25b?
Run: CUDA_VISIBLE_DEVICES=0 python precision/mxfp6_verify.py
"""
import glob, torch
from safetensors import safe_open
from msaq_mxfp8_ppl import msaq_mxfp8, msaq_mxint8, BLOCK

DEV = "cuda"


def qsnr(x, xq):
    e = (x - xq).double().pow(2).mean()
    return 10.0 * torch.log10(x.double().pow(2).mean() / e.clamp(min=1e-45)).item()


# ---- (A) independent E2M3: enumerate the actual FP6 grid, per-32-block E8M0, nearest-code ----
def _fp6_grid(eb, mb):
    """All non-negative representable magnitudes of E{eb}M{mb} (incl subnormals), as a sorted tensor."""
    bias = (1 << (eb - 1)) - 1
    vals = {0.0}
    for ef in range(0, (1 << eb)):
        for mf in range(0, (1 << mb)):
            if ef == 0:
                v = (mf / (1 << mb)) * (2.0 ** (1 - bias))          # subnormal
            else:
                v = (1.0 + mf / (1 << mb)) * (2.0 ** (ef - bias))   # normal
            vals.add(v)
    return torch.tensor(sorted(vals), dtype=torch.float32, device=DEV)


def e2m3_clean(x, eb=2, mb=3):
    """Standard MXFP6 quant, fully independent of msaq_mxfp8: per-32-block E8M0 scale that maps the
    block max onto the format max, then snap each scaled element to the NEAREST point of the real
    E{eb}M{mb} grid (sign-magnitude). No sharing, no per-tensor scale."""
    grid = _fp6_grid(eb, mb)                                       # (G,) magnitudes
    maxval = grid[-1].item()
    maxexp = ((1 << eb) - 1) - ((1 << (eb - 1)) - 1)
    xf = x.reshape(-1, BLOCK).to(torch.float32)
    absmax = xf.abs().amax(-1, keepdim=True).clamp(min=1e-30)
    s = torch.exp2(torch.floor(torch.log2(absmax)) - float(maxexp))  # E8M0 (power of 2), per 32-block
    y = (xf / s)
    sgn = torch.sign(y); ay = y.abs().clamp(max=maxval)
    # nearest grid magnitude
    idx = torch.bucketize(ay, grid)
    lo = (idx - 1).clamp(0, grid.numel() - 1); hi = idx.clamp(0, grid.numel() - 1)
    gl, gh = grid[lo], grid[hi]
    snapped = torch.where((ay - gl).abs() <= (gh - ay).abs(), gl, gh)
    return ((sgn * snapped) * s).reshape(x.shape), s


# ---- (C) neutral 6.25b baseline: per-32-block E8M0 + symmetric 6-bit int (no sharing) ----
def mxint6_clean(x):
    xf = x.reshape(-1, BLOCK).to(torch.float32)
    absmax = xf.abs().amax(-1, keepdim=True).clamp(min=1e-30)
    s = torch.exp2(torch.ceil(torch.log2(absmax / 31.0)))          # E8M0; 31 = 6-bit signed max
    return (torch.round(xf / s).clamp(-31, 31) * s).reshape(x.shape)


def bits(name):
    return {"E2M3(msaq)": 6.25, "E2M3(clean)": 6.25, "MXINT6": 6.25, "E3M4-MSAQ6.0": 6.0,
            "E3M4-MSAQ6.5": 6.5, "MXINT8-MSAQ6.0": 6.0, "E4M3(8.25)": 8.25}[name]


FORMATS = [
    ("E2M3(msaq)",     lambda x: msaq_mxfp8(x, 0, 1, 2, 3)),
    ("E2M3(clean)",    lambda x: e2m3_clean(x)[0]),
    ("MXINT6",         mxint6_clean),
    ("E3M4-MSAQ6.0",   lambda x: msaq_mxfp8(x, 3, 4, 3, 4, efb_iters=2)),
    ("E3M4-MSAQ6.5",   lambda x: msaq_mxfp8(x, 2, 8, 3, 4, efb_iters=2)),   # u2/mg8 = 6.5b
    ("MXINT8-MSAQ6.0", lambda x: msaq_mxint8(x, 3, 4)),
    ("E4M3(8.25)",     lambda x: msaq_mxfp8(x, 0, 1, 4, 3)),
]


def report(tag, W):
    W = W.to(DEV).float()
    print(f"\n== {tag}  shape={tuple(W.shape)}  absmax={W.abs().max():.3g} ==")
    print(f"   {'format':>16} {'bits':>5} {'QSNR(dB)':>9}")
    for nm, fn in FORMATS:
        print(f"   {nm:>16} {bits(nm):>5.2f} {qsnr(W, fn(W).float()):>9.2f}")


def main():
    torch.manual_seed(0)
    # (A) impl cross-check
    Wt = torch.randn(2048, 2048, device=DEV) * 0.02
    a = msaq_mxfp8(Wt, 0, 1, 2, 3).float(); b = e2m3_clean(Wt)[0].float()
    print(f"(A) E2M3 msaq vs independent-clean: max|Δ|={(a-b).abs().max():.2e}  "
          f"(0 ⇒ msaq_mxfp8(u=0,2,3) IS standard MXFP6-E2M3)")
    # (B) scale is E8M0 (power of two)?
    _, s = e2m3_clean(Wt)
    log2s = torch.log2(s); is_pow2 = (log2s - log2s.round()).abs().max().item()
    print(f"(B) E2M3 per-block scale: blocks={s.numel()} (=numel/32={Wt.numel()//BLOCK}), "
          f"max frac(log2 s)={is_pow2:.1e} (0 ⇒ pure E8M0 power-of-2, per-32-block, NOT per-tensor)")

    report("synthetic Gaussian W*0.02", Wt)
    # (D) real Llama weights
    f = sorted(glob.glob("/home/yubin/.cache/huggingface/hub/models--NousResearch--Meta-Llama-3.1-8B/"
                         "snapshots/*/model-00001-of-00004.safetensors"))[0]
    with safe_open(f, framework="pt") as fh:
        for key in fh.keys():
            if key.endswith("layers.0.mlp.down_proj.weight") or key.endswith("layers.0.self_attn.q_proj.weight"):
                report(f"real Llama {key.split('model.')[-1]}", fh.get_tensor(key))


if __name__ == "__main__":
    main()
