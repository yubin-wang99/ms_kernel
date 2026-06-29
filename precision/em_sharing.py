"""EM_sharing: two-level micro-format = native MX base (gb=32, E8M0) + contraction-axis residual
reconstructed with DECOUPLED exponent/mantissa granularities.

    R = W - W_hat,    W_hat = base(W)   (FP4/FP6 native, or INT4/INT6 control)
    R_hat[k,n] = m[n, floor(k/gm)] * 2^{ e[n, floor(k/ge)] }

    m  = signed u-bit mantissa (significand), one per gm elements along K
    e  = be-bit residual exponent (envelope),  one per ge elements along K

Free variables {u, gm, ge, be} with gm,ge | 32 and one nesting the other. This GENERALIZES
`two_tier_ppl.two_tier`, which fixed ge=32 & a full E8M0 residual scale (i.e. be=8, gm=ge): there
mantissa and exponent shared ONE granularity. EM_sharing decouples them so storage splits into two
independent terms  B_res = u/gm + be/ge  — letting us put "fine" on whichever axis the base grid
rewards. Dual hypothesis (spec §4):
  - INT base  (uniform grid):     residual envelope FLAT     -> fine on MANTISSA (gm small, ge=32,be=0)
  - FP  base  (non-uniform grid): envelope ~ 2^elem-exponent -> fine on EXPONENT (ge small,be>0; gm=32)

This file is the precision gate: calibration-free reconstruction QSNR over the {u,gm,ge,be} grid on
real Llama-3.1-8B weights, to settle the dual hypothesis BEFORE the iso-bit PPL sweep. Bit-accounting
(per-block residual anchor: free-vs-honest) does NOT affect QSNR — it only shifts the bits axis — so
the dual verdict is accounting-independent.

Selftest (no model; validates the math on GPU):
    .venv/bin/python precision/em_sharing.py --selftest
Dual-hypothesis QSNR on real weights:
    CUDA_VISIBLE_DEVICES=0 .venv/bin/python precision/em_sharing.py
"""
import glob, sys, torch
from safetensors import safe_open
from mxfp6_verify import _fp6_grid
from two_tier_ppl import _snap_grid, two_tier

DEV = "cuda" if torch.cuda.is_available() else "cpu"
BLOCK = 32

# base spec: 'fp' -> E{eb}M{mb} grid; 'int' -> signed n-bit uniform
BASES = {
    "FP4":  dict(kind="fp",  eb=2, mb=1),   # E2M1 / NVFP4 tier (non-uniform, native)
    "FP6":  dict(kind="fp",  eb=2, mb=3),   # E2M3 tier         (non-uniform, native)
    "INT4": dict(kind="int", n=4),          # uniform grid (control / accuracy frontier; not native)
    "INT6": dict(kind="int", n=6),          # uniform grid (control / accuracy frontier; not native)
}


def _base_encode(xf, spec):
    """Per-block E8M0 scale s + base snap in the scaled domain. Returns (s, snap_fn) where snap_fn maps
    a scaled-domain tensor to its base codes (same shape). xf is [rows, BLOCK]."""
    absmax = xf.abs().amax(-1, keepdim=True).clamp(min=1e-30)
    if spec["kind"] == "fp":
        eb, mb = spec["eb"], spec["mb"]
        grid = _fp6_grid(eb, mb)
        maxval = grid[-1].item()
        maxexp = ((1 << eb) - 1) - ((1 << (eb - 1)) - 1)
        s = torch.exp2(torch.floor(torch.log2(absmax)) - float(maxexp))   # E8M0
        def snap(t):
            sgn = torch.sign(t); ay = t.abs().clamp(max=maxval)
            return sgn * _snap_grid(ay, grid)
        return s, snap
    else:
        qmax = (1 << (spec["n"] - 1)) - 1
        s = torch.exp2(torch.ceil(torch.log2(absmax / qmax)))             # E8M0
        def snap(t):
            return torch.round(t).clamp(-qmax, qmax)
        return s, snap


def em_quant(x, base, u, gm, ge, be, efb_iters=2):
    """EM_sharing reconstruction. base in BASES. u-bit signed mantissa per gm, be-bit residual
    exponent per ge, both along the contraction axis (last). u=0 -> pure native base.

    Closed-form, calibration-free (spec §5 option 1, reconstruction-L2). efb_iters>0 re-snaps the
    native base against the fixed residual (coordinate descent; base stays native)."""
    assert gm in (4, 8, 16, 32) and ge in (4, 8, 16, 32)
    spec = BASES[base]
    xf = x.reshape(-1, BLOCK).to(torch.float32)
    s, snap = _base_encode(xf, spec)
    y = xf / s                                                            # scaled domain (block scale 1)
    bcode = snap(y)
    if u == 0:
        return (bcode * s).reshape(x.shape)

    hi = (1 << (u - 1)) - 1                                               # signed u-bit: [lo, hi]
    lo = -(1 << (u - 1))
    R, C = xf.shape                                                       # C == BLOCK
    nemax = (1 << be) - 1 if be else 0                                    # be-bit downward delta range

    def encode_residual(r):
        # --- exponent envelope e[k] : be-bit per-ge microexponent, downward delta from block ref ---
        eg = r.reshape(R, C // ge, ge)
        amax_ge = eg.abs().amax(-1, keepdim=True).clamp(min=1e-30)        # per-ge residual envelope
        raw = torch.floor(torch.log2(amax_ge / max(hi, 1)))              # exp so hi*2^e ~ envelope
        ref = torch.floor(torch.log2(r.abs().amax(-1, keepdim=True).clamp(min=1e-30) / max(hi, 1)))
        if be:
            delta = (ref.unsqueeze(1) - raw).round().clamp(0, nemax)
            e_ge = (ref.unsqueeze(1) - delta)                            # [R, C//ge, 1]
        else:
            e_ge = ref.unsqueeze(1).expand(-1, C // ge, -1)             # single per-block exponent
        e = e_ge.expand(-1, -1, ge).reshape(R, C)                        # broadcast to elements
        scale = torch.exp2(e)
        # --- mantissa m : DC of exponent-normalized residual, one signed u-bit value per gm ---
        rn = (r / scale).reshape(R, C // gm, gm)
        m = rn.mean(-1, keepdim=True).round().clamp(lo, hi)
        m = m.expand(-1, -1, gm).reshape(R, C)
        return m * scale                                                 # R_hat in scaled domain

    rhat = encode_residual(y - bcode)
    for _ in range(efb_iters):
        bcode = snap(y - rhat)                                           # efb: re-snap native base
        rhat = encode_residual(y - bcode)
    return ((bcode + rhat) * s).reshape(x.shape)


def bits_em(base, u, gm, ge, be, anchor="free"):
    """B_total = B_base + B_res.  B_res = u/gm + be/ge  (spec §2).
    B_base = format bits + E8M0 block scale 8/32. anchor='honest' adds a per-block residual exponent
    reference 8/32 when the residual is active (the §7 metadata-honesty variant)."""
    spec = BASES[base]
    fmt = (1 + spec["eb"] + spec["mb"]) if spec["kind"] == "fp" else spec["n"]
    b_base = fmt + 8.0 / BLOCK
    if u == 0:
        return b_base
    b_res = u / gm + be / ge
    if anchor == "honest":
        b_res += 8.0 / BLOCK
    return b_base + b_res


def qsnr(x, xq):
    e = (x - xq).double().pow(2).mean()
    return 10.0 * torch.log10(x.double().pow(2).mean() / e.clamp(min=1e-45)).item()


# ----------------------------------------------------------------------------------------- selftest
def selftest():
    torch.manual_seed(0)
    # synthetic with per-row scale heterogeneity (so envelope varies -> dual hypothesis has signal)
    W = torch.randn(4096, 4096, device=DEV) * 0.02
    Wt = (torch.randn(4096, 4096, device=DEV) * 0.02) * (1 + 3 * torch.rand(4096, 1, device=DEV) ** 4)
    print(f"=== em_sharing selftest (device={DEV}) ===")

    # (1) u=0 == native base (each base) and == two_tier u0 for FP
    print("\n  (1) u=0 reductions:")
    for base in BASES:
        a = em_quant(W, base, 0, 8, 8, 0)
        spec = BASES[base]
        if spec["kind"] == "fp":
            b = two_tier(W, spec["eb"], spec["mb"], 0)
            print(f"     {base:>5} u0 == two_tier native: max|d|={(a-b).abs().max().item():.1e}")
        else:
            # native int = round-to-grid * s, recomputed independently
            xf = W.reshape(-1, BLOCK).float(); am = xf.abs().amax(-1, keepdim=True).clamp(min=1e-30)
            qm = (1 << (spec["n"] - 1)) - 1; s = torch.exp2(torch.ceil(torch.log2(am / qm)))
            b = (torch.round(xf / s).clamp(-qm, qm) * s).reshape(W.shape)
            print(f"     {base:>5} u0 == native int snap:  max|d|={(a-b).abs().max().item():.1e}")

    # (2) efb recon-MSE non-increasing (FP4, exp-fine config)
    print("\n  (2) efb recon-MSE monotone (FP4 u3 gm32 ge8 be2; should be non-increasing):")
    for it in range(4):
        mse = (W - em_quant(W, "FP4", 3, 32, 8, 2, efb_iters=it)).pow(2).mean().item()
        print(f"       iters={it}: MSE={mse:.6e}")

    # (3) DUAL HYPOTHESIS at (near) matched B_res: mantissa-fine vs exp-fine, per base
    print("\n  (3) dual hypothesis  (QSNR on Wt, u=3; higher wins):")
    print(f"       {'base':>5} | {'mant-fine':>10} {'exp-fine':>10}  predicted  matches?")
    print(f"       {'':>5} | {'gm8 ge32 be0':>10} {'gm32 ge8 be2':>10}")
    for base in BASES:
        q_mant = qsnr(Wt, em_quant(Wt, base, 3, 8, 32, 0))
        q_exp  = qsnr(Wt, em_quant(Wt, base, 3, 32, 8, 2))
        pred = "mant-fine" if BASES[base]["kind"] == "int" else "exp-fine"
        win = "mant-fine" if q_mant > q_exp else "exp-fine"
        print(f"       {base:>5} | {q_mant:>10.2f} {q_exp:>10.2f}  {pred:>9}  {'YES' if win == pred else 'no'}")


# ------------------------------------------------------------------------------- real-weight QSNR
def _load_layers():
    f = sorted(glob.glob("/home/yubin/.cache/huggingface/hub/models--NousResearch--Meta-Llama-3.1-8B/"
                         "snapshots/*/model-00001-of-00004.safetensors"))[0]
    out = {}
    with safe_open(f, framework="pt") as fh:
        for key in fh.keys():
            if key.endswith("layers.0.self_attn.q_proj.weight"):
                out["q_proj"] = fh.get_tensor(key)
            if key.endswith("layers.0.mlp.down_proj.weight"):
                out["down_proj"] = fh.get_tensor(key)
    return out


def main():
    layers = _load_layers()
    # (desc, gm, ge, be): mantissa-fine vs exp-fine allocations of the residual bits
    CFGS = [
        ("mant-fine gm8  ge32 be0",  8, 32, 0),
        ("mant-fine gm4  ge32 be0",  4, 32, 0),
        ("exp-fine  gm32 ge8  be2", 32,  8, 2),
        ("exp-fine  gm32 ge8  be3", 32,  8, 3),
        ("balanced  gm8  ge8  be2",  8,  8, 2),
    ]
    for lname, Wt in layers.items():
        W = Wt.to(DEV).float()
        print(f"\n== {lname}  shape={tuple(W.shape)}  (dual-hypothesis QSNR, calibration-free) ==")
        header = "  ".join(f"{b:>7}" for b in BASES)
        print(f"   {'config':>26} {'u':>2} {'B_res':>6}   {header}")
        for desc, gm, ge, be in CFGS:
            for u in (2, 3):
                br = u / gm + be / ge
                cells = [f"{qsnr(W, em_quant(W, base, u, gm, ge, be)):>7.2f}" for base in BASES]
                print(f"   {desc:>26} {u:>2} {br:>6.3f}   {'  '.join(cells)}")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest()
    else:
        main()
