"""Bits-vs-QSNR Pareto frontier: CORRECTED MSAQ (FP efb + INT upper_bits_correction) vs plain
MXINT vs hardware-native MXFP6/8. Settles "does corrected MSAQ beat MXINT6 at slightly more bits,
and does E2M3 still dominate?" on real Llama-3.1-8B weights.

Run: CUDA_VISIBLE_DEVICES=0 python precision/mxfp6_frontier.py
"""
import glob, torch
from safetensors import safe_open
from msaq_mxfp8_ppl import msaq_mxfp8, msaq_mxint8, BLOCK
from error_correction_mechanism import upper_bits_correction   # CORRECTED INT MSAQ (MSB compensation)
from mxfp6_verify import e2m3_clean, mxint6_clean

DEV = "cuda"
def qsnr(x, xq):
    e = (x - xq).double().pow(2).mean()
    return 10.0 * torch.log10(x.double().pow(2).mean() / e.clamp(min=1e-45)).item()

def msaq_mxint8_efb(x, u, mg, efb_iters=2):
    """INT8-MSAQ with ERROR-FEEDBACK coordinate descent (the INT analog of msaq_mxfp8's efb).
    For INT8 all elements share one linear scale, so wshare = plain mean (uniform weight); the gain
    is purely re-rounding the upper code after `shared` is fixed. efb_iters=0 == naive mean."""
    xf = x.reshape(-1, BLOCK).float()
    absmax = xf.abs().amax(-1, keepdim=True).clamp(min=1e-30)
    q_max = (1 << (7 - u)) - 1
    s_base = torch.exp2(torch.floor(torch.log2(absmax / 64.0)))
    s_un = s_base * float(1 << u)
    s_min, s_max = -(1 << (u - 1)), (1 << (u - 1)) - 1
    q_un = torch.round(xf / s_un).clamp(-q_max, q_max)
    shared = None
    for it in range(max(1, efb_iters + 1)):
        res = xf - q_un * s_un
        shared = torch.round(res.reshape(res.shape[0], -1, mg).mean(-1, keepdim=True)
                             .expand(-1, -1, mg).reshape(res.shape) / s_base).clamp(s_min, s_max)
        if it == efb_iters:
            break
        q_un = torch.round((xf - shared * s_base) / s_un).clamp(-q_max, q_max)   # efb: re-round upper
    return (q_un * s_un + shared * s_base).reshape(x.shape)

def bint(u, mg): return (8 - u) + u / mg + 8.0 / 32      # INT8/E3M4-MSAQ bits (1+eb+mb=8)

def mxint_n(x, n):                                        # plain n-bit signed int, per-block E8M0
    xf = x.reshape(-1, BLOCK).float(); am = xf.abs().amax(-1, keepdim=True).clamp(min=1e-30)
    qmax = (1 << (n - 1)) - 1
    s = torch.exp2(torch.ceil(torch.log2(am / qmax)))
    return (torch.round(xf / s).clamp(-qmax, qmax) * s).reshape(x.shape)

# (name, bits, fn).  *=HW-native MX element.  efb/UBC = corrected; mean = naive
ENTRIES = []
def add(nm, b, fn): ENTRIES.append((nm, b, fn))
add("MXFP6-E2M3*",    6.25, lambda x: e2m3_clean(x)[0])
add("MXFP6-E3M2*",    6.25, lambda x: msaq_mxfp8(x, 0, 1, 3, 2))
add("MXFP8-E4M3*",    8.25, lambda x: msaq_mxfp8(x, 0, 1, 4, 3))
add("MXINT6",         6.25, mxint6_clean)
add("MXINT7",         7.25, lambda x: mxint_n(x, 7))
add("MXINT8",         8.25, lambda x: msaq_mxint8(x, 0, 1))
# corrected FP MSAQ (E3M4 + efb=2) across bit budgets
for (u, mg) in [(3, 4), (2, 8), (2, 4), (2, 2), (1, 8)]:
    add(f"E3M4-MSAQ.efb u{u}/{mg}", bint(u, mg), lambda x, u=u, mg=mg: msaq_mxfp8(x, u, mg, 3, 4, efb_iters=2))
# corrected INT MSAQ (upper_bits_correction = single MSB-compensation pass)
for (u, mg) in [(3, 4), (2, 8), (2, 4), (2, 2)]:
    add(f"INT8-MSAQ.UBC u{u}/{mg}", bint(u, mg), lambda x, u=u, mg=mg: upper_bits_correction(x, u, mg))
# INT MSAQ with efb (iterated coordinate descent — the INT analog of the FP efb)
for (u, mg) in [(3, 4), (2, 8), (2, 4), (2, 2)]:
    add(f"INT8-MSAQ.efb u{u}/{mg}", bint(u, mg), lambda x, u=u, mg=mg: msaq_mxint8_efb(x, u, mg, efb_iters=2))
# naive INT MSAQ (mean) for contrast
for (u, mg) in [(3, 4), (2, 8), (2, 4)]:
    add(f"INT8-MSAQ.mean u{u}/{mg}", bint(u, mg), lambda x, u=u, mg=mg: msaq_mxint8(x, u, mg))

def report(tag, W):
    W = W.to(DEV).float()
    rows = [(nm, b, qsnr(W, fn(W).float())) for nm, b, fn in ENTRIES]
    # Pareto frontier: an entry is dominated if another has >= QSNR at <= bits
    front = set()
    for i, (_, bi, qi) in enumerate(rows):
        if not any((bj <= bi + 1e-9 and qj >= qi + 1e-9) for j, (_, bj, qj) in enumerate(rows) if j != i):
            front.add(i)
    print(f"\n== {tag}  (sorted by bits; ★=Pareto frontier) ==")
    print(f"   {'format':>20} {'bits':>5} {'QSNR':>7}  frontier")
    for i, (nm, b, q) in enumerate(sorted(rows, key=lambda r: (r[1], -r[2]))):
        oi = rows.index((nm, b, q))
        print(f"   {nm:>20} {b:>5.2f} {q:>7.2f}  {'★' if oi in front else ''}")

def main():
    torch.manual_seed(0)
    report("synthetic Gaussian W*0.02", torch.randn(4096, 4096, device=DEV) * 0.02)
    f = sorted(glob.glob("/home/yubin/.cache/huggingface/hub/models--NousResearch--Meta-Llama-3.1-8B/"
                         "snapshots/*/model-00001-of-00004.safetensors"))[0]
    with safe_open(f, framework="pt") as fh:
        for key in fh.keys():
            if key.endswith("layers.0.mlp.down_proj.weight") or key.endswith("layers.0.self_attn.q_proj.weight"):
                report(f"real Llama {key.split('model.')[-1]}", fh.get_tensor(key))

if __name__ == "__main__":
    main()
