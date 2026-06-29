"""Middle-bit mantissa sharing vs LSB sharing — weight-tensor QSNR.

Current MSAQ / naive-MS always share the LOWEST `sb` bits of the MXINT8 code
(an LSB window).  This script generalises the shared window to start at an
arbitrary bit offset `lo`:

    8-bit code  =  [ top (8-sb-lo) | shared sb | low lo ]
                     per-element     SHARED       per-element

  lo = 0  ->  share the bottom sb bits          == current LSB scheme
  lo > 0  ->  keep the bottom lo bits per-elem,  share a MIDDLE sb-bit window

The per-block bit budget is identical for every `lo` at fixed (sb, mg):
`(8-sb)` unshared bits/elem + `sb*ceil(BLOCK/mg)` shared bits.  Only WHICH
sb-bit window is shared changes, so lo=0 vs lo>0 is an iso-bit comparison and
isolates the effect of sharing a higher-significance window.

Two recombination families, each generalised from lightms_qsnr.py:
  naive_win  : unsigned bit-field replace (OR-style).  lo=0 == naive_ms.
  msaq_win   : signed residual-mean (ADD-style).       lo=0 == light/MSAQ.

QSNR(dB) = 10*log10( sum||W||^2 / sum||W-dequant(W)||^2 ) over all Linear weights.
Run: CUDA_VISIBLE_DEVICES=1 python precision/midbit_qsnr.py
"""
import glob, os, torch
from safetensors import safe_open

BLOCK = 32
# NousResearch mirror = identical Llama-3.1-8B base weights (ungated).
SNAP = glob.glob(os.path.expanduser(
    "~/.cache/huggingface/hub/models--NousResearch--Meta-Llama-3.1-8B/snapshots/*"))[0]
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def _blocks(x):
    return x.reshape(-1, BLOCK).to(torch.float32)

def _mxint8_scale(xf):                                   # OCP E8M0 for 8-bit signed
    max_abs = xf.abs().amax(-1, keepdim=True).clamp(min=1e-30)
    return torch.exp2(torch.floor(torch.log2(max_abs)) - 6.0)


# ---- unsigned family: share an sb-bit field at offset `lo` (OR-style) --------
def naive_win(x, sb, mg, lo):                            # lo=0 -> naive_ms (lightms_qsnr.py)
    xf = _blocks(x)
    s = _mxint8_scale(xf)
    q = torch.round(xf / s).clamp(-127, 127).to(torch.int32)
    patt = q & 0xFF                                      # two's-complement 8-bit pattern
    fmask = ((1 << sb) - 1) << lo                        # the sb-bit window
    mid = (patt & fmask) >> lo                           # unsigned field in [0, 2^sb)
    grp = mid.reshape(mid.shape[0], -1, mg)
    shared = (((grp.sum(-1, keepdim=True) + mg // 2) // mg)
              .clamp(0, (1 << sb) - 1).expand(-1, -1, mg).reshape(mid.shape))
    patt2 = (patt & ~fmask) | (shared << lo)             # replace only the window
    q2 = torch.where(patt2 >= 128, patt2 - 256, patt2)
    return (q2.float() * s).reshape(x.shape)


# ---- signed family: round-to-nearest top, INT-mean shared mid (ADD-style) ----
def msaq_win(x, sb, mg, lo):                             # lo=0 -> light-MS / MSAQ (INT-mean)
    xf = _blocks(x)
    s = _mxint8_scale(xf)
    q8 = torch.round(xf / s)                             # full 8-bit code (continuous->int)
    hi_step = float(1 << (lo + sb))                      # step of the per-elem UPPER grid
    lo_step = float(1 << lo)                             # step the shared window resolves
    q_top_max = (1 << (7 - lo - sb)) - 1
    q_hi = torch.round(q8 / hi_step).clamp(-q_top_max, q_top_max)
    R = q8 - q_hi * hi_step                              # residual (code units), ~[-hi/2, hi/2]
    mid = torch.round(R / lo_step)                       # per-elem mid code
    low = R - mid * lo_step                              # per-elem low remainder (kept exactly)
    s_min, s_max = -(1 << (sb - 1)), (1 << (sb - 1)) - 1
    mid = mid.clamp(s_min, s_max)
    grp = mid.reshape(mid.shape[0], -1, mg)
    shared = (torch.round(grp.mean(-1, keepdim=True))
              .clamp(s_min, s_max).expand(-1, -1, mg).reshape(mid.shape))
    q_hat = q_hi * hi_step + shared * lo_step + low
    return (q_hat * s).reshape(x.shape)


def qsnr_acc(W, fn, sb, mg, lo):
    Wq = fn(W, sb, mg, lo)
    sig = W.float().pow(2).sum().item()
    noi = (W.float() - Wq.float()).pow(2).sum().item()
    return sig, noi


LINEAR_KEYS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
def linear_weights():
    for f in sorted(glob.glob(os.path.join(SNAP, "*.safetensors"))):
        with safe_open(f, framework="pt", device=DEV) as g:
            for k in g.keys():
                if k.endswith(".weight") and any(p in k for p in LINEAR_KEYS):
                    yield k, g.get_tensor(k)


if __name__ == "__main__":
    # (sb, mg) base configs; for each, sweep lo over all windows that fit
    # (lo + sb <= 7 keeps >=1 signed top bit). lo=0 is the LSB baseline.
    base = [(sb, mg) for sb in (2, 3, 4) for mg in (2, 4, 8)]
    families = (("naive", naive_win), ("msaq", msaq_win))
    LO_MAX = 4
    # collect the (sb,mg,lo) cells per family
    cells = []
    for sb, mg in base:
        for lo in range(0, LO_MAX + 1):
            if lo + sb <= 7:
                cells.append((sb, mg, lo))

    acc = {(fam, c): [0.0, 0.0] for fam, _ in families for c in cells}
    n = 0
    for name, W in linear_weights():
        n += 1
        for fam, fn in families:
            for (sb, mg, lo) in cells:
                sg, no = qsnr_acc(W, fn, sb, mg, lo)
                acc[(fam, (sb, mg, lo))][0] += sg
                acc[(fam, (sb, mg, lo))][1] += no
        del W

    db = lambda a: 10.0 * torch.log10(torch.tensor(a[0] / max(a[1], 1e-12))).item()
    print(f"Llama-3.1-8B weight QSNR (dB). block={BLOCK}, {DEV}, {n} linear tensors")
    print(f"Window: [top(8-sb-lo) | shared sb | low lo].  lo=0 == LSB sharing (baseline)\n")
    for fam, _ in families:
        print(f"===== family: {fam}  ({'OR-style unsigned' if fam=='naive' else 'signed INT-mean (MSAQ)'}) =====")
        print(f"{'sb':>2} {'mg':>3} | " + " ".join(f"lo={lo:<6}" for lo in range(LO_MAX + 1))
              + " |  best-lo  delta_vs_LSB")
        for sb, mg in base:
            row = {}
            for lo in range(0, LO_MAX + 1):
                if (sb, mg, lo) in [(s_, m_, l_) for (s_, m_, l_) in cells if (s_, m_) == (sb, mg)]:
                    row[lo] = db(acc[(fam, (sb, mg, lo))])
            base_lsb = row[0]
            cellstr = " ".join(f"{row[lo]:>8.3f}" if lo in row else f"{'--':>8}"
                               for lo in range(LO_MAX + 1))
            best_lo = max(row, key=row.get)
            print(f"{sb:>2} {mg:>3} | {cellstr} |  lo={best_lo} "
                  f"{row[best_lo]-base_lsb:>+8.3f} dB")
        print()
