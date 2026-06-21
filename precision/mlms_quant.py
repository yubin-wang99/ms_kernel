"""Multi-level mantissa sharing quant defs: naive-multi vs MSAQ-multi (multi-level-MS).

naive-multi : ssnf_core hierarchical bit-plane sharing on the unsigned MXINT8 code
              (shares the literal mantissa bit-fields, LSB-first, per ml_mg granularity).
MSAQ-multi  : our MSAQ idea lifted to the multi-level structure. Round the unshared
              upper bits to NEAREST (not truncate), form the centered signed residual
              q_r = q - q_upper*2^sum_ml, then peel it into the per-level shares by
              SUCCESSIVE SIGNED-RESIDUAL quantization, MSB-first (most-significant residual
              bits get the finest grouping). Reconstruction is the same cheap form as naive:
                  (q_upper*2^sum_ml + sum_k share_k*2^shift_k) * scale
              so DECODE cost == naive (integer shift-adds + 1 FP mul); only encode changes.

Both share the exact same storage / bits-per-element accounting.
Run: CUDA_VISIBLE_DEVICES=0 python precision/mlms_quant.py   (weight-tensor QSNR on Llama)
"""
import glob, math, os, torch
from safetensors import safe_open
from ssnf_core import ssnf_quant, compute_bits_per_elem, _apply_ssnf_share

BLOCK = 32
P = 8                                   # partition_bits (twos-complement MXINT8)


# ---- shared scale ------------------------------------------------------------
def _e8m0_scale(xf):                    # OCP E8M0 for 8-bit signed: 2^(floor(log2 amax) - 6)
    amax = xf.abs().amax(-1, keepdim=True).clamp(min=1e-30)
    return torch.exp2(torch.floor(torch.log2(amax)) - 6.0)


# ---- naive multi-level (framework bit-plane sharing) -------------------------
def naive_ml(x, ml_bitwidth, ml_mg):
    return ssnf_quant(x, num_format='ssnf', block_size=BLOCK, elem_bitwidth=8,
                      ml_bitwidth=list(ml_bitwidth), ml_mg=list(ml_mg),
                      ml_sharingmode=['round_mean'] * len(ml_bitwidth),
                      encoding='twos_complement', rounding_mode='determ')


# ---- MSAQ multi-level (residual then hierarchical sharing) --------------------
# MSAQ idea = quantize the UNSHARED upper with its own scale (round-to-nearest, like
# single-level MSAQ), form the centered signed residual, quantize it to a clean
# sum_ml-bit signed integer (FULL range -> no coverage gap), then run the SAME proven
# multi-level bit-plane sharing on that residual. Decode is the cheap shared form:
#   (q_upper*2^sum_ml + residual_shared) * s   == naive decode cost.
def msaq_ml(x, ml_bitwidth, ml_mg, carry_compensate=False):
    xf = x.reshape(-1, BLOCK).to(torch.float32)
    s = _e8m0_scale(xf)
    sum_ml = sum(ml_bitwidth)
    s_un = s * float(1 << sum_ml)
    qmax_up = (1 << (P - 1 - sum_ml)) - 1                     # 7-sum_ml magnitude bits + sign
    q_upper = torch.round(xf / s_un).clamp(-qmax_up, qmax_up) # unshared upper (round-to-nearest)
    r = xf - q_upper * s_un                                   # FP residual, centered at 0
    smin, smax = -(1 << (sum_ml - 1)), (1 << (sum_ml - 1)) - 1
    q_res = torch.round(r / s).clamp(smin, smax)             # signed sum_ml-bit residual (full range)
    # hierarchical bit-plane sharing of the residual (partition_bits == sum_ml: all bits shared)
    q_res_shared = _apply_ssnf_share(
        q_res.to(torch.int64), elem_bitwidth=sum_ml,
        ml_bitwidth=list(ml_bitwidth), ml_mg=list(ml_mg),
        ml_sharingmode=['round_mean'] * len(ml_bitwidth),
        carry_compensate=carry_compensate, encoding='twos_complement')
    q_hat = q_upper * float(1 << sum_ml) + q_res_shared.to(torch.float32)
    return (q_hat * s).reshape(x.shape)


def bpe(ml_bitwidth, ml_mg):
    return compute_bits_per_elem(8, list(ml_bitwidth), list(ml_mg), BLOCK, 8, 'twos_complement')


# ---- weight-tensor QSNR on real Llama weights --------------------------------
SNAP = glob.glob(os.path.expanduser(
    "~/.cache/huggingface/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/*"))
LINEAR_KEYS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")

# Fair sweep: 3 shared bits (5 unshared, the robust region), split hierarchically.
# Iso-bpe with single-level u3/mgX so the comparison isolates the HIERARCHY benefit
# (different granularity per bit-significance) rather than fewer unshared bits.
SWEEP = [
    (2, [2, 1], [8, 8]),    # bpe 5.625  iso single u3/mg8
    (2, [1, 2], [8, 8]),    # bpe 5.625  (LSB shared coarsely vs finer)
    (2, [2, 1], [16, 16]),  # bpe 5.4375
    (2, [1, 2], [32, 16]),  # bpe 5.40
    (2, [2, 1], [16, 32]),  # bpe 5.40
    (2, [2, 1], [32, 32]),  # bpe 5.34   iso single u3/mg32 (KV frontier)
    (2, [1, 2], [32, 32]),  # bpe 5.34
]


def _weights():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    for f in sorted(glob.glob(os.path.join(SNAP[0], "*.safetensors"))):
        with safe_open(f, framework="pt", device=dev) as g:
            for k in g.keys():
                if k.endswith(".weight") and any(p in k for p in LINEAR_KEYS):
                    yield g.get_tensor(k)


if __name__ == "__main__":
    print(f"Llama-3.1-8B weight QSNR (dB), block={BLOCK}\n")
    print(f"{'depth':>5} {'bw':>8} {'mg':>8} {'bpe':>5} | {'naive':>8} {'MSAQ':>8} | {'gain':>6}")
    acc = {tuple(map(tuple, (bw, mg))): {'n': [0.0, 0.0], 'm': [0.0, 0.0]}
           for _, bw, mg in SWEEP}
    for W in _weights():
        for _, bw, mg in SWEEP:
            key = tuple(map(tuple, (bw, mg)))
            for tag, fn in (('n', naive_ml), ('m', msaq_ml)):
                Wq = fn(W, bw, mg)
                acc[key][tag][0] += W.float().pow(2).sum().item()
                acc[key][tag][1] += (W.float() - Wq.float()).pow(2).sum().item()
        del W
    db = lambda a: 10.0 * math.log10(a[0] / max(a[1], 1e-12))
    for d, bw, mg in SWEEP:
        key = tuple(map(tuple, (bw, mg)))
        qn, qm = db(acc[key]['n']), db(acc[key]['m'])
        print(f"{d:>5} {str(bw):>8} {str(mg):>8} {bpe(bw, mg):>5.2f} | "
              f"{qn:>8.3f} {qm:>8.3f} | {qm - qn:>+6.3f}")
