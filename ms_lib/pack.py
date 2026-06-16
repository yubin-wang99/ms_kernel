# ms_lib/pack.py
#
# MSAQ-signed numerics + packing layer (offline / preprocessing).
#
# Migrated VERBATIM from the certified §1 NUMERICS and §2 PACKING sections of
# mantissa_sharing_kernel.py. These functions are the bit-exact ground truth:
# decompose/reconstruct reproduce error_correction_mechanism.MSAQ_signed, and
# the packing is the out-innermost SoA dense LSB-first layout the CUDA kernels
# consume. DO NOT "improve" these — the kernels (csrc/) and the oracle
# (reference.py) are both certified against this exact math.
#
#   decompose / reconstruct      : MSAQ-s quantization numerics (single FP-round)
#   pack_weight / unpack_weight  : SoA plane packing (upper, shared, scale_exp)
#   dequant_weight / weight_int8 : W-only (-> float) and W+A (-> int8) restore
#   pack_kv                      : per-head packing for the attention kernel
#   per_block_bits / per_elem_bits / effective_bits_per_elem : bit accounting

import math
import numpy as np

# --- OCP MX constants (shared across the whole project) ----------------------
BLOCK = 32        # OCP MX block size
E_MAX = 6         # MXINT8 (elem_bitwidth=8): E_max = n - 2 = 6


# =============================================================================
#  NUMERICS  (matches MSAQ_signed)
# =============================================================================
def _e8m0_scale(max_abs):
    """E8M0 shared scale for an MXINT8 block: 2^(floor(log2(max_abs)) - 6)."""
    max_abs = np.maximum(np.asarray(max_abs, dtype=np.float64), 1e-30)
    exp = np.floor(np.log2(max_abs)) - E_MAX
    exp = np.clip(exp, -127, 127)
    return np.exp2(exp)


def decompose(x_blocked, u, gs):
    """float blocks (B,32) -> (scale (B,1), q_upper (B,32) int, r_shared (B,n_group) int).

    Single FP-domain rounding to avoid double-rounding (Figueroa):
      (1) coarse grid at scale*2^u -> upper code  q_upper  ((8-u)-bit signed)
      (2) residual = x - dequant(coarse)
      (3) group-mean of residual over gs
      (4) quantize that mean at step `scale` -> shared code r_shared (u-bit signed)
    Reconstruction: (q_upper*2^u + r_shared) * scale  (a valid MXINT8 word).
    """
    xf = np.asarray(x_blocked, dtype=np.float64)
    B = xf.shape[0]
    n_group = BLOCK // gs
    max_abs = np.abs(xf).max(axis=1, keepdims=True)
    s_base = _e8m0_scale(max_abs)                                   # (B,1)
    s_unshared = s_base * (1 << u)
    q_max = (1 << (7 - u)) - 1
    q_upper = np.clip(np.round(xf / s_unshared), -q_max, q_max).astype(np.int64)
    x_unshared = q_upper * s_unshared
    residual = xf - x_unshared
    res_avg = residual.reshape(B, n_group, gs).mean(axis=2)         # (B,n_group)
    s_min, s_max = -(1 << (u - 1)), (1 << (u - 1)) - 1
    r_shared = np.clip(np.round(res_avg / s_base), s_min, s_max).astype(np.int64)
    return s_base, q_upper, r_shared


def reconstruct(scale, q_upper, r_shared, u, gs):
    """Inverse of decompose -> float blocks (B,32)."""
    r_exp = np.repeat(r_shared, gs, axis=1)                         # (B,32)
    return (q_upper * (1 << u) + r_exp).astype(np.float64) * np.asarray(scale)


def per_block_bits(u, gs):
    """Bits to store one 32-block: E8M0 scale + per-element upper + per-group shared."""
    n_group = BLOCK // gs
    return 8 + (8 - u) * BLOCK + u * n_group


def per_elem_bits(u, gs):
    return per_block_bits(u, gs) / BLOCK


# =============================================================================
#  PACKING  (out-innermost SoA, dense LSB-first)
# =============================================================================
def _pack_codes_lsb(codes, width):
    """codes int[..., n] (two's-complement width-bit) -> uint8[..., ceil(n*width/8)]."""
    n = codes.shape[-1]
    c = (codes & ((width and (1 << width) - 1) or 0)).astype(np.uint64)
    bit = ((c[..., :, None] >> np.arange(width, dtype=np.uint64)) & 1).astype(np.uint8)
    bit = bit.reshape(*codes.shape[:-1], n * width)
    return np.packbits(bit, axis=-1, bitorder="little")


def _unpack_codes_lsb(buf, n, width):
    """Inverse of _pack_codes_lsb -> signed int[..., n]."""
    bits = np.unpackbits(buf, axis=-1, bitorder="little")
    bits = bits[..., : n * width].reshape(*buf.shape[:-1], n, width)
    shifts = np.arange(width, dtype=np.int64)
    vals = (bits.astype(np.int64) << shifts).sum(axis=-1)
    half = 1 << (width - 1)
    return np.where(vals >= half, vals - (1 << width), vals)


def pack_weight(W, u, gs):
    """W float[OUT, K] (K%32==0) -> dict of out-innermost packed planes + meta.

    Reused for KV: pack one head's K or V as [Lk, D] (OUT=Lk, K=D) so the 32-blocks
    run along head_dim and the token axis is innermost (coalesced)."""
    OUT, K = W.shape
    assert K % BLOCK == 0, "K (in_features / head_dim) must be a multiple of 32"
    nb = K // BLOCK
    wbits = 8 - u
    n_group = BLOCK // gs

    blocks = W.reshape(OUT, nb, BLOCK).reshape(OUT * nb, BLOCK)
    scale, q_upper, r_shared = decompose(blocks, u, gs)
    exp = np.rint(np.log2(scale.reshape(-1))).astype(np.int64).reshape(OUT, nb)
    qu = q_upper.reshape(OUT, nb, BLOCK)
    rs = r_shared.reshape(OUT, nb, n_group)

    UB = (BLOCK * wbits) // 8
    SB = math.ceil(n_group * u / 8)
    up = _pack_codes_lsb(qu, wbits)                # [OUT, nb, UB]
    sh = _pack_codes_lsb(rs, u)                    # [OUT, nb, SB]

    return dict(
        scale_exp=np.ascontiguousarray(exp.T.astype(np.int8)),     # [nb, OUT]
        upper=np.ascontiguousarray(up.transpose(1, 2, 0)),         # [nb, UB, OUT]
        shared=np.ascontiguousarray(sh.transpose(1, 2, 0)),        # [nb, SB, OUT]
        OUT=OUT, K=K, nb=nb, u=u, gs=gs, wbits=wbits,
        UB=UB, SB=SB, n_group=n_group,
    )


def unpack_weight(p):
    """Packed planes -> (scale [OUT,nb], q_upper [OUT,nb,32], r_shared [OUT,nb,n_group])."""
    OUT, nb, u, gs, wbits, n_group = (p["OUT"], p["nb"], p["u"], p["gs"],
                                      p["wbits"], p["n_group"])
    exp = p["scale_exp"].T.astype(np.int64)                        # [OUT, nb]
    up = np.ascontiguousarray(p["upper"].transpose(2, 0, 1))       # [OUT, nb, UB]
    sh = np.ascontiguousarray(p["shared"].transpose(2, 0, 1))      # [OUT, nb, SB]
    q_upper = _unpack_codes_lsb(up, BLOCK, wbits)                  # [OUT, nb, 32]
    r_shared = _unpack_codes_lsb(sh, n_group, u)                  # [OUT, nb, n_group]
    scale = (2.0 ** exp)                                           # [OUT, nb]
    return scale, q_upper, r_shared


def dequant_weight(p):
    """Packed -> dense float weight [OUT, K]  (W-only reconstruction)."""
    scale, q_upper, r_shared = unpack_weight(p)
    gs, u = p["gs"], p["u"]
    r_exp = np.repeat(r_shared, gs, axis=2)                        # [OUT,nb,32]
    qfull = q_upper * (1 << u) + r_exp                            # [OUT,nb,32]
    return (qfull.astype(np.float64) * scale[:, :, None]).reshape(p["OUT"], p["K"])


def weight_int8(p):
    """Packed -> (qW int [OUT,nb,32], scale_w [OUT,nb])  (W+A reconstruction).

    qW = upper*2^u + shared_expanded  is the reusable MXINT8 integer word."""
    scale, q_upper, r_shared = unpack_weight(p)
    gs, u = p["gs"], p["u"]
    r_exp = np.repeat(r_shared, gs, axis=2)
    qfull = q_upper * (1 << u) + r_exp                            # [OUT,nb,32], int8 range
    return qfull, scale


def pack_kv(KV, u, gs):
    """KV float[H, L, D] (D%32==0) -> stacked head-major planes for the attn kernel.
    Per head this is pack_weight on [L, D]: blocks along head_dim, token innermost.

    TOKEN-MAJOR byte order (Stage 4a): the byte planes are laid out with the BYTES
    axis INNERMOST ([H,nb,L,UB] not [H,nb,UB,L]) so that, for a fixed key/token, a
    warp's 32 head_dim threads read a CONTIGUOUS byte span -> coalesced loads. The
    bit-packing inside each block is unchanged (same codes), only the plane axis
    order differs; the kernel addresses it as base + key*UB + byteIdx.

    (Stage 4b register-aligned packing was tried and REVERTED: word-alignment
    padding raised bytes ~26% and `bfe` did not help — the KV decode is bound by
    load-latency/MLP, not extraction-instruction count. See change.md Phase 5.)"""
    H, L, D = KV.shape
    per = [pack_weight(KV[h], u, gs) for h in range(H)]
    # per-head upper/shared are [nb,UB,L]/[nb,SB,L]; move BYTES innermost -> [nb,L,UB]
    up = np.stack([q["upper"] for q in per]).transpose(0, 1, 3, 2)             # [H,nb,L,UB]
    sh = np.stack([q["shared"] for q in per]).transpose(0, 1, 3, 2)            # [H,nb,L,SB]
    return dict(
        scale_exp=np.ascontiguousarray(np.stack([q["scale_exp"] for q in per])),  # [H,nb,L]
        upper=np.ascontiguousarray(up),                                        # [H,nb,L,UB]
        shared=np.ascontiguousarray(sh),                                       # [H,nb,L,SB]
        H=H, L=L, D=D, nb=per[0]["nb"], u=u, gs=gs, wbits=per[0]["wbits"],
        UB=per[0]["UB"], SB=per[0]["SB"], n_group=per[0]["n_group"],
        _per=per,
    )


def effective_bits_per_elem(p):
    total = p["scale_exp"].size + p["upper"].size + p["shared"].size
    denom = (p["OUT"] * p["K"]) if "OUT" in p else (p["H"] * p["L"] * p["D"])
    return total * 8 / denom


# =============================================================================
#  PLAIN MXINT8  (baseline format: int8 mantissa + E8M0 scale, NO sharing)
# =============================================================================
#  The matched-optimization baseline for the kernels. Same out-innermost SoA
#  plane layout as the MSAQ pack, but the mantissa is a full int8 stored
#  directly — so the MXINT8 kernels are byte-for-byte identical to the MSAQ
#  kernels EXCEPT they skip the sub-byte unpack (a direct int8 load replaces
#  ms::unpack_ms_weight_elem). This isolates the unpack overhead. Per element:
#  8 bits + 8/32 scale = 8.25 b/elem (vs e.g. MSAQ u3gs8 = 5.625).
def _quant_mxint8_blocks(blocks):
    """float blocks (B,32) -> (scale (B,1), q (B,32) int)."""
    xf = np.asarray(blocks, np.float64)
    s = _e8m0_scale(np.abs(xf).max(axis=1, keepdims=True))
    q = np.clip(np.round(xf / s), -127, 127).astype(np.int64)
    return s, q


def pack_weight_mxint8(W):
    """W float[OUT,K] (K%32==0) -> out-innermost MXINT8 planes:
       scale_exp [nb,OUT] int8,  qweight [nb,32,OUT] int8."""
    OUT, K = W.shape
    assert K % BLOCK == 0, "K must be a multiple of 32"
    nb = K // BLOCK
    blocks = W.reshape(OUT, nb, BLOCK).reshape(OUT * nb, BLOCK)
    s, q = _quant_mxint8_blocks(blocks)
    exp = np.rint(np.log2(s.reshape(-1))).astype(np.int64).reshape(OUT, nb)
    q = q.reshape(OUT, nb, BLOCK)
    return dict(
        scale_exp=np.ascontiguousarray(exp.T.astype(np.int8)),          # [nb, OUT]
        qweight=np.ascontiguousarray(q.transpose(1, 2, 0).astype(np.int8)),  # [nb, 32, OUT]
        OUT=OUT, K=K, nb=nb,
    )


def dequant_weight_mxint8(p):
    """MXINT8 packed -> dense float [OUT,K]."""
    exp = p["scale_exp"].T.astype(np.int64)                              # [OUT, nb]
    q = np.ascontiguousarray(p["qweight"].transpose(2, 0, 1)).astype(np.float64)  # [OUT,nb,32]
    return (q * (2.0 ** exp)[:, :, None]).reshape(p["OUT"], p["K"])


def weight_int8_mxint8(p):
    """MXINT8 packed -> (qW int [OUT,nb,32], scale_w [OUT,nb])  (W+A path)."""
    exp = p["scale_exp"].T.astype(np.int64)                              # [OUT, nb]
    qW = np.ascontiguousarray(p["qweight"].transpose(2, 0, 1)).astype(np.int64)   # [OUT,nb,32]
    return qW, (2.0 ** exp)


def pack_kv_mxint8(KV):
    """KV float[H,L,D] -> stacked MXINT8 head-major planes (mirror of pack_kv)."""
    H, L, D = KV.shape
    per = [pack_weight_mxint8(KV[h]) for h in range(H)]
    # token-major (Stage 4a): mantissa axis innermost -> [H,nb,L,32] for coalescing
    qw = np.stack([q["qweight"] for q in per]).transpose(0, 1, 3, 2)           # [H,nb,L,32]
    return dict(
        scale_exp=np.ascontiguousarray(np.stack([q["scale_exp"] for q in per])),  # [H,nb,L]
        qweight=np.ascontiguousarray(qw),                                      # [H,nb,L,32]
        H=H, L=L, D=D, nb=per[0]["nb"], _per=per,
    )
