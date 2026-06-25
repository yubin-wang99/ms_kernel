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

    d = dict(
        scale_exp=np.ascontiguousarray(exp.T.astype(np.int8)),     # [nb, OUT]
        upper=np.ascontiguousarray(up.transpose(1, 2, 0)),         # [nb, UB, OUT]
        shared=np.ascontiguousarray(sh.transpose(1, 2, 0)),        # [nb, SB, OUT]
        OUT=OUT, K=K, nb=nb, u=u, gs=gs, wbits=wbits,
        UB=UB, SB=SB, n_group=n_group,
    )
    # Wide-load GEMV plane: COLUMN-MAJOR [nb, OUT, UB] so a thread reads its
    # column's whole UB-byte block in ONE (u4: int4) or a few (u2/u3: 4-aligned
    # uint32) wide, coalesced loads instead of UB byte-strided reads. Just a
    # transpose of the dense plane (no re-packing); bytes/codes identical ->
    # oracle unaffected. Built for all u (u4 = int4 width; u2/u3 word-loaded).
    d["upper_cm"] = np.ascontiguousarray(d["upper"].transpose(0, 2, 1))   # [nb, OUT, UB]
    d["shared_cm"] = np.ascontiguousarray(d["shared"].transpose(0, 2, 1)) # [nb, OUT, SB]
    return d


def pack_weight_ra(W, u, gs):
    """REGISTER-ALIGNED pack: SAME codes as pack_weight (signed q_upper/r_shared),
    but the upper plane is padded so each (8-u)-bit code lies WHOLLY inside one 32-bit
    word (CPW = floor(32/(8-u)) codes/word, NW words) -> the kernel extracts each code
    with ONE bfe.s32 (HW sign-extend), no straddle / rolling-buffer / mask / sign-extend.
    shared stays dense (n_group*u <= 8 -> already 1 straddle-free byte). +~20% upper
    bytes (u3 20->24, u2 24->28) buys ~4->1 ALU ops/elem. Decode == pack_weight."""
    OUT, K = W.shape
    assert K % BLOCK == 0
    nb = K // BLOCK; wbits = 8 - u; n_group = BLOCK // gs
    CPW = 32 // wbits; NW = (BLOCK + CPW - 1) // CPW
    blocks = W.reshape(OUT, nb, BLOCK).reshape(OUT * nb, BLOCK)
    scale, q_upper, r_shared = decompose(blocks, u, gs)
    exp = np.rint(np.log2(scale.reshape(-1))).astype(np.int64).reshape(OUT, nb)
    qu = q_upper.reshape(OUT, nb, BLOCK); rs = r_shared.reshape(OUT, nb, n_group)
    # register-aligned upper: words[..,wi] |= (code & mask) << ((k%CPW)*wbits)
    mask = (1 << wbits) - 1
    words = np.zeros((OUT, nb, NW), dtype=np.uint32)
    for k in range(BLOCK):
        wi = k // CPW; bp = (k % CPW) * wbits
        words[..., wi] |= ((qu[..., k].astype(np.uint64) & mask).astype(np.uint32) << bp)
    upper_ra = words.view(np.uint8).reshape(OUT, nb, NW * 4)        # little-endian bytes
    sh = _pack_codes_lsb(rs, u)                                     # [OUT,nb,SB] dense (1 byte)
    SB = sh.shape[-1]
    return dict(
        scale_exp=np.ascontiguousarray(exp.T.astype(np.int8)),     # [nb, OUT]
        upper_ra_cm=np.ascontiguousarray(upper_ra.transpose(1, 0, 2)),  # [nb, OUT, NW*4]
        shared_cm=np.ascontiguousarray(sh.transpose(1, 0, 2)),     # [nb, OUT, SB]
        OUT=OUT, K=K, nb=nb, u=u, gs=gs, wbits=wbits, CPW=CPW, NW=NW, SB=SB, n_group=n_group,
    )


def dequant_weight_ra(p):
    """register-aligned packed -> dense float [OUT,K] (oracle)."""
    OUT, nb, u, gs, wbits, CPW, NW, n_group = (p["OUT"], p["nb"], p["u"], p["gs"],
                                               p["wbits"], p["CPW"], p["NW"], p["n_group"])
    exp = p["scale_exp"].T.astype(np.int64)                        # [OUT,nb]
    words = np.ascontiguousarray(p["upper_ra_cm"].transpose(1, 0, 2)).reshape(OUT, nb, NW, 4).view(np.uint32).reshape(OUT, nb, NW)
    mask = (1 << wbits) - 1; half = 1 << (wbits - 1)
    q_upper = np.empty((OUT, nb, BLOCK), np.int64)
    for k in range(BLOCK):
        wi = k // CPW; bp = (k % CPW) * wbits
        v = ((words[..., wi] >> bp) & mask).astype(np.int64)
        q_upper[..., k] = np.where(v >= half, v - (1 << wbits), v)   # sign-extend
    sh = np.ascontiguousarray(p["shared_cm"].transpose(1, 0, 2))    # [OUT,nb,SB]
    r_shared = _unpack_codes_lsb(sh, n_group, u)                    # signed
    r_exp = np.repeat(r_shared, gs, axis=2)
    qfull = q_upper * (1 << u) + r_exp
    return (qfull.astype(np.float64) * (2.0 ** exp)[:, :, None]).reshape(OUT, p["K"])


def decompose_naive(x_blocked, u, gs):
    """NAIVE mantissa-sharing: quantize to MXINT8 int8 q8, then split q8 into a
    per-element high part + group-shared low u bits (the low bits are AVERAGED).
      q8       = clip(round(x/s), +-127)              (plain MXINT8 int8)
      upper[k] = q8[k] >> u                           (arithmetic; signed (8-u)-bit)
      low[k]   = q8[k] & (2^u - 1)                    (UNSIGNED u-bit, [0,2^u))
      r_shared = clip(round(mean_g(low)), 0, 2^u-1)   (one UNSIGNED u-bit per group)
    Reconstruct: (upper*2^u + r_shared)*s  (== upper<<u | r_shared). UNSIGNED shared,
    so it reuses the MS-unsigned unpack (concat). Same stored plane layout as MSAQ;
    the ONLY difference vs decompose_unsigned is HOW the codes are derived (from the
    quantized int8, integer-only) — the simplest possible sharing, for bottleneck
    isolation (kernel time = MS-unsigned/MSAQ; the decompose math is free at runtime)."""
    xf = np.asarray(x_blocked, dtype=np.float64)
    B = xf.shape[0]; n_group = BLOCK // gs
    s_base = _e8m0_scale(np.abs(xf).max(axis=1, keepdims=True))
    q8 = np.clip(np.round(xf / s_base), -127, 127).astype(np.int64)
    upper = q8 >> u                                   # arithmetic shift (Python int >> is arithmetic)
    low = q8 - (upper << u)                            # == q8 & (2^u-1), unsigned [0,2^u)
    low_avg = low.reshape(B, n_group, gs).mean(axis=2)
    r_shared = np.clip(np.round(low_avg), 0, (1 << u) - 1).astype(np.int64)
    return s_base, upper, r_shared


def pack_weight_naive(W, u, gs):
    """naive-ms pack: SAME plane layout/keys as pack_weight_unsigned (upper_cm/
    shared_cm/scale_exp, shared UNSIGNED). Read by the MS-unsigned unpack kernels."""
    OUT, K = W.shape
    assert K % BLOCK == 0
    nb = K // BLOCK; wbits = 8 - u; n_group = BLOCK // gs
    blocks = W.reshape(OUT, nb, BLOCK).reshape(OUT * nb, BLOCK)
    scale, upper, r_shared = decompose_naive(blocks, u, gs)
    exp = np.rint(np.log2(scale.reshape(-1))).astype(np.int64).reshape(OUT, nb)
    qu = upper.reshape(OUT, nb, BLOCK); rs = r_shared.reshape(OUT, nb, n_group)
    UB = (BLOCK * wbits) // 8; SB = math.ceil(n_group * u / 8)
    up = _pack_codes_lsb(qu, wbits); sh = _pack_codes_lsb(rs, u)
    d = dict(
        scale_exp=np.ascontiguousarray(exp.T.astype(np.int8)),
        upper=np.ascontiguousarray(up.transpose(1, 2, 0)),
        shared=np.ascontiguousarray(sh.transpose(1, 2, 0)),
        OUT=OUT, K=K, nb=nb, u=u, gs=gs, wbits=wbits, UB=UB, SB=SB, n_group=n_group,
    )
    d["upper_cm"] = np.ascontiguousarray(d["upper"].transpose(0, 2, 1))
    d["shared_cm"] = np.ascontiguousarray(d["shared"].transpose(0, 2, 1))
    return d


def dequant_weight_naive(p):
    """naive-ms packed -> dense float [OUT,K] (oracle). Identical decode to UNSIGNED."""
    return dequant_weight_unsigned(p)


def decompose_unsigned(x_blocked, u, gs):
    """MS-UNSIGNED variant: FLOOR the upper code (residual stays >=0) so the shared
    code is UNSIGNED and reconstruction is a pure bit-concat (q_upper<<u | r_shared),
    no signed add / no shared sign-extend. Same E8M0 scale & sharing as decompose.
      q_upper = clip(floor(x/s_unshared), -2^(7-u), 2^(7-u)-1)   (8-u)-bit signed
      r_shared = clip(round(mean_g(x - q_upper*s_unshared)/s_base), 0, 2^u-1)  u-bit UNSIGNED
    Reconstruct: (q_upper*2^u + r_shared)*s_base  (valid int8; low u bits = r_shared)."""
    xf = np.asarray(x_blocked, dtype=np.float64)
    B = xf.shape[0]; n_group = BLOCK // gs
    s_base = _e8m0_scale(np.abs(xf).max(axis=1, keepdims=True))
    s_unshared = s_base * (1 << u)
    q_lo, q_hi = -(1 << (7 - u)), (1 << (7 - u)) - 1
    q_upper = np.clip(np.floor(xf / s_unshared), q_lo, q_hi).astype(np.int64)   # FLOOR -> residual>=0
    residual = xf - q_upper * s_unshared                                        # in [0, s_unshared)
    res_avg = residual.reshape(B, n_group, gs).mean(axis=2)
    r_shared = np.clip(np.round(res_avg / s_base), 0, (1 << u) - 1).astype(np.int64)  # UNSIGNED u-bit
    return s_base, q_upper, r_shared


def pack_weight_unsigned(W, u, gs):
    """MS-unsigned pack: SAME plane layout/keys as pack_weight (upper_cm/shared_cm/
    scale_exp) but shared codes are UNSIGNED (decompose_unsigned). The unsigned-unpack
    kernel reads these with OR-concat (no shared sign-extend). dequant_weight_unsigned
    is the oracle."""
    OUT, K = W.shape
    assert K % BLOCK == 0
    nb = K // BLOCK; wbits = 8 - u; n_group = BLOCK // gs
    blocks = W.reshape(OUT, nb, BLOCK).reshape(OUT * nb, BLOCK)
    scale, q_upper, r_shared = decompose_unsigned(blocks, u, gs)
    exp = np.rint(np.log2(scale.reshape(-1))).astype(np.int64).reshape(OUT, nb)
    qu = q_upper.reshape(OUT, nb, BLOCK); rs = r_shared.reshape(OUT, nb, n_group)
    UB = (BLOCK * wbits) // 8; SB = math.ceil(n_group * u / 8)
    up = _pack_codes_lsb(qu, wbits); sh = _pack_codes_lsb(rs, u)   # rs unsigned -> mask keeps bits
    d = dict(
        scale_exp=np.ascontiguousarray(exp.T.astype(np.int8)),
        upper=np.ascontiguousarray(up.transpose(1, 2, 0)),
        shared=np.ascontiguousarray(sh.transpose(1, 2, 0)),
        OUT=OUT, K=K, nb=nb, u=u, gs=gs, wbits=wbits, UB=UB, SB=SB, n_group=n_group,
    )
    d["upper_cm"] = np.ascontiguousarray(d["upper"].transpose(0, 2, 1))
    d["shared_cm"] = np.ascontiguousarray(d["shared"].transpose(0, 2, 1))
    return d


def dequant_weight_unsigned(p):
    """MS-unsigned packed -> dense float [OUT,K] (oracle). shared is UNSIGNED."""
    OUT, nb, u, gs, wbits, n_group = (p["OUT"], p["nb"], p["u"], p["gs"], p["wbits"], p["n_group"])
    exp = p["scale_exp"].T.astype(np.int64)
    up = np.ascontiguousarray(p["upper"].transpose(2, 0, 1))
    sh = np.ascontiguousarray(p["shared"].transpose(2, 0, 1))
    q_upper = _unpack_codes_lsb(up, BLOCK, wbits)                  # signed (8-u)-bit
    # shared UNSIGNED: unpack bits without sign extension
    bits = np.unpackbits(sh, axis=-1, bitorder="little")[..., : n_group * u].reshape(*sh.shape[:-1], n_group, u)
    r_shared = (bits.astype(np.int64) << np.arange(u, dtype=np.int64)).sum(axis=-1)   # [OUT,nb,ng], 0..2^u-1
    r_exp = np.repeat(r_shared, gs, axis=2)
    qfull = q_upper * (1 << u) + r_exp
    return (qfull.astype(np.float64) * (2.0 ** exp)[:, :, None]).reshape(OUT, p["K"])


def pack_weight_relayout(W, u, gs):
    """NIBBLE-ALIGNED re-layout of pack_weight (SAME weights, bit-exact, byte-neutral).

    Splits the per-element (8-u)-bit upper code q_upper into a nibble-aligned high
    part and a small dense low part so the bulk (high nibble) can be SIMD-dequantized:
      upper4 = q_upper >> (4-u)        signed 4-bit (nibble; bits 4-7 of the int8 weight)
      low_un = q_upper & (2^(4-u)-1)   unsigned (4-u)-bit (bits u..3)
      shared = r_shared                signed u-bit per group (bits 0..u-1, unchanged)
    Reconstruct: w = upper4*16 + low_un*2^u + shared  (== pack_weight's q_upper*2^u+shared).
    Planes are COLUMN-MAJOR [nb, OUT, .] like *_cm so a thread wide-loads its column.
    """
    OUT, K = W.shape
    assert K % BLOCK == 0
    nb = K // BLOCK; n_group = BLOCK // gs; lu_bits = 4 - u
    blocks = W.reshape(OUT, nb, BLOCK).reshape(OUT * nb, BLOCK)
    scale, q_upper, r_shared = decompose(blocks, u, gs)
    exp = np.rint(np.log2(scale.reshape(-1))).astype(np.int64).reshape(OUT, nb)
    qu = q_upper.reshape(OUT, nb, BLOCK)
    upper4 = (qu >> lu_bits).astype(np.int64)                      # signed 4-bit
    low_un = (qu & ((1 << lu_bits) - 1)).astype(np.int64)          # unsigned (4-u)-bit
    rs = r_shared.reshape(OUT, nb, n_group)
    # high-nibble plane: store BIASED (upper4+8 in [0,15]) so the SIMD dequant gets
    # the signed value via a uniform (nibble-8) with no per-nibble two's-complement
    # branch. 32 nibbles -> 16 bytes (k even=low nibble, odd=high).
    hu = ((upper4 + 8) & 0xF).astype(np.uint8)                     # [OUT,nb,32]
    hi4 = (hu[..., 0::2] | (hu[..., 1::2] << 4)).astype(np.uint8)  # [OUT,nb,16]
    lu = _pack_codes_lsb(low_un, lu_bits)                          # [OUT,nb, ceil(32*lu_bits/8)]
    sh = _pack_codes_lsb(rs, u)                                    # [OUT,nb,SB]
    return dict(
        scale_exp=np.ascontiguousarray(exp.T.astype(np.int8)),     # [nb, OUT]
        hi4_cm=np.ascontiguousarray(hi4.transpose(1, 0, 2)),       # [nb, OUT, 16]
        lowun_cm=np.ascontiguousarray(lu.transpose(1, 0, 2)),      # [nb, OUT, lu_B]
        shared_cm=np.ascontiguousarray(sh.transpose(1, 0, 2)),     # [nb, OUT, SB]
        OUT=OUT, K=K, nb=nb, u=u, gs=gs, lu_bits=lu_bits, n_group=n_group,
        HB=16, LUB=lu.shape[-1], SB=sh.shape[-1],
    )


def dequant_weight_relayout(p):
    """Re-layout planes -> dense float [OUT,K]; oracle for the CUDA nibble kernel."""
    OUT, nb, u, gs, lu_bits, n_group = (p["OUT"], p["nb"], p["u"], p["gs"],
                                        p["lu_bits"], p["n_group"])
    exp = p["scale_exp"].T.astype(np.int64)                        # [OUT,nb]
    hi4 = np.ascontiguousarray(p["hi4_cm"].transpose(1, 0, 2))     # [OUT,nb,16]
    lo = np.ascontiguousarray(p["lowun_cm"].transpose(1, 0, 2))    # [OUT,nb,LUB]
    sh = np.ascontiguousarray(p["shared_cm"].transpose(1, 0, 2))   # [OUT,nb,SB]
    # unpack high nibbles (signed 4-bit)
    lown = (hi4 & 0xF).astype(np.int64); highn = (hi4 >> 4).astype(np.int64)
    upper4 = np.empty((OUT, nb, BLOCK), np.int64)
    upper4[..., 0::2] = lown; upper4[..., 1::2] = highn
    upper4 = upper4 - 8                                            # un-bias (stored as upper4+8)
    low_un = _unpack_codes_lsb(lo, BLOCK, lu_bits) if lu_bits else np.zeros((OUT, nb, BLOCK), np.int64)
    low_un &= (1 << lu_bits) - 1                                   # unsigned
    shared = _unpack_codes_lsb(sh, n_group, u)                    # [OUT,nb,n_group]
    sh_exp = np.repeat(shared, gs, axis=2)
    w_int = upper4 * 16 + low_un * (1 << u) + sh_exp
    return (w_int.astype(np.float64) * (2.0 ** exp)[:, :, None]).reshape(OUT, p["K"])


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


def pack_kv_relayout(KV, u, gs):
    """KV float[H,L,D] -> nibble-relayout token-major planes (BYTES innermost) for the
    u2/u3 relayout KV-decode kernel. Per head = pack_weight_relayout on [L,D]; the _cm
    planes are already [nb, L, BYTES] (token-major) so just stack over H."""
    H, L, D = KV.shape
    per = [pack_weight_relayout(KV[h], u, gs) for h in range(H)]
    return dict(
        scale_exp=np.ascontiguousarray(np.stack([q["scale_exp"] for q in per])),  # [H,nb,L]
        hi4=np.ascontiguousarray(np.stack([q["hi4_cm"] for q in per])),            # [H,nb,L,16]
        lowun=np.ascontiguousarray(np.stack([q["lowun_cm"] for q in per])),        # [H,nb,L,LUB]
        shared=np.ascontiguousarray(np.stack([q["shared_cm"] for q in per])),      # [H,nb,L,SB]
        H=H, L=L, D=D, nb=per[0]["nb"], u=u, gs=gs, lu_bits=per[0]["lu_bits"],
        HB=16, LUB=per[0]["LUB"], SB=per[0]["SB"], n_group=per[0]["n_group"],
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
    qw = np.ascontiguousarray(q.transpose(1, 2, 0).astype(np.int8))      # [nb, 32, OUT]
    return dict(
        scale_exp=np.ascontiguousarray(exp.T.astype(np.int8)),          # [nb, OUT]
        qweight=qw,                                                     # [nb, 32, OUT] row-major
        # column-major twin [nb, OUT, 32]: a column's 32 int8 bytes are CONTIGUOUS, so the
        # decode/GEMM kernels wide-load them (1 int4x2) instead of 32 strided 1-byte reads.
        qweight_cm=np.ascontiguousarray(q.transpose(1, 0, 2).astype(np.int8)),  # [nb, OUT, 32]
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
