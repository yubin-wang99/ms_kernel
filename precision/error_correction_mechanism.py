# error_correction_mechanism.py
#
# Quantization primitives (error-correction mechanisms) for the
# Single-Level Mantissa Sharing experiments. All functions operate on
# BF16 tensors, promote to FP32 internally for numerical stability, and
# return tensors in the original dtype.
#
# Implemented mechanisms (LSB-first naming, u == ml1_bitwidth):
#   1. quant_mxint8                       - MXINT8 baseline (OCP)
#   2. single_level_mantissa_sharing      - SSNF single-level LSB sharing
#                                           via ssnf_core.ssnf_quant
#                                           (no MSB compensation)
#   3. upper_bits_correction              - Single-level LSB sharing +
#                                           fractional-eps folded into MSBs
#   4. MSAQ_unsigned                      - Mantissa-Sharing-Aware Quant,
#                                           FP-residual avg, unsigned u-bit
#   5. MSAQ_signed                        - Mantissa-Sharing-Aware Quant,
#                                           FP-residual avg, signed u-bit
#
# Bitwidth-reduction baselines (single-rounding / double-rounding) live in
# mantissa_bitwidth_reduction.py and are imported separately by the
# experiment drivers.

import math
import torch

try:
    from ssnf_main.ForGit.src.ssnf_core import ssnf_quant
except ImportError:                      # local layout (ssnf_core.py beside this file)
    from ssnf_core import ssnf_quant


# ─────────────────────────────────────────────────────────────────────────────
# Block-size constant (must match the calling experiment script)
# ─────────────────────────────────────────────────────────────────────────────
BLOCK_SIZE = 32


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────
def _mxint8_scale_ocp(x_flat: torch.Tensor) -> torch.Tensor:
    max_abs = x_flat.abs().amax(dim=-1, keepdim=True).clamp(min=1e-30)
    shared_exp = torch.floor(torch.log2(max_abs)) - 6.0
    return torch.exp2(shared_exp)


def _to_blocks_f32(x: torch.Tensor):
    """
    Promote BF16/FP -> FP32 view of shape (-1, BLOCK_SIZE), preserving
    the original shape/dtype for later restoration.
    """
    orig_shape = x.shape
    orig_dtype = x.dtype
    xf = x.reshape(-1, BLOCK_SIZE).to(torch.float32)
    return xf, orig_shape, orig_dtype


def _back_to_orig(y_f32: torch.Tensor, orig_shape, orig_dtype) -> torch.Tensor:
    """Restore decoded result to the original shape/dtype (usually BF16)."""
    return y_f32.reshape(orig_shape).to(orig_dtype)


# ─────────────────────────────────────────────────────────────────────────────
# 1. MXINT8 baseline
# ─────────────────────────────────────────────────────────────────────────────
def quant_mxint8(x: torch.Tensor) -> torch.Tensor:
    xf, orig, dt = _to_blocks_f32(x)
    s = _mxint8_scale_ocp(xf)
    q = torch.round(xf / s).clamp(-127, 127)
    return _back_to_orig(q * s, orig, dt)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Single-Level Mantissa Sharing (Orig-Sharing, no MSB compensation)
# ─────────────────────────────────────────────────────────────────────────────
def single_level_mantissa_sharing(
    x: torch.Tensor,
    ml1_bitwidth: int,
    ml1_mg: int,
    sharing_mode: str = 'round_mean',
    rounding_mode: str = 'determ',
    encoding: str = 'twos_complement',
) -> torch.Tensor:
    """
    Single-level SSNF sharing, delegated to ssnf_core.ssnf_quant.

    Equivalent to a single-level SSNF sharing in the chosen bit-encoding domain:
      - ml1 (LSB) : ml1_bitwidth bits, shared across ml1_mg adjacent elements
                    via the user-selected sharing_mode.
      - Above ml1 : the remaining (8 - ml1_bitwidth) bits are kept per-element
                    with no sharing. ssnf_core treats levels above
                    sum(ml_bitwidth) as an implicit unshared region.

    Args:
        x              : block-shaped float tensor (..., block_size).
        ml1_bitwidth   : number of LSB bits forming ml1 (u).
        ml1_mg         : group size for ml1 sharing.
        sharing_mode   : LSB sharing aggregator
                         ('mean' / 'max' / 'min' / 'majority' /
                          'round_mean').
        rounding_mode  : 'determ' (round-to-nearest) or 'stoc'.
        encoding       : 'twos_complement' or 'sign_magnitude'.
    """
    orig_shape = x.shape
    orig_dtype = x.dtype
    xf = x.reshape(-1, BLOCK_SIZE).to(torch.float32)

    y = ssnf_quant(
        xf,
        num_format='ssnf',
        block_size=BLOCK_SIZE,
        elem_bitwidth=8,
        ml_bitwidth=[ml1_bitwidth],          # LSB only; upper (8-ml1) bits unshared
        ml_mg=[ml1_mg],
        ml_sharingmode=[sharing_mode],
        encoding=encoding,
        rounding_mode=rounding_mode,
    )
    return y.reshape(orig_shape).to(orig_dtype)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Upper-Bits Correction (single-level LSB sharing + fractional eps -> MSB)
# ─────────────────────────────────────────────────────────────────────────────
def upper_bits_correction(x: torch.Tensor, ml1_bitwidth: int, ml1_mg: int) -> torch.Tensor:
    """
    Upper-bits correction: single-level LSB sharing with MSB compensation.

    On top of the integer-domain LSB sharing of `single_level_mantissa_sharing`,
    the error introduced by sharing -- plus the fractional residual epsilon
    that is lost during the initial integer round -- is folded back into the
    upper ((8 - ml1_bitwidth)) bits so that the overall reconstruction is
    closer to the original FP value.

    Intermediate tensors are released with explicit `del` to shorten their
    peak-memory lifetimes; the algorithm is otherwise bit-identical to the
    original.

    Args:
        x            : block-shaped float tensor (..., block_size).
        ml1_bitwidth : number of LSB bits forming ml1 (u).
        ml1_mg       : group size for ml1 sharing (number of adjacent
                       elements that share the same ml1 bit-pattern).
    """
    xf, orig, dt = _to_blocks_f32(x)
    s    = _mxint8_scale_ocp(xf)

    xf_scaled = xf / s
    del xf  # original block view no longer needed

    q_init = torch.round(xf_scaled).clamp(-127, 127).to(torch.int32)
    epsilon = xf_scaled - q_init.float()
    del xf_scaled

    # Step = 1/16: int4 two's-complement codes [-8, 7] cover [-0.5, +0.4375],
    # entirely inside the theoretical round-to-nearest residual range [-0.5, 0.5).
    eps_scale = 0.5 / 8.0
    eps_quantized = torch.round(epsilon / eps_scale).clamp(-8, 7) * eps_scale
    del epsilon  # no longer needed after quantization

    # ml1: shared LSB bits via round-half-up mean
    ml1_init = q_init & ((1 << ml1_bitwidth) - 1)
    splits = torch.split(ml1_init, ml1_mg, dim=1)
    shared_splits = []
    for sp in splits:
        gs_actual = sp.shape[1]
        shared_sp = (sp.sum(dim=-1, keepdim=True) + gs_actual // 2) // gs_actual
        shared_splits.append(shared_sp.expand_as(sp))
    ml1_bits = torch.cat(shared_splits, dim=1)
    del ml1_init, splits, shared_splits  # python lists + their tensors

    # Integer + fractional residual that the (sharing + rounding) discarded
    residual_integer = q_init - ml1_bits
    del q_init  # no longer needed after residual computed
    residual_precise_integer = residual_integer.float() + eps_quantized
    del residual_integer, eps_quantized

    # Fold the precise residual into the upper bits
    q_max = (1 << (7 - ml1_bitwidth)) - 1
    corrected_upper = torch.round(residual_precise_integer / float(1 << ml1_bitwidth)) \
                            .clamp(-q_max, q_max).to(torch.int32)
    del residual_precise_integer

    q_reconstruction = (corrected_upper * (1 << ml1_bitwidth) + ml1_bits).clamp(-127, 127)
    del corrected_upper, ml1_bits
    return _back_to_orig(q_reconstruction.float() * s, orig, dt)

# ─────────────────────────────────────────────────────────────────────────────
# 4. Mantissa-Sharing-Aware Quantization (Unsigned, FP-residual mean)
# ─────────────────────────────────────────────────────────────────────────────
def MSAQ_unsigned(x: torch.Tensor,
                  ml1_bitwidth: int,
                  ml1_mg: int) -> torch.Tensor:
    """
    Mantissa-Sharing-Aware Quantization (unsigned residual storage).

    The MSB part of MXINT8 is quantized at reduced precision
    (8 - ml1_bitwidth bits), and the FP-domain residual is averaged within
    each group of ml1_mg adjacent elements and stored as an unsigned
    ml1_bitwidth-bit code (using a half-step offset trick to make the
    residual non-negative).

    Args:
        x            : block-shaped float tensor (..., block_size).
        ml1_bitwidth : number of bits used for the shared LSB residual (u).
        ml1_mg       : group size of ml1 sharing.
    """
    xf, orig, dt = _to_blocks_f32(x)
    s_base         = _mxint8_scale_ocp(xf)
    s_for_unshared = s_base * float(1 << ml1_bitwidth)

    q_max      = (1 << (7 - ml1_bitwidth)) - 1
    q_unshared = torch.round(xf / s_for_unshared).clamp(-q_max, q_max).to(torch.int32)
    x_unshared = q_unshared.float() * s_for_unshared

    residual_fp = xf - x_unshared

    splits     = torch.split(residual_fp, ml1_mg, dim=1)
    avg_splits = [sp.mean(dim=-1, keepdim=True).expand_as(sp) for sp in splits]
    residual_fp_avg = torch.cat(avg_splits, dim=1)

    # Unsigned offset trick: shift residual into [0, s_for_unshared] before encoding
    levels                    = 1 << ml1_bitwidth
    step                      = s_base
    residual_unsigned         = residual_fp_avg + s_for_unshared / 2.0
    residual_unsigned_integer = torch.round(residual_unsigned / step) \
                                      .clamp(0, levels - 1).to(torch.int32)

    residual_decoded = residual_unsigned_integer.float() * step - s_for_unshared / 2.0
    x_reconstructed  = x_unshared + residual_decoded
    return _back_to_orig(x_reconstructed, orig, dt)


# ─────────────────────────────────────────────────────────────────────────────
# 5. Mantissa-Sharing-Aware Quantization (Signed, FP-residual mean)
# ─────────────────────────────────────────────────────────────────────────────
def MSAQ_signed(x: torch.Tensor,
                ml1_bitwidth: int,
                ml1_mg: int) -> torch.Tensor:
    """
    Mantissa-Sharing-Aware Quantization (signed residual storage).

    Same as `MSAQ_unsigned`, but the per-group averaged residual is stored
    as a signed (two's-complement) ml1_bitwidth-bit code, so no half-step
    offset is needed.

    Args:
        x            : block-shaped float tensor (..., block_size).
        ml1_bitwidth : number of bits used for the shared LSB residual (u).
        ml1_mg       : group size of ml1 sharing.
    """
    xf, orig, dt = _to_blocks_f32(x)
    s_base         = _mxint8_scale_ocp(xf)
    s_for_unshared = s_base * float(1 << ml1_bitwidth)

    q_max      = (1 << (7 - ml1_bitwidth)) - 1
    q_unshared = torch.round(xf / s_for_unshared).clamp(-q_max, q_max).to(torch.int32)
    x_unshared = q_unshared.float() * s_for_unshared

    residual_fp = xf - x_unshared

    splits     = torch.split(residual_fp, ml1_mg, dim=1)
    avg_splits = [sp.mean(dim=-1, keepdim=True).expand_as(sp) for sp in splits]
    residual_fp_avg = torch.cat(avg_splits, dim=1)

    # Signed encoding: residual already centered around zero
    step = s_base
    s_min = -(1 << (ml1_bitwidth - 1))
    s_max =  (1 << (ml1_bitwidth - 1)) - 1
    residual_signed_integer = torch.round(residual_fp_avg / step) \
                                    .clamp(s_min, s_max).to(torch.int32)

    residual_decoded = residual_signed_integer.float() * step
    x_reconstructed  = x_unshared + residual_decoded
    return _back_to_orig(x_reconstructed, orig, dt)


# ─────────────────────────────────────────────────────────────────────────────
# 6. light-MS (signed) — INTEGER-residual mean (cheap online quant)
# ─────────────────────────────────────────────────────────────────────────────
def lightMS_signed(x: torch.Tensor,
                   ml1_bitwidth: int,
                   ml1_mg: int) -> torch.Tensor:
    """
    Light Mantissa-Sharing-Aware Quantization (signed, INTEGER-residual mean).

    Identical stored format to `MSAQ_signed` ((7-u)-bit unshared + u-bit signed
    shared), so DEQUANT (`upper*2^u + shared`) is unchanged. The only difference
    is the AVERAGING: where `MSAQ_signed` takes the FP-domain mean of the residual
    and then quantizes it, light-MS quantizes each element's residual to a signed
    u-bit integer FIRST and then takes the INTEGER (round-to-nearest) group mean.
    This removes the FP group-mean from the online-quant path (KV append / W+A
    activation pre-pass) — INT ops only — at near-identical accuracy:
      - vs MSAQ : within ~0.1-0.6% PPL across weight/activation/KV (wikitext-2);
      - vs naive (MXINT8 low-bit share): much better, esp. on activations (2x).
    See precision/lightms_*.py.

    Args:
        x            : block-shaped float tensor (..., block_size).
        ml1_bitwidth : number of bits for the shared LSB residual (u).
        ml1_mg       : group size of ml1 sharing.
    """
    xf, orig, dt = _to_blocks_f32(x)
    s_base         = _mxint8_scale_ocp(xf)
    s_for_unshared = s_base * float(1 << ml1_bitwidth)

    q_max      = (1 << (7 - ml1_bitwidth)) - 1
    q_unshared = torch.round(xf / s_for_unshared).clamp(-q_max, q_max).to(torch.int32)
    x_unshared = q_unshared.float() * s_for_unshared

    residual_fp = xf - x_unshared
    step  = s_base
    s_min = -(1 << (ml1_bitwidth - 1))
    s_max =  (1 << (ml1_bitwidth - 1)) - 1

    # light-MS: per-element residual -> signed u-bit INT first ...
    residual_int = torch.round(residual_fp / step).clamp(s_min, s_max).to(torch.int32)
    # ... then INTEGER (round-to-nearest) group mean -> shared u-bit code.
    splits     = torch.split(residual_int, ml1_mg, dim=1)
    avg_splits = [torch.round(sp.float().mean(dim=-1, keepdim=True))
                       .clamp(s_min, s_max).expand_as(sp) for sp in splits]
    residual_signed_integer = torch.cat(avg_splits, dim=1)

    residual_decoded = residual_signed_integer.float() * step
    x_reconstructed  = x_unshared + residual_decoded
    return _back_to_orig(x_reconstructed, orig, dt)