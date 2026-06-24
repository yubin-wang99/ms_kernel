# mantissa_bitwidth_reduction.py
#
# Definition-only module for the two mantissa-bitwidth-reduction comparison
# baselines used alongside the mantissa-sharing mechanisms. This file does
# not run any sweep; it just exposes the primitives so that experiment
# drivers (e.g. single_level_mantissa_sharing.py) can import and apply them.
#
# Both primitives reduce MXINT8 to an effective (8 - reduced_bitwidth)-bit
# representation. They differ in *where* the bitwidth reduction happens
# relative to the MXINT round step:
#
# ─── single rounding ─────────────────────────────────────────────────────────
#   single_rounding_bitwidth_reduction:
#       High-precision FP value
#         │
#         │  divide by enlarged scale (s_base * 2^reduced_bitwidth)
#         │  round once, clamp
#         ▼
#       Reduced-bitwidth integer  (one round trip, FP -> integer directly
#                                  at the reduced grid)
#
#   The FP value is quantized straight to the reduced grid; the
#   intermediate MXINT8 integer is never instantiated.
#
# ─── double rounding ─────────────────────────────────────────────────────────
#   double_rounding_bitwidth_reduction  (a.k.a. Slice-and-Scale MXINT,
#                                        Xu et al., arXiv'26):
#       High-precision FP value
#         │
#         │  divide by s_base
#         │  round once, clamp                   <-- round #1
#         ▼
#       MXINT8 integer
#         │
#         │  divide by 2^reduced_bitwidth
#         │  round once, clamp                   <-- round #2
#         ▼
#       Reduced-bitwidth integer  (two round trips, FP -> MXINT8 ->
#                                  reduced grid)
#
#   The FP value is first quantized to the full MXINT8 grid, and the
#   reduced-bitwidth result is obtained by rounding *on that integer grid*.
#
# Both produce the same per-element bit cost; they differ in the LSB
# round-off pattern and therefore in the resulting noise distribution.

import torch

from ssnf_main.ForGit.src.error_correction_mechanism import (
    BLOCK_SIZE,
    _mxint8_scale_ocp,
    _to_blocks_f32,
    _back_to_orig,
)


# ─────────────────────────────────────────────────────────────────────────────
# Single-rounding bitwidth reduction (scale-first)
# ─────────────────────────────────────────────────────────────────────────────
def single_rounding_bitwidth_reduction(x: torch.Tensor, reduced_bitwidth: int) -> torch.Tensor:
    """
    Single-rounding bitwidth reduction.

    High-precision FP -> reduced-bitwidth integer in one rounding step.
    The shared scale is enlarged by 2^reduced_bitwidth before rounding,
    so the bottom `reduced_bitwidth` LSB positions of the MXINT8 grid
    are never instantiated; MXINT8 reduces to an effective
    (8 - reduced_bitwidth)-bit format directly.

    Args:
        x                : block-shaped float tensor (..., block_size).
        reduced_bitwidth : number of LSB bits effectively removed.
    """
    xf, orig, dt = _to_blocks_f32(x)
    s_base       = _mxint8_scale_ocp(xf)
    s_reduced    = s_base * float(1 << reduced_bitwidth)
    q_max        = (1 << (7 - reduced_bitwidth)) - 1
    q            = torch.round(xf / s_reduced).clamp(-q_max, q_max).to(torch.int32)
    return _back_to_orig(q.float() * s_reduced, orig, dt)


# ─────────────────────────────────────────────────────────────────────────────
# Double-rounding bitwidth reduction (Slice-and-Scale MXINT, Xu+ '26)
# ─────────────────────────────────────────────────────────────────────────────
def double_rounding_bitwidth_reduction(x: torch.Tensor, reduced_bitwidth: int) -> torch.Tensor:
    """
    Double-rounding bitwidth reduction (a.k.a. Slice-and-Scale MXINT,
    Xu et al., arXiv'26).

    High-precision FP -> MXINT8 integer (round #1) -> reduced-bitwidth
    integer (round #2). The full MXINT8 grid is materialised first, and
    the reduced-bitwidth result is obtained by rounding the integer
    mantissa by 2^reduced_bitwidth on that grid.

    Args:
        x                : block-shaped float tensor (..., block_size).
        reduced_bitwidth : number of LSB bits sliced off (round #2 grain).
    """
    xf, orig, dt = _to_blocks_f32(x)
    s_base       = _mxint8_scale_ocp(xf)

    # Round #1: full MXINT8 quantization
    q_init  = torch.round(xf / s_base).clamp(-127, 127).to(torch.int32)

    # Round #2: slice-and-scale on the integer grid
    denom    = float(1 << reduced_bitwidth)
    q_sliced = torch.round(q_init.float() / denom).to(torch.int32)

    # Scale compensation and clamp into the reduced range
    s_reduced = s_base * denom
    q_max     = (1 << (7 - reduced_bitwidth)) - 1
    q_sliced  = q_sliced.clamp(-q_max, q_max)

    return _back_to_orig(q_sliced.float() * s_reduced, orig, dt)