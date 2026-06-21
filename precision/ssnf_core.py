# === ssnf_core.py ===
#
# Scaled squared numeric format (SSNF) quantization core.
#
# ─── Naming conventions ───────────────────────────────────────────────────────
#   elem_bitwidth     : per-element signed-integer bit-width (e.g. 8 for MXINT8).
#                       Does NOT change when sharing is applied.
#   bits_per_elem     : effective per-element bit cost AFTER sharing, including
#                       the per-block scale overhead. See `compute_bits_per_elem`.
#                       For MXINT8 baseline (block_size=32, scale 8-bit,
#                       twos_complement): 8 (mantissa) + 8/32 (scale) = 8.25.
#                       For ssnf with ml_depth=1, ml1_bitwidth=2, ml1_mg=2,
#                       twos_complement:
#                         6 (unshared) + 2/2 (shared) + 8/32 (scale) = 7.25.
#                       Note: with encoding='sign_magnitude', partition_bits
#                       drops to (elem_bitwidth - 1) and an extra 1 bit per
#                       element is added for the per-element sign.
#   s                 : block-wise OCP shared scale factor (8-bit exponent, E8M0).
#
#   ml_depth          : mantissa sharing level depth (number of LSB levels that
#                       participate in sharing). Above ml_depth, bits are kept
#                       per-element with no sharing.
#                       Example: ml_depth=1 means only one level of sharing
#                       granularity exists (the lowest one).
#   mlk               : k-th mantissa level (1-indexed from LSB).
#                       ml1 = level containing the LSB.
#                       ml2 = next level above ml1, etc.
#                       For a 4-sharing-level config, levels are ml1..ml4.
#   ml_bitwidth       : list of length ml_depth, [ml1_bitwidth, ml2_bitwidth, ...].
#                       Each entry is the number of bits at that level.
#                       Example: ml_depth=4 with [2,2,2,2] -> ml1..ml4 each hold
#                       2 bits.
#   ml_mg             : list of length ml_depth, [ml1_mg, ml2_mg, ...].
#                       Each entry is the GROUP SIZE at that level, i.e. how
#                       many adjacent elements share a value at this level.
#                       mlN_mg = 1 means no sharing at level N (passthrough).
#                       Example: ml1_mg=2 means LSB-level bits are shared
#                       across every 2 adjacent elements.
#   ml_sharingmode    : list of length ml_depth, [ml1_sharingmode, ...].
#                       Reduction operator per level. Supported values:
#                         'max', 'min', 'mean', 'round_mean', 'majority'
#                       When the corresponding mlN_mg == 1, the sharingmode
#                       is ignored and the level passes through unchanged.
#
#   encoding          : 'twos_complement' (default) | 'sign_magnitude'
#                       Selects the bit-domain in which the mantissa partition
#                       is performed. See _apply_ssnf_share for semantics.
#   carry_compensate  : bool. If True, subtract 1 from the next-higher
#                       level (ml2) wherever the ml1 sharing turned a 0
#                       into the all-ones pattern (2^bits-1), as a single
#                       one-step boundary correction. Only ml1 -> ml2 is
#                       affected; no further propagation occurs.
#
# ─── Scale formula (OCP MX standard) ──────────────────────────────────────────
#   For an n-bit signed integer element:
#       E_max     = n - 2
#       shared_exp = floor(log2(max_abs)) - E_max
#       s          = 2 ** shared_exp     (clamped to E8M0 range)
#
# ─── Supported num_format values ──────────────────────────────────────────────
#   'fp32'                                  : pass-through
#   'mxint4' / 'mxint6' / 'mxint7' / 'mxint8' : MX signed integer (OCP scale)
#   'ssnf'                                  : SSNF hierarchical sharing
#
# Public API:
#   SEED, same_seeds, ssnf_quant, compute_bits_per_elem

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import os
import math
from typing import List, Optional, Sequence

# --- 시드 고정 ---
SEED = 1111
def same_seeds(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

same_seeds(SEED)


class rounding_modes:
    STOC, DETERM = 'stoc', 'determ'
    modes = [STOC, DETERM]


def round_tensor(t, mode, device):
    """Deterministic or stochastic rounding."""
    if mode == rounding_modes.STOC:
        if device == "cpu":
            same_seeds(SEED)
            sampled = torch.FloatTensor(t.size(), device=device).uniform_(-0.5, 0.5)
        else:
            sampled = torch.empty_like(t).uniform_(-0.5, 0.5)
        return sampled.add_(t).round()
    elif mode == rounding_modes.DETERM:
        return t.round()
    else:
        raise NotImplementedError(f"Rounding mode {mode} is not implemented")


# --- 1D block utility ---
def block_tensor(t: torch.Tensor, block_size: int):
    """Flatten t and reshape into (-1, block_size), zero-padding the tail."""
    orig_shape = t.shape
    t_flat = t.flatten()
    num_elements = t_flat.numel()

    eff_block_size = num_elements if block_size == -1 else block_size

    pad_len = (eff_block_size - (num_elements % eff_block_size)) % eff_block_size
    if pad_len > 0:
        t_flat = F.pad(t_flat, (0, pad_len), 'constant', 0)

    return t_flat.reshape(-1, eff_block_size), orig_shape, pad_len


def unblock_tensor(t_blocked: torch.Tensor, orig_shape: torch.Size, pad_len: int):
    """Inverse of block_tensor: drop padding and restore original shape."""
    t_flat = t_blocked.flatten()
    if pad_len > 0:
        t_flat = t_flat[:-pad_len]
    return t_flat.reshape(orig_shape)


# --- OCP MX shared-scale (E8M0) ---
def _ocp_scale_e8m0(t_abs_max_blk: torch.Tensor, elem_bitwidth: int, device, dtype):
    """
    OCP MX standard shared-exponent scale for an n-bit signed integer element.

    Args:
        t_abs_max_blk : per-block max-abs, shape (B, 1).
        elem_bitwidth : signed integer bit-width n.
    Formula:
        E_max      = n - 2                  (= floor(log2(2^(n-1)-1)) for n>=2)
        shared_exp = floor(log2(max_abs)) - E_max
        s          = 2 ** shared_exp        (clamped to [-127, 127])
    """
    if elem_bitwidth < 2:
        raise ValueError(f"elem_bitwidth must be >= 2, got {elem_bitwidth}")

    max_abs = torch.where(
        t_abs_max_blk == 0,
        torch.tensor(1e-30, device=device, dtype=t_abs_max_blk.dtype),
        t_abs_max_blk,
    )
    e_max = float(elem_bitwidth - 2)
    shared_exp = torch.floor(torch.log2(max_abs)) - e_max
    shared_exp = torch.clamp(shared_exp, min=-127.0, max=127.0)
    s = torch.exp2(shared_exp)
    return s.to(dtype)


# --- Plain MX signed-integer quantizer (for num_format='mxintN') ---
def _quantize_mx_int_ocp(t_blocked: torch.Tensor, elem_bitwidth: int, device):
    """MX signed-integer quantizer with OCP E8M0 shared scale."""
    orig_dtype = t_blocked.dtype
    xf = t_blocked.to(torch.float32)

    max_abs = xf.abs().amax(dim=-1, keepdim=True)
    s = _ocp_scale_e8m0(max_abs, elem_bitwidth, device, torch.float32)

    quant_max = (1 << (elem_bitwidth - 1)) - 1
    q = torch.round(xf / s).clamp(-quant_max, quant_max)
    return (q * s).to(orig_dtype)


# --- Mantissa-level bit-shift table ---
def _build_level_shifts(ml_bitwidth: Sequence[int], partition_bits: int) -> List[int]:
    """
    Build per-level right-shift amounts for bit extraction.
    ml_bitwidth is ordered LSB-first: ml_bitwidth[0] == ml1 == LSB level.
    Returns shifts[0..ml_depth-1] where shifts[k] is the right-shift for level k.
    Levels above sum(ml_bitwidth) (i.e. unsharing region) are implicit.
    """
    if sum(ml_bitwidth) > partition_bits:
        raise ValueError(
            f"sum(ml_bitwidth)={sum(ml_bitwidth)} exceeds partition_bits={partition_bits}"
        )
    shifts, off = [], 0
    for b in ml_bitwidth:
        shifts.append(off)
        off += b
    return shifts


# --- Group slicing utility ---
def _group_slices(block_size: int, mg: int):
    """
    Yield (start, end) index pairs along block dim such that each slice covers
    `mg` adjacent elements (the last slice may be shorter if block_size % mg != 0).
    """
    mg = max(1, min(mg, block_size))
    n_groups = math.ceil(block_size / mg)
    for g in range(n_groups):
        s = g * mg
        e = min((g + 1) * mg, block_size)
        if s < e:
            yield s, e


# --- Per-level group-sharing reduction ---
def _apply_share(level_part: torch.Tensor, mg: int,
                 sharingmode: str, bitwidth: int) -> torch.Tensor:
    """
    Apply per-group sharing on the bit-extracted `level_part`.

    Args:
        level_part  : (B, BLK) integer tensor, values in [0, 2^bitwidth - 1].
        mg          : group size (number of adjacent elements that share).
                      mg <= 1 means no sharing -> passthrough.
        sharingmode : one of {'max', 'min', 'mean', 'round_mean', 'majority'}.
        bitwidth    : number of bits at this level (used for 'majority' num_classes).
    """
    # mg=1 (or 0) means no sharing
    if mg <= 1:
        return level_part

    B, BLK = level_part.shape
    out = torch.empty_like(level_part)
    for s, e in _group_slices(BLK, mg):
        seg = level_part[:, s:e]
        gs_actual = seg.shape[1]

        if sharingmode == 'max':
            rep, _ = seg.max(dim=1, keepdim=True)
        elif sharingmode == 'min':
            rep, _ = seg.min(dim=1, keepdim=True)
        elif sharingmode == 'mean':
            # Hardware-friendly binary-tree mean: pairwise add-and-shift
            # ((a + b) >> 1) repeated until a single value remains. For
            # power-of-two `mg` this is the floor-mean of the group; for
            # non-power-of-two `mg` it is a weighted average (the trailing
            # "leftover" elements carry larger weight because they pass
            # through fewer averaging rounds).
            current = seg
            while current.shape[1] > 1:
                evens = current[:, 0::2]
                odds  = current[:, 1::2]
                min_len = min(evens.shape[1], odds.shape[1])
                merged = (evens[:, :min_len] + odds[:, :min_len]) >> 1
                if current.shape[1] % 2 != 0:
                    leftover = current[:, -1:]
                    current = torch.cat([merged, leftover], dim=1)
                else:
                    current = merged
            rep = current
        elif sharingmode == 'round_mean':
            # Round-half-up mean (assumes non-negative bit-extracted inputs)
            seg64 = seg.to(torch.int64)
            rep = ((seg64.sum(dim=1, keepdim=True) + gs_actual // 2) // gs_actual).to(seg.dtype)
        elif sharingmode == 'majority':
            num_classes = 1 << bitwidth
            counts = F.one_hot(seg.to(torch.int64), num_classes=num_classes).sum(dim=1)
            rep = counts.argmax(dim=1, keepdim=True).to(seg.dtype)
        else:
            raise ValueError(f"unknown sharingmode: {sharingmode!r}")
        out[:, s:e] = rep
    return out


# --- Core: SSNF bit-level hierarchical sharing ---
def _apply_ssnf_share(q_in: torch.Tensor,
                      elem_bitwidth: int,
                      ml_bitwidth: Sequence[int],
                      ml_mg: Sequence[int],
                      ml_sharingmode: Sequence[str],
                      carry_compensate: bool,
                      encoding: str) -> torch.Tensor:
    """
    Bit-level hierarchical sharing on bit-partitioned integer tensors.

    Input encoding:
      'twos_complement' : q_in is signed int in [-(2^(n-1)-1), +(2^(n-1)-1)].
                          Reinterpreted internally as an unsigned n-bit pattern
                          (e.g. -5 with n=8 -> 0b11111011 = 251). All n bits,
                          including the MSB (sign bit), participate in the
                          partition.
      'sign_magnitude'  : q_in is non-negative magnitude in [0, 2^(n-1)-1].
                          The sign is stored separately; only the (n-1)
                          magnitude bits participate in the partition.

    Output: integer tensor in the same encoding as the input.

    ml_bitwidth/ml_mg/ml_sharingmode are LSB-first (index 0 == ml1 == LSB level).
    """
    if encoding == 'twos_complement':
        partition_bits = elem_bitwidth
    elif encoding == 'sign_magnitude':
        partition_bits = elem_bitwidth - 1
    else:
        raise ValueError(f"unknown encoding: {encoding!r}")

    ml_depth = len(ml_bitwidth)
    sum_ml_bits = sum(ml_bitwidth)
    # Allow ml_bitwidth sum to exceed partition_bits; the upper virtual bits
    # are simply 0. (Preserves original behavior for [2,2,2,2] + sign_magnitude.)
    width_for_share = max(sum_ml_bits, partition_bits)

    # Re-interpret q_in as unsigned bit pattern for bit-partition
    if encoding == 'twos_complement':
        mask_all = (1 << elem_bitwidth) - 1
        q_uint = q_in.to(torch.int64) & mask_all
    else:
        q_uint = q_in.to(torch.int64)

    # Memory optimization: downcast where safe
    if width_for_share <= 15:
        q_uint = q_uint.to(torch.int16)
    elif width_for_share <= 31:
        q_uint = q_uint.to(torch.int32)

    shifts = _build_level_shifts(ml_bitwidth, width_for_share)

    # Initialize comp with the UNSHARED upper bits preserved from q_uint.
    # If sum(ml_bitwidth) < partition_bits, the bits above the sharing region
    # must pass through per-element (no sharing applied to them).
    # If sum(ml_bitwidth) >= partition_bits, there is no unshared region and
    # comp starts at zero (everything will be filled by the sharing loop).
    if sum_ml_bits < partition_bits:
        upper_shift = sum_ml_bits
        # Mask covering bits [sum_ml_bits, partition_bits)
        upper_mask  = ((1 << (partition_bits - sum_ml_bits)) - 1) << upper_shift
        comp = q_uint & upper_mask
    else:
        comp = torch.zeros_like(q_uint)

    carry_in = torch.zeros_like(q_uint)

    # Iterate LSB -> upper (ml1 -> ml_{depth})
    for k in range(ml_depth):
        shift = shifts[k]
        bits  = ml_bitwidth[k]
        mg    = ml_mg[k]
        mode  = ml_sharingmode[k]
        mask  = (1 << bits) - 1

        part = (q_uint >> shift) & mask
        part_w_carry = torch.clamp(part - carry_in, min=0)
        part_shared = _apply_share(part_w_carry, mg, mode, bits)

        comp = comp | (part_shared << shift)

        # Compute boundary-correction signal only at ml1 (LSB level) when
        # requested. The next iteration (ml2) subtracts this from its
        # part, effectively cancelling a single 0 -> 2^bits-1 carry that
        # the ml1 sharing introduced. No further levels see a carry.
        if carry_compensate and k == 0:
            max_val_level = (1 << bits) - 1
            carry_condition = (part_w_carry == 0) & (part_shared == max_val_level)
            carry_in = carry_condition.to(q_uint.dtype)
        else:
            carry_in.zero_()

        del part, part_w_carry, part_shared

    comp = comp.to(torch.int64)

    # Convert back to original encoding
    if encoding == 'twos_complement':
        # Reinterpret unsigned n-bit pattern as signed
        half = 1 << (elem_bitwidth - 1)
        full = 1 << elem_bitwidth
        comp = torch.where(comp >= half, comp - full, comp)
        # Clamp to symmetric signed range [-quant_max, +quant_max]
        # (bit-sharing may produce -2^(n-1), which is outside the MX range)
        quant_max = (1 << (elem_bitwidth - 1)) - 1
        comp = torch.clamp(comp, -quant_max, +quant_max)

    return comp


# --- Float -> integer quantization with OCP scale ---
def _float_to_int_ocp(t_blocked: torch.Tensor,
                      elem_bitwidth: int,
                      rounding_mode: str,
                      device,
                      encoding: str):
    """
    Quantize block-wise float input to signed integer using OCP shared scale.

    Returns:
        encoding == 'twos_complement' : (q_int, None, s)
                                        q_int in [-quant_max, +quant_max]
        encoding == 'sign_magnitude'  : (q_mag, sign, s)
                                        q_mag in [0, quant_max]
                                        sign  in {-1, +1}
    """
    orig_dtype = t_blocked.dtype
    xf = t_blocked.to(torch.float32)

    max_abs = xf.abs().amax(dim=-1, keepdim=True)
    s = _ocp_scale_e8m0(max_abs, elem_bitwidth, device, torch.float32)

    quant_max = (1 << (elem_bitwidth - 1)) - 1
    t_norm = xf / s

    if encoding == 'twos_complement':
        q = round_tensor(t_norm, rounding_mode, device).to(torch.int64)
        q = torch.clamp(q, -quant_max, +quant_max)
        return q, None, s.to(orig_dtype)
    elif encoding == 'sign_magnitude':
        sign = torch.sign(t_blocked)
        sign[sign == 0] = 1
        q_mag = round_tensor(t_norm.abs(), rounding_mode, device).to(torch.int64)
        q_mag = torch.clamp(q_mag, 0, quant_max)
        return q_mag, sign, s.to(orig_dtype)
    else:
        raise ValueError(f"unknown encoding: {encoding!r}")


# --- Argument unpacking with defaults ---
def unpack_quant_args(kwargs):
    quant_args = {}
    defaults = [
        ('num_format',       'fp32'),
        ('rounding_mode',    'determ'),
        ('device',           'cpu'),
        ('block_size',       32),
        ('elem_bitwidth',    8),
        ('ml_bitwidth',      None),                # list, LSB-first
        ('ml_mg',            None),                # list, LSB-first
        ('ml_sharingmode',   None),                # list, LSB-first
        ('carry_compensate', False),
        ('encoding',         'twos_complement'),
    ]
    for arg, default in defaults:
        quant_args[arg] = kwargs.get(arg, default)
    return quant_args


def _validate_ssnf_lists(ml_bitwidth, ml_mg, ml_sharingmode):
    """Validate the three per-level lists have matching length and types."""
    if ml_bitwidth is None or ml_mg is None or ml_sharingmode is None:
        raise ValueError(
            "For num_format='ssnf', all of ml_bitwidth, ml_mg, ml_sharingmode "
            "must be provided as lists of equal length."
        )
    ml_depth = len(ml_bitwidth)
    if len(ml_mg) != ml_depth or len(ml_sharingmode) != ml_depth:
        raise ValueError(
            f"Length mismatch: ml_bitwidth={len(ml_bitwidth)}, "
            f"ml_mg={len(ml_mg)}, ml_sharingmode={len(ml_sharingmode)}"
        )
    return ml_depth


# === Main Entry Point ===
def float_to_format_tiled(t: torch.Tensor, **kwargs):
    """
    Quantize tensor `t` according to the format in `kwargs['num_format']`.

    Supported num_format:
      'fp32'                                : pass-through
      'mxint4' / 'mxint6' / 'mxint7' / 'mxint8'
                                            : OCP-scaled MX signed integer
      'ssnf'                                : SSNF hierarchical sharing
                                              (requires ml_bitwidth, ml_mg,
                                              ml_sharingmode lists)
    """
    args = unpack_quant_args(kwargs)
    fmt = args['num_format'].lower()
    if fmt == 'fp32':
        return t

    device = t.device
    bs = args['block_size'] if args['block_size'] != -1 else 32

    # --- MXINT branch ---
    if fmt.startswith('mxint') and fmt != 'mxint':
        if   '8' in fmt: ebw = 8
        elif '7' in fmt: ebw = 7
        elif '6' in fmt: ebw = 6
        elif '4' in fmt: ebw = 4
        else: raise ValueError(f"Unknown MXINT format: {fmt}")
        t_blk, orig, pad = block_tensor(t, bs)
        t_out = _quantize_mx_int_ocp(t_blk, ebw, device)
        return unblock_tensor(t_out, orig, pad)

    # --- SSNF branch ---
    if fmt == 'ssnf':
        elem_bitwidth = args['elem_bitwidth']
        encoding      = args['encoding']
        if encoding not in ('twos_complement', 'sign_magnitude'):
            raise ValueError(f"unknown encoding: {encoding!r}")

        _validate_ssnf_lists(args['ml_bitwidth'], args['ml_mg'], args['ml_sharingmode'])

        t_blocked, orig_shape, pad_len = block_tensor(t, bs)

        q_or_mag, sign, s = _float_to_int_ocp(
            t_blocked,
            elem_bitwidth=elem_bitwidth,
            rounding_mode=args['rounding_mode'],
            device=args['device'],
            encoding=encoding,
        )

        q_final = _apply_ssnf_share(
            q_or_mag,
            elem_bitwidth=elem_bitwidth,
            ml_bitwidth=args['ml_bitwidth'],
            ml_mg=args['ml_mg'],
            ml_sharingmode=args['ml_sharingmode'],
            carry_compensate=args['carry_compensate'],
            encoding=encoding,
        )

        if encoding == 'twos_complement':
            t_q_blocked = q_final.to(t.dtype) * s.to(t.dtype)
        else:
            t_q_blocked = (q_final.to(t.dtype) * s.to(t.dtype)) * sign.to(t.dtype)

        return unblock_tensor(t_q_blocked, orig_shape, pad_len)

    raise NotImplementedError(
        f"num_format {fmt!r} is not supported. "
        f"Supported: 'fp32', 'mxint4', 'mxint6', 'mxint7', 'mxint8', 'ssnf'."
    )


class SSNFQuant(torch.autograd.Function):
    @staticmethod
    def forward(ctx, t, quant_config: dict):
        ctx.quant_config = quant_config
        if quant_config.get('num_format', 'fp32') == 'fp32':
            return t
        return float_to_format_tiled(t, **quant_config)
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


def ssnf_quant(t: torch.Tensor, **kwargs):
    """
    Public entry point. See `float_to_format_tiled` for accepted kwargs.
    """
    if not t.requires_grad:
        if kwargs.get('num_format', 'fp32') == 'fp32':
            return t
        return float_to_format_tiled(t, **kwargs)
    else:
        return SSNFQuant.apply(t, kwargs)


# --- Bit budget helper ---
def compute_bits_per_elem(elem_bitwidth: int,
                          ml_bitwidth: Optional[Sequence[int]] = None,
                          ml_mg: Optional[Sequence[int]] = None,
                          block_size: int = 32,
                          scale_bits: int = 8,
                          encoding: str = 'twos_complement') -> float:
    """
    Compute the effective per-element bit cost after sharing.

    The cost is the sum of:
        unshared_bits      : per-element bits above the sharing region,
                             = max(partition_bits - sum(ml_bitwidth), 0)
        shared_bits        : per-element amortised cost of the shared region,
                             = sum_k [ ml_bitwidth[k] / ml_mg[k] ]
        sign_bit_per_elem  : 1 if encoding == 'sign_magnitude' else 0
        scale_overhead     : scale_bits / block_size

    partition_bits depends on encoding:
        'twos_complement' : partition_bits = elem_bitwidth
                            (all n bits participate, including the sign bit)
        'sign_magnitude'  : partition_bits = elem_bitwidth - 1
                            (only magnitude bits are partitioned; the
                             sign is stored per-element separately)

    Notes:
        - mlN_mg <= 1 contributes its full ml_bitwidth[k] per element
          (no sharing at that level).
        - If sum(ml_bitwidth) > partition_bits, the excess bits are
          "virtual upper bits" that are always zero in ssnf_core and
          carry no storage cost, so they are not counted in shared_bits.

    Examples:
        MXINT8 baseline (no sharing):
            compute_bits_per_elem(8) == 8.25
            compute_bits_per_elem(8, encoding='sign_magnitude') == 8.25
        SSNF ml1_bitwidth=2, ml1_mg=2 over MXINT8 (twos_complement):
            compute_bits_per_elem(8, [2], [2]) == 7.25
        SSNF ml_bitwidth=[2,2,2,2], ml_mg=[2,2,2,2] over MXINT8
        (sign_magnitude, partition_bits=7, virtual upper bit unused):
            ((2+2+2+2) clipped to 7) sharing region averaged over groups
            + 1 sign + 0.25 scale.
    """
    if encoding == 'twos_complement':
        partition_bits = elem_bitwidth
        sign_bit_per_elem = 0.0
    elif encoding == 'sign_magnitude':
        partition_bits = elem_bitwidth - 1
        sign_bit_per_elem = 1.0
    else:
        raise ValueError(f"unknown encoding: {encoding!r}")

    overhead = scale_bits / block_size

    if ml_bitwidth is None or ml_mg is None:
        # No sharing at all -> full per-element cost is partition_bits
        # of mantissa + sign_bit_per_elem + scale overhead.
        return partition_bits + sign_bit_per_elem + overhead

    if len(ml_bitwidth) != len(ml_mg):
        raise ValueError("ml_bitwidth and ml_mg must have the same length")

    sum_ml = sum(ml_bitwidth)
    # Bits in the sharing region that actually carry information.
    effective_sum_ml = min(sum_ml, partition_bits)
    unshared_bits = partition_bits - effective_sum_ml

    # Amortised cost of the shared region. Walk LSB-first and only count
    # bits that fall within partition_bits (anything above is virtual).
    shared_bits = 0.0
    bits_seen = 0
    for b, g in zip(ml_bitwidth, ml_mg):
        if bits_seen >= partition_bits:
            break
        eff_b = min(b, partition_bits - bits_seen)
        if g <= 1:
            shared_bits += eff_b           # no sharing -> full cost
        else:
            shared_bits += eff_b / g       # 1 shared slice per group of g
        bits_seen += b

    return unshared_bits + shared_bits + sign_bit_per_elem + overhead


# --- Quantization target selector (shared by QSNR hooks and PPL forward-patch) ---
def is_quant_target_linear(name: str, mod: nn.Module) -> bool:
    """Single source of truth for which Linear submodules are quantized.

    Used by both the QSNR forward-hook and the PPL forward-patch paths so that
    the two measurements always cover the identical layer set. Targets decoder
    block Linears (``model.layers.*``) and explicitly excludes ``lm_head``.
    """
    if not isinstance(mod, nn.Linear):
        return False
    if "lm_head" in name:
        return False
    if "model.layers" not in name:
        return False
    return True


print("ssnf_core.py (new naming: ml_depth/mlN_bitwidth/mlN_mg/mlN_sharingmode/encoding) loaded")