# multi_level_mantissa_sharing.py
#
# Multi-Level Mantissa Sharing sweep, narrowed for bits_per_elem ∈ [5, 7).
#
# Compared to the previous version of this script, the sweep is now driven
# by an explicit static list `SWEEP_TRIPLETS` of (ml_depth, ml_bitwidth, ml_mg)
# triplets rather than enumerated from a constants × monotonic-combinations
# grid. This is needed because:
#   - The previous sweep fixed ml_bitwidth = [2]*ml_depth at every level and
#     swept ml_mg. None of those configs satisfied the BF16 +3% PPL bar
#     (closest: depth=2, mg=(2,2), bpe=6.25 -> PPL +27.6%), and the
#     depth=3/4 results diverged catastrophically.
#   - The new sweep allows mixed per-level ml_bitwidth ∈ {1, 2}, exploring
#     bw=[1,1], [1,2], [2,1], [2,2] (depth=2) and [1,1,1] (depth=3 only).
#   - Configs are chosen so multiple bw variants land on the same bpe band
#     (iso-bpe horizontal comparison) across [5, 7) at 0.25 resolution.
#
# - Fixed: elem_bitwidth = 8 (MXINT8 baseline as reference)
# - Per-level mantissa sharing still uses a single user-selected sharingmode
#   (applied identically to every ml level). NB: with bw=1 levels (16 of 21
#   sweep configs), 'mean' (pairwise floor) and 'round_mean' (round-half-up)
#   diverge in opposite directions on {0,1} pairs; 'round_mean' is the
#   recommended default.
# - Interactive prompts at start-up for:
#     * sharing_mode      ∈ {mean, max, min, majority, round_mean (default)}
#     * rounding_mode     ∈ {determ (default), stoc}
#     * encoding          ∈ {twos_complement (default), sign_magnitude}
#     * metric_mode       ∈ {both (default), qsnr_only, ppl_only}
#
# - Metrics: QSNR and/or Perplexity (sliding-window on WikiText-2)
# - Outputs: CSV + plots, annotated with the active modes both
#            in filenames, in the dataframe (Mode columns), and in plot titles.

import gc
import math
import os
import sys
import types
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from ssnf_main.ForGit.src.ssnf_core import SEED, same_seeds, ssnf_quant, compute_bits_per_elem, is_quant_target_linear
from ssnf_main.ForGit.src.ssnf_sdpa_patch import (
    patch_sdpa_for_ssnf, unpatch_sdpa,
    record_skip, reset_skip_log, dump_skip_report,
)


# ═════════════════════════════════════════════════════════════════════════════
# Global config
# ═════════════════════════════════════════════════════════════════════════════
MODEL_NAME        = "meta-llama/Llama-3.1-8B-Instruct"
DEVICE            = "cuda" if torch.cuda.is_available() else "cpu"
NUM_CALIB_SAMPLES = 16
MAX_SEQ_LEN       = 128
BLOCK_SIZE        = 32

# Fixed experiment parameters
ELEM_BITWIDTH = 8

# Static sweep list: each entry is (ml_depth, ml_bitwidth, ml_mg), all
# LSB-first. Selected for bits_per_elem ∈ [5, 7) with iso-bpe horizontal
# comparison across bw variants. See module docstring for the design.
SWEEP_TRIPLETS = [
    # bpe = 6.875  -- depth=2 [1,1] family top
    (2, [1, 1],    [8,  2]),

    # bpe = 6.75   -- 3-way iso-bpe (depth=2) + depth=3
    (2, [1, 1],    [4,  4]),
    (2, [1, 2],    [2,  2]),
    (2, [2, 1],    [2,  2]),
    (3, [1, 1, 1], [2,  2, 2]),

    # bpe = 6.50   -- iso-bpe + depth=3
    (2, [1, 1],    [8,  8]),
    (2, [1, 2],    [4,  2]),
    (3, [1, 1, 1], [4,  2, 2]),

    # bpe = 6.25   -- iso-bpe + depth=3
    # ([2,2] mg=(2,2) is the previous-sweep best with PPL +27.6%)
    (2, [2, 1],    [4,  2]),
    (2, [2, 2],    [2,  2]),
    (3, [1, 1, 1], [4,  4, 2]),

    # bpe = 6.00   -- 4-way: bw split direction × within-bw mg distribution
    (2, [1, 2],    [4,  4]),     # bw symmetric-mg
    (2, [2, 1],    [8,  2]),     # bw reversed,    mg asymmetric
    (2, [2, 1],    [4,  4]),     # bw reversed,    mg symmetric
    (3, [1, 1, 1], [4,  4, 4]),

    # bpe = 5.75   -- iso-bpe
    (2, [2, 1],    [8,  4]),
    (2, [2, 2],    [4,  2]),

    # bpe = 5.50   -- iso-bpe
    (2, [2, 1],    [16, 8]),
    (2, [2, 2],    [8,  2]),

    # bpe = 5.25   -- single point (lower-bound exploration)
    (2, [2, 2],    [4,  4]),

    # bpe = 5.00   -- single point (lower-bound boundary)
    (2, [2, 2],    [8,  4]),
]

# Perplexity sliding-window settings (only used when metric_mode includes PPL)
PPL_SEQ_LEN   = 2048
PPL_STRIDE    = 2048
PPL_MAX_STEPS = 10**9


# ═════════════════════════════════════════════════════════════════════════════
# Interactive mode selection
# ═════════════════════════════════════════════════════════════════════════════
SHARING_MODE_OPTIONS  = ["round_mean", "mean", "max", "min", "majority"]
ROUNDING_MODE_OPTIONS = ["determ", "stoc"]
ENCODING_OPTIONS      = ["twos_complement", "sign_magnitude"]
METRIC_MODE_OPTIONS   = ["both", "qsnr_only", "ppl_only"]


def _prompt_mode(label: str, options: list, default: str) -> str:
    """
    Interactively pick a mode from `options`. Acceptance rules:
      - empty input  or 'y' / 'Y'        -> default
      - exact match in `options`         -> that option
      - anything else                    -> re-prompt
    """
    others = [o for o in options if o != default]
    others_str = ", ".join(others)
    print(f"\n=== {label} ===")
    print(f"  default : {default}")
    print(f"  others  : {others_str}")
    while True:
        ans = input(f"Use default '{default}'? [y / or mode name]: ").strip()
        if ans == "" or ans.lower() == "y":
            print(f"  -> using {label} = {default}")
            return default
        if ans in options:
            print(f"  -> using {label} = {ans}")
            return ans
        print(f"  !! '{ans}' is not a valid option. Try again.")


def prompt_all_modes() -> dict:
    """Run the four interactive prompts and return a dict of mode choices.

    Default sharing_mode is 'round_mean' because 16 of the 21 sweep configs
    include at least one bw=1 level; on bw=1 the 'mean' operator (pairwise
    floor) and 'round_mean' (round-half-up) round in opposite directions
    on {0,1} pairs.
    """
    print("=" * 70)
    print("Multi-Level Mantissa Sharing Experiment - Mode Selection")
    print("=" * 70)
    return {
        "sharing_mode":  _prompt_mode("sharing_mode",  SHARING_MODE_OPTIONS,  "round_mean"),
        "rounding_mode": _prompt_mode("rounding_mode", ROUNDING_MODE_OPTIONS, "determ"),
        "encoding":      _prompt_mode("encoding",      ENCODING_OPTIONS,      "twos_complement"),
        "metric_mode":   _prompt_mode("metric_mode",   METRIC_MODE_OPTIONS,   "both"),
    }


# ═════════════════════════════════════════════════════════════════════════════
# Bit-budget helpers
# ═════════════════════════════════════════════════════════════════════════════
def bits_mxint8() -> int:
    """MXINT8 baseline bits-per-block (block_size * 8 mantissa + 8 scale)."""
    return 8 + 8 * BLOCK_SIZE


def bits_multi_level_sharing(ml_bitwidth: list, ml_mg: list,
                             encoding: str = 'twos_complement') -> int:
    """
    Bits-per-block for a multi-level mantissa-shared MXINT8.

    With encoding='twos_complement', all `ELEM_BITWIDTH` bits participate
    in the partition. With encoding='sign_magnitude', only the
    (ELEM_BITWIDTH - 1) magnitude bits are partitioned and the sign is
    stored separately, costing one extra bit per element.

    Above the shared region, max(partition_bits - sum(ml_bitwidth), 0)
    bits are kept per element. Each shared level contributes
    `ml_bitwidth[k]` bits per group of `ml_mg[k]` elements. Bits of the
    shared region that fall above `partition_bits` ("virtual upper bits"
    in ssnf_core) are always zero and carry no storage cost.
    """
    if encoding == 'twos_complement':
        partition_bits   = ELEM_BITWIDTH
        sign_bits_block  = 0
    elif encoding == 'sign_magnitude':
        partition_bits   = ELEM_BITWIDTH - 1
        sign_bits_block  = BLOCK_SIZE          # 1 sign bit per element
    else:
        raise ValueError(f"unknown encoding: {encoding!r}")

    sum_ml          = sum(ml_bitwidth)
    effective_sum   = min(sum_ml, partition_bits)
    unshared_bits   = partition_bits - effective_sum

    bits = 8 + unshared_bits * BLOCK_SIZE + sign_bits_block

    # Walk LSB-first; only bits below partition_bits carry storage cost.
    bits_seen = 0
    for b, g in zip(ml_bitwidth, ml_mg):
        if bits_seen >= partition_bits:
            break
        eff_b = min(b, partition_bits - bits_seen)
        bits += eff_b * math.ceil(BLOCK_SIZE / g)
        bits_seen += b

    return bits


# ═════════════════════════════════════════════════════════════════════════════
# Experiment config builder
# ═════════════════════════════════════════════════════════════════════════════
def get_experiment_configs(modes: dict) -> list:
    """
    Build the full config list:
      1. MXINT8 baseline (always twos_complement encoding regardless of the
         chosen SSNF encoding; the scored reference baseline.)
      2. SSNF sweep over SWEEP_TRIPLETS = list of (ml_depth, ml_bitwidth, ml_mg).
    """
    configs = []

    # --- 1. MXINT8 baseline ---
    bpe_baseline = compute_bits_per_elem(elem_bitwidth=ELEM_BITWIDTH,
                                         ml_bitwidth=None, ml_mg=None,
                                         block_size=BLOCK_SIZE, scale_bits=8,
                                         encoding='twos_complement')
    configs.append({
        "name":          "MXINT8\n(baseline)",
        "group":         "baseline",
        "ml_depth":      0,
        "ml_bitwidth":   None,
        "ml_mg":         None,
        "total_bits":    bits_mxint8(),
        "bits_per_elem": bpe_baseline,
        "fn":            (lambda x: ssnf_quant(
                              x,
                              num_format='mxint8',
                              block_size=BLOCK_SIZE,
                              elem_bitwidth=ELEM_BITWIDTH,
                              rounding_mode=modes['rounding_mode'],
                          )),
    })

    # --- 2. SSNF sweep over the static triplet list ---
    for ml_depth, ml_bitwidth, ml_mg in SWEEP_TRIPLETS:
        assert len(ml_bitwidth) == ml_depth and len(ml_mg) == ml_depth, \
            f"length mismatch in triplet (depth={ml_depth}, bw={ml_bitwidth}, mg={ml_mg})"
        ml_sharingmode = [modes['sharing_mode']] * ml_depth
        tb  = bits_multi_level_sharing(ml_bitwidth, ml_mg, encoding=modes['encoding'])
        bpe = compute_bits_per_elem(
            elem_bitwidth=ELEM_BITWIDTH,
            ml_bitwidth=ml_bitwidth, ml_mg=ml_mg,
            block_size=BLOCK_SIZE, scale_bits=8,
            encoding=modes['encoding'],
        )
        bw_str = "_".join(str(b) for b in ml_bitwidth)
        mg_str = "_".join(str(g) for g in ml_mg)
        config_name = f"d{ml_depth}\nbw=({bw_str})\nmg=({mg_str})\n({tb}b)"
        configs.append({
            "name":          config_name,
            "group":         f"depth{ml_depth}",
            "ml_depth":      ml_depth,
            "ml_bitwidth":   tuple(ml_bitwidth),
            "ml_mg":         tuple(ml_mg),
            "total_bits":    tb,
            "bits_per_elem": bpe,
            "fn":            (lambda _bw, _mg, _sm: lambda x: ssnf_quant(
                                  x,
                                  num_format='ssnf',
                                  block_size=BLOCK_SIZE,
                                  elem_bitwidth=ELEM_BITWIDTH,
                                  ml_bitwidth=list(_bw),
                                  ml_mg=list(_mg),
                                  ml_sharingmode=list(_sm),
                                  encoding=modes['encoding'],
                                  rounding_mode=modes['rounding_mode'],
                              ))(ml_bitwidth, ml_mg, ml_sharingmode),
        })
    return configs


# ═════════════════════════════════════════════════════════════════════════════
# Activation-quant safety wrapper
# ═════════════════════════════════════════════════════════════════════════════
def _maybe_quantize_activation(x: torch.Tensor, quant_fn,
                               name=None) -> torch.Tensor:
    """Apply quant_fn only if the last dim is a multiple of BLOCK_SIZE.
    Skipped tensors are logged (diagnostic only; measured values unchanged)."""
    if x.shape[-1] % BLOCK_SIZE != 0:
        record_skip("linear", name, x.shape)
        return x
    return quant_fn(x)


# ═════════════════════════════════════════════════════════════════════════════
# QSNR measurement (Weight + Activation, BF16 target)
# ═════════════════════════════════════════════════════════════════════════════
def measure_qsnr(model, inputs, quant_fn) -> float:
    signal = 0.0
    noise  = 0.0
    def make_hook(fn, layer_name):
        def hook_fn(mod, inp, out):
            nonlocal signal, noise
            x_bf = inp[0].detach()
            w_bf = mod.weight.detach()
            bias = mod.bias.detach() if mod.bias is not None else None

            y_ref = F.linear(x_bf, w_bf, bias).to(torch.float32)

            x_q = _maybe_quantize_activation(x_bf, fn, name=layer_name)
            w_q = fn(w_bf)
            y_q = F.linear(x_q, w_q, bias).to(torch.float32)

            signal += y_ref.pow(2).sum().item()
            noise  += (y_ref - y_q).pow(2).sum().item()
        return hook_fn
    hooks = []
    for name, mod in model.named_modules():
        if is_quant_target_linear(name, mod):
            hooks.append(mod.register_forward_hook(make_hook(quant_fn, name)))
    with torch.no_grad():
        model(**inputs)
    for h in hooks:
        h.remove()
    if noise == 0:
        noise = 1e-12
    return 10.0 * torch.log10(torch.tensor(signal / noise)).item()


# ═════════════════════════════════════════════════════════════════════════════
# Perplexity measurement (sliding window, Weight + Activation, BF16 target)
# ═════════════════════════════════════════════════════════════════════════════
def _patch_linear_forward(model: nn.Module, quant_fn) -> dict:
    original_forwards = {}

    def _make_quantized_forward(fn, layer_name):
        def quantized_forward(self, input):
            x_q = _maybe_quantize_activation(input, fn, name=layer_name)
            w_q = fn(self.weight)
            if w_q.dtype != self.weight.dtype:
                w_q = w_q.to(self.weight.dtype)
            out = F.linear(x_q, w_q, self.bias)
            del w_q, x_q
            return out
        return quantized_forward

    for name, mod in model.named_modules():
        if not is_quant_target_linear(name, mod): continue
        original_forwards[name] = mod.forward
        mod.forward = types.MethodType(
            _make_quantized_forward(quant_fn, name), mod)

    return original_forwards


def _unpatch_linear_forward(model: nn.Module, original_forwards: dict) -> None:
    for name, mod in model.named_modules():
        if name in original_forwards:
            mod.forward = original_forwards[name]


def _measure_ppl_sliding_window(model, encodings, quant_fn) -> float:
    orig_forwards = _patch_linear_forward(model, quant_fn)
    seq_len = encodings.input_ids.size(1)
    nlls = []
    try:
        with torch.no_grad():
            pbar = tqdm(range(0, seq_len, PPL_STRIDE), desc="  PPL steps", leave=False)
            step_count = 0
            for i in pbar:
                if step_count >= PPL_MAX_STEPS: break
                begin_loc = max(i + PPL_STRIDE - PPL_SEQ_LEN, 0)
                end_loc   = min(i + PPL_STRIDE, seq_len)
                trg_len   = end_loc - i
                if trg_len < PPL_STRIDE: continue
                input_ids  = encodings.input_ids[:, begin_loc:end_loc].to(DEVICE)
                target_ids = input_ids.clone()
                target_ids[:, :-trg_len] = -100
                model.gradient_checkpointing_enable()
                outputs = model(input_ids, labels=target_ids)
                model.gradient_checkpointing_disable()
                nlls.append(outputs.loss.detach().cpu() * trg_len)
                step_count += 1
                del outputs, input_ids, target_ids
                gc.collect(); torch.cuda.empty_cache()
    finally:
        _unpatch_linear_forward(model, orig_forwards)
        gc.collect(); torch.cuda.empty_cache()
    if not nlls:
        return float("nan")
    total_trg_len = PPL_STRIDE * len(nlls)
    return torch.exp(torch.stack(nlls).sum() / total_trg_len).item()


# ═════════════════════════════════════════════════════════════════════════════
# Plotting helpers
# ═════════════════════════════════════════════════════════════════════════════
GROUP_COLORS = {
    "baseline": "#4878CF",
    "depth2":   "#FF8C00",
    "depth3":   "#2CA02C",
    "depth4":   "#9467BD",
}
GROUP_LABELS = {
    "baseline": "MXINT8 Baseline",
    "depth2":   "ml_depth=2",
    "depth3":   "ml_depth=3",
    "depth4":   "ml_depth=4",
}
GROUP_MARKER = {
    "baseline": "D",
    "depth2":   "o",
    "depth3":   "s",
    "depth4":   "^",
}


def _depths_in_sweep() -> list:
    """Return sorted unique ml_depth values present in SWEEP_TRIPLETS.

    Replaces the previous fixed ML_DEPTH_VALUES = [2, 3, 4] constant so
    that plot panels match the actual sweep contents.
    """
    return sorted({d for d, _, _ in SWEEP_TRIPLETS})


def _draw_integer_boundary_dividers(ax, df_sub: pd.DataFrame, y_top: float):
    """
    Draw vertical dashed dividers between adjacent bars whose
    floor(bits_per_elem) differs, and annotate each band with its
    integer band label (e.g. "5 ≤ bpe < 6 bits/elem").

    df_sub must already be sorted by bits_per_elem (descending in our
    convention). Bars sit at x = 0, 1, 2, ..., len(df_sub)-1.
    """
    bpe_values = df_sub["bits_per_elem"].tolist()
    n = len(bpe_values)
    if n == 0:
        return

    floors = [int(math.floor(v)) for v in bpe_values]

    # Find boundaries: pairs (i, i+1) where floor differs
    # Boundary x position is between bar i and bar i+1, i.e. at i + 0.5
    boundaries = []
    for i in range(n - 1):
        if floors[i] != floors[i + 1]:
            boundaries.append(i + 0.5)

    # Draw vertical dashed lines at each boundary
    for bx in boundaries:
        ax.axvline(x=bx, color="#444444", linestyle="--",
                   linewidth=1.0, alpha=0.55, zorder=1)

    # Compute band ranges and annotate
    band_starts = [0] + [int(b + 0.5) for b in boundaries]
    band_ends   = [int(b + 0.5) - 1 for b in boundaries] + [n - 1]

    label_y = y_top * 0.985 if y_top > 0 else 1.0
    for start, end in zip(band_starts, band_ends):
        if end < start:
            continue
        floor_val = floors[start]
        center_x  = (start + end) / 2.0
        ax.text(
            center_x, label_y,
            f"{floor_val} ≤ bpe < {floor_val + 1}",
            ha="center", va="top",
            fontsize=9, color="#333333", fontweight="bold",
            alpha=0.85,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                      edgecolor="#888888", alpha=0.75, linewidth=0.5),
        )


def _build_bar_panel_for_depth(ax, df_sub: pd.DataFrame, baseline_value: float,
                               metric: str, ylabel: str, title: str,
                               higher_better: bool, depth_color: str):
    """
    Render a single ml_depth subplot.

    df_sub: rows belonging to one ml_depth, already sorted by bits_per_elem
            descending.
    baseline_value: scalar of the metric on MXINT8 baseline (for reference line).
    """
    n = len(df_sub)
    x = list(range(n))
    values = df_sub[metric].tolist()

    bars = ax.bar(x, values, color=depth_color, edgecolor="black",
                  linewidth=0.4, zorder=2)

    # Compute y range
    finite_vals = [v for v in values
                   if isinstance(v, (int, float)) and math.isfinite(v)]
    if isinstance(baseline_value, (int, float)) and math.isfinite(baseline_value):
        finite_vals.append(baseline_value)
    if finite_vals:
        y_min_data = min(finite_vals)
        y_max_data = max(finite_vals)
        y_range = max(y_max_data - y_min_data, 1.0)
        y_bottom = y_min_data - 0.05 * y_range if higher_better else max(0, y_min_data - 0.05 * y_range)
        y_top    = y_max_data + 0.18 * y_range  # extra room for band labels
        ax.set_ylim(y_bottom, y_top)
    else:
        y_top = 1.0

    # Per-bar value labels
    for b, v in zip(bars, values):
        if not (isinstance(v, (int, float)) and math.isfinite(v)):
            continue
        ax.annotate(f"{v:.2f}",
                    (b.get_x() + b.get_width() / 2., v),
                    ha="center", va="bottom",
                    xytext=(0, 3), textcoords="offset points",
                    fontsize=7, fontweight="bold", zorder=3)

    # Baseline reference line
    if isinstance(baseline_value, (int, float)) and math.isfinite(baseline_value):
        ax.axhline(y=baseline_value, color=GROUP_COLORS["baseline"],
                   linestyle="-", linewidth=1.6, alpha=0.85, zorder=1.5,
                   label=f"MXINT8 baseline ({baseline_value:.2f})")

    # Integer-band dividers + labels
    _draw_integer_boundary_dividers(ax, df_sub, y_top=ax.get_ylim()[1])

    # X-axis cosmetics
    ax.set_xticks(x)
    ax.set_xticklabels(df_sub["Configuration"], rotation=80, ha="right", fontsize=6.5)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.grid(axis="y", linestyle="--", alpha=0.35, zorder=0)
    ax.set_title(title, fontsize=12, fontweight="bold", pad=10)

    direction = "↑ Higher is Better" if higher_better else "↓ Lower is Better"
    ax.text(0.99, 0.98, direction, transform=ax.transAxes,
            fontsize=9, color="#333333", ha="right", va="top",
            style="italic", alpha=0.7)

    ax.legend(loc="lower left", fontsize=9, framealpha=0.9)


def _split_by_depth(df: pd.DataFrame) -> dict:
    """
    Partition df into {depth: df_d} for each ml_depth present in the sweep,
    each sorted by bits_per_elem DESCENDING. Baseline is excluded (handled
    separately).
    """
    out = {}
    for depth in _depths_in_sweep():
        sub = df[df["Group"] == f"depth{depth}"].copy()
        sub = sub.sort_values(by="bits_per_elem", ascending=False, kind="mergesort")
        sub = sub.reset_index(drop=True)
        out[depth] = sub
    return out


def _baseline_value(df: pd.DataFrame, metric: str) -> float:
    base_rows = df[df["Group"] == "baseline"]
    if len(base_rows) == 0:
        return float("nan")
    v = base_rows.iloc[0][metric]
    return float(v) if isinstance(v, (int, float)) else float("nan")


def _build_legend_handles():
    handles = [Patch(facecolor=GROUP_COLORS["baseline"], edgecolor="black",
                     label=GROUP_LABELS["baseline"])]
    for depth in _depths_in_sweep():
        key = f"depth{depth}"
        handles.append(Patch(facecolor=GROUP_COLORS.get(key, "#888888"),
                             edgecolor="black",
                             label=GROUP_LABELS.get(key, key)))
    return handles


def _mode_tag(modes: dict) -> str:
    """
    Short tag for filenames. Values only (no prefix), underscore-separated:
        {rounding_mode}_{encoding}_{sharing_mode}
    Underscores inside individual values (e.g. 'sign_magnitude',
    'round_mean') are rewritten as hyphens so the tag uses '_' purely as a
    field separator.
    Note: metric_mode is intentionally excluded — plot suffixes (__QSNR /
    __PPL / __scatter) carry that information instead.
    """
    enc   = modes["encoding"].replace("_", "-")
    share = modes["sharing_mode"].replace("_", "-")
    return f"{modes['rounding_mode']}_{enc}_{share}"


def _mode_subtitle(modes: dict) -> str:
    """Human-readable mode summary for plot subtitles."""
    return (f"sharing_mode={modes['sharing_mode']} | "
            f"rounding_mode={modes['rounding_mode']} | "
            f"encoding={modes['encoding']} | "
            f"metric_mode={modes['metric_mode']}")


def _plot_metric_by_depth(df: pd.DataFrame, modes: dict, metric: str,
                          ylabel: str, title_main: str, higher_better: bool,
                          out_path: str):
    """
    Render a multi-row figure: one bar subplot per ml_depth present in the
    sweep. Each subplot is sorted by bits_per_elem desc, with integer-band
    dividers and a MXINT8 baseline horizontal reference line.
    """
    parts = _split_by_depth(df)
    baseline_val = _baseline_value(df, metric)

    depths = _depths_in_sweep()
    if not depths:
        print(f"  (no SSNF rows to plot for metric={metric!r})")
        return

    # Figure width follows the widest subplot
    max_n = max((len(parts[d]) for d in depths), default=1)
    fig_width = max(28, int(max_n * 0.32))
    fig, axes = plt.subplots(nrows=len(depths), ncols=1,
                             figsize=(fig_width, max(8, 7 * len(depths))))
    # axes is a single Axes if nrows==1
    if len(depths) == 1:
        axes = [axes]

    for ax, depth in zip(axes, depths):
        sub_df = parts[depth]
        sub_title = f"{title_main}  -  ml_depth={depth}  (n={len(sub_df)} configs)"
        _build_bar_panel_for_depth(
            ax, sub_df, baseline_val,
            metric=metric, ylabel=ylabel, title=sub_title,
            higher_better=higher_better,
            depth_color=GROUP_COLORS.get(f"depth{depth}", "#888888"),
        )

    fig.suptitle(
        f"{title_main} (W+A, BF16)\n{_mode_subtitle(modes)}\n"
        f"x-axis: bits_per_elem DESC | dashed vertical lines mark integer bands of bits_per_elem",
        fontsize=13, fontweight="bold", y=1.005,
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Saved plot -> '{out_path}'")
    plt.close()


def _plot_qsnr_bar(df: pd.DataFrame, modes: dict, out_path: str):
    _plot_metric_by_depth(
        df, modes,
        metric="QSNR (dB)", ylabel="QSNR (dB) ↑",
        title_main="Multi-Level Mantissa Sharing - QSNR",
        higher_better=True, out_path=out_path,
    )


def _plot_ppl_bar(df: pd.DataFrame, modes: dict, out_path: str):
    _plot_metric_by_depth(
        df, modes,
        metric="Perplexity", ylabel="Perplexity ↓",
        title_main="Multi-Level Mantissa Sharing - Perplexity (WikiText-2)",
        higher_better=False, out_path=out_path,
    )


def _plot_qsnr_vs_ppl_scatter(df: pd.DataFrame, modes: dict, out_path: str):
    fig, ax = plt.subplots(figsize=(14, 9))
    plotted_groups = set()
    for _, row in df.iterrows():
        qsnr = row["QSNR (dB)"]
        ppl  = row["Perplexity"]
        if not (isinstance(qsnr, (int, float)) and math.isfinite(qsnr) and
                isinstance(ppl, (int, float)) and math.isfinite(ppl)):
            continue
        grp = row["Group"]
        color  = GROUP_COLORS.get(grp, "#888888")
        marker = GROUP_MARKER.get(grp, "o")
        ms = 220 if grp == "baseline" else 90
        label = GROUP_LABELS.get(grp, grp) if grp not in plotted_groups else None
        ax.scatter(qsnr, ppl, c=color, s=ms, marker=marker,
                   edgecolors="black", linewidths=0.5,
                   label=label, zorder=3)
        plotted_groups.add(grp)
        if grp == "baseline":
            ax.annotate("MXINT8 baseline",
                        xy=(qsnr, ppl),
                        xytext=(8, 8), textcoords="offset points",
                        fontsize=10, fontweight="bold")

    ax.legend(loc="upper left", fontsize=10, framealpha=0.9, title="ml_depth")
    ax.set_xlabel("QSNR (dB)  ↑ Higher is Better", fontsize=12)
    ax.set_ylabel("Perplexity  ↓ Lower is Better", fontsize=12)
    ax.set_title(
        f"Multi-Level Mantissa Sharing - QSNR vs Perplexity\n{_mode_subtitle(modes)}",
        fontsize=13, fontweight="bold", pad=18,
    )
    ax.grid(True, linestyle="--", alpha=0.35)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Saved plot -> '{out_path}'")
    plt.close()


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════
def run_experiment():
    # 0) Interactive mode selection
    modes = prompt_all_modes()
    measure_qsnr_flag = modes["metric_mode"] in ("both", "qsnr_only")
    measure_ppl_flag  = modes["metric_mode"] in ("both", "ppl_only")

    same_seeds(SEED)

    reset_skip_log()
    # 1) Model + tokenizer
    print(f"\nLoading model: {MODEL_NAME} ...")
    print("*** BF16-target + Weight & Activation quantization ***")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16, device_map=DEVICE
    )
    model.eval()

    # 2) Dataset
    print("Loading dataset (WikiText-2) ...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")

    if measure_qsnr_flag:
        qsnr_inputs = tokenizer(
            dataset["text"][:NUM_CALIB_SAMPLES],
            return_tensors="pt", padding=True,
            truncation=True, max_length=MAX_SEQ_LEN,
        ).to(DEVICE)
    else:
        qsnr_inputs = None

    if measure_ppl_flag:
        print("Tokenizing WikiText-2 test set for PPL ...")
        ppl_full_text = "\n\n".join(dataset["text"])
        ppl_encodings = tokenizer(ppl_full_text, return_tensors="pt")
        print(f"  Total tokens: {ppl_encodings.input_ids.size(1):,}  "
              f"(seq_len={PPL_SEQ_LEN}, stride={PPL_STRIDE})")
    else:
        ppl_encodings = None

    # 3) Build configs
    configs = get_experiment_configs(modes)
    print(f"\nTotal configurations: {len(configs)}")

    results = []
    print(f"\n=== Sweep: Multi-Level Mantissa Sharing ===")
    print(f"    {_mode_subtitle(modes)}\n")

    for cfg in configs:
        label = cfg["name"].replace("\n", " ")
        total_bits  = cfg["total_bits"]
        bpe         = cfg["bits_per_elem"]
        savings     = bits_mxint8() - total_bits
        # Reference baseline is MXINT8 (always twos_complement), regardless
        # of the SSNF encoding chosen for the sweep configs.
        savings_bpe = compute_bits_per_elem(elem_bitwidth=ELEM_BITWIDTH,
                                            block_size=BLOCK_SIZE,
                                            scale_bits=8,
                                            encoding='twos_complement') - bpe

        # ---- SDPA patch using the same quant config as Linear ----
        if cfg["group"] == "baseline":
            sdpa_args = dict(
                num_format='mxint8',
                block_size=BLOCK_SIZE,
                elem_bitwidth=ELEM_BITWIDTH,
                rounding_mode=modes['rounding_mode'],
            )
        else:
            ml_depth    = cfg["ml_depth"]
            ml_bitwidth = list(cfg["ml_bitwidth"])
            ml_mg       = list(cfg["ml_mg"])
            sdpa_args = dict(
                num_format='ssnf',
                block_size=BLOCK_SIZE,
                elem_bitwidth=ELEM_BITWIDTH,
                ml_bitwidth=ml_bitwidth,
                ml_mg=ml_mg,
                ml_sharingmode=[modes['sharing_mode']] * ml_depth,
                encoding=modes['encoding'],
                rounding_mode=modes['rounding_mode'],
            )

        # ---- QSNR ----
        qsnr_val = float("nan")
        if measure_qsnr_flag:
            print(f"> [QSNR] {label} ...", end=" ", flush=True)
            patch_sdpa_for_ssnf(quant_args_act=sdpa_args)
            try:
                qsnr_val = measure_qsnr(model, qsnr_inputs, cfg["fn"])
            finally:
                unpatch_sdpa()
            print(f"QSNR={qsnr_val:.2f} dB  "
                  f"bits_per_elem={bpe:.4f}  "
                  f"savings={savings}b ({savings_bpe:+.4f}/elem)")
            torch.cuda.empty_cache(); gc.collect()

        # ---- Perplexity ----
        ppl_val = float("nan")
        if measure_ppl_flag:
            print(f"  [PPL]  {label} ...")
            patch_sdpa_for_ssnf(quant_args_act=sdpa_args)
            try:
                try:
                    ppl_val = _measure_ppl_sliding_window(model, ppl_encodings, cfg["fn"])
                    print(f"         PPL={ppl_val:.4f}  "
                          f"bits_per_elem={bpe:.4f}  "
                          f"savings={savings}b ({savings_bpe:+.4f}/elem)")
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        print(f"         OOM - skipping.")
                        ppl_val = float("nan")
                    else:
                        raise
            finally:
                unpatch_sdpa()
            torch.cuda.empty_cache(); gc.collect()

        results.append({
            "Configuration":   cfg["name"],
            "Group":           cfg["group"],
            "ml_depth":        cfg["ml_depth"],
            "ml_bitwidth":     "" if cfg["ml_bitwidth"] is None
                                  else "(" + ",".join(str(b) for b in cfg["ml_bitwidth"]) + ")",
            "ml_mg":           "" if cfg["ml_mg"] is None
                                  else "(" + ",".join(str(g) for g in cfg["ml_mg"]) + ")",
            "Total Bits":      total_bits,
            "Bit Savings":     savings,
            "bits_per_elem":   bpe,
            "QSNR (dB)":       qsnr_val,
            "Perplexity":      ppl_val,
            # Mode annotation columns (same value on every row, but explicit)
            "Mode_sharing":    modes["sharing_mode"],
            "Mode_rounding":   modes["rounding_mode"],
            "Mode_encoding":   modes["encoding"],
            "Mode_metric":     modes["metric_mode"],
        })

    df = pd.DataFrame(results)

    # 4) Save outputs (filenames carry the mode tag)
    tag = _mode_tag(modes)
    csv_path = f"multi_level_mantissa_sharing__{tag}.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved -> '{csv_path}'")

    show_cols = ["Configuration", "ml_depth", "ml_bitwidth", "ml_mg",
                 "Total Bits", "Bit Savings", "bits_per_elem"]
    if measure_qsnr_flag: show_cols.append("QSNR (dB)")
    if measure_ppl_flag:  show_cols.append("Perplexity")
    print(df[show_cols].to_string(index=False))

    # 5) Plots (separately for QSNR and PPL; scatter only when both are present)
    if measure_qsnr_flag:
        _plot_qsnr_bar(df, modes,
                       out_path=f"multi_level_mantissa_sharing__{tag}__QSNR.png")
    if measure_ppl_flag:
        _plot_ppl_bar(df, modes,
                      out_path=f"multi_level_mantissa_sharing__{tag}__PPL.png")
    if measure_qsnr_flag and measure_ppl_flag:
        _plot_qsnr_vs_ppl_scatter(df, modes,
                                  out_path=f"multi_level_mantissa_sharing__{tag}__scatter.png")

    # Diagnostic: unconditional, independent of metric_mode / plot branches.
    dump_skip_report(header="multi_level_mantissa_sharing")


if __name__ == "__main__":
    run_experiment()