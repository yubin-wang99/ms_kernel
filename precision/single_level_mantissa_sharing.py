# single_level_mantissa_sharing.py
#
# Experiment: Single-level (ml_depth=1) Mantissa Sharing with additional
# error-correction mechanisms.
#
# This script assumes ml_depth=1 and sweeps:
#   ml1_bitwidth ∈ {1, 2, 3, 4}
#   ml1_mg       ∈ {2, 4, 8, 16, 32}
#
# 6 evaluation targets (imported from error_correction_mechanism.py and
# mantissa_bitwidth_reduction.py):
#   1. BF16 reference (PPL only; identity fn, no quantization)
#   2. quant_mxint8                              - MXINT8 baseline
#   3. single_level_mantissa_sharing             - SSNF LSB sharing, no MSB comp.
#   4. MSAQ_signed                               - FP-residual mean, signed u-bit
#                                                  (MXINT8 bit-level compatible
#                                                  via 2's complement)
#   5. single_rounding_bitwidth_reduction        - Scale-first bitwidth reduction
#                                                  (FP → reduced grid in one
#                                                  round trip)
#   6. double_rounding_bitwidth_reduction        - Slice-and-Scale MXINT
#                                                  (Xu+ '26; two rounds: FP →
#                                                  MXINT8 → reduced grid)
#
# Interactive prompts (Multi-Level style) for:
#   * quant_target      ∈ {weight_act (default), weight_only}
#                         weight_act  : Linear weight + Linear input
#                                        activation are quantized. SDPA
#                                        Q / K / V and the post-softmax
#                                        attention probability stay BF16
#                                        (SmoothQuant / LLM.int8()
#                                        convention: Linear inputs are
#                                        quantized; attention internals
#                                        are left untouched).
#                         weight_only : only Linear weights are quantized;
#                                        activations and SDPA Q/K/V/attn
#                                        stay BF16 throughout.
#   * sharing_mode      ∈ {mean (default), max, min, majority, round_mean, none}
#   * rounding_mode     ∈ {determ (default), stoc}
#   * encoding          ∈ {twos_complement (default), sign_magnitude}
#   * metric_mode       ∈ {both (default), qsnr_only, ppl_only}
#
# NOTE: Only mechanism #3 (single_level_mantissa_sharing) is SSNF-backed
# and consumes sharing_mode / rounding_mode / encoding -- those modes are
# forwarded into ssnf_quant. MXINT8, MSAQ-S, and the two bitwidth-reduction
# baselines (single_rounding / double_rounding) implement their own
# rounding / encoding logic directly and do NOT consume these modes;
# they are still recorded in CSV/plot metadata for reproducibility.
# The interactive prompts are kept identical to the Multi-Level script
# so the two experiments share the same control surface.
#
# Multiple models are evaluated sequentially. After each model is fully
# measured its weights/tokenizer are freed (del + gc + empty_cache) before
# the next model is loaded, to avoid accumulating GPU memory. Per-model
# CSV/plot files are written (model slug embedded in the filename).
#
# *** BF16-TARGET + (WEIGHT&ACT or WEIGHT-ONLY) QUANTIZATION VERSION ***

import gc
import math
import os
import re
import types
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from ssnf_main.ForGit.src.ssnf_core import SEED, same_seeds, is_quant_target_linear
from ssnf_main.ForGit.src.ssnf_sdpa_patch import (
    patch_sdpa_with_callable, patch_sdpa_kv_only, unpatch_sdpa,
    record_skip, reset_skip_log, dump_skip_report,
)
from ssnf_main.ForGit.src.error_correction_mechanism import (
    BLOCK_SIZE,
    quant_mxint8,
    single_level_mantissa_sharing,
    MSAQ_signed,
)
from ssnf_main.ForGit.src.mantissa_bitwidth_reduction import (
    single_rounding_bitwidth_reduction,
    double_rounding_bitwidth_reduction,
)


# ─────────────────────────────────────────────────────────────────────────────
# Global config
# ─────────────────────────────────────────────────────────────────────────────
# Models evaluated sequentially (load -> measure -> free -> next).
# Order: Qwen first — Qwen2.5 exhibits the largest KV-quantization
# sensitivity (RoPE base = 1M, QKV bias, 7:1 GQA), so running it first
# surfaces the most diagnostic results early in a sweep.
MODEL_NAMES = [
    # "Qwen/Qwen2.5-7B-Instruct",
    "meta-llama/Llama-3.1-8B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.3",
    "google/gemma-2-9b-it",
]
DEVICE            = "cuda" if torch.cuda.is_available() else "cpu"
NUM_CALIB_SAMPLES = 16
MAX_SEQ_LEN       = 128
MXINT8_TOTAL_BITS = 264

# Sweep ranges. ml1_mg is common to every scope; the ml1_bitwidth band is
# per quantization scope, because each scope has a different accuracy-robust
# LSB-sharing depth (weight-only tolerates the most sharing, KV cache the
# least). Selected at config-build time via `sweep_for(quant_target)`.
ML1_MG_VALUES = [2, 4, 8, 16, 32]
ML1_BITWIDTH_BY_SCOPE = {
    "weight_only":   [2, 3, 4],
    "weight_act":    [2, 3],
    "kv_only":       [3, 4, 5],
    "full_act":      [2, 3],
}


def sweep_for(quant_target: str):
    """(ml1_bitwidth_values, ml1_mg_values) for a quantization scope."""
    return ML1_BITWIDTH_BY_SCOPE[quant_target], ML1_MG_VALUES


# Mechanism toggle: the 1Level (single_level_mantissa_sharing) sweep is OFF
# by default — only MSAQ-S is swept among the mantissa-sharing variants. The
# config-construction code below is retained, so re-enabling needs no file
# edit: launch with `INCLUDE_1LEVEL=1 python -m ...` to add the 1Level rows
# back alongside MSAQ-S.
INCLUDE_1LEVEL = os.environ.get("INCLUDE_1LEVEL", "0") == "1"


# TEMP per-(model, scope) sweep override. When a (model-substring, scope) key
# matches the run, it REPLACES that scope's default ml1_bitwidth / ml1_mg from
# ML1_BITWIDTH_BY_SCOPE for that model only. Empty this dict (or delete the
# entry) to restore the default scope sweep everywhere.
SWEEP_OVERRIDE = {
    ("Mistral", "weight_act"): ([4], [2, 4, 8, 16, 32]),
}


def effective_sweep(quant_target: str, model_name: str, override=None):
    """Scope sweep with optional overrides.

    Priority: explicit `override` (per-preset / per-run, e.g. CURATED_SWEEP_OVERRIDE)
    > per-model SWEEP_OVERRIDE > scope default `sweep_for(quant_target)`.
    Returns (ml1_bitwidth_values, ml1_mg_values).
    """
    if override is not None:
        return override
    for (model_key, scope), (bw, mg) in SWEEP_OVERRIDE.items():
        if scope == quant_target and model_key.lower() in model_name.lower():
            return bw, mg
    return sweep_for(quant_target)

# Perplexity: prefill/decode workload, sliding-window over WikiText-2 test set.
# Mirrors the decode-dominant request in msaq_footprint_traffic.py
# (P_IN=800 prefill, G_OUT=3880 decode). Each window feeds PREFILL+DECODE
# tokens through the (quantized) model and scores ONLY the last DECODE
# positions; the first PREFILL positions are loss-masked context whose K/V
# are still quantized and attended to. Causal masking makes the per-token
# NLL identical whether positions are fed all-at-once (prefill/teacher-
# forcing) or one-at-a-time (decode), so this faithfully measures PPL under
# the prefill=800 / decode=3880 regime; the split only selects which
# positions contribute to the loss.
PPL_PREFILL   = 800
PPL_DECODE    = 3880
PPL_SEQ_LEN   = PPL_PREFILL + PPL_DECODE   # 4680-token window
PPL_STRIDE    = PPL_DECODE                 # 3880 scored positions per window
PPL_MAX_STEPS = 64

# Memory guard for the PPL loss. At the 4680-token window a full-vocab logits
# tensor [1, 4680, V] plus its fp32 upcast inside CrossEntropyLoss overflows
# 24 GB for large-vocab models (Gemma-2: V=256k -> ~7 GB transient on top of
# the ~18.5 GB weights -> OOM). For vocab >= PPL_CHUNK_VOCAB we compute the LM
# loss in sequence chunks via lm_head + cross-entropy, never materializing the
# full-sequence logits; this is mathematically identical to the HF loss (same
# per-token log-softmax, same final-logit soft-capping). Llama-3.1 (128k) /
# Mistral (32k) stay on the standard HF path unchanged.
PPL_CHUNK_VOCAB  = 200_000   # Gemma-2 (256k) -> chunked; Llama/Mistral -> standard
PPL_LOGITS_CHUNK = 512       # sequence positions per lm_head/CE chunk

# Optional per-model-type PPL decode cap (escape hatch). With native SDPA
# (O(L) memory, no materialized [L, L] score matrix) plus the chunked LM loss
# and use_cache=False, every model — including Gemma-2-9B — fits the full
# PPL_DECODE window on a 24 GB card, so no cap is applied by default. If a
# specific model still OOMs on your GPU, add an entry here to shrink its
# decode, e.g. {"gemma2": 1920}; otherwise the global PPL_DECODE is used.
PPL_DECODE_BY_TYPE = {}

# PPL plot y-axis cap. Bars whose PPL exceeds this value are clipped to
# the cap so the BF16 / 0.97 quality-threshold line remains legible; the
# actual (un-clipped) PPL is still printed in the bar annotation.
PPL_PLOT_YMAX = 20.0

# KV cache quantization block axis.
#   "mx_standard" : OCP MX default — ch_axis = -1 for both K and V.
#                   Blocks of BLOCK_SIZE contiguous elements run along
#                   the head_dim (D) axis; each token's D is split into
#                   D // BLOCK_SIZE groups, per-token scales. This is
#                   the dot-product reduction axis of Q·Kᵀ, so it aligns
#                   with hardware-native MX GEMM dataflow (NVIDIA
#                   Blackwell mma.mNkK, AMD Quark OCP_MXINT8Spec default).
#   "kivi"        : KIVI-style per-channel grouping for K AND V — blocks
#                   of BLOCK_SIZE tokens at a fixed channel index. RoPE's
#                   channel-wise outlier pattern is preserved across the
#                   block, mitigating long-RoPE quantization failures
#                   observed on Qwen2.5. Simplified variant where K and
#                   V share the same axis; useful for axis-ablation
#                   diagnostics. NOT hardware-native MX.
#   "kivi_split"  : KIVI-paper-faithful — K on sequence axis (per-channel),
#                   V on head_dim axis (per-token). KIVI (Liu et al.,
#                   ICML 2024) shows K and V have different outlier
#                   structures: K outliers are channel-wise (consistent
#                   across tokens at the same channel), V outliers are
#                   token-wise (some tokens have disproportionate norm
#                   regardless of channel). This variant matches the
#                   paper's recommendation.
#
# Reads from the KV_AXIS env var so the axis can be toggled at the
# command line without editing this file, e.g.:
#     KV_AXIS=kivi_split CUDA_VISIBLE_DEVICES=1 python -m ssnf_main...sharing
# Unset (or any unknown value) falls back to "mx_standard".
KV_AXIS = os.environ.get("KV_AXIS", "mx_standard").lower()
if KV_AXIS not in ("mx_standard", "kivi", "kivi_split"):
    print(f"[warn] Unknown KV_AXIS={KV_AXIS!r}; falling back to 'mx_standard'.")
    KV_AXIS = "mx_standard"

# Generation-style PPL — used for KV-cache quantization modes
# (kv_only, weight_act_kv) where the quantization effect propagates through
# autoregressive decode steps as the cache grows. For each generation
# sample we prefill PPL_GEN_PROMPT_LEN tokens, then autoregressively decode
# PPL_GEN_LENGTH steps using the ground-truth next token as input
# (teacher-forcing-of-generation). At every decode step the entire KV
# cache passes through the SDPA quantization patch, so the quantization
# error accumulates exactly as it would during real inference. Total
# measured tokens are capped at PPL_GEN_MAX_TOKENS to keep wall-time
# tractable (generation is ~10–20× slower than sliding-window PPL).
PPL_GEN_PROMPT_LEN = 512
PPL_GEN_LENGTH     = 128
PPL_GEN_STRIDE     = PPL_GEN_PROMPT_LEN + PPL_GEN_LENGTH   # no overlap
PPL_GEN_MAX_TOKENS = 1024   # = 8 samples at PPL_GEN_LENGTH=128


def _model_slug(model_name: str) -> str:
    """Filesystem-safe short tag for a HF model id (e.g.
    'meta-llama/Llama-3.1-8B-Instruct' -> 'Llama-3.1-8B-Instruct')."""
    base = model_name.split("/")[-1]
    return re.sub(r"[^0-9A-Za-z._-]+", "-", base)


# ─────────────────────────────────────────────────────────────────────────────
# Interactive mode selection (mirrors multi_level_mantissa_sharing.py)
# ─────────────────────────────────────────────────────────────────────────────
SHARING_MODE_OPTIONS  = ["mean", "max", "min", "majority", "round_mean"]
ROUNDING_MODE_OPTIONS = ["determ", "stoc"]
ENCODING_OPTIONS      = ["twos_complement", "sign_magnitude"]
METRIC_MODE_OPTIONS   = ["both", "qsnr_only", "ppl_only"]
# Quantization scope:
#   weight_act    : Linear weight + Linear input activation are quantized.
#                   SDPA Q / K / V and the post-softmax attention probability
#                   stay BF16 (SmoothQuant / LLM.int8() convention: attention
#                   internals are left untouched).
#   weight_only   : Linear weight only — activations and SDPA Q/K/V/attn stay BF16.
#   weight_act_kv : weight_act + KV cache quantization. SDPA's K and V tensors
#                   (RoPE-applied, head-reshaped — i.e. the tensors that would
#                   be stored in the past_key_values cache during generation)
#                   are quantized via `patch_sdpa_kv_only`. SDPA Q and the
#                   post-softmax attention probability remain BF16.
#   kv_only       : KV cache only. Linear weights and input activations stay
#                   BF16; only SDPA's K and V are quantized. Useful for
#                   isolating the KV-quantization quality impact. QSNR is
#                   not meaningful in this mode (Linear outputs are unchanged
#                   end-to-end), so it is skipped automatically.
#   full_act      : weight_act + full attention-internal quantization. On top
#                   of Linear weight + input, SDPA's Q, K, V AND the
#                   post-softmax attention probability are all quantized via
#                   `patch_sdpa_with_callable` (the unfused SDPA path). This is
#                   the ONLY scope that quantizes the activation×activation
#                   matmuls (Q·Kᵀ and attn·V); every other scope leaves
#                   attention internals in BF16. Always uses the MX head_dim
#                   axis for all four tensors (KV_AXIS is ignored).
QUANT_TARGET_OPTIONS  = ["weight_act", "weight_only", "kv_only", "full_act"]

# Curated presets selectable at the quant_target prompt, in addition to the
# real scopes and "all". Each preset lists (model-substring, [scopes]) pairs
# and runs ONLY those (model, scope) combinations — handy for re-measuring a
# specific subset. Model substrings are matched against MODEL_NAMES at runtime;
# entries with no match (e.g. a commented-out model) are skipped with a notice.
CURATED_SCOPE_SETS = {
    # scope1: re-measure Gemma across all scopes + Mistral's weight_act only
    # (the latter carries the temporary ml1_bitwidth=4 SWEEP_OVERRIDE).
    "scope1": [
        ("gemma",   ["weight_only", "weight_act", "kv_only"]),
        ("Mistral", ["weight_act"]),
    ],
    # scope2: every model's weight_act only, at ml1_bitwidth=4 (see
    # CURATED_SWEEP_OVERRIDE below).
    "scope2": [
        ("Llama",   ["weight_act"]),
        ("Mistral", ["weight_act"]),
        ("gemma",   ["weight_act"]),
    ],
}

# Optional uniform sweep override applied to every (model, scope) run of a
# whole curated preset. Takes precedence over ML1_BITWIDTH_BY_SCOPE and the
# per-model SWEEP_OVERRIDE for that preset's runs. Outputs from an overridden
# preset get a bit-width marker in the filename (see _mode_tag) so they never
# overwrite the default-sweep CSV/PNG of the same scope.
CURATED_SWEEP_OVERRIDE = {
    "scope2": ([4], [2, 4, 8, 16, 32]),
}

# `all` runs every real scope across every model; a curated preset runs only
# its listed pairs. Each run writes its OWN per-(model,scope) CSV/PNG files
# (scope is embedded in _mode_tag, so outputs never collide). Both are
# expanded into explicit (model, scope) pairs in run_experiment.
QUANT_TARGET_PROMPT_OPTIONS = QUANT_TARGET_OPTIONS + ["all"] + list(CURATED_SCOPE_SETS)


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
    """Run the interactive prompts and return a dict of mode choices."""
    print("=" * 70)
    print("Single-Level Mantissa Sharing Experiment - Mode Selection")
    print("=" * 70)
    return {
        "quant_target":  _prompt_mode("quant_target",  QUANT_TARGET_PROMPT_OPTIONS,  "weight_act"),
        "sharing_mode":  _prompt_mode("sharing_mode",  SHARING_MODE_OPTIONS,  "mean"),
        "rounding_mode": _prompt_mode("rounding_mode", ROUNDING_MODE_OPTIONS, "determ"),
        "encoding":      _prompt_mode("encoding",      ENCODING_OPTIONS,      "twos_complement"),
        "metric_mode":   _prompt_mode("metric_mode",   METRIC_MODE_OPTIONS,   "both"),
    }


def _mode_tag(modes: dict) -> str:
    """
    Short tag for filenames. Values only (no prefix), underscore-separated:
        {quant_target}_{rounding_mode}_{encoding}_{sharing_mode}[_{kv_axis}]
    Underscores inside individual values (e.g. 'sign_magnitude',
    'round_mean', 'weight_act', 'weight_only') are rewritten as hyphens
    so the tag uses '_' purely as a field separator.
    The `kv_axis` suffix is appended only for KV-cache quantization modes
    (kv_only, weight_act_kv) so MX-axis and KIVI-axis results don't
    silently overwrite each other.
    Note: metric_mode is intentionally excluded — plot suffixes (__QSNR /
    __PPL / __scatter) carry that information instead.
    """
    qt    = modes["quant_target"].replace("_", "-")
    enc   = modes["encoding"].replace("_", "-")
    share = modes["sharing_mode"].replace("_", "-")
    base  = f"{qt}_{modes['rounding_mode']}_{enc}_{share}"
    if modes["quant_target"] in ("kv_only", "weight_act_kv"):
        base = f"{base}_axis-{KV_AXIS.replace('_', '-')}"
    # An explicit sweep override (e.g. a curated preset like scope2) shares the
    # quant_target of a normal run, so mark the bit-width band in the filename
    # to avoid overwriting the default-sweep CSV/PNG of the same scope.
    ov = modes.get("_sweep_override")
    if ov is not None:
        base = f"{base}_bw" + "-".join(str(b) for b in ov[0])
    return base


def _mode_subtitle(modes: dict) -> str:
    """Human-readable mode summary for plot subtitles."""
    base = (f"quant_target={modes['quant_target']} | "
            f"sharing_mode={modes['sharing_mode']} | "
            f"rounding_mode={modes['rounding_mode']} | "
            f"encoding={modes['encoding']} | "
            f"metric_mode={modes['metric_mode']}")
    if modes["quant_target"] in ("kv_only", "weight_act_kv"):
        base += f" | kv_axis={KV_AXIS}"
    return base


def _scope_short_tag(quant_target: str) -> str:
    """Compact scope tag for plot titles and stdout banners."""
    return {
        "weight_only":   "W-only",
        "weight_act":    "W+A",
        "weight_act_kv": "W+A+KV",
        "kv_only":       "KV-only",
        "full_act":      "W+A+Attn",
    }[quant_target]


def _sweep_range_tag(quant_target: str, model_name: str, override=None) -> str:
    """Render the current scope's sweep ranges for plot titles, e.g.
    'ml1_bitwidth ∈ {3,4,5}, ml1_mg ∈ {2,4,8,16,32}'."""
    bw_values, mg_values = effective_sweep(quant_target, model_name, override)
    bw = ",".join(str(v) for v in bw_values)
    mg = ",".join(str(v) for v in mg_values)
    return f"ml1_bitwidth ∈ {{{bw}}}, ml1_mg ∈ {{{mg}}}"


# ─────────────────────────────────────────────────────────────────────────────
# Activation quantization helper
# ─────────────────────────────────────────────────────────────────────────────
def _maybe_quantize_activation(x: torch.Tensor, quant_fn,
                               name=None) -> torch.Tensor:
    """Quantize the block-aligned head of the last dim; pass the tail through.

    Tail-quantization (unified with single_level_vision.py). The last dim is
    split into a block-aligned head `x[..., :k]`
    (k = floor(last_dim/BLOCK_SIZE) * BLOCK_SIZE) which IS quantized, and a
    remainder tail `x[..., k:]` of length `last_dim % BLOCK_SIZE` which is
    passed through unchanged. Slicing to a multiple of BLOCK_SIZE keeps the
    `block_tensor` flatten aligned with row boundaries, so the head is
    quantized without cross-row block contamination; only the short tail
    escapes quantization.

    - last_dim a multiple of BLOCK_SIZE  -> whole tensor quantized (k =
      last_dim, no tail).
    - last_dim < BLOCK_SIZE              -> k = 0, nothing to quantize; the
      entire tensor is the tail and is passed through.
    - otherwise                          -> head quantized, tail passed
      through.

    Only the tail (the part that escapes quantization) is recorded via
    `record_skip` under the single site tag "tail" so dump_skip_report
    surfaces exactly which sites / shapes were measured at higher precision
    than nominal.
    """
    last = x.shape[-1]
    k = (last // BLOCK_SIZE) * BLOCK_SIZE

    if k == last:
        return quant_fn(x)

    if k == 0:
        record_skip("tail", name, x.shape)
        return x

    head = x[..., :k]
    tail = x[..., k:]
    record_skip("tail", name, tail.shape)
    head_q = quant_fn(head)
    if head_q.dtype != x.dtype:
        head_q = head_q.to(x.dtype)
    return torch.cat([head_q, tail], dim=-1)


def _make_sdpa_quant_fn(fn):
    """
    Wrap a config's quant_fn for use inside the SDPA patch.

    The SDPA path quantizes Q/K/V/attn tensors of shape (B, H, L, S/D).
    The ECM primitives reshape the last dim into blocks of BLOCK_SIZE,
    so we apply the same tail-quantization as the Linear-side
    `_maybe_quantize_activation`: the block-aligned head of the last dim
    is quantized and only the `last_dim % BLOCK_SIZE` tail passes through
    unchanged (recorded under the single site tag "tail").
    """
    def _wrapped(x, site="sdpa:?"):
        last = x.shape[-1]
        k = (last // BLOCK_SIZE) * BLOCK_SIZE

        if k == last:
            return fn(x)

        if k == 0:
            record_skip("tail", None, x.shape)
            return x

        head = x[..., :k]
        tail = x[..., k:]
        record_skip("tail", None, tail.shape)
        head_q = fn(head)
        if head_q.dtype != x.dtype:
            head_q = head_q.to(x.dtype)
        return torch.cat([head_q, tail], dim=-1)
    return _wrapped


def _make_sdpa_quant_fn_kivi(fn):
    """
    KIVI-axis variant of `_make_sdpa_quant_fn`.

    Input shape on the SDPA path is (B, H, L, D). The MX-axis wrapper
    quantizes contiguous blocks along D (the last dim). The KIVI variant
    transposes the tensor so that L becomes the last dim — the ECM `fn`
    then sees blocks of BLOCK_SIZE along the sequence (token) axis, i.e.
    per-channel grouping in KIVI terminology (a fixed channel index, a
    contiguous window of tokens share one scale). After quantization the
    tensor is transposed back so SDPA sees the original layout.

    Tail handling mirrors the MX-axis wrapper, but now operates on the
    sequence axis: if L is not a multiple of BLOCK_SIZE, the last
    `L % BLOCK_SIZE` tokens pass through unchanged and the skip is
    logged. This is a simplification compared to KIVI's streaming
    approach (which fills groups as the cache grows); for a single
    snapshot of K, V at SDPA entry the two are functionally equivalent
    on the block-aligned head.

    NOTE: K and V are both routed through this wrapper. KIVI's original
    paper recommends K per-channel + V per-token, but here we use a
    single KIVI axis for both to keep the diagnostic clean — the
    question being answered is "does the channel-axis grouping alone
    recover Qwen?", not "what is the best KIVI-faithful recipe?".
    """
    def _wrapped(x, site="sdpa:?"):
        # (B, H, L, D) -> (B, H, D, L) so L is contiguous in the last dim.
        x_swap = x.transpose(-2, -1)
        L_axis = x_swap.shape[-1]
        k = (L_axis // BLOCK_SIZE) * BLOCK_SIZE

        if k == L_axis:
            y_swap = fn(x_swap)
        elif k == 0:
            # Sequence too short for a single full block — skip entirely.
            record_skip("tail", None, x.shape)
            return x
        else:
            head = x_swap[..., :k]                 # (B, H, D, k) — block-aligned tokens
            tail = x_swap[..., k:]                 # (B, H, D, L - k) — trailing tokens
            record_skip("tail", None, tail.shape)
            head_q = fn(head)
            if head_q.dtype != x.dtype:
                head_q = head_q.to(x.dtype)
            y_swap = torch.cat([head_q, tail], dim=-1)

        # Swap back to (B, H, L, D). .contiguous() ensures downstream
        # SDPA receives a memory-contiguous tensor (transpose returns a
        # view with non-contiguous strides).
        return y_swap.transpose(-2, -1).contiguous()
    return _wrapped


def _make_sdpa_quant_fn_kivi_split(fn):
    """
    KIVI-paper-faithful variant: K is quantized along the sequence axis
    (per-channel) while V is quantized along the head_dim axis
    (per-token).

    KIVI (Liu et al., ICML 2024) shows that K and V exhibit different
    outlier distributions:
      • K has channel-wise outliers that stay consistent across tokens
        after RoPE → per-channel (sequence-axis) grouping preserves the
        channel statistics → block scale is dominated by the channel's
        outlier magnitude, not by RoPE-induced mixing of high/low
        frequencies within a single token.
      • V has token-wise outliers (some tokens carry disproportionate
        value norms regardless of channel) → per-token (head_dim-axis)
        grouping isolates those outlier tokens to their own block scale.

    This wrapper dispatches by `site`: K goes through the KIVI-axis
    wrapper (sequence-axis grouping), V goes through the MX-axis wrapper
    (head_dim-axis grouping). Implementation reuses the two existing
    wrappers as inner functions — no logic duplication.

    For non-KV sites ("sdpa:Q", "sdpa:attn") the MX-axis path is used as
    a safe default; the kv_only / weight_act_kv pipelines only invoke
    "sdpa:K" and "sdpa:V" anyway, so the other branches are unreachable
    in normal use.
    """
    mx_wrapper   = _make_sdpa_quant_fn(fn)        # for V (head_dim axis)
    kivi_wrapper = _make_sdpa_quant_fn_kivi(fn)   # for K (sequence axis)

    def _wrapped(x, site="sdpa:?"):
        if site == "sdpa:K":
            return kivi_wrapper(x, site)
        return mx_wrapper(x, site)
    return _wrapped


# ─────────────────────────────────────────────────────────────────────────────
# Bit-budget helpers
# ─────────────────────────────────────────────────────────────────────────────
def bits_mxint8() -> int:
    return 8 + 8 * BLOCK_SIZE
def bits_grouped_sharing(ml1_bitwidth: int, ml1_mg: int,
                         encoding: str = 'twos_complement') -> int:
    """
    Bits-per-block for a single-level mantissa-shared MXINT8.

    Mirrors multi_level_mantissa_sharing.bits_multi_level_sharing for the
    one-level (ml_depth=1) case so the two experiments report bit costs
    on the same basis.

    With encoding='twos_complement' all 8 bits participate in the
    partition. With encoding='sign_magnitude' only the 7 magnitude bits
    are partitioned and the sign is stored separately (one extra bit per
    element); the bottom `ml1_bitwidth` bits above `partition_bits` (none
    in the current sweep, since ml1_bitwidth <= 4 < 7) would be virtual
    upper bits at zero cost.

    NOTE: only single_level_mantissa_sharing (SSNF-backed) consumes
    `encoding`; MSAQ-S uses its own bit-budget formula
    (`bits_mantissa_share`) which is independent of encoding choice. The
    bitwidth-reduction baselines (single_rounding / double_rounding) strip
    `reduced_bitwidth` LSB positions of the MXINT8 mantissa entirely, so
    their per-block bit cost is `bits_mxint8() - reduced_bitwidth * BLOCK_SIZE`
    and is computed inline at the call site.
    """
    if encoding == 'twos_complement':
        partition_bits  = 8
        sign_bits_block = 0
    elif encoding == 'sign_magnitude':
        partition_bits  = 7              # 8-bit elem, MSB is the sign
        sign_bits_block = BLOCK_SIZE     # 1 sign bit per element
    else:
        raise ValueError(f"unknown encoding: {encoding!r}")

    effective_ml  = min(ml1_bitwidth, partition_bits)
    unshared_bits = partition_bits - effective_ml

    bits = 8 + unshared_bits * BLOCK_SIZE + sign_bits_block
    bits += effective_ml * math.ceil(BLOCK_SIZE / ml1_mg)
    return bits
def bits_mantissa_share(ml1_bitwidth: int, ml1_mg: int) -> int:
    """
    Bits-per-block for MSAQ (signed or unsigned). Both variants store the
    same number of bits per shared residual; only the encoding of those
    bits (sign-magnitude offset vs two's-complement) differs.
    """
    return 8 + (8 - ml1_bitwidth) * BLOCK_SIZE + ml1_bitwidth * math.ceil(BLOCK_SIZE / ml1_mg)


# ─────────────────────────────────────────────────────────────────────────────
# Experiment configurations
# ─────────────────────────────────────────────────────────────────────────────
def get_experiment_configs(modes: dict, model_name: str):
    configs = []
    # 0. BF16 reference (no quantization). The model is loaded in bfloat16,
    #    so the identity fn leaves it at native BF16. Its PPL is the
    #    pure-BF16 PPL and becomes `baseline_ppl` (the first finite PPL),
    #    which the PPL plot uses to draw the BF16 / 0.97 quality threshold.
    #    QSNR is NOT measured for this row (PPL-only); it is also excluded
    #    from the QSNR bar and the QSNR-vs-PPL scatter.
    configs.append({
        "name":       "BF16\n(no quant)\n(ref)",
        "group":      "bf16_ref",
        "total_bits": MXINT8_TOTAL_BITS,   # bit_savings=0 → plot/Pareto safe
        "fn":         (lambda x: x),       # identity = no quantization
    })
    # 1. MXINT8 baseline
    configs.append({
        "name":       "MXINT8\n(baseline)\n(264b)",
        "group":      "baseline",
        "total_bits": bits_mxint8(),
        "fn":         quant_mxint8,
    })

    # 2-3. Mantissa-sharing mechanisms (sweep over ml1_bitwidth × ml1_mg)
    # NOTE: only single_level_mantissa_sharing is SSNF-backed and therefore
    # consumes sharing_mode / rounding_mode / encoding. MSAQ_signed
    # implements its own rounding/encoding logic directly and is not
    # parameterized by these modes.
    sm = modes['sharing_mode']
    rm = modes['rounding_mode']
    en = modes['encoding']
    bw_values, mg_values = effective_sweep(modes["quant_target"], model_name, modes.get("_sweep_override"))
    for ml1_bitwidth in bw_values:
        for ml1_mg in mg_values:
            # 1Level (SSNF-backed) is retained but OFF by default; only
            # appended when INCLUDE_1LEVEL is set. It honours the chosen
            # encoding for bit-budget accounting (MSAQ-S uses its own formula).
            if INCLUDE_1LEVEL:
                tb_ssnf = bits_grouped_sharing(ml1_bitwidth, ml1_mg, encoding=en)
                configs.append({
                    "name":       f"1Level\nml1_bitwidth={ml1_bitwidth},ml1_mg={ml1_mg}\n({tb_ssnf}b)",
                    "group":      "single_level_mantissa_sharing",
                    "total_bits": tb_ssnf,
                    "fn":         (lambda _b, _g, _sm, _rm, _en:
                                    lambda x: single_level_mantissa_sharing(
                                        x, _b, _g,
                                        sharing_mode=_sm,
                                        rounding_mode=_rm,
                                        encoding=_en,
                                    ))(ml1_bitwidth, ml1_mg, sm, rm, en),
                })

            configs.append({
                "name":       f"MSAQ-S\nml1_bitwidth={ml1_bitwidth},ml1_mg={ml1_mg}\n({bits_mantissa_share(ml1_bitwidth, ml1_mg)}b)",
                "group":      "MSAQ_signed",
                "total_bits": bits_mantissa_share(ml1_bitwidth, ml1_mg),
                "fn":         (lambda _b, _g: lambda x: MSAQ_signed(x, _b, _g))(ml1_bitwidth, ml1_mg),
            })

    # 4-5. Bitwidth-reduction baselines (sweep over reduced_bitwidth only —
    # these mechanisms have no ml1_mg dimension, so they appear once per
    # bitwidth value rather than once per (bitwidth, mg) cell). Per-block
    # cost equals MXINT8 with `reduced_bitwidth` LSB positions stripped
    # entirely, i.e. bits_mxint8() - reduced_bitwidth * BLOCK_SIZE.
    for reduced_bitwidth in bw_values:
        tb_br = bits_mxint8() - reduced_bitwidth * BLOCK_SIZE

        configs.append({
            "name":       f"SR-BR\nreduced_bitwidth={reduced_bitwidth}\n({tb_br}b)",
            "group":      "single_rounding_bitwidth_reduction",
            "total_bits": tb_br,
            "fn":         (lambda _b: lambda x: single_rounding_bitwidth_reduction(x, _b))(reduced_bitwidth),
        })

        configs.append({
            "name":       f"DR-BR\nreduced_bitwidth={reduced_bitwidth}\n({tb_br}b)",
            "group":      "double_rounding_bitwidth_reduction",
            "total_bits": tb_br,
            "fn":         (lambda _b: lambda x: double_rounding_bitwidth_reduction(x, _b))(reduced_bitwidth),
        })

    return configs


# ─────────────────────────────────────────────────────────────────────────────
# QSNR measurement  (BF16-target; scope controlled by `quant_target`)
# ─────────────────────────────────────────────────────────────────────────────
def measure_qsnr(model, inputs, quant_fn, quant_target: str = "weight_act") -> float:
    """
    QSNR over per-layer Linear outputs vs the BF16 reference.

    quant_target == 'weight_act' / 'weight_act_kv'
                                : both `x` and `w` are quantized.
    quant_target == 'weight_only': `x` stays BF16, only `w` is quantized.
    quant_target == 'kv_only'   : neither `x` nor `w` is quantized — this
                                  function should not be called in that mode
                                  (the caller skips it).
    """
    signal = 0.0
    noise  = 0.0
    quantize_act = quant_target in ("weight_act", "weight_act_kv", "full_act")
    def make_hook(fn, layer_name):
        def hook_fn(mod, inp, out):
            nonlocal signal, noise
            x_bf = inp[0].detach()
            w_bf = mod.weight.detach()
            bias = mod.bias.detach() if mod.bias is not None else None

            y_ref = F.linear(x_bf, w_bf, bias).to(torch.float32)

            if quantize_act:
                x_q = _maybe_quantize_activation(x_bf, fn, name=layer_name)
            else:
                x_q = x_bf
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


# ─────────────────────────────────────────────────────────────────────────────
# Perplexity measurement  (sliding-window, BF16-target; scope by `quant_target`)
# ─────────────────────────────────────────────────────────────────────────────
def _patch_linear_forward(model: nn.Module, quant_fn,
                          quant_target: str = "weight_act") -> dict:
    """
    Monkey-patch every quant-target Linear's forward so that:
      quant_target == 'weight_act' / 'weight_act_kv'
                                    : both `input` and `self.weight` are quantized.
      quant_target == 'weight_only' : `input` passes through unchanged; only
                                       `self.weight` is quantized.

    Not called for `quant_target == 'kv_only'` — see `_measure_ppl_sliding_window`.
    """
    original_forwards = {}
    quantize_act = quant_target in ("weight_act", "weight_act_kv", "full_act")

    def _make_quantized_forward(fn, layer_name):
        if quantize_act:
            def quantized_forward(self, input):
                x_q = _maybe_quantize_activation(input, fn, name=layer_name)
                w_q = fn(self.weight)
                if w_q.dtype != self.weight.dtype:
                    w_q = w_q.to(self.weight.dtype)
                out = F.linear(x_q, w_q, self.bias)
                del w_q, x_q
                return out
        else:
            def quantized_forward(self, input):
                w_q = fn(self.weight)
                if w_q.dtype != self.weight.dtype:
                    w_q = w_q.to(self.weight.dtype)
                out = F.linear(input, w_q, self.bias)
                del w_q
                return out
        return quantized_forward

    for name, mod in model.named_modules():
        if not is_quant_target_linear(name, mod):
            continue
        original_forwards[name] = mod.forward
        mod.forward = types.MethodType(
            _make_quantized_forward(quant_fn, name), mod)

    return original_forwards


def _unpatch_linear_forward(model: nn.Module, original_forwards: dict) -> None:
    for name, mod in model.named_modules():
        if name in original_forwards:
            mod.forward = original_forwards[name]


def _ppl_window(model):
    """Per-model PPL window: (prefill, decode, seq_len, stride).

    decode is capped per model_type via PPL_DECODE_BY_TYPE (e.g. Gemma-2),
    otherwise the global PPL_DECODE is used. prefill stays PPL_PREFILL.
    """
    mtype  = getattr(model.config, "model_type", "")
    decode = PPL_DECODE_BY_TYPE.get(mtype, PPL_DECODE)
    prefill = PPL_PREFILL
    return prefill, decode, prefill + decode, decode


def _window_sum_nll(model, input_ids, target_ids, trg_len, use_chunked):
    """Summed NLL over the scored (label != -100) positions of one window.

    Returns a CPU scalar tensor equal to (mean CE over scored tokens) * trg_len,
    i.e. the same quantity the old ``outputs.loss * trg_len`` produced.

    use_chunked=False : standard HF path (model(..., labels=...).loss).
    use_chunked=True  : run the base transformer to get hidden states, then
                        apply lm_head + cross-entropy in PPL_LOGITS_CHUNK-sized
                        sequence chunks so the full-vocab logits are never
                        materialized at once. Replicates HF exactly: causal
                        shift (predict token t+1 from position t), optional
                        final-logit soft-capping, ignore_index=-100.
    """
    if not use_chunked:
        outputs = model(input_ids, labels=target_ids)
        return outputs.loss.detach().cpu() * trg_len

    # Hidden states only — never build the full [1, L, V] logits tensor.
    hidden = model.model(input_ids=input_ids, use_cache=False).last_hidden_state   # [1, L, H]
    shift_hidden = hidden[:, :-1, :]
    shift_labels = target_ids[:, 1:]
    softcap = getattr(model.config, "final_logit_softcapping", None)

    total = torch.zeros((), dtype=torch.float32, device=hidden.device)
    L = shift_hidden.size(1)
    for c0 in range(0, L, PPL_LOGITS_CHUNK):
        h   = shift_hidden[:, c0:c0 + PPL_LOGITS_CHUNK, :]
        lab = shift_labels[:, c0:c0 + PPL_LOGITS_CHUNK]
        logits = model.lm_head(h)
        if softcap is not None:
            logits = torch.tanh(logits / softcap) * softcap
        total = total + F.cross_entropy(
            logits.float().reshape(-1, logits.size(-1)),
            lab.reshape(-1),
            ignore_index=-100,
            reduction="sum",
        )
        del logits, h, lab
    del hidden, shift_hidden, shift_labels
    return total.detach().cpu()


def _measure_ppl_sliding_window(model, encodings, quant_fn,
                                quant_target: str = "weight_act") -> float:
    # kv_only mode: Linear forwards stay BF16. The SDPA patch (applied by
    # the caller) is the only quantization site.
    patch_linear = (quant_target != "kv_only")
    orig_forwards = (_patch_linear_forward(model, quant_fn, quant_target=quant_target)
                     if patch_linear else {})

    seq_len = encodings.input_ids.size(1)
    nlls    = []
    # Large-vocab models (e.g. Gemma-2, V=256k) compute the LM loss in sequence
    # chunks to avoid OOM on the full-vocab logits at the 4680-token window;
    # smaller-vocab models (Llama/Mistral) use the standard HF loss unchanged.
    use_chunked = getattr(model.config, "vocab_size", 0) >= PPL_CHUNK_VOCAB
    # Per-model PPL window. Gemma-2's decode is capped (PPL_DECODE_BY_TYPE) so
    # the L*L attention matrix fits in memory; the invariant
    # ppl_seq_len - ppl_stride == ppl_prefill still holds, so starting the loop
    # at ppl_prefill keeps every window's prefill context intact.
    ppl_prefill, ppl_decode, ppl_seq_len, ppl_stride = _ppl_window(model)

    try:
        with torch.no_grad():
            # Start at PPL_PREFILL (not 0) so EVERY scored window carries a
            # full PPL_PREFILL-token prefill context: at i = PPL_PREFILL the
            # window begin_loc lands exactly at 0 (since PPL_SEQ_LEN - PPL_STRIDE
            # == PPL_PREFILL), giving [0, PREFILL+DECODE) with the first PREFILL
            # positions masked. This removes the first-window prefill=0 artifact
            # of the plain range(0, ...) tiling; every counted window now feeds
            # PREFILL+DECODE tokens and scores only the last DECODE positions.
            pbar       = tqdm(range(ppl_prefill, seq_len, ppl_stride), desc="  PPL steps", leave=False)
            step_count = 0

            for i in pbar:
                if step_count >= PPL_MAX_STEPS:
                    break

                begin_loc = max(i + ppl_stride - ppl_seq_len, 0)
                end_loc   = min(i + ppl_stride, seq_len)
                trg_len   = end_loc - i

                if trg_len < ppl_stride:
                    continue

                input_ids  = encodings.input_ids[:, begin_loc:end_loc].to(DEVICE)
                target_ids = input_ids.clone()
                target_ids[:, :-trg_len] = -100

                model.gradient_checkpointing_enable()
                window_nll = _window_sum_nll(model, input_ids, target_ids,
                                             trg_len, use_chunked)
                model.gradient_checkpointing_disable()

                nlls.append(window_nll)
                step_count += 1

                del window_nll, input_ids, target_ids
                gc.collect()
                torch.cuda.empty_cache()

    finally:
        if patch_linear:
            _unpatch_linear_forward(model, orig_forwards)
        gc.collect()
        torch.cuda.empty_cache()

    if not nlls:
        return float("nan")

    total_trg_len = ppl_stride * len(nlls)
    ppl = torch.exp(torch.stack(nlls).sum() / total_trg_len)
    return ppl.item()


def _measure_generation_ppl(model, encodings, quant_fn,
                            quant_target: str = "kv_only") -> float:
    """
    Generation-style PPL for KV-cache quantization modes.

    The sliding-window PPL function measures a single teacher-forcing
    forward per window — every K, V tensor is computed fresh and used
    exactly once, so KV-cache quantization error never accumulates and
    cannot affect the measurement in a way that reflects real inference.

    This function instead prefills a prompt and then autoregressively
    decodes PPL_GEN_LENGTH steps. At each decode step:
      • model is called with the previous ground-truth target token as
        input and the accumulated past_key_values from the prior step.
      • the model's SDPA call at every layer sees the full (prefill +
        already-decoded) K, V cache; the SDPA patch (applied by the
        caller) quantizes K, V before the attention computation, so the
        quantization error from earlier tokens enters the attention
        context for the current token.
      • the log-probability of the ground-truth next target token is
        accumulated.

    Because round-to-nearest quantization is idempotent, re-quantizing
    cached K, V at every step is functionally identical to storing
    quantized K, V in the cache from the start (which is what real
    KV-cache quantization implementations do). The PPL this function
    returns is therefore directly comparable to KV-cache-quantization
    PPL numbers reported elsewhere.

    quant_target controls Linear-side quantization the same way as
    `_measure_ppl_sliding_window`:
      • kv_only        : Linear weight & input stay BF16.
      • weight_act_kv  : Linear weight & input are also quantized
                         (via _patch_linear_forward).
    """
    patch_linear = quant_target == "weight_act_kv"
    orig_forwards = (_patch_linear_forward(model, quant_fn, quant_target=quant_target)
                     if patch_linear else {})

    seq_len           = encodings.input_ids.size(1)
    total_log_prob    = 0.0
    total_tokens      = 0
    n_samples         = 0
    window_size       = PPL_GEN_PROMPT_LEN + PPL_GEN_LENGTH

    try:
        with torch.no_grad():
            window_starts = list(range(0, seq_len - window_size, PPL_GEN_STRIDE))
            pbar = tqdm(window_starts, desc="  gen-PPL samples", leave=False)
            for win_start in pbar:
                if total_tokens >= PPL_GEN_MAX_TOKENS:
                    break

                prompt_ids = encodings.input_ids[
                    :, win_start : win_start + PPL_GEN_PROMPT_LEN
                ].to(DEVICE)
                target_ids = encodings.input_ids[
                    :, win_start + PPL_GEN_PROMPT_LEN :
                      win_start + PPL_GEN_PROMPT_LEN + PPL_GEN_LENGTH
                ].to(DEVICE)

                # ── Prefill ──────────────────────────────────────────────
                # The SDPA patch (applied by the caller) quantizes K, V
                # for every layer during this single forward.
                outputs = model(prompt_ids, use_cache=True)
                past    = outputs.past_key_values
                # Last position's logits predict the first target token.
                last_logits = outputs.logits[:, -1, :].float()
                del outputs

                # ── Autoregressive decode ────────────────────────────────
                for t in range(PPL_GEN_LENGTH):
                    target_token = target_ids[:, t : t + 1]   # (1, 1)
                    log_probs    = torch.log_softmax(last_logits, dim=-1)
                    lp           = log_probs.gather(-1, target_token).squeeze()
                    total_log_prob += lp.item()
                    total_tokens   += 1
                    del log_probs, lp

                    if t == PPL_GEN_LENGTH - 1:
                        # No need for another forward — we've already used
                        # the last ground-truth token to score it.
                        break

                    # Next step: feed the ground-truth target token in
                    # and let the cache (with quantized K, V) grow.
                    outputs = model(
                        target_token,
                        past_key_values=past,
                        use_cache=True,
                    )
                    past        = outputs.past_key_values
                    last_logits = outputs.logits[:, -1, :].float()
                    del outputs

                del past, last_logits, prompt_ids, target_ids
                n_samples += 1
                torch.cuda.empty_cache()
                gc.collect()

    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print(f"         OOM during generation PPL at sample {n_samples}.")
            torch.cuda.empty_cache()
            gc.collect()
        else:
            raise
    finally:
        if patch_linear:
            _unpatch_linear_forward(model, orig_forwards)
        gc.collect()
        torch.cuda.empty_cache()

    if total_tokens == 0:
        return float("nan")

    return math.exp(-total_log_prob / total_tokens)


# ─────────────────────────────────────────────────────────────────────────────
# Per-model run (load -> measure all configs -> save -> free)
# ─────────────────────────────────────────────────────────────────────────────
def _run_one_model(model_name: str, modes: dict, dataset,
                    measure_qsnr_flag: bool, measure_ppl_flag: bool) -> None:
    """
    Load `model_name`, measure every config, write per-model CSV/plots, then
    free the model. `baseline_qsnr` / `baseline_ppl` are local so each model
    uses its OWN MXINT8 baseline as the reference row.
    """
    slug = _model_slug(model_name)
    quant_target = modes["quant_target"]
    scope_tag    = _scope_short_tag(quant_target)
    # ── Two SDPA-side quantization knobs ────────────────────────────────────
    # sdpa_quantized : full Q / K / V / attn-weight quantization through
    #                  `patch_sdpa_with_callable`. Reserved for future
    #                  experiments — kept off by default so attention
    #                  internals stay BF16 under the standard SmoothQuant /
    #                  LLM.int8() convention.
    # kv_quantized   : KV-cache-only quantization through
    #                  `patch_sdpa_kv_only`. SDPA's K and V (RoPE-applied,
    #                  head-reshaped) are quantized; Q and the post-softmax
    #                  attention probability stay BF16. Activated by
    #                  `quant_target` ∈ {weight_act_kv, kv_only}.
    # The two paths are mutually exclusive; only one F.scaled_dot_product_
    # attention patch can be active at a time.
    sdpa_quantized = quant_target == "full_act"
    kv_quantized   = quant_target in ("weight_act_kv", "kv_only")
    print(f"\nLoading model: {model_name} ...")
    if quant_target == "weight_act":
        print("*** BF16-TARGET + LINEAR WEIGHT & ACTIVATION QUANTIZATION MODE ***")
        print("    (SDPA Q/K/V and post-softmax attn stay BF16; "
              "only Linear-layer inputs are quantized.)")
    elif quant_target == "weight_only":
        print("*** BF16-TARGET + WEIGHT-ONLY QUANTIZATION MODE "
              "(activations & SDPA Q/K/V/attn stay BF16) ***")
    elif quant_target == "weight_act_kv":
        print("*** BF16-TARGET + LINEAR WEIGHT & ACTIVATION & KV-CACHE "
              "QUANTIZATION MODE ***")
        print("    (Linear weight + input + SDPA K/V are quantized; "
              "SDPA Q and post-softmax attn stay BF16.)")
    elif quant_target == "full_act":
        print("*** BF16-TARGET + FULL ACTIVATION QUANTIZATION MODE ***")
        print("    (Linear weight + input AND SDPA Q/K/V + post-softmax attn "
              "are all quantized; nothing in attention stays BF16.)")
    else:  # kv_only
        print("*** BF16-TARGET + KV-CACHE-ONLY QUANTIZATION MODE ***")
        print("    (Only SDPA K/V are quantized; Linear weight, input, "
              "SDPA Q, and post-softmax attn stay BF16.)")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # All models load with native SDPA. SDPA's fused kernel uses O(L) memory
    # (it never materializes the [L, L] attention-score matrix), which is what
    # lets Gemma-2-9B fit the 4680-token PPL window on a 24 GB card; eager
    # builds several [L, L] tensors per layer and OOMs at this length. The cost
    # is that the SDPA code path omits Gemma-2's attention logit soft-capping —
    # a small accuracy effect that cancels between the BF16 baseline and the
    # quantized configs (the quantization-quality delta is what we report).
    # For kv_only the KV-cache patch replaces F.scaled_dot_product_attention,
    # so loading SDPA is also required for that patch to fire.
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map=DEVICE,
        attn_implementation="sdpa",
    )
    model.eval()

    # ── QSNR calibration inputs ──────────────────────────────────────────────
    if measure_qsnr_flag:
        qsnr_inputs = tokenizer(
            dataset["text"][:NUM_CALIB_SAMPLES],
            return_tensors="pt", padding=True,
            truncation=True, max_length=MAX_SEQ_LEN,
        ).to(DEVICE)
    else:
        qsnr_inputs = None

    # ── PPL: full test set joined → single sliding-window pass ───────────────
    if measure_ppl_flag:
        print("Tokenizing WikiText-2 test set for PPL evaluation ...")
        ppl_full_text = "\n\n".join(dataset["text"])
        ppl_encodings = tokenizer(ppl_full_text, return_tensors="pt")
        _pf, _dec, _sl, _st = _ppl_window(model)
        _capped = " [capped]" if _dec != PPL_DECODE else ""
        print(f"  Total tokens: {ppl_encodings.input_ids.size(1):,}  "
              f"(seq_len={_sl}, stride={_st}, max_steps={PPL_MAX_STEPS})")
        print(f"  PPL mode    : teacher-forcing sliding window "
              f"(prefill={_pf}, decode={_dec}{_capped})")
        if quant_target == "kv_only":
            # KV is quantized at every SDPA call during the teacher-forcing
            # forward. On the default head_dim axis the grouping is
            # per-token-independent, so it matches decode-time KV-cache
            # quantization exactly. KV_AXIS=kivi / kivi_split are unsupported.
            axis_caption = {
                "mx_standard": "head_dim grouping, OCP MX standard",
                "kivi":        "token-axis grouping, K=V same axis (KIVI simplified)",
                "kivi_split":  "K=token-axis, V=head_dim (KIVI paper recommendation)",
            }[KV_AXIS]
            print(f"  KV axis     : {KV_AXIS}  ({axis_caption})")
    else:
        ppl_encodings = None

    configs       = get_experiment_configs(modes, model_name)
    results       = []
    baseline_qsnr = None
    baseline_ppl  = None
    print(f"\n=== Single-Level Mantissa Sharing Experiment (QSNR + PPL) [{scope_tag}, BF16] ===")
    print(f"    model = {model_name}")
    print(f"    {_mode_subtitle(modes)}")
    print(f"    block_size={BLOCK_SIZE}, calib_samples={NUM_CALIB_SAMPLES}")
    _bw_values, _mg_values = effective_sweep(quant_target, model_name, modes.get("_sweep_override"))
    print(f"    ml1_bitwidth ∈ {_bw_values}, ml1_mg ∈ {_mg_values}")
    print(f"    total configurations: {len(configs)}\n")
    for cfg in configs:
        label       = cfg["name"].replace("\n", " ")
        total_bits  = cfg["total_bits"]
        bit_savings = MXINT8_TOTAL_BITS - total_bits

        # SDPA-side patching: two mutually exclusive paths.
        # - sdpa_quantized (currently off by default): full Q/K/V/attn
        #   quantization. Reserved for future experiments.
        # - kv_quantized: KV-cache-only quantization (K, V only) for
        #   quant_target ∈ {weight_act_kv, kv_only}.
        # The wrapper is shared by both paths but selected by KV_AXIS:
        #   mx_standard : K, V both on head_dim axis (MX default)
        #   kivi        : K, V both on sequence axis (KIVI simplified)
        #   kivi_split  : K on sequence axis, V on head_dim axis
        #                 (KIVI paper recommendation)
        # All three wrappers add the same block-size tail guard around
        # the underlying ECM.
        if sdpa_quantized or kv_quantized:
            if kv_quantized and KV_AXIS == "kivi":
                wrapper = _make_sdpa_quant_fn_kivi
            elif kv_quantized and KV_AXIS == "kivi_split":
                wrapper = _make_sdpa_quant_fn_kivi_split
            else:
                wrapper = _make_sdpa_quant_fn
            sdpa_quant_fn = wrapper(cfg["fn"])
        else:
            sdpa_quant_fn = None

        # ── QSNR ─────────────────────────────────────────────────────────────
        # The BF16 reference row is PPL-only: it exists solely to anchor the
        # PPL quality threshold (BF16_PPL / 0.97). QSNR is neither measured
        # nor recorded for it, and it is excluded from the QSNR / scatter
        # plots (see _plot_qsnr_bar / _plot_qsnr_vs_ppl_scatter).
        #
        # `kv_only` is also skipped here: with Linear weight and input both
        # at BF16, every per-layer Linear output is identical to the BF16
        # reference, so the QSNR signal/noise ratio is ill-defined
        # (numerator finite, denominator → 0). KV-cache quantization
        # quality is captured downstream by PPL.
        is_bf16_ref = (cfg["group"] == "bf16_ref")
        skip_qsnr   = is_bf16_ref or (quant_target == "kv_only")
        qsnr = float("nan")
        qsnr_loss = float("nan")
        loss_per_saved = float("nan")
        if measure_qsnr_flag and not skip_qsnr:
            print(f"> [QSNR] {label} ...", end=" ", flush=True)
            if sdpa_quantized:
                patch_sdpa_with_callable(sdpa_quant_fn)
            elif kv_quantized:
                patch_sdpa_kv_only(sdpa_quant_fn)
            try:
                qsnr = measure_qsnr(model, qsnr_inputs, cfg["fn"],
                                    quant_target=quant_target)
            finally:
                if sdpa_quantized or kv_quantized:
                    unpatch_sdpa()
            if baseline_qsnr is None:
                baseline_qsnr = qsnr
            qsnr_loss      = baseline_qsnr - qsnr
            loss_per_saved = qsnr_loss / bit_savings if bit_savings > 0 else 0.0
            print(
                f"QSNR={qsnr:.2f} dB  "
                f"loss={qsnr_loss:.3f} dB  "
                f"savings={bit_savings}b  "
                f"loss/saved={loss_per_saved:.5f} dB/bit"
            )
            torch.cuda.empty_cache()
            gc.collect()
        elif is_bf16_ref:
            print(f"> [QSNR] {label} ... skipped (BF16 ref is PPL-only)")
        elif quant_target == "kv_only":
            print(f"> [QSNR] {label} ... skipped "
                  f"(kv_only: Linear outputs match BF16 reference)")

        # ── Perplexity ───────────────────────────────────────────────────────
        # All scopes (weight_only / weight_act / kv_only) use the same
        # prefill=PPL_PREFILL / decode=PPL_DECODE teacher-forcing sliding
        # window. For kv_only, KV is quantized along the head_dim axis (OCP MX
        # standard, the KV_AXIS default); that grouping is per-token-independent,
        # so a teacher-forcing forward attends to exactly the same quantized
        # K/V that decode-time KV-cache quantization would produce — i.e. the
        # measurement matches real inference without a slow autoregressive loop.
        # (Generation-style PPL via _measure_generation_ppl is retained below
        # but no longer wired in; KV_AXIS=kivi / kivi_split are unsupported on
        # this path.)
        ppl_fn      = _measure_ppl_sliding_window
        ppl_fn_name = "PPL"

        ppl = float("nan")
        ppl_delta = float("nan")
        if measure_ppl_flag:
            print(f"  [{ppl_fn_name}]  {label} ...")
            if sdpa_quantized:
                patch_sdpa_with_callable(sdpa_quant_fn)
            elif kv_quantized:
                patch_sdpa_kv_only(sdpa_quant_fn)
            try:
                try:
                    ppl = ppl_fn(model, ppl_encodings, cfg["fn"],
                                 quant_target=quant_target)
                    print(f"         PPL={ppl:.4f}")
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        print(f"         OOM — skipping.")
                        ppl = float("nan")
                    else:
                        raise
            finally:
                if sdpa_quantized or kv_quantized:
                    unpatch_sdpa()

            if baseline_ppl is None and math.isfinite(ppl):
                baseline_ppl = ppl
            ppl_delta = ((ppl - baseline_ppl)
                         if (baseline_ppl is not None and math.isfinite(ppl))
                         else float("nan"))

            torch.cuda.empty_cache()
            gc.collect()

        results.append({
            "Model":                   model_name,
            "Configuration":           cfg["name"],
            "Group":                   cfg["group"],
            "Total Bits":              total_bits,
            "Perplexity":              ppl,
            "PPL Delta":               ppl_delta,
            "Bit Savings":             bit_savings,
            "QSNR (dB)":               qsnr,
            "QSNR Loss (dB)":          qsnr_loss,
            # Mode annotation columns (same value on every row, but explicit)
            "Mode_quant_target":       modes["quant_target"],
            "Mode_sharing":            modes["sharing_mode"],
            "Mode_rounding":           modes["rounding_mode"],
            "Mode_encoding":           modes["encoding"],
            "Mode_metric":             modes["metric_mode"],
        })
    df = pd.DataFrame(results)

    tag = _mode_tag(modes)
    csv_path = f"single_level_mantissa_sharing__{slug}__{tag}.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved → '{csv_path}'")

    show_cols = ["Configuration", "Total Bits"]
    if measure_ppl_flag:
        show_cols += ["Perplexity", "PPL Delta"]
    show_cols += ["Bit Savings"]
    if measure_qsnr_flag:
        show_cols += ["QSNR (dB)", "QSNR Loss (dB)"]
    print(df[show_cols].to_string(index=False))

    if measure_qsnr_flag:
        _plot_qsnr_bar(df, modes, model_name,
                       out_path=f"single_level_mantissa_sharing__{slug}__{tag}__QSNR.png")
    if measure_ppl_flag:
        _plot_ppl_bar(df, modes, model_name,
                      out_path=f"single_level_mantissa_sharing__{slug}__{tag}__PPL.png")
    if measure_qsnr_flag and measure_ppl_flag:
        _plot_qsnr_vs_ppl_scatter(df, modes, model_name,
                                  out_path=f"single_level_mantissa_sharing__{slug}__{tag}__scatter.png")

    # Diagnostic: unconditional, independent of metric_mode / plot branches.
    dump_skip_report(header=f"single_level_mantissa_sharing [{slug}]")

    # ── Free the model before the next one is loaded ─────────────────────────
    del model, tokenizer, qsnr_inputs, ppl_encodings, configs, results, df
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def run_experiment():
    # 0) Interactive mode selection — asked ONCE, shared across all models.
    modes = prompt_all_modes()
    measure_qsnr_flag = modes["metric_mode"] in ("both", "qsnr_only")
    measure_ppl_flag  = modes["metric_mode"] in ("both", "ppl_only")

    same_seeds(SEED)
    print("Loading dataset (WikiText-2) ...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")

    # Resolve quant_target into the explicit list of (model, scope) runs:
    #   • a single real scope -> that scope across every model in MODEL_NAMES
    #   • "all"               -> every real scope across every model
    #   • a curated preset    -> only the (model, scope) pairs it lists
    #                            (CURATED_SCOPE_SETS, e.g. "scope1")
    # Each run writes its OWN per-(model,scope) CSV/PNG (scope + model slug are
    # embedded in the filenames), so nothing overwrites anything else.
    qt = modes["quant_target"]
    if qt == "all":
        run_pairs = [(m, sc, None) for sc in QUANT_TARGET_OPTIONS for m in MODEL_NAMES]
    elif qt in CURATED_SCOPE_SETS:
        preset_sweep = CURATED_SWEEP_OVERRIDE.get(qt)  # None or (bw, mg)
        run_pairs = []
        for model_key, scope_list in CURATED_SCOPE_SETS[qt]:
            m = next((mm for mm in MODEL_NAMES
                      if model_key.lower() in mm.lower()), None)
            if m is None:
                print(f"  [{qt}] skipping '{model_key}': "
                      f"no matching entry in MODEL_NAMES")
                continue
            run_pairs += [(m, sc, preset_sweep) for sc in scope_list]
    else:
        run_pairs = [(m, qt, None) for m in MODEL_NAMES]

    print("=" * 70)
    print(f"quant_target = {qt}  ->  {len(run_pairs)} (model, scope) run(s):")
    for i, (m, sc, sw) in enumerate(run_pairs, 1):
        sw_tag = f"  [sweep override: ml1_bitwidth={sw[0]}]" if sw is not None else ""
        print(f"  {i}. {m}  |  {sc} ({_scope_short_tag(sc)}){sw_tag}")
    print("=" * 70)

    for idx, (model_name, scope, sweep) in enumerate(run_pairs, 1):
        scope_modes = dict(modes, quant_target=scope)
        if sweep is not None:
            scope_modes["_sweep_override"] = sweep
        print("\n" + "#" * 70)
        print(f"# [{idx}/{len(run_pairs)}] MODEL: {model_name}  | SCOPE: {scope}")
        print("#" * 70)
        # Re-seed per (model, scope) so each run is independently reproducible.
        same_seeds(SEED)
        reset_skip_log()
        _run_one_model(
            model_name, scope_modes, dataset,
            measure_qsnr_flag=measure_qsnr_flag,
            measure_ppl_flag=measure_ppl_flag,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Plot styling constants
# ─────────────────────────────────────────────────────────────────────────────
GROUP_COLORS = {
    "bf16_ref":                              "#000000",
    "baseline":                              "#4878CF",
    "single_level_mantissa_sharing":         "#8C564B",
    "MSAQ_signed":                           "#9467BD",
    "single_rounding_bitwidth_reduction":    "#2CA02C",
    "double_rounding_bitwidth_reduction":    "#FF7F0E",
}
GROUP_LABELS = {
    "bf16_ref":                              "BF16 (no quant, reference)",
    "baseline":                              "MXINT8 Baseline",
    "single_level_mantissa_sharing":         "Single-Level Mantissa Sharing (No Comp)",
    "MSAQ_signed":                           "MSAQ-S (FP-resid, u-bit signed)",
    "single_rounding_bitwidth_reduction":    "Single-Rounding Bitwidth Reduction",
    "double_rounding_bitwidth_reduction":    "Double-Rounding Bitwidth Reduction (Slice-and-Scale)",
}
U_ALPHA = {1: 1.0, 2: 0.75, 3: 0.50, 4: 0.30, 5: 0.18, 6: 0.10}

GROUP_MARKER = {
    "bf16_ref":                              "s",
    "baseline":                              "D",
    "single_level_mantissa_sharing":         "o",
    "MSAQ_signed":                           "P",
    "single_rounding_bitwidth_reduction":    "v",
    "double_rounding_bitwidth_reduction":    "^",
}

def _u_from_name(name: str) -> int:
    """
    Extract the per-config bit-width parameter from a config name.

    Recognised tokens (in order of preference):
      - "ml1_bitwidth=X"       : sharing groups (mantissa-sharing variants)
      - "reduced_bitwidth=X"   : bitwidth-reduction groups
                                 (single_rounding / double_rounding)
    Backward-compat tokens for older saved CSVs:
      - "ml1_bw=X", "trunc_bitwidth=X", "tb=X", "u="
    """
    for part in name.split("\n"):
        part = part.strip().replace(",", " ").replace(")", " ").replace("(", " ")
        for tok in part.split():
            for prefix in ("ml1_bitwidth=", "reduced_bitwidth=",
                           "ml1_bw=", "trunc_bitwidth=", "tb=", "u="):
                if tok.startswith(prefix):
                    try:
                        return int(tok.split("=")[1])
                    except ValueError:
                        pass
    return 1


def _alpha_for_row(row) -> float:
    if row["Group"] == "baseline":
        return 0.88
    return U_ALPHA.get(_u_from_name(row["Configuration"]), 0.88)


def _build_bar_panel(ax, df, metric, ylabel, title, higher_better,
                     ymax_cap: float = None):
    """
    Bar-panel renderer shared by the QSNR and PPL absolute-value plots.

    ymax_cap : optional. When set, bar heights are clipped to this value
               (so the y-axis can be fixed at [y_lo, ymax_cap] and small
               features like the BF16 quality-threshold line stay visible)
               but the in-bar annotation still prints the un-clipped value
               so no information is lost. Clipped bars get a red annotation
               as a visual flag.
    """
    n      = len(df)
    colors = [GROUP_COLORS[g] for g in df["Group"]]
    alphas = [_alpha_for_row(row) for _, row in df.iterrows()]
    labels = df["Configuration"].tolist()
    vals   = df[metric].tolist()
    finite_vals = [v for v in vals if isinstance(v, (int, float)) and math.isfinite(v)]

    def _clip(v):
        if not (isinstance(v, (int, float)) and math.isfinite(v)):
            return 0.0
        return min(v, ymax_cap) if ymax_cap is not None else v

    for i, (val, color, alpha) in enumerate(zip(vals, colors, alphas)):
        plot_val = _clip(val)
        ax.bar(i, plot_val, color=color, edgecolor="black",
               linewidth=0.6, alpha=alpha)

    ax.set_title(title, fontsize=14, fontweight="bold", pad=10)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_xticks(range(n))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    if ymax_cap is not None:
        # Fixed y range; finite outliers above the cap are visualised as
        # full-height bars with their true value annotated above the bar.
        in_cap_vals = [v for v in finite_vals if v <= ymax_cap]
        if in_cap_vals:
            vmin = min(in_cap_vals)
            y_lo = (vmin - abs(vmin) * 0.1) if higher_better else max(0.0, vmin - abs(vmin) * 0.1)
        else:
            y_lo = 0.0
        ax.set_ylim(y_lo, ymax_cap)
    elif finite_vals:
        vmin, vmax = min(finite_vals), max(finite_vals)
        margin = abs(vmax - vmin) * 0.10 if vmax != vmin else (abs(vmax) * 0.1 if vmax else 1.0)
        y_lo = (vmin - margin) if higher_better else max(0.0, vmin - margin)
        ax.set_ylim(y_lo, vmax + margin)

    def fmt(v):
        if not (isinstance(v, (int, float)) and math.isfinite(v)):
            return "N/A"
        return f"{v:.4f}" if abs(v) < 1 else f"{v:.3f}"

    for i, val in enumerate(vals):
        plot_val = _clip(val)
        is_clipped = (isinstance(val, (int, float)) and math.isfinite(val)
                      and ymax_cap is not None and val > ymax_cap)
        # Clipped bars (true value > ymax_cap) get a much more visible
        # annotation: larger font, red text on a white background box with
        # a red border, sitting just above the capped bar top so the real
        # PPL is impossible to miss even when many bars are clipped at
        # the same height.
        ax.annotate(
            fmt(val),
            xy=(i, plot_val),
            ha="center", va="bottom" if plot_val >= 0 else "top",
            fontsize=(9 if is_clipped else 6),
            fontweight="bold",
            color=("red" if is_clipped else "black"),
            xytext=(0, 5 if is_clipped else (3 if plot_val >= 0 else -9)),
            textcoords="offset points",
            bbox=(dict(boxstyle="round,pad=0.2", facecolor="white",
                       edgecolor="red", linewidth=0.8, alpha=0.95)
                  if is_clipped else None),
            zorder=10,
        )

    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks(range(n))
    ax2.set_xticklabels(
        [f"−{r['Bit Savings']}b" for _, r in df.iterrows()],
        fontsize=6, color="#555555",
    )
    ax2.tick_params(length=0)

    direction = "↑ Higher is Better" if higher_better else "↓ Lower is Better"
    ax.text(0.99, 0.98, direction, transform=ax.transAxes,
            fontsize=9, color="#333333", ha="right", va="top",
            style="italic", alpha=0.7)


def _build_legend_handles():
    handles = [
        Patch(facecolor=GROUP_COLORS["bf16_ref"],                              alpha=0.88, edgecolor="black",
              label=GROUP_LABELS["bf16_ref"]),
        Patch(facecolor=GROUP_COLORS["baseline"],                              alpha=0.88, edgecolor="black",
              label=GROUP_LABELS["baseline"]),
        Patch(facecolor=GROUP_COLORS["single_level_mantissa_sharing"],         alpha=1.00, edgecolor="black",
              label=GROUP_LABELS["single_level_mantissa_sharing"] + " (ml1_bitwidth=1)"),
        Patch(facecolor=GROUP_COLORS["MSAQ_signed"],                           alpha=1.00, edgecolor="black",
              label=GROUP_LABELS["MSAQ_signed"] + " (ml1_bitwidth=1)"),
        Patch(facecolor=GROUP_COLORS["single_rounding_bitwidth_reduction"],    alpha=1.00, edgecolor="black",
              label=GROUP_LABELS["single_rounding_bitwidth_reduction"] + " (reduced_bitwidth=1)"),
        Patch(facecolor=GROUP_COLORS["double_rounding_bitwidth_reduction"],    alpha=1.00, edgecolor="black",
              label=GROUP_LABELS["double_rounding_bitwidth_reduction"] + " (reduced_bitwidth=1)"),
        Patch(facecolor="#555555", alpha=0.75, edgecolor="black", label="Faded Color: bitwidth=2"),
        Patch(facecolor="#555555", alpha=0.50, edgecolor="black", label="Lighter Color: bitwidth=3"),
        Patch(facecolor="#555555", alpha=0.30, edgecolor="black", label="Lightest Color: bitwidth=4"),
    ]
    return handles


# ─────────────────────────────────────────────────────────────────────────────
# (1) Absolute QSNR bar chart
# ─────────────────────────────────────────────────────────────────────────────
def _plot_qsnr_bar(df: pd.DataFrame, modes: dict, model_name: str, out_path: str):
    # The BF16 ref row is PPL-only (no QSNR measured); drop it so it does
    # not appear as an empty/zero bar on the QSNR chart.
    df = df[df["Group"] != "bf16_ref"].reset_index(drop=True)
    n = len(df)
    fig_width = max(28, int(n * 0.55))
    fig, ax = plt.subplots(figsize=(fig_width, 9))

    scope = _scope_short_tag(modes["quant_target"])
    _build_bar_panel(
        ax, df,
        metric="QSNR (dB)",
        ylabel="QSNR (dB) ↑",
        title=f"Absolute QSNR [{scope}, BF16] — {_sweep_range_tag(modes['quant_target'], model_name, modes.get('_sweep_override'))}",
        higher_better=True,
    )

    fig.legend(handles=_build_legend_handles(), loc="upper right",
               bbox_to_anchor=(0.99, 0.99), fontsize=9, framealpha=0.9)
    fig.suptitle(
        f"Single-Level Mantissa Sharing  —  Absolute QSNR  [{scope}, BF16]\n"
        f"model = {model_name}\n"
        f"{_mode_subtitle(modes)}\n"
        "(Top axis: bit savings vs MXINT8 = 264 bits/block)",
        fontsize=13, fontweight="bold", y=1.02,
    )

    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Saved plot → '{out_path}'")
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# (2) Absolute Perplexity bar chart
# ─────────────────────────────────────────────────────────────────────────────
def _plot_ppl_bar(df: pd.DataFrame, modes: dict, model_name: str, out_path: str):
    n = len(df)
    fig_width = max(28, int(n * 0.55))
    fig, ax = plt.subplots(figsize=(fig_width, 9))

    scope = _scope_short_tag(modes["quant_target"])
    _build_bar_panel(
        ax, df,
        metric="Perplexity",
        ylabel="Perplexity ↓",
        title=f"Absolute Perplexity [{scope}, BF16] — {_sweep_range_tag(modes['quant_target'], model_name, modes.get('_sweep_override'))}",
        higher_better=False,
        ymax_cap=PPL_PLOT_YMAX,
    )

    # BF16 quality threshold: the BF16-ref row carries the pure-BF16 PPL.
    # "Keep 97% of BF16 quality" → PPL must stay at/below  BF16_PPL / 0.97
    # (PPL is lower-is-better, so a 3% quality loss is a higher PPL ceiling).
    # The y-axis is fixed at [y_lo, PPL_PLOT_YMAX] so the threshold line stays
    # prominent; bars whose true PPL exceeds PPL_PLOT_YMAX are clipped to the
    # cap and annotated with their real value (in red) by _build_bar_panel.
    thr_line = None
    bf16_rows = df[df["Group"] == "bf16_ref"]
    if not bf16_rows.empty:
        bf16_ppl = bf16_rows["Perplexity"].iloc[0]
        if isinstance(bf16_ppl, (int, float)) and math.isfinite(bf16_ppl) and bf16_ppl > 0:
            thr_line = bf16_ppl / 0.97
            ax.axhline(
                thr_line, color="red", linestyle="--", linewidth=1.8,
                alpha=0.9, zorder=5,
            )
            ax.text(
                0.005, thr_line, f" 97% BF16 quality (PPL ≤ {thr_line:.3f})",
                transform=ax.get_yaxis_transform(),
                color="red", fontsize=9, fontweight="bold",
                va="bottom", ha="left", zorder=6,
            )

    fig.legend(handles=_build_legend_handles(), loc="upper right",
               bbox_to_anchor=(0.99, 0.99), fontsize=9, framealpha=0.9)
    fig.suptitle(
        f"Single-Level Mantissa Sharing  —  Absolute Perplexity  [{scope}, BF16]\n"
        f"model = {model_name}\n"
        f"{_mode_subtitle(modes)}\n"
        "(WikiText-2 sliding window  |  Top axis: bit savings vs MXINT8 = 264 bits/block"
        + (f"  |  red = BF16/0.97 = {thr_line:.3f})" if thr_line is not None else ")"),
        fontsize=13, fontweight="bold", y=1.02,
    )

    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Saved plot → '{out_path}'")
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# (3) QSNR (x) vs Perplexity (y) scatter — with Pareto frontier
# ─────────────────────────────────────────────────────────────────────────────
def _plot_qsnr_vs_ppl_scatter(df: pd.DataFrame, modes: dict, model_name: str, out_path: str):
    # The BF16 ref row has no QSNR (PPL-only), so it cannot be placed on a
    # QSNR-vs-PPL scatter; exclude it.
    df = df[df["Group"] != "bf16_ref"].reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(14, 9))

    plotted_groups = set()

    for _, row in df.iterrows():
        qsnr = row["QSNR (dB)"]
        ppl  = row["Perplexity"]
        if not (isinstance(qsnr, (int, float)) and math.isfinite(qsnr) and
                isinstance(ppl,  (int, float)) and math.isfinite(ppl)):
            continue

        grp    = row["Group"]
        color  = GROUP_COLORS[grp]
        marker = GROUP_MARKER[grp]
        alpha  = _alpha_for_row(row)

        if grp == "baseline":
            ms = 220
        else:
            u  = _u_from_name(row["Configuration"])
            ms = max(45, 170 - u * 22)

        label = GROUP_LABELS[grp] if grp not in plotted_groups else None
        ax.scatter(qsnr, ppl,
                   c=color, alpha=alpha, s=ms,
                   marker=marker,
                   edgecolors="black", linewidths=0.5,
                   label=label, zorder=3)
        plotted_groups.add(grp)

        if grp == "baseline":
            ax.annotate(
                row["Configuration"].split("\n")[0],
                xy=(qsnr, ppl),
                xytext=(8, 8), textcoords="offset points",
                fontsize=10, fontweight="bold",
            )

    # ── Pareto frontier ──────────────────────────────────────────────────────
    finite_rows = df[
        df["QSNR (dB)"].apply(lambda v: isinstance(v, (int, float)) and math.isfinite(v)) &
        df["Perplexity"].apply(lambda v: isinstance(v, (int, float)) and math.isfinite(v))
    ].copy()

    if not finite_rows.empty:
        sorted_rows = finite_rows.sort_values("QSNR (dB)", ascending=False)
        pareto_pts  = []
        min_ppl     = float("inf")
        for _, r in sorted_rows.iterrows():
            if r["Perplexity"] <= min_ppl:
                min_ppl = r["Perplexity"]
                pareto_pts.append(
                    (r["QSNR (dB)"], r["Perplexity"], r["Bit Savings"])
                )

        if pareto_pts:
            pareto_pts_sorted = sorted(pareto_pts, key=lambda p: p[0])
            px = [p[0] for p in pareto_pts_sorted]
            py = [p[1] for p in pareto_pts_sorted]
            ax.plot(px, py,
                    color="#333333", linewidth=1.6,
                    linestyle="--", alpha=0.55, zorder=2,
                    label="Pareto frontier")

            for qx, qy, bs in pareto_pts_sorted:
                ax.annotate(
                    f"−{bs}b",
                    xy=(qx, qy),
                    xytext=(0, -14), textcoords="offset points",
                    fontsize=7.5, color="#333333", ha="center",
                    fontweight="bold",
                )

    legend1 = ax.legend(loc="upper left", fontsize=9, framealpha=0.9,
                        title="Method", title_fontsize=10)
    ax.add_artist(legend1)

    u_handles = [
        Line2D([0], [0], marker="o", color="w",
               markerfacecolor="#555555", markersize=10,
               markeredgecolor="black",
               alpha=U_ALPHA.get(u, 0.88),
               label=f"ml1_bitwidth={u}")
        for u in effective_sweep(modes["quant_target"], model_name, modes.get("_sweep_override"))[0]
    ]
    ax.legend(handles=u_handles, loc="lower left",
              fontsize=9, framealpha=0.9,
              title="LSB bits (ml1_bitwidth)", title_fontsize=10)

    shape_caption = (
        "Markers:  ◆ Baseline   "
        "● Single-Level Mantissa Sharing   ✚ MSAQ-S   "
        "▽ Single-Rounding BR   △ Double-Rounding BR"
    )
    ax.text(0.01, 1.02, shape_caption, transform=ax.transAxes,
            fontsize=9, color="#333333", ha="left", va="bottom",
            style="italic")

    ax.set_xlabel("QSNR (dB)  ↑ Higher is Better", fontsize=12)
    # All scopes (incl. kv_only) now use teacher-forcing sliding-window PPL.
    ax.set_ylabel("Perplexity  ↓ Lower is Better", fontsize=12)
    scope_long = {
        "weight_act":    "Weight + Linear-Input Activation Quantization (SDPA = BF16)",
        "weight_only":   "Weight-Only Quantization (activations & SDPA = BF16)",
        "weight_act_kv": "Weight + Linear-Input Activation + KV-Cache (SDPA K/V) Quantization",
        "kv_only":       "KV-Cache (SDPA K/V) Only Quantization (weight & activations = BF16)",
        "full_act":      "Full Activation Quantization (weight + input + SDPA Q/K/V/attn)",
    }[modes["quant_target"]]
    ax.set_title(
        f"QSNR vs Perplexity  [{scope_long}, BF16-target]\n"
        f"model = {model_name}\n"
        f"{_mode_subtitle(modes)}\n"
        f"{_sweep_range_tag(modes['quant_target'], model_name, modes.get('_sweep_override'))}  |  "
        "Dashed = Pareto frontier (bit savings labeled)",
        fontsize=13, fontweight="bold", pad=24,
    )
    ax.grid(True, linestyle="--", alpha=0.35)

    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Saved plot → '{out_path}'")
    plt.close()


if __name__ == "__main__":
    run_experiment()