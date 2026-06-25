# ssnf_sdpa_patch.py
import math
import os
import torch
import torch.nn.functional as F
from ssnf_main.ForGit.src.ssnf_core import ssnf_quant

_ORIG_SDPA = None

# Query-block size for the unfused SDPA path (_unfused_sdpa_callable). That path
# materializes the [Lq, S] attention-score matrix and, when the post-softmax
# attn is quantized (full_act), an FP32 copy of it — the O(L²) tensors that
# native fused SDPA avoids. Processing the query axis in blocks of this many
# positions bounds peak memory to ~[_SDPA_Q_BLOCK, S] per step instead of
# [Lq, S], which lets full_act run the full-length PPL window on a 24 GB card.
# softmax and the attn block-scale quantization are per-query-row independent,
# so blocking is numerically exact. Override at the command line without editing
# this file via SDPA_Q_BLOCK=<n>; lower it if a model still OOMs, raise it for
# fewer Python iterations when VRAM allows.
_SDPA_Q_BLOCK = int(os.environ.get("SDPA_Q_BLOCK", "512"))


# ─────────────────────────────────────────────────────────────────────────────
# Skip-logging registry  (diagnostic only — does NOT change any measured value)
# ─────────────────────────────────────────────────────────────────────────────
# Records every tensor that was passed through UNQUANTIZED because its last
# dim was not a multiple of the quantizer block size. Keyed by
# (site, last_dim) so the report stays short; `names` collects the distinct
# layer names seen at that site (when a name is available).
#
# Linear-side call sites carry a real module name (e.g.
# "model.layers.0.mlp.gate_proj"). SDPA-side call sites have no module name,
# so they use fixed tags ("sdpa:Q" / "sdpa:K" / "sdpa:V" / "sdpa:attn").

_SKIP_LOG: dict = {}


def reset_skip_log() -> None:
    """Clear the skip registry. Call once before each config's measurement."""
    _SKIP_LOG.clear()


def record_skip(site: str, name, shape) -> None:
    """
    Register one skip event.

    Args:
        site  : call-site tag, e.g. "linear" or "sdpa:Q".
        name  : module name if known, else None.
        shape : the tensor's shape (torch.Size or tuple).
    """
    last_dim = int(shape[-1]) if len(shape) else -1
    key = (site, last_dim)
    entry = _SKIP_LOG.get(key)
    if entry is None:
        entry = {"count": 0, "names": set()}
        _SKIP_LOG[key] = entry
    entry["count"] += 1
    if name is not None:
        entry["names"].add(name)


def dump_skip_report(header: str = "") -> None:
    """
    Print a compact summary of all skipped tensors. Safe to call when the
    registry is empty (prints a single 'no skips' line).
    """
    title = "Block-size SKIP report"
    if header:
        title += f"  [{header}]"
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)
    if not _SKIP_LOG:
        print("  (no tensors were skipped — every quantized tensor was "
              "block-aligned)")
        print("=" * 70)
        return
    for (site, last_dim), entry in sorted(_SKIP_LOG.items()):
        names = entry["names"]
        if names:
            shown = sorted(names)
            name_str = ", ".join(shown[:8])
            if len(shown) > 8:
                name_str += f", … (+{len(shown) - 8} more)"
            name_part = f"  names={{{name_str}}}"
        else:
            name_part = "  names=<unavailable for this call site>"
        print(f"  site={site:<12} last_dim={last_dim:<6} "
              f"skipped×{entry['count']}{name_part}")
    print("=" * 70)


# ─────────────────────────────────────────────────────────────────────────────
# Site-aware callable invoker
# ─────────────────────────────────────────────────────────────────────────────
def _call_qfn(quant_fn, t, site):
    """
    Invoke a user-supplied quant callable, passing `site` only if the
    callable accepts it. This lets the standard wrapper
    (`_make_sdpa_quant_fn`) tag Q/K/V/attn skips distinctly while keeping
    plain (tensor)->tensor callables working unchanged.
    """
    try:
        return quant_fn(t, site=site)
    except TypeError:
        return quant_fn(t)


# ─────────────────────────────────────────────────────────────────────────────
# Internal SDPA body (callable-based)
# ─────────────────────────────────────────────────────────────────────────────
def _unfused_sdpa_callable(query, key, value, quant_fn,
                           attn_mask=None, dropout_p: float = 0.0,
                           is_causal: bool = False):
    """
    Memory-optimized unfused SDPA. Q/K/V are already quantized by the
    caller; the attention weight `attn` is also quantized here via
    `quant_fn(attn)` before the final matmul with `value`.

    `quant_fn` is any callable with signature (tensor) -> tensor.

    The query axis is processed in blocks of `_SDPA_Q_BLOCK` positions so the
    [Lq, S] score / attn matrices (and the FP32 attn-quantization copy) are
    never materialized in full. softmax and the attn block-scale quantization
    are per-query-row independent, so this is numerically identical to the
    all-at-once computation (modulo FP matmul accumulation order) while keeping
    peak memory near [_SDPA_Q_BLOCK, S] per step. This is what lets the full
    Q/K/V/attn quantization scope (full_act) run the long PPL window without
    OOMing a 24 GB card.
    """
    # --- 0) Grouped-Query Attention (GQA) head replication ---
    if query.dim() == 4 and key.dim() == 4 and value.dim() == 4:
        Bq, Hq, Lq, Dq = query.shape
        Bk, Hk, Sk, Dk = key.shape
        if Hq != Hk:
            rep = Hq // Hk
            key   = key.repeat_interleave(rep, dim=1)
            value = value.repeat_interleave(rep, dim=1)

    d_k     = query.size(-1)
    sqrt_dk = math.sqrt(d_k)
    Lq      = query.shape[-2]
    S       = key.shape[-2]
    key_t   = key.transpose(-2, -1)              # view, reused every block

    # Full-length additive causal mask, built once and sliced per query block.
    # Only this single [Lq, S] tensor is held (not a per-head [Hq, Lq, S]
    # stack), so it does not reintroduce the blow-up that query-blocking removes.
    causal_full = None
    if is_causal:
        mask_val = torch.finfo(query.dtype).min
        causal_full = torch.triu(
            torch.full((Lq, S), mask_val, dtype=query.dtype, device=query.device),
            diagonal=1,
        )

    out = torch.empty(
        (query.shape[0], query.shape[1], Lq, value.shape[-1]),
        dtype=value.dtype, device=query.device,
    )

    for q0 in range(0, Lq, _SDPA_Q_BLOCK):
        q1 = min(q0 + _SDPA_Q_BLOCK, Lq)

        # 1) attention scores for this query block
        scores = torch.matmul(query[:, :, q0:q1, :], key_t)
        scores.div_(sqrt_dk)

        # 2) optional attention mask. Slice the query axis only when it is real;
        #    a size-1 query axis broadcasts over all query positions as-is.
        if attn_mask is not None:
            if attn_mask.shape[-2] == 1:
                scores.add_(attn_mask)
            else:
                scores.add_(attn_mask[..., q0:q1, :])

        # 3) causal mask for this query block
        if causal_full is not None:
            scores.add_(causal_full[q0:q1, :].view(1, 1, q1 - q0, S))

        # 4) softmax (per query row)
        attn = torch.softmax(scores, dim=-1)
        del scores

        # 5) dropout
        if dropout_p > 0.0:
            attn = F.dropout(attn, p=dropout_p, training=True)

        # 6) quantize this block's attention weight
        attn_q = _call_qfn(quant_fn, attn, "sdpa:attn")
        del attn

        # 7) value is already quantized upstream; 8) block output matmul
        out[:, :, q0:q1, :] = torch.matmul(attn_q, value)
        del attn_q

    del causal_full, key_t
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Original dict-based API (kept for Multi-Level compatibility)
# ─────────────────────────────────────────────────────────────────────────────
# Block size for the dict-based ssnf_quant path. Mirrors the BLOCK_SIZE used
# by the experiment scripts (ssnf_core defaults to 32, and every driver here
# uses block_size=32), so the guard below matches the underlying quantizer.
_SSNF_DICT_BLOCK_SIZE = 32


def _ssnf_quant_guarded(t, quant_args_act, site):
    """
    ssnf_quant with the SAME block-size guard the callable-based SDPA path
    uses: if the tensor's last dim is not a multiple of the quantizer block
    size, pass it through unchanged and record the skip.

    Previously `_unfused_sdpa`/`patch_sdpa_for_ssnf` called ssnf_quant with
    no guard, so a non-block-aligned attn weight (last dim = seq_len) would
    either crash in reshape or, when seq_len happened to be a multiple of
    32, be quantized — an asymmetry vs the callable path. This guard makes
    the dict-based path behave like the callable one and logs every skip.
    """
    bs = quant_args_act.get('block_size', _SSNF_DICT_BLOCK_SIZE)
    if bs in (None, -1):
        bs = _SSNF_DICT_BLOCK_SIZE
    if t.shape[-1] % bs != 0:
        record_skip(site, None, t.shape)
        return t
    return ssnf_quant(t, **quant_args_act)


def _unfused_sdpa(query, key, value, quant_args_act,
                  attn_mask=None, dropout_p: float = 0.0, is_causal: bool = False):
    """Legacy ssnf_quant-dict variant. Equivalent to _unfused_sdpa_callable
    with quant_fn = lambda x: ssnf_quant(x, **quant_args_act), now with the
    same block-size guard + skip logging as the callable path."""
    return _unfused_sdpa_callable(
        query, key, value,
        quant_fn=lambda t: _ssnf_quant_guarded(t, quant_args_act, "sdpa:attn"),
        attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal,
    )


def patch_sdpa_for_ssnf(quant_args_act: dict):
    """
    Patch F.scaled_dot_product_attention to quantize Q/K/V and the
    attention weight using `ssnf_quant(t, **quant_args_act)`.

    Q/K/V/attn whose last dim is not a multiple of the quantizer block
    size are passed through unchanged and recorded in the skip registry
    (matching the callable-based path's guard semantics).
    """
    global _ORIG_SDPA
    if _ORIG_SDPA is not None:
        return
    _ORIG_SDPA = F.scaled_dot_product_attention

    def _sdpa_patched(query, key, value, attn_mask=None, dropout_p=0.0,
                      is_causal=False, **kwargs):
        q_q = _ssnf_quant_guarded(query, quant_args_act, "sdpa:Q")
        k_q = _ssnf_quant_guarded(key,   quant_args_act, "sdpa:K")
        v_q = _ssnf_quant_guarded(value, quant_args_act, "sdpa:V")
        return _unfused_sdpa(
            q_q, k_q, v_q,
            quant_args_act=quant_args_act,
            attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal,
        )

    F.scaled_dot_product_attention = _sdpa_patched


# ─────────────────────────────────────────────────────────────────────────────
# Callable-based API (used by Single-Level so any quant function works)
# ─────────────────────────────────────────────────────────────────────────────
def patch_sdpa_with_callable(quant_fn):
    """
    Patch F.scaled_dot_product_attention to quantize Q/K/V and the
    attention weight using an arbitrary callable `quant_fn`.

    The callable must have signature (torch.Tensor) -> torch.Tensor.
    SDPA Q/K/V/attn tensors have shape (B, H, L, S/D); their last
    dimension may not be a multiple of the underlying quantizer's
    block size (e.g. head_dim values like 80 or 96). The caller is
    responsible for wrapping the raw quant function with whatever
    block-size guard the quantizer expects -- e.g. passing the
    tensor through unchanged when the last dim is not a multiple of
    BLOCK_SIZE (see `_make_sdpa_quant_fn` in
    single_level_mantissa_sharing.py for the standard wrapper).

    Skip logging for this path is performed by the caller-supplied
    wrapper (`_make_sdpa_quant_fn`), which has access to BLOCK_SIZE.
    """
    global _ORIG_SDPA
    if _ORIG_SDPA is not None:
        return
    _ORIG_SDPA = F.scaled_dot_product_attention

    def _sdpa_patched(query, key, value, attn_mask=None, dropout_p=0.0,
                      is_causal=False, **kwargs):
        q_q = _call_qfn(quant_fn, query, "sdpa:Q")
        k_q = _call_qfn(quant_fn, key,   "sdpa:K")
        v_q = _call_qfn(quant_fn, value, "sdpa:V")
        return _unfused_sdpa_callable(
            q_q, k_q, v_q,
            quant_fn=quant_fn,
            attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal,
        )

    F.scaled_dot_product_attention = _sdpa_patched

def patch_sdpa_kv_only(quant_fn):
    """
    KV cache 양자화 경로: SDPA에 들어가는 K, V만 양자화한다.
    Q와 post-softmax attention weight는 BF16 그대로 두고, 원본 fused
    SDPA(_ORIG_SDPA)를 호출하므로 _unfused_sdpa_callable 경로의 비용은
    들지 않는다.

    quant_fn signature는 patch_sdpa_with_callable과 동일 — (tensor) -> tensor.
    block-size 가드/skip 로깅은 호출 측 래퍼(_make_sdpa_quant_fn)가 책임진다.
    """
    global _ORIG_SDPA
    if _ORIG_SDPA is not None:
        return
    _ORIG_SDPA = F.scaled_dot_product_attention

    def _sdpa_patched(query, key, value, attn_mask=None, dropout_p=0.0,
                      is_causal=False, **kwargs):
        k_q = _call_qfn(quant_fn, key,   "sdpa:K")
        v_q = _call_qfn(quant_fn, value, "sdpa:V")
        return _ORIG_SDPA(
            query, k_q, v_q,
            attn_mask=attn_mask, dropout_p=dropout_p,
            is_causal=is_causal, **kwargs,
        )

    F.scaled_dot_product_attention = _sdpa_patched

def unpatch_sdpa():
    global _ORIG_SDPA
    if _ORIG_SDPA is not None:
        F.scaled_dot_product_attention = _ORIG_SDPA
        _ORIG_SDPA = None