# ms_lib/reference.py
#
# The verification ORACLE. Pure NumPy ground-truth results that every CUDA /
# CUTLASS kernel in csrc/ must reproduce (the role selftest_triton's "mirror"
# played before). Migrated VERBATIM from the certified §3 NUMPY MIRRORS and the
# §6 independent port in mantissa_sharing_kernel.py.
#
#   wonly_matmul   : W-only   Y = X @ dequant(W)^T          (M=1 == GEMV)
#   quant_act      : runtime activation quant (MXINT8 default; MSAQ-s if share)
#   wa_matmul      : W+A      per-block (scale_a*scale_w) * int8-dot
#   kv_attention   : fused-dequant attention (causal online-softmax reference)
#   _msaq_signed_ref : independent re-derivation of the numerics, used to
#                      cross-check pack.decompose+reconstruct (NOT shared code).
#
# Layering: reference.py -> pack.py (one direction; no cycle). reference.py owns
# "the ground-truth result"; pack.py owns "how a tensor becomes packed bytes".

import math
import numpy as np

from ms_lib.pack import (
    BLOCK, E_MAX,
    _e8m0_scale, decompose,
    dequant_weight, weight_int8,
    dequant_weight_mxint8, weight_int8_mxint8,
)


# =============================================================================
#  MATMUL / ATTENTION ORACLES  (certified transcription targets for the kernels)
# =============================================================================
def wonly_matmul(p, X):
    """W-only: X [M,K] -> Y [M,OUT].  Y = X @ dequant(W)^T.  (M=1 == GEMV.)"""
    W_dq = dequant_weight(p)                                       # [OUT, K]
    return np.asarray(X, np.float64) @ W_dq.T


def quant_act(X, u=None, gs=None, share=False):
    """Runtime activation quant. X [M,K] -> (qX int [M,nb,32], scale_a [M,nb]).
    share=False -> MXINT8 (kernel default); share=True -> MSAQ-s (accuracy match)."""
    M, K = X.shape
    nb = K // BLOCK
    xb = np.asarray(X, np.float64).reshape(M, nb, BLOCK)
    if not share:
        s = _e8m0_scale(np.abs(xb).max(axis=2, keepdims=True))     # [M,nb,1]
        q = np.clip(np.round(xb / s), -127, 127).astype(np.int64)
        return q, s[..., 0]
    s2, qu, rs = decompose(xb.reshape(M * nb, BLOCK), u, gs)
    r_exp = np.repeat(rs, gs, axis=1)
    qfull = (qu * (1 << u) + r_exp).reshape(M, nb, BLOCK)
    return qfull, s2.reshape(M, nb)


def wa_matmul(p, X, share_act=False):
    """W+A: X [M,K] -> Y [M,OUT].  Per block: (scale_a*scale_w) * int8-dot."""
    qW, sW = weight_int8(p)                                        # [OUT,nb,32],[OUT,nb]
    qX, sX = quant_act(X, p["u"], p["gs"], share=share_act)        # [M,nb,32],[M,nb]
    intdot = np.einsum("mbk,obk->mbo", qX.astype(np.int64), qW.astype(np.int64))
    return np.einsum("mbo,mb,ob->mo", intdot.astype(np.float64), sX, sW)


def kv_attention(Q, pK, pV, causal=True):
    """Fused-dequant attention mirror. Q [H,Lq,D]; pK,pV from pack_kv.
    scores = Q @ Kdq^T / sqrt(D) -> (causal) softmax -> @ Vdq.  Returns [H,Lq,D]."""
    H, Lq, D = Q.shape
    out = np.zeros((H, Lq, D), dtype=np.float64)
    for h in range(H):
        Kdq = dequant_weight(pK["_per"][h])                        # [Lk, D]
        Vdq = dequant_weight(pV["_per"][h])                        # [Lk, D]
        Lk = Kdq.shape[0]
        scores = (np.asarray(Q[h], np.float64) @ Kdq.T) / math.sqrt(D)   # [Lq,Lk]
        if causal:                                                  # query t sees keys <= t
            m = np.triu(np.ones((Lq, Lk)), k=1 + (Lk - Lq))
            scores = np.where(m > 0, -np.inf, scores)
        scores = scores - scores.max(axis=1, keepdims=True)
        pmat = np.exp(scores)
        pmat /= pmat.sum(axis=1, keepdims=True)
        out[h] = pmat @ Vdq
    return out


# =============================================================================
#  MXINT8 BASELINE ORACLES  (same math as above, weight in plain MXINT8)
# =============================================================================
def wonly_matmul_mxint8(p, X):
    """W-only MXINT8: X [M,K] -> Y [M,OUT] = X @ dequant_mxint8(W)^T."""
    return np.asarray(X, np.float64) @ dequant_weight_mxint8(p).T


def wa_matmul_mxint8(p, X):
    """W+A MXINT8: int8 weight (no sharing) x MXINT8 activation, per-block dot."""
    qW, sW = weight_int8_mxint8(p)                                # [OUT,nb,32],[OUT,nb]
    qX, sX = quant_act(X, share=False)                            # [M,nb,32],[M,nb]
    intdot = np.einsum("mbk,obk->mbo", qX.astype(np.int64), qW.astype(np.int64))
    return np.einsum("mbo,mb,ob->mo", intdot.astype(np.float64), sX, sW)


def kv_attention_mxint8(Q, pK, pV, causal=True):
    """Attention with K/V stored in plain MXINT8 (mirror of kv_attention)."""
    H, Lq, D = Q.shape
    out = np.zeros((H, Lq, D), dtype=np.float64)
    for h in range(H):
        Kdq = dequant_weight_mxint8(pK["_per"][h])
        Vdq = dequant_weight_mxint8(pV["_per"][h])
        Lk = Kdq.shape[0]
        scores = (np.asarray(Q[h], np.float64) @ Kdq.T) / math.sqrt(D)
        if causal:
            m = np.triu(np.ones((Lq, Lk)), k=1 + (Lk - Lq))
            scores = np.where(m > 0, -np.inf, scores)
        scores = scores - scores.max(axis=1, keepdims=True)
        pmat = np.exp(scores)
        pmat /= pmat.sum(axis=1, keepdims=True)
        out[h] = pmat @ Vdq
    return out


# =============================================================================
#  INDEPENDENT CROSS-CHECK  (re-derivation of the numerics, NOT shared code)
# =============================================================================
def _msaq_signed_ref(x_blocked, u, gs):
    """Independent port of MSAQ_signed (cross-checks decompose+reconstruct)."""
    xf = np.asarray(x_blocked, np.float64)
    B = xf.shape[0]
    amax = np.maximum(np.abs(xf).max(1, keepdims=True), 1e-30)
    sb = 2.0 ** (np.floor(np.log2(amax)) - E_MAX)
    sfu = sb * (1 << u)
    qmax = (1 << (7 - u)) - 1
    qun = np.clip(np.round(xf / sfu), -qmax, qmax)
    xun = qun * sfu
    res = xf - xun
    ng = BLOCK // gs
    ravg = res.reshape(B, ng, gs).mean(2)
    smin, smax = -(1 << (u - 1)), (1 << (u - 1)) - 1
    ri = np.clip(np.round(ravg / sb), smin, smax)
    return xun + np.repeat(ri, gs, 1) * sb
