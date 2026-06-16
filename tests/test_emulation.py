# tests/test_emulation.py
#
# CPU LOGIC GATE for the CUDA kernels — runs WITHOUT a GPU.
#
# Each test below mirrors the EXACT arithmetic and byte-addressing of a .cu
# kernel, independently re-deriving the MSAQ-s unpack straight from the raw
# packed planes (p["upper"]/["shared"]/["scale_exp"] — the same tensors
# ms_lib.ops hands the kernel), NOT via pack.dequant_weight. The numpy index
# computed here, e.g. upper[blk, bi, o], is the same C-contiguous flat offset
# (blk*UB+bi)*OUT+o that ms::unpack_ms_weight_elem computes — so a match here
# proves the kernel's flat-index math and reconstruction logic against the
# oracle (ms_lib.reference). Everything is fp64: this isolates LOGIC; the
# fp32-accumulate / bf16 / CUDA-exec gap is what the GPU *_vs_oracle gates
# (test_w/test_wa/test_kv) measure on the RTX 3090.
#
# Pass bar: rel_fro < 1e-9 (exact-arithmetic agreement up to fp64 rounding).

import math
import numpy as np

from ms_lib.pack import pack_weight, pack_kv, BLOCK, E_MAX
from ms_lib.reference import wonly_matmul, wa_matmul, kv_attention
from conftest import rel_fro

LOGIC_TOL = 1e-9


def _sx(v, bits):                       # sign-extend the low `bits` of v
    s = 1 << (bits - 1)
    return (int(v) ^ s) - s


def _unpack(upper, shared, blk, col, k, u, gs, UB, SB):
    """Mirror of ms::unpack_ms_weight_elem (and unpack_ms_kv_elem) on planes
    shaped [nb, UB|SB, COL] (COL = OUT for weights, L for KV)."""
    wb = 8 - u
    bit0 = k * wb; bi = bit0 >> 3; off = bit0 & 7
    code = int(upper[blk, bi, col]) >> off
    if off + wb > 8 and bi + 1 < UB:
        code |= int(upper[blk, bi + 1, col]) << (8 - off)
    code = _sx(code & ((1 << wb) - 1), wb)
    g = k // gs; sbit = g * u; sbi = sbit >> 3; soff = sbit & 7
    sc = int(shared[blk, sbi, col]) >> soff
    if soff + u > 8 and sbi + 1 < SB:
        sc |= int(shared[blk, sbi + 1, col]) << (8 - soff)
    sc = _sx(sc & ((1 << u) - 1), u)
    return code * (1 << u) + sc


def _unpack_kv(upper, shared, blk, key, k, u, gs, UB, SB):
    """Mirror of ms::unpack_ms_kv_elem on TOKEN-MAJOR KV planes shaped
    [nb, L, UB|SB] (BYTES innermost; Stage 4a). Same codes as _unpack, but the
    byte for a given key is at [blk, key, bi] (flat (blk*L+key)*UB+bi) — exactly
    the kernel's base + key*UB + bi addressing."""
    wb = 8 - u
    bit0 = k * wb; bi = bit0 >> 3; off = bit0 & 7
    code = int(upper[blk, key, bi]) >> off
    if off + wb > 8 and bi + 1 < UB:
        code |= int(upper[blk, key, bi + 1]) << (8 - off)
    code = _sx(code & ((1 << wb) - 1), wb)
    g = k // gs; sbit = g * u; sbi = sbit >> 3; soff = sbit & 7
    sc = int(shared[blk, key, sbi]) >> soff
    if soff + u > 8 and sbi + 1 < SB:
        sc |= int(shared[blk, key, sbi + 1]) << (8 - soff)
    sc = _sx(sc & ((1 << u) - 1), u)
    return code * (1 << u) + sc


def _wpack(rng, u, gs, OUT=128, K=256):
    W = (rng.standard_normal((OUT, K)) * rng.uniform(0.2, 4.0, (OUT, 1))).astype(np.float32)
    return W, pack_weight(W, u, gs)


# ---- w_gemv: y[o] = sum_{blk,k} unpack*2^exp * x[k]  (M=1) -------------------
def test_w_gemv_logic(rng, cfg):
    u, gs = cfg
    _, p = _wpack(rng, u, gs)
    up, sh, se = p["upper"], p["shared"], p["scale_exp"].astype(np.int64)
    UB, SB, nb, OUT, K = p["UB"], p["SB"], p["nb"], p["OUT"], p["K"]
    x = rng.standard_normal(K).astype(np.float64)
    y = np.zeros(OUT)
    for o in range(OUT):
        acc = 0.0
        for blk in range(nb):
            scale = 2.0 ** int(se[blk, o])
            for k in range(BLOCK):
                acc += _unpack(up, sh, blk, o, k, u, gs, UB, SB) * scale * x[blk * BLOCK + k]
        y[o] = acc
    assert rel_fro(y, wonly_matmul(p, x[None])[0]) < LOGIC_TOL


# ---- wonly_gemm: same reconstruction, full M --------------------------------
def test_wonly_gemm_logic(rng, cfg):
    u, gs = cfg
    M = 16
    _, p = _wpack(rng, u, gs)
    up, sh, se = p["upper"], p["shared"], p["scale_exp"].astype(np.int64)
    UB, SB, nb, OUT, K = p["UB"], p["SB"], p["nb"], p["OUT"], p["K"]
    X = rng.standard_normal((M, K)).astype(np.float64)
    Y = np.zeros((M, OUT))
    for m in range(M):
        for o in range(OUT):
            acc = 0.0
            for blk in range(nb):
                scale = 2.0 ** int(se[blk, o])
                for k in range(BLOCK):
                    acc += _unpack(up, sh, blk, o, k, u, gs, UB, SB) * scale * X[m, blk * BLOCK + k]
            Y[m, o] = acc
    assert rel_fro(Y, wonly_matmul(p, X)) < LOGIC_TOL


# ---- wa_gemm: on-the-fly MXINT8 activation quant + per-block int dot ---------
def test_wa_gemm_logic(rng, cfg):
    u, gs = cfg
    M = 16
    _, p = _wpack(rng, u, gs)
    up, sh, se = p["upper"], p["shared"], p["scale_exp"].astype(np.int64)
    UB, SB, nb, OUT, K = p["UB"], p["SB"], p["nb"], p["OUT"], p["K"]
    X = rng.standard_normal((M, K)).astype(np.float64)
    Y = np.zeros((M, OUT))
    for m in range(M):
        for o in range(OUT):
            acc = 0.0
            for blk in range(nb):
                xb = X[m, blk * BLOCK:(blk + 1) * BLOCK]
                amax = max(np.abs(xb).max(), 1e-30)
                ea = max(min(math.floor(math.log2(amax)) - E_MAX, 127), -127)
                sa = 2.0 ** ea
                sw = 2.0 ** int(se[blk, o])
                idot = 0
                for k in range(BLOCK):
                    qx = int(np.clip(np.round(xb[k] / sa), -127, 127))
                    idot += qx * _unpack(up, sh, blk, o, k, u, gs, UB, SB)
                acc += idot * sa * sw
            Y[m, o] = acc
    assert rel_fro(Y, wa_matmul(p, X, share_act=False)) < LOGIC_TOL


# ---- kv_decode: online softmax with fused K/V unpack (token-major [H,nb,L,*])
def test_kv_decode_logic(rng, cfg):
    u, gs = cfg
    H, Lk, D = 4, 40, 64
    Kt = (rng.standard_normal((H, Lk, D)) * 0.5).astype(np.float32)
    Vt = (rng.standard_normal((H, Lk, D)) * 0.5).astype(np.float32)
    Q = (rng.standard_normal((H, D)) * 0.5).astype(np.float64)
    pK, pV = pack_kv(Kt, u, gs), pack_kv(Vt, u, gs)
    UB, SB = pK["UB"], pK["SB"]
    sm = 1.0 / math.sqrt(D)
    out = np.zeros((H, D))
    for h in range(H):
        kse, vse = pK["scale_exp"][h].astype(np.int64), pV["scale_exp"][h].astype(np.int64)
        ku, kh = pK["upper"][h], pK["shared"][h]
        vu, vh = pV["upper"][h], pV["shared"][h]
        for e in range(D):                                  # one "thread" / head_dim elem
            blk, k = e // BLOCK, e % BLOCK
            m_i, l_i, acc = -np.inf, 0.0, 0.0
            for j in range(Lk):
                # score_j = sm * sum_ee q[ee] * Kdq[ee, j]   (the block reduction)
                s = 0.0
                for ee in range(D):
                    b2, k2 = ee // BLOCK, ee % BLOCK
                    s += Q[h, ee] * (_unpack_kv(ku, kh, b2, j, k2, u, gs, UB, SB) * 2.0 ** int(kse[b2, j]))
                score = s * sm
                m_new = max(m_i, score)
                alpha = math.exp(m_i - m_new)
                p_ = math.exp(score - m_new)
                vdq = _unpack_kv(vu, vh, blk, j, k, u, gs, UB, SB) * 2.0 ** int(vse[blk, j])
                l_i = l_i * alpha + p_
                acc = acc * alpha + p_ * vdq
                m_i = m_new
            out[h, e] = acc / l_i
    assert rel_fro(out, kv_attention(Q[:, None, :], pK, pV, causal=True)[:, 0, :]) < LOGIC_TOL