# tests/test_kv.py
#
# KV-cache scope tests (highest-risk kernel — validate the oracle first):
#   1. oracle      : kv_attention (causal online-softmax on dequant K/V) is
#                    self-consistent with the direct dequant-then-attend result
#   2. kernel gate : ms_lib.ops.kv_decode_attention vs the oracle  (CUDA flash)
#
# KV packs each head as [L, D] (blocks along head_dim, token innermost), so the
# kernel's per-key dequant reuses the exact W-only unpack path.

import math
import numpy as np
import pytest

from ms_lib.pack import pack_kv, dequant_weight
from ms_lib.reference import kv_attention
from conftest import rel_fro, bf16np, REL_FRO_TOL, requires_kernel


# ---- 1. oracle self-consistency: fused mirror == direct dequant attention ---
def test_kv_attention_oracle(rng, cfg):
    u, gs = cfg
    H, Lk, D = 3, 40, 64                          # head_dim 64 = 2 blocks
    Kt = (rng.standard_normal((H, Lk, D)) * 0.5).astype(np.float32)
    Vt = (rng.standard_normal((H, Lk, D)) * 0.5).astype(np.float32)
    Q = (rng.standard_normal((H, Lk, D)) * 0.5).astype(np.float32)   # prefill (Lq=Lk)
    pK, pV = pack_kv(Kt, u, gs), pack_kv(Vt, u, gs)
    att = kv_attention(Q, pK, pV, causal=True)

    ref = np.zeros_like(att)
    for h in range(H):
        Kd = dequant_weight(pK["_per"][h])
        Vd = dequant_weight(pV["_per"][h])
        sc = (Q[h].astype(np.float64) @ Kd.T) / math.sqrt(D)
        msk = np.triu(np.ones((Lk, Lk)), k=1)
        sc = np.where(msk > 0, -np.inf, sc)
        sc -= sc.max(1, keepdims=True)
        pm = np.exp(sc)
        pm /= pm.sum(1, keepdims=True)
        ref[h] = pm @ Vd
    assert np.allclose(att, ref, rtol=1e-9, atol=1e-7)


# ---- 2. kernel gate: CUDA flash-decode attention vs the oracle (skips w/o GPU)
@requires_kernel
def test_kv_decode_attention_vs_oracle(rng, cfg):
    import torch
    from ms_lib import ops
    u, gs = cfg
    H, Lk, D = 4, 40, 64
    Kt = (rng.standard_normal((H, Lk, D)) * 0.5).astype(np.float32)
    Vt = (rng.standard_normal((H, Lk, D)) * 0.5).astype(np.float32)
    Q = (rng.standard_normal((H, D)) * 0.5).astype(np.float32)       # single decode step
    pK, pV = pack_kv(Kt, u, gs), pack_kv(Vt, u, gs)
    qt = torch.from_numpy(Q).to(torch.bfloat16).cuda()
    got = ops.kv_decode_attention(qt, pK, pV).float().cpu().numpy()
    ref = kv_attention(bf16np(Q)[:, None, :], pK, pV, causal=True)[:, 0, :]
    assert rel_fro(got, ref) < REL_FRO_TOL


# ---- 3. KV write (prefill) kernel: planes byte-exact to pack_kv + end-to-end --
@requires_kernel
def test_kv_write_vs_pack(rng, cfg):
    import torch
    u, gs = cfg
    H, L, D = 4, 48, 128
    K = (rng.standard_normal((H, L, D)) * rng.uniform(0.3, 3, (H, 1, 1))).astype(np.float32)
    Kb = torch.from_numpy(K).to(torch.bfloat16).cuda()
    nb = D // 32
    se, up, sh = torch.ops.msaq.kv_write(Kb, H, L, D, nb, u, gs)
    p = pack_kv(Kb.float().cpu().numpy(), u, gs)           # MSAQ-s planes (oracle)
    assert np.array_equal(se.cpu().numpy(), p["scale_exp"])    # byte-exact
    assert np.array_equal(up.cpu().numpy(), p["upper"])
    assert np.array_equal(sh.cpu().numpy(), p["shared"])


@requires_kernel
def test_kv_write_then_decode_vs_oracle(rng, cfg):
    import torch
    from ms_lib import ops
    u, gs = cfg
    H, Lk, D = 8, 96, 128
    K = (rng.standard_normal((H, Lk, D)) * 0.5).astype(np.float32)
    V = (rng.standard_normal((H, Lk, D)) * 0.5).astype(np.float32)
    Q = (rng.standard_normal((H, D)) * 0.5).astype(np.float32)
    Kb, Vb = (torch.from_numpy(a).to(torch.bfloat16).cuda() for a in (K, V))
    qt, nb = torch.from_numpy(Q).to(torch.bfloat16).cuda(), D // 32
    ks, ku, kh = torch.ops.msaq.kv_write(Kb, H, Lk, D, nb, u, gs)   # write cache via kernel
    vs, vu, vh = torch.ops.msaq.kv_write(Vb, H, Lk, D, nb, u, gs)
    got = torch.ops.msaq.kv_decode_attention(qt, ks, ku, kh, vs, vu, vh,
                                             H, H, Lk, D, nb, u, gs).float().cpu().numpy()
    pK = pack_kv(Kb.float().cpu().numpy(), u, gs)
    pV = pack_kv(Vb.float().cpu().numpy(), u, gs)
    ref = kv_attention(bf16np(Q)[:, None, :], pK, pV, causal=True)[:, 0, :]
    assert rel_fro(got, ref) < REL_FRO_TOL


# ---- 3b. GQA decode: Hq query heads attend to Hkv kv heads (q head h -> h//g) --
@requires_kernel
def test_kv_decode_gqa_vs_oracle(rng, cfg):
    import torch
    u, gs = cfg
    Hq, Hkv, Lk, D = 8, 2, 96, 128                # group = 4
    g = Hq // Hkv
    K = (rng.standard_normal((Hkv, Lk, D)) * 0.5).astype(np.float32)
    V = (rng.standard_normal((Hkv, Lk, D)) * 0.5).astype(np.float32)
    Q = (rng.standard_normal((Hq, D)) * 0.5).astype(np.float32)
    pK, pV = pack_kv(K, u, gs), pack_kv(V, u, gs)
    ks, ku, kh = (torch.from_numpy(pK[k]).cuda() for k in ("scale_exp", "upper", "shared"))
    vs, vu, vh = (torch.from_numpy(pV[k]).cuda() for k in ("scale_exp", "upper", "shared"))
    qt, nb = torch.from_numpy(Q).to(torch.bfloat16).cuda(), D // 32
    got = torch.ops.msaq.kv_decode_attention(qt, ks, ku, kh, vs, vu, vh,
                                             Hq, Hkv, Lk, D, nb, u, gs).float().cpu().numpy()
    # reference: each q head h attends to kv head h//g (single-query causal == full)
    ref = np.zeros((Hq, D), np.float64)
    for h in range(Hq):
        Kd, Vd = dequant_weight(pK["_per"][h // g]), dequant_weight(pV["_per"][h // g])
        sc = (bf16np(Q[h]).astype(np.float64) @ Kd.T) / math.sqrt(D)
        p = np.exp(sc - sc.max()); p /= p.sum()
        ref[h] = p @ Vd
    assert rel_fro(got, ref) < REL_FRO_TOL


# ---- 4. KV append (decode): per-token in-place quantize, byte-exact to write ---
@requires_kernel
def test_kv_append_vs_pack(rng, cfg):
    import torch
    u, gs = cfg
    H, L, D = 4, 48, 128
    K = (rng.standard_normal((H, L, D)) * rng.uniform(0.3, 3, (H, 1, 1))).astype(np.float32)
    Kb = torch.from_numpy(K).to(torch.bfloat16).cuda()
    nb, wbits = D // 32, 8 - u
    UB, SB = 32 * wbits // 8, ((32 // gs) * u + 7) // 8
    # pre-allocate the cache (capacity == final length so the slot stride matches)
    se = torch.empty((H, nb, L), dtype=torch.int8, device="cuda")
    up = torch.empty((H, nb, L, UB), dtype=torch.uint8, device="cuda")
    sh = torch.empty((H, nb, L, SB), dtype=torch.uint8, device="cuda")
    for pos in range(L):                                  # decode loop: one token / step
        torch.ops.msaq.kv_append(Kb[:, pos, :].contiguous(), se, up, sh,
                                 H, D, nb, pos, L, u, gs)
    p = pack_kv(Kb.float().cpu().numpy(), u, gs)          # whole-tensor write (oracle)
    assert np.array_equal(se.cpu().numpy(), p["scale_exp"])    # byte-exact to write
    assert np.array_equal(up.cpu().numpy(), p["upper"])
    assert np.array_equal(sh.cpu().numpy(), p["shared"])


@requires_kernel
def test_kv_append_then_decode_vs_oracle(rng, cfg):
    import torch
    u, gs = cfg
    H, Lk, D = 8, 96, 128
    K = (rng.standard_normal((H, Lk, D)) * 0.5).astype(np.float32)
    V = (rng.standard_normal((H, Lk, D)) * 0.5).astype(np.float32)
    Q = (rng.standard_normal((H, D)) * 0.5).astype(np.float32)
    Kb, Vb = (torch.from_numpy(a).to(torch.bfloat16).cuda() for a in (K, V))
    qt, nb, wbits = torch.from_numpy(Q).to(torch.bfloat16).cuda(), D // 32, 8 - u
    UB, SB = 32 * wbits // 8, ((32 // gs) * u + 7) // 8
    cache = lambda: (torch.empty((H, nb, Lk), dtype=torch.int8, device="cuda"),
                     torch.empty((H, nb, Lk, UB), dtype=torch.uint8, device="cuda"),
                     torch.empty((H, nb, Lk, SB), dtype=torch.uint8, device="cuda"))
    ks, ku, kh = cache()
    vs, vu, vh = cache()
    for pos in range(Lk):                                 # build both caches token-by-token
        torch.ops.msaq.kv_append(Kb[:, pos, :].contiguous(), ks, ku, kh, H, D, nb, pos, Lk, u, gs)
        torch.ops.msaq.kv_append(Vb[:, pos, :].contiguous(), vs, vu, vh, H, D, nb, pos, Lk, u, gs)
    got = torch.ops.msaq.kv_decode_attention(qt, ks, ku, kh, vs, vu, vh,
                                             H, H, Lk, D, nb, u, gs).float().cpu().numpy()
    pK = pack_kv(Kb.float().cpu().numpy(), u, gs)
    pV = pack_kv(Vb.float().cpu().numpy(), u, gs)
    ref = kv_attention(bf16np(Q)[:, None, :], pK, pV, causal=True)[:, 0, :]
    assert rel_fro(got, ref) < REL_FRO_TOL