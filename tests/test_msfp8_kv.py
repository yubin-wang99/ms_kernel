# tests/test_msfp8_kv.py
#
# MXFP8-MSAQ (E3M4) KV-cache kernels — 1:1 analogs of the MXINT8-MSAQ KV kernels
# (test_kv.py). E3M4 has mb=4, so the shared-mantissa width u in {1,2,3} (u4 is the
# INT-only config). Same plane layout as the INT path; the only difference is the
# per-element FP8 decode (and the V P·V runs in float, since int8-staging is INT-only).
#
#   1. oracle      : kv_attention_msfp8 self-consistent with direct dequant attention
#   2. decode      : ops.msfp8_kv_decode (CUDA flash) vs the oracle
#   3. GQA / batched / write / append / kdot   — mirror test_kv.py

import math
import numpy as np
import pytest

from ms_lib.pack import pack_kv_msfp8, dequant_weight_msfp8
from ms_lib.reference import kv_attention_msfp8
from conftest import rel_fro, bf16np, REL_FRO_TOL, requires_kernel

# E3M4-valid (u, gs): u in {1,2,3}; gs a power of two
FP8_CFGS = [(1, 8), (2, 4), (3, 4), (2, 8), (3, 8), (1, 16)]
FP8_IDS = [f"u{u}_gs{gs}" for (u, gs) in FP8_CFGS]


@pytest.fixture(params=FP8_CFGS, ids=FP8_IDS)
def fcfg(request):
    return request.param


# ---- 1. oracle self-consistency: fused mirror == direct dequant attention ---
def test_msfp8_kv_oracle(rng, fcfg):
    u, gs = fcfg
    H, Lk, D = 3, 40, 64                           # head_dim 64 = 2 blocks
    Kt = (rng.standard_normal((H, Lk, D)) * 0.5).astype(np.float32)
    Vt = (rng.standard_normal((H, Lk, D)) * 0.5).astype(np.float32)
    Q = (rng.standard_normal((H, Lk, D)) * 0.5).astype(np.float32)
    pK, pV = pack_kv_msfp8(Kt, u, gs), pack_kv_msfp8(Vt, u, gs)
    att = kv_attention_msfp8(Q, pK, pV, causal=True)
    ref = np.zeros_like(att)
    for h in range(H):
        Kd, Vd = dequant_weight_msfp8(pK["_per"][h]), dequant_weight_msfp8(pV["_per"][h])
        sc = (Q[h].astype(np.float64) @ Kd.T) / math.sqrt(D)
        sc = np.where(np.triu(np.ones((Lk, Lk)), k=1) > 0, -np.inf, sc)
        sc -= sc.max(1, keepdims=True)
        pm = np.exp(sc); pm /= pm.sum(1, keepdims=True)
        ref[h] = pm @ Vd
    assert np.allclose(att, ref, rtol=1e-9, atol=1e-7)


# ---- 2. kernel gate: CUDA flash-decode attention vs the oracle --------------
@requires_kernel
def test_msfp8_kv_decode_vs_oracle(rng, fcfg):
    import torch
    from ms_lib import ops
    u, gs = fcfg
    H, Lk, D = 4, 40, 64
    Kt = (rng.standard_normal((H, Lk, D)) * 0.5).astype(np.float32)
    Vt = (rng.standard_normal((H, Lk, D)) * 0.5).astype(np.float32)
    Q = (rng.standard_normal((H, D)) * 0.5).astype(np.float32)
    pK, pV = pack_kv_msfp8(Kt, u, gs), pack_kv_msfp8(Vt, u, gs)
    qt = torch.from_numpy(Q).to(torch.bfloat16).cuda()
    got = ops.msfp8_kv_decode(qt, pK, pV).float().cpu().numpy()
    ref = kv_attention_msfp8(bf16np(Q)[:, None, :], pK, pV, causal=True)[:, 0, :]
    assert rel_fro(got, ref) < REL_FRO_TOL


# ---- 3. GQA decode: Hq query heads attend to Hkv kv heads (h -> h//g) -------
@requires_kernel
def test_msfp8_kv_decode_gqa_vs_oracle(rng, fcfg):
    import torch
    u, gs = fcfg
    Hq, Hkv, Lk, D = 8, 2, 96, 128                 # group = 4
    g = Hq // Hkv
    K = (rng.standard_normal((Hkv, Lk, D)) * 0.5).astype(np.float32)
    V = (rng.standard_normal((Hkv, Lk, D)) * 0.5).astype(np.float32)
    Q = (rng.standard_normal((Hq, D)) * 0.5).astype(np.float32)
    pK, pV = pack_kv_msfp8(K, u, gs), pack_kv_msfp8(V, u, gs)
    ks, ku, kh = (torch.from_numpy(pK[k]).cuda() for k in ("scale_exp", "upper", "shared"))
    vs, vu, vh = (torch.from_numpy(pV[k]).cuda() for k in ("scale_exp", "upper", "shared"))
    qt, nb = torch.from_numpy(Q).to(torch.bfloat16).cuda(), D // 32
    got = torch.ops.msaq.msfp8_kv_decode_attention(qt, ks, ku, kh, vs, vu, vh,
                                                   Hq, Hkv, Lk, D, nb, u, gs).float().cpu().numpy()
    ref = np.zeros((Hq, D), np.float64)
    for h in range(Hq):
        Kd, Vd = dequant_weight_msfp8(pK["_per"][h // g]), dequant_weight_msfp8(pV["_per"][h // g])
        sc = (bf16np(Q[h]).astype(np.float64) @ Kd.T) / math.sqrt(D)
        p = np.exp(sc - sc.max()); p /= p.sum()
        ref[h] = p @ Vd
    assert rel_fro(got, ref) < REL_FRO_TOL


# ---- 4. batched decode: grid.z = batch -------------------------------------
@requires_kernel
def test_msfp8_kv_decode_batched_vs_oracle(rng, fcfg):
    import torch
    u, gs = fcfg
    B, H, Lk, D = 3, 4, 64, 128
    nb = D // 32
    outs, refs = [], []
    ks_l, ku_l, kh_l, vs_l, vu_l, vh_l, q_l = [], [], [], [], [], [], []
    for bi in range(B):
        K = (rng.standard_normal((H, Lk, D)) * 0.5).astype(np.float32)
        V = (rng.standard_normal((H, Lk, D)) * 0.5).astype(np.float32)
        Q = (rng.standard_normal((H, D)) * 0.5).astype(np.float32)
        pK, pV = pack_kv_msfp8(K, u, gs), pack_kv_msfp8(V, u, gs)
        ks_l.append(pK["scale_exp"]); ku_l.append(pK["upper"]); kh_l.append(pK["shared"])
        vs_l.append(pV["scale_exp"]); vu_l.append(pV["upper"]); vh_l.append(pV["shared"])
        q_l.append(Q)
        refs.append(kv_attention_msfp8(bf16np(Q)[:, None, :], pK, pV, causal=True)[:, 0, :])
    st = lambda L: torch.from_numpy(np.stack(L)).cuda()
    qt = torch.from_numpy(np.stack(q_l)).to(torch.bfloat16).cuda()
    got = torch.ops.msaq.msfp8_kv_decode_attention_batched(
        qt, st(ks_l), st(ku_l), st(kh_l), st(vs_l), st(vu_l), st(vh_l),
        B, H, H, Lk, D, nb, u, gs).float().cpu().numpy()
    assert rel_fro(got, np.stack(refs)) < REL_FRO_TOL


# ---- 5. KV write (prefill) -> decode roundtrip vs oracle --------------------
@requires_kernel
def test_msfp8_kv_write_then_decode_vs_oracle(rng, fcfg):
    import torch
    u, gs = fcfg
    H, Lk, D = 8, 96, 128
    K = (rng.standard_normal((H, Lk, D)) * 0.5).astype(np.float32)
    V = (rng.standard_normal((H, Lk, D)) * 0.5).astype(np.float32)
    Q = (rng.standard_normal((H, D)) * 0.5).astype(np.float32)
    Kb, Vb = (torch.from_numpy(a).to(torch.bfloat16).cuda() for a in (K, V))
    qt, nb = torch.from_numpy(Q).to(torch.bfloat16).cuda(), D // 32
    ks, ku, kh = torch.ops.msaq.msfp8_kv_write(Kb, H, Lk, D, nb, u, gs)
    vs, vu, vh = torch.ops.msaq.msfp8_kv_write(Vb, H, Lk, D, nb, u, gs)
    got = torch.ops.msaq.msfp8_kv_decode_attention(qt, ks, ku, kh, vs, vu, vh,
                                                   H, H, Lk, D, nb, u, gs).float().cpu().numpy()
    pK = pack_kv_msfp8(Kb.float().cpu().numpy(), u, gs)
    pV = pack_kv_msfp8(Vb.float().cpu().numpy(), u, gs)
    ref = kv_attention_msfp8(bf16np(Q)[:, None, :], pK, pV, causal=True)[:, 0, :]
    assert rel_fro(got, ref) < REL_FRO_TOL


# ---- 5b. GPU write planes == CPU pack_kv_msfp8 (byte-exact encode) ----------
@requires_kernel
def test_msfp8_kv_write_vs_pack(rng, fcfg):
    import torch
    u, gs = fcfg
    H, L, D = 4, 48, 128
    K = (rng.standard_normal((H, L, D)) * rng.uniform(0.3, 3, (H, 1, 1))).astype(np.float32)
    Kb = torch.from_numpy(K).to(torch.bfloat16).cuda()
    nb = D // 32
    se, up, sh = torch.ops.msaq.msfp8_kv_write(Kb, H, L, D, nb, u, gs)
    p = pack_kv_msfp8(Kb.float().cpu().numpy(), u, gs)
    # GPU encoder is fp32; the CPU pack is fp64 -> allow a tiny mismatch fraction at
    # rounding boundaries (decode roundtrip in 5/6 is the strict correctness gate).
    for name, got in (("scale_exp", se), ("upper", up), ("shared", sh)):
        a, b = got.cpu().numpy(), p[name]
        frac = np.mean(a != b)
        assert frac < 5e-3, f"{name}: {frac:.2%} bytes differ (u{u}/gs{gs})"


# ---- 6. KV append (decode loop) -> decode roundtrip + lcap-stride invariance -
@requires_kernel
def test_msfp8_kv_append_then_decode_vs_oracle(rng, fcfg):
    import torch
    u, gs = fcfg
    H, Lk, D = 8, 96, 128
    nb, wbits = D // 32, 8 - u
    UB, SB = 32 * wbits // 8, ((32 // gs) * u + 7) // 8
    K = (rng.standard_normal((H, Lk, D)) * 0.5).astype(np.float32)
    V = (rng.standard_normal((H, Lk, D)) * 0.5).astype(np.float32)
    Q = (rng.standard_normal((H, D)) * 0.5).astype(np.float32)
    Kb, Vb = (torch.from_numpy(a).to(torch.bfloat16).cuda() for a in (K, V))
    qt = torch.from_numpy(Q).to(torch.bfloat16).cuda()

    def cache(Lcap):
        return (torch.zeros((H, nb, Lcap), dtype=torch.int8, device="cuda"),
                torch.zeros((H, nb, Lcap, UB), dtype=torch.uint8, device="cuda"),
                torch.zeros((H, nb, Lcap, SB), dtype=torch.uint8, device="cuda"))

    def build(Lcap):
        ks, ku, kh = cache(Lcap); vs, vu, vh = cache(Lcap)
        for pos in range(Lk):
            torch.ops.msaq.msfp8_kv_append(Kb[:, pos, :].contiguous(), ks, ku, kh, H, D, nb, pos, Lcap, u, gs)
            torch.ops.msaq.msfp8_kv_append(Vb[:, pos, :].contiguous(), vs, vu, vh, H, D, nb, pos, Lcap, u, gs)
        return ks, ku, kh, vs, vu, vh

    ks, ku, kh, vs, vu, vh = build(Lk)
    got = torch.ops.msaq.msfp8_kv_decode_attention(qt, ks, ku, kh, vs, vu, vh,
                                                   H, H, Lk, D, nb, u, gs).float().cpu().numpy()
    pK = pack_kv_msfp8(Kb.float().cpu().numpy(), u, gs)
    pV = pack_kv_msfp8(Vb.float().cpu().numpy(), u, gs)
    ref = kv_attention_msfp8(bf16np(Q)[:, None, :], pK, pV, causal=True)[:, 0, :]
    assert rel_fro(got, ref) < REL_FRO_TOL
    # capacity-invariance: attending Lk keys is independent of cache stride Lcap
    ks2, ku2, kh2, vs2, vu2, vh2 = build(256)
    got2 = torch.ops.msaq.msfp8_kv_decode_attention(qt, ks2, ku2, kh2, vs2, vu2, vh2,
                                                    H, H, Lk, D, nb, u, gs, 256).float().cpu().numpy()
    assert np.array_equal(got, got2)


# ---- 7. K-dot probe vs oracle (raw q·K per key, no softmax) -----------------
@requires_kernel
def test_msfp8_kv_kdot_vs_oracle(rng, fcfg):
    import torch
    u, gs = fcfg
    H, Lk, D = 4, 64, 128
    nb = D // 32
    K = (rng.standard_normal((H, Lk, D)) * 0.5).astype(np.float32)
    Q = (rng.standard_normal((H, D)) * 0.5).astype(np.float32)
    pK = pack_kv_msfp8(K, u, gs)
    ks, ku, kh = (torch.from_numpy(pK[k]).cuda() for k in ("scale_exp", "upper", "shared"))
    qt = torch.from_numpy(Q).to(torch.bfloat16).cuda()
    got = torch.ops.msaq.msfp8_kv_kdot(qt, ks, ku, kh, 1, H, H, Lk, D, nb, u, gs).float().cpu().numpy()[0]
    ref = np.zeros((H, Lk), np.float64)
    for h in range(H):
        Kd = dequant_weight_msfp8(pK["_per"][h])
        ref[h] = bf16np(Q[h]).astype(np.float64) @ Kd.T
    assert rel_fro(got, ref) < REL_FRO_TOL
