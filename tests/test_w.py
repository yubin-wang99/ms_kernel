# tests/test_w.py
#
# W-only scope tests (the simplest, build-first kernel target):
#   1. numerics      : pack.decompose+reconstruct == reference._msaq_signed_ref
#   2. roundtrip     : dequant_weight(pack_weight(W)) == reconstruct(decompose(W))
#   3. oracle        : wonly_matmul self-consistency vs the dense product
#   4. kernel gate   : ms_lib.ops.wonly_gemv / wonly_gemm vs the oracle  (CUDA)
#
# Tests 1-3 are NumPy-only and run anywhere. Test 4 skips without the compiled
# backend (see conftest.requires_kernel).

import numpy as np
import pytest

from ms_lib.pack import (BLOCK, decompose, reconstruct, pack_weight, dequant_weight)
from ms_lib.reference import wonly_matmul, _msaq_signed_ref
from conftest import rel_fro, bf16np, REL_FRO_TOL, requires_kernel


# ---- 1. numerics: shared decompose/reconstruct == independent port ----------
def test_numerics_match_independent_port(rng, cfg):
    u, gs = cfg
    xb = rng.standard_normal((512, BLOCK)) * rng.uniform(0.1, 8.0, (512, 1))
    got = reconstruct(*decompose(xb, u, gs), u, gs)
    ref = _msaq_signed_ref(xb, u, gs)
    assert np.array_equal(got, ref)


# ---- 2. packing roundtrip: unpack(pack(W)) is bit-exact ---------------------
def test_pack_roundtrip_bit_exact(rng, cfg):
    u, gs = cfg
    OUT, K = 256, 512
    W = (rng.standard_normal((OUT, K)) * rng.uniform(0.2, 6.0, (OUT, 1))).astype(np.float32)
    p = pack_weight(W, u, gs)
    gold = reconstruct(*decompose(W.reshape(OUT * (K // BLOCK), BLOCK), u, gs),
                       u, gs).reshape(OUT, K)
    assert np.array_equal(dequant_weight(p), gold)


# ---- 3. oracle self-consistency: matmul == dense reconstructed product ------
def test_wonly_matmul_oracle(rng, cfg):
    u, gs = cfg
    OUT, K, M = 128, 256, 4
    W = (rng.standard_normal((OUT, K)) * rng.uniform(0.2, 4.0, (OUT, 1))).astype(np.float32)
    p = pack_weight(W, u, gs)
    gold = dequant_weight(p)
    X = rng.standard_normal((M, K)).astype(np.float32)
    assert np.allclose(wonly_matmul(p, X), X.astype(np.float64) @ gold.T,
                       rtol=1e-9, atol=1e-7)


# ---- 4. kernel gate: CUDA gemv / gemm vs the oracle (skips without GPU) ------
@requires_kernel
def test_wonly_gemv_vs_oracle(rng, cfg):
    import torch
    from ms_lib import ops
    u, gs = cfg
    OUT, K = 128, 256
    W = (rng.standard_normal((OUT, K)) * rng.uniform(0.2, 4.0, (OUT, 1))).astype(np.float32)
    p = pack_weight(W, u, gs)
    x = rng.standard_normal(K).astype(np.float32)
    xt = torch.from_numpy(x).to(torch.bfloat16).cuda()
    got = ops.wonly_gemv(p, xt).float().cpu().numpy()
    ref = wonly_matmul(p, bf16np(x)[None])[0]
    assert rel_fro(got, ref) < REL_FRO_TOL


@requires_kernel
def test_wonly_gemm_vs_oracle(rng, cfg):
    import torch
    from ms_lib import ops
    u, gs = cfg
    OUT, K, M = 128, 256, 16
    W = (rng.standard_normal((OUT, K)) * rng.uniform(0.2, 4.0, (OUT, 1))).astype(np.float32)
    p = pack_weight(W, u, gs)
    X = rng.standard_normal((M, K)).astype(np.float32)
    Xt = torch.from_numpy(X).to(torch.bfloat16).cuda()
    got = ops.wonly_gemm(p, Xt).float().cpu().numpy()
    ref = wonly_matmul(p, bf16np(X))
    assert rel_fro(got, ref) < REL_FRO_TOL