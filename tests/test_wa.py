# tests/test_wa.py
#
# W+A scope tests:
#   1. oracle      : wa_matmul (per-block int8-dot) == dense product of
#                    dequant(W) and dequant(X_MXINT8)
#   2. kernel gate : ms_lib.ops.wa_gemm vs the oracle  (CUDA / CUTLASS IMMA)
#
# DESIGN NOTE (carried from the kernel file): the W+A activation is quantized to
# plain MXINT8, NOT MSAQ-s shared — the activation is produced and consumed
# inside the matmul (never stored), so sharing its low bits saves no bandwidth
# and gives the same int-dot speed while lowering precision. So the gate uses
# share_act=False to match the kernel. (reference.quant_act exposes share=True
# only to reproduce the fake-quant accuracy numbers.)

import numpy as np
import pytest

from ms_lib.pack import BLOCK, pack_weight, pack_weight_mxint8, dequant_weight
from ms_lib.reference import wa_matmul, wa_matmul_mxint8, quant_act
from conftest import rel_fro, bf16np, REL_FRO_TOL, requires_kernel


# ---- 1. oracle: int8-dot == dense product of the two dequantized operands ---
def test_wa_matmul_oracle(rng, cfg):
    u, gs = cfg
    OUT, K, M = 128, 256, 16
    W = (rng.standard_normal((OUT, K)) * rng.uniform(0.2, 4.0, (OUT, 1))).astype(np.float32)
    p = pack_weight(W, u, gs)
    gold = dequant_weight(p)
    X = rng.standard_normal((M, K)).astype(np.float32)
    Ywa = wa_matmul(p, X, share_act=False)
    qX, sX = quant_act(X, u, gs, share=False)
    Xdq = (qX.astype(np.float64) * sX[:, :, None]).reshape(M, K)
    assert np.allclose(Ywa, Xdq @ gold.T, rtol=1e-9, atol=1e-6)


# ---- 2. kernel gate: CUTLASS IMMA wa_gemm vs the oracle (skips without GPU) --
@requires_kernel
def test_wa_gemm_vs_oracle(rng, cfg):
    import torch
    from ms_lib import ops
    u, gs = cfg
    OUT, K, M = 128, 256, 16
    W = (rng.standard_normal((OUT, K)) * rng.uniform(0.2, 4.0, (OUT, 1))).astype(np.float32)
    p = pack_weight(W, u, gs)
    X = rng.standard_normal((M, K)).astype(np.float32)
    Xt = torch.from_numpy(X).to(torch.bfloat16).cuda()
    got = ops.wa_gemm(p, Xt).float().cpu().numpy()
    ref = wa_matmul(p, bf16np(X), share_act=True)   # MSAQ path: activation is MSAQ-s
    assert rel_fro(got, ref) < REL_FRO_TOL


# ---- 3. W+A decode GEMV: MSAQ-s activation pre-pass + wide-load int-dot ------
@requires_kernel
def test_wa_gemv_vs_oracle(rng, cfg):
    import torch
    from ms_lib import ops
    u, gs = cfg
    OUT, K = 256, 512
    W = (rng.standard_normal((OUT, K)) * rng.uniform(0.2, 4.0, (OUT, 1))).astype(np.float32)
    p = pack_weight(W, u, gs)
    x = rng.standard_normal(K).astype(np.float32)
    xt = torch.from_numpy(x).to(torch.bfloat16).cuda()
    got = ops.wa_gemv(p, xt).float().cpu().numpy()
    ref = wa_matmul(p, bf16np(x)[None], share_act=True)[0]   # MSAQ-s activation
    assert rel_fro(got, ref) < REL_FRO_TOL


# ---- 4. matched MXINT8 W+A GEMV baseline (plain int8 activation + int-dot) ---
@requires_kernel
def test_mxint8_wa_gemv_vs_oracle(rng, cfg):
    import torch
    from ms_lib import ops
    u, gs = cfg
    OUT, K = 256, 512
    W = (rng.standard_normal((OUT, K)) * rng.uniform(0.2, 4.0, (OUT, 1))).astype(np.float32)
    pm = pack_weight_mxint8(W)
    x = rng.standard_normal(K).astype(np.float32)
    xt = torch.from_numpy(x).to(torch.bfloat16).cuda()
    got = ops.mxint8_wa_gemv(pm, xt).float().cpu().numpy()
    ref = wa_matmul_mxint8(pm, bf16np(x)[None])[0]           # plain MXINT8 activation
    assert rel_fro(got, ref) < REL_FRO_TOL