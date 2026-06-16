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