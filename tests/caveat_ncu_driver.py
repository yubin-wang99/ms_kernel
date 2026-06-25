#!/usr/bin/env python3
"""Run exactly ONE target kernel (warmup is the same kernel) so `ncu -c 1` captures it.
Usage: python tests/caveat_ncu_driver.py <case>
cases: w_dec_ms w_dec_mx w_pre_ms w_pre_mx wa_dec_ms kv_ms kv_mx
"""
import sys, numpy as np, torch
sys.path.insert(0, "."); sys.path.insert(0, "..")
from ms_lib.pack import pack_weight, pack_weight_mxint8, pack_kv, pack_kv_mxint8
from ms_lib import ops; assert ops.available()
OPS = torch.ops.msaq
def C(a): return torch.from_numpy(a).cuda()
case = sys.argv[1]; OUT = K = 4096
rng = np.random.default_rng(0)

if case.startswith("w_") or case.startswith("wa_"):
    W = (rng.standard_normal((OUT, K)) * rng.uniform(0.2, 4.0, (OUT, 1))).astype(np.float32)
    u, gs = (3, 16) if case.startswith("w_") else (2, 8)
    pm, px = pack_weight(W, u, gs), pack_weight_mxint8(W)
    s, upc, shc = C(pm["scale_exp"]), C(pm["upper_cm"]), C(pm["shared_cm"])
    sx, qw, qwc = C(px["scale_exp"]), C(px["qweight"]), C(px["qweight_cm"])
    nb, nbx = pm["nb"], px["nb"]
    M = 1 if "_dec_" in case else 512
    X = C(rng.standard_normal((M, K)).astype(np.float32)).to(torch.bfloat16); x1 = X[0].contiguous()
    if case == "w_dec_ms":  fn = lambda: OPS.wonly_gemv_wide(x1, s, upc, shc, OUT, nb, u, gs)
    elif case == "w_dec_mx": fn = lambda: OPS.mxint8_gemv(x1, sx, qw, OUT, nbx)
    elif case == "w_pre_ms": fn = lambda: OPS.wonly_gemm_tc(X, s, upc, shc, M, OUT, K, nb, u, gs)
    elif case == "w_pre_mx": fn = lambda: OPS.mxint8_gemm(X, sx, qw, M, OUT, K, nbx)
    elif case == "wa_dec_ms": fn = lambda: OPS.wa_gemv(x1, s, upc, shc, OUT, nb, u, gs)
    else: raise SystemExit("bad case")
elif case.startswith("kv_"):
    u, gs, Hq, Hkv, Lk, D = 4, 2, 32, 8, 16384, 128
    Kf = (rng.standard_normal((Hkv, Lk, D)) * 0.5).astype(np.float32)
    Vf = (rng.standard_normal((Hkv, Lk, D)) * 0.5).astype(np.float32)
    pmK, pmV = pack_kv(Kf, u, gs), pack_kv(Vf, u, gs)
    pxK, pxV = pack_kv_mxint8(Kf), pack_kv_mxint8(Vf)
    ks, ku, kh = C(pmK["scale_exp"]), C(pmK["upper"]), C(pmK["shared"])
    vs, vu, vh = C(pmV["scale_exp"]), C(pmV["upper"]), C(pmV["shared"])
    kxs, kxq = C(pxK["scale_exp"]), C(pxK["qweight"]); vxs, vxq = C(pxV["scale_exp"]), C(pxV["qweight"])
    nb = pmK["nb"]; q = C(rng.standard_normal((Hq, D)).astype(np.float32)).to(torch.bfloat16)
    if case == "kv_ms": fn = lambda: OPS.kv_decode_attention(q, ks, ku, kh, vs, vu, vh, Hq, Hkv, Lk, D, nb, u, gs)
    elif case == "kv_mx": fn = lambda: OPS.mxint8_kv_decode(q, kxs, kxq, vxs, vxq, Hq, Hkv, Lk, D, nb)
    else: raise SystemExit("bad case")
else:
    raise SystemExit("bad case")

for _ in range(3):
    fn(); torch.cuda.synchronize()
