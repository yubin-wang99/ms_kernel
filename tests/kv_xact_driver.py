"""One-shot driver for ncu transaction/sector analysis.
WHICH = gemv_msaq|gemv_mx|kv_msaq|kv_mx ; U, GS env.
"""
import os, torch, numpy as np
from ms_lib import ops; assert ops.available()
from ms_lib.pack import pack_weight, pack_weight_mxint8, pack_kv, pack_kv_mxint8
OPS = torch.ops.msaq
torch.cuda.init()
def cuda(a): return torch.from_numpy(a).cuda()
which = os.environ.get("WHICH", "gemv_msaq")
u = int(os.environ.get("U", "4")); gs = int(os.environ.get("GS", "8"))
rng = np.random.default_rng(0)

if which.startswith("gemv"):
    OUT = K = 4096; nb = K // 32
    W = (rng.standard_normal((OUT, K)) * 0.02).astype(np.float32)
    x = cuda((rng.standard_normal((K,)) * 1.0).astype(np.float32)).to(torch.bfloat16)
    if which == "gemv_msaq":
        p = pack_weight(W, u, gs)
        s, upc, shc = cuda(p["scale_exp"]), cuda(p["upper_cm"]), cuda(p["shared_cm"])
        fn = lambda: OPS.wonly_gemv_wide(x, s, upc, shc, OUT, nb, u, gs)
    else:
        p = pack_weight_mxint8(W)
        s, qw = cuda(p["scale_exp"]), cuda(p["qweight"])
        fn = lambda: OPS.mxint8_gemv(x, s, qw, OUT, nb)
else:  # kv read, single decode step, H=Hkv=8 (no GQA re-read), Lk=4096
    H = Hkv = 8; Lk = 4096; D = 128; nb = D // 32
    Kt = (rng.standard_normal((H, Lk, D)) * 0.5).astype(np.float32)
    Vt = (rng.standard_normal((H, Lk, D)) * 0.5).astype(np.float32)
    q = cuda((rng.standard_normal((H, D)) * 0.5).astype(np.float32)).to(torch.bfloat16)
    if which == "kv_msaq":
        pk, pv = pack_kv(Kt, u, gs), pack_kv(Vt, u, gs)
        a = [cuda(pk[k]) for k in ("scale_exp", "upper", "shared")]
        b = [cuda(pv[k]) for k in ("scale_exp", "upper", "shared")]
        fn = lambda: OPS.kv_decode_attention(q, *a, *b, H, Hkv, Lk, D, nb, u, gs)
    else:
        pk, pv = pack_kv_mxint8(Kt), pack_kv_mxint8(Vt)
        a = [cuda(pk[k]) for k in ("scale_exp", "qweight")]
        b = [cuda(pv[k]) for k in ("scale_exp", "qweight")]
        fn = lambda: OPS.mxint8_kv_decode(q, *a, *b, H, Hkv, Lk, D, nb)

for _ in range(30): fn()
torch.cuda.synchronize(); fn(); torch.cuda.synchronize()
