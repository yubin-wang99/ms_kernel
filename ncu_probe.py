# 최소 probe: 분석할 커널만 1회씩 launch (timing은 ncu가 함)
import sys; sys.path.insert(0, ".")
import numpy as np, torch
from ms_lib.pack import pack_weight, pack_weight_mxint8
from ms_lib import ops  # registers torch.ops.msaq

rng = np.random.default_rng(0)
OUT, K, u, gs = 4096, 4096, 3, 8
W = (rng.standard_normal((OUT, K)) * rng.uniform(0.2, 4.0, (OUT, 1))).astype(np.float32)
pm, px = pack_weight(W, u, gs), pack_weight_mxint8(W)
s  = torch.from_numpy(pm["scale_exp"]).cuda(); up = torch.from_numpy(pm["upper"]).cuda()
sh = torch.from_numpy(pm["shared"]).cuda()
sx = torch.from_numpy(px["scale_exp"]).cuda(); qw = torch.from_numpy(px["qweight"]).cuda()
x  = torch.from_numpy(rng.standard_normal(K).astype(np.float32)).to(torch.bfloat16).cuda()

torch.cuda.synchronize()
torch.ops.msaq.wonly_gemv(x, s, up, sh, OUT, pm["nb"], u, gs)   # MSAQ  -> wonly_gemv_splitk_kernel + gemv_combine_kernel
torch.ops.msaq.mxint8_gemv(x, sx, qw, OUT, px["nb"])            # MXINT8 -> mxint8_gemv_splitk_kernel + mxint8_gemv_combine_kernel
torch.cuda.synchronize()
