# ncu probe: launch the SCALAR tiled W-only prefill GEMM once so Nsight Compute
# can profile occupancy of the 64x64 tile (cfg=1, picked for M<256).
#   MS_GEMM_SCALAR=1  -> wonly_gemm_cm_cuda -> DISPATCH_TILE(wonly_gemm_tiled_cm)
#   M<256 (or MS_TILE_CFG=1) -> tile 64x64, reg 4x4, 256 threads/block
import sys, os; sys.path.insert(0, ".")
import numpy as np, torch
from ms_lib.pack import pack_weight
from ms_lib import ops

rng = np.random.default_rng(0)
OUT, K, u, gs = 4096, 4096, 3, 8
M = int(os.environ.get("PROBE_M", "128"))            # <256 -> cfg 1 (64x64)
W = (rng.standard_normal((OUT, K)) * rng.uniform(0.2, 4.0, (OUT, 1))).astype(np.float32)
p = pack_weight(W, u, gs)
X = torch.from_numpy(rng.standard_normal((M, K)).astype(np.float32)).to(torch.bfloat16).cuda()

# warmup (not profiled — ncu uses --launch-skip/--launch-count to pick the target)
torch.cuda.synchronize()
for _ in range(3):
    Y = ops.wonly_gemm(p, X)
torch.cuda.synchronize()
Y = ops.wonly_gemm(p, X)                              # <-- profiled launch
torch.cuda.synchronize()
print("done", tuple(Y.shape), "M=", M)
