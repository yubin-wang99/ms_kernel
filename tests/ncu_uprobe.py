# ncu probe: launch the wide-load W-only GEMV once (measured) for a given u.
# Warms up so the measured launch excludes JIT/alloc; ncu picks the last launch.
#   U=<2|3|4> GS=<n> python tests/ncu_uprobe.py
import os, sys; sys.path.insert(0, ".")
import numpy as np, torch
from ms_lib.pack import pack_weight
from ms_lib import ops

u  = int(os.environ.get("U", "3"))
gs = int(os.environ.get("GS", "8"))
OUT, K = 4096, 4096

rng = np.random.default_rng(0)
W = (rng.standard_normal((OUT, K)) * rng.uniform(0.2, 4.0, (OUT, 1))).astype(np.float32)
p = pack_weight(W, u, gs)
x = torch.from_numpy(rng.standard_normal(K).astype(np.float32)).to(torch.bfloat16).cuda()

# warmup (these launches are skipped via --launch-skip)
for _ in range(5):
    ops.wonly_gemv(p, x)
torch.cuda.synchronize()
# measured launch (the one ncu profiles)
ops.wonly_gemv(p, x)
torch.cuda.synchronize()
