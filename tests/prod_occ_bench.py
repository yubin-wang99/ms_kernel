"""Time the PRODUCTION wonly_gemv (torch.ops.msaq -> wide kernel) under occupancy knobs.
Knobs are env-only (no rebuild): MS_GEMV_SPLITK_MULT (split-K mult, default 16),
MS_GEMV_SEPSC (separated-scale on/off). Set them BEFORE importing ms_cuda."""
import os, sys; sys.path.insert(0, ".")
import numpy as np, torch
from ms_lib.pack import pack_weight
from ms_lib import ops

u  = int(os.environ.get("U", "2"))
gs = int(os.environ.get("GS", "8"))
OUT, K = 4096, 4096
rng = np.random.default_rng(0)
W = (rng.standard_normal((OUT, K)) * rng.uniform(0.2, 4.0, (OUT, 1))).astype(np.float32)
p = pack_weight(W, u, gs)
x = torch.from_numpy(rng.standard_normal(K).astype(np.float32)).to(torch.bfloat16).cuda()

# hoist the H2D uploads OUT of the timed loop (ops.wonly_gemv re-uploads every call)
s   = torch.from_numpy(p["scale_exp"]).cuda()
up  = torch.from_numpy(p["upper_cm"]).cuda()
sh  = torch.from_numpy(p["shared_cm"]).cuda()
OUTn, nb = int(p["OUT"]), int(p["nb"])
call = lambda: torch.ops.msaq.wonly_gemv_wide(x, s, up, sh, OUTn, nb, u, gs)

def bench(iters=500):
    for _ in range(50): call()
    torch.cuda.synchronize()
    e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
    e0.record()
    for _ in range(iters): call()
    e1.record(); torch.cuda.synchronize()
    return e0.elapsed_time(e1) / iters * 1e3

t = bench()
print(f"u={u} gs={gs}  SPLITK_MULT={os.environ.get('MS_GEMV_SPLITK_MULT','16')} "
      f"SEPSC={os.environ.get('MS_GEMV_SEPSC','(default)')}  ->  {t:7.2f} us")
