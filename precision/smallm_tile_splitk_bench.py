# small-M prefill W-only GEMM: tile-config and split-K latency sweep.
# Paths (all on the SAME u3/gs16 packed weight, OUT=K=4096):
#   wmma          : default production path  torch.ops.msaq.wonly_gemm_tc (64x64, no split-K)
#   scalar-NxN    : wonly_gemm_cm with MS_TILE_CFG = 0/1/5  (32x32 / 64x64 / 128x128, no split-K)
#   skinny sk=S   : wonly_gemm_fused_skinny with MS_FUSED_SPLITK = S  (split-K; M<=64 only, MT<=4)
# Latency = median over 60 timed iters (CUDA events), 15 warmup.
import sys, os; sys.path.insert(0, ".")
import numpy as np, torch, statistics
from ms_lib.pack import pack_weight
from ms_lib import ops  # noqa

rng = np.random.default_rng(0)
OUT, K, u, gs = 4096, 4096, 3, 16
W = (rng.standard_normal((OUT, K)) * rng.uniform(0.2, 4.0, (OUT, 1))).astype(np.float32)
p = pack_weight(W, u, gs)
dev = "cuda"
s   = torch.from_numpy(p["scale_exp"]).to(dev)
upc = torch.from_numpy(p["upper_cm"]).to(dev)
shc = torch.from_numpy(p["shared_cm"]).to(dev)
NB  = int(p["nb"])
O = torch.ops.msaq

def time_call(fn, iters=60, warmup=15):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    ev = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(iters)]
    for a, b in ev:
        a.record(); fn(); b.record()
    torch.cuda.synchronize()
    return statistics.median(a.elapsed_time(b) * 1e3 for a, b in ev)  # us

M_LIST = [8, 16, 32, 64, 128, 192]
SK_LIST = [1, 2, 4, 8, 16]
rows = []
for M in M_LIST:
    X = torch.from_numpy(rng.standard_normal((M, K)).astype(np.float32)).to(torch.bfloat16).to(dev)
    args = (M, OUT, K, NB, u, gs)
    rec = {"M": M}
    # WMMA default
    os.environ.pop("MS_TILE_CFG", None)
    rec["wmma"] = time_call(lambda: O.wonly_gemm_tc(X, s, upc, shc, *args))
    # scalar tiles
    for cfg, tag in [(0, "sc32"), (1, "sc64"), (5, "sc128")]:
        os.environ["MS_TILE_CFG"] = str(cfg)
        rec[tag] = time_call(lambda: O.wonly_gemm_cm(X, s, upc, shc, *args))
    os.environ.pop("MS_TILE_CFG", None)
    # skinny split-K (M<=64 only)
    if M <= 64:
        for S in SK_LIST:
            os.environ["MS_FUSED_SPLITK"] = str(S)
            rec[f"sk{S}"] = time_call(lambda: O.wonly_gemm_fused_skinny(X, s, upc, shc, *args))
        os.environ.pop("MS_FUSED_SPLITK", None)
    rows.append(rec)
    del X

cols = ["wmma", "sc32", "sc64", "sc128"] + [f"sk{S}" for S in SK_LIST]
hdr = f"{'M':>4} | " + " ".join(f"{c:>8}" for c in cols)
print(f"\nsmall-M W-only GEMM latency (us, median).  OUT=K=4096, u3/gs16, {torch.cuda.get_device_name(0)}")
print(f"wmma=default(tc)  sc*=scalar MS_TILE_CFG  sk*=skinny MS_FUSED_SPLITK\n")
print(hdr); print("-" * len(hdr))
for r in rows:
    cells = []
    for c in cols:
        cells.append(f"{r[c]:>8.1f}" if c in r else f"{'--':>8}")
    best = min((r[c], c) for c in cols if c in r)
    print(f"{r['M']:>4} | " + " ".join(cells) + f"   best={best[1]}({best[0]:.1f}us)")
