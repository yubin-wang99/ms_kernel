#!/usr/bin/env python3
"""MXFP8-MSAQ (E3M4) FUSED GEMV/GEMM latency vs MXINT8-MSAQ, plain MXINT8, BF16.

The isolated dequant (msfp8_decode_bench) is write-bound and equal to INT8-MSAQ. This bench
measures the FUSED kernels where the unpack OVERLAPS compute:
  * W-only decode GEMV (M=1 wide, M>1 batched): unpack overlaps the dot
  * prefill GEMM (wmma_pipe): column dequant overlaps the tensor-core MMA
E3M4 reads the SAME bytes as MXINT8-MSAQ (field width 8-u == 1+eb+(mb-u)); only the per-element
reconstruction ALU differs (ldexpf). Config u3/gs4 = 6.0 bits/elem (the accuracy-winning point).

Run: CUDA_VISIBLE_DEVICES=0 python tests/msfp8_gemv_gemm_bench.py
"""
import sys, os, numpy as np, torch
sys.path.insert(0, "."); sys.path.insert(0, "..")
from ms_lib.pack import pack_weight, pack_weight_msfp8, pack_weight_mxint8
from ms_lib import ops; assert ops.available()
OPS = torch.ops.msaq
def C(a): return torch.from_numpy(a).cuda()

def T(fn, iters=200, warm=30):
    for _ in range(warm): fn()
    torch.cuda.synchronize(); s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    s.record()
    for _ in range(iters): fn()
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) / iters  # ms

OUT = K = 4096
u, gs = 3, 4
rng = np.random.default_rng(0)
W = (rng.standard_normal((OUT, K)) * rng.uniform(0.2, 4.0, (OUT, 1))).astype(np.float32)
pf, pm, px = pack_weight_msfp8(W, u, gs), pack_weight(W, u, gs), pack_weight_mxint8(W)
nb, nbx = pf["nb"], px["nb"]
sf, ufc, sfc = C(pf["scale_exp"]), C(pf["upper_cm"]), C(pf["shared_cm"])
sm, umc, smc = C(pm["scale_exp"]), C(pm["upper_cm"]), C(pm["shared_cm"])
sx, qw, qwc = C(px["scale_exp"]), C(px["qweight"]), C(px["qweight_cm"])
Wdeq = OPS.msfp8_dequant_bf16(sf, ufc, sfc, OUT, K, nb, u, gs).t().contiguous()  # [OUT,K] E3M4-decoded

# ---- correctness: fused E3M4 GEMV/GEMM == (x @ Wdeq^T) ----
x1 = torch.randn(K, device="cuda", dtype=torch.bfloat16)
ref1 = (x1.float() @ Wdeq.float().t())
got1 = OPS.msfp8_gemv_wide(x1, sf, ufc, sfc, OUT, nb, u, gs).float()
rel = (got1 - ref1).norm() / ref1.norm()
X8 = torch.randn(8, K, device="cuda", dtype=torch.bfloat16)
gb = OPS.msfp8_gemv_batched(X8, sf, ufc, sfc, 8, OUT, nb, u, gs).float()
rb = (X8.float() @ Wdeq.float().t())
relb = (gb - rb).norm() / rb.norm()
X128 = torch.randn(128, K, device="cuda", dtype=torch.bfloat16)
gm = OPS.msfp8_gemm(X128, sf, ufc, sfc, 128, OUT, K, nb, u, gs).float()
rm = (X128.float() @ Wdeq.float().t())
relm = (gm - rm).norm() / rm.norm()
print(f"[correctness vs x@Wdeq^T] wide(M1) rel={rel:.2e} | batched(M8) rel={relb:.2e} | gemm(M128) rel={relm:.2e}\n")

print(f"FUSED GEMV/GEMM latency (OUT={OUT}, K={K}, u{u}/gs{gs}=6.0b)  [us; ratio = E3M4/INT8-MSAQ]")
print(f"{'M':>5} | {'BF16':>8} {'plainMX':>8} {'INT8-MSAQ':>9} {'E3M4':>8} | {'E3M4/MSAQ':>9} {'E3M4/bf16':>9}")
# decode GEMV: M=1 wide, M>1 batched
for M in (1, 4, 8, 16, 32):
    X = torch.randn(M, K, device="cuda", dtype=torch.bfloat16); x1 = X[0].contiguous()
    Wbf = C(W).to(torch.bfloat16)
    bf = T(lambda: X @ Wbf.t())
    if M == 1:
        mx = T(lambda: OPS.mxint8_gemv(x1, sx, qw, OUT, nbx))
        ms = T(lambda: OPS.wonly_gemv_wide(x1, sm, umc, smc, OUT, nb, u, gs))
        fp = T(lambda: OPS.msfp8_gemv_wide(x1, sf, ufc, sfc, OUT, nb, u, gs))
    else:
        mx = T(lambda: OPS.mxint8_gemv_batched(X, sx, qw, M, OUT, nbx))
        ms = T(lambda: OPS.wonly_gemv_batched(X, sm, umc, smc, M, OUT, nb, u, gs))
        fp = T(lambda: OPS.msfp8_gemv_batched(X, sf, ufc, sfc, M, OUT, nb, u, gs))
    print(f"{M:>5} | {bf*1e3:8.1f} {mx*1e3:8.1f} {ms*1e3:9.1f} {fp*1e3:8.1f} | {fp/ms:9.3f} {fp/bf:9.2f}")
# prefill GEMM (wmma_pipe)
print(f"  --- prefill GEMM (wmma_pipe) ---")
for M in (128, 256, 512, 1024):
    X = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    Wbf = C(W).to(torch.bfloat16)
    bf = T(lambda: X @ Wbf.t())
    mx = T(lambda: OPS.mxint8_gemm(X, sx, qw, M, OUT, K, nbx))
    ms = T(lambda: OPS.wonly_gemm_tc(X, sm, umc, smc, M, OUT, K, nb, u, gs))
    fp = T(lambda: OPS.msfp8_gemm(X, sf, ufc, sfc, M, OUT, K, nb, u, gs))
    print(f"{M:>5} | {bf*1e3:8.1f} {mx*1e3:8.1f} {ms*1e3:9.1f} {fp*1e3:8.1f} | {fp/ms:9.3f} {fp/bf:9.2f}")

# FAIR batched GEMV: at gs4 the INT8-MSAQ batched kernel has no (u,gs)-specialized path and
# falls back to the slow generic unpack -> unfair. Re-measure at gs16 where BOTH have the fast
# uspec path (note: gs16 = 5.5b, lower accuracy than the gs4=6.0b point; this isolates the
# overlap behavior of the FP reconstruction, not the deployed accuracy config).
print(f"\nFAIR batched GEMV at u3/gs16 (both fast uspec paths; 5.5b)  [us]")
gsf = 16
pf16, pm16 = pack_weight_msfp8(W, u, gsf), pack_weight(W, u, gsf)
sf6, uf6, sf6c = C(pf16["scale_exp"]), C(pf16["upper_cm"]), C(pf16["shared_cm"])
sm6, um6, sm6c = C(pm16["scale_exp"]), C(pm16["upper_cm"]), C(pm16["shared_cm"])
print(f"{'M':>5} | {'INT8-MSAQ':>9} {'E3M4':>8} | {'E3M4/MSAQ':>9}")
for M in (1, 4, 8, 16, 32):
    X = torch.randn(M, K, device="cuda", dtype=torch.bfloat16); x1 = X[0].contiguous()
    if M == 1:
        ms = T(lambda: OPS.wonly_gemv_wide(x1, sm6, um6, sm6c, OUT, nb, u, gsf))
        fp = T(lambda: OPS.msfp8_gemv_wide(x1, sf6, uf6, sf6c, OUT, nb, u, gsf))
    else:
        ms = T(lambda: OPS.wonly_gemv_batched(X, sm6, um6, sm6c, M, OUT, nb, u, gsf))
        fp = T(lambda: OPS.msfp8_gemv_batched(X, sf6, uf6, sf6c, M, OUT, nb, u, gsf))
    print(f"{M:>5} | {ms*1e3:9.1f} {fp*1e3:8.1f} | {fp/ms:9.3f}")
