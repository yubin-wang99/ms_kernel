#!/usr/bin/env python3
"""MXFP8-MSAQ (E3M4) decode kernel: correctness + isolated decode latency vs MXINT8.

Compares the per-element DEQUANT (unpack -> bf16) of three 6.0-bit formats at u3/gs4:
  * MXFP8-MSAQ E3M4 : msfp8_dequant_bf16   (new; FP8 element + shared low-mantissa)
  * MXINT8-MSAQ     : ms_dequant_bf16      (INT8 element + shared low-bits; same bit-stream)
  * plain MXINT8    : mxint8_dequant_bf16  (8.25b byte-aligned reference; the deployed fast path)
The first two share the SAME packed layout (upper field 8-u bits, shared u-bit), so the latency
delta isolates the FP reconstruction ALU (exp/mantissa split + ldexpf) vs the INT path.

Correctness: dequant(kernel) is compared to precision/msaq_mxfp8_ppl.py's msaq_mxfp8(e3m4) — the
exact function the PPL used — by QSNR (dB). High QSNR => the kernel decodes the same format.

Run: CUDA_VISIBLE_DEVICES=0 python tests/msfp8_decode_bench.py
"""
import sys, os, numpy as np, torch
sys.path.insert(0, "."); sys.path.insert(0, "..")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "precision"))
from ms_lib.pack import pack_weight, pack_weight_msfp8, pack_weight_mxint8
from ms_lib import ops; assert ops.available()
from msaq_mxfp8_ppl import msaq_mxfp8           # the PPL's reference encoder
OPS = torch.ops.msaq
def C(a): return torch.from_numpy(a).cuda()

def T(fn, iters=300, warm=50):
    for _ in range(warm): fn()
    torch.cuda.synchronize(); s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    s.record()
    for _ in range(iters): fn()
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) / iters

def qsnr(x, xq):
    err = (x - xq).double().pow(2).mean()
    return 10.0 * torch.log10(x.double().pow(2).mean() / err.clamp(min=1e-45)).item()

OUT = K = 4096
u, gs = 3, 4                                     # 6.0 bits/elem (E3M4: 1+3+1 + 3/4 + 8/32)

rng = np.random.default_rng(0)
# realistic weight: per-row scale spread -> high intra-block dynamic range (the regime FP8 wins)
W = (rng.standard_normal((OUT, K)) * rng.uniform(0.2, 4.0, (OUT, 1))).astype(np.float32)

pf = pack_weight_msfp8(W, u, gs)                 # E3M4
pm = pack_weight(W, u, gs)                        # MXINT8-MSAQ
px = pack_weight_mxint8(W)                        # plain MXINT8
nb, nbx = pf["nb"], px["nb"]

sf, ufc, sfc = C(pf["scale_exp"]), C(pf["upper_cm"]), C(pf["shared_cm"])
sm, umc, smc = C(pm["scale_exp"]), C(pm["upper_cm"]), C(pm["shared_cm"])
sx, qwc = C(px["scale_exp"]), C(px["qweight_cm"])

# ---- correctness: kernel dequant vs the PPL's msaq_mxfp8(e3m4) ----
Wt = torch.from_numpy(W).cuda()
ref = msaq_mxfp8(Wt, u, gs, eb=3, mb=4).float()                       # [OUT,K] reference values
deq = OPS.msfp8_dequant_bf16(sf, ufc, sfc, OUT, K, nb, u, gs).float()  # kernel -> [K,OUT]
deq = deq.t().contiguous()                                            # -> [OUT,K]
print(f"[correctness] MXFP8-MSAQ E3M4 kernel vs msaq_mxfp8 reference: QSNR = {qsnr(ref, deq):.1f} dB "
      f"(max abs err {(ref-deq).abs().max().item():.2e})")
deq_int = OPS.ms_dequant_bf16(sm, umc, smc, OUT, K, nb, u, gs).float().t().contiguous()
print(f"[sanity]      MXINT8-MSAQ kernel reconstruct error vs its packer: "
      f"QSNR = {qsnr(torch.from_numpy(W).cuda(), deq_int):.1f} dB (bf16-rounded)\n")

# ---- isolated decode latency (M-independent unpack -> bf16 [K,OUT]) ----
t_fp = T(lambda: OPS.msfp8_dequant_bf16(sf, ufc, sfc, OUT, K, nb, u, gs))
t_ms = T(lambda: OPS.ms_dequant_bf16(sm, umc, smc, OUT, K, nb, u, gs))
t_mx = T(lambda: OPS.mxint8_dequant_bf16(sx, qwc, OUT, K, nbx))
print(f"isolated DEQUANT latency  (OUT={OUT}, K={K}, u{u}/gs{gs}, {OUT*K/1e6:.1f}M elems -> bf16)")
print(f"   MXFP8-MSAQ E3M4 (6.0b) : {t_fp*1e3:7.2f} us")
print(f"   MXINT8-MSAQ     (6.0b) : {t_ms*1e3:7.2f} us   (E3M4 / INT8-MSAQ = {t_fp/t_ms:.3f}x)")
print(f"   plain MXINT8    (8.25b): {t_mx*1e3:7.2f} us   (E3M4 / plain MXINT8 = {t_fp/t_mx:.3f}x)")
