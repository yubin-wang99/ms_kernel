#!/usr/bin/env python3
# tests/benchmark.py  —  3-way macro latency for the MSAQ-signed kernels.
#
#   python tests/benchmark.py
#
# Per scope, times THREE kernels with torch.cuda.Event and prints them with the
# two ratios that matter:
#   * baseline : cuBLAS BF16 (linear) / SDPA BF16 (attention) — the tuned-library
#                reference (upper bound; very fast).
#   * MXINT8   : our int8 baseline kernel — SAME structure as MSAQ, direct int8
#                read (no unpack).
#   * MSAQ     : our mantissa-shared kernel — adds the sub-byte unpack.
#
# THE KEY NUMBER is MSAQ / MXINT8: same optimization level, so the ratio
# isolates "fewer bytes read (MSAQ wins) vs unpack overhead (MXINT8 wins)". At
# GEMV granularity the prior finding is that unpack overhead can dominate, i.e.
# MSAQ/MXINT8 > 1 even though MSAQ reads fewer bytes. MSAQ / cuBLAS is reported
# too but is NOT the matched comparison (cuBLAS is a different optimization
# level entirely — both our kernels are correctness-first, no tensor cores yet).
#
# MEASUREMENT DISCIPLINE: all packed planes are moved to the GPU ONCE before
# timing (H2D-per-iter is a ~48x artifact); warmup precedes timing.
#
# NCU (hardware metrics, no source edits) — e.g.:
#   ncu --set full python tests/benchmark.py
#   key metrics: smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct
#   (memory-bound), smsp__warp_issue_stalled_math_pipe_throttle_per_warp_active
#   .pct (unpack/bfe bound), sm__warps_active.avg.pct_of_peak_sustained_active
#   (occupancy).

import math
import sys
import numpy as np

try:
    import torch
    import torch.nn.functional as F
except Exception:
    print("[benchmark] torch unavailable — run on the RTX 3090.")
    sys.exit(0)

sys.path.insert(0, ".")
sys.path.insert(0, "..")
from ms_lib.pack import (pack_weight, dequant_weight, pack_kv,
                         pack_weight_mxint8, pack_kv_mxint8)
from ms_lib import ops


def measure_latency(fn, *args, iters=100, warmup=20):
    """Mean per-call latency in ms via cuda.Event (args already on device)."""
    for _ in range(warmup):
        fn(*args)
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn(*args)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def _cuda(np_arr):
    return torch.from_numpy(np_arr).cuda()


def _row(tag, t_msaq, t_mx, t_bl, bl_name):
    print(f"  {tag}")
    print(f"      MSAQ {t_msaq:8.4f} ms | MXINT8 {t_mx:8.4f} ms | {bl_name} {t_bl:8.4f} ms")
    print(f"      MSAQ/MXINT8 = {t_msaq / t_mx:6.2f}x   (>1 -> unpack overhead wins; "
          f"<1 -> bandwidth saving wins)   MSAQ/{bl_name} = {t_msaq / t_bl:6.1f}x")


def bench_wonly_gemv(u=3, gs=8, OUT=4096, K=4096):
    rng = np.random.default_rng(0)
    W = (rng.standard_normal((OUT, K)) * rng.uniform(0.2, 4.0, (OUT, 1))).astype(np.float32)
    pm, px = pack_weight(W, u, gs), pack_weight_mxint8(W)
    s = _cuda(pm["scale_exp"])
    sx, qw = _cuda(px["scale_exp"]), _cuda(px["qweight"])
    x = torch.from_numpy(rng.standard_normal(K).astype(np.float32)).to(torch.bfloat16).cuda()
    Wbf = torch.from_numpy(W).to(torch.bfloat16).cuda()
    # GEMV uses the wide-load (column-major) path for ALL u (Phase 14/16/19); move
    # the cm planes to the GPU ONCE (the ops.py wrapper would re-copy per call).
    up_cm, sh_cm = _cuda(pm["upper_cm"]), _cuda(pm["shared_cm"])
    t_msaq = measure_latency(lambda: torch.ops.msaq.wonly_gemv_wide(x, s, up_cm, sh_cm, OUT, pm["nb"], u, gs))
    t_mx = measure_latency(lambda: torch.ops.msaq.mxint8_gemv(x, sx, qw, OUT, px["nb"]))
    t_bl = measure_latency(lambda: x @ Wbf.t())
    _row(f"W-only GEMV  (OUT={OUT}, K={K}, MSAQ u{u}gs{gs})", t_msaq, t_mx, t_bl, "cuBLAS")


def bench_wonly_gemm(u=3, gs=8, M=512, OUT=4096, K=4096):
    rng = np.random.default_rng(2)
    W = (rng.standard_normal((OUT, K)) * rng.uniform(0.2, 4.0, (OUT, 1))).astype(np.float32)
    pm, px = pack_weight(W, u, gs), pack_weight_mxint8(W)
    s, up, sh = _cuda(pm["scale_exp"]), _cuda(pm["upper"]), _cuda(pm["shared"])
    sx, qw = _cuda(px["scale_exp"]), _cuda(px["qweight"])
    X = torch.from_numpy(rng.standard_normal((M, K)).astype(np.float32)).to(torch.bfloat16).cuda()
    Wbf = torch.from_numpy(W).to(torch.bfloat16).cuda()
    t_msaq = measure_latency(lambda: torch.ops.msaq.wonly_gemm(X, s, up, sh, M, OUT, K, pm["nb"], u, gs))
    t_mx = measure_latency(lambda: torch.ops.msaq.mxint8_gemm(X, sx, qw, M, OUT, K, px["nb"]))
    t_bl = measure_latency(lambda: X @ Wbf.t())
    _row(f"W-only GEMM  (M={M}, OUT={OUT}, K={K}, MSAQ u{u}gs{gs})", t_msaq, t_mx, t_bl, "cuBLAS")


def bench_wa_gemm(u=2, gs=8, M=512, OUT=4096, K=4096):
    rng = np.random.default_rng(1)
    W = (rng.standard_normal((OUT, K)) * rng.uniform(0.2, 4.0, (OUT, 1))).astype(np.float32)
    pm, px = pack_weight(W, u, gs), pack_weight_mxint8(W)
    s, up, sh = _cuda(pm["scale_exp"]), _cuda(pm["upper"]), _cuda(pm["shared"])
    sx, qw = _cuda(px["scale_exp"]), _cuda(px["qweight"])
    X = torch.from_numpy(rng.standard_normal((M, K)).astype(np.float32)).to(torch.bfloat16).cuda()
    Wbf = torch.from_numpy(W).to(torch.bfloat16).cuda()
    t_msaq = measure_latency(lambda: torch.ops.msaq.wa_gemm(X, s, up, sh, M, OUT, K, pm["nb"], u, gs))
    t_mx = measure_latency(lambda: torch.ops.msaq.mxint8_wa_gemm(X, sx, qw, M, OUT, K, px["nb"]))
    t_bl = measure_latency(lambda: X @ Wbf.t())
    _row(f"W+A GEMM     (M={M}, OUT={OUT}, K={K}, MSAQ u{u}gs{gs})", t_msaq, t_mx, t_bl, "cuBLAS")


def bench_kv_decode(u=3, gs=8, H=8, Lk=4096, D=128):
    rng = np.random.default_rng(3)
    K = (rng.standard_normal((H, Lk, D)) * 0.5).astype(np.float32)
    V = (rng.standard_normal((H, Lk, D)) * 0.5).astype(np.float32)
    q = (rng.standard_normal((H, D)) * 0.5).astype(np.float32)
    pmK, pmV = pack_kv(K, u, gs), pack_kv(V, u, gs)
    pxK, pxV = pack_kv_mxint8(K), pack_kv_mxint8(V)
    ks, ku, kh = _cuda(pmK["scale_exp"]), _cuda(pmK["upper"]), _cuda(pmK["shared"])
    vs, vu, vh = _cuda(pmV["scale_exp"]), _cuda(pmV["upper"]), _cuda(pmV["shared"])
    kxs, kxq = _cuda(pxK["scale_exp"]), _cuda(pxK["qweight"])
    vxs, vxq = _cuda(pxV["scale_exp"]), _cuda(pxV["qweight"])
    qt = torch.from_numpy(q).to(torch.bfloat16).cuda()
    Kbf = torch.from_numpy(K).to(torch.bfloat16).cuda()
    Vbf = torch.from_numpy(V).to(torch.bfloat16).cuda()
    q4 = qt.unsqueeze(1)
    t_msaq = measure_latency(lambda: torch.ops.msaq.kv_decode_attention(
        qt, ks, ku, kh, vs, vu, vh, H, H, Lk, D, pmK["nb"], u, gs))
    t_mx = measure_latency(lambda: torch.ops.msaq.mxint8_kv_decode(
        qt, kxs, kxq, vxs, vxq, H, H, Lk, D, pxK["nb"]))
    t_bl = measure_latency(lambda: F.scaled_dot_product_attention(q4, Kbf, Vbf))
    _row(f"KV decode    (H={H}, Lk={Lk}, D={D}, MSAQ u{u}gs{gs})", t_msaq, t_mx, t_bl, "SDPA")


def main():
    if not torch.cuda.is_available():
        print("[benchmark] no CUDA device — run on the RTX 3090.")
        return
    if not ops.available():
        print("[benchmark] ms_cuda not built — run: python setup.py build_ext --inplace")
        return
    print(f"[benchmark] {torch.cuda.get_device_name(0)} | torch {torch.__version__}")
    print("  (lower ms is better; MSAQ/MXINT8 is the matched-optimization comparison)\n")
    bench_wonly_gemv()
    bench_wonly_gemm()
    bench_wa_gemm()
    bench_kv_decode()


if __name__ == "__main__":
    main()
