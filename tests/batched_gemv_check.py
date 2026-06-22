"""Correctness + latency of the batched-decode GEMV (W-only B>1 win).
Correctness: wonly_gemv_batched(X[M,K]) == row-wise wide GEMV, and rel_fro vs bf16 oracle.
Latency: batched GEMV vs wonly_gemm(M=B) vs bf16 (X@W) at decode batch M -> show the win.
Run: CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python tests/batched_gemv_check.py
"""
import numpy as np, torch
from ms_lib.pack import pack_weight, pack_weight_mxint8, dequant_weight
from ms_lib import ops; assert ops.available()
OPS = torch.ops.msaq
DEV = "cuda"


def _t(fn, it=100):
    s = torch.cuda.Event(True); e = torch.cuda.Event(True)
    torch.cuda.synchronize(); s.record()
    for _ in range(it): fn()
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) / it


def check(OUT=4096, K=4096, u=4, gs=2):
    rng = np.random.default_rng(0)
    W = (rng.standard_normal((OUT, K)) * 0.02).astype(np.float32)
    pm = pack_weight(W, u, gs); px = pack_weight_mxint8(W)
    s = torch.from_numpy(pm["scale_exp"]).cuda(); upc = torch.from_numpy(pm["upper_cm"]).cuda(); shc = torch.from_numpy(pm["shared_cm"]).cuda()
    sx = torch.from_numpy(px["scale_exp"]).cuda(); qw = torch.from_numpy(px["qweight"]).cuda()
    Wdq = torch.from_numpy(dequant_weight(pm)).cuda().to(torch.bfloat16)   # MSAQ-dequant weight (bf16) [OUT,K]
    nb = K // 32
    print(f"OUT={OUT} K={K} u{u}/gs{gs}  (rel_fro vs MSAQ-dequant oracle; <2e-2 OK)")
    for M in (2, 8, 16, 32, 48):
        X = torch.randn(M, K, dtype=torch.bfloat16, device=DEV)
        yb = OPS.wonly_gemv_batched(X, s, upc, shc, M, OUT, nb, u, gs)        # [M,OUT]
        # oracle: X @ Wdq^T (same dequant the kernel uses)
        ref = (X.float() @ Wdq.float().t())
        rel = (yb.float() - ref).norm() / (ref.norm() + 1e-9)
        # row-wise consistency vs single-row wide GEMV
        y0 = torch.stack([OPS.wonly_gemv_wide(X[m], s, upc, shc, OUT, nb, u, gs) for m in range(M)])
        rrow = (yb.float() - y0.float()).norm() / (y0.float().norm() + 1e-9)
        ymx = OPS.mxint8_gemv_batched(X, sx, qw, M, OUT, nb)
        print(f"  M={M:>2}: rel_fro {rel:.2e} | vs row-wise {rrow:.2e} | mxint8 batched ok {tuple(ymx.shape)} "
              f"{'PASS' if rel < 2e-2 and rrow < 1e-3 else 'FAIL'}")


def latency(OUT=4096, K=4096, u=4, gs=2):
    rng = np.random.default_rng(0)
    W = (rng.standard_normal((OUT, K)) * 0.02).astype(np.float32)
    pm = pack_weight(W, u, gs); px = pack_weight_mxint8(W)
    s = torch.from_numpy(pm["scale_exp"]).cuda(); upc = torch.from_numpy(pm["upper_cm"]).cuda(); shc = torch.from_numpy(pm["shared_cm"]).cuda()
    up = torch.from_numpy(pm["upper"]).cuda(); sh = torch.from_numpy(pm["shared"]).cuda()
    sx = torch.from_numpy(px["scale_exp"]).cuda(); qw = torch.from_numpy(px["qweight"]).cuda()
    Wt = torch.from_numpy(W).to(torch.bfloat16).cuda().t().contiguous()
    nb = K // 32
    print(f"\nlatency (us) OUT={OUT} K={K} u{u}/gs{gs}: msaq-batched-GEMV vs msaq-GEMM(M) vs bf16(cuBLAS)")
    print(f"{'M':>3} | {'msaq batGEMV':>12} {'msaq GEMM':>10} {'mxint8 batGEMV':>14} {'bf16':>8} | bat/bf bat/gemm")
    for M in (1, 8, 16, 32, 48):
        X = torch.randn(M, K, dtype=torch.bfloat16, device=DEV)
        f_bat = lambda: OPS.wonly_gemv_batched(X, s, upc, shc, M, OUT, nb, u, gs)
        f_gemm = lambda: OPS.wonly_gemm(X, s, up, sh, M, OUT, K, nb, u, gs)
        f_mxb = lambda: OPS.mxint8_gemv_batched(X, sx, qw, M, OUT, nb)
        f_bf = lambda: X @ Wt
        for f in (f_bat, f_gemm, f_mxb, f_bf): f()
        tb = min(_t(f_bat), _t(f_bat)); tg = min(_t(f_gemm), _t(f_gemm))
        tmx = min(_t(f_mxb), _t(f_mxb)); tf = min(_t(f_bf), _t(f_bf))
        print(f"{M:>3} | {tb*1e3:>12.1f} {tg*1e3:>10.1f} {tmx*1e3:>14.1f} {tf*1e3:>8.1f} | "
              f"{tb/tf:>5.2f} {tb/tg:>6.2f}")


if __name__ == "__main__":
    check()
    latency()                          # attn/o proj OUT=K=4096
    latency(OUT=14336, K=4096)         # MLP gate/up
    latency(OUT=4096, K=14336)         # MLP down
