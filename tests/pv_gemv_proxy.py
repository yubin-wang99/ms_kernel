"""Q2 proxy: if V were channel-major [d, token], P.V = out[d]=Σ_t p[t]·V[d,t] is a
GEMV (weight=V [OUT=D, K=Lk], x=p). Run the existing MSAQ wide GEMV vs MXINT8 GEMV
at that shape to test whether channel-major makes P.V a coalesced MSAQ win (~0.6x).
Run: CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python tests/pv_gemv_proxy.py
"""
import time, numpy as np, torch
from ms_lib import ops; assert ops.available()
from ms_lib.pack import pack_weight, pack_weight_mxint8
OPS = torch.ops.msaq
def cuda(a): return torch.from_numpy(a).cuda()


def _t(fn, it=200):
    s = torch.cuda.Event(True); e = torch.cuda.Event(True)
    torch.cuda.synchronize(); s.record()
    for _ in range(it): fn()
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) / it


def cross(fa, fb, warm=3.0):
    t0 = time.time()
    while time.time() - t0 < warm: fa(); fb()
    return min(_t(fa), _t(fa)), min(_t(fb), _t(fb))


def bw(t_ms, n): return n / (t_ms * 1e-3) / 1e9


def run(OUT, K, u=4, gs=8, label=""):
    rng = np.random.default_rng(0)
    V = (rng.standard_normal((OUT, K)) * 0.5).astype(np.float32)   # V[d, token] (channel-major)
    p = rng.random(K).astype(np.float32); p /= p.sum()            # softmax-like probabilities = x
    x = cuda(p).to(torch.bfloat16)
    nb = K // 32
    pm = pack_weight(V, u, gs); s, upc, shc = cuda(pm["scale_exp"]), cuda(pm["upper_cm"]), cuda(pm["shared_cm"])
    px = pack_weight_mxint8(V); xs, xq = cuda(px["scale_exp"]), cuda(px["qweight"])
    msaq = lambda: OPS.wonly_gemv_wide(x, s, upc, shc, OUT, nb, u, gs)
    mx = lambda: OPS.mxint8_gemv(x, xs, xq, OUT, nb)
    # correctness (MSAQ result vs V@p on dequant)
    ref = (V.astype(np.float64) @ p.astype(np.float64))
    got = msaq().float().cpu().numpy()
    rel = np.linalg.norm(got - ref) / (np.linalg.norm(ref) + 1e-9)
    UB = 32 * (8 - u) // 8; SB = ((32 // gs) * u + 7) // 8
    bm = OUT * nb * (UB + SB) + OUT * nb
    bx = OUT * nb * 32 + OUT * nb
    tm, tx = cross(msaq, mx)
    print(f"  OUT={OUT:5d} K={K:6d} {label}: rel {rel:.1e} | "
          f"MXINT8 {tx*1e3:6.1f}us ({bw(tx,bx):4.0f}) | "
          f"MSAQ {tm*1e3:6.1f}us ({bw(tm,bm):4.0f}) | ratio {tm/tx:.3f} "
          f"{'WIN' if tm < tx else 'loss'}")


if __name__ == "__main__":
    torch.cuda.init()
    print("P.V-as-GEMV proxy (channel-major V). ratio=MSAQ/MXINT8, lower=MSAQ wins.\n"
          "[per-head: OUT=D=128]")
    for Lk in (4096, 8192, 16384, 32768):
        run(128, Lk, label="(1 head)")
    print("[8-kv-head proxy: OUT=8*128=1024 (bigger, BW-bound)]")
    for Lk in (4096, 8192, 16384):
        run(1024, Lk, label="(8 heads)")
    print("[reference: weight-GEMV shape OUT=K=4096 (known ~0.63 win)]")
    run(4096, 4096, label="(weight)")
