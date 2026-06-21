"""(ii) footprint -> memory-bound GEMV speed. W-only GEMV at u in {2,3,4} (gs8) vs MXINT8.
Shows whether the robust-aggressive weight config (u3/gs8, 0.69x bytes) actually speeds up.
Run: CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python tests/gemv_u_bench.py
"""
import time, numpy as np, torch
from ms_lib.pack import pack_weight, pack_weight_mxint8
from ms_lib import ops; assert ops.available()
OPS = torch.ops.msaq
def cuda(a): return torch.from_numpy(a).cuda()

def _t(fn, it=300):
    s = torch.cuda.Event(True); e = torch.cuda.Event(True)
    torch.cuda.synchronize(); s.record()
    for _ in range(it): fn()
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) / it

def cross(fns, warm=3.0):
    t0 = time.time()
    while time.time() - t0 < warm:
        for f in fns: f()
    return [min(_t(f), _t(f)) for f in fns]

def run(OUT=4096, K=4096, gs=8):
    rng = np.random.default_rng(0)
    W = (rng.standard_normal((OUT, K)) * 0.02).astype(np.float32)
    x = cuda((rng.standard_normal((K,))).astype(np.float32)).to(torch.bfloat16)
    nb = K // 32
    px = pack_weight_mxint8(W); xs, xq = cuda(px["scale_exp"]), cuda(px["qweight"])
    mx = lambda: OPS.mxint8_gemv(x, xs, xq, OUT, nb)
    bytes_mx = OUT * nb * 32 + OUT * nb
    print(f"=== W-only GEMV {OUT}x{K} gs{gs} (vs MXINT8 {bytes_mx/1e6:.2f}MB) ===")
    fns = [mx]; tags = ["MXINT8"]; byts = [bytes_mx]
    for u in (2, 3, 4):
        p = pack_weight(W, u, gs)
        s, upc, shc = cuda(p["scale_exp"]), cuda(p["upper_cm"]), cuda(p["shared_cm"])
        fns.append((lambda s=s, upc=upc, shc=shc, u=u: OPS.wonly_gemv_wide(x, s, upc, shc, OUT, nb, u, gs)))
        UB = 32 * (8 - u) // 8; SB = ((32 // gs) * u + 7) // 8
        byts.append(OUT * nb * (UB + SB) + OUT * nb); tags.append(f"MSAQ u{u}")
    ts = cross(fns)
    tmx = ts[0]
    for tag, t, b in zip(tags, ts, byts):
        bw = b / (t * 1e-3) / 1e9
        rel = f"  {t/tmx:.3f}x MX" if tag != "MXINT8" else ""
        print(f"  {tag:9s}: {t*1e3:6.2f}us  {b/1e6:5.2f}MB ({b/bytes_mx:.2f}x)  {bw:4.0f} GB/s{rel}")

if __name__ == "__main__":
    torch.cuda.init()
    for sz in (4096, 14336):
        run(OUT=sz, K=4096, gs=8)
        print()
