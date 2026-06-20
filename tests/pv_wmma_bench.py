"""Tensor-core P.V win-track: bf16 WMMA P.V with token-major V (per-token group,
accurate/fair) unpacked to a d-major shared tile. MSAQ vs MXINT8 across M=batch*G.
Hypothesis: WMMA makes P.V memory-bound -> MSAQ reads 0.58x V bytes -> win at M>=16.
Run: CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python tests/pv_wmma_bench.py
"""
import time, numpy as np, torch
from ms_lib.pack import pack_kv, pack_kv_mxint8, dequant_weight
from ms_lib import ops; assert ops.available()
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

def setup(Hkv, M, D, Lk, u=4, gs=8, seed=0):
    rng = np.random.default_rng(seed)
    nb = D // 32
    V = (rng.standard_normal((Hkv, Lk, D)) * 0.5).astype(np.float32)     # [Hkv, token, d]
    Pr = rng.random((Hkv, M, Lk)).astype(np.float32); Pr /= Pr.sum(-1, keepdims=True)
    Pt = cuda(Pr).to(torch.bfloat16)
    pm = pack_kv(V, u, gs); px = pack_kv_mxint8(V)
    vs, vu, vh = cuda(pm["scale_exp"]), cuda(pm["upper"]), cuda(pm["shared"])
    xs, xq = cuda(px["scale_exp"]), cuda(px["qweight"])
    msaq = lambda: OPS.pv_wmma(Pt, vs, vu, vh, Hkv, M, D, Lk, nb, u, gs)
    mx   = lambda: OPS.pv_wmma_mx(Pt, xs, xq, Hkv, M, D, Lk, nb)
    UB = 32*(8-u)//8; SB = ((32//gs)*u+7)//8
    bm = Hkv*nb*Lk*(UB+SB) + Hkv*nb*Lk      # V upper/shared + scale (read once)
    bx = Hkv*nb*Lk*32 + Hkv*nb*Lk
    return dict(msaq=msaq, mx=mx, V=V, Pr=Pr, Pt=Pt, pm=pm, bm=bm, bx=bx, nb=nb)

def check():
    Hkv, M, D, Lk = 2, 32, 128, 256
    d = setup(Hkv, M, D, Lk, seed=3)
    got = d["msaq"]().float().cpu().numpy()                       # [Hkv, M, D]
    bad = 0
    for h in range(Hkv):
        Vdq = dequant_weight(d["pm"]["_per"][h])                  # [Lk, D]  (token, d)
        ref = d["Pr"][h] @ Vdq                                    # [M, D]
        rel = np.linalg.norm(got[h] - ref) / (np.linalg.norm(ref) + 1e-9)
        print(f"  head {h}: WMMA P.V vs P@dequant(V)  rel_fro {rel:.2e}  {'OK' if rel < 3e-2 else 'FAIL'}")
        bad += rel >= 3e-2
    return bad

if __name__ == "__main__":
    torch.cuda.init()
    print("[correctness]")
    check()
    print("\n[batch sweep] Hkv=8 D=128 Lk=4096, M=batch*G. ratio=MSAQ/MXINT8, lower=MSAQ wins")
    for Lk in (4096, 8192):
        for M in (16, 32, 64, 128):
            d = setup(8, M, 128, Lk)
            tm, tx = cross(d["msaq"], d["mx"])
            print(f"  Lk{Lk} M={M:3d}: MXINT8 {tx*1e3:6.1f}us ({bw(tx,d['bx']):4.0f}) | "
                  f"MSAQ {tm*1e3:6.1f}us ({bw(tm,d['bm']):4.0f}) | ratio {tm/tx:.3f} "
                  f"{'WIN' if tm < tx else 'loss'}")
            del d; torch.cuda.empty_cache()
