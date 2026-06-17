"""Phase 18 A/B: key-per-thread WIDE vs cp.async vs MXINT8 (warm cross-measure).
Run: CUDA_VISIBLE_DEVICES=1 python tests/kv_wide_bench.py
"""
import os, time, numpy as np, torch
from ms_lib.pack import pack_kv, pack_kv_mxint8
from ms_lib import ops
assert ops.available()

def _cuda(a): return torch.from_numpy(a).cuda()

def _t(fn, it=300):
    s = torch.cuda.Event(True); e = torch.cuda.Event(True)
    torch.cuda.synchronize(); s.record()
    for _ in range(it): fn()
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e)/it

def cross(fa, fb, warm=3.0):
    # warm both (cancels cold-clock bias, see Phase 17), then time each separately
    t0 = time.time()
    while time.time() - t0 < warm: fa(); fb()
    ta = min(_t(fa), _t(fa)); tb = min(_t(fb), _t(fb))
    return ta, tb

def run(u=4, gs=8, H=8, Lk=4096, D=128):
    rng = np.random.default_rng(3)
    K = (rng.standard_normal((H, Lk, D))*0.5).astype(np.float32)
    V = (rng.standard_normal((H, Lk, D))*0.5).astype(np.float32)
    q = (rng.standard_normal((H, D))*0.5).astype(np.float32)
    pmK, pmV = pack_kv(K, u, gs), pack_kv(V, u, gs)
    pxK, pxV = pack_kv_mxint8(K), pack_kv_mxint8(V)
    ks,ku,kh = _cuda(pmK["scale_exp"]),_cuda(pmK["upper"]),_cuda(pmK["shared"])
    vs,vu,vh = _cuda(pmV["scale_exp"]),_cuda(pmV["upper"]),_cuda(pmV["shared"])
    kxs,kxq = _cuda(pxK["scale_exp"]),_cuda(pxK["qweight"])
    vxs,vxq = _cuda(pxV["scale_exp"]),_cuda(pxV["qweight"])
    qt = torch.from_numpy(q).to(torch.bfloat16).cuda()
    nb, nbx = pmK["nb"], pxK["nb"]
    msaq = lambda: torch.ops.msaq.kv_decode_attention(qt,ks,ku,kh,vs,vu,vh,H,Lk,D,nb,u,gs)
    mx   = lambda: torch.ops.msaq.mxint8_kv_decode(qt,kxs,kxq,vxs,vxq,H,Lk,D,nbx)

    UB = 32*(8-u)//8; SB = ((32//gs)*u+7)//8
    bytes_msaq = H*nb*Lk*(UB+SB)*2 + H*nb*Lk*2          # K+V upper/sh + scales(x2 planes)
    bytes_mx   = H*nbx*Lk*32*2 + H*nbx*Lk*2
    def bw(t_ms, nbytes): return nbytes/(t_ms*1e-3)/1e9  # GB/s

    print(f"\n=== u{u} gs{gs} H{H} Lk{Lk} D{D}  (MSAQ {(UB+SB)} B/blk vs MXINT8 32) ===")
    # wide (default) vs mxint8
    os.environ["MS_KV_WIDE"]="1"
    tw, tx = cross(msaq, mx)
    # cpasync vs mxint8
    os.environ["MS_KV_WIDE"]="0"; os.environ["MS_KV_CPASYNC"]="1"
    tc, tx2 = cross(msaq, mx)
    os.environ["MS_KV_WIDE"]="1"
    print(f"  MXINT8        : {tx*1000:7.2f} us  ({bw(tx,bytes_mx):5.0f} GB/s)")
    print(f"  MSAQ cp.async : {tc*1000:7.2f} us  ({bw(tc,bytes_msaq):5.0f} GB/s)  {tc/tx2:.2f}x MX")
    print(f"  MSAQ WIDE (A) : {tw*1000:7.2f} us  ({bw(tw,bytes_msaq):5.0f} GB/s)  {tw/tx:.2f}x MX"
          f"   [wide/cpasync = {tw/tc:.2f}x]")

if __name__ == "__main__":
    torch.cuda.init()
    print(f"[kv_wide_bench] {torch.cuda.get_device_name(0)}")
    run(Lk=4096)
    run(Lk=16384)
    run(Lk=16384, H=16)   # 151 MB footprint (>> L2): pure HBM regime
