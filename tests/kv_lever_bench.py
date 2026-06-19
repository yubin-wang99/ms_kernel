"""Fair KV-read bench: MSAQ (wide / gqa) vs MXINT8, current op signatures.
Run: CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python tests/kv_lever_bench.py
Env knobs (read by the kernel launcher): MS_KV_WIDE, MS_KV_GQA, MS_KV_CPASYNC, MS_KV_DIAG.
"""
import os, time, numpy as np, torch
from ms_lib.pack import pack_kv, pack_kv_mxint8
from ms_lib import ops
assert ops.available(), "ms_cuda backend not built"

def _cuda(a): return torch.from_numpy(a).cuda()

def _t(fn, it=200):
    s = torch.cuda.Event(True); e = torch.cuda.Event(True)
    torch.cuda.synchronize(); s.record()
    for _ in range(it): fn()
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e)/it

def cross(fa, fb, warm=3.0):
    t0 = time.time()
    while time.time() - t0 < warm: fa(); fb()
    return min(_t(fa), _t(fa)), min(_t(fb), _t(fb))

def setup(u=4, gs=8, H=8, Hkv=8, Lk=4096, D=128, seed=3):
    rng = np.random.default_rng(seed)
    K = (rng.standard_normal((Hkv, Lk, D))*0.5).astype(np.float32)
    V = (rng.standard_normal((Hkv, Lk, D))*0.5).astype(np.float32)
    q = (rng.standard_normal((H, D))*0.5).astype(np.float32)
    pmK, pmV = pack_kv(K, u, gs), pack_kv(V, u, gs)
    pxK, pxV = pack_kv_mxint8(K), pack_kv_mxint8(V)
    ks,ku,kh = _cuda(pmK["scale_exp"]),_cuda(pmK["upper"]),_cuda(pmK["shared"])
    vs,vu,vh = _cuda(pmV["scale_exp"]),_cuda(pmV["upper"]),_cuda(pmV["shared"])
    kxs,kxq = _cuda(pxK["scale_exp"]),_cuda(pxK["qweight"])
    vxs,vxq = _cuda(pxV["scale_exp"]),_cuda(pxV["qweight"])
    qt = torch.from_numpy(q).to(torch.bfloat16).cuda()
    nb, nbx = pmK["nb"], pxK["nb"]
    msaq = lambda: torch.ops.msaq.kv_decode_attention(qt,ks,ku,kh,vs,vu,vh,H,Hkv,Lk,D,nb,u,gs)
    mx   = lambda: torch.ops.msaq.mxint8_kv_decode(qt,kxs,kxq,vxs,vxq,H,Hkv,Lk,D,nbx)
    UB = 32*(8-u)//8; SB = ((32//gs)*u+7)//8
    bytes_msaq = Hkv*nb*Lk*(UB+SB)*2 + Hkv*nb*Lk*2
    bytes_mx   = Hkv*nbx*Lk*32*2 + Hkv*nbx*Lk*2
    return dict(msaq=msaq, mx=mx, qt=qt, bytes_msaq=bytes_msaq, bytes_mx=bytes_mx,
                UB=UB, SB=SB, args_m=(ks,ku,kh,vs,vu,vh), args_x=(kxs,kxq,vxs,vxq),
                u=u, gs=gs, H=H, Hkv=Hkv, Lk=Lk, D=D, nb=nb, nbx=nbx)

def bw(t_ms, nbytes): return nbytes/(t_ms*1e-3)/1e9

def run(u=4, gs=8, H=8, Hkv=8, Lk=4096, D=128, label=""):
    d = setup(u, gs, H, Hkv, Lk, D)
    msaq, mx = d["msaq"], d["mx"]
    print(f"\n=== u{u} gs{gs} H{H} Hkv{Hkv} Lk{Lk} D{D}  {label}"
          f"  (MSAQ {d['UB']+d['SB']} B/blk vs MXINT8 32) ===")
    tm, tx = cross(msaq, mx)
    print(f"  MXINT8     : {tx*1000:7.2f} us  ({bw(tx,d['bytes_mx']):5.0f} GB/s)")
    print(f"  MSAQ       : {tm*1000:7.2f} us  ({bw(tm,d['bytes_msaq']):5.0f} GB/s)  {tm/tx:.3f}x MX"
          f"  {'WIN' if tm<tx else 'loss'}")
    return tm, tx

if __name__ == "__main__":
    torch.cuda.init()
    mode = os.environ.get("KV_MODE", "wide")
    if mode == "wide":
        os.environ["MS_KV_WIDE"]="1"; os.environ.pop("MS_KV_GQA", None)
    elif mode == "gqa":
        os.environ["MS_KV_GQA"]="1"; os.environ["MS_KV_WIDE"]="1"
    print(f"[mode={mode}]  MS_KV_WIDE={os.environ.get('MS_KV_WIDE')} MS_KV_GQA={os.environ.get('MS_KV_GQA')}")
    for u in (4, 2):
        for Lk in (1056, 2848, 4680):
            run(u=u, H=8, Hkv=8, Lk=Lk, D=128)
    # GQA-shaped (Hq=32, Hkv=8) — the real decode shape
    print("\n----- GQA shape Hq=32 Hkv=8 -----")
    for u in (4,):
        for Lk in (2848, 4680):
            run(u=u, H=32, Hkv=8, Lk=Lk, D=128)
