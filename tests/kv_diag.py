"""KV decode bottleneck decomposition (memceil-style).
diag=1: memory ceiling (cp.async staging only, no unpack, no exp)
diag=2: + unpack (dequant), no exp
diag=0: full (unpack + softmax exp)
Run on GPU1: CUDA_VISIBLE_DEVICES=1 python tests/kv_diag.py
"""
import os, numpy as np, torch
from ms_lib.pack import pack_kv, pack_kv_mxint8
from ms_lib import ops  # registers torch.ops.msaq.*
assert ops.available(), "ms_cuda not built"

def _cuda(a): return torch.from_numpy(a).cuda()

def measure(fn, it=200, warm=30):
    for _ in range(warm): fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(True); e = torch.cuda.Event(True)
    s.record()
    for _ in range(it): fn()
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) / it

def run(u, gs, H=8, Lk=4096, D=128):
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
    nb = pmK["nb"]
    msaq = lambda: torch.ops.msaq.kv_decode_attention(qt, ks, ku, kh, vs, vu, vh, H, Lk, D, nb, u, gs)
    mx   = lambda: torch.ops.msaq.mxint8_kv_decode(qt, kxs, kxq, vxs, vxq, H, Lk, D, pxK["nb"])

    def t_diag(d):
        os.environ["MS_KV_DIAG"] = str(d)
        return measure(msaq)
    t1 = t_diag(1); t2 = t_diag(2); t0 = t_diag(0)
    os.environ["MS_KV_DIAG"] = "0"
    tx = measure(mx)
    # UB bytes/block: upper plane is the dominant HBM traffic
    UB = 32 * (8 - u) // 8
    SB = ((32 // gs) * u + 7) // 8
    ub_mx = 32  # MXINT8 int8 mantissa = 1 byte/elem
    print(f"\n=== MSAQ u{u} gs{gs}  (UB={UB}+SB={SB}={UB+SB} B/blk vs MXINT8 {ub_mx} B/blk) ===")
    print(f"  diag1 memory ceiling   : {t1*1000:7.2f} us")
    print(f"  diag2 +unpack(no exp)  : {t2*1000:7.2f} us   (unpack = {(t2-t1)*1000:+.2f})")
    print(f"  diag0 full (+exp)      : {t0*1000:7.2f} us   (exp    = {(t0-t2)*1000:+.2f})")
    print(f"  MXINT8 baseline (full) : {tx*1000:7.2f} us")
    print(f"  --> MSAQ/MXINT8 = {t0/tx:.2f}x   | memory share of MSAQ = {t1/t0*100:.0f}%")
    print(f"  --> breakdown: mem {t1/t0*100:4.0f}% | unpack {(t2-t1)/t0*100:4.0f}% | exp+rest {(t0-t2)/t0*100:4.0f}%")
    return t0, tx

if __name__ == "__main__":
    torch.cuda.init()
    print(f"[kv_diag] {torch.cuda.get_device_name(0)}")
    for u, gs in [(4, 8), (3, 8), (2, 4)]:
        run(u, gs)
