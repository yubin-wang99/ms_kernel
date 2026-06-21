"""Packing-friendliness of KV-decode read across the ROBUST single-level configs.
Question (user): does a less-aggressive but packing-friendly config (u4/gs2, nibble-aligned
unshared) give a clearer inference-time win than the most-aggressive robust config
(u3/gs32, smaller footprint but 5-bit fields straddling bytes -> extraction)?

Accuracy (from precision/ KV boundary, wikitext-2, 3% bar): u3/gs32 +1.24%, u3/gs8 ~+2%,
u4/gs2 +2.72% (all robust); u4/gs8 NOT robust (+5%, shown only as the speed ceiling).
Run: CUDA_VISIBLE_DEVICES=0 python tests/kv_pack_bench.py
"""
import os, time, numpy as np, torch
from ms_lib.pack import pack_kv, pack_kv_mxint8
from ms_lib import ops; assert ops.available()

def _cuda(a): return torch.from_numpy(a).cuda()
def _t(fn, it=300):
    s = torch.cuda.Event(True); e = torch.cuda.Event(True)
    torch.cuda.synchronize(); s.record()
    for _ in range(it): fn()
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) / it
def warm(fns, sec=3.0):
    t0 = time.time()
    while time.time() - t0 < sec:
        for f in fns: f()

# (u, gs, accuracy-note)
CFGS = [
    (2, 32, "robust  ~+0.5%"),
    (3, 32, "robust  +1.24% (most-aggressive robust)"),
    (3,  8, "robust  ~+2%"),
    (4,  2, "robust  +2.72% (nibble / packing-friendly)"),
    (4,  8, "NOT robust +5% (speed ceiling)"),
]

def run(H=8, Lk=16384, D=128):
    rng = np.random.default_rng(3)
    K = (rng.standard_normal((H, Lk, D)) * 0.5).astype(np.float32)
    V = (rng.standard_normal((H, Lk, D)) * 0.5).astype(np.float32)
    q = (rng.standard_normal((H, D)) * 0.5).astype(np.float32)
    qt = torch.from_numpy(q).to(torch.bfloat16).cuda()
    pxK, pxV = pack_kv_mxint8(K), pack_kv_mxint8(V)
    kxs, kxq = _cuda(pxK["scale_exp"]), _cuda(pxK["qweight"])
    vxs, vxq = _cuda(pxV["scale_exp"]), _cuda(pxV["qweight"])
    nbx = pxK["nb"]
    mx = lambda: torch.ops.msaq.mxint8_kv_decode(qt, kxs, kxq, vxs, vxq, H, H, Lk, D, nbx)
    bytes_mx = H * nbx * Lk * 32 * 2 + H * nbx * Lk * 2
    bw = lambda t_ms, nb: nb / (t_ms * 1e-3) / 1e9
    os.environ["MS_KV_WIDE"] = "1"

    print(f"=== KV decode  H{H} Lk{Lk} D{D}  (MXINT8 footprint {bytes_mx/1e6:.1f} MB) ===")
    warm([mx]); tmx = min(_t(mx), _t(mx))
    print(f"  {'config':14s} {'B/blk':>5} {'foot':>5} | {'time':>8} {'GB/s':>6} {'vs MX':>7} | accuracy")
    print(f"  {'MXINT8':14s} {'32':>5} {'1.00x':>5} | {tmx*1e3:7.2f}us {bw(tmx,bytes_mx):6.0f} {'1.00x':>7} | exact")
    for u, gs, note in CFGS:
        pmK, pmV = pack_kv(K, u, gs), pack_kv(V, u, gs)
        ks, ku, kh = _cuda(pmK["scale_exp"]), _cuda(pmK["upper"]), _cuda(pmK["shared"])
        vs, vu, vh = _cuda(pmV["scale_exp"]), _cuda(pmV["upper"]), _cuda(pmV["shared"])
        nb = pmK["nb"]
        msaq = lambda ks=ks, ku=ku, kh=kh, vs=vs, vu=vu, vh=vh, nb=nb, u=u, gs=gs: \
            torch.ops.msaq.kv_decode_attention(qt, ks, ku, kh, vs, vu, vh, H, H, Lk, D, nb, u, gs)
        UB = 32 * (8 - u) // 8; SB = ((32 // gs) * u + 7) // 8
        nbytes = H * nb * Lk * (UB + SB) * 2 + H * nb * Lk * 2
        warm([msaq, mx]); t = min(_t(msaq), _t(msaq))
        print(f"  u{u}/gs{gs:<2} {'':6s} {UB+SB:>5} {nbytes/bytes_mx:.2f}x | "
              f"{t*1e3:7.2f}us {bw(t,nbytes):6.0f} {t/tmx:6.2f}x | {note}")

if __name__ == "__main__":
    torch.cuda.init()
    print(f"[kv_pack_bench] {torch.cuda.get_device_name(0)}\n")
    run(Lk=16384)
    print()
    run(Lk=16384, H=16)
