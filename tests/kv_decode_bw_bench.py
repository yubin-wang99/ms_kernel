#!/usr/bin/env python3
"""KV-cache decode: MSAQ vs MXINT8 — latency, ACHIEVED bandwidth, and byte ratio.

Decode attention reads the whole KV cache each step -> bandwidth-bound. If MSAQ stores X% fewer
bytes, an ideal kernel should be ~1/(1-X) faster. This bench shows (a) the ACTUAL packed-byte ratio
(MS/MX) and (b) the ACTUAL decode-latency ratio, so we can see how much of the byte saving converts
to speed. GQA Hq=32/Hkv=8, D=128.

Run: CUDA_VISIBLE_DEVICES=0 python tests/kv_decode_bw_bench.py
"""
import sys, numpy as np, torch
sys.path.insert(0, "."); sys.path.insert(0, "..")
from ms_lib.pack import pack_kv, pack_kv_mxint8
from ms_lib import ops; assert ops.available()
OPS = torch.ops.msaq

def T(fn, iters=100, warm=20):
    for _ in range(warm): fn()
    torch.cuda.synchronize(); s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    s.record()
    for _ in range(iters): fn()
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) / iters     # ms

def stackp(packs, keys):
    return {k: torch.from_numpy(np.stack([p[k] for p in packs])).cuda() for k in keys}

def nbytes(d, keys):       # bytes of one seq's K (or V) planes
    return sum(int(np.prod(d[k].shape)) * d[k].dtype.itemsize for k in keys)

Hq, Hkv, D = 32, 8, 128
CONFIGS = [(2, 4), (3, 16), (4, 16), (4, 32)]
print(f"GQA Hq={Hq} Hkv={Hkv} D={D} | MXINT8 = 8.25 b/elem baseline")
for Lk in (4096, 16384):
    rng = np.random.default_rng(3)
    for B in (1, 8):
        Kf = (rng.standard_normal((B, Hkv, Lk, D)) * 0.5).astype(np.float32)
        Vf = (rng.standard_normal((B, Hkv, Lk, D)) * 0.5).astype(np.float32)
        nb = D // 32
        q = torch.randn(B, Hq, D, device="cuda", dtype=torch.bfloat16)
        xk = stackp([pack_kv_mxint8(Kf[b]) for b in range(B)], ("scale_exp", "qweight"))
        xv = stackp([pack_kv_mxint8(Vf[b]) for b in range(B)], ("scale_exp", "qweight"))
        mx_bytes = (nbytes(pack_kv_mxint8(Kf[0]), ("scale_exp", "qweight")) +
                    nbytes(pack_kv_mxint8(Vf[0]), ("scale_exp", "qweight"))) * B
        mx = T(lambda: OPS.mxint8_kv_decode_batched(q, xk["scale_exp"], xk["qweight"],
                       xv["scale_exp"], xv["qweight"], B, Hq, Hkv, Lk, D, nb, Lk))
        print(f"\nLk={Lk:>6} B={B}:  MXINT8 {mx*1e3:7.1f}us  ({mx_bytes/mx/1e9*1e3:6.0f} GB/s eff)")
        print(f"   {'cfg':>9} {'bytes MS/MX':>11} {'MS us':>8} {'MX/MS spd':>9} {'GB/s':>7}  (want spd≈1/(MS/MX))")
        for (u, gs) in CONFIGS:
            mk = stackp([pack_kv(Kf[b], u, gs) for b in range(B)], ("scale_exp", "upper", "shared"))
            mv = stackp([pack_kv(Vf[b], u, gs) for b in range(B)], ("scale_exp", "upper", "shared"))
            ms_bytes = (nbytes(pack_kv(Kf[0], u, gs), ("scale_exp", "upper", "shared")) +
                        nbytes(pack_kv(Vf[0], u, gs), ("scale_exp", "upper", "shared"))) * B
            ms = T(lambda: OPS.kv_decode_attention_batched(q, mk["scale_exp"], mk["upper"], mk["shared"],
                           mv["scale_exp"], mv["upper"], mv["shared"], B, Hq, Hkv, Lk, D, nb, u, gs, Lk))
            print(f"   u{u}/gs{gs:<3} {ms_bytes/mx_bytes:>10.3f}  {ms*1e3:7.1f} {mx/ms:>8.3f}x {ms_bytes/ms/1e9*1e3:>6.0f}"
                  f"   ideal {mx_bytes/ms_bytes:.2f}x")
