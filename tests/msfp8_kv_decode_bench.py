#!/usr/bin/env python3
"""MXFP8-MSAQ (E3M4) KV decode vs MXINT8-MSAQ KV decode — MSAQ-vs-MSAQ (NOT plain MXINT8).

Both formats use the SAME packed layout at a given (u,gs): upper field 8-u bits, shared u-bit,
E8M0 scale -> IDENTICAL stored bytes / plane sizes / DRAM traffic. So this bench isolates the
ONLY differences on the KV-read path:
  * per-element DECODE ALU: INT shift+add  vs  FP8 E3M4 exp/mantissa split + ldexpf
  * V handling: INT int8-staged V (v8/vt, integer P·V)  vs  FP8 float P·V (int8-staging can't port)
K-dot: INT (u,gs)-specialized stream + sepsc  vs  FP8 stream_block_uspec_fp8_e3m4 (no sepsc).

Comparable (u,gs): E3M4 has mb=4 -> u in {1,2,3}; INT MSAQ has u in {2,3,4}. Overlap u in {2,3}.
We use the SPECIALIZED gs in {8,16} (post-fix INT v8 path; gs16 is the deployed config).

Run: CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. python tests/msfp8_kv_decode_bench.py
"""
import os, time, numpy as np, torch
from ms_lib.pack import pack_kv, pack_kv_msfp8, dequant_weight, dequant_weight_msfp8
from ms_lib.reference import kv_attention, kv_attention_msfp8
from ms_lib import ops; assert ops.available()
OPS = torch.ops.msaq


def _t(fn, it=100):
    s = torch.cuda.Event(True); e = torch.cuda.Event(True)
    torch.cuda.synchronize(); s.record()
    for _ in range(it): fn()
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) / it


def cross(fa, fb, warm=3.0):
    t0 = time.time()
    while time.time() - t0 < warm: fa(); fb()
    return min(_t(fa), _t(fa)), min(_t(fb), _t(fb))


def stack_planes(packs, keys):
    return {k: torch.from_numpy(np.stack([p[k] for p in packs])).cuda() for k in keys}


def bytes_kv(B, Hkv, nb, Lk, u, gs):
    UB = 32 * (8 - u) // 8; SB = ((32 // gs) * u + 7) // 8
    return B * Hkv * nb * Lk * (UB + SB) * 2 + B * Hkv * nb * Lk * 2   # K+V planes + scales


def bw(t_ms, n): return n / (t_ms * 1e-3) / 1e9


# ---- correctness sanity: each kernel vs its OWN numpy oracle (different formats) ----
def check(u, gs, Hq=8, Hkv=2, Lk=96, D=128, seed=7):
    rng = np.random.default_rng(seed); nb = D // 32; g = Hq // Hkv
    K = (rng.standard_normal((Hkv, Lk, D)) * 0.5).astype(np.float32)
    V = (rng.standard_normal((Hkv, Lk, D)) * 0.5).astype(np.float32)
    Q = (rng.standard_normal((Hq, D)) * 0.5).astype(np.float32)
    qt = torch.from_numpy(Q).to(torch.bfloat16).cuda()
    bf16 = lambda a: torch.from_numpy(a).to(torch.bfloat16).float().numpy()
    out = {}
    for tag, packfn, op, oracle, dq in (
        ("INT ", pack_kv, OPS.kv_decode_attention, kv_attention, dequant_weight),
        ("FP8 ", pack_kv_msfp8, OPS.msfp8_kv_decode_attention, kv_attention_msfp8, dequant_weight_msfp8)):
        pK, pV = packfn(K, u, gs), packfn(V, u, gs)
        pl = lambda p: [torch.from_numpy(p[k]).cuda() for k in ("scale_exp", "upper", "shared")]
        ks, ku, kh = pl(pK); vs, vu, vh = pl(pV)
        got = op(qt, ks, ku, kh, vs, vu, vh, Hq, Hkv, Lk, D, nb, u, gs).float().cpu().numpy()
        # GQA reference: q head h -> kv head h//g
        ref = np.zeros((Hq, D), np.float64)
        import math
        for h in range(Hq):
            Kd, Vd = dq(pK["_per"][h // g]), dq(pV["_per"][h // g])
            sc = (bf16(Q[h]).astype(np.float64) @ Kd.T) / math.sqrt(D)
            p = np.exp(sc - sc.max()); p /= p.sum(); ref[h] = p @ Vd
        rel = np.linalg.norm(got - ref) / np.linalg.norm(ref)
        out[tag] = rel
    print(f"  u{u}/gs{gs}: INT rel_fro {out['INT ']:.2e} | FP8 rel_fro {out['FP8 ']:.2e}  "
          f"{'OK' if max(out.values()) < 2e-2 else 'FAIL'}")


def setup(B, u, gs, Hq=32, Hkv=8, Lk=4096, D=128, seed=0):
    rng = np.random.default_rng(seed); nb = D // 32
    K = (rng.standard_normal((B, Hkv, Lk, D)) * 0.5).astype(np.float32)
    V = (rng.standard_normal((B, Hkv, Lk, D)) * 0.5).astype(np.float32)
    q = (rng.standard_normal((B, Hq, D)) * 0.5).astype(np.float32)
    iK = [pack_kv(K[b], u, gs) for b in range(B)];        iV = [pack_kv(V[b], u, gs) for b in range(B)]
    fK = [pack_kv_msfp8(K[b], u, gs) for b in range(B)];  fV = [pack_kv_msfp8(V[b], u, gs) for b in range(B)]
    KEYS = ("scale_exp", "upper", "shared")
    ik = stack_planes(iK, KEYS); iv = stack_planes(iV, KEYS)
    fk = stack_planes(fK, KEYS); fv = stack_planes(fV, KEYS)
    qt = torch.from_numpy(q).to(torch.bfloat16).cuda()
    iop = lambda: OPS.kv_decode_attention_batched(qt, ik["scale_exp"], ik["upper"], ik["shared"],
              iv["scale_exp"], iv["upper"], iv["shared"], B, Hq, Hkv, Lk, D, nb, u, gs)
    fop = lambda: OPS.msfp8_kv_decode_attention_batched(qt, fk["scale_exp"], fk["upper"], fk["shared"],
              fv["scale_exp"], fv["upper"], fv["shared"], B, Hq, Hkv, Lk, D, nb, u, gs)
    nbytes = bytes_kv(B, Hkv, nb, Lk, u, gs)
    return dict(iop=iop, fop=fop, nbytes=nbytes)


def run(u, gs, Lk=4096):
    print(f"\n=== Hq32 Hkv8 D128 Lk{Lk}  u{u}/gs{gs}  (lower us better; ratio=FP8/INT, both MSAQ) ===")
    for B in (1, 8, 32):
        d = setup(B, u, gs, Lk=Lk)
        ti, tf = cross(d["iop"], d["fop"])
        print(f"  B={B:2d}: INT-MSAQ {ti*1e3:8.1f}us ({bw(ti,d['nbytes']):4.0f} GB/s) | "
              f"FP8-MSAQ {tf*1e3:8.1f}us ({bw(tf,d['nbytes']):4.0f} GB/s) | "
              f"ratio {tf/ti:.3f} {'FP8 faster' if tf < ti else 'INT faster'}")
        del d; torch.cuda.empty_cache()


if __name__ == "__main__":
    torch.cuda.init()
    print("[correctness — each kernel vs its own numpy oracle]")
    for u, gs in [(3, 16), (3, 8), (2, 16), (2, 8)]:
        check(u, gs)
    print("\n[latency — MXINT8-MSAQ vs MXFP8-MSAQ at matched (u,gs); identical bytes -> isolates decode/V-path]")
    for u, gs in [(3, 16), (3, 8), (2, 16), (2, 8)]:
        run(u, gs)
