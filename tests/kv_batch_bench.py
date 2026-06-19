"""KV-read batch/seqlen sweep: batched flash-decode, MSAQ vs MXINT8.
Hypothesis: batch supplies the occupancy the single-token decode lacks (one-wave),
pushing KV read toward BW-bound where MSAQ's ~0.58x bytes convert to ~0.58x time.
Run: CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python tests/kv_batch_bench.py
"""
import os, time, numpy as np, torch
from ms_lib.pack import pack_kv, pack_kv_mxint8
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


def setup(B, u=4, gs=8, Hq=32, Hkv=8, Lk=4096, D=128, seed=0):
    rng = np.random.default_rng(seed)
    nb = D // 32
    K = (rng.standard_normal((B, Hkv, Lk, D)) * 0.5).astype(np.float32)
    V = (rng.standard_normal((B, Hkv, Lk, D)) * 0.5).astype(np.float32)
    q = (rng.standard_normal((B, Hq, D)) * 0.5).astype(np.float32)
    pmK = [pack_kv(K[b], u, gs) for b in range(B)]; pmV = [pack_kv(V[b], u, gs) for b in range(B)]
    pxK = [pack_kv_mxint8(K[b]) for b in range(B)]; pxV = [pack_kv_mxint8(V[b]) for b in range(B)]
    mk = stack_planes(pmK, ("scale_exp", "upper", "shared")); mv = stack_planes(pmV, ("scale_exp", "upper", "shared"))
    xk = stack_planes(pxK, ("scale_exp", "qweight"));         xv = stack_planes(pxV, ("scale_exp", "qweight"))
    qt = torch.from_numpy(q).to(torch.bfloat16).cuda()
    UB = 32 * (8 - u) // 8; SB = ((32 // gs) * u + 7) // 8
    bmsaq = B * Hkv * nb * Lk * (UB + SB) * 2 + B * Hkv * nb * Lk * 2
    bmx   = B * Hkv * nb * Lk * 32 * 2 + B * Hkv * nb * Lk * 2
    msaq = lambda: OPS.kv_decode_attention_batched(qt, mk["scale_exp"], mk["upper"], mk["shared"],
              mv["scale_exp"], mv["upper"], mv["shared"], B, Hq, Hkv, Lk, D, nb, u, gs)
    mx = lambda: OPS.mxint8_kv_decode_batched(qt, xk["scale_exp"], xk["qweight"],
              xv["scale_exp"], xv["qweight"], B, Hq, Hkv, Lk, D, nb)
    return dict(msaq=msaq, mx=mx, bmsaq=bmsaq, bmx=bmx, qt=qt, mk=mk, mv=mv, u=u, gs=gs,
                Hq=Hq, Hkv=Hkv, Lk=Lk, D=D, nb=nb)


def check(u=4, gs=8):
    """batched B=2 slice b must equal the single-token kernel on that slice."""
    d = setup(2, u, gs, Hq=32, Hkv=8, Lk=512, D=128, seed=7)
    got = d["msaq"]().float().cpu().numpy()                       # [B,Hq,D]
    mk, mv = d["mk"], d["mv"]
    for b in range(2):
        sing = OPS.kv_decode_attention(d["qt"][b], mk["scale_exp"][b], mk["upper"][b], mk["shared"][b],
                  mv["scale_exp"][b], mv["upper"][b], mv["shared"][b], 32, 8, 512, 128, d["nb"], u, gs).float().cpu().numpy()
        rel = np.linalg.norm(got[b] - sing) / (np.linalg.norm(sing) + 1e-9)
        print(f"  batched vs single-token, slice {b}: rel_fro {rel:.2e}  {'OK' if rel < 1e-4 else 'FAIL'}")


def bw(t_ms, n): return n / (t_ms * 1e-3) / 1e9


def run(Lk=4096, u=4, gs=8):
    print(f"\n=== batch sweep: Hq32 Hkv8 D128 Lk{Lk} u{u} gs{gs}  (lower us better; ratio=MSAQ/MXINT8) ===")
    for B in (1, 4, 8, 16, 32):
        d = setup(B, u, gs, Hq=32, Hkv=8, Lk=Lk, D=128)
        tm, tx = cross(d["msaq"], d["mx"])
        print(f"  B={B:2d}: MXINT8 {tx*1e3:7.1f}us ({bw(tx,d['bmx']):4.0f} GB/s) | "
              f"MSAQ {tm*1e3:7.1f}us ({bw(tm,d['bmsaq']):4.0f} GB/s) | "
              f"ratio {tm/tx:.3f} {'WIN' if tm < tx else 'loss'}")
        del d; torch.cuda.empty_cache()


if __name__ == "__main__":
    torch.cuda.init()
    print("[correctness]"); check(4, 8); check(2, 8)
    for Lk in (4096,):
        for gs in (8, 32):
            run(Lk=Lk, u=4, gs=gs)
