"""FUSED online K/Q-rotation: TRUE marginal cost on the decode hot path.

Standalone rotation (rot_kv_bench.py) was launch-bound at ~9us/launch — that is the
launch tax, not the math. Here the rotation rides EXISTING decode launches:
  * K-rotation fused into kv_append      -> marginal = kv_append_rot - kv_append
  * Q-rotation fused into attn prologue  -> marginal = attn(MS_KV_QROT=1) - attn(=0)
So the deltas below are the real added cost (no extra launch).

Correctness (end-to-end): rotating BOTH Q and K preserves QK^T (V un-rotated), so the
attention output with (qrot=1 + H-rotated K cache) must match the no-rotation output.
Plus kv_append_rot must match a host-rotated kv_append.

Run: CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python tests/rot_fused_bench.py
"""
import os, sys, numpy as np, torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from scipy.linalg import hadamard
from ms_lib.pack import pack_kv
from ms_lib import ops; assert ops.available()
from conftest import rel_fro
OPS = torch.ops.msaq
DEV = "cuda"
Hn = (hadamard(128).astype(np.float32) / np.sqrt(128.0))   # orthonormal H128


def bf16(a):  # f32 view of the bf16-rounded array (what the kernel actually consumes)
    return torch.from_numpy(a).to(torch.bfloat16).float().numpy()


def _t(fn, it=300):
    s = torch.cuda.Event(True); e = torch.cuda.Event(True)
    torch.cuda.synchronize(); s.record()
    for _ in range(it): fn()
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) / it  # ms


def stack_planes(packs, keys):
    return {k: torch.from_numpy(np.stack([p[k] for p in packs])).cuda() for k in keys}


def correctness(u=4, gs=8, Hq=32, Hkv=8, D=128, Lk=200):
    """Two ISOLATED checks (quant noise can't masquerade as a rotation bug):
      (A) Q-prologue FWHT: same rotated-K cache, kernel-rotated q vs host-rotated q.
          No quant divergence (cache identical) -> must be tight (bf16/FWHT order only).
      (B) append_rot quant path: decode over append_rot cache vs over pack_kv(bf16(K)@H)
          cache, same un-rotated q. Both quantize ~same rotated K; diff is FWHT-vs-matmul
          fp32 order at quant boundaries -> small."""
    nb = D // 32
    rng = np.random.default_rng(0)
    Kc = (rng.standard_normal((Hkv, Lk, D)) * 0.5).astype(np.float32)
    Vc = (rng.standard_normal((Hkv, Lk, D)) * 0.5).astype(np.float32)
    qn = (rng.standard_normal((Hq, D)) * 0.5).astype(np.float32)
    def planes(p): return {k: torch.from_numpy(p[k]).unsqueeze(0).cuda() for k in ("scale_exp", "upper", "shared")}
    def attn_b(qnp, mkp, mvp, qrot):
        os.environ["MS_KV_QROT"] = str(qrot)
        qb = torch.from_numpy(qnp).to(torch.bfloat16).unsqueeze(0).cuda()
        return OPS.kv_decode_attention_batched(qb, mkp["scale_exp"], mkp["upper"], mkp["shared"],
            mvp["scale_exp"], mvp["upper"], mvp["shared"], 1, Hq, Hkv, Lk, D, nb, u, gs).float().cpu().numpy()

    mv = planes(pack_kv(Vc, u, gs))
    mkr = planes(pack_kv((Kc.reshape(-1, D) @ Hn).reshape(Hkv, Lk, D), u, gs))   # rotated-K cache
    # (A) identical cache; rotate q in-kernel vs on host
    got = attn_b(qn, mkr, mv, 1)
    ref = attn_b(qn @ Hn, mkr, mv, 0)
    rA = rel_fro(got, ref)
    print(f"(A) Q-prologue FWHT  : kernel-qrot vs host-qrot (same cache)  rel_fro {rA:.2e} "
          f"{'PASS' if rA < 2e-2 else 'FAIL'}")

    # (B) append_rot cache vs host-rotated pack_kv cache, same un-rotated q (qrot=0)
    # cache stride == Lk so it matches pack_kv planes (decode default Lcap=Lk).
    ks = torch.empty((Hkv, nb, Lk), dtype=torch.int8, device=DEV)
    UB, SB = 32 * (8 - u) // 8, ((32 // gs) * u + 7) // 8
    ku = torch.empty((Hkv, nb, Lk, UB), dtype=torch.uint8, device=DEV)
    kh = torch.empty((Hkv, nb, Lk, SB), dtype=torch.uint8, device=DEV)
    Kb = torch.from_numpy(Kc).to(torch.bfloat16).cuda()
    for p in range(Lk):
        OPS.kv_append_rot(Kb[:, p, :].contiguous(), ks, ku, kh, Hkv, D, nb, p, Lk, u, gs)
    ca = {"scale_exp": ks.unsqueeze(0), "upper": ku.unsqueeze(0), "shared": kh.unsqueeze(0)}
    # oracle host path: kernel rotates bf16(K) in fp32 -> match with bf16(Kc)@Hn
    ch = planes(pack_kv((bf16(Kc).reshape(-1, D) @ Hn).reshape(Hkv, Lk, D), u, gs))
    gotB = attn_b(qn, ca, mv, 0)
    refB = attn_b(qn, ch, mv, 0)
    rB = rel_fro(gotB, refB)
    print(f"(B) append_rot path  : kernel-append_rot vs host pack_kv(bf16(K)@H)  rel_fro {rB:.2e} "
          f"{'PASS' if rB < 2e-2 else 'FAIL'}\n")
    os.environ["MS_KV_QROT"] = "0"


def bench_append(Hkv=8, D=128, u=4, gs=8, Lcap=4096, pos=2000):
    nb = D // 32
    p = pack_kv(np.zeros((Hkv, Lcap, D), np.float32), u, gs)
    c = {k: torch.from_numpy(p[k]).cuda() for k in ("scale_exp", "upper", "shared")}
    X = torch.randn(Hkv, D, dtype=torch.bfloat16, device=DEV)
    plain = lambda: OPS.kv_append(X, c["scale_exp"], c["upper"], c["shared"], Hkv, D, nb, pos, Lcap, u, gs)
    rot   = lambda: OPS.kv_append_rot(X, c["scale_exp"], c["upper"], c["shared"], Hkv, D, nb, pos, Lcap, u, gs)
    plain(); rot()
    tp = min(_t(plain), _t(plain)); tr = min(_t(rot), _t(rot))
    print(f"K-append  : plain {tp*1e3:6.2f}us | +rot {tr*1e3:6.2f}us | MARGINAL {(tr-tp)*1e3:+6.2f}us")


def bench_attn(B, Lk, Hq=32, Hkv=8, D=128, u=4, gs=8, seed=0):
    rng = np.random.default_rng(seed)
    nb = D // 32
    K = (rng.standard_normal((B, Hkv, Lk, D)) * 0.5).astype(np.float32)
    V = (rng.standard_normal((B, Hkv, Lk, D)) * 0.5).astype(np.float32)
    q = torch.from_numpy((rng.standard_normal((B, Hq, D)) * 0.5).astype(np.float32)).to(torch.bfloat16).cuda()
    mk = stack_planes([pack_kv(K[b], u, gs) for b in range(B)], ("scale_exp", "upper", "shared"))
    mv = stack_planes([pack_kv(V[b], u, gs) for b in range(B)], ("scale_exp", "upper", "shared"))
    def run(qrot):
        os.environ["MS_KV_QROT"] = str(qrot)
        return lambda: OPS.kv_decode_attention_batched(q, mk["scale_exp"], mk["upper"], mk["shared"],
            mv["scale_exp"], mv["upper"], mv["shared"], B, Hq, Hkv, Lk, D, nb, u, gs)
    f0, f1 = run(0), run(1)
    os.environ["MS_KV_QROT"] = "0"; f0()  # warm
    os.environ["MS_KV_QROT"] = "1"; f1()
    os.environ["MS_KV_QROT"] = "0"; t0 = min(_t(f0), _t(f0))
    os.environ["MS_KV_QROT"] = "1"; t1 = min(_t(f1), _t(f1))
    os.environ["MS_KV_QROT"] = "0"
    print(f"B={B:>2} Lk={Lk:>5} | attn {t0*1e3:8.2f}us | +Qrot {t1*1e3:8.2f}us | "
          f"MARGINAL {(t1-t0)*1e3:+6.2f}us ({100*(t1-t0)/t0:+5.2f}%)")


if __name__ == "__main__":
    print("=== correctness ===")
    correctness()
    print("=== K-rotation fused into kv_append (true marginal) ===")
    bench_append()
    print("\n=== Q-rotation fused into attn prologue (true marginal) ===")
    for B in (1, 8, 32):
        for Lk in (1024, 4096, 16384):
            bench_attn(B, Lk)
        print()
