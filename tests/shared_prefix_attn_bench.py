"""Shared-prefix decode attention (2-pass tensor-core): N requests share one KV cache
(prefix caching / beam) -> M = N*G query rows per kv-head. Q.K WMMA -> softmax ->
P.V WMMA. MSAQ vs MXINT8, across 3 model KV shapes x N x Lk. Token-major KV (accurate).
Run: CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python tests/shared_prefix_attn_bench.py
"""
import time, numpy as np, torch
from ms_lib.pack import pack_kv, pack_kv_mxint8, dequant_weight
from ms_lib import ops; assert ops.available()
OPS = torch.ops.msaq
def cuda(a): return torch.from_numpy(a).cuda()

# (name, Hkv, G, D) — KV shape per model (G = GQA group = Hq/Hkv)
MODELS = [("llama31_8b", 8, 4, 128), ("gemma2_9b", 8, 2, 256), ("mistral_7b", 8, 4, 128)]

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

def softmax_bf16(scores):                      # scores [Hkv, M, Lk] fp32 -> P bf16
    return torch.softmax(scores, dim=-1).to(torch.bfloat16)

def setup(Hkv, M, D, Lk, u=4, gs=8, seed=0):
    rng = np.random.default_rng(seed)
    nb = D // 32
    K = (rng.standard_normal((Hkv, Lk, D)) * 0.5).astype(np.float32)   # shared KV (one per head)
    V = (rng.standard_normal((Hkv, Lk, D)) * 0.5).astype(np.float32)
    Q = (rng.standard_normal((Hkv, M, D)) * 0.5).astype(np.float32)
    Qt = cuda(Q).to(torch.bfloat16)
    pk, pv = pack_kv(K, u, gs), pack_kv(V, u, gs)
    xk, xv = pack_kv_mxint8(K), pack_kv_mxint8(V)
    ks, ku, kh = (cuda(pk[k]) for k in ("scale_exp", "upper", "shared"))
    vs, vu, vh = (cuda(pv[k]) for k in ("scale_exp", "upper", "shared"))
    kxs, kxq = cuda(xk["scale_exp"]), cuda(xk["qweight"])
    vxs, vxq = cuda(xv["scale_exp"]), cuda(xv["qweight"])
    def msaq():
        sc = OPS.qk_wmma(Qt, ks, ku, kh, Hkv, M, D, Lk, nb, u, gs)
        P = softmax_bf16(sc)
        return OPS.pv_wmma(P, vs, vu, vh, Hkv, M, D, Lk, nb, u, gs)
    def mx():
        sc = OPS.qk_wmma_mx(Qt, kxs, kxq, Hkv, M, D, Lk, nb)
        P = softmax_bf16(sc)
        return OPS.pv_wmma_mx(P, vxs, vxq, Hkv, M, D, Lk, nb)
    UB = 32*(8-u)//8; SB = ((32//gs)*u+7)//8
    bm = Hkv*nb*Lk*(UB+SB)*2 + Hkv*nb*Lk*2     # K+V (read once, shared)
    bx = Hkv*nb*Lk*32*2 + Hkv*nb*Lk*2
    return dict(msaq=msaq, mx=mx, Qt=Qt, pk=pk, pv=pv, K=K, V=V, bm=bm, bx=bx, nb=nb)

def check():
    Hkv, G, D, Lk, N = 4, 4, 128, 256, 4
    M = N * G
    d = setup(Hkv, M, D, Lk, seed=5)
    got = d["msaq"]().float().cpu().numpy()         # [Hkv, M, D]
    bad = 0
    for h in range(Hkv):
        Kd = dequant_weight(d["pk"]["_per"][h]); Vd = dequant_weight(d["pv"]["_per"][h])
        sc = (d["Qt"][h].float().cpu().numpy() @ Kd.T) / np.sqrt(D)
        sc -= sc.max(1, keepdims=True); p = np.exp(sc); p /= p.sum(1, keepdims=True)
        ref = p @ Vd
        rel = np.linalg.norm(got[h] - ref) / (np.linalg.norm(ref) + 1e-9)
        print(f"  head {h}: 2-pass attn vs ref  rel_fro {rel:.2e}  {'OK' if rel < 3e-2 else 'FAIL'}")
        bad += rel >= 3e-2
    return bad

def bw(t_ms, n): return n / (t_ms * 1e-3) / 1e9

if __name__ == "__main__":
    import os
    torch.cuda.init()
    qk = os.environ.get("MS_QK_SCALAR", "0")
    print(f"[correctness]  (Q.K = {'SCALAR' if qk=='1' else 'WMMA'})"); check()
    print(f"\n[shared-prefix attention, Q.K={'scalar' if qk=='1' else 'wmma'}] M=N*G. ratio=MSAQ/MXINT8 (full attn), lower=MSAQ wins")
    for name, Hkv, G, D in MODELS:
        print(f"\n  --- {name}: Hkv={Hkv} G={G} D={D} ---")
        for Lk in (2048, 4096):
            for N in (8, 16, 32):
                M = N * G
                d = setup(Hkv, M, D, Lk)
                tm, tx = cross(d["msaq"], d["mx"])
                print(f"    Lk{Lk} N={N:2d} (M={M:3d}): MXINT8 {tx*1e3:6.1f}us | MSAQ {tm*1e3:6.1f}us | "
                      f"ratio {tm/tx:.3f} {'WIN' if tm < tx else 'loss'}")
                del d; torch.cuda.empty_cache()
