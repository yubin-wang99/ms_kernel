"""Online K/Q-rotation kernel: correctness + added-latency on the decode hot path.

The accuracy study (precision/rot_results.md) found head-dim H128 rotation of KV-KEY
is the structural MSAQ win (kills channel outliers, makes nibble u4 robust). Realizing
it online costs decode-hot-path time: every step the new token's Q [Hq,128] and K
[Hkv,128] must be Hadamard-rotated (post-RoPE, Q mirrored so QK^T is preserved).

This measures that tax: rotation-kernel time vs the kv_decode_attention it is added to,
across context length and batch. STANDALONE rotation = UPPER BOUND (separate launch);
fused into QK epilogue / kv_append it would be cheaper.

Run: CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python tests/rot_kv_bench.py
"""
import numpy as np, torch
from scipy.linalg import hadamard
from ms_lib.pack import pack_kv
from ms_lib import ops; assert ops.available()
OPS = torch.ops.msaq
DEV = "cuda"


def _t(fn, it=200):
    s = torch.cuda.Event(True); e = torch.cuda.Event(True)
    torch.cuda.synchronize(); s.record()
    for _ in range(it): fn()
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) / it  # ms


def correctness():
    H = (hadamard(128).astype(np.float32) / np.sqrt(128.0))   # orthonormal
    rng = np.random.default_rng(0)
    x = (rng.standard_normal((40, 128)) * 0.5).astype(np.float32)
    ref = x @ H                                                # rotate last dim
    got = OPS.hadamard_rotate(torch.from_numpy(x).to(torch.bfloat16).cuda()).float().cpu().numpy()
    rel = np.abs(got - ref).max() / (np.abs(ref).max() + 1e-9)
    # bit-faithful pair check: (Q@H)(K@H)^T == Q@K^T (the property the kernel must keep)
    q = (rng.standard_normal((4, 128)) * 0.5).astype(np.float32)
    k = (rng.standard_normal((6, 128)) * 0.5).astype(np.float32)
    qr = OPS.hadamard_rotate(torch.from_numpy(q).to(torch.bfloat16).cuda()).float().cpu().numpy()
    kr = OPS.hadamard_rotate(torch.from_numpy(k).to(torch.bfloat16).cuda()).float().cpu().numpy()
    pair = np.abs(qr @ kr.T - q @ k.T).max() / (np.abs(q @ k.T).max() + 1e-9)
    print(f"correctness: rel-err vs H-matmul = {rel:.2e} | QK^T-preservation rel-err = {pair:.2e}")
    print(f"            {'PASS' if rel < 2e-2 and pair < 3e-2 else 'FAIL (bf16 tol)'}\n")


def stack_planes(packs, keys):
    return {k: torch.from_numpy(np.stack([p[k] for p in packs])).cuda() for k in keys}


def bench(B, Lk, Hq=32, Hkv=8, D=128, u=4, gs=8, seed=0):
    rng = np.random.default_rng(seed)
    nb = D // 32
    K = (rng.standard_normal((B, Hkv, Lk, D)) * 0.5).astype(np.float32)
    V = (rng.standard_normal((B, Hkv, Lk, D)) * 0.5).astype(np.float32)
    q = torch.from_numpy((rng.standard_normal((B, Hq, D)) * 0.5).astype(np.float32)).to(torch.bfloat16).cuda()
    pmK = [pack_kv(K[b], u, gs) for b in range(B)]; pmV = [pack_kv(V[b], u, gs) for b in range(B)]
    mk = stack_planes(pmK, ("scale_exp", "upper", "shared")); mv = stack_planes(pmV, ("scale_exp", "upper", "shared"))

    # new-token Q/K for this decode step (the only tensors that get rotated)
    qnew = torch.randn(B * Hq, D, dtype=torch.bfloat16, device=DEV)
    knew = torch.randn(B * Hkv, D, dtype=torch.bfloat16, device=DEV)

    attn = lambda: OPS.kv_decode_attention_batched(
        q, mk["scale_exp"], mk["upper"], mk["shared"], mv["scale_exp"], mv["upper"], mv["shared"],
        B, Hq, Hkv, Lk, D, nb, u, gs)
    # full per-step rotation tax: rotate Q (Hq rows) AND K (Hkv rows) of the new token
    rot = lambda: (OPS.hadamard_rotate(qnew), OPS.hadamard_rotate(knew))
    rot_k_only = lambda: OPS.hadamard_rotate(knew)
    qknew = torch.cat([qnew, knew], 0)             # both in ONE launch (fusion proxy)
    rot_fused = lambda: OPS.hadamard_rotate(qknew)

    for f in (attn, rot, rot_k_only, rot_fused): f()  # warmup
    t_attn = min(_t(attn), _t(attn))
    t_rot = min(_t(rot), _t(rot))
    t_k = min(_t(rot_k_only), _t(rot_k_only))
    t_f = min(_t(rot_fused), _t(rot_fused))
    print(f"B={B:>2} Lk={Lk:>5} | attn {t_attn*1e3:7.2f}us | Q+K(2 launch) {t_rot*1e3:6.2f}us "
          f"({100*t_rot/t_attn:4.1f}%) | Q+K(1 launch) {t_f*1e3:6.2f}us ({100*t_f/t_attn:4.1f}%) "
          f"| K-only {t_k*1e3:6.2f}us ({100*t_k/t_attn:4.1f}%)")


if __name__ == "__main__":
    correctness()
    print("decode-step added latency (rot is 2 launches Q+K; K-only is 1) vs kv_decode_attention:\n")
    for B in (1, 8, 32):
        for Lk in (1024, 4096, 16384):
            bench(B, Lk)
        print()
