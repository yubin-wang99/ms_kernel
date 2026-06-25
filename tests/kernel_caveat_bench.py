#!/usr/bin/env python3
"""Per-kernel latency + PHASE breakdown vs BF16 / MXINT8, swept over batch.

For each kernel kind it times the three implementations (BF16 cuBLAS/SDPA, MXINT8, MSAQ) and isolates the
sub-phases that are separately callable as ops:
  * weight UNPACK/DEQUANT : ms_dequant_bf16 (MSAQ)  vs  mxint8_dequant_bf16 (MXINT8)
  * activation QUANT       : quant_act (MSAQ-s runtime activation quant)
  * KV PACK (append)       : kv_append (MSAQ)        vs  mxint8_kv_append (MXINT8)
The fused kernels overlap unpack with compute, so (unpack+quant) can EXCEED total — that gap *is* the
overlap, and is reported. Robust per-scope configs: W-only u3/gs16, W+A u2/gs8, KV u4/gs2.
Run: CUDA_VISIBLE_DEVICES=0 python tests/kernel_caveat_bench.py > tests/kernel_caveat_bench.txt 2>&1
"""
import sys, os, numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, "."); sys.path.insert(0, "..")
from ms_lib.pack import pack_weight, pack_weight_mxint8, pack_kv, pack_kv_mxint8
from ms_lib import ops; assert ops.available()
OPS = torch.ops.msaq
def C(a): return torch.from_numpy(a).cuda()

def T(fn, iters=200, warm=30):
    for _ in range(warm): fn()
    torch.cuda.synchronize(); s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    s.record()
    for _ in range(iters): fn()
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) / iters     # ms

OUT = K = 4096

def weight(u, gs, seed=0):
    rng = np.random.default_rng(seed)
    W = (rng.standard_normal((OUT, K)) * rng.uniform(0.2, 4.0, (OUT, 1))).astype(np.float32)
    pm, px = pack_weight(W, u, gs), pack_weight_mxint8(W)
    return W, pm, px

def planes(pm, px):
    s = C(pm["scale_exp"]); upc = C(pm["upper_cm"]); shc = C(pm["shared_cm"])
    up = C(pm["upper"]); sh = C(pm["shared"])
    sx = C(px["scale_exp"]); qw = C(px["qweight"]); qwc = C(px["qweight_cm"])
    return s, upc, shc, up, sh, sx, qw, qwc, pm["nb"], px["nb"]

# ───────────────────────── weight matmul: decode (GEMV/batched) ────────────────
def decode_family(tag, u, gs, wa=False):
    W, pm, px = weight(u, gs)
    s, upc, shc, up, sh, sx, qw, qwc, nb, nbx = planes(pm, px)
    Wbf = C(W).to(torch.bfloat16)
    # isolated phases (M-independent for weight unpack; quant scales with M)
    t_unp_ms = T(lambda: OPS.ms_dequant_bf16(s, upc, shc, OUT, K, nb, u, gs))
    t_unp_mx = T(lambda: OPS.mxint8_dequant_bf16(sx, qwc, OUT, K, nbx))
    print(f"\n## {tag}  (OUT={OUT}, K={K}, u{u}/gs{gs})  [decode family]")
    print(f"   weight-unpack isolated:  MSAQ ms_dequant {t_unp_ms*1e3:7.1f}us | "
          f"MXINT8 dequant {t_unp_mx*1e3:7.1f}us")
    print(f"   {'M':>4} | {'BF16':>8} {'MXINT8':>8} {'MSAQ':>8} | {'ms/bf':>6} {'ms/mq':>6} | "
          f"{'quant(ms)':>9}")
    for M in (1, 4, 16, 32, 64):
        X = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
        x1 = X[0].contiguous()
        if wa:
            tq = T(lambda: OPS.quant_act(X, M, K, nb, u, gs))
            if M == 1:
                ms = T(lambda: OPS.wa_gemv(x1, s, upc, shc, OUT, nb, u, gs))
                mx = T(lambda: OPS.mxint8_wa_gemv(x1, sx, qw, OUT, nbx))
            else:
                ms = T(lambda: OPS.wa_gemv_batched(X, s, upc, shc, M, OUT, nb, u, gs))
                mx = T(lambda: OPS.mxint8_wa_gemv_batched(X, sx, qw, M, OUT, nbx))
        else:
            tq = 0.0
            if M == 1:
                ms = T(lambda: OPS.wonly_gemv_wide(x1, s, upc, shc, OUT, nb, u, gs))
                mx = T(lambda: OPS.mxint8_gemv(x1, sx, qw, OUT, nbx))
            else:
                ms = T(lambda: OPS.wonly_gemv_batched(X, s, upc, shc, M, OUT, nb, u, gs))
                mx = T(lambda: OPS.mxint8_gemv_batched(X, sx, qw, M, OUT, nbx))
        bf = T(lambda: X @ Wbf.t())
        print(f"   {M:>4} | {bf*1e3:8.1f} {mx*1e3:8.1f} {ms*1e3:8.1f} | {ms/bf:6.2f} {ms/mx:6.2f} | "
              f"{tq*1e3:9.1f}")

# ───────────────────────── weight matmul: prefill (GEMM) ──────────────────────
def prefill_family(tag, u, gs, wa=False):
    W, pm, px = weight(u, gs)
    s, upc, shc, up, sh, sx, qw, qwc, nb, nbx = planes(pm, px)
    Wbf = C(W).to(torch.bfloat16)
    print(f"\n## {tag}  (OUT={OUT}, K={K}, u{u}/gs{gs})  [prefill GEMM family]")
    print(f"   {'M':>4} | {'BF16':>8} {'MXINT8':>8} {'MSAQ':>8} | {'ms/bf':>6} {'ms/mq':>6} | "
          f"{'quant(ms)':>9}")
    for M in (128, 256, 512, 1024):
        X = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
        if wa:
            tq = T(lambda: OPS.quant_act(X, M, K, nb, u, gs))
            ms = T(lambda: OPS.wa_gemm_cm(X, s, upc, shc, M, OUT, K, nb, u, gs))
            mx = T(lambda: OPS.mxint8_wa_gemm(X, sx, qw, M, OUT, K, nbx))
        else:
            tq = 0.0
            ms = T(lambda: OPS.wonly_gemm_tc(X, s, upc, shc, M, OUT, K, nb, u, gs))
            mx = T(lambda: OPS.mxint8_gemm(X, sx, qw, M, OUT, K, nbx))
        bf = T(lambda: X @ Wbf.t())
        print(f"   {M:>4} | {bf*1e3:8.1f} {mx*1e3:8.1f} {ms*1e3:8.1f} | {ms/bf:6.2f} {ms/mx:6.2f} | "
              f"{tq*1e3:9.1f}")

# ───────────────────────── KV-cache decode attention (GQA Hq=32/Hkv=8) ─────────
def stackp(packs, keys):
    return {k: torch.from_numpy(np.stack([p[k] for p in packs])).cuda() for k in keys}

def kv_family(u, gs, Hq=32, Hkv=8):
    D = 128; g = Hq // Hkv
    print(f"\n## KV-cache decode attention  (Hq={Hq}, Hkv={Hkv}, D={D}, u{u}/gs{gs})  [per-seq KV cache]")
    print(f"   {'Lk':>6} {'B':>3} | {'BF16':>8} {'MXINT8':>8} {'MSAQ':>8} | {'ms/bf':>6} {'ms/mq':>6} | "
          f"{'append(ms)':>10} {'append(mx)':>10}")
    for Lk, Bs in ((2048, (1, 8, 32)), (16384, (1, 8))):
        rng = np.random.default_rng(3)
        for B in Bs:
            Kf = (rng.standard_normal((B, Hkv, Lk, D)) * 0.5).astype(np.float32)
            Vf = (rng.standard_normal((B, Hkv, Lk, D)) * 0.5).astype(np.float32)
            mk = stackp([pack_kv(Kf[b], u, gs) for b in range(B)], ("scale_exp", "upper", "shared"))
            mv = stackp([pack_kv(Vf[b], u, gs) for b in range(B)], ("scale_exp", "upper", "shared"))
            xk = stackp([pack_kv_mxint8(Kf[b]) for b in range(B)], ("scale_exp", "qweight"))
            xv = stackp([pack_kv_mxint8(Vf[b]) for b in range(B)], ("scale_exp", "qweight"))
            nb = D // 32
            q = torch.randn(B, Hq, D, device="cuda", dtype=torch.bfloat16)
            ms = T(lambda: OPS.kv_decode_attention_batched(q, mk["scale_exp"], mk["upper"], mk["shared"],
                       mv["scale_exp"], mv["upper"], mv["shared"], B, Hq, Hkv, Lk, D, nb, u, gs, Lk))
            mx = T(lambda: OPS.mxint8_kv_decode_batched(q, xk["scale_exp"], xk["qweight"],
                       xv["scale_exp"], xv["qweight"], B, Hq, Hkv, Lk, D, nb, Lk))
            Kb = C(Kf).to(torch.bfloat16).repeat_interleave(g, 1)   # [B,Hq,Lk,D]
            Vb = C(Vf).to(torch.bfloat16).repeat_interleave(g, 1)
            bf = T(lambda: F.scaled_dot_product_attention(q.unsqueeze(2), Kb, Vb))
            ap = apx = f"{'':>10}"
            if B == 1:   # append (packing) one token into a single-seq cache
                xn = torch.randn(Hkv, D, device="cuda", dtype=torch.bfloat16)
                ks, ku, kh = mk["scale_exp"][0], mk["upper"][0], mk["shared"][0]
                kxs, kxq = xk["scale_exp"][0], xk["qweight"][0]
                ap = f"{T(lambda: OPS.kv_append(xn, ks, ku, kh, Hkv, D, nb, 0, Lk, u, gs)) * 1e3:10.1f}"
                apx = f"{T(lambda: OPS.mxint8_kv_append(xn, kxs, kxq, Hkv, D, nb, 0, Lk)) * 1e3:10.1f}"
            print(f"   {Lk:>6} {B:>3} | {bf*1e3:8.1f} {mx*1e3:8.1f} {ms*1e3:8.1f} | {ms/bf:6.2f} {ms/mx:6.2f} | {ap} {apx}")

if __name__ == "__main__":
    print(f"[caveat] {torch.cuda.get_device_name(0)} | torch {torch.__version__} | times in microseconds")
    decode_family("W-only", 3, 16, wa=False)
    decode_family("W+A",    2, 8,  wa=True)
    prefill_family("W-only", 3, 16, wa=False)
    prefill_family("W+A",    2, 8,  wa=True)
    kv_family(4, 2, Hq=32, Hkv=8)
