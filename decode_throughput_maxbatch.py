#!/usr/bin/env python3
"""Decode throughput (tokens/s) at EACH format's MAX-BATCH (capacity frontier).

Combines the two pieces: (1) max-B from capacity (how many seqs fit), (2) decode-step latency at
that B (how fast each step is). throughput = B_max / decode_step_time. Shows that MSAQ runs ~2x the
batch AND the per-step KV read is cheaper -> the capacity advantage converts to real decode tok/s.

One decode LAYER = linear GEMMs (QKV/O/MLP, M=B) + attention (our KV-decode kernel); x num_layers.
Memory: only ONE layer's weights+KV are allocated (timing is per-layer x L) -> high B fits easily.
Linears use bf16 cuBLAS for ALL formats (at B>=16 the deployed path is dequant+cuBLAS, ~format-equal),
so the format-specific cost is the ATTENTION (KV read). Run:
  CUDA_VISIBLE_DEVICES=0 python decode_throughput_maxbatch.py --ctx 1536
"""
import argparse, numpy as np, torch, torch.nn.functional as F
from ms_lib.pack import pack_kv, pack_kv_mxint8
from ms_lib import ops; assert ops.available()
OPS = torch.ops.msaq

# Llama-3.1-8B layer dims
LAYERS, Hq, Hkv, hd, hidden, inter = 32, 32, 8, 128, 4096, 14336
LIN = [("qkv", hidden, hidden + 2 * Hkv * hd), ("o", hidden, hidden),
       ("gate_up", hidden, 2 * inter), ("down", inter, hidden)]   # fused qkv & gate_up (as deployed)

def C(a): return torch.from_numpy(a).cuda()
def T(fn, it, w):
    for _ in range(w): fn()
    torch.cuda.synchronize(); s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    s.record()
    for _ in range(it): fn()
    e.record(); torch.cuda.synchronize(); return s.elapsed_time(e) / it  # ms

def lin_ms(B):
    """one layer's linear GEMMs at M=B (bf16 cuBLAS; common dequant+cuBLAS path at B>=16)."""
    t = 0.0
    for _, K, N in LIN:
        X = torch.randn(B, K, device="cuda", dtype=torch.bfloat16)
        W = torch.randn(N, K, device="cuda", dtype=torch.bfloat16)
        t += T(lambda: X @ W.t(), 50, 10)
    return t

def attn_ms(B, L, fmt, u, gs):
    nb = hd // 32
    rng = np.random.default_rng(0)
    Kf = (rng.standard_normal((B, Hkv, L, hd)) * 0.5).astype(np.float32)
    Vf = (rng.standard_normal((B, Hkv, L, hd)) * 0.5).astype(np.float32)
    q = torch.randn(B, Hq, hd, device="cuda", dtype=torch.bfloat16)
    def sp(packs, keys): return {k: torch.from_numpy(np.stack([p[k] for p in packs])).cuda() for k in keys}
    if fmt == "bf16":
        g = Hq // Hkv
        Kb = C(Kf).to(torch.bfloat16).repeat_interleave(g, 1); Vb = C(Vf).to(torch.bfloat16).repeat_interleave(g, 1)
        return T(lambda: F.scaled_dot_product_attention(q.unsqueeze(2), Kb, Vb), 50, 10)
    if fmt == "mxint8":
        xk = sp([pack_kv_mxint8(Kf[b]) for b in range(B)], ("scale_exp", "qweight"))
        xv = sp([pack_kv_mxint8(Vf[b]) for b in range(B)], ("scale_exp", "qweight"))
        return T(lambda: OPS.mxint8_kv_decode_batched(q, xk["scale_exp"], xk["qweight"], xv["scale_exp"], xv["qweight"],
                                                      B, Hq, Hkv, L, hd, nb, L), 50, 10)
    mk = sp([pack_kv(Kf[b], u, gs) for b in range(B)], ("scale_exp", "upper", "shared"))
    mv = sp([pack_kv(Vf[b], u, gs) for b in range(B)], ("scale_exp", "upper", "shared"))
    return T(lambda: OPS.kv_decode_attention_batched(q, mk["scale_exp"], mk["upper"], mk["shared"],
                     mv["scale_exp"], mv["upper"], mv["shared"], B, Hq, Hkv, L, hd, nb, u, gs, L), 50, 10)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ctx", type=int, default=1536)
    a = ap.parse_args()
    L = a.ctx
    # (name, fmt, u, gs, B_max from capacity_maxbatch.py at this ctx)  -- 1536 defaults; pass others via edit
    BMAX = {1536: {"bf16": 40, "mxint8": 154, "msaq45": 314, "msaq544": 261},
            1152: {"bf16": 53, "mxint8": 204, "msaq45": 417, "msaq544": 346}}.get(L)
    if BMAX is None:
        print(f"# no preset B_max for ctx={L}; run capacity_maxbatch.py --ctx {L} and edit BMAX"); return
    rows = [("bf16 (16/16b)", "bf16", 0, 0, BMAX["bf16"]),
            ("MXINT8 (8.25b)", "mxint8", 0, 0, BMAX["mxint8"]),
            ("MSAQ KV4.5/gs16", "msaq", 4, 16, BMAX["msaq45"]),
            ("MSAQ KV5.44/gs16", "msaq", 3, 16, BMAX["msaq544"])]
    print(f"# {torch.cuda.get_device_name(0)} | Llama-8B, context={L}, {LAYERS} layers")
    print(f"# decode step = {LAYERS} x (linear GEMMs M=B + attention). throughput = B_max / step_time.\n")
    print(f"{'format':>18} {'B_max':>6} {'lin/lyr':>8} {'attn/lyr':>9} {'step(ms)':>9} {'tok/s':>9} | {'tok/s vs MXINT8':>15}")
    base = None
    for nm, fmt, u, gs, B in rows:
        lm = lin_ms(B); am = attn_ms(B, L, fmt, u, gs)
        step = LAYERS * (lm + am)                  # ms per decode step (whole batch advances 1 token)
        toks = B / (step / 1000.0)                 # tokens/s
        if fmt == "mxint8": base = toks
        r = f"{toks/base:.2f}x" if base else "-"
        print(f"{nm:>18} {B:>6} {lm*1e3:>7.1f}u {am*1e3:>8.1f}u {step:>8.2f} {toks:>9.0f} | {r:>15}")
    print(f"\n# tok/s vs MXINT8: combines (more batch) x (per-step speed). >1 = MSAQ delivers more decode throughput")
    print(f"# at the capacity frontier — the capacity win CONVERTED to tokens/s (not just fixed-batch).")

if __name__ == "__main__":
    main()
