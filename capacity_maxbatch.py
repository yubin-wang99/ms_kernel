#!/usr/bin/env python3
"""Measure MAX SERVABLE BATCH per format by ACTUAL GPU allocation (no kernel, no vLLM).

max-batch is a MEMORY-FOOTPRINT question, not a compute one. We allocate, on the real GPU:
  weights (quantized-resident, format's bytes) + KV cache for B seqs at context L (format's bytes)
  + a per-seq decode workspace, and BINARY-SEARCH the largest B that fits before OOM.
This captures real allocator/fragmentation behavior. Assumes CHUNKED prefill (serving standard) so
the prefill-activation peak is bounded and the decode KV+weight footprint is the binding resource.

Run: python capacity_maxbatch.py            (auto-detects GPU; Llama-3.1-8B, context 1536 default)
"""
import argparse, torch

MODELS = {  # params, layers, Hkv, head_dim, hidden
    "llama8b":  (8.03e9, 32, 8, 128, 4096),
    "llama70b": (70.6e9, 80, 8, 128, 8192),
}
# (name, weight_bits, kv_bits)  — MXINT8 and MXFP8(E4M3) are BOTH 8.25b -> identical footprint.
FORMATS = [
    ("MXINT8/MXFP8 (8.25b)", 8.25, 8.25),
    ("MSAQ W6.25/KV4.50",     6.25, 4.50),   # accuracy-safe weights + u4 KV
    ("MSAQ W6.25/KV5.44",     6.25, 5.44),   # + u3/gs16 KV
    ("MSAQ KV-only 4.50",     8.25, 4.50),   # isolate the KV effect (weights = MXINT8)
]

def try_fit(weight_bytes, per_seq_bytes, B, dev):
    """Actually allocate weights + B*per_seq on the GPU. Return True if it fits."""
    bufs = []
    try:
        bufs.append(torch.empty(int(weight_bytes), dtype=torch.uint8, device=dev))
        # allocate per-seq KV+workspace in chunks (mimic paged blocks; avoids one giant alloc)
        remain = int(per_seq_bytes) * B
        chunk = 256 * 1024 * 1024
        while remain > 0:
            n = min(chunk, remain); bufs.append(torch.empty(n, dtype=torch.uint8, device=dev)); remain -= n
        torch.cuda.synchronize()
        ok = True
    except RuntimeError:
        ok = False
    del bufs; torch.cuda.empty_cache()
    return ok

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="llama8b", choices=MODELS)
    ap.add_argument("--ctx", type=int, default=1536, help="context = L_in + L_out")
    ap.add_argument("--act_per_seq_kb", type=float, default=512,
                    help="decode workspace reserved per active seq (chunked-prefill assumption)")
    ap.add_argument("--reserve_gb", type=float, default=1.0, help="fixed runtime/cuda reserve")
    a = ap.parse_args()
    dev = "cuda"
    P, layers, Hkv, hd, hidden = MODELS[a.model]
    total = torch.cuda.get_device_properties(0).total_memory
    name = torch.cuda.get_device_name(0)
    # leave reserve for CUDA context + runtime
    budget = total - a.reserve_gb * 1e9
    kv_elems_tok = 2 * layers * Hkv * hd
    act_per_seq = a.act_per_seq_kb * 1024

    print(f"# {name}  total {total/1e9:.1f}GB  (reserve {a.reserve_gb}GB -> budget {budget/1e9:.1f}GB)")
    print(f"# {a.model}, context={a.ctx} tokens, decode workspace {a.act_per_seq_kb}KB/seq (chunked-prefill)\n")
    print(f"{'format':>22} {'Wgt(GB)':>8} {'KV/seq':>8} | {'B_max':>6}  vs baseline")
    base = None
    for nm, wb, kvb in FORMATS:
        W = P * wb / 8
        kv_seq = kv_elems_tok * kvb / 8 * a.ctx          # bytes/seq for KV at context L
        per_seq = kv_seq + act_per_seq
        # binary search max B that fits (weights + B*per_seq <= budget), then VERIFY by real alloc
        lo, hi = 0, int((budget - W) / per_seq) + 2 if budget > W else 0
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if try_fit(W, per_seq, mid, dev): lo = mid
            else: hi = mid - 1
        Bmax = lo
        if base is None: base = Bmax
        ratio = f"{Bmax/base:.2f}x" if base else "-"
        print(f"{nm:>22} {W/1e9:>8.2f} {kv_seq/1e6:>6.0f}MB | {Bmax:>6}  {ratio}")
    print(f"\n# B_max = largest batch that fits (real GPU alloc). ratio>1 = MSAQ runs batches MXINT8/MXFP8 cannot.")
    print(f"# Sweep context: --ctx; the KV effect grows with context. Isolate KV: 'KV-only' row (weights=MXINT8).")

if __name__ == "__main__":
    main()
