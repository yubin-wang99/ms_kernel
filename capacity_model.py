#!/usr/bin/env python3
"""Analytical CAPACITY MODEL: KV/weight bit reduction -> more concurrent sequences at fixed HBM
-> higher serving throughput. The paper-grade "real strength" argument the per-op latency numbers
miss (those were fixed-batch). No GPU/kernel needed — pure first-order accounting.

Outputs, per (model, GPU, format):
  - weights footprint, KV bytes/token
  - B_max(L): max concurrent sequences that fit at context length L   [CAPACITY, kernel-free]
  - projected decode throughput at B_max                              [needs a BW assumption]
  - a context-length sweep showing the crossover (short ctx: weights dominate -> tie;
    long ctx: KV dominates -> low-bit KV wins big)

⚠️ FAIRNESS: set each format's (w_bits, kv_bits) to an ISO-ACCURACY point (same PPL/task acc),
NOT iso-bits. The whole argument is "same accuracy, fewer bits -> more capacity". The defaults
below are seeded from this repo's measurements; replace with your accuracy-matched budgets.

Run: python capacity_model.py            (or  --model llama70b --gpu h100 --lout 256)
"""
import argparse

# ---- model presets (GQA): params, layers, kv-heads, head_dim ---------------------------------
MODELS = {
    # name:        (params,   layers, Hkv, head_dim)
    "llama8b":     (8.03e9,   32,     8,   128),
    "llama70b":    (70.6e9,   80,     8,   128),
    "qwen32b":     (32.8e9,   64,     8,   128),
}
# ---- GPU presets: HBM bytes, mem bandwidth (B/s) ---------------------------------------------
GPUS = {
    "h100":   (80e9,  3.35e12),
    "a100":   (80e9,  2.039e12),
    "h200":   (141e9, 4.8e12),
    "b200":   (192e9, 8.0e12),
}
# ---- formats: (name, weight_bits/elem, kv_bits/elem, accuracy note) ---------------------------
#   * bf16 = no quant. MXINT8 = the baseline the reviewer compares to. The rest are this work's
#     iso-accuracy points (EDIT to your accuracy-matched budgets; notes = measured PPL delta).
FORMATS = [
    ("bf16",            16.0, 16.0, "lossless ref"),
    ("MXINT8 (W+KV)",    8.25, 8.25, "~lossless; the reviewer baseline"),
    ("W6.25 / KV8.25",   6.25, 8.25, "MXFP6-E2M3 weights, INT8 KV"),
    ("W6.25 / KV5.44",   6.25, 5.44, "E2M3 W + u3/gs16 KV (+~1% PPL)"),   # this work
    ("W6.25 / KV4.50",   6.25, 4.50, "E2M3 W + u4 KV (+~1.4% PPL)"),       # this work (aggressive KV)
]

def human(b):  # bytes -> GB
    return b / 1e9

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="llama8b", choices=MODELS)
    ap.add_argument("--gpu",   default="h100", choices=GPUS)
    ap.add_argument("--util",  type=float, default=0.90, help="usable HBM fraction (vLLM gpu_mem_util)")
    ap.add_argument("--workspace_gb", type=float, default=2.0, help="reserved non-KV activation/runtime")
    ap.add_argument("--bw_eff", type=float, default=0.70, help="achievable fraction of peak mem BW (decode)")
    ap.add_argument("--lout", type=int, default=256, help="decode tokens per request (for req/s)")
    ap.add_argument("--ctx", default="1024,4096,16384,65536,131072", help="context lengths to sweep")
    a = ap.parse_args()

    P, layers, Hkv, hd = MODELS[a.model]
    HBM, BW = GPUS[a.gpu]
    avail = HBM * a.util - a.workspace_gb * 1e9
    kv_elems_per_token = 2 * layers * Hkv * hd      # K and V, all layers
    Ls = [int(x) for x in a.ctx.split(",")]

    print(f"# Capacity model — {a.model} on {a.gpu} "
          f"(HBM {human(HBM):.0f}GB x util {a.util} - workspace {a.workspace_gb}GB = {human(avail):.1f}GB usable)")
    print(f"# KV = 2*layers*Hkv*head_dim = {kv_elems_per_token:,} elem/token; "
          f"decode BW {BW/1e12:.2f}TB/s x eff {a.bw_eff}; L_out={a.lout}")
    print(f"# ⚠️ compare formats at ISO-ACCURACY (rows below seeded from this repo; edit to your matched budgets)\n")

    def kv_bytes_tok(kv_bits): return kv_elems_per_token * kv_bits / 8.0
    def w_bytes(w_bits):       return P * w_bits / 8.0

    # ---- per-format summary table ----
    print(f"{'format':>18} {'Wgt(GB)':>8} {'KV B/tok':>9} | " + " ".join(f"Bmax@{L//1024}k" if L>=1024 else f"Bmax@{L}" for L in Ls))
    rows = {}
    for nm, wb, kvb, note in FORMATS:
        W = w_bytes(wb); kvt = kv_bytes_tok(kvb)
        bmax = []
        for L in Ls:
            room = avail - W
            b = int(max(0, room) // (L * kvt)) if kvt > 0 and room > 0 else 0
            bmax.append(b)
        rows[nm] = (W, kvt, bmax)
        print(f"{nm:>18} {human(W):>8.1f} {kvt/1024:>8.1f}K | " + " ".join(f"{b:>7,}" for b in bmax))

    # ---- B_max ratio vs MXINT8 (the headline: how many MORE sequences) ----
    base = rows["MXINT8 (W+KV)"][2]
    print(f"\n## B_max ratio vs MXINT8 (>1 = fit more sequences at same accuracy & HBM)")
    print(f"{'format':>18} | " + " ".join(f"{L//1024}k" if L>=1024 else f"{L}" for L in Ls))
    for nm, *_ in [(f[0],) for f in FORMATS]:
        bm = rows[nm][2]
        print(f"{nm:>18} | " + " ".join(f"{(b/bb):>4.2f}x" if bb else "  -  " for b, bb in zip(bm, base)))

    # ---- projected decode throughput at the capacity frontier ----
    #   memory-bound decode step: read weights once + each seq's full KV.
    #   t_step = (W + B*L*kv_bytes/tok) / (BW*eff);  tok/s = B/t_step;  req/s = tok/s / L_out
    print(f"\n## Projected decode throughput at B=B_max  (memory-bound roofline; tok/s | req/s)")
    print(f"{'format':>18} | " + " ".join(f"{L//1024}k" if L>=1024 else f"{L}" for L in Ls))
    tput = {}
    for nm, wb, kvb, note in FORMATS:
        W, kvt, bmax = rows[nm]
        cells, toks = [], []
        for L, B in zip(Ls, bmax):
            if B == 0: cells.append("   OOM   "); toks.append(0.0); continue
            t_step = (W + B * L * kvt) / (BW * a.bw_eff)          # seconds per decode step
            tok_s = B / t_step                                    # tokens/s (whole batch advances 1 tok)
            req_s = tok_s / a.lout
            cells.append(f"{tok_s/1e3:>5.0f}k|{req_s:>4.0f}")
            toks.append(tok_s)
        tput[nm] = toks
        print(f"{nm:>18} | " + " ".join(cells))

    bt = tput["MXINT8 (W+KV)"]
    print(f"\n## Throughput ratio vs MXINT8 (the paper headline — RPS gain from capacity)")
    print(f"{'format':>18} | " + " ".join(f"{L//1024}k" if L>=1024 else f"{L}" for L in Ls))
    for nm, *_ in [(f[0],) for f in FORMATS]:
        tt = tput[nm]
        print(f"{nm:>18} | " + " ".join(f"{(t/b):>4.2f}x" if b else "  -  " for t, b in zip(tt, bt)))

    # ---- dual view: max context at a fixed serving batch (the other compelling capacity figure) ----
    print(f"\n## Max context length at fixed batch (how long a context fits in HBM)")
    print(f"{'format':>18} | " + " ".join(f"B={B}" for B in (8, 32, 128)))
    for nm, wb, kvb, note in FORMATS:
        W, kvt, _ = rows[nm]
        cells = []
        for B in (8, 32, 128):
            room = avail - W
            Lmax = int(max(0, room) // (B * kvt)) if kvt > 0 and room > 0 else 0
            cells.append(f"{Lmax//1024:>5}k" if Lmax >= 1024 else f"{Lmax:>5}")
        print(f"{nm:>18} | " + "   ".join(cells))

    print("\n# HOW TO READ (the honest framing):")
    print("# - B_max RATIO is ~constant in L (B_max ∝ 1/(L·kv_bytes) -> L cancels): low-bit KV admits a")
    print("#   ~fixed multiple MORE sequences -> ~fixed throughput multiple at the capacity frontier.")
    print("# - But capacity only BINDS at LONG ctx / LARGE model: short ctx -> B_max is huge (1000s) so")
    print("#   compute/scheduler caps you first (capacity moot); long ctx -> B_max is tiny (single digits)")
    print("#   and IS the bottleneck -> the bit-ratio win becomes the real, decisive RPS win.")
    print("# - So the paper plot to show: B_max (and throughput) vs context length, with the low-bit curve")
    print("#   staying servable (B_max>=1) far past where MXINT8 OOMs. This is what fixed-batch latency hides.")
    print("# - Pair with the accuracy-vs-bits Pareto (these bits MUST be iso-accuracy) to rebut 'just use INT4'.")


if __name__ == "__main__":
    main()
