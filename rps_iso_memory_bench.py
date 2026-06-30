#!/usr/bin/env python3
"""Iso-memory B_max -> RPS decomposition bench  (new methodology, 2026-06-30).

Replaces the fixed-batch sweep {1,8,16,32} with the RPS DECOMPOSITION:

        RPS  =  B_max / (O * t_step)

measured at *each format's own* B_max under a SHARED memory budget M_avail
(iso-memory operating point, not iso-batch). The two RPS channels are reported
SEPARATELY so the reader can see where a gain comes from:

  * B_max channel  -- capacity headroom. MSAQ's byte saving converts here
    UNCONDITIONALLY (fewer bytes/seq -> more seqs fit; pure arithmetic, no
    bandwidth/latency paradox).
  * t_step channel -- decode step latency. Latency/ridge-bound at this anchor;
    the KV-decode plateau lives here (the "KV decode paradox").

3 cases x {BF16, MXINT8, mantissa-shared}, each measured at its B_max:
  Case 1  Weight     : W_total drops  -> frees mem for KV -> B_max up (k = BF16-KV).
  Case 2  KV         : k (per-seq KV) drops -> B_max up   (W_total = BF16 16GB).
  Case 3  Weight+KV  : both drop      -> B_max max.

B_max:  analytical closed form (final_methodology.md S2) + real-GPU-alloc verify.
t_step: real kernels. One decode layer = linear GEMMs (M=B) + attention (KV read),
        timed once and x n_layer (only one layer's footprint is resident, so the
        full B_max batch fits for timing). Linears use bf16 cuBLAS for ALL formats
        (the deployed dequant+cuBLAS path is ~format-equal at B>=16; this is mildly
        CONSERVATIVE for the weight-quant fused path which actually wins at B>=16).
        Weight quantization therefore affects t_step ONLY through B_max (capacity);
        the format-specific step cost is the attention (KV) kernel.

Run:  CUDA_VISIBLE_DEVICES=0 python rps_iso_memory_bench.py [--lout 128] [--lin 1024]
"""
import argparse, json, numpy as np, torch, torch.nn.functional as F
from ms_lib.pack import pack_kv, pack_kv_mxint8
from ms_lib import ops; assert ops.available()
OPS = torch.ops.msaq

# ---- Llama-3.1-8B ----------------------------------------------------------
LAYERS, Hq, Hkv, hd, hidden, inter = 32, 32, 8, 128, 4096, 14336
P_W   = 8.03e9                                   # weight params
KV_ELEM_PER_TOK = 2 * LAYERS * Hkv * hd          # K and V, all layers = 65,536
LIN = [("qkv", hidden, hidden + 2 * Hkv * hd), ("o", hidden, hidden),
       ("gate_up", hidden, 2 * inter), ("down", inter, hidden)]   # fused, as deployed

# ---- iso-memory budget -----------------------------------------------------
VRAM_GB   = 24.0          # nominal card capacity (GB = 1e9 B), matches final_methodology S2.6
F_USABLE  = 0.86          # gpu_mem_util 0.9 x paged-KV usable ~0.96 -> ~0.86 (M_avail = 20.6 GB)
M_AVAIL   = F_USABLE * VRAM_GB * 1e9

# ---- cases: (name, b_w, b_kv, kv_fmt, u, gs) -------------------------------
#   b_w/b_kv = EFFECTIVE bits/elem (metadata amortized).  kv_fmt picks the attn kernel.
#   Weight quant enters t_step only via B_max (linears = bf16 cuBLAS for all).
CASES = {
    "Weight": [
        ("BF16",   16.0,    16.0,   "bf16",   0,  0),
        ("MXINT8",  8.25,   16.0,   "bf16",   0,  0),   # W8.25, KV stays BF16
        ("MSAQ",    5.4375, 16.0,   "bf16",   0,  0),   # W (3,16)=5.4bpe, KV BF16
    ],
    "KV": [
        ("BF16",   16.0,    16.0,   "bf16",   0,  0),
        ("MXINT8", 16.0,     8.25,  "mxint8", 0,  0),   # W BF16, KV 8.25
        ("MSAQ",   16.0,     5.4375,"msaq",   3, 16),   # W BF16, KV (3,16)=5.4bpe
    ],
    "Weight+KV": [
        ("BF16",   16.0,    16.0,   "bf16",   0,  0),
        ("MXINT8",  8.25,    8.25,  "mxint8", 0,  0),
        ("MSAQ",    6.5,     6.5,   "msaq",   2,  8),   # W+KV (2,8)=6.5bpe
    ],
}

# ---- timing helpers --------------------------------------------------------
def C(a): return torch.from_numpy(a).cuda()
def T(fn, it, w):
    for _ in range(w): fn()
    torch.cuda.synchronize(); s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    s.record()
    for _ in range(it): fn()
    e.record(); torch.cuda.synchronize(); return s.elapsed_time(e) / it   # ms

def lin_ms(B):
    """one layer's linear GEMMs at M=B (bf16 cuBLAS = deployed dequant+cuBLAS path, B>=16)."""
    t = 0.0
    for _, K, N in LIN:
        X = torch.randn(B, K, device="cuda", dtype=torch.bfloat16)
        W = torch.randn(N, K, device="cuda", dtype=torch.bfloat16)
        t += T(lambda: X @ W.t(), 30, 8)
    return t

def attn_ms(B, L, fmt, u, gs):
    """one layer's decode attention (format-specific KV read) at batch B, context L."""
    nb = hd // 32
    rng = np.random.default_rng(0)
    Kf = (rng.standard_normal((B, Hkv, L, hd)) * 0.5).astype(np.float32)
    Vf = (rng.standard_normal((B, Hkv, L, hd)) * 0.5).astype(np.float32)
    q = torch.randn(B, Hq, hd, device="cuda", dtype=torch.bfloat16)
    def sp(packs, keys): return {k: torch.from_numpy(np.stack([p[k] for p in packs])).cuda() for k in keys}
    if fmt == "bf16":
        g = Hq // Hkv
        Kb = C(Kf).to(torch.bfloat16).repeat_interleave(g, 1); Vb = C(Vf).to(torch.bfloat16).repeat_interleave(g, 1)
        return T(lambda: F.scaled_dot_product_attention(q.unsqueeze(2), Kb, Vb), 30, 8)
    if fmt == "mxint8":
        xk = sp([pack_kv_mxint8(Kf[b]) for b in range(B)], ("scale_exp", "qweight"))
        xv = sp([pack_kv_mxint8(Vf[b]) for b in range(B)], ("scale_exp", "qweight"))
        return T(lambda: OPS.mxint8_kv_decode_batched(q, xk["scale_exp"], xk["qweight"], xv["scale_exp"], xv["qweight"],
                                                      B, Hq, Hkv, L, hd, nb, L), 30, 8)
    mk = sp([pack_kv(Kf[b], u, gs) for b in range(B)], ("scale_exp", "upper", "shared"))
    mv = sp([pack_kv(Vf[b], u, gs) for b in range(B)], ("scale_exp", "upper", "shared"))
    return T(lambda: OPS.kv_decode_attention_batched(q, mk["scale_exp"], mk["upper"], mk["shared"],
                     mv["scale_exp"], mv["upper"], mv["shared"], B, Hq, Hkv, L, hd, nb, u, gs, L), 30, 8)

# ---- capacity --------------------------------------------------------------
def w_bytes(b_w):          return P_W * b_w / 8.0
def kv_bytes_req(b_kv, L): return KV_ELEM_PER_TOK * b_kv / 8.0 * L          # bytes / request
def b_max_analytic(b_w, b_kv, L):
    room = M_AVAIL - w_bytes(b_w)
    return int(max(0, room) // kv_bytes_req(b_kv, L))

def step_ms(B, L, kv_fmt, u, gs, reps):
    """median t_step over `reps` (one layer's lin+attn) x n_layer; damps 140W clock drift."""
    vals = []
    for _ in range(reps):
        vals.append(LAYERS * (lin_ms(B) + attn_ms(B, L, kv_fmt, u, gs)))
    return float(np.median(vals))

def verify_alloc(b_w, b_kv, L, B):
    """Confirm weights + B requests' KV actually allocate on the real device (no OOM)."""
    bufs, ok = [], True
    try:
        bufs.append(torch.empty(int(w_bytes(b_w)), dtype=torch.uint8, device="cuda"))
        remain = int(kv_bytes_req(b_kv, L) * B); chunk = 256 * 1024 * 1024
        while remain > 0:
            n = min(chunk, remain); bufs.append(torch.empty(n, dtype=torch.uint8, device="cuda")); remain -= n
        torch.cuda.synchronize()
    except RuntimeError:
        ok = False
    del bufs; torch.cuda.empty_cache()
    return ok

# ---- main ------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lin",  type=int, default=1024)
    ap.add_argument("--lout", type=int, default=128)
    ap.add_argument("--no-verify", action="store_true", help="skip real-alloc B_max verification")
    ap.add_argument("--reps", type=int, default=3, help="median over N t_step measurements (clock-drift control)")
    ap.add_argument("--out", default="rps_iso_memory_results.jsonl")
    a = ap.parse_args()
    L_seq, O = a.lin + a.lout, a.lout
    dev = torch.cuda.get_device_name(0)
    real_total = torch.cuda.get_device_properties(0).total_memory

    print(f"# {dev} | Llama-3.1-8B | L_in={a.lin} L_out={O} L_seq={L_seq} | {LAYERS} layers")
    print(f"# M_avail = {F_USABLE} x {VRAM_GB}GB = {M_AVAIL/1e9:.2f}GB (nominal); device total {real_total/1e9:.2f}GB")
    print(f"# RPS = B_max / (O x t_step). Two channels reported separately.\n")

    # BF16 reference (W16/KV16) is identical across all 3 cases -> measure ONCE and reuse,
    # so the normalization isn't polluted by case-to-case clock drift.
    B_bf16 = b_max_analytic(16.0, 16.0, L_seq)
    t_bf16 = step_ms(B_bf16, L_seq, "bf16", 0, 0, a.reps)
    ref = dict(B=B_bf16, t_step=t_bf16, rps=(B_bf16 / (t_bf16 / 1000.0)) / O)
    bf16_fit = "-" if a.no_verify else ("ok" if verify_alloc(16.0, 16.0, L_seq, B_bf16) else "OOM")
    print(f"# BF16 reference (shared): B_max={B_bf16}  t_step={t_bf16:.2f}ms  RPS={ref['rps']:.2f}\n")

    records = []
    for case, fmts in CASES.items():
        print(f"## Case: {case}")
        print(f"{'format':>8} {'b_w':>6} {'b_kv':>6} {'Wgt(GB)':>8} {'KV/req':>8} {'B_max':>6} {'fit?':>5} "
              f"{'t_step(ms)':>10} {'tok/s':>8} {'RPS':>7} {'RPS/BF16':>9} {'B_max/BF16':>10}")
        for nm, b_w, b_kv, kv_fmt, u, gs in fmts:
            B = b_max_analytic(b_w, b_kv, L_seq)
            if nm == "BF16":                            # reuse shared reference (identical config)
                fit, t_step = bf16_fit, t_bf16
                lm = am = float("nan")
            else:
                fit = "-" if a.no_verify else ("ok" if verify_alloc(b_w, b_kv, L_seq, B) else "OOM")
                lm = lin_ms(B); am = attn_ms(B, L_seq, kv_fmt, u, gs)   # for the JSONL breakdown
                t_step = step_ms(B, L_seq, kv_fmt, u, gs, a.reps)
            tok_s = B / (t_step / 1000.0)
            rps = tok_s / O
            rps_r = rps / ref["rps"]; bmax_r = B / ref["B"]
            print(f"{nm:>8} {b_w:>6.2f} {b_kv:>6.2f} {w_bytes(b_w)/1e9:>8.2f} {kv_bytes_req(b_kv,L_seq)/1e6:>6.0f}MB "
                  f"{B:>6} {fit:>5} {t_step:>10.2f} {tok_s:>8.0f} {rps:>7.2f} {rps_r:>8.2f}x {bmax_r:>9.2f}x")
            records.append(dict(case=case, format=nm, b_w=b_w, b_kv=b_kv, kv_fmt=kv_fmt,
                                W_GB=w_bytes(b_w)/1e9, KV_MB_req=kv_bytes_req(b_kv,L_seq)/1e6,
                                B_max=B, alloc_fit=fit, lin_ms=lm, attn_ms=am, t_step_ms=t_step,
                                tok_s=tok_s, rps=rps, rps_vs_bf16=rps_r, bmax_vs_bf16=bmax_r,
                                L_in=a.lin, L_out=O, L_seq=L_seq))
        # two-channel decomposition vs MXINT8 (the reviewer baseline)
        mx = next(r for r in records if r["case"] == case and r["format"] == "MXINT8")
        ms = next(r for r in records if r["case"] == case and r["format"] == "MSAQ")
        cap_ch  = ms["B_max"] / mx["B_max"]                 # capacity (B_max) channel
        step_ch = mx["t_step_ms"] / ms["t_step_ms"]         # step-latency channel (inverse t_step)
        rps_ch  = ms["rps"] / mx["rps"]
        print(f"   vs MXINT8 -> RPS {rps_ch:.2f}x = B_max ch {cap_ch:.2f}x  x  t_step ch {step_ch:.2f}x"
              f"   (t_step {mx['t_step_ms']:.1f}->{ms['t_step_ms']:.1f} ms)\n")
        records.append(dict(case=case, format="_decomp_vs_mxint8", cap_channel=cap_ch,
                            step_channel=step_ch, rps_gain=rps_ch))

    with open(a.out, "w") as f:
        for r in records: f.write(json.dumps(r) + "\n")
    print(f"# wrote {a.out}")
    print(f"# READ: B_max channel = capacity headroom (MSAQ wins unconditionally, pure arithmetic).")
    print(f"#       t_step channel = step latency (ridge/plateau-bound at L_seq={L_seq}; the KV-decode paradox).")

if __name__ == "__main__":
    main()
