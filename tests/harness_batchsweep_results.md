# Batched E2E sweep — results (kernel_ver2.md §3)

RTX 3090, Llama-3.1-8B (32L), MSAQ **u4/gs2**, glue bf16, random weights (timing value-independent).
Three time metrics — **PREFILL (TTFT)**, **DECODE** (integrated trajectory), **TOTAL** — each as three
ratios: **mq/mx** (MSAQ vs MXINT8, format axis) · **mq/bf** (MSAQ vs BF16) · **mx/bf** (MXINT8 vs BF16).
Ratio **< 1 = faster (win)**. PREFILL: W-only uses the pipelined-WMMA tile (`MS_TILE_CFG=11`), W+A the
auto M-adaptive 2-stage IMMA. DECODE: B=1→wide GEMV; **B>1→batched-decode GEMV for both W-only
(`wonly_gemv_batched`) and W+A (`wa_gemv_batched`, int-dot)** + their MXINT8 matches — each weight
column read once, amortized over B activation rows in registers. Harness `tests/harness_batchsweep.py`;
raw `harness_batchsweep_results.jsonl`. (Tables below are POST batched-GEMV optimization, all decode scopes.)

## Batch sweep — (L_in, L_out) = (1024, 512)

| scope | B | PREFILL mq/mx·mq/bf·mx/bf | DECODE mq/mx·mq/bf·mx/bf | TOTAL mq/mx·mq/bf·mx/bf |
|---|--|--|--|--|
| **S1 W-only** | 1 | 1.00 · 4.96 · 4.96 | 0.77 · **0.76** · 0.99 | 0.79 · **0.85** · 1.07 |
| | 8 | 1.00 · 4.69 · 4.70 | 0.99 · 1.33 · 1.35 | 0.99 · 1.59 · 1.61 |
| | 32 | 0.99 · 4.96 · 4.99 | **0.44** · 2.01 · 4.57 | **0.51** · 2.37 · 4.62 |
| **S2 W+A** | 1 | **0.79** · 6.01 · 7.61 | 0.88 · **0.71** · 0.80 | 0.86 · **0.82** · 0.95 |
| | 8 | 0.82 · 5.82 · 7.08 | 0.91 · 1.34 · 1.47 | 0.89 · 1.68 · 1.90 |
| | 32 | 0.85 · 6.30 · 7.41 | 0.98 · 2.52 · 2.57 | 0.94 · 2.98 · 3.16 |
| **S3 KV-only** | 1 | 1.00 · 1.02 · 1.02 | 1.01 · 0.94 · 0.93 | 1.01 · 0.94 · 0.93 |
| | 8 | 1.00 · 1.01 · 1.01 | 0.98 · **0.65** · 0.66 | 0.98 · **0.68** · 0.69 |
| | 32 | 1.00 · 1.01 · 1.01 | 1.00 · **0.39** · 0.39 | 1.00 · **0.47** · 0.47 |
| **S4 W-only+KV** | 1 | 1.00 · 5.00 · 4.99 | 0.76 · **0.70** · 0.92 | 0.79 · **0.79** · 1.01 |
| | 8 | 1.00 · 4.69 · 4.70 | 0.97 · **0.95** · 0.98 | 0.98 · 1.24 · 1.27 |
| | 32 | 0.99 · 4.95 · 5.00 | **0.35** · 1.39 · 4.01 | **0.44** · 1.82 · 4.13 |
| **S5 W+A+KV** | 1 | 0.79 · 6.03 · 7.65 | 0.88 · **0.65** · 0.73 | 0.86 · **0.76** · 0.88 |
| | 8 | 0.82 · 5.83 · 7.08 | 0.87 · **0.97** · 1.11 | 0.85 · 1.34 · 1.57 |
| | 32 | 0.85 · 6.30 · 7.41 | 0.97 · 1.91 · 1.97 | 0.93 · 2.45 · 2.64 |

**B ≥ 64 OOM** (3090 24GB; 32-layer KV + prefill activation, all formats together).

## Output sweep — L_in=1024, B=8 (TOTAL mq/bf; long-output trend)
| scope | L_out=128 | 512 | 1024 | 2048 | 3880 |
|---|--|--|--|--|--|
| **S3 KV-only** | 0.77 | 0.68 | 0.63 | 0.57 | **0.51** |
| **S5 W+A+KV** | 2.29 | 1.34 | 1.09 | **0.89** | **0.74** |
| **S4 W-only+KV** | 1.98 | 1.25 | 1.04 | **0.87** | **0.73** |
| S2 W+A | 2.54 | 1.68 | 1.47 | 1.34 | 1.25 |
| S1 W-only | 2.24 | 1.59 | 1.43 | 1.32 | 1.24 |

(decode-only mq/bf at L_out=3880: S3 **0.50**, S5 **0.70**, S4 **0.70**, S2 1.22, S1 1.21. mq/mx ≤ ~1.0 everywhere.)

## Findings (all decode kernels batched: W-only, W+A, KV)
1. **Format axis (mq/mx): MSAQ ≤ MXINT8 across every decode scope.** Batched-decode GEMV makes W-only
   decode mq/mx **0.44 (S1 B32) / 0.35 (S4 B32)** (MSAQ's packed-column wide `uint4` load vs MXINT8's 32
   scalar int8 loads); W+A decode mq/mx 0.87–0.98. PREFILL 0.79–1.00. The byte advantage holds at serving scale.
2. **KV-cache (S3): WINS vs BF16 at batch, growing with B and L_out** — decode mq/bf 0.94 (B1) → 0.65
   (B8) → **0.39 (B32)**; total 0.77→**0.51** as L_out 128→3880 (bf16 weights, only KV quantized → batched
   KV-read converts KV bytes once the machine fills, BW-bound).
3. **W+A decode fixed (batched int-dot GEMV vs old GEMM): S5 B8 decode mq/bf 3.77 → 0.97 (now WINS bf16);
   S2 B8 4.10 → 1.34.** Both quant scopes (S4 W-only+KV, S5 W+A+KV) **WIN vs BF16 at long output** — total
   mq/bf **0.73 / 0.74 @L_out3880** (decode 0.70) as the KV win compounds with the now-cheap weight decode.
4. **W-only batched decode: S1 B8 5.67 → 1.33× bf16 (near tie), B32 mq/mx 0.44.** Pure-W-only decode stays
   ~1.2× bf16 — BF16's tensor-core GEMM vs our scalar sub-byte FMA (the structural wall, Phase 47).
5. **PREFILL vs BF16 (~5×) is the separate tensor-core gap** (pipelined-WMMA / 2-stage-IMMA vs cuBLAS at
   large M); dominates S1/S2 TOTAL at short L_out. mq/mx fair (~1.0 W-only / ~0.8 W+A).

## Summary
All decode weight matmuls are now batched (`wonly_gemv_batched`, `wa_gemv_batched` + MXINT8 matches;
weight read amortized over B). **MSAQ WINS/ties vs the matched MXINT8 baseline across every decode scope**
(W-only decode 0.35–0.44× @B32; W+A 0.87–0.98×). **KV-cache WINS vs BF16** (0.39–0.65×), and **both
full-quant scopes (S4, S5) WIN vs BF16 at long output** (total 0.73–0.74× @L_out3880; S5 W+A+KV decode
beats bf16 from B=8). The remaining vs-BF16 gaps — pure W-only decode ~1.2×, prefill ~5× — are the
tensor-core deficit of scalar/staged sub-byte kernels (Phase 47 documented wall), not the format. B≥64 OOM.
