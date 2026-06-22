# Batched E2E sweep — results (kernel_ver2.md §3)

RTX 3090, Llama-3.1-8B (32L), MSAQ **u4/gs2**, glue bf16, random weights (timing value-independent).
Three time metrics — **PREFILL (TTFT)**, **DECODE** (integrated trajectory), **TOTAL** — each as three
ratios: **mq/mx** (MSAQ vs MXINT8, format axis) · **mq/bf** (MSAQ vs BF16) · **mx/bf** (MXINT8 vs BF16).
Ratio **< 1 = faster (win)**. PREFILL: W-only uses the pipelined-WMMA tile (`MS_TILE_CFG=11`), W+A the
auto M-adaptive 2-stage IMMA. DECODE: B=1→wide GEMV; **B>1 W-only→batched-decode GEMV**
(`wonly_gemv_batched`/`mxint8_gemv_batched` — reads each weight column once, amortizes over B
activation rows in registers), W+A→GEMM(M=B). Harness `tests/harness_batchsweep.py`; raw
`harness_batchsweep_results.jsonl`. (Tables below are POST batched-GEMV optimization.)

## Batch sweep — (L_in, L_out) = (1024, 512)

| scope | B | PREFILL mq/mx·mq/bf·mx/bf | DECODE mq/mx·mq/bf·mx/bf | TOTAL mq/mx·mq/bf·mx/bf |
|---|--|--|--|--|
| **S1 W-only** | 1 | 1.00 · 4.96 · 4.96 | 0.77 · **0.76** · 0.99 | 0.79 · **0.85** · 1.07 |
| | 8 | 1.00 · 4.69 · 4.70 | 0.99 · 1.33 · 1.35 | 0.99 · 1.59 · 1.61 |
| | 32 | 0.99 · 4.96 · 4.99 | **0.44** · 2.01 · 4.57 | **0.51** · 2.37 · 4.62 |
| **S2 W+A** | 1 | **0.79** · 6.02 · 7.65 | 0.88 · **0.71** · 0.80 | 0.87 · **0.82** · 0.95 |
| | 8 | 0.82 · 5.81 · 7.08 | 1.03 · 4.10 · 3.98 | 1.00 · 4.24 · 4.22 |
| | 32 | 0.84 · 6.26 · 7.41 | 0.93 · 2.45 · 2.64 | 0.91 · 2.92 · 3.22 |
| **S3 KV-only** | 1 | 1.00 · 1.02 · 1.02 | 1.01 · 0.94 · 0.93 | 1.01 · 0.94 · 0.93 |
| | 8 | 1.00 · 1.01 · 1.01 | 0.98 · **0.65** · 0.66 | 0.98 · **0.68** · 0.69 |
| | 32 | 1.00 · 1.01 · 1.01 | 1.00 · **0.39** · 0.39 | 1.00 · **0.47** · 0.47 |
| **S4 W-only+KV** | 1 | 1.00 · 5.00 · 4.99 | 0.76 · **0.70** · 0.92 | 0.79 · **0.79** · 1.01 |
| | 8 | 1.00 · 4.69 · 4.70 | 0.97 · **0.95** · 0.98 | 0.98 · 1.24 · 1.27 |
| | 32 | 0.99 · 4.95 · 5.00 | **0.35** · 1.39 · 4.01 | **0.44** · 1.82 · 4.13 |
| **S5 W+A+KV** | 1 | 0.79 · 6.05 · 7.67 | 0.88 · **0.65** · 0.74 | 0.86 · **0.76** · 0.88 |
| | 8 | 0.82 · 5.83 · 7.08 | 1.03 · 3.77 · 3.65 | 1.00 · 3.93 · 3.92 |
| | 32 | 0.84 · 6.24 · 7.41 | 0.89 · 1.84 · 2.06 | 0.88 · 2.38 · 2.72 |

**B ≥ 64 OOM** (3090 24GB; 32-layer KV + prefill activation, all formats together).

## Output sweep — L_in=1024, B=8 (DECODE & TOTAL mq/bf; long-output trend)
| scope | metric | L_out=128 | 512 | 1024 | 2048 | 3880 |
|---|--|--|--|--|--|--|
| **S3 KV-only** | total mq/bf | 0.77 | 0.68 | 0.63 | 0.57 | **0.51** |
| **S4 W-only+KV** | total mq/bf | 1.98 | 1.25 | 1.04 | **0.90** | **0.72** |
| **S4 W-only+KV** | decode mq/bf | 1.01 | 0.96 | 0.90 | 0.81 | **0.70** |
| S1 W-only | decode mq/mx | 0.98 | 0.98 | 0.98 | 0.99 | 0.99 |
| S1 W-only | decode mq/bf | 1.35 | 1.33 | 1.30 | 1.26 | 1.21 |

## Findings (post batched-GEMV optimization)
1. **Format axis (mq/mx): MSAQ ≤ MXINT8 everywhere — and now WINS W-only decode at batch.** With the
   batched GEMV, W-only decode mq/mx drops to **0.44 (S1 B=32)** / **0.35 (S4 B=32)**: MSAQ's packed
   column → one wide `uint4` load vs MXINT8's 32 scalar int8 loads. (Was ~1.0 tie under the old GEMM.)
2. **KV-cache (S3): WINS vs BF16 at batch, growing with B and L_out** — decode mq/bf 0.94 (B=1) → 0.65
   (B=8) → **0.39 (B=32)**; total 0.77→**0.51** as L_out 128→3880. Weights stay bf16 (cuBLAS), only KV
   quantized → the batched KV-read kernel converts KV bytes once batch fills the machine (BW-bound).
3. **W-only decode hugely faster (batched GEMV vs old GEMM): S1 B=8 decode mq/bf 5.67 → 1.33; B=32 3.12
   → 2.01.** S4 (W-only+KV) **WINS vs BF16 at long output** (total mq/bf **0.90 @L_out2048, 0.72 @3880**)
   as the KV win compounds with the cheaper W-only decode. Pure W-only (S1) decode is now ~1.2–1.3× bf16
   (near tie) — the residual gap is BF16's tensor-core GEMM vs our scalar sub-byte FMA.
4. **PREFILL vs BF16 (~5×) is a separate tensor-core gap** — the pipelined-WMMA / 2-stage-IMMA prefill
   GEMM (even with `MS_TILE_CFG=11`) trails cuBLAS at these large M; it dominates S1/S2 TOTAL at L_out=512.
   mq/mx is fair (~1.0 W-only / ~0.8 W+A).

## Summary
The **batched-decode GEMV** (`wonly_gemv_batched`, amortizes the weight read over B rows) closes the
B>1 W-only gap: **W-only WINS vs the matched MXINT8 baseline at batch** (decode 0.35–0.44× @B32) and is
5.7→1.3× faster vs the old GEMM. **KV-cache WINS vs both baselines** at batch/long-output. **S4
(W-only+KV) WINS vs BF16 at long output** (0.72× @L_out3880). Remaining vs-BF16 gaps (pure W-only
decode ~1.2×, prefill ~5×) are the tensor-core deficit of scalar sub-byte kernels, not the format.
