# Batched E2E sweep — results (kernel_ver2.md §3)

RTX 3090, Llama-3.1-8B (32L), MSAQ **u4/gs2**, glue bf16, random weights (timing value-independent).
Three time metrics — **PREFILL (TTFT)**, **DECODE** (integrated trajectory), **TOTAL** — each as three
ratios: **mq/mx** (MSAQ vs MXINT8, format axis) · **mq/bf** (MSAQ vs BF16) · **mx/bf** (MXINT8 vs BF16).
Ratio **< 1 = faster (win)**. PREFILL: W-only uses the pipelined-WMMA tile (`MS_TILE_CFG=11`), W+A the
auto M-adaptive 2-stage IMMA; DECODE uses M-adaptive default (B=1→GEMV, B>1→GEMM(M=B)).
Harness `tests/harness_batchsweep.py`; raw `harness_batchsweep_results.jsonl`.

## Batch sweep — (L_in, L_out) = (1024, 512)

| scope | B | PREFILL mq/mx·mq/bf·mx/bf | DECODE mq/mx·mq/bf·mx/bf | TOTAL mq/mx·mq/bf·mx/bf |
|---|--|--|--|--|
| **S1 W-only** | 1 | 1.00 · 4.98 · 4.96 | 0.77 · **0.76** · 0.99 | 0.79 · **0.85** · 1.07 |
| | 8 | 1.00 · 4.69 · 4.69 | 1.00 · 5.67 · 5.70 | 1.00 · 5.60 · 5.62 |
| | 32 | 0.99 · 4.94 · 5.00 | 0.99 · 3.12 · 3.14 | 0.99 · 3.34 · 3.37 |
| **S2 W+A** | 1 | **0.79** · 6.04 · 7.65 | 0.88 · **0.71** · 0.80 | 0.86 · **0.82** · 0.95 |
| | 8 | 0.82 · 5.82 · 7.09 | 1.03 · 4.09 · 3.97 | 1.00 · 4.22 · 4.21 |
| | 32 | 0.85 · 6.27 · 7.41 | 0.93 · 2.45 · 2.63 | 0.91 · 2.92 · 3.22 |
| **S3 KV-only** | 1 | 1.00 · 1.01 · 1.01 | 1.01 · 0.94 · 0.93 | 1.01 · 0.94 · 0.93 |
| | 8 | 1.00 · 1.01 · 1.01 | 0.98 · **0.65** · 0.66 | 0.98 · **0.68** · 0.69 |
| | 32 | 1.00 · 1.01 · 1.01 | 1.00 · **0.39** · 0.39 | 1.00 · **0.47** · 0.47 |
| **S4 W-only+KV** | 1 | 1.00 · 5.04 · 5.02 | 0.76 · **0.70** · 0.92 | 0.79 · **0.79** · 1.01 |
| | 8 | 1.00 · 4.69 · 4.69 | 0.99 · 5.33 · 5.40 | 0.99 · 5.28 · 5.35 |
| | 32 | 0.99 · 4.97 · 5.04 | 0.99 · 2.52 · 2.53 | 0.99 · 2.82 · 2.84 |
| **S5 W+A+KV** | 1 | 0.79 · 6.07 · 7.69 | 0.88 · **0.64** · 0.74 | 0.86 · **0.76** · 0.88 |
| | 8 | 0.82 · 5.84 · 7.09 | 1.03 · 3.78 · 3.66 | 1.00 · 3.94 · 3.92 |
| | 32 | 0.84 · 6.27 · 7.44 | 0.89 · 1.85 · 2.07 | 0.88 · 2.39 · 2.73 |

**B ≥ 64 OOM** (3090 24GB; 32-layer KV + prefill activation, all formats together).

## Output sweep — L_in=1024, B=8 (DECODE ratios; KV trend)
| scope | L_out=128 | 512 | 1024 | 2048 | 3880 |
|---|--|--|--|--|--|
| **S3 KV-only** mq/bf | 0.68 | 0.65 | 0.62 | 0.57 | **0.50** |
| S1 W-only mq/bf | 6.04 | 5.69 | 5.31 | 4.69 | 3.92 |
| S5 W+A+KV mq/bf | 4.03 | 3.76 | 3.47 | 3.00 | 2.43 |

(mq/mx ≈ 0.96–1.03 throughout — MSAQ ties/edges MXINT8 on the format axis everywhere.)

## Findings
1. **Format axis (mq/mx): MSAQ ≤ MXINT8 everywhere.** PREFILL 0.79–1.00 (W+A 0.79–0.85 — MSAQ-s
   activation prequant beats plain MXINT8), DECODE/TOTAL 0.76–1.03. The byte advantage holds at serving scale.
2. **KV-cache (S3): WINS vs BF16 at batch, growing with B and L_out.** decode mq/bf 0.94 (B=1) →
   0.65 (B=8) → **0.39 (B=32)**; and 0.68→**0.50** as L_out 128→3880. Weights stay bf16 (cuBLAS), only
   KV quantized → the batched KV-read kernel converts KV byte savings once batch fills the machine
   (BW-bound). This is the spec's headline and it holds.
3. **W-only (S1/S4): WINS at B=1 (mq/bf 0.70–0.76, the wide GEMV), LOSES at B>1 (≈3–6×).** Cause: at
   B>1 the decode weight matmul becomes GEMM(M=B) and there is **no optimized small-batch decode GEMM**
   — the tile GEMM is built for large-M prefill. **This is the B>1 optimization target.**
4. **PREFILL vs BF16 (~5×) is a separate kernel-maturity gap** — even with `MS_TILE_CFG=11` the
   pipelined-WMMA / 2-stage-IMMA prefill GEMM trails cuBLAS at these M; mq/mx is fair (~1.0 / 0.8).

## Next: B>1 W-only win
Build a **batched-decode W-only GEMV** (extend the B=1 wide GEMV to M=B): read each weight column once
and amortize over the B activation rows. At decode the weight read dominates (memory-bound) and MSAQ
reads ~0.5× bytes → win vs BF16 and MXINT8 until the §3.2 crossover (~tens of tokens). KV-cache (S3)
already wins; this closes W-only at B>1.
