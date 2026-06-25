# Per-scope E2E latency v2 (S1-S6, incl. AA) — NVIDIA GeForce RTX 3090

Llama-3.1-8B 32L, L_in=1024, L_out=128, per-scope robust (u=4 where robust).
Current kernels: prefill = dequant-weight + cuBLAS (ties bf16); batched decode B in 2..15 = shared-activation wide GEMV (40us@M8, beats bf16 46us); B>=16 = dequant+cuBLAS; KV decode = (u,gs)-specialized wide-load. ms in µs; ratio <1 = MSAQ faster.

**Decode kernel fixes (this revision), B=8 weight scopes flipped to winning.** Two memory-pattern
fixes to the batched-decode GEMV (B in 2..15), both found via ncu (L1 pegged ~87-89%, DRAM <20%):
(1) **shared-activation** — each output-column thread was reloading `x[m,kk]` from global; staging the
`[MR][BLOCK]` activation tile in shared once per K-block (broadcast to all 128 threads) collapsed the L1
bound: W-only GEMV M=8 **81→40µs** (beats bf16 46). (2) **W+A dequant-in-staging** — at decode the int8
dot has no advantage (`(qa·qw)·sa·sw == (qa·sa)·(qw·sw)`) and the int path cost 70µs (IMAD<FFMA, int8
shared reads, `idot[MR]`+`acc[MR]` = 2× accumulators); folding `sa` into the staged activation
(`As=qx·sa`, float) and running the W-only float MAC took W+A GEMV **70→31µs**. Both applied symmetrically
to the MXINT8 baseline. Net B=8 total mq/bf: **S1 1.12→0.87→0.88, S2 W+A 1.17→1.12→0.89, S4 0.94→0.67,
S5/S6 0.98→0.93→0.68** — all six scopes now beat bf16 at B=8. (Crossover ~M=10 vs bf16; B>=16 still uses
dequant+cuBLAS, where bf16's tensor-core GEMM leads.)

**S6 W+A+KV+AA** = full quant incl. attention activation×activation (Q,K,P,V). The decode attention kernel reads Q in bf16, so **AA decode latency == KV-decode (== S5)**; AA adds accuracy cost (~+0.9–1.0pp PPL, `precision/aa_attn_results.md`), not decode latency. Prefill attention is bf16 SDPA (AA-prefill loses to it). So S6 ≈ S5 in latency — that equality is the finding.

## S1 W-only  (MSAQ u3/gs16)
| B | prefill bf/mx/mq | decode bf/mx/mq | total bf/mx/mq | mq/bf | mq/mx | mx/bf |
|--:|--|--|--|--:|--:|--:|
| 1 | 257/281/278 | 2880/1874/1564 | 3137/2155/1842 | 0.59 | 0.85 | 0.69 |
| 8 | 2018/2034/2026 | 5730/4776/4753 | 7747/6810/6779 | 0.88 | 1.00 | 0.88 |
| 32 | 7793/7796/7769 | 12515/15638/15445 | 20308/23434/23214 | 1.14 | 0.99 | 1.15 |

## S2 W+A  (MSAQ u2/gs8)
| B | prefill bf/mx/mq | decode bf/mx/mq | total bf/mx/mq | mq/bf | mq/mx | mx/bf |
|--:|--|--|--|--:|--:|--:|
| 1 | 257/281/279 | 2895/1904/1753 | 3153/2185/2032 | 0.64 | 0.93 | 0.69 |
| 8 | 2019/2034/2027 | 5716/4794/4828 | 7735/6827/6856 | 0.89 | 1.00 | 0.88 |
| 32 | 7800/7804/7762 | 12509/15661/15575 | 20309/23465/23337 | 1.15 | 0.99 | 1.16 |

## S3 KV-only  (MSAQ u4/gs2)
| B | prefill bf/mx/mq | decode bf/mx/mq | total bf/mx/mq | mq/bf | mq/mx | mx/bf |
|--:|--|--|--|--:|--:|--:|
| 1 | 258/259/259 | 2890/2720/2743 | 3148/2979/3002 | 0.95 | 1.01 | 0.95 |
| 8 | 2022/2035/2043 | 5723/3925/3861 | 7746/5960/5903 | 0.76 | 0.99 | 0.77 |
| 32 | 7815/7893/7920 | 12524/5187/5232 | 20339/13080/13152 | 0.65 | 1.01 | 0.64 |

## S4 W-only+KV  (MSAQ u2/gs8)
| B | prefill bf/mx/mq | decode bf/mx/mq | total bf/mx/mq | mq/bf | mq/mx | mx/bf |
|--:|--|--|--|--:|--:|--:|
| 1 | 257/282/281 | 2881/1707/1576 | 3138/1989/1857 | 0.59 | 0.93 | 0.63 |
| 8 | 2020/2047/2041 | 5720/2867/3166 | 7740/4914/5207 | 0.67 | 1.06 | 0.63 |
| 32 | 7800/7842/7815 | 12502/8346/9163 | 20302/16188/16978 | 0.84 | 1.05 | 0.80 |

## S5 W+A+KV  (MSAQ u2/gs8)
| B | prefill bf/mx/mq | decode bf/mx/mq | total bf/mx/mq | mq/bf | mq/mx | mx/bf |
|--:|--|--|--|--:|--:|--:|
| 1 | 257/283/281 | 2881/1742/1640 | 3138/2024/1921 | 0.61 | 0.95 | 0.65 |
| 8 | 2017/2045/2041 | 5723/2917/3234 | 7740/4962/5276 | 0.68 | 1.06 | 0.64 |
| 32 | 7791/7834/7806 | 12485/8363/9165 | 20276/16197/16971 | 0.84 | 1.05 | 0.80 |

## S6 W+A+KV+AA  (MSAQ u2/gs8)
| B | prefill bf/mx/mq | decode bf/mx/mq | total bf/mx/mq | mq/bf | mq/mx | mx/bf |
|--:|--|--|--|--:|--:|--:|
| 1 | 257/282/282 | 2892/1745/1641 | 3149/2027/1923 | 0.61 | 0.95 | 0.64 |
| 8 | 2018/2046/2044 | 5718/2911/3235 | 7736/4957/5279 | 0.68 | 1.06 | 0.64 |
| 32 | 7791/7831/7809 | 12482/8350/9151 | 20273/16181/16960 | 0.84 | 1.05 | 0.80 |

(mq=MSAQ mx=MXINT8 bf=bf16. prefill=TTFT(1024 tok), decode=integrated over 128 steps, total=sum.)
