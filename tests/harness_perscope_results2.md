# Per-scope E2E latency v2 (S1-S6, incl. AA) — NVIDIA GeForce RTX 3090

Llama-3.1-8B 32L, L_in=1024, L_out=128, per-scope robust (u=4 where robust).
Current kernels: prefill = dequant-weight + cuBLAS (ties bf16); batched decode B∈2..15 =
SHARED-ACTIVATION wide GEMV (see below); B>=16 = dequant+cuBLAS; KV decode = (u,gs)-specialized
wide-load. ms in µs; ratio <1 = MSAQ faster.

**Decode shared-activation fix (this revision).** The batched decode GEMV (B>1) had each output-column
thread reload the activation `x[m,kk]` from global — ncu showed L1 pegged at 87% while DRAM sat at 17%
(the *activation*, not the quantized weight, bounded it). Staging the `[MR][BLOCK]` activation tile in
shared once per K-block (broadcast to all 128 column threads) collapsed that: the W-only decode GEMV at
M=8 went **81µs → 40µs, now beating bf16 cuBLAS (46µs)**. This flipped the **B=8 W-only scopes from
losing to winning**: S1 1.12→**0.87**, S4 0.94→**0.67**, S5/S6 0.98→**0.93** (decode-only S1 6.7k→4.7k µs).
The fix was applied symmetrically to the MXINT8 baseline (fair). Crossover: the shared-activation GEMV
beats bf16 up to ~M=10 and beats dequant+cuBLAS up to ~M=20; beyond that the cuda-core MAC compute grows
(occupancy-bound at M=32) while bf16 stays tensor-core memory-bound, so B>=16 keeps dequant+cuBLAS.

**S6 W+A+KV+AA** = full quant incl. attention activation×activation (Q,K,P,V). The decode attention kernel reads Q in bf16, so **AA decode latency == KV-decode (== S5)**; AA adds accuracy cost (~+0.9–1.0pp PPL, `precision/aa_attn_results.md`), not decode latency. Prefill attention is bf16 SDPA (AA-prefill loses to it). So S6 ≈ S5 in latency — that equality is the finding.

## S1 W-only  (MSAQ u3/gs16)
| B | prefill bf/mx/mq | decode bf/mx/mq | total bf/mx/mq | mq/bf | mq/mx | mx/bf |
|--:|--|--|--|--:|--:|--:|
| 1 | 256/282/278 | 2881/1875/1562 | 3138/2157/1840 | 0.59 | 0.85 | 0.69 |
| 8 | 2016/2034/2024 | 5737/4764/4723 | 7753/6798/6747 | 0.87 | 0.99 | 0.88 |
| 32 | 7781/7785/7759 | 12497/15621/15447 | 20278/23406/23206 | 1.14 | 0.99 | 1.15 |

## S2 W+A  (MSAQ u2/gs8)
| B | prefill bf/mx/mq | decode bf/mx/mq | total bf/mx/mq | mq/bf | mq/mx | mx/bf |
|--:|--|--|--|--:|--:|--:|
| 1 | 258/281/279 | 2895/1904/1754 | 3153/2185/2033 | 0.64 | 0.93 | 0.69 |
| 8 | 2022/2037/2030 | 5724/6541/6639 | 7746/8577/8669 | 1.12 | 1.01 | 1.11 |
| 32 | 7788/7795/7755 | 12499/15646/15568 | 20287/23441/23323 | 1.15 | 0.99 | 1.16 |

## S3 KV-only  (MSAQ u4/gs2)
| B | prefill bf/mx/mq | decode bf/mx/mq | total bf/mx/mq | mq/bf | mq/mx | mx/bf |
|--:|--|--|--|--:|--:|--:|
| 1 | 259/261/260 | 2892/2721/2744 | 3151/2982/3004 | 0.95 | 1.01 | 0.95 |
| 8 | 2021/2037/2038 | 5729/3931/3867 | 7749/5968/5905 | 0.76 | 0.99 | 0.77 |
| 32 | 7811/7891/7915 | 12523/5190/5233 | 20334/13082/13148 | 0.65 | 1.01 | 0.64 |

## S4 W-only+KV  (MSAQ u2/gs8)
| B | prefill bf/mx/mq | decode bf/mx/mq | total bf/mx/mq | mq/bf | mq/mx | mx/bf |
|--:|--|--|--|--:|--:|--:|
| 1 | 258/282/282 | 2890/1718/1577 | 3147/2000/1858 | 0.59 | 0.93 | 0.64 |
| 8 | 2020/2047/2041 | 5725/2868/3172 | 7745/4915/5213 | 0.67 | 1.06 | 0.63 |
| 32 | 7799/7846/7817 | 12512/8366/9168 | 20310/16212/16985 | 0.84 | 1.05 | 0.80 |

## S5 W+A+KV  (MSAQ u2/gs8)
| B | prefill bf/mx/mq | decode bf/mx/mq | total bf/mx/mq | mq/bf | mq/mx | mx/bf |
|--:|--|--|--|--:|--:|--:|
| 1 | 257/282/282 | 2895/1744/1641 | 3152/2027/1923 | 0.61 | 0.95 | 0.64 |
| 8 | 2019/2048/2042 | 5728/4774/5152 | 7747/6822/7193 | 0.93 | 1.05 | 0.88 |
| 32 | 7795/7839/7814 | 12510/8369/9172 | 20305/16207/16987 | 0.84 | 1.05 | 0.80 |

## S6 W+A+KV+AA  (MSAQ u2/gs8)
| B | prefill bf/mx/mq | decode bf/mx/mq | total bf/mx/mq | mq/bf | mq/mx | mx/bf |
|--:|--|--|--|--:|--:|--:|
| 1 | 257/282/282 | 2882/1741/1641 | 3139/2023/1923 | 0.61 | 0.95 | 0.64 |
| 8 | 2020/2050/2044 | 5731/4778/5165 | 7751/6828/7208 | 0.93 | 1.06 | 0.88 |
| 32 | 7808/7852/7827 | 12525/8368/9186 | 20332/16220/17013 | 0.84 | 1.05 | 0.80 |

(mq=MSAQ mx=MXINT8 bf=bf16. prefill=TTFT(1024 tok), decode=integrated over 128 steps, total=sum.)
