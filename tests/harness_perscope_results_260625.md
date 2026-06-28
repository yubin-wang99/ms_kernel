# Per-scope E2E latency v2 (S1-S6, incl. AA) — NVIDIA RTX PRO 4000 Blackwell

Llama-3.1-8B 32L, L_in=1024, L_out=128, per-scope robust (u=4 where robust).
Current kernels: prefill = dequant-weight + cuBLAS (ties bf16); batched decode B in 2..15 = shared-activation wide GEMV (W-only 40us@M8; W+A dequant-in-staging float MAC 44us@M8 incl quant_act; both beat bf16 46us); B>=16 = dequant+cuBLAS; KV decode = (u,gs)-specialized wide-load. ms in µs; ratio <1 = MSAQ faster.

**S6 W+A+KV+AA** = full quant incl. attention activation×activation (Q,K,P,V). The decode attention kernel reads Q in bf16, so **AA decode latency == KV-decode (== S5)**; AA adds accuracy cost (~+0.9–1.0pp PPL, `precision/aa_attn_results.md`), not decode latency. Prefill attention is bf16 SDPA (AA-prefill loses to it). So S6 ≈ S5 in latency — that equality is the finding.

## S1 W-only  (MSAQ u3/gs16)
| B | prefill bf/mx/mq | decode bf/mx/mq | total bf/mx/mq | mq/bf | mq/mx | mx/bf |
|--:|--|--|--|--:|--:|--:|
| 1 | 258/291/287 | 3613/2249/1845 | 3871/2540/2132 | 0.55 | 0.84 | 0.66 |
| 8 | 2227/2255/2242 | 5979/4976/4979 | 8205/7231/7221 | 0.88 | 1.00 | 0.88 |
| 16 | 4428/4457/4432 | 8772/8916/7937 | 13200/13373/12370 | 0.94 | 0.92 | 1.01 |
| 32 | 7628/7631/7586 | 14373/14481/13470 | 22001/22113/21056 | 0.96 | 0.95 | 1.01 |

## S2 W+A  (MSAQ u2/gs8)
| B | prefill bf/mx/mq | decode bf/mx/mq | total bf/mx/mq | mq/bf | mq/mx | mx/bf |
|--:|--|--|--|--:|--:|--:|
| 1 | 257/293/292 | 3625/2287/2007 | 3882/2580/2299 | 0.59 | 0.89 | 0.66 |
| 8 | 2233/2254/2253 | 6002/4957/4992 | 8235/7211/7245 | 0.88 | 1.00 | 0.88 |
| 16 | 4427/4459/4436 | 8789/8928/8031 | 13217/13386/12467 | 0.94 | 0.93 | 1.01 |
| 32 | 7627/7630/7566 | 14384/14497/13554 | 22012/22127/21120 | 0.96 | 0.95 | 1.01 |

## S3 KV-only  (MSAQ u4/gs16)
| B | prefill bf/mx/mq | decode bf/mx/mq | total bf/mx/mq | mq/bf | mq/mx | mx/bf |
|--:|--|--|--|--:|--:|--:|
| 1 | 257/258/262 | 3624/3571/3452 | 3881/3829/3714 | 0.96 | 0.97 | 0.99 |
| 8 | 2215/2218/2213 | 5940/4947/3889 | 8156/7165/6103 | 0.75 | 0.85 | 0.88 |
| 16 | 4434/4465/4453 | 8804/5210/4371 | 13238/9675/8824 | 0.67 | 0.91 | 0.73 |
| 32 | 7639/7709/7687 | 14402/7044/5334 | 22042/14753/13021 | 0.59 | 0.88 | 0.67 |

## S4 W-only+KV  (MSAQ u2/gs8)
| B | prefill bf/mx/mq | decode bf/mx/mq | total bf/mx/mq | mq/bf | mq/mx | mx/bf |
|--:|--|--|--|--:|--:|--:|
| 1 | 257/296/294 | 3628/2197/1873 | 3885/2493/2167 | 0.56 | 0.87 | 0.64 |
| 8 | 2235/2265/2260 | 5977/3832/3299 | 8212/6097/5559 | 0.68 | 0.91 | 0.74 |
| 16 | 4427/4485/4457 | 8782/5273/4619 | 13209/9757/9077 | 0.69 | 0.93 | 0.74 |
| 32 | 7614/7686/7599 | 14354/7058/6049 | 21968/14744/13648 | 0.62 | 0.93 | 0.67 |

## S5 W+A+KV  (MSAQ u2/gs8)
| B | prefill bf/mx/mq | decode bf/mx/mq | total bf/mx/mq | mq/bf | mq/mx | mx/bf |
|--:|--|--|--|--:|--:|--:|
| 1 | 257/295/294 | 3620/2209/1875 | 3877/2504/2168 | 0.56 | 0.87 | 0.65 |
| 8 | 2233/2264/2260 | 5931/3837/3349 | 8164/6101/5608 | 0.69 | 0.92 | 0.75 |
| 16 | 4427/4483/4455 | 8778/5271/4615 | 13205/9754/9070 | 0.69 | 0.93 | 0.74 |
| 32 | 7618/7670/7605 | 14372/7046/6055 | 21990/14716/13660 | 0.62 | 0.93 | 0.67 |

## S6 W+A+KV+AA  (MSAQ u2/gs8)
| B | prefill bf/mx/mq | decode bf/mx/mq | total bf/mx/mq | mq/bf | mq/mx | mx/bf |
|--:|--|--|--|--:|--:|--:|
| 1 | 257/295/294 | 3622/2209/1876 | 3879/2504/2170 | 0.56 | 0.87 | 0.65 |
| 8 | 2233/2258/2260 | 5934/3836/3351 | 8167/6094/5610 | 0.69 | 0.92 | 0.75 |
| 16 | 4424/4484/4452 | 8783/5275/4615 | 13207/9759/9067 | 0.69 | 0.93 | 0.74 |
| 32 | 7611/7680/7608 | 14353/7057/6052 | 21964/14737/13660 | 0.62 | 0.93 | 0.67 |

(mq=MSAQ mx=MXINT8 bf=bf16. prefill=TTFT(1024 tok), decode=integrated over 128 steps, total=sum.)
