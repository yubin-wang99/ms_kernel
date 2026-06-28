# Per-scope E2E latency v2 (S1-S6, incl. AA) — NVIDIA RTX PRO 4000 Blackwell

Llama-3.1-8B 32L, L_in=1024, L_out=512, per-scope robust (u=4 where robust).
Current kernels: prefill = dequant-weight + cuBLAS (ties bf16); batched decode B in 2..15 = shared-activation wide GEMV (W-only 40us@M8; W+A dequant-in-staging float MAC 44us@M8 incl quant_act; both beat bf16 46us); B>=16 = dequant+cuBLAS; KV decode = (u,gs)-specialized wide-load. ms in µs; ratio <1 = MSAQ faster.

**S6 W+A+KV+AA** = full quant incl. attention activation×activation (Q,K,P,V). The decode attention kernel reads Q in bf16, so **AA decode latency == KV-decode (== S5)**; AA adds accuracy cost (~+0.9–1.0pp PPL, `precision/aa_attn_results.md`), not decode latency. Prefill attention is bf16 SDPA (AA-prefill loses to it). So S6 ≈ S5 in latency — that equality is the finding.

## S1 W-only  (MSAQ u3/gs16)
| B | prefill bf/mx/mq | decode bf/mx/mq | total bf/mx/mq | mq/bf | mq/mx | mx/bf |
|--:|--|--|--|--:|--:|--:|
| 1 | 258/290/288 | 14728/9147/7550 | 14986/9437/7838 | 0.52 | 0.83 | 0.63 |
| 8 | 2227/2256/2243 | 25971/21615/21188 | 28198/23871/23430 | 0.83 | 0.98 | 0.85 |
| 16 | 4384/4407/4373 | 38854/39405/35478 | 43238/43812/39851 | 0.92 | 0.91 | 1.01 |
| 32 | 7663/7633/7553 | 64899/65347/61128 | 72561/72980/68682 | 0.95 | 0.94 | 1.01 |

## S2 W+A  (MSAQ u2/gs8)
| B | prefill bf/mx/mq | decode bf/mx/mq | total bf/mx/mq | mq/bf | mq/mx | mx/bf |
|--:|--|--|--|--:|--:|--:|
| 1 | 254/293/289 | 14770/9314/8193 | 15024/9606/8481 | 0.56 | 0.88 | 0.64 |
| 8 | 2220/2221/2222 | 25570/21599/21733 | 27790/23820/23955 | 0.86 | 1.01 | 0.86 |
| 16 | 4388/4411/4372 | 38846/39420/35923 | 43234/43830/40296 | 0.93 | 0.92 | 1.01 |
| 32 | 7656/7606/7556 | 64872/65272/61615 | 72529/72877/69172 | 0.95 | 0.95 | 1.00 |

## S3 KV-only  (MSAQ u4/gs16)
| B | prefill bf/mx/mq | decode bf/mx/mq | total bf/mx/mq | mq/bf | mq/mx | mx/bf |
|--:|--|--|--|--:|--:|--:|
| 1 | 255/256/262 | 14775/14507/13948 | 15029/14763/14210 | 0.95 | 0.96 | 0.98 |
| 8 | 2197/2197/2195 | 25593/20717/15844 | 27789/22914/18040 | 0.65 | 0.79 | 0.82 |
| 16 | 4399/4444/4438 | 38939/22019/18121 | 43338/26463/22558 | 0.52 | 0.85 | 0.61 |
| 32 | 7685/7775/7739 | 64994/30380/22417 | 72679/38155/30156 | 0.41 | 0.79 | 0.52 |

## S4 W-only+KV  (MSAQ u2/gs8)
| B | prefill bf/mx/mq | decode bf/mx/mq | total bf/mx/mq | mq/bf | mq/mx | mx/bf |
|--:|--|--|--|--:|--:|--:|
| 1 | 256/295/292 | 14779/8906/7588 | 15035/9201/7880 | 0.52 | 0.86 | 0.61 |
| 8 | 2238/2221/2260 | 25644/16207/13745 | 27883/18428/16005 | 0.57 | 0.87 | 0.66 |
| 16 | 4387/4439/4408 | 38862/22234/19635 | 43249/26673/24043 | 0.56 | 0.90 | 0.62 |
| 32 | 7639/7685/7575 | 64807/30584/26306 | 72446/38269/33882 | 0.47 | 0.89 | 0.53 |

## S5 W+A+KV  (MSAQ u2/gs8)
| B | prefill bf/mx/mq | decode bf/mx/mq | total bf/mx/mq | mq/bf | mq/mx | mx/bf |
|--:|--|--|--|--:|--:|--:|
| 1 | 254/294/292 | 14754/8979/7614 | 15007/9274/7906 | 0.53 | 0.85 | 0.62 |
| 8 | 2228/2227/2252 | 25555/16245/13942 | 27783/18472/16195 | 0.58 | 0.88 | 0.66 |
| 16 | 4379/4433/4408 | 38823/22236/19623 | 43202/26669/24031 | 0.56 | 0.90 | 0.62 |
| 32 | 7628/7697/7576 | 64890/30573/26301 | 72518/38271/33877 | 0.47 | 0.89 | 0.53 |

## S6 W+A+KV+AA  (MSAQ u2/gs8)
| B | prefill bf/mx/mq | decode bf/mx/mq | total bf/mx/mq | mq/bf | mq/mx | mx/bf |
|--:|--|--|--|--:|--:|--:|
| 1 | 254/295/292 | 14760/8980/7612 | 15014/9274/7904 | 0.53 | 0.85 | 0.62 |
| 8 | 2231/2237/2256 | 25626/16245/13959 | 27857/18482/16216 | 0.58 | 0.88 | 0.66 |
| 16 | 4384/4435/4410 | 38843/22217/19604 | 43227/26652/24015 | 0.56 | 0.90 | 0.62 |
| 32 | 7642/7689/7577 | 64918/30564/26308 | 72560/38253/33885 | 0.47 | 0.89 | 0.53 |

(mq=MSAQ mx=MXINT8 bf=bf16. prefill=TTFT(1024 tok), decode=integrated over 128 steps, total=sum.)
