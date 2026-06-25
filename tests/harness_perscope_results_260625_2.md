# Per-scope E2E latency v2 (S1-S6, incl. AA) — NVIDIA RTX PRO 4000 Blackwell

Llama-3.1-8B 32L, L_in=1024, L_out=128, per-scope robust (u=4 where robust).
Current kernels: prefill = dequant-weight + cuBLAS (ties bf16); batched decode B in 2..15 = shared-activation wide GEMV (W-only 40us@M8; W+A dequant-in-staging float MAC 44us@M8 incl quant_act; both beat bf16 46us); B>=16 = dequant+cuBLAS; KV decode = (u,gs)-specialized wide-load. ms in µs; ratio <1 = MSAQ faster.

**S6 W+A+KV+AA** = full quant incl. attention activation×activation (Q,K,P,V). The decode attention kernel reads Q in bf16, so **AA decode latency == KV-decode (== S5)**; AA adds accuracy cost (~+0.9–1.0pp PPL, `precision/aa_attn_results.md`), not decode latency. Prefill attention is bf16 SDPA (AA-prefill loses to it). So S6 ≈ S5 in latency — that equality is the finding.

## S1 W-only  (MSAQ u3/gs16)
| B | prefill bf/mx/mq | decode bf/mx/mq | total bf/mx/mq | mq/bf | mq/mx | mx/bf |
|--:|--|--|--|--:|--:|--:|
| 1 | 256/288/285 | 3608/2245/1842 | 3865/2533/2127 | 0.55 | 0.84 | 0.66 |
| 8 | 2213/2237/2225 | 5971/4960/4983 | 8184/7196/7207 | 0.88 | 1.00 | 0.88 |
| 16 | 4471/4438/4434 | 8718/8866/7890 | 13189/13304/12324 | 0.93 | 0.93 | 1.01 |
| 32 | 7549/7543/7497 | 14313/14409/13367 | 21861/21952/20864 | 0.95 | 0.95 | 1.00 |

## S2 W+A  (MSAQ u2/gs8)
| B | prefill bf/mx/mq | decode bf/mx/mq | total bf/mx/mq | mq/bf | mq/mx | mx/bf |
|--:|--|--|--|--:|--:|--:|
| 1 | 254/291/290 | 3619/2287/2007 | 3873/2578/2296 | 0.59 | 0.89 | 0.67 |
| 8 | 2226/2247/2239 | 6016/5039/5137 | 8242/7286/7375 | 0.89 | 1.01 | 0.88 |
| 16 | 4406/4413/4428 | 8734/8881/7987 | 13139/13294/12415 | 0.94 | 0.93 | 1.01 |
| 32 | 7540/7544/7464 | 14306/14394/13465 | 21846/21938/20929 | 0.96 | 0.95 | 1.00 |

## S3 KV-only  (MSAQ u4/gs2)
| B | prefill bf/mx/mq | decode bf/mx/mq | total bf/mx/mq | mq/bf | mq/mx | mx/bf |
|--:|--|--|--|--:|--:|--:|
| 1 | 254/261/261 | 3620/3587/3480 | 3874/3848/3740 | 0.97 | 0.97 | 0.99 |
| 8 | 2228/2247/2249 | 5974/4929/3984 | 8202/7175/6233 | 0.76 | 0.87 | 0.87 |
| 16 | 4375/4413/4416 | 8757/5198/4629 | 13132/9611/9045 | 0.69 | 0.94 | 0.73 |
| 32 | 7564/7650/7654 | 14324/7019/5710 | 21888/14669/13363 | 0.61 | 0.91 | 0.67 |

## S4 W-only+KV  (MSAQ u2/gs8)
| B | prefill bf/mx/mq | decode bf/mx/mq | total bf/mx/mq | mq/bf | mq/mx | mx/bf |
|--:|--|--|--|--:|--:|--:|
| 1 | 254/294/292 | 3621/2198/1871 | 3875/2492/2163 | 0.56 | 0.87 | 0.64 |
| 8 | 2224/2261/2246 | 6011/3839/3395 | 8235/6100/5641 | 0.68 | 0.92 | 0.74 |
| 16 | 4425/4433/4455 | 8727/5249/4600 | 13152/9682/9055 | 0.69 | 0.94 | 0.74 |
| 32 | 7542/7602/7531 | 14293/7040/6025 | 21835/14642/13556 | 0.62 | 0.93 | 0.67 |

## S5 W+A+KV  (MSAQ u2/gs8)
| B | prefill bf/mx/mq | decode bf/mx/mq | total bf/mx/mq | mq/bf | mq/mx | mx/bf |
|--:|--|--|--|--:|--:|--:|
| 1 | 260/294/292 | 3631/2208/1874 | 3891/2502/2166 | 0.56 | 0.87 | 0.64 |
| 8 | 2225/2257/2244 | 6014/3853/3450 | 8238/6110/5695 | 0.69 | 0.93 | 0.74 |
| 16 | 4421/4436/4468 | 8728/5252/4594 | 13149/9689/9062 | 0.69 | 0.94 | 0.74 |
| 32 | 7536/7602/7519 | 14295/7040/6020 | 21831/14643/13539 | 0.62 | 0.92 | 0.67 |

## S6 W+A+KV+AA  (MSAQ u2/gs8)
| B | prefill bf/mx/mq | decode bf/mx/mq | total bf/mx/mq | mq/bf | mq/mx | mx/bf |
|--:|--|--|--|--:|--:|--:|
| 1 | 254/293/291 | 3614/2208/1874 | 3868/2501/2165 | 0.56 | 0.87 | 0.65 |
| 8 | 2222/2255/2244 | 6007/3884/3449 | 8228/6139/5693 | 0.69 | 0.93 | 0.75 |
| 16 | 4414/4435/4452 | 8728/5247/4599 | 13141/9683/9051 | 0.69 | 0.93 | 0.74 |
| 32 | 7542/7609/7524 | 14304/7049/6022 | 21846/14658/13545 | 0.62 | 0.92 | 0.67 |

(mq=MSAQ mx=MXINT8 bf=bf16. prefill=TTFT(1024 tok), decode=integrated over 128 steps, total=sum.)
