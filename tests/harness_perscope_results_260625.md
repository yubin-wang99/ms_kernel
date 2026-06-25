# Per-scope E2E latency v2 (S1-S6, incl. AA) — NVIDIA RTX PRO 4000 Blackwell

Llama-3.1-8B 32L, L_in=1024, L_out=128, per-scope robust (u=4 where robust).
Current kernels: prefill = dequant-weight + cuBLAS (ties bf16); batched decode B in 2..15 = shared-activation wide GEMV (W-only 40us@M8; W+A dequant-in-staging float MAC 44us@M8 incl quant_act; both beat bf16 46us); B>=16 = dequant+cuBLAS; KV decode = (u,gs)-specialized wide-load. ms in µs; ratio <1 = MSAQ faster.

**S6 W+A+KV+AA** = full quant incl. attention activation×activation (Q,K,P,V). The decode attention kernel reads Q in bf16, so **AA decode latency == KV-decode (== S5)**; AA adds accuracy cost (~+0.9–1.0pp PPL, `precision/aa_attn_results.md`), not decode latency. Prefill attention is bf16 SDPA (AA-prefill loses to it). So S6 ≈ S5 in latency — that equality is the finding.

## S1 W-only  (MSAQ u3/gs16)
| B | prefill bf/mx/mq | decode bf/mx/mq | total bf/mx/mq | mq/bf | mq/mx | mx/bf |
|--:|--|--|--|--:|--:|--:|
| 16 | 4428/4470/4464 | 8774/9058/7877 | 13202/13529/12341 | 0.93 | 0.91 | 1.02 |
| 32 | 7542/7554/7513 | 14303/14427/13393 | 21844/21980/20907 | 0.96 | 0.95 | 1.01 |

## S2 W+A  (MSAQ u2/gs8)
| B | prefill bf/mx/mq | decode bf/mx/mq | total bf/mx/mq | mq/bf | mq/mx | mx/bf |
|--:|--|--|--|--:|--:|--:|
| 16 | 4403/4404/4423 | 8809/8926/8005 | 13212/13330/12429 | 0.94 | 0.93 | 1.01 |
| 32 | 7571/7565/7494 | 14328/14454/13490 | 21900/22019/20984 | 0.96 | 0.95 | 1.01 |

## S3 KV-only  (MSAQ u4/gs2)
| B | prefill bf/mx/mq | decode bf/mx/mq | total bf/mx/mq | mq/bf | mq/mx | mx/bf |
|--:|--|--|--|--:|--:|--:|
| 16 | 4402/4430/4426 | 8814/5213/4639 | 13216/9643/9065 | 0.69 | 0.94 | 0.73 |
| 32 | 7590/7666/7667 | 14363/7031/5709 | 21953/14696/13376 | 0.61 | 0.91 | 0.67 |

## S4 W-only+KV  (MSAQ u2/gs8)
| B | prefill bf/mx/mq | decode bf/mx/mq | total bf/mx/mq | mq/bf | mq/mx | mx/bf |
|--:|--|--|--|--:|--:|--:|
| 16 | 4411/4436/4429 | 8816/5275/4616 | 13227/9711/9046 | 0.68 | 0.93 | 0.73 |
| 32 | 7570/7621/7537 | 14337/7060/6032 | 21907/14681/13569 | 0.62 | 0.92 | 0.67 |

## S5 W+A+KV  (MSAQ u2/gs8)
| B | prefill bf/mx/mq | decode bf/mx/mq | total bf/mx/mq | mq/bf | mq/mx | mx/bf |
|--:|--|--|--|--:|--:|--:|
| 16 | 4395/4433/4440 | 8802/5279/4616 | 13197/9712/9056 | 0.69 | 0.93 | 0.74 |
| 32 | 7554/7614/7535 | 14327/7057/6034 | 21881/14671/13569 | 0.62 | 0.92 | 0.67 |

## S6 W+A+KV+AA  (MSAQ u2/gs8)
| B | prefill bf/mx/mq | decode bf/mx/mq | total bf/mx/mq | mq/bf | mq/mx | mx/bf |
|--:|--|--|--|--:|--:|--:|
| 16 | 4393/4430/4454 | 8804/5267/4615 | 13197/9697/9069 | 0.69 | 0.94 | 0.73 |
| 32 | 7565/7619/7540 | 14330/7061/6028 | 21895/14681/13568 | 0.62 | 0.92 | 0.67 |

(mq=MSAQ mx=MXINT8 bf=bf16. prefill=TTFT(1024 tok), decode=integrated over 128 steps, total=sum.)
