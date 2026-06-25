# Per-scope E2E latency v2 (S1-S6, incl. AA) — NVIDIA GeForce RTX 3090

Llama-3.1-8B 32L, L_in=1024, L_out=128, per-scope robust (u=4 where robust).
Current kernels: prefill = dequant-weight + cuBLAS (ties bf16); batched decode B in 2..15 = shared-activation wide GEMV (W-only 40us@M8; W+A dequant-in-staging float MAC 44us@M8 incl quant_act; both beat bf16 46us); B>=16 = dequant+cuBLAS; KV decode = (u,gs)-specialized wide-load. ms in µs; ratio <1 = MSAQ faster.

**S6 W+A+KV+AA** = full quant incl. attention activation×activation (Q,K,P,V). The decode attention kernel reads Q in bf16, so **AA decode latency == KV-decode (== S5)**; AA adds accuracy cost (~+0.9–1.0pp PPL, `precision/aa_attn_results.md`), not decode latency. Prefill attention is bf16 SDPA (AA-prefill loses to it). So S6 ≈ S5 in latency — that equality is the finding.

## S1 W-only  (MSAQ u3/gs16)
| B | prefill bf/mx/mq | decode bf/mx/mq | total bf/mx/mq | mq/bf | mq/mx | mx/bf |
|--:|--|--|--|--:|--:|--:|
| 1 | 254/280/280 | 2881/1876/1562 | 3135/2157/1842 | 0.59 | 0.85 | 0.69 |
| 8 | 2020/2036/2028 | 5733/4784/4725 | 7753/6821/6753 | 0.87 | 0.99 | 0.88 |
| 32 | 7795/7803/7772 | 12512/15635/15441 | 20306/23438/23213 | 1.14 | 0.99 | 1.15 |

## S2 W+A  (MSAQ u2/gs8)
| B | prefill bf/mx/mq | decode bf/mx/mq | total bf/mx/mq | mq/bf | mq/mx | mx/bf |
|--:|--|--|--|--:|--:|--:|
| 1 | 257/281/279 | 2894/1904/1753 | 3151/2184/2032 | 0.64 | 0.93 | 0.69 |
| 8 | 2019/2034/2026 | 5720/4799/4812 | 7739/6833/6838 | 0.88 | 1.00 | 0.88 |
| 32 | 7799/7801/7769 | 12511/15660/15575 | 20311/23461/23344 | 1.15 | 1.00 | 1.16 |

## S3 KV-only  (MSAQ u4/gs2)
| B | prefill bf/mx/mq | decode bf/mx/mq | total bf/mx/mq | mq/bf | mq/mx | mx/bf |
|--:|--|--|--|--:|--:|--:|
| 1 | 257/260/260 | 2895/2720/2743 | 3152/2980/3003 | 0.95 | 1.01 | 0.95 |
| 8 | 2020/2034/2040 | 5721/3934/3865 | 7740/5968/5904 | 0.76 | 0.99 | 0.77 |
| 32 | 7821/7899/7925 | 12519/5197/5253 | 20340/13095/13178 | 0.65 | 1.01 | 0.64 |

## S4 W-only+KV  (MSAQ u2/gs8)
| B | prefill bf/mx/mq | decode bf/mx/mq | total bf/mx/mq | mq/bf | mq/mx | mx/bf |
|--:|--|--|--|--:|--:|--:|
| 1 | 258/283/282 | 2895/1704/1578 | 3153/1986/1859 | 0.59 | 0.94 | 0.63 |
| 8 | 2019/2049/2045 | 5721/2867/3175 | 7739/4915/5221 | 0.67 | 1.06 | 0.64 |
| 32 | 7800/7848/7819 | 12503/8355/9167 | 20303/16202/16986 | 0.84 | 1.05 | 0.80 |

## S5 W+A+KV  (MSAQ u2/gs8)
| B | prefill bf/mx/mq | decode bf/mx/mq | total bf/mx/mq | mq/bf | mq/mx | mx/bf |
|--:|--|--|--|--:|--:|--:|
| 1 | 257/282/281 | 2880/1741/1641 | 3137/2023/1922 | 0.61 | 0.95 | 0.65 |
| 8 | 2018/2046/2041 | 5716/2904/3233 | 7734/4950/5274 | 0.68 | 1.07 | 0.64 |
| 32 | 7800/7846/7820 | 12489/8355/9172 | 20289/16201/16992 | 0.84 | 1.05 | 0.80 |

## S6 W+A+KV+AA  (MSAQ u2/gs8)
| B | prefill bf/mx/mq | decode bf/mx/mq | total bf/mx/mq | mq/bf | mq/mx | mx/bf |
|--:|--|--|--|--:|--:|--:|
| 1 | 257/282/282 | 2881/1744/1637 | 3138/2026/1919 | 0.61 | 0.95 | 0.65 |
| 8 | 2019/2046/2044 | 5720/2908/3227 | 7739/4953/5271 | 0.68 | 1.06 | 0.64 |
| 32 | 7796/7848/7819 | 12490/8358/9171 | 20286/16206/16990 | 0.84 | 1.05 | 0.80 |

(mq=MSAQ mx=MXINT8 bf=bf16. prefill=TTFT(1024 tok), decode=integrated over 128 steps, total=sum.)
