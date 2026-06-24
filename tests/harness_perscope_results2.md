# Per-scope E2E latency v2 (S1-S6, incl. AA) — NVIDIA GeForce RTX 3090

Llama-3.1-8B 32L, L_in=1024, L_out=128, per-scope robust (u=4 where robust).
Current kernels: prefill = dequant-weight + cuBLAS (ties bf16); batched decode (B>=16) = dequant+cuBLAS; KV decode = (u,gs)-specialized wide-load. ms in µs; ratio <1 = MSAQ faster.

**S6 W+A+KV+AA** = full quant incl. attention activation×activation (Q,K,P,V). The decode attention kernel reads Q in bf16, so **AA decode latency == KV-decode (== S5)**; AA adds accuracy cost (~+0.9–1.0pp PPL, `precision/aa_attn_results.md`), not decode latency. Prefill attention is bf16 SDPA (AA-prefill loses to it). So S6 ≈ S5 in latency — that equality is the finding.

## S1 W-only  (MSAQ u3/gs16)
| B | prefill bf/mx/mq | decode bf/mx/mq | total bf/mx/mq | mq/bf | mq/mx | mx/bf |
|--:|--|--|--|--:|--:|--:|
| 1 | 255/280/280 | 2883/1878/1562 | 3138/2158/1842 | 0.59 | 0.85 | 0.69 |
| 8 | 2019/2035/2029 | 5723/6702/6673 | 7742/8737/8702 | 1.12 | 1.00 | 1.13 |
| 32 | 7792/7797/7760 | 12487/15623/15408 | 20279/23420/23168 | 1.14 | 0.99 | 1.15 |

## S2 W+A  (MSAQ u2/gs8)
| B | prefill bf/mx/mq | decode bf/mx/mq | total bf/mx/mq | mq/bf | mq/mx | mx/bf |
|--:|--|--|--|--:|--:|--:|
| 1 | 258/281/279 | 2895/1904/1753 | 3152/2185/2033 | 0.64 | 0.93 | 0.69 |
| 8 | 2018/2032/2028 | 5722/7269/6997 | 7739/9302/9024 | 1.17 | 0.97 | 1.20 |
| 32 | 7794/7793/7754 | 12497/15615/15573 | 20291/23408/23327 | 1.15 | 1.00 | 1.15 |

## S3 KV-only  (MSAQ u4/gs2)
| B | prefill bf/mx/mq | decode bf/mx/mq | total bf/mx/mq | mq/bf | mq/mx | mx/bf |
|--:|--|--|--|--:|--:|--:|
| 1 | 257/259/259 | 2892/2721/2743 | 3149/2980/3002 | 0.95 | 1.01 | 0.95 |
| 8 | 2021/2033/2041 | 5721/3926/3855 | 7742/5959/5895 | 0.76 | 0.99 | 0.77 |
| 32 | 7808/7885/7911 | 12506/5172/5227 | 20314/13057/13138 | 0.65 | 1.01 | 0.64 |

## S4 W-only+KV  (MSAQ u2/gs8)
| B | prefill bf/mx/mq | decode bf/mx/mq | total bf/mx/mq | mq/bf | mq/mx | mx/bf |
|--:|--|--|--|--:|--:|--:|
| 1 | 258/283/284 | 2880/1705/1578 | 3138/1988/1863 | 0.59 | 0.94 | 0.63 |
| 8 | 2015/2044/2037 | 5713/4957/5212 | 7728/7001/7249 | 0.94 | 1.04 | 0.91 |
| 32 | 7792/7834/7803 | 12479/8356/9165 | 20271/16190/16968 | 0.84 | 1.05 | 0.80 |

## S5 W+A+KV  (MSAQ u2/gs8)
| B | prefill bf/mx/mq | decode bf/mx/mq | total bf/mx/mq | mq/bf | mq/mx | mx/bf |
|--:|--|--|--|--:|--:|--:|
| 1 | 257/282/282 | 2881/1744/1637 | 3137/2027/1919 | 0.61 | 0.95 | 0.65 |
| 8 | 2017/2043/2038 | 5716/5501/5514 | 7732/7544/7552 | 0.98 | 1.00 | 0.98 |
| 32 | 7788/7828/7806 | 12478/8342/9162 | 20266/16170/16968 | 0.84 | 1.05 | 0.80 |

## S6 W+A+KV+AA  (MSAQ u2/gs8)
| B | prefill bf/mx/mq | decode bf/mx/mq | total bf/mx/mq | mq/bf | mq/mx | mx/bf |
|--:|--|--|--|--:|--:|--:|
| 1 | 257/282/281 | 2881/1745/1641 | 3139/2027/1922 | 0.61 | 0.95 | 0.65 |
| 8 | 2016/2043/2039 | 5721/5503/5529 | 7737/7546/7568 | 0.98 | 1.00 | 0.98 |
| 32 | 7788/7831/7807 | 12489/8359/9169 | 20277/16190/16976 | 0.84 | 1.05 | 0.80 |

(mq=MSAQ mx=MXINT8 bf=bf16. prefill=TTFT(1024 tok), decode=integrated over 128 steps, total=sum.)
