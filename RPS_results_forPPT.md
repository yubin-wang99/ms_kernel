# RPS results (PPT) — MSAQ vs MXINT8 (and vs bf16) by batch

Offline request throughput (RPS) speedup, per scope × batch. `mq`=MSAQ `mx`=MXINT8 `bf`=bf16. **mq/mx** = MSAQ over MXINT8 (the competitive comparison); **mq/bf** = MSAQ over fp16 baseline (headline). >1 = MSAQ faster. Two workloads: L_out=128 (prefill-heavy) / 512 (decode-heavy). Llama-3.1-8B 32L, L_in=1024, RTX PRO 4000 Blackwell. Source: `RPS_results.md`.

> Note: the isolated KV-read **kernel** win is larger (S3 B32 ≈1.9× = ratio 0.52, byte-roofline); RPS dilutes it because prefill is a format-tie and bf16 weight-GEMVs are format-neutral. The longer L_out, the closer RPS gets to the kernel win.

## S1 W-only
*weight-only (bf16 KV)*

| B | mq/mx (L128) | mq/mx (L512) | mq/bf (L128) | mq/bf (L512) |
|--:|--:|--:|--:|--:|
| 1 | 1.19× | 1.20× | 1.82× | 1.91× |
| 8 | 1.00× | 1.02× | 1.14× | 1.20× |
| 16 | 1.08× | 1.10× | 1.07× | 1.08× |
| 32 | 1.05× | 1.06× | 1.04× | 1.06× |

## S3 KV-only
*KV-cache quant — u4/gs16 (vpack)*

| B | mq/mx (L128) | mq/mx (L512) | mq/bf (L128) | mq/bf (L512) |
|--:|--:|--:|--:|--:|
| 1 | 1.03× | 1.04× | 1.05× | 1.06× |
| 8 | 1.17× | 1.27× | 1.34× | 1.54× |
| 16 | 1.10× | 1.17× | 1.50× | 1.92× |
| 32 | 1.13× | 1.27× | 1.69× | 2.41× |

## S4 W-only+KV
*weight + KV quant*

| B | mq/mx (L128) | mq/mx (L512) | mq/bf (L128) | mq/bf (L512) |
|--:|--:|--:|--:|--:|
| 1 | 1.15× | 1.17× | 1.79× | 1.91× |
| 8 | 1.10× | 1.15× | 1.48× | 1.74× |
| 16 | 1.07× | 1.11× | 1.46× | 1.80× |
| 32 | 1.08× | 1.13× | 1.61× | 2.14× |

## S6 W+A+KV+AA
*full quant incl. attn-activation*

| B | mq/mx (L128) | mq/mx (L512) | mq/bf (L128) | mq/bf (L512) |
|--:|--:|--:|--:|--:|
| 1 | 1.15× | 1.17× | 1.79× | 1.90× |
| 8 | 1.09× | 1.14× | 1.46× | 1.72× |
| 16 | 1.08× | 1.11× | 1.46× | 1.80× |
| 32 | 1.08× | 1.13× | 1.61× | 2.14× |

