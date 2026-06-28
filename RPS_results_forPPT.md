# RPS results (PPT) — MSAQ/bf16 & MXINT8/bf16 by batch

Offline request throughput (RPS) speedup vs **bf16**, per scope × batch. `mq`=MSAQ, `mx`=MXINT8. **>1 = faster than bf16.** Two workloads: L_out=128 (prefill-heavy) and L_out=512 (decode-heavy). Llama-3.1-8B 32L, L_in=1024, RTX PRO 4000 Blackwell. Source: `RPS_results.md`.

## S1 W-only
*weight-only (bf16 baseline KV)*

| B | mq/bf (L128) | mx/bf (L128) | mq/bf (L512) | mx/bf (L512) |
|--:|--:|--:|--:|--:|
| 1 | 1.82× | 1.52× | 1.91× | 1.59× |
| 8 | 1.14× | 1.13× | 1.20× | 1.18× |
| 16 | 1.07× | 0.99× | 1.08× | 0.99× |
| 32 | 1.04× | 0.99× | 1.06× | 0.99× |

## S3 KV-only
*KV-cache quant — u4/gs16 (vpack)*

| B | mq/bf (L128) | mx/bf (L128) | mq/bf (L512) | mx/bf (L512) |
|--:|--:|--:|--:|--:|
| 1 | 1.05× | 1.01× | 1.06× | 1.02× |
| 8 | 1.34× | 1.14× | 1.54× | 1.21× |
| 16 | 1.50× | 1.37× | 1.92× | 1.64× |
| 32 | 1.69× | 1.49× | 2.41× | 1.90× |

## S4 W-only+KV
*weight + KV quant*

| B | mq/bf (L128) | mx/bf (L128) | mq/bf (L512) | mx/bf (L512) |
|--:|--:|--:|--:|--:|
| 1 | 1.79× | 1.56× | 1.91× | 1.63× |
| 8 | 1.48× | 1.35× | 1.74× | 1.51× |
| 16 | 1.46× | 1.35× | 1.80× | 1.62× |
| 32 | 1.61× | 1.49× | 2.14× | 1.89× |

## S6 W+A+KV+AA
*full quant incl. attention-activation*

| B | mq/bf (L128) | mx/bf (L128) | mq/bf (L512) | mx/bf (L512) |
|--:|--:|--:|--:|--:|
| 1 | 1.79× | 1.55× | 1.90× | 1.62× |
| 8 | 1.46× | 1.34× | 1.72× | 1.51× |
| 16 | 1.46× | 1.35× | 1.80× | 1.62× |
| 32 | 1.61× | 1.49× | 2.14× | 1.90× |

