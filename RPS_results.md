# Offline throughput (RPS) — per-scope batch sweep

Re-label of the per-scope latency sweeps as **offline throughput at batch B** (requests/s). Derived from the per-scope `torch.cuda.Event` latencies (`tests/harness_perscope_results_260625*.jsonl`); **no re-measurement** (RPS = `B·1000/total_ms` is an exact transform of B and latency). `mq`=MSAQ, `mx`=MXINT8, `bf`=bf16. **S3 KV uses u4/gs16** (vpack, byte-roofline KV-read).

## Definitions & caveats

- **RPS (req/s)** = `B·1000 / total_ms`, `total_ms = ttft + Σ decode-step latency`. Each batch retires B requests.
- **decode (tok/s)** = `B·L_out·1000 / decode_ms` — pure generation throughput; isolates decode from prefill.
- **Offline/saturated, static batch** — throughput ceiling at fixed B, not SLA-constrained online RPS.
- **ratio > 1 = MSAQ faster.** For KV scopes (S3–S6) the win GROWS with L_out: prefill is a format-tie, so the longer the decode, the less the KV win is diluted (compare the two workloads below).
- **B≤32**: B≥64 OOMs on 24 GB.

## Workload: L_in=1024, L_out=128  (prefill-heavy)

### S1 W-only

**Request throughput — RPS (req/s)**

| B | bf16 | MXINT8 | MSAQ | mq/bf | mq/mx | mx/bf |
|--:|--:|--:|--:|--:|--:|--:|
| 1 | 0.258 | 0.394 | 0.469 | 1.82× | 1.19× | 1.52× |
| 8 | 0.975 | 1.106 | 1.108 | 1.14× | 1.00× | 1.13× |
| 16 | 1.212 | 1.196 | 1.293 | 1.07× | 1.08× | 0.99× |
| 32 | 1.454 | 1.447 | 1.520 | 1.04× | 1.05× | 0.99× |

**RPS-vs-B curve** (sparkline, norm to scope max 1.52 req/s; B = 1, 8, 16, 32)

```
  bf16    ▂▆▇█   0.26 → 1.45 req/s
  mxint8  ▃▆▇█   0.39 → 1.45 req/s
  msaq    ▃▆▇█   0.47 → 1.52 req/s
```

**Decode throughput (tok/s)** — generation-phase only

| B | bf16 | MXINT8 | MSAQ | mq/bf | mq/mx |
|--:|--:|--:|--:|--:|--:|
| 1 | 35.4 | 56.9 | 69.4 | 1.96× | 1.22× |
| 8 | 171.3 | 205.8 | 205.7 | 1.20× | 1.00× |
| 16 | 233.5 | 229.7 | 258.0 | 1.11× | 1.12× |
| 32 | 285.0 | 282.8 | 304.1 | 1.07× | 1.08× |

### S2 W+A

**Request throughput — RPS (req/s)**

| B | bf16 | MXINT8 | MSAQ | mq/bf | mq/mx | mx/bf |
|--:|--:|--:|--:|--:|--:|--:|
| 1 | 0.258 | 0.388 | 0.435 | 1.69× | 1.12× | 1.50× |
| 8 | 0.971 | 1.109 | 1.104 | 1.14× | 1.00× | 1.14× |
| 16 | 1.211 | 1.195 | 1.283 | 1.06× | 1.07× | 0.99× |
| 32 | 1.454 | 1.446 | 1.515 | 1.04× | 1.05× | 0.99× |

**RPS-vs-B curve** (sparkline, norm to scope max 1.52 req/s; B = 1, 8, 16, 32)

```
  bf16    ▂▆▇█   0.26 → 1.45 req/s
  mxint8  ▃▆▇█   0.39 → 1.45 req/s
  msaq    ▃▆▇█   0.44 → 1.52 req/s
```

**Decode throughput (tok/s)** — generation-phase only

| B | bf16 | MXINT8 | MSAQ | mq/bf | mq/mx |
|--:|--:|--:|--:|--:|--:|
| 1 | 35.3 | 56.0 | 63.8 | 1.81× | 1.14× |
| 8 | 170.6 | 206.6 | 205.1 | 1.20× | 0.99× |
| 16 | 233.0 | 229.4 | 255.0 | 1.09× | 1.11× |
| 32 | 284.8 | 282.5 | 302.2 | 1.06× | 1.07× |

### S3 KV-only

**Request throughput — RPS (req/s)**

| B | bf16 | MXINT8 | MSAQ | mq/bf | mq/mx | mx/bf |
|--:|--:|--:|--:|--:|--:|--:|
| 1 | 0.258 | 0.261 | 0.269 | 1.05× | 1.03× | 1.01× |
| 8 | 0.981 | 1.117 | 1.311 | 1.34× | 1.17× | 1.14× |
| 16 | 1.209 | 1.654 | 1.813 | 1.50× | 1.10× | 1.37× |
| 32 | 1.452 | 2.169 | 2.458 | 1.69× | 1.13× | 1.49× |

**RPS-vs-B curve** (sparkline, norm to scope max 2.46 req/s; B = 1, 8, 16, 32)

```
  bf16    ▁▄▄▅   0.26 → 1.45 req/s
  mxint8  ▁▄▆█   0.26 → 2.17 req/s
  msaq    ▁▅▆█   0.27 → 2.46 req/s
```

**Decode throughput (tok/s)** — generation-phase only

| B | bf16 | MXINT8 | MSAQ | mq/bf | mq/mx |
|--:|--:|--:|--:|--:|--:|
| 1 | 35.3 | 35.8 | 37.1 | 1.05× | 1.03× |
| 8 | 172.4 | 207.0 | 263.3 | 1.53× | 1.27× |
| 16 | 232.6 | 393.1 | 468.5 | 2.01× | 1.19× |
| 32 | 284.4 | 581.5 | 767.9 | 2.70× | 1.32× |

### S4 W-only+KV

**Request throughput — RPS (req/s)**

| B | bf16 | MXINT8 | MSAQ | mq/bf | mq/mx | mx/bf |
|--:|--:|--:|--:|--:|--:|--:|
| 1 | 0.257 | 0.401 | 0.461 | 1.79× | 1.15× | 1.56× |
| 8 | 0.974 | 1.312 | 1.439 | 1.48× | 1.10× | 1.35× |
| 16 | 1.211 | 1.640 | 1.763 | 1.46× | 1.07× | 1.35× |
| 32 | 1.457 | 2.170 | 2.345 | 1.61× | 1.08× | 1.49× |

**RPS-vs-B curve** (sparkline, norm to scope max 2.34 req/s; B = 1, 8, 16, 32)

```
  bf16    ▁▄▅▅   0.26 → 1.46 req/s
  mxint8  ▂▅▆█   0.40 → 2.17 req/s
  msaq    ▂▅▇█   0.46 → 2.34 req/s
```

**Decode throughput (tok/s)** — generation-phase only

| B | bf16 | MXINT8 | MSAQ | mq/bf | mq/mx |
|--:|--:|--:|--:|--:|--:|
| 1 | 35.3 | 58.3 | 68.3 | 1.94× | 1.17× |
| 8 | 171.3 | 267.2 | 310.4 | 1.81× | 1.16× |
| 16 | 233.2 | 388.4 | 443.3 | 1.90× | 1.14× |
| 32 | 285.3 | 580.4 | 677.2 | 2.37× | 1.17× |

### S5 W+A+KV

**Request throughput — RPS (req/s)**

| B | bf16 | MXINT8 | MSAQ | mq/bf | mq/mx | mx/bf |
|--:|--:|--:|--:|--:|--:|--:|
| 1 | 0.258 | 0.399 | 0.461 | 1.79× | 1.15× | 1.55× |
| 8 | 0.980 | 1.311 | 1.426 | 1.46× | 1.09× | 1.34× |
| 16 | 1.212 | 1.640 | 1.764 | 1.46× | 1.08× | 1.35× |
| 32 | 1.455 | 2.174 | 2.343 | 1.61× | 1.08× | 1.49× |

**RPS-vs-B curve** (sparkline, norm to scope max 2.34 req/s; B = 1, 8, 16, 32)

```
  bf16    ▁▄▅▅   0.26 → 1.46 req/s
  mxint8  ▂▅▆█   0.40 → 2.17 req/s
  msaq    ▂▅▇█   0.46 → 2.34 req/s
```

**Decode throughput (tok/s)** — generation-phase only

| B | bf16 | MXINT8 | MSAQ | mq/bf | mq/mx |
|--:|--:|--:|--:|--:|--:|
| 1 | 35.4 | 57.9 | 68.3 | 1.93× | 1.18× |
| 8 | 172.7 | 266.9 | 305.8 | 1.77× | 1.15× |
| 16 | 233.3 | 388.5 | 443.8 | 1.90× | 1.14× |
| 32 | 285.0 | 581.3 | 676.5 | 2.37× | 1.16× |

### S6 W+A+KV+AA

**Request throughput — RPS (req/s)**

| B | bf16 | MXINT8 | MSAQ | mq/bf | mq/mx | mx/bf |
|--:|--:|--:|--:|--:|--:|--:|
| 1 | 0.258 | 0.399 | 0.461 | 1.79× | 1.15× | 1.55× |
| 8 | 0.979 | 1.313 | 1.426 | 1.46× | 1.09× | 1.34× |
| 16 | 1.211 | 1.640 | 1.765 | 1.46× | 1.08× | 1.35× |
| 32 | 1.457 | 2.171 | 2.343 | 1.61× | 1.08× | 1.49× |

**RPS-vs-B curve** (sparkline, norm to scope max 2.34 req/s; B = 1, 8, 16, 32)

```
  bf16    ▁▄▅▅   0.26 → 1.46 req/s
  mxint8  ▂▅▆█   0.40 → 2.17 req/s
  msaq    ▂▅▇█   0.46 → 2.34 req/s
```

**Decode throughput (tok/s)** — generation-phase only

| B | bf16 | MXINT8 | MSAQ | mq/bf | mq/mx |
|--:|--:|--:|--:|--:|--:|
| 1 | 35.3 | 57.9 | 68.2 | 1.93× | 1.18× |
| 8 | 172.6 | 266.9 | 305.6 | 1.77× | 1.14× |
| 16 | 233.2 | 388.3 | 443.8 | 1.90× | 1.14× |
| 32 | 285.4 | 580.4 | 676.8 | 2.37× | 1.17× |

## Workload: L_in=1024, L_out=512  (decode-heavy)

### S1 W-only

**Request throughput — RPS (req/s)**

| B | bf16 | MXINT8 | MSAQ | mq/bf | mq/mx | mx/bf |
|--:|--:|--:|--:|--:|--:|--:|
| 1 | 0.067 | 0.106 | 0.128 | 1.91× | 1.20× | 1.59× |
| 8 | 0.284 | 0.335 | 0.341 | 1.20× | 1.02× | 1.18× |
| 16 | 0.370 | 0.365 | 0.401 | 1.08× | 1.10× | 0.99× |
| 32 | 0.441 | 0.438 | 0.466 | 1.06× | 1.06× | 0.99× |

**RPS-vs-B curve** (sparkline, norm to scope max 0.47 req/s; B = 1, 8, 16, 32)

```
  bf16    ▂▅▇█   0.07 → 0.44 req/s
  mxint8  ▂▆▇█   0.11 → 0.44 req/s
  msaq    ▃▆▇█   0.13 → 0.47 req/s
```

**Decode throughput (tok/s)** — generation-phase only

| B | bf16 | MXINT8 | MSAQ | mq/bf | mq/mx |
|--:|--:|--:|--:|--:|--:|
| 1 | 34.8 | 56.0 | 67.8 | 1.95× | 1.21× |
| 8 | 157.7 | 189.5 | 193.3 | 1.23× | 1.02× |
| 16 | 210.8 | 207.9 | 230.9 | 1.10× | 1.11× |
| 32 | 252.5 | 250.7 | 268.0 | 1.06× | 1.07× |

### S2 W+A

**Request throughput — RPS (req/s)**

| B | bf16 | MXINT8 | MSAQ | mq/bf | mq/mx | mx/bf |
|--:|--:|--:|--:|--:|--:|--:|
| 1 | 0.067 | 0.104 | 0.118 | 1.77× | 1.13× | 1.56× |
| 8 | 0.288 | 0.336 | 0.334 | 1.16× | 0.99× | 1.17× |
| 16 | 0.370 | 0.365 | 0.397 | 1.07× | 1.09× | 0.99× |
| 32 | 0.441 | 0.439 | 0.463 | 1.05× | 1.05× | 1.00× |

**RPS-vs-B curve** (sparkline, norm to scope max 0.46 req/s; B = 1, 8, 16, 32)

```
  bf16    ▂▅▇█   0.07 → 0.44 req/s
  mxint8  ▂▆▇█   0.10 → 0.44 req/s
  msaq    ▃▆▇█   0.12 → 0.46 req/s
```

**Decode throughput (tok/s)** — generation-phase only

| B | bf16 | MXINT8 | MSAQ | mq/bf | mq/mx |
|--:|--:|--:|--:|--:|--:|
| 1 | 34.7 | 55.0 | 62.5 | 1.80× | 1.14× |
| 8 | 160.2 | 189.6 | 188.5 | 1.18× | 0.99× |
| 16 | 210.9 | 207.8 | 228.0 | 1.08× | 1.10× |
| 32 | 252.6 | 251.0 | 265.9 | 1.05× | 1.06× |

### S3 KV-only

**Request throughput — RPS (req/s)**

| B | bf16 | MXINT8 | MSAQ | mq/bf | mq/mx | mx/bf |
|--:|--:|--:|--:|--:|--:|--:|
| 1 | 0.067 | 0.068 | 0.070 | 1.06× | 1.04× | 1.02× |
| 8 | 0.288 | 0.349 | 0.443 | 1.54× | 1.27× | 1.21× |
| 16 | 0.369 | 0.605 | 0.709 | 1.92× | 1.17× | 1.64× |
| 32 | 0.440 | 0.839 | 1.061 | 2.41× | 1.27× | 1.90× |

**RPS-vs-B curve** (sparkline, norm to scope max 1.06 req/s; B = 1, 8, 16, 32)

```
  bf16    ▁▃▃▄   0.07 → 0.44 req/s
  mxint8  ▁▃▅▇   0.07 → 0.84 req/s
  msaq    ▁▄▆█   0.07 → 1.06 req/s
```

**Decode throughput (tok/s)** — generation-phase only

| B | bf16 | MXINT8 | MSAQ | mq/bf | mq/mx |
|--:|--:|--:|--:|--:|--:|
| 1 | 34.7 | 35.3 | 36.7 | 1.06× | 1.04× |
| 8 | 160.0 | 197.7 | 258.5 | 1.62× | 1.31× |
| 16 | 210.4 | 372.0 | 452.1 | 2.15× | 1.22× |
| 32 | 252.1 | 539.3 | 730.9 | 2.90× | 1.36× |

### S4 W-only+KV

**Request throughput — RPS (req/s)**

| B | bf16 | MXINT8 | MSAQ | mq/bf | mq/mx | mx/bf |
|--:|--:|--:|--:|--:|--:|--:|
| 1 | 0.067 | 0.109 | 0.127 | 1.91× | 1.17× | 1.63× |
| 8 | 0.287 | 0.434 | 0.500 | 1.74× | 1.15× | 1.51× |
| 16 | 0.370 | 0.600 | 0.665 | 1.80× | 1.11× | 1.62× |
| 32 | 0.442 | 0.836 | 0.944 | 2.14× | 1.13× | 1.89× |

**RPS-vs-B curve** (sparkline, norm to scope max 0.94 req/s; B = 1, 8, 16, 32)

```
  bf16    ▁▃▄▄   0.07 → 0.44 req/s
  mxint8  ▁▄▆█   0.11 → 0.84 req/s
  msaq    ▂▅▆█   0.13 → 0.94 req/s
```

**Decode throughput (tok/s)** — generation-phase only

| B | bf16 | MXINT8 | MSAQ | mq/bf | mq/mx |
|--:|--:|--:|--:|--:|--:|
| 1 | 34.6 | 57.5 | 67.5 | 1.95× | 1.17× |
| 8 | 159.7 | 252.7 | 298.0 | 1.87× | 1.18× |
| 16 | 210.8 | 368.4 | 417.2 | 1.98× | 1.13× |
| 32 | 252.8 | 535.7 | 622.8 | 2.46× | 1.16× |

### S5 W+A+KV

**Request throughput — RPS (req/s)**

| B | bf16 | MXINT8 | MSAQ | mq/bf | mq/mx | mx/bf |
|--:|--:|--:|--:|--:|--:|--:|
| 1 | 0.067 | 0.108 | 0.126 | 1.90× | 1.17× | 1.62× |
| 8 | 0.288 | 0.433 | 0.494 | 1.72× | 1.14× | 1.50× |
| 16 | 0.370 | 0.600 | 0.666 | 1.80× | 1.11× | 1.62× |
| 32 | 0.441 | 0.836 | 0.945 | 2.14× | 1.13× | 1.89× |

**RPS-vs-B curve** (sparkline, norm to scope max 0.94 req/s; B = 1, 8, 16, 32)

```
  bf16    ▁▃▄▄   0.07 → 0.44 req/s
  mxint8  ▁▄▆█   0.11 → 0.84 req/s
  msaq    ▂▅▆█   0.13 → 0.94 req/s
```

**Decode throughput (tok/s)** — generation-phase only

| B | bf16 | MXINT8 | MSAQ | mq/bf | mq/mx |
|--:|--:|--:|--:|--:|--:|
| 1 | 34.7 | 57.0 | 67.2 | 1.94× | 1.18× |
| 8 | 160.3 | 252.1 | 293.8 | 1.83× | 1.17× |
| 16 | 211.0 | 368.4 | 417.5 | 1.98× | 1.13× |
| 32 | 252.5 | 535.9 | 622.9 | 2.47× | 1.16× |

### S6 W+A+KV+AA

**Request throughput — RPS (req/s)**

| B | bf16 | MXINT8 | MSAQ | mq/bf | mq/mx | mx/bf |
|--:|--:|--:|--:|--:|--:|--:|
| 1 | 0.067 | 0.108 | 0.127 | 1.90× | 1.17× | 1.62× |
| 8 | 0.287 | 0.433 | 0.493 | 1.72× | 1.14× | 1.51× |
| 16 | 0.370 | 0.600 | 0.666 | 1.80× | 1.11× | 1.62× |
| 32 | 0.441 | 0.837 | 0.944 | 2.14× | 1.13× | 1.90× |

**RPS-vs-B curve** (sparkline, norm to scope max 0.94 req/s; B = 1, 8, 16, 32)

```
  bf16    ▁▃▄▄   0.07 → 0.44 req/s
  mxint8  ▁▄▆█   0.11 → 0.84 req/s
  msaq    ▂▅▆█   0.13 → 0.94 req/s
```

**Decode throughput (tok/s)** — generation-phase only

| B | bf16 | MXINT8 | MSAQ | mq/bf | mq/mx |
|--:|--:|--:|--:|--:|--:|
| 1 | 34.7 | 57.0 | 67.3 | 1.94× | 1.18× |
| 8 | 159.8 | 252.1 | 293.4 | 1.84× | 1.16× |
| 16 | 210.9 | 368.7 | 417.9 | 1.98× | 1.13× |
| 32 | 252.4 | 536.0 | 622.8 | 2.47× | 1.16× |

## Workload contrast — MSAQ/MXINT8 RPS by L_out

How much the KV-quant win grows when decode dominates total (prefill = tie). Each cell = RPS mq/mx.

**S1 W-only**

| L_out | B=1 | B=8 | B=16 | B=32 |
|---|--:|--:|--:|--:|
| 128 | 1.19× | 1.00× | 1.08× | 1.05× |
| 512 | 1.20× | 1.02× | 1.10× | 1.06× |

**S2 W+A**

| L_out | B=1 | B=8 | B=16 | B=32 |
|---|--:|--:|--:|--:|
| 128 | 1.12× | 1.00× | 1.07× | 1.05× |
| 512 | 1.13× | 0.99× | 1.09× | 1.05× |

**S3 KV-only**

| L_out | B=1 | B=8 | B=16 | B=32 |
|---|--:|--:|--:|--:|
| 128 | 1.03× | 1.17× | 1.10× | 1.13× |
| 512 | 1.04× | 1.27× | 1.17× | 1.27× |

**S4 W-only+KV**

| L_out | B=1 | B=8 | B=16 | B=32 |
|---|--:|--:|--:|--:|
| 128 | 1.15× | 1.10× | 1.07× | 1.08× |
| 512 | 1.17× | 1.15× | 1.11× | 1.13× |

**S5 W+A+KV**

| L_out | B=1 | B=8 | B=16 | B=32 |
|---|--:|--:|--:|--:|
| 128 | 1.15× | 1.09× | 1.08× | 1.08× |
| 512 | 1.17× | 1.14× | 1.11× | 1.13× |

**S6 W+A+KV+AA**

| L_out | B=1 | B=8 | B=16 | B=32 |
|---|--:|--:|--:|--:|
| 128 | 1.15× | 1.09× | 1.08× | 1.08× |
| 512 | 1.17× | 1.14× | 1.11× | 1.13× |

---

## S3 KV-only — rotation-free u3 variants (L_in=1024, L_out=128)

KV **u3** (5-bit unshared) is robust **without** H128 rotation (gs16 +1.20% PPL, gs32 +1.19%), whereas
**u4** needs rotation (+2.58% with, +6.44% without — see `KV_cache_analysis.md`). How does a
rotation-free u3 KV compare to the deployed **u4/gs16** in RPS? (S3 = bf16 weights, MSAQ KV.)

**Request throughput — RPS (req/s):**

| B | bf16 | MXINT8 | u4/gs16 (4.5b, +rot) | u3/gs16 (5.5b, **no-rot**) | u3/gs32 (5.5b, **no-rot**) |
|--:|--:|--:|--:|--:|--:|
| 1 | 0.258 | 0.261 | 0.269 | 0.270 | 0.265 |
| 8 | 0.981 | 1.117 | 1.311 | 1.304 | 1.241 |
| 16 | 1.209 | 1.654 | 1.813 | 1.801 | 1.692 |
| 32 | 1.452 | 2.169 | 2.458 | **2.465** | 2.311 |

**Speedup (mq/mx · mq/bf):**

| B | u4/gs16 | u3/gs16 (no-rot) | u3/gs32 (no-rot) |
|--:|--:|--:|--:|
| 8 | 1.17× · 1.34× | 1.17× · 1.33× | 1.11× · 1.27× |
| 16 | 1.10× · 1.50× | 1.09× · 1.49× | 1.02× · 1.40× |
| 32 | 1.13× · 1.69× | **1.14× · 1.70×** | 1.07× · 1.59× |

**Findings:**
- **u3/gs16 ≈ u4/gs16 in RPS** (B32 2.465 vs 2.458; mq/mx 1.14 vs 1.13) — matches the deployed config
  while being **rotation-free** and **more accurate** (+1.20% vs +2.58% PPL). Cost: **+1.0 bits/elem**
  (5.5 vs 4.5). The extra byte doesn't slow RPS because KV decode is L1TEX/staging-bound (the ~4%
  isolated-decode gap dilutes to ~0 in this prefill-heavy L_out=128 workload).
- **u3/gs32 is worse** (B32 RPS 2.311 vs u3/gs16 2.465) despite identical bytes (both 5.5b) — the gs32
  kernel path is slower with no accuracy gain. **u3/gs16 dominates u3/gs32.**
- Bottom line: **u3/gs16 is a compelling rotation-free alternative to u4/gs16** — same RPS, no QuaRot
  rotation, better accuracy, at +1 bit/elem.

