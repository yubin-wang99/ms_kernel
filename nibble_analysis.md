# Does nibble alignment matter for WEIGHT read? — No.

Empirical study (RTX PRO 4000 Blackwell, OUT=K=4096, measured 2026-06-28). "Nibble alignment" =
**u4** (the unshared field is 4 bits = a clean nibble → 2 byte-aligned fields, cheap `bfe` unpack)
vs **u2/u3** straddle (5/6-bit unshared field crosses byte boundaries → streaming rolling-buffer
unpack). Question: does the cheap nibble unpack make weight READ faster?

`mq/mx` = MSAQ / MXINT8 time. Bits/elem: u2≈6.5, u3≈5.5, u4≈4.5 (vs MXINT8 8.25).

---

## TL;DR

**Nibble alignment does NOT matter for weight read.** Every weight-read vehicle is bound by
something *other* than the quantized read/unpack (output write, WMMA compute, or launch latency), so
the cheap nibble unpack never converts to time. Measured u4 (nibble) directly against u2/u3 (straddle)
on every vehicle, including a u4/gs16 spec added to the deployed throughput kernel: u4 is identical
(fused_skinny) or even *slower* (decode GEMV), never a meaningful win. This is the **opposite** of KV
read, where nibble (u4) is decisive because the unpack/staging is on the critical path.

---

## Measurements — every weight-read vehicle

### 1. Pure dequant read (`ms_dequant_bf16`) — prefill & B≥16 fallback vehicle
Reads quantized weight → materializes bf16 [OUT,K] for cuBLAS.

| config | bits/elem | time | nibble? | mq/mx |
|---|--:|--:|---|--:|
| MXINT8 | 8.25 | 52.1 µs | — | 1.00 |
| u2/gs16 | 6.50 | 37.7 µs | straddle | 0.72 |
| u3/gs16 | 5.50 | 36.6 µs | straddle | 0.70 |
| **u4/gs16** | 4.50 | **35.0 µs** | **nibble** | 0.67 |
| u2/gs8 | 6.50 | 38.5 µs | straddle | 0.74 |
| u3/gs8 | 5.75 | 36.7 µs | straddle | 0.71 |
| **u4/gs8** | 4.75 | **36.1 µs** | **nibble** | 0.69 |

**1.44× fewer bytes (u2→u4) buys only 1.10× time** (38.5→35.0 µs). Nibble u4 is only **~2–4% faster**
than straddle u3. **Why: this kernel is bound by the bf16 OUTPUT WRITE** (OUT×K×2 = 33.5 MB) — far
larger than the quantized read (9–13 MB) — so the read/unpack difference (bytes, nibble) is mostly
invisible.

### 2. Decode GEMV (`wonly_gemv_wide`, M=1) — small-batch read
| config | time | mq/mx |
|---|--:|--:|
| MXINT8 | 22.6 µs | 1.00 |
| u2 (gs8/16) | 16.5 µs | 0.73 |
| u3 (gs8/16) | 16.5 µs | 0.73 |
| **u4 (gs8/16)** | **24.7 µs** | **1.09** |

**Nibble u4 is SLOWER, not faster** (24.7 vs 16.5 µs). At M=1 the kernel is launch/latency-bound; the
u4 wide-load path carries more overhead, so straddle u2/u3 win. u2==u3 exactly → not byte-bound either.

### 3. Deployed throughput read (`wonly_gemm_fused_skinny`, B≥16) — the decisive test
Stock kernel compiles only the deployed straddle configs (u3/gs16, u2/gs8); a **u4/gs16 spec was
added** (`csrc/wa_gemm.cu` dispatch, bit-exact rel_fro 3.4e-5) purely to isolate nibble vs straddle
at the SAME gs.

| B | config | bits | time | nibble? | mq/mx |
|--:|---|--:|--:|---|--:|
| 16 | MXINT8 | 8.25 | 49.5 µs | — | 1.00 |
| 16 | u2/gs8 | 6.50 | 30.8 µs | straddle | 0.62 |
| 16 | u3/gs16 | 5.50 | 30.7 µs | straddle | 0.62 |
| 16 | **u4/gs16** | **4.50** | **30.8 µs** | **nibble** | 0.62 |
| 32 | u2/gs8 | 6.50 | 35.8 µs | straddle | 0.67 |
| 32 | u3/gs16 | 5.50 | 35.6 µs | straddle | 0.67 |
| 32 | **u4/gs16** | **4.50** | **35.8 µs** | **nibble** | 0.67 |

**u4 (nibble, 4.5b), u3 (straddle, 5.5b), u2 (straddle, 6.5b) are ALL IDENTICAL.** A 1.44× byte
difference (u2→u4) and the nibble vs straddle distinction both produce **zero** time difference → the
kernel is bound by the **WMMA compute tile**, not the weight read. Nibble alignment is irrelevant here
even when explicitly measured. (u4 weight is also accuracy-dead at +6.9% PPL, so it's never deployed —
but the point stands: even if it were, the read is identical.)

### 4. (Non-deployed) fused-tile GEMM (`wonly_gemm`, M=512)
u4 1.73 < u3 1.99 < u2 2.02 ms — here nibble *is* faster. But this path is **12× slower than cuBLAS**
(0.16 ms) and is not the deployed prefill (which uses dequant→cuBLAS). Irrelevant to deployment.

---

## Why weight read is nibble-insensitive (binding constraint per vehicle)

| vehicle | bottleneck | nibble effect |
|---|---|---|
| dequant→bf16 | bf16 output WRITE (33.5 MB ≫ 9–13 MB read) | ~2–4% (invisible) |
| GEMV M=1 | launch / latency (tiny) | u4 slower (path overhead) |
| fused_skinny B≥16 | WMMA compute tile (read not byte-bound) | none (u2/gs8==u3/gs16; no u4) |

In none of them is the quantized **read/unpack** the critical path, so the nibble's cheap-`bfe`
advantage never converts. Plus weight accuracy forces u2/u3 (u4 dead), so deployment is non-nibble.

## Contrast — KV read, where nibble DOES matter

| | nibble (u4) decisive? | binding constraint |
|---|---|---|
| **weight read** | ❌ no | bf16 write / WMMA compute / launch |
| **KV read** | ✅ yes | per-element unpack + L1TEX shared staging (on the critical path) |

KV decode is L1TEX/shared-bound on per-element unpack+staging; u4 (nibble) enables the fast **vpack**
V-staging path → u4/gs16 wins (KV-read mq/mx 0.52, byte-roofline). See `KV_cache_analysis.md`.
**Rule: nibble alignment helps only where unpack is the bottleneck. Weight read isn't; KV read is.**

---

## Reproduce
```bash
cd ~/ms_kernel && source .venv/bin/activate
CUDA_VISIBLE_DEVICES=N PYTHONPATH=. python tests/benchmark.py    # GEMV / GEMM rows
# dequant + fused_skinny sweeps: inline ms_dequant_bf16 / wonly_gemm_fused_skinny over (u,gs)
# (fused_skinny compiles only u3/gs16 & u2/gs8 — the deployed configs).
```
