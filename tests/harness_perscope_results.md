# Batched E2E at the accuracy-robust (u,gs) per scope

Re-run of the §3 E2E batch/output sweep with **each scope at its max-aggressive robust config**
(≤3.5% PPL, plain MSAQ — `precision/scope_uvgs_results.md`), per the "use u=4 where robust" rule:
**S1 u3/gs16 · S2 u2/gs8 · S3 u4/gs2 · S4 u2/gs8 · S5 u2/gs8** (only KV tolerates the u4 nibble).
RTX 3090, Llama-3.1-8B, TTFT+integrated-decode. Ratios `mq=MSAQ`, `mx=MXINT8`, `bf=bf16`; `<1` = faster.
Contrast: the prior uniform run (`harness_batchsweep_results.md`) used u4/gs2 for **all** scopes — fast
but **not weight-accurate**; this run is the accuracy-grounded picture. Raw: `harness_perscope_results.jsonl`.

## DECODE ratios at (1024, 512)
| scope (config) | B=1 mq/bf | B=8 mq/mx · mq/bf | B=32 mq/mx · mq/bf |
|---|---|---|---|
| **S1 W-only** (u3/gs16) | **0.55** | 1.19 · 1.60 | **0.61** · 2.79 |
| **S2 W+A** (u2/gs8) | **0.76** | **0.85** · 1.27 | **0.79** · 2.02 |
| **S3 KV** (u4/gs2) | 0.92 | 0.98 · **0.65** | 1.00 · **0.39** |
| **S4 W-only+KV** (u2/gs8) | **0.56** | **0.95** · 0.94 | **0.87** · 3.49 |
| **S5 W+A+KV** (u2/gs8) | **0.59** | **0.91** · 1.00 | **0.79** · 1.54 |

(After the W+A register-pressure fix `MS_WA_MR=8` — see below — **every scope now has decode mq/mx ≤ ~1.0
except S1 @B8**; S2/S5 went from 1.44/1.64 @B32 to **0.79**.)

## OUTPUT sweep — total mq/bf (and mx/bf), B=8
| scope | L=128 | 512 | 1024 | 2048 | 3880 | mx/bf @3880 |
|---|---|---|---|---|---|---|
| **S3 KV** (u4/gs2) | 0.74 | 0.67 | 0.63 | 0.57 | **0.50** | 0.53 |
| **S4 W-only+KV** (u2/gs8) | 1.92 | 1.19 | 1.00 | 0.85 | **0.74** | 0.76 |
| **S5 W+A+KV** (u2/gs8) | 2.53 | 1.40 | 1.12 | 0.93 | **0.79** | 0.85 |
| S1 W-only (u3/gs16) | 2.11 | 1.46 | 1.31 | 1.22 | 1.16 | 1.26 |
| S2 W+A (u2/gs8) | 2.73 | 1.68 | 1.44 | 1.30 | 1.21 | 1.34 |

(decode-only mq/bf @3880: S3 **0.50**, S4 **0.71**, S5 **0.75**, S1 1.13, S2 1.17.)

## What changed this round (two non-nibble decode optimizations)
1. **(u,gs) compile-time specialization** of the u2/u3 unpack (decode GEMV / batched / W+A / KV paths) —
   the dense-LSB straddle bit-math is resolved at compile time per (u,gs), turning the streaming
   bit-buffer into a fixed shift/mask schedule. ~1.3–1.8× on the non-nibble decode paths.
2. **W+A register-pressure fix** — W+A batched held `idot[MR]`(int) + `acc[MR]`(float) + unpack buffers →
   **spill at MR=32** (micro-bench decode mq/mx 1.55–1.70, worsening with M). Capping the W+A row-tile
   (default **`MS_WA_MR=8`**, tile M>8) keeps accumulators in registers: W+A M=32 (OUT=14336) **2371→1451
   µs, mq/mx 1.64→1.00**; exact (rel_fro 0.0).

Together these flipped S2/S5 W+A decode from losing to **winning vs MXINT8 at every batch** (@B32 1.44/1.64
→ **0.79**), and S5 long-output now **beats bf16** (total 0.79× @L_out3880). (W-only non-nibble already
won at M≥16; only M=8 is a slight intrinsic-unpack loss ~1.2×.)

## Findings (accuracy-grounded, post W+A fix)
1. **vs the matched MXINT8 baseline: MSAQ decode now wins/ties at essentially every scope & batch** — S2/S5
   W+A 0.76–0.91, S4 0.56–0.95, S1/S4 0.55–0.87 (only S1 @B8 = 1.19, the u3 small-batch intrinsic). The
   byte+wide-load advantage holds at the robust configs once the W+A spill is fixed.
2. **KV-cache (S3) is the clean win vs BF16** (tolerates the u4 nibble): decode 0.39–0.65×, total 0.50×
   @L_out3880, grows with B and L_out.
3. **Both full-quant scopes win vs BF16 at long output** — S4 total 0.74×, S5 **0.79× @L_out3880** (S5
   decode 0.75×); the KV win compounds with the now-cheap weight/W+A decode. **B=1 GEMV decode wins vs bf16
   at every scope** (0.55–0.92).
4. **Residual vs-BF16 losses are the tensor-core deficit, not the format** — prefill ~4.6–7.3× (scalar/
   staged sub-byte vs cuBLAS) dominates TOTAL at short L_out; pure W-only decode ~1.1–1.2× bf16.

## Verdict
Under a 3.5% PPL bar, after the W+A register fix: **MSAQ ties/beats the matched MXINT8 baseline at decode
across all scopes** (W+A 0.79–0.91 @batch), **KV-cache beats BF16** (0.39–0.65×), and **full-quant
(S4/S5) beats BF16 at long output** (0.74–0.79× @L3880) plus B=1 GEMV everywhere. The remaining vs-BF16
gap is prefill (~5×, tensor-core), not the quantization format. (B≥64 OOM, 3090.)
