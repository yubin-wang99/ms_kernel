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
| **S1 W-only** (u3/gs16) | **0.84** | 1.19 · 1.60 | **0.60** · 2.77 |
| **S2 W+A** (u2/gs8) | **0.84** | 1.08 · 1.58 | 1.44 · 3.70 |
| **S3 KV** (u4/gs2) | 0.94 | 0.98 · **0.65** | 1.00 · **0.39** |
| **S4 W-only+KV** (u2/gs8) | **0.84** | 1.39 · 1.37 | **0.58** · 2.31 |
| **S5 W+A+KV** (u2/gs8) | **0.81** | 1.20 · 1.33 | 1.64 · 3.22 |

## OUTPUT sweep — total mq/bf (and mx/bf), B=8
| scope | L=128 | 512 | 1024 | 2048 | 3880 | mx/bf @3880 |
|---|---|---|---|---|---|---|
| **S3 KV** (u4/gs2) | 0.77 | 0.68 | 0.63 | 0.57 | **0.51** | 0.53 |
| S4 W-only+KV (u2/gs8) | 2.30 | 1.63 | 1.43 | 1.24 | 1.07 | **0.76** |
| S5 W+A+KV (u2/gs8) | 2.84 | 1.77 | 1.48 | 1.26 | 1.07 | 0.85 |
| S1 W-only (u3/gs16) | 2.43 | 1.84 | 1.67 | 1.53 | 1.41 | 1.25 |
| S2 W+A (u2/gs8) | 3.00 | 1.99 | 1.73 | 1.55 | 1.41 | 1.34 |

## Findings (the honest, accuracy-grounded result)
1. **KV-cache (S3) is the only clean E2E win at robust accuracy.** KV tolerates the **u4 nibble** (u4/gs2
   +2.89% PPL), so its kernel-optimal config IS accuracy-valid: decode **0.39–0.65× bf16** (growing with
   B and L_out), total **0.51× @L_out3880**, tie vs MXINT8. This is the headline that survives the 3.5%
   accuracy bar.
2. **Weight / W+A scopes are accuracy-bound to u2/u3 — and there MSAQ's latency edge largely evaporates
   at decode.** At u2/u3 (non-nibble) the byte advantage shrinks (u2 = 6.5 bits/elem vs MXINT8 8.25, only
   ~0.79×) AND the streaming sub-byte unpack is heavier than MXINT8's direct int8 read, so decode mq/mx
   often **> 1** (S2/S5 1.05–1.64; S1/S4 win only at B=32, 0.58–0.60). vs bf16 these scopes lose at B>1
   (prefill 4.6–7.3×, decode 1.0–3.7×); at S4 long-output MXINT8 even beats bf16 (mx/bf 0.76) while
   MSAQ-u2 does not (mq/bf 1.07) — the non-nibble unpack overhead.
3. **B=1 decode (GEMV) still wins vs bf16 at every scope** (mq/bf 0.81–0.94) — the memory-bound single-
   token path, regardless of config.
4. **Contrast with uniform u4/gs2** (`harness_batchsweep_results.md`): that run showed W-only decode
   mq/mx 0.35–0.44 @B32 — but u4 is **not weight-accurate** (weight u4/gs2 +6.04% PPL). At the
   accuracy-robust weight config (u3/u2) the same scope is 0.58–0.60 (B32) / >1 (B8). So the earlier
   weight-scope latency wins were at a non-robust config; **only KV's win is simultaneously fast and accurate.**

## Verdict
Under a 3.5% PPL bar, MSAQ's E2E latency advantage is **concentrated in the KV-cache scope** (u4-nibble-
robust → 0.39–0.65× bf16 at batch, growing with output length) plus the B=1 GEMV decode. Weight and W+A
are accuracy-pinned to u2/u3, where the smaller byte saving and the non-nibble unpack overhead make MSAQ
tie/lose vs MXINT8 and bf16 at B>1 decode. Closing the weight/W+A gap needs either u4 made weight-robust
(rotation / MX two-level — not in these kernels) or a faster non-nibble decode unpack. (B≥64 OOM, 3090.)
