# Kernel inventory & latency vs BF16 / MXINT8 — 2026-06-24

Read from the current tree. **RTX 3090, Llama-3.1-8B**, accuracy-robust per-scope configs
(**S1 `u3/gs16` · S2/S4/S5 `u2/gs8` · S3 `u4/gs2`**; only KV tolerates the u4 nibble). Ratio `< 1` = faster.
Sources: `csrc/*.cu`, `change.md`, `tests/harness_perscope_results.md`, `tests/harness_batchsweep_results.md`,
`precision/aa_attn_results.md`, `tests/aa_kernel_bench.py`, `precision/aa_u4_ppl.py`.

Two-axis comparison: **mq/mx** = MSAQ vs the *matched* MXINT8 baseline (format axis) · **mq/bf** = MSAQ vs BF16
(cuBLAS GEMM / flash SDPA).

---

## Prefill stage (compute-bound, tensor-core)

| Kernel kind (csrc fn) | Optimization level | vs MXINT8 | vs BF16 (cuBLAS/flash) |
|---|---|---|---|
| **W-only GEMM** (`wonly_gemm_wmma_pipe`, `wonly_gemm_tiled[_cm]`) | High: shared-mem tiling (1 unpack/tile) + **BF16 WMMA software pipeline** (`MS_TILE_CFG=11`, next-tile unpack ∥ MMA) + divide→shift | **0.93–1.00 WIN** (u4 0.93, u2/u3 ~0.98–1.00) | **~5× LOSE** |
| **W+A GEMM** (`wa_imma`, `wa_gemm_tiled`) | High: **2-stage INT8 IMMA** — activation pre-quantized to MSAQ-s (`quant_act_msaq`, memory-bound prologue) + weight unpack double-buffered behind int8×int8 MMA | **0.79 WIN** | **~6–7.6× LOSE** |
| **Prefill self-attention AA** (`qk_wmma` + `pv_wmma`) | Medium WMMA, but **structurally negative**: quantizing softmax `P` is O(L²) and `[H,L,L]` scores must be materialized (flash avoids both) | n/a | **1.4–2.7× LOSE (documented-negative)** |
| (baseline) MXINT8 GEMM / attn | `mxint8_gemm_wmma_pipe`, `mxint8_wa_imma`, `qk_wmma_mx`, `pv_wmma_mx` | — | ~5–8× |

**Prefill verdict:** beats matched MXINT8 (W+A 0.79, W-only ~1.0), but **every sub-byte GEMM loses to cuBLAS
BF16 (~5×)** — the tensor-core staging wall (sub-byte operand must be unpacked to a bf16/int8 tile, cancelling
the DRAM byte saving). AA prefill loses to bf16 outright.

---

## Decode stage (memory-bound, GEMV + KV attention)

> **Batched-decode dispatch.** The W-only and W+A decode rows below ARE the batched-decode kernels. The
> harness dispatches on batch B: **B=1 → wide/split-K GEMV** (`wonly_gemv_wide/_splitk`, `wa_gemv_wide`);
> **B>1 → batched GEMV** (`wonly_gemv_batched[_uspec]`, `wa_gemv_batched[_uspec]`) — weight column read once and
> amortized over B activation rows held in `acc[MR]` registers. MXINT8 counterparts: `mxint8_gemv_batched`,
> `mxint8_wa_gemv_batched`. The B=32 numbers (W-only 0.44/0.35, W+A 0.79) come from these batched paths.

| Kernel kind (csrc fn) | Optimization level | vs MXINT8 | vs BF16 |
|---|---|---|---|
| **W-only GEMV** (B=1 `wonly_gemv_wide/_splitk`; B>1 `wonly_gemv_batched[_uspec]`) | High: **batched-decode GEMV** (MR registers) + wide `uint4` load + split-K + streaming/uspec unpack + divide→shift | **0.35–0.84 WIN** (B32 0.44/0.35; robust u3 ~0.82) | 0.55–0.76 @B1 WIN / ~1.1–1.3× @batch·short LOSE |
| **W+A GEMV** (B=1 `wa_gemv_wide[_uspec]`; B>1 `wa_gemv_batched[_uspec]`) | High: batched int-dot GEMV + **`MS_WA_MR=8` register-spill cap** (mq/mx 1.64→1.00) + compile-time u2/u3 specialization | **0.79–0.98 WIN** | 0.65–0.71 @B1 WIN; beats bf16 at long output (decode 0.70 @L_out=3880) |
| **KV-cache decode attention** (`kv_decode_wide/_gqa/_split/_warpT/_cpasync` + `kv_decode_combine`; append `kv_append[_rot]`) | Highest: split-KV + token-major coalescing + two-pass barrier-light + **key-per-thread wide `uint4` read** + **vpack transposed-V staging** + compile-time VPACK template (reg 163→127) | 0.47–1.01 (isolated 0.47–0.79 WIN; E2E ~0.98–1.01 tie) | **0.13–0.94 WIN** (batch 0.39–0.65; long-output 0.50) |
| **Tensor-core decode GEMV** (`wonly_gemv_tc`) | documented-negative — no native sub-byte MMA on RTX 3090; staging wall | — | 3.5–8.8× LOSE (not pursued) |
| (baseline) MXINT8 decode | `mxint8_gemv_batched`, `mxint8_wa_gemv_batched`, `mxint8_kv_*` | — | — |

**Decode verdict:** beats matched MXINT8 at essentially every scope & batch. **KV-cache (S3) is the clean win
vs BF16** (tolerates the u4 nibble: 0.39–0.65× at batch, 0.50× at long output). W-only / W+A beat bf16 at B=1
and at long output; at short output cuBLAS's prefill-GEMM advantage makes them ~1.1–1.3×.

---

## One-line takeaway

The latency win lives entirely in **memory-bound decode** (above all KV-cache). **Compute-bound prefill GEMM
beats MXINT8 but loses to cuBLAS BF16** because of the tensor-core staging wall, and **prefill AA is a
structural documented-negative** (O(L²) softmax-P quant). The MSAQ-over-MXINT8 edge needs the u4 nibble, which
only KV-cache accuracy tolerates — for W/W+A/AA accuracy needs u2, so those win on bytes only, not on the
nibble fast path.
