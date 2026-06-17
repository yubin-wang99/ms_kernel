# Measurement methodology & configuration

What every latency number in `change.md` / `Readme.md` is measured under. Only
factors that actually move the numbers are listed.

---

## 1. Hardware / software environment

| | |
|---|---|
| GPU | NVIDIA GeForce RTX 3090 (GA102, **Ampere `sm_86`**) |
| SMs | **82** |
| Memory | 24 GB GDDR6X, **peak HBM BW ≈ 936 GB/s** (384-bit, 19.5 Gbps) |
| L2 cache | **6 MB** (matters — see §6 residency caveat) |
| Max SM clock | 2100 MHz; sustained boost under load **≈ 1950 MHz** (see §5 clock) |
| Driver | 535.183.01 |
| CUDA toolkit | **11.8** (`nvcc` V11.8.89) |
| PyTorch | **2.5.1** (CUDA 11.8 build), cuDNN 9.1.0 |
| Build flags | `nvcc -O3 -gencode arch=compute_86,code=sm_86 --use_fast_math -std=c++17` |

**GPU selection.** All numbers are taken on **GPU index 1** (`CUDA_VISIBLE_DEVICES=1`).
GPU 0 on this box is frequently occupied by external jobs (OOM / ~2× slowdown),
which corrupts timing; GPU 1 is kept idle for clean runs.

---

## 2. What is being measured

Three kernels per scope, all timed the same way:

- **MSAQ** — our mantissa-shared sub-byte kernel (the thing under design). Reads
  the packed `upper`/`shared` planes and reconstructs the weight/KV on the fly.
- **MXINT8** — the **matched baseline**: the *same* kernel structure (same tiling /
  split-K / flash-decode), but reads a plain 1-byte INT8 mantissa with no unpack.
  **`MSAQ / MXINT8` is THE headline ratio** — same optimization level, so it
  isolates "fewer bytes (MSAQ wins) vs unpack overhead (MXINT8 wins)". `<1.0` =
  MSAQ faster.
- **BF16 reference** — a tuned vendor library, **not** a matched comparison (it
  uses tensor cores; our kernels are FP32 CUDA-core). Reported for context only:
  - GEMV / GEMM / W+A → **cuBLAS** via `torch`: `x @ W.t()` (BF16 in, BF16 out).
  - KV decode → **SDPA**: `torch.nn.functional.scaled_dot_product_attention`
    (BF16 flash attention).

### What we design (MSAQ-signed format)
Each weight/KV element is `(upper · 2^u + shared) · scale` — a valid MXINT8 word:
- `upper` : `8−u`-bit signed code, **one per element**.
- `shared` : `u`-bit signed code, **shared by a `gs`-element group**.
- `scale` : E8M0 (`int8` exponent), one per 32-element block.
- Swept params: **`u ∈ {2,3,4}`** (`wbits = 8−u`), **`gs`** = power of two (default **8**).
- Block size `BLOCK = 32` (OCP MX), `E_MAX = 6`.

**Bytes per 32-element block** (the source of MSAQ's advantage), `gs=8`:

| | upper (UB) | shared (SB) | + scale | vs MXINT8 33 B | ratio (UB+SB / 32) |
|---|---|---|---|---|---|
| u4 | 16 | 2 | 1 | 18 (+1) | **0.56** |
| u3 | 20 | 2 | 1 | 22 (+1) | **0.69** |
| u2 | 24 | 1 | 1 | 25 (+1) | **0.78** |
| MXINT8 | 32 (int8) | — | 1 | 32 (+1) | 1.00 |

---

## 3. Workload configurations

`OUT` = output features, `K` = input features, `M` = tokens (batch×seq).
Canonical problem size is **`OUT = K = 4096`** (one transformer FFN/attn projection).

### Decode (one new token at a time)
| scope | config | meaning |
|---|---|---|
| **W-only GEMV** | `M = 1`, `OUT = K = 4096` | single-token decode Linear (batch 1, the weight read dominates) |
| **KV decode** | `H = 8` heads, `D = 128`, **`Lk = 4096` and `16384`**, `Lq = 1` | one decode step attending to the full `Lk` cached context; no causal mask (newest token sees all). **Lk=16384 is the reported (pure-HBM) number; Lk=4096 is L2-resident — see §6.** |

- KV element count: per head `Lk·D`, ×`H` heads, ×2 (K and V). At `Lk=16384`,
  `H=8`, `D=128` → `8·16384·128·2 ≈ 33.5 M` quantized elements;
  `nb = D/32 = 4` blocks per token. Footprint scales `H·Lk` (5 MB → 151 MB swept
  to confirm the HBM regime).
- Batch is **1 sequence**; "batch size" for GEMV decode is `M=1`.

### Prefill (many tokens at once)
| scope | config | meaning |
|---|---|---|
| **W-only GEMM** | **`M = 512`**, `OUT = K = 4096` | 512-token prefill Linear |
| **W+A GEMM** | **`M = 512`**, `OUT = K = 4096` | same, with on-the-fly MXINT8 activation quant (INT dot) |

- Weight element count: `OUT·K = 4096·4096 ≈ 16.8 M` = `(K/32)·OUT = 128·4096
  = 524 288` blocks. Prefill `M` is swept `16…1024` in tile-size studies
  (Phase 10), but **`M=512` is the canonical headline**.

---

## 4. Numeric precision flow (affects both correctness and speed)

- Inputs `x`/`X`/`q` are **bf16**; packed planes are `uint8` (`upper`,`shared`,
  `qweight`) + `int8` (`scale_exp`).
- Compute accumulates in **FP32** (CUDA cores; no tensor cores in the matched
  kernels), output cast back to **bf16**.
- W+A path quantizes activations to MXINT8 on the fly and does an INT dot folded
  with the power-of-two `sa·sw` scales (bit-exact to the integer dot).
- BF16 baselines (cuBLAS/SDPA) run their own tensor-core paths in bf16.

---

## 5. Timing procedure

1. **Planes moved to GPU once**, before timing. Copying the packed planes H2D
   *inside* the timed loop is a ~48× artifact; never done.
2. **Warm-up before timing**: both compared kernels are run back-to-back for
   **~2.5–3 s with no idle gap**, so the SM clock reaches and *holds* boost.
   (Idle-then-first-kernel runs at an un-boosted clock and mis-times — the
   correction that retired several earlier "wins" as artifacts.)
3. **Per-call latency** via `torch.cuda.Event` over a fixed iteration count
   (100–300 for fast kernels, 50 for the multi-ms GEMMs): `elapsed / iters`.
4. **`min` of 2 timing blocks** per kernel (drops the occasional scheduler blip).
5. **Cross / order check.** Kernels are timed in steady state; for the headline
   ratios the two kernels are measured in **both orders** (A-first and B-first).
   A real result is order-independent; a clock artifact flips with order. This is
   automated in `tests/kv_clock_verify.py` (KV) and was used to confirm the
   crossovers.

**Clock note.** SM clocks **cannot be locked** here (no permission:
`nvidia-smi -lgc` denied), so absolute µs drift run-to-run by ~10–15% as boost
varies. The defense is (a) warm-up to hold boost, (b) measure the matched pair in
the same warm window, (c) verify order-independence. Absolute numbers are
indicative; the **same-window ratio** is the robust quantity.

---

## 6. Regimes & caveats that change the numbers

- **L2 residency (KV).** The 6 MB L2 holds a *small* KV footprint. At `Lk=4096`
  the MSAQ footprint (~6 MB) fits L2 while the larger MXINT8 footprint (~8.6 MB)
  thrashes, which *flatters* MSAQ — a benchmark artifact, not a bandwidth win.
  **Report `Lk=16384`** (both ≫ L2 → pure HBM); footprint is swept to 151 MB to
  confirm the regime is stable.
- **Matched vs vendor.** `MSAQ/MXINT8` is the claim; `MSAQ/cuBLAS` and
  `MSAQ/SDPA` are ~8–14× and ~0.2–0.3× respectively only because the vendor path
  uses tensor cores (GEMM) or is a different algorithm (SDPA). Don't read them as
  the design target.
- **`u` dependence.** All ratios are reported per `u` (2/3/4); `gs=8` unless
  noted. u4 is the cheapest (nibble-aligned), u2 the most bytes — the crossover
  margin shrinks with smaller `u`.

---

## 7. Correctness gating (every perf change is certified first)

- **CPU logic**: `tests/test_emulation.py` mirrors each kernel's exact byte
  addressing vs the NumPy oracle (`rel_fro < 1e-9`).
- **GPU vs oracle**: `tests/test_{w,wa,kv}.py::*_vs_oracle` on the 3090
  (`rel_fro < 2e-2`, fed the same bf16-rounded inputs).
- **A/B bit-exactness**: a new fast path is diffed against the certified path
  (`max|diff|` typically `0.0`, or `~4e-3` when only bf16 accumulation order
  differs).

---

## 8. Reproduce

```bash
# build
python setup.py build_ext --inplace
# headline 3-way table (default configs in §3)
CUDA_VISIBLE_DEVICES=1 python tests/benchmark.py
# warm cross-measured, clock/order-verified (KV)
CUDA_VISIBLE_DEVICES=1 python tests/kv_clock_verify.py
# bottleneck decomposition (KV memory/unpack/exp split)
CUDA_VISIBLE_DEVICES=1 MS_KV_DIAG=1 python tests/kv_diag.py
```

Notes: `tests/benchmark.py` moves planes to the GPU once and reports
`MSAQ / MXINT8` and `MSAQ / {cuBLAS,SDPA}` per scope. Env toggles for A/B:
`MS_KV_WIDE`, `MS_KV_CPASYNC`, `MS_KV_SPLIT_MULT`, `MS_GEMV_SPLITK_MULT`,
`MS_TILE_CFG`.
