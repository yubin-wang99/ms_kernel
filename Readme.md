# ms/ — MSAQ-signed inference kernels (ColTrain)

CUDA C++ / CUTLASS kernels for the three MSAQ-signed quantization scopes
(weight-only, weight+activation, KV-cache), with a NumPy oracle for
verification. This tree is the CUDA pivot of the old single-file
`mantissa_sharing_kernel.py` (Triton) — Triton couldn't map bit-extraction to
hardware instructions, so the kernels move to CUDA (`bfe`/`prmt`, `__hfma2`,
warp reductions) and CUTLASS (tensor cores).

## Results — all four scopes now beat the matched MXINT8 baseline at every `u`

The campaign goal — *fewer bytes (MSAQ-signed) → faster wall-clock* than a
matched 1-byte MXINT8 kernel — is met for **all four scopes (W-only GEMV, KV
decode, W-only GEMM, W+A GEMM) at every `u`** (and the decode kernels beat
cuBLAS/SDPA BF16 too). The wins are bit-exact (no repack, no padding) and came
from finding the *real* bottleneck by measurement, not by the layout fix the
design notes assumed.

> **결과 총정리 (세 문서):**
> - [`weight_scope_results.md`](weight_scope_results.md) — weight matmul scope(W/GEMV/GEMM) **win**
>   (GEMV u4 0.63, W+A GEMM u4 0.79 …)과 "왜 이기는가".
> - [`kv_read_attempts.md`](kv_read_attempts.md) — KV-read는 Phase 32–40 "tie/불가능"이었으나
>   **Phase 41–44에서 single-token decode WIN으로 정정**(아래 KV decode 표).
> - [`tests/kv_pack_results.md`](tests/kv_pack_results.md) — 그 win 스택(nibble u4/gs2 + sepsc + vpack)의
>   레버별 측정·ncu 진단·정직한 negative까지 전부.

**KV decode** (Phase 41–44) — `D=128`, the autoregressive single-token decode path (the real TPOT
driver). The Phase-18 "압승" was an MXINT8 under-optimization artifact and Phase 32–40 then closed it
as a fair **tie**; Phase 41–44 reopened it by **measuring the real bottleneck with ncu** — it is
**L1/TEX shared traffic + occupancy, not bandwidth** (DRAM only ~10–20%). The fair-and-accurate win
came from a 3-lever stack on the robust **nibble `u4/gs2`** config (the most packing-friendly config
that stays within 3% wikitext PPL):

- **sepsc** — separated-scale K dot: `Σ q·(up·2^u+sh)·s = s·(2^u·Σ q·up + Σ_g sh·qg)`, query
  group-sums `qg` precomputed once.
- **vpack** — packed-transposed nibble V staging: stage V as packed sub-byte (not reconstructed int8),
  decode in Pass-2 registers; smaller smem (13 vs 16.5 KB) → higher occupancy.

One wide kernel handles both **MHA** (`Hq=Hkv`) and **GQA** (`Hq>Hkv`, e.g. Llama-3.1-8B's 32/8) via
the `Hq/Hkv` branch — both win. Bit-exact (no repack), `test_kv` 72/72.

| config (u4/gs2) | MSAQ / MXINT8 |
|---|---|
| MHA  H8  Lk4096  | **0.95–0.97×** |
| MHA  H8  Lk16384 | **0.82×** |
| MHA  H16 Lk16384 | **0.87–0.90×** |
| GQA 32/8 Lk4096  | **0.88–0.90×** |
| GQA 32/8 Lk16384 | **0.93–0.94×** |
| GQA 32/8 batched | **B≤8 win 0.84–0.93×** · B≥16 near-tie/slight-loss 1.00–1.04× |

> Honest boundaries (`kv_read_attempts.md`, `tests/kv_pack_results.md`): high-batch (B≥16) is
> compute+shared-saturated where MSAQ's irreducible bfe-decode is a pure tax (DRAM ~10% → no headroom),
> so it is a near-tie. The **batched/tensor-core prefill regime stays a 3090 hardware wall** — no native
> sub-byte MMA, so both formats build the same MMA input tile and the byte advantage dies at the MMA;
> it is also not the decode path. design-A KV-reuse kernel is a gated documented-negative (occupancy-bound).

**Online K-rotation** (Hadamard H₁₂₈, post-RoPE, Q mirrored) — *expands the robust frontier toward
the fastest KV config at ≈ free latency.* A full head-dim Hadamard rotation of the KV-**Key** kills
the persistent channel outliers (QuaRot mechanism), making the **nibble `u4/gs2`** config — exactly
the one the vpack KV-decode kernel wins with — robust: wikitext PPL **+5.14% (FAIL) → +1.89%** (`u4/mg8`
KV; `precision/rot_results.md`). It is a pair op so accuracy is preserved: rotate K before quant+append
and mirror Q before QKᵀ — `(Q·H)(K·H)ᵀ = Q·Kᵀ`. V stays un-rotated (accuracy-irrelevant).

*Accuracy verified end-to-end on the **actual online path** the kernel runs* (orthonormal `Hₙ=H/√D`,
`Q@Hₙ` in the attn prologue + `msaq(K@Hₙ)` in append — not the offline dequant-fold proxy, and not
bit-identical to it since `1/√128` isn't a power of two). wikitext PPL (Llama-3.1-8B) reproduces the
fold's win — in fact marginally **better** everywhere (online ≤ fold by 0.01–0.20 pp): K `u4/mg8`
+4.63%→**+1.76%**, KV `u4/mg8` +5.14%→**+1.89%**, KV `u3/mg8` +1.25%→**+0.43%**. So `MS_KV_QROT` /
`kv_append_rot` deliver the win for real (`precision/rot_qrot_ppl.py`).

The cost lands on the latency-bound decode hot path, so it was measured (`precision/rot_kv_latency.md`):
- **Standalone** (`torch.ops.msaq.hadamard_rotate`, FWHT — 7 butterfly stages, not a 128² matmul):
  ~9 µs/launch, **flat** across (B, Lk) — Q+K in one launch (8.9 µs) == K alone == ½ of two launches,
  i.e. the cost is **pure launch overhead**, not the math.
- **Fused** (rides launches the decode already pays): K-rotation into `kv_append`
  (`kv_append_rot`, byte-exact, rel_fro 0.0) → marginal **−0.03 µs (noise)**; Q-rotation into the
  attn prologue (`MS_KV_QROT=1`) → marginal **≤ ~1 µs / ≤ ~1%**, at/below attn run-to-run jitter.

So the u4-KV robustness gain is bought for **essentially zero latency** when fused. `test_kv` 72/72,
default path unchanged. Benches `tests/rot_{kv,fused}_bench.py`.

**W-only GEMV** (Phase 14/16/20) — decode `M=1`, `OUT=K=4096`. u4 uses a single
int4 load + nibble `bfe`; u2/u3 were *extraction*-bound (the byte-straddle unpack,
not the load — a perfectly-coalesced plane-split gave zero speedup), fixed by a
**streaming bit-buffer unpack** (one shift+mask per code, shared advanced per
group). No repack — only the kernel's inner loop changed.

| `u` | MSAQ / MXINT8 | MSAQ / cuBLAS |
|---|---|---|
| u2 | **0.84×** (was 1.50) | 0.88× |
| u3 | **0.82×** (was 1.43) | 0.86× |
| u4 | **0.56×** | 0.59× |

RTX 3090 (sm_86), warm cross-measured (clock-/order-verified —
`tests/kv_clock_verify.py`); bit-exact vs the certified path; all GPU +
emulation tests pass. Full settings (sizes, regimes, timing procedure) in
[`methodology.md`](methodology.md).

**Prefill GEMM** (M=512). **W-only**: a software-pipelined BF16 WMMA (opt-in
`MS_TILE_CFG=11`) overlaps the next tile's streaming-unpack with the current
tile's tensor-core MMA — crosses at every `u` (u2 0.98×, u3 0.98×, u4 0.93×),
~1.6× faster than the FP32 tile. **W+A** crosses too, via a **2-stage** design
(Phase 26–27): a Stage-0 pre-pass quantizes the activation to the MSAQ-s
mantissa-sharing format (each element once; the MXINT8 baseline keeps plain
MXINT8 — a format difference, not an optimization), so the Stage-1 INT8-IMMA
GEMM's only heavy prologue work is the weight unpack — which double-buffering
hides behind the MMA. Result: u2 **0.85×**, u3 **0.84×**, u4 **0.71×**; the
weight unpack is 100% hidden (the B-stage cost is purely byte-proportional, so
MSAQ's fewer bytes win) and the pre-pass is ~1% of total. See `change.md`
Phase 22–27.
Full phase-by-phase history (split-K → coalescing → two-pass → cp.async →
wide-load → streaming unpack) and the 4-kernel scoreboard are in `change.md`.

> **weight matmul scope(W/GEMV/GEMM) 결과 총정리:** [`weight_scope_results.md`](weight_scope_results.md)
> — 설계·lever·수치(GEMV u4 0.63, W+A GEMM u4 0.79 …)와 "왜 이기는가". KV-read decode win은
> [`kv_read_attempts.md`](kv_read_attempts.md) + [`tests/kv_pack_results.md`](tests/kv_pack_results.md).

**Batched E2E inference (Phase 47, `harness_batchsweep`)** — full 32-layer Llama-3.1-8B forward,
**TTFT (prefill) + integrated decode = total inference time** (not isolated kernels), over the
`kernel_ver2.md` §3 serving sweep: batch `B∈{1,8,32}` at `(L_in,L_out)=(1024,512)` + output sweep
`L_out∈{128..3880}` at `B=8`, for 5 scopes × {bf16, MXINT8, MSAQ u4/gs2}. Ratios `mq=MSAQ`, `mx=MXINT8`,
`bf=bf16` (`<1` = faster). At decode `B>1` the per-token weight matmul becomes GEMM(M=B); a
**batched-decode GEMV** (`wonly_gemv_batched`/`wa_gemv_batched`, weight column read once and amortized
over the B rows in registers) replaces it for both W-only and W+A.

| decode ratio | S1 W-only | S4 W-only+KV | S3 KV-only | S5 W+A+KV |
|---|---|---|---|---|
| **mq/mx** (vs MXINT8, B=32) | **0.44** | **0.35** | 1.00 | 0.97 |
| **mq/bf** (vs bf16, B=8) | 1.32 | 0.95 | **0.65** | **0.97** |
| **mq/bf total, L_out=3880** (B=8) | 1.24 | **0.73** | **0.51** | **0.74** |

- **vs the matched MXINT8 baseline: MSAQ wins/ties at every decode scope** — W-only decode 0.35–0.44× @B32
  (packed-column wide `uint4` load vs MXINT8's 32 scalar int8 loads), W+A 0.87–0.98×.
- **vs bf16: KV-cache wins at batch** (0.39–0.65×, growing with B and `L_out`), and **both full-quant
  scopes (S4, S5) win at long output** (total 0.73–0.74× @`L_out`3880; S5 W+A+KV decode beats bf16 from B=8).
- **Honest walls (not the format):** pure W-only decode stays ~1.2× bf16 and prefill ~5× — the tensor-core
  deficit of scalar/staged sub-byte kernels (a dedicated tensor-core decode GEMV hits the unpack→bf16
  staging wall; `change.md` Phase 47, documented-negative). `B≥64` OOM on the 3090 (24 GB, 32-layer KV).
  Full ratio tables (prefill/decode/total × mq/mx·mq/bf·mx/bf) in [`tests/harness_batchsweep_results.md`](tests/harness_batchsweep_results.md).

**Accuracy-grounded E2E (each scope at its robust `(u,gs)`).** The sweep above fixes `u4/gs2` everywhere
(kernel-optimal, but `u4` is **not weight-accurate**). Re-running each scope at its **max-aggressive config
within 3.5% PPL** (`precision/scope_uvgs_results.md`: S1 `u3/gs16`, S2/S4/S5 `u2/gs8`, **S3 `u4/gs2`** —
only KV tolerates the u4 nibble) gives the honest fast-**and**-accurate picture:

| scope (robust cfg) | decode mq/bf B=1 | B=8 mq/mx·mq/bf | B=32 mq/mx·mq/bf | total mq/bf @L3880 |
|---|---|---|---|---|
| **S3 KV** (u4/gs2) | 0.94 | 0.98 · **0.65** | 1.00 · **0.39** | **0.51** |
| S1 W-only (u3/gs16) | **0.84** | 1.19 · 1.60 | 0.60 · 2.77 | 1.41 |
| S2 W+A (u2/gs8) | **0.84** | 1.08 · 1.58 | 1.44 · 3.70 | 1.41 |
| S4 W-only+KV (u2/gs8) | **0.84** | 1.39 · 1.37 | 0.58 · 2.31 | 1.07 |
| S5 W+A+KV (u2/gs8) | **0.81** | 1.20 · 1.33 | 1.64 · 3.22 | 1.07 |

- **Under the 3.5% PPL bar, the clean fast-and-accurate win is KV-cache (S3)** — it tolerates the u4
  nibble so its kernel-optimal config is accuracy-valid: decode 0.39–0.65× bf16 (grows with B and L_out),
  tie vs MXINT8. Plus **B=1 GEMV decode wins vs bf16 at every scope** (0.81–0.94×).
- **Weight / W+A are accuracy-pinned to `u2/u3` (non-nibble)** — the byte saving shrinks (u2 ≈ 0.79× MXINT8
  bytes) and the streaming sub-byte unpack is heavier than direct int8, so decode `mq/mx` often **> 1** at
  B>1 (win only at B=32). So the earlier uniform-`u4/gs2` W-only win (mq/mx 0.35–0.44 @B32) was at a
  config that is **not** weight-accurate; only KV's win is simultaneously fast and accurate.
  Full tables: [`tests/harness_perscope_results.md`](tests/harness_perscope_results.md).

## `u` / `gs` per scope (our setup)

MSAQ is parameterized by `u` (per-element *unshared* upper bits — fewer `u` ⇒ fewer bytes ⇒ more
aggressive) and `gs` (= `mg`, the *shared*-code group size — smaller `gs` ⇒ finer shared scale ⇒ more
accurate, more bytes). We set them on **two axes**:

**Latency / E2E (timing is value-independent → one config compared apples-to-apples):**
| run | scope coverage | `u` / `gs` |
|---|---|---|
| **E2E batched** (`harness_batchsweep`) | all 5 scopes (S1–S5) | **`u4/gs2`** (the packing-friendly nibble the vpack KV-decode kernel wins with) |
| E2E batch-1 (`harness.py`) | all scopes | sweep `u∈{2,3,4}` × `gs∈{2,8,32}` |
| kernel microbench (`tests/benchmark.py`) | per kernel (sweepable defaults) | GEMV `u3/gs8`, W-only GEMM `u3/gs8`, W+A GEMM `u2/gs8`, KV `u3/gs8` |

**Accuracy-robust `u`/`gs` per scope** (within 3% wikitext PPL, block=32 — the design target):
| scope | plain MSAQ (E8M0) | with the unlocking lever |
|---|---|---|
| **KV-cache** | **`u4/gs2`** ✓ (`u3/gs8` ✓; `u4/gs8` fails +5.1%) | `u4/gs8` robust via online K-rotation (+1.9%) **or** MX two-level `d2=1` (all `gs`) |
| **weight** | `u3/gs8` ✓ (`u4` fails) | `u4` robust via MX two-level `d2=2` at `gs≤4` |
| **weight+act** | `u2/gs8` (hardest scope) | `u4/gs2` robust only via two-level `d2=2` + rotation |
| **weight+KV** | `u3/gs8` (bounded by weight) | follows weight/KV levers |

So the **latency headline uses `u4/gs2` everywhere**; **accuracy then says which scope can actually run
`u4`** — KV yes (rotation/two-level), weight/W+A only with the extra levers. Per-scope numbers + the
levers: [`precision/u4_robustness_study.md`](precision/u4_robustness_study.md),
[`precision/rot_results.md`](precision/rot_results.md).

## Where to look (results + design)

| topic | file |
|---|---|
| **E2E batched latency** (uniform u4/gs2, prefill/decode/total × ratios) | [`tests/harness_batchsweep_results.md`](tests/harness_batchsweep_results.md), harness `tests/harness_batchsweep.py` |
| **E2E at accuracy-robust (u,gs) per scope** (`--perscope`) | [`tests/harness_perscope_results.md`](tests/harness_perscope_results.md) |
| packing/unpacking format (planes, u4-nibble vs u<4-straddle, unpack ops) | [`packing_explained.md`](packing_explained.md) |
| E2E batch-1 (TTFT/TPOT, 4 scenarios × 3 models) | [`harness_results.md`](harness_results.md), `tests/harness.py` |
| weight-matmul kernel wins (GEMV/GEMM/W+A) | [`weight_scope_results.md`](weight_scope_results.md) |
| KV-read decode win (nibble u4/gs2 + sepsc + vpack) | [`kv_read_attempts.md`](kv_read_attempts.md), [`tests/kv_pack_results.md`](tests/kv_pack_results.md) |
| **u=4 accuracy robustness per scope** (block/scale/rotation/MX two-level) | [`precision/u4_robustness_study.md`](precision/u4_robustness_study.md) |
| Hadamard K-rotation (accuracy + ≈free online cost) | [`precision/rot_results.md`](precision/rot_results.md), [`precision/rot_kv_latency.md`](precision/rot_kv_latency.md) |
| **phase-by-phase design history** (every kernel decision incl. Phase 47 batched-decode) | [`change.md`](change.md) |
| serving workload spec (B, L_in/L_out axes) | [`kernel_ver2.md`](kernel_ver2.md) §3 |
| fairness audit (what differs from MXINT8 besides element handling) | [`for_fair_comparison.md`](for_fair_comparison.md) |
| kernels / numerics / op registration | `csrc/*.cu`, `ms_lib/pack.py`, `csrc/pybind.cpp` |

## Layout

```
ms/
├── csrc/                     C++ / CUDA / CUTLASS backend
│   ├── core/ms_utils.cuh     shared MSAQ-s unpack primitive (+ profiling macros)
│   ├── w_gemv.cu             [pure CUDA] W-only decode GEMV
│   ├── wa_gemm.cu            W+A GEMM + W-only prefill GEMM
│   ├── kv_attention.cu       [pure CUDA] KV-cache flash-decode attention (+ fused K-rotation append)
│   ├── rotate.cu             [pure CUDA] online Hadamard K/Q-rotation (standalone, FWHT)
│   └── pybind.cpp            registers torch.ops.msaq.*
├── ms_lib/                   Python frontend
│   ├── pack.py               numerics + offline packing (NumPy ground truth)
│   ├── reference.py          matmul/attention ORACLE (kernel verification target)
│   └── ops.py                torch wrappers over the compiled kernels
├── tests/                    pytest gates (per scope) + benchmark.py
└── setup.py                  build (torch.utils.cpp_extension)
```

## Old → new (from `mantissa_sharing_kernel.py`)

| old section | new home |
|---|---|
| §1 numerics (`decompose`, `reconstruct`, `_e8m0_scale`, `per_*_bits`) | `ms_lib/pack.py` |
| §2 packing (`pack_weight`, `unpack_weight`, `dequant_weight`, `weight_int8`, `pack_kv`) | `ms_lib/pack.py` |
| §3 mirrors (`wonly_matmul`, `quant_act`, `wa_matmul`, `kv_attention`) + §6 `_msaq_signed_ref` | `ms_lib/reference.py` |
| §4 Triton kernels | replaced by `csrc/*.cu` |
| §5 wrappers (`run_*`) | `ms_lib/ops.py` |
| §6 `selftest_numpy` / `selftest_triton` | `tests/test_{w,wa,kv}.py` + `tests/benchmark.py` |

## Two layers

* **Verified (NumPy).** `pack.py` + `reference.py` are migrated verbatim from
  the certified file and reproduce `ALL CHECKS PASSED`. They are the oracle.
  Runs anywhere (no GPU): `python -m pytest` → NumPy gates pass, kernel gates
  skip.
* **GPU-unvalidated (CUDA).** The `.cu` kernels reuse the certified
  `ms::unpack_ms_weight_elem`, so they are correct *by construction* against the
  oracle; only their CUDA execution is unproven until certified on the 3090.

### `.cu` completeness (honest status)

> **Note:** the table below is the *original correctness-first baseline* status.
> The kernels have since been optimized and GPU-certified — see **Results** above
> and `change.md` (Phase 1–27). All four scopes (`w_gemv.cu`, `kv_attention.cu`,
> W-only GEMM and W+A GEMM in `wa_gemm.cu`) now beat the matched MXINT8 baseline
> at every `u`.

All four kernels' **logic is verified on CPU** by `tests/test_emulation.py`
(`rel_fro < 1e-9` vs the oracle, mirroring each kernel's exact byte-addressing
and arithmetic). What remains is CUDA *execution* — compile + the GPU
`*_vs_oracle` gates on the 3090.

| file | status |
|---|---|
| `ms_utils.cuh` | unpack primitive **complete & matches the oracle's bit-math** |
| `w_gemv.cu` | **optimized & GPU-certified** — split-K + wide-load + streaming bit-buffer unpack; beats matched MXINT8 at all `u` (Phase 14/16/20) |
| `wonly_gemm` / `wa_gemm` (`wa_gemm.cu`) | **optimized & GPU-certified** — W-only pipelined BF16 WMMA (cfg=11); W+A 2-stage (MSAQ-s pre-quant + pipelined INT8 IMMA). Both beat matched MXINT8 at all `u` (Phase 23/26/27) |
| `kv_attention.cu` | **optimized & GPU-certified** — split-K flash-decode + key-per-thread wide-read; single-token decode beats matched MXINT8 (MHA·GQA) via nibble u4/gs2 + sepsc + vpack (Phase 41–44). No tensor-core QKᵀ/PV (3090 hardware wall, not the decode path) |

All `.cu` are **UNVALIDATED-ON-GPU** until the GPU gates run. None is the
optimized path: `w_gemv` still needs split-K + `__shfl_down` + `__hfma2`;
`wa_gemm` needs the CUTLASS INT8-IMMA / BF16 tensor-core path; `kv_attention`
needs tiled tensor-core Q·Kᵀ / P·V. They exist to lock the addressing + wiring
and be certified, then optimized.

## Build · test · certify (on the RTX 3090, sm_86)

```bash
python -m pytest                          # oracle + CPU logic gates (no GPU needed) + GPU gates
python -m pytest tests/test_emulation.py  # CPU logic gate only — kernel arithmetic vs oracle
python setup.py build_ext --inplace       # builds ms_cuda*.so (torch.ops.msaq.*)
python -m pytest tests/test_w.py -q       # target one scope's GPU gate after editing it
python tests/benchmark.py                 # cuda.Event ms latency vs cuBLAS/SDPA
```

Two correctness layers: `test_emulation.py` proves each kernel's logic on CPU
(runs anywhere); the per-scope `tests/test_*.py::*_vs_oracle` gates certify CUDA
execution on the 3090 (`rel_fro < 2e-2` vs the oracle, fed the same bf16-rounded
inputs). Build → target the one GPU gate for the kernel you touched → fix →
repeat. CUTLASS work additionally needs `CUTLASS_DIR=/path/to/cutlass` at build
time (the current baselines build without it).

## Open design fork — packing layout

`ms_utils.cuh` decodes the **current** dense LSB-first, byte-strided pack
(`pack.py`), which is correct but **cannot** use single-op `bfe.s32` /
128-bit vectorized loads — a code's bytes aren't contiguous in a register. The
doc's `bfe` + `cp.async` + `ldmatrix` plan needs a **register-aligned,
XOR-swizzled** repack. When adopting it: (1) re-lay-out `pack.py`, (2)
**re-certify** the new layout through the `test_*::*roundtrip*` /
`*_vs_oracle*` gates, (3) flip `MSAQ_USE_BFE` and fill the `bfe` path. Do not
let the kernels assume the new layout before the roundtrip gate is green.

## Profiling

1. `tests/benchmark.py` — `torch.cuda.Event` ms latency (the headline number).
2. `ncu` — warp-stall / occupancy attribution, no source edits (see the header
   comment in `benchmark.py` for metric sets).
3. `clock64()` spans in `ms_utils.cuh` under `-DENABLE_PROFILING` — last resort
   for a single bit-op span; it perturbs scheduling, so never report it as the
   latency.