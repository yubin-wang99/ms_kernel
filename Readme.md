# ms/ — MSAQ-signed inference kernels (ColTrain)

CUDA C++ / CUTLASS kernels for the three MSAQ-signed quantization scopes
(weight-only, weight+activation, KV-cache), with a NumPy oracle for
verification. This tree is the CUDA pivot of the old single-file
`mantissa_sharing_kernel.py` (Triton) — Triton couldn't map bit-extraction to
hardware instructions, so the kernels move to CUDA (`bfe`/`prmt`, `__hfma2`,
warp reductions) and CUTLASS (tensor cores).

## Layout

```
ms/
├── csrc/                     C++ / CUDA / CUTLASS backend
│   ├── core/ms_utils.cuh     shared MSAQ-s unpack primitive (+ profiling macros)
│   ├── w_gemv.cu             [pure CUDA] W-only decode GEMV
│   ├── wa_gemm.cu            W+A GEMM + W-only prefill GEMM
│   ├── kv_attention.cu       [pure CUDA] KV-cache flash-decode attention
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

All four kernels' **logic is verified on CPU** by `tests/test_emulation.py`
(`rel_fro < 1e-9` vs the oracle, mirroring each kernel's exact byte-addressing
and arithmetic). What remains is CUDA *execution* — compile + the GPU
`*_vs_oracle` gates on the 3090.

| file | status |
|---|---|
| `ms_utils.cuh` | unpack primitive **complete & matches the oracle's bit-math** |
| `w_gemv.cu` | correctness-first draft, **logic-verified**; x via L2 broadcast (no shared-mem K ceiling) |
| `wonly_gemm` / `wa_gemm` (`wa_gemm.cu`) | correctness-first baselines, **logic-verified** (no tensor cores yet) |
| `kv_attention.cu` | correctness-first flash-decode, **logic-verified** (no tensor-core QKᵀ/PV yet) |

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