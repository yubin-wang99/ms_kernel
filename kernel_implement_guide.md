# MSAQ-s Kernel Implementation Guide

> Reference for the CUDA / CUTLASS inference-kernel implementation of MSAQ-s
> (Mantissa Sharing-Aware Quantization, signed). Faithful English transcription
> of the 2026-06-07 kernel design notes. Structure and technical detail are
> preserved as written.

## Current Build Environment

- **Environment**
    - hardware: RTX 3090, 24 GB VRAM
    - implementation: PyTorch
        - BF16: cuBLAS, SDPA
        - MXINT8, MS: Triton
    - target:
        - weight-only
        - weight and activation
        - KV cache

## Development Direction

- **Goal**
    - MS inference time << MXINT8 inference time

- **Sub-goal**
    - Latency reduction from reduced memory traffic > unpacking latency
        - This optimization is especially impactful in decode, where memory read determines the total time.

- **Direction: Triton → CUDA C++ and CUTLASS**
    - Rationale:
        - Triton optimizes high-level tensor ops well but cannot directly direct hardware-instruction mapping. So even a hand-written bit-extraction kernel is lowered by the compiler into general-purpose logic ops, which increases the cycle count.
        - Also, when sub-byte extraction logic is implemented in Triton, the compiler may allocate many registers, which lowers the number of active warps and reduces GPU occupancy (parallel-execution efficiency).
        - CUDA can control bit-extraction so it runs in a single cycle, using PTX instructions such as `bit field extract (BFE)` or `byte permute (prmt)`.
    - Implementation:
        - W-only GEMV and KV cache: CUDA C++
        - W+A GEMM and prefill: CUTLASS, to exploit Tensor Cores
            - Rationale:
                - Tensor Core optimization is hard in plain CUDA C++:
                    - requires complex data-layout rearrangement to avoid bank conflicts,
                    - requires pipelining that overlaps the global→shared memory read with compute,
                    - requires logic that distributes data per warp while minimizing register allocation.
                - CUTLASS is a C++ template library that NVIDIA has pre-tuned to maximize Tensor Core performance.
                - Unlike a fixed black-box library (e.g., cuBLAS), CUTLASS is modular in structure, so user-defined code can be inserted in the middle of the compute pipeline.
    - Keep the PyTorch environment:
        - Use `torch.utils.cpp_extension` and Pybind11 to compile the CUDA / CUTLASS C++ kernels so they can be called from the wrapper functions of the Python code.
        - AWQ and GPTQ also use this style of combination.

## CUDA-Based Kernel Pipeline

CUDA is used to control the memory-bound regions where matrix-vector dot products and complex bit operations are mixed.

### 1) W-only GEMV (Decode stage, batch size = 1)

In the decode stage, the speed of reading weights from HBM determines the overall speed of the Linear operation.

- **Step 1: Asynchronous, parallel load (Split-K applied)**
    - Partition the matrix's K dimension (input channels) via Split-K so that multiple thread blocks load, in parallel from global memory into registers, the packed weights (`upper`, `shared`), the `scale`, and the activation vector $X$ (BF16) from HBM.

- **Step 2: In-register unpacking**
    - Use the `bfe.s32` (Bit Field Extract) PTX instruction to extract, per element, the upper bits ($8-u$) and the shared lower bits ($u$) from the bitstream packed inside a 32-bit register.
    - Add the two extracted values to reconstruct the INT8 integer value in the form $c \cdot 2^u + r$.

- **Step 3: In-register dequantization**
    - Cast the INT8 integer to BF16 and multiply by the loaded `scale` to obtain the final BF16 weight.
        - The INT8 bits cannot be placed directly into the BF16 mantissa field.
        - Reasons:
            - INT8 is two's-complement; BF16 is sign-magnitude.
            - To place the value in the mantissa field, you must undo two's complement to obtain the magnitude and extract the sign bit.
            - Implementation overhead (substantially increases register usage):
                - sign-bit extraction (`shr` and `and`)
                - conversion to absolute value when negative (`abs`)
                - locating the highest set bit (`clz`, Count Leading Zeros)
                - shift to align to the mantissa field (`shl` or `shr`)
                - adjusting the exponent (scale exponent) by the shift amount (`add` and `sub`)
                - merging the sign, adjusted exponent, and mantissa into one (`or` and `shl`)

- **Step 4: MAC (Multiply-Accumulate)**
    - Compute the reconstructed weight and the activation vector $X$ elements with the `__hfma2` (vectorized BF16 FMA) instruction and add into the register accumulator.

- **Step 5: Warp reduction and store**
    - When the computation completes, use `__shfl_down_sync()` to gather the partial sums computed by the threads within a warp, and store to the HBM output tensor with an atomic add to combine the partitioned K dimension.

### 2) KV Cache Quantization / Dequantization (Attention stage)

In attention, writing KV data to the cache differs from reading the cache during generation.

- **[Write pipeline — at prefill]**
    - **Step 1:** Perform projection and RoPE (positional encoding) on the K, V tensors in BF16.
    - **Step 2:** Compute `max-abs` over the head dimension (block size 32) of the K, V tensors and run MSAQ-s quantization to derive `upper`, `shared`, and `scale`.
    - **Step 3:** Write the quantized data to the KV-cache region of HBM in packed format.

- **[Read & compute pipeline — at decode (Fused Attention)]**
    - **Step 1 (Load & Dequant):** Read the Q (BF16) tensor from HBM and, at the same time, read the packed K, V data from the KV-cache region. As soon as the K, V data is loaded into registers, dequantize it on the fly to BF16 via the `bfe.s32` instruction and a multiply by `scale`.
    - **Step 2 (Q·K^T):** Compute the dot product between Q and the just-reconstructed K, then compute FlashAttention-style online softmax to obtain the probabilities.
    - **Step 3 (Softmax·V):** Compute the dot product of the probabilities with the on-the-fly-reconstructed V, and store the final output to HBM.

## CUTLASS-Based Kernel Pipeline

CUTLASS is used to maximize Tensor Core efficiency in the matrix-matrix multiply (GEMM) regions, where the compute volume grows sharply as many tokens are processed.

### 1) W+A GEMM

A structure that converts both weight and activation data to INT8 and performs INT8 Tensor Core operations (IMMA).

- **Step 1: Pipelined data load (Global → Shared)**
    - Use `cp.async` to asynchronously load the packed weight tile and the BF16 activation tile from global memory (HBM) into Shared Memory. Tensor Core compute and the memory copy overlap.

- **Step 2: Prologue (Custom Load & On-the-fly Quantization)**
    - **Weight load:** When loading a tile from Shared Memory into registers, inject the unpacking logic via a custom iterator. Immediately reconstruct the data to INT8 via `bfe.s32`. (Dequantization is not performed here; the integer state is kept.)
    - **Activation load:** When loading from Shared Memory into registers, find the max over each 32-element block and quantize the BF16 data in real time to INT8 (MXINT8 format) in the range [-127, 127]. Keep each block's scale in a separate register.

- **Step 3: Mainloop (INT8 Tensor Core Math)**
    - Feed the unpacked weight (INT8) and the quantized activation (INT8) as inputs to the `mma.sync` (INT8 IMMA) Tensor Core instruction, perform tile-wise matmul, and accumulate into INT32 registers.

- **Step 4: Epilogue (Dequantization & Store)**
    - Multiply the INT32 tile accumulation result by the stored W scale block and A scale block to scale-restore (dequantize) to BF16, then write to HBM as the output tensor.

### 2) Prefill (W-only GEMM)

The first stage, where many prompt tokens are input at once: a matrix-matrix multiply (GEMM), but the activation data is not quantized and BF16 Tensor Cores are used.

- **Step 1: Pipelined data load (Global → Shared)**
    - Move the weight (MSAQ-s packed tile) and the activation tile (BF16) composed of many tokens to Shared Memory via the asynchronous memory-copy instruction (`cp.async`).

- **Step 2: Prologue (Weight Dequantization)**
    - When loading the weight data from Shared Memory into registers, perform both unpacking and scaling. That is, combine the upper / lower bits with `bfe.s32`, then multiply by `scale` to fully reconstruct to **BF16 format**.
    - The activation data is already BF16, so only alignment to the Tensor Core input format is performed, with no separate conversion.

- **Step 3: Mainloop (BF16 Tensor Core Math)**
    - Take the BF16-reconstructed weight tile and the BF16 activation tile as inputs and run the `mma.sync` instruction (BF16 FMA). Results are accumulated into FP32 registers to preserve internal precision.

- **Step 4: Epilogue (Store)**
    - Cast the FP32 accumulator tile values to the BF16 data type and store to global memory (HBM) to pass to the next layer.

## Differences Between the MXINT8 and MSAQ-s Kernels

With MXINT8 as the baseline, MSAQ-s requires an intermediate step that, in registers, separates the bitstream into per-element `upper` codes and per-group `shared` codes, then synthesizes them into a single 8-bit integer ($c \cdot 2^u + r$).

### 1. W-only GEMV (Decode stage)

The region that reads weights from memory and performs a matrix-vector dot product with the BF16 activation vector.

- **MXINT8 (Baseline):** Load INT8 weight and scale from global memory → cast INT8 to BF16 → multiply by scale → run BF16 FMA instruction.
- **MSAQ-s (added form):** Load packed weight from global memory → **call the PTX instruction (`bfe.s32`) to extract the element's `upper` bits and the group's `shared` bits → reconstruct INT8 via shift and add** → cast INT8 to BF16 → multiply by scale → run BF16 FMA instruction.
- **Summary:** Immediately after the load, a bit-masking and integer-synthesis cycle is added in registers.

### 2. KV Cache decoding (Attention stage)

The region that, during generation (decode), reads and dequantizes the KV cache to compute attention scores.

- **MXINT8 (Baseline):**
    - **Write:** Simply round the K, V tensors to INT8 by per-block max and store to HBM.
    - **Read:** Load INT8 data from HBM → BF16 conversion and scale multiply.
- **MSAQ-s (added form):**
    - **Write:** When storing the K, V tensors, add a **multi-step operation: coarse-grid quantization at $2^u$ scale → group-mean of the resulting residual → clamp the mean into the `shared` code**.
    - **Read:** When reading the packed data from HBM, add the **bit extraction and INT8 synthesis via `bfe.s32`** — the same as W-only GEMV — then convert to BF16.
- **Summary:** Double quantization (coarse quantization + residual mean) is added to the cache-write pipeline, and sub-byte unpacking is added to the read pipeline.

### 3. W+A GEMM (Compute-bound)

The region that converts both weight and activation data to INT8 to drive the Tensor Cores (IMMA).

- **MXINT8 (Baseline):** Load INT8 weight from Shared Memory → keep in INT8 form in registers. The activation data is quantized to INT8 by finding the per-block max → INT8 Tensor Core (IMMA) operation.
- **MSAQ-s (added form):** Load packed weight from Shared Memory → **inject `bfe.s32`-based unpacking code inside the iterator logic that brings data into registers, synthesizing to INT8** → pass the synthesized INT8 data to the IMMA instruction. (Activation quantization is kept identical to the MXINT8 method.)
- **Summary:** A bit-manipulation instruction is added explicitly to the step of the CUTLASS prologue module that moves weights from Shared Memory to registers. The mainloop and epilogue are completely identical to the baseline.

### 4. Prefill (W-only GEMM)

The region that dequantizes weights to use BF16 Tensor Cores, to process many input tokens.

- **MXINT8 (Baseline):** Load INT8 weight from Shared Memory → BF16 conversion and scale multiply → run BF16 Tensor Core operation together with the BF16 activation tile awaiting computation.
- **MSAQ-s (added form):** Load packed weight from Shared Memory → **extract bits with `bfe.s32` at the register-load step and synthesize into INT8 integers** → convert the synthesized INT8 integers to BF16 and multiply by scale → run BF16 Tensor Core operation.
- **Summary:** As in W+A GEMM, this scenario adds the logic that unpacks the packed bits into the ordinary 8-bit integer form right before the weight tile is dequantized.

## Packing and Unpacking Design Direction

The memory layout (packing) and reconstruction (unpacking) structure is the single most important factor determining the kernel's register pressure and global-memory access efficiency. The following presents the optimal design direction to fully exploit the hardware characteristics of CUDA and CUTLASS on the RTX 3090 (Ampere).

### 1. Packing design: 128-bit alignment and Structure of Arrays (SoA)

To hide global-memory latency (~300–400 cycles), you must extract maximum bandwidth from a single memory transaction.

- **SoA planar separation (Planar Layout):**
    - Do not interleave the `upper` bits, `shared` bits, and `scale` value into a single struct (AoS). Reading from global memory would then also load unnecessary data, lowering cache efficiency. Store the `upper` plane, `shared` plane, and `scale` plane as completely separate tensors in memory.

- **128-bit (16-byte) vectorized-load optimization:**
    - CUDA threads achieve the highest bandwidth efficiency when reading data in `int4` or `float4` (128-bit) units. Design the packed data so that it falls exactly into contiguous 128-bit chunks in memory.

- **Minimizing cross-boundary bit splits:**
    - The trickiest part. For example, when $u=3$ the `upper` code is 5 bits. Packing in 5-bit units inevitably straddles byte or 32-bit word boundaries. Resolving this by loading two registers and then performing a shift and OR wastes cycles.
    - *Solution:* When packing along the weight's K dimension (in 32-element blocks), consider either allowing some padding bits so that a whole number of $N$ element codes fit within the 32-bit register a thread will read, or rearranging the data order (swizzling) to match the thread mapping.

### 2. Unpacking design: the `bfe` instruction and JIT (Just-In-Time) reconstruction

Unpacking must be done in registers right before the computation, and the key is to minimize register allocation to keep the active-warp count high.

- **Instruction choice: `bfe.s32` (Bit Field Extract)**
    - Extracting data from the bitstream by combining bit shifts (`>>`, `<<`) and masking (`&`) needs 2–3 instructions. Using the PTX hardware instruction `bfe.s32` handles bit extraction including sign-extension in a single instruction (~4-cycle latency).

- **JIT (Just-In-Time) reconstruction and register pressure:**
    - To sustain the maximum active warps per SM on the RTX 3090, it is ideal to hold register usage per thread to 64 or fewer. Unpacking an entire tile at once and stacking it in registers exhausts the register file and sharply reduces occupancy.
    - *Solution:* Inside the mainloop (K-dimension loop), unpack only the minimum-size chunk needed (e.g., a 16×8 tile) with `bfe.s32` right before it feeds the FMA or IMMA instruction, and schedule it so the registers are immediately reused (overwritten) once the computation finishes.

### 3. Avoiding the CUTLASS and Shared Memory bottleneck

In the CUTLASS setting (W+A GEMM and Prefill), Shared Memory intervenes between global-memory access and unpacking.

- **Latency hiding with `cp.async`:**
    - When moving data from global memory to Shared Memory, use `cp.async` to overlap compute and memory movement. To use this instruction fully, the packed format must perfectly satisfy 16-byte alignment.

- **Avoiding Shared Memory bank conflicts:**
    - When loading the packed `upper` or `shared` bits from Shared Memory into registers (using `ldmatrix`, etc.), processing speed degrades severely if multiple threads access the same memory bank simultaneously. It is essential to store the packed data, in an offline step, in an XOR-swizzled layout that matches the Tensor Core's thread-access pattern.

- **Injecting unpacking inside the custom iterator:**
    - The unpacking code should be inserted not into the CUTLASS mainloop itself, but into the `load` function of the custom iterator that brings data from Shared Memory into registers, so it is handled inside the prologue pipeline and its overhead is hidden.

## Project Structure

### Overview

```
ms/
├── csrc/                      # 1. C++ / CUDA / CUTLASS kernel sources (backend)
│   ├── core/
│   │   └── ms_utils.cuh       # [shared header] common inline functions, e.g. bfe.s32 unpacking
│   ├── w_gemv.cu              # [pure CUDA] W-only decode matrix-vector dot product
│   ├── wa_gemm.cu             # [CUTLASS] W+A GEMM and prefill
│   ├── kv_attention.cu        # [pure CUDA] KV cache FlashAttention decoding
│   └── pybind.cpp             # [PyBind11] interface connecting C++ functions to Python
├── ms_lib/                    # 2. Python package (frontend)
│   ├── __init__.py
│   ├── ops.py                 # PyTorch wrapper functions that call the compiled C++ kernels
│   ├── pack.py                # weight and KV packing logic (for offline preprocessing)
│   └── reference.py           # (existing NumPy mirror) pure Python / NumPy logic for verification
├── tests/                     # 3. unit tests and benchmark
│   ├── test_w.py              # cross-validate reference.py output against the W-only GEMV kernel output
│   ├── test_wa.py
│   ├── test_kv.py
│   └── benchmark.py           # latency and throughput measurement script
└── setup.py                   # 4. build script (torch.utils.cpp_extension)
```

### `csrc/` (CUDA / C++ kernel layer)

- **Isolating `ms_utils.cuh`:** The core of MSAQ-s — "the logic that extracts `upper` and `shared` from the bitstream and synthesizes INT8 (using `bfe.s32`)" — is used everywhere: GEMV, GEMM, and Attention. Rather than rewriting it each time, declare it as a C++ `__device__ __forceinline__` function and `#include` it in each `.cu` file; this makes modification easy.
- **Separating CUTLASS from pure-CUDA files:** CUTLASS templates have very long compile times. Keeping `wa_gemm.cu` and `w_gemv.cu` separate prevents the heavy CUTLASS code from being recompiled when only the GEMV logic changes.

### `ms_lib/` (Python frontend layer)

- **`pack.py`:** Functions like the existing `pack_weight`, `pack_kv` are offline-natured code that runs only once — when saving the model to disk or before inference. Manage it separately from the kernel-driving logic.
- **`ops.py`:** Wraps the C++ binary built by `setup.py` to match the PyTorch tensor format, e.g. `torch.ops.msaq.wonly_gemv(...)`.
- **`reference.py`:** Do not discard the NumPy logic from the existing code (`_msaq_signed_ref`, `wonly_matmul`, etc.); collect it here. When writing C++ kernels, computed results frequently go wrong, and this file serves as the oracle (answer key).

### `tests/` (verification layer)

The existing `selftest_numpy()` and `selftest_triton()` must be fully split into modular `pytest`-based tests.
- When developing C++ kernels, it is very hard to trace where a segmentation fault or value error occurred. Separate the test files by feature — W-only, W+A, KV — to build an environment where you can quickly target and test a specific kernel right after a build.

### `setup.py` (build system)

Use PyTorch's `cpp_extension` module to compile the `.cu` and `.cpp` files in the `csrc/` folder into a single `.so` dynamic-library file. Define here the CUTLASS include paths and the necessary compile flags (`-O3`, `-arch=sm_86`, etc.).

## Kernel Optimization

### Which metric?

Choose metrics by splitting into micro (inside the kernel) and macro (outside the kernel) levels.

- **Clock cycles:**
    - **Use:** Measuring how much time a specific code block **inside the kernel** consumes — unpacking (`bfe.s32`), quantization ops, FMA (add / multiply), etc.
    - **Advantage:** Provides the most precise measurement, at hardware-cycle granularity.

- **Latency / Time (milliseconds ms / microseconds us):**
    - **Use:** Measuring the execution time of an entire W-only GEMV kernel, or an entire W+A GEMM kernel, to compare speed against cuBLAS (the reference point).

- **Instruction count / warp stalls (instruction count and stall causes):**
    - **Use:** Checking register pressure or memory-I/O bottlenecks (memory-bound) without inserting measurement code directly into the source.

### Where to place the measurement code in the project?

#### A. In-kernel cycle measurement (Micro Profiling)

**Location:** `.cu` and `.cuh` files inside `csrc/`

Use the CUDA built-in `clock64()` to compute the cycle difference before and after a specific region. Because this code wastes registers whenever it runs, it must be isolated via a compile macro (`#ifdef`) so it operates only in debug / optimization mode.

```cpp
// csrc/core/ms_utils.cuh (example)
#ifdef ENABLE_PROFILING
    unsigned long long start_time = clock64();
#endif

// 1. Unpacking logic (bfe.s32, etc.)
int32_t unpacked_w = unpack_ms_weight(packed_data);

#ifdef ENABLE_PROFILING
    unsigned long long unpack_cycles = clock64() - start_time;
    // store the result into a profiling-only buffer in global memory
    profiling_buffer[threadIdx.x] = unpack_cycles;
#endif
```

- **Caution:** Inserting `clock64()` changes the compiler's instruction scheduling and register allocation, which can introduce a small discrepancy from the original kernel performance. Use it only to set optimization direction.

#### B. Python-level kernel-latency measurement (Macro Profiling)

**Location:** `tests/benchmark.py`

Where you measure the final speed (latency) the user perceives. Use PyTorch's `torch.cuda.Event` to accurately measure the asynchronously-executing GPU operation time.

```python
# tests/benchmark.py (example)
import torch
import ms_lib.ops as ops

def measure_latency(func, *args, iters=100, warmup=10):
    # Warmup (cache warm-up)
    for _ in range(warmup):
        func(*args)

    torch.cuda.synchronize()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    for _ in range(iters):
        func(*args)
    end_event.record()

    torch.cuda.synchronize()
    # divide by iters to get the average per-call latency (ms)
    return start_event.elapsed_time(end_event) / iters

# run the measurement
latency_ms = measure_latency(ops.wonly_gemv, packed_W, X_bf16)
print(f"W-only GEMV Latency: {latency_ms:.4f} ms")
```

#### C. Using NVIDIA Nsight Compute (NCU) (Hardware Metrics)

**Location:** Terminal environment (no separate source-code modification)

To check whether unpacking and memory loads overlap well in the pipeline (latency hiding), embedding metrics in the source is not enough. Connecting NVIDIA's hardware profiler (NCU) to `tests/benchmark.py` and running it is the most recommended method.

Run the command below in the terminal to extract metrics.

```bash
# terminal example (inside the tests folder)
ncu --set full --metrics sm__cycles_active.avg.pct_of_peak_sustained_elapsed python benchmark.py
```

- **Key NCU metrics to check:**
    - `smsp__warp_issue_stalled_wait_desc.avg` (fraction of warps stalled waiting for a memory load)
    - `smsp__warp_issue_stalled_math_pipe.avg` (fraction stalled on register-op bottleneck — rises with overuse of `bfe.s32`)
    - `achieved_occupancy` (check the active-warp count)

#### Summary and design direction

1. Create `tests/benchmark.py` in the repository and, as the top priority, measure and record the **end-to-end latency in milliseconds (ms) using `torch.cuda.Event`**.
2. Analyze detailed bottleneck causes (which instruction is slow, whether registers are scarce) by running the **`ncu` (Nsight Compute) profiler** in the terminal, instead of modifying the source code directly.
3. Only when you must precisely tune the cycle count of the unpacking logic itself, in special situations, temporarily insert an `#ifdef` block and `clock64()` inside the `csrc/` code to perform a micro-benchmark.