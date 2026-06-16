// csrc/w_gemv.cu  —  [pure CUDA]  W-only decode matrix-vector product (Batch=1)
//
// Decode Linear: y[OUT] = x[K] @ dequant(W)^T. Weight read from HBM dominates,
// so the fused path keeps the reconstructed weight in registers and never
// re-materializes BF16 weight in HBM (doc §"W-only GEMV (Decode)").
//
// STATUS: GPU-UNVALIDATED DRAFT. The numerics + addressing reuse the certified
// ms::unpack_ms_weight_elem (bit-exact vs ms_lib.pack), so this is correct by
// construction against the oracle; only the CUDA execution is unproven. Certify
// on the RTX 3090 with tests/test_w.py::test_wonly_gemv_vs_oracle.
//
// THIS DRAFT'S STRATEGY vs the doc's 5-stage split-K plan:
//   The out-innermost SoA layout means thread `o` reads plane[...*OUT + o], so
//   adjacent threads read adjacent bytes -> naturally COALESCED. So this draft
//   uses one-thread-per-output-column with an FP32 accumulator and x broadcast
//   through L2 (no shared-mem K ceiling) — the simplest mapping that is both
//   correct and coalesced. The doc's split-K + __shfl_down_sync warp-reduce + atomicAdd is
//   the OCCUPANCY optimization to add when OUT is too small to fill the SMs;
//   structure it in once correctness is certified. Likewise __hfma2 (vectorized
//   bf16 FMA) and the bit-stuffing dequant replace the float path below as the
//   micro-optimization (profile first; see README).

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include "core/ms_utils.cuh"

namespace {

constexpr int BLOCK = 32;

// SPLIT-K (occupancy): one thread per output column `o`, but the K-reduction is
// split across blockIdx.y = `sp` (one K-slice each), so the grid is
// base_blocks*splitK instead of base_blocks -> fills the SMs. Each (o, sp) block
// reduces only its block range and writes a PARTIAL sum; gemv_combine_kernel sums
// the splitK partials (no atomics, per the KV lesson). x[k] is read at the SAME
// address by every column (warp broadcast), streaming cheaply through L2.
//
// (Register-blocking — COLS output columns/thread for MLP — was tried and
// reverted: it spiked the register-heavy unpack to 114 regs, collapsing
// occupancy 48->17 warps, net-negative. Same lesson as Stage 4b: per-thread ILP
// levers don't beat the unpack's intrinsic throughput cost. See change.md Phase 7.)
__global__ void wonly_gemv_splitk_kernel(
        const __nv_bfloat16* __restrict__ x,     // [K]
        const int8_t*  __restrict__ scale_exp,   // [NB, OUT]
        const uint8_t* __restrict__ upper,       // [NB, UB, OUT]
        const uint8_t* __restrict__ shared,      // [NB, SB, OUT]
        float* __restrict__ partial,             // [splitK, OUT]
        int OUT, int NB, int u, int gs, int UB, int SB, int splitK) {

    const int o  = blockIdx.x * blockDim.x + threadIdx.x;
    const int sp = blockIdx.y;                    // K-slice this block owns
    if (o >= OUT) return;

    const int per = (NB + splitK - 1) / splitK;   // 32-blocks per split
    const int b0  = sp * per;
    const int b1  = min(b0 + per, NB);

    float acc = 0.0f;
    for (int blk = b0; blk < b1; ++blk) {
        const float scale = ms::e8m0_to_scale(scale_exp[blk * OUT + o]);
        #pragma unroll
        for (int k = 0; k < BLOCK; ++k) {
            const int w = ms::unpack_ms_weight_elem(upper, shared, blk, o, k,
                                                    OUT, u, gs, UB, SB);
            const float xv = __bfloat162float(x[blk * BLOCK + k]);
            acc += (static_cast<float>(w) * scale) * xv;
        }
    }
    partial[(long)sp * OUT + o] = acc;            // partial; combine sums over sp
}

// Sum the splitK partial columns into the final y[o] (linear reduction; GEMV
// needs no softmax rescale, unlike the KV combine). One thread per output column.
__global__ void gemv_combine_kernel(
        const float* __restrict__ partial, __nv_bfloat16* __restrict__ y,
        int OUT, int splitK) {
    const int o = blockIdx.x * blockDim.x + threadIdx.x;
    if (o >= OUT) return;
    float acc = 0.0f;
    for (int sp = 0; sp < splitK; ++sp) acc += partial[(long)sp * OUT + o];
    y[o] = __float2bfloat16(acc);
}

} // namespace

// Host launcher. Signature matches ms_lib.ops.wonly_gemv / the pybind schema.
torch::Tensor wonly_gemv_cuda(
        torch::Tensor x,          // bf16 [K]
        torch::Tensor scale_exp,  // int8  [NB, OUT]
        torch::Tensor upper,      // uint8 [NB, UB, OUT]
        torch::Tensor shared,     // uint8 [NB, SB, OUT]
        int64_t OUT, int64_t NB, int64_t u, int64_t gs) {
    TORCH_CHECK(x.is_cuda() && upper.is_cuda(), "inputs must be CUDA tensors");
    TORCH_CHECK(x.scalar_type() == torch::kBFloat16, "x must be bf16");
    const int wbits = 8 - (int)u;
    const int UB = BLOCK * wbits / 8;
    const int SB = ((BLOCK / (int)gs) * (int)u + 7) / 8;

    auto y = torch::empty({OUT}, x.options());
    const int threads = 128;
    const int blocks = (int)((OUT + threads - 1) / threads);

    // split the K-reduction -> grid (blocks, splitK) so the block count fills SMs
    const int splitK = ms::gemv_splitk_count(blocks, (int)NB);
    auto partial = torch::empty({(int64_t)splitK, OUT},
                                x.options().dtype(torch::kFloat32));

    wonly_gemv_splitk_kernel<<<dim3(blocks, splitK), threads>>>(
        reinterpret_cast<const __nv_bfloat16*>(x.data_ptr<at::BFloat16>()),
        scale_exp.data_ptr<int8_t>(),
        upper.data_ptr<uint8_t>(),
        shared.data_ptr<uint8_t>(),
        partial.data_ptr<float>(),
        (int)OUT, (int)NB, (int)u, (int)gs, UB, SB, splitK);

    gemv_combine_kernel<<<blocks, threads>>>(
        partial.data_ptr<float>(),
        reinterpret_cast<__nv_bfloat16*>(y.data_ptr<at::BFloat16>()),
        (int)OUT, splitK);
    return y;
}
