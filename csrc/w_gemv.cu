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
#include <cuda_pipeline.h>
#include <cstdlib>
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

// ---- cp.async double-buffered split-K GEMV (hide unpack behind memory) ------
//   Same split-K, but each block's packed bytes (upper/shared/scale for its 128
//   columns) are PREFETCHED to shared via cp.async while the previous block is
//   being unpacked+accumulated -> the global loads overlap the unpack ALU, so the
//   unpack hides under memory (the "memory-bound regime" where MSAQ's fewer bytes
//   pay off). The block's tile is staged as-is ([BYTES][128]); unpack reads it
//   with OUT=128, blk=0 (reusing unpack_ms_weight_elem). Requires OUT % 128 == 0
//   and blockDim == 128. Writes partial[sp,o]; combine sums over sp (unchanged).
__global__ void wonly_gemv_cpasync_kernel(
        const __nv_bfloat16* __restrict__ x,
        const int8_t*  __restrict__ scale_exp,
        const uint8_t* __restrict__ upper,
        const uint8_t* __restrict__ shared,
        float* __restrict__ partial,
        int OUT, int NB, int u, int gs, int UB, int SB, int splitK) {
    const int o0  = blockIdx.x * 128;
    const int lane = threadIdx.x;                 // 0..127  -> column o = o0+lane
    const int o   = o0 + lane;
    const int sp  = blockIdx.y;
    const int per = (NB + splitK - 1) / splitK;
    const int b0  = sp * per, b1 = min(b0 + per, NB);
    if (b0 >= b1) { if (o < OUT) partial[(long)sp * OUT + o] = 0.0f; return; }

    extern __shared__ uint8_t smem[];
    const int bufBytes = (UB + SB) * 128 + 128;   // upper | shared | scale
    uint8_t* buf[2] = { smem, smem + bufBytes };

    // stage block `blk`'s tile into buffer `b` via cp.async (128 threads)
    auto stage = [&](int blk, uint8_t* b) {
        uint8_t* up_s = b;
        uint8_t* sh_s = b + UB * 128;
        int8_t*  sc_s = (int8_t*)(b + (UB + SB) * 128);
        const uint8_t* gup = upper     + (long)blk * UB * OUT + o0;
        const uint8_t* gsh = shared    + (long)blk * SB * OUT + o0;
        const int8_t*  gsc = scale_exp + (long)blk * OUT      + o0;
        for (int c = lane; c < UB * 8; c += 128) { int r = c >> 3, ch = c & 7;
            __pipeline_memcpy_async(up_s + r*128 + ch*16, gup + (long)r*OUT + ch*16, 16); }
        for (int c = lane; c < SB * 8; c += 128) { int r = c >> 3, ch = c & 7;
            __pipeline_memcpy_async(sh_s + r*128 + ch*16, gsh + (long)r*OUT + ch*16, 16); }
        for (int ch = lane; ch < 8; ch += 128)
            __pipeline_memcpy_async(sc_s + ch*16, gsc + ch*16, 16);
    };

    stage(b0, buf[0]);
    __pipeline_commit();

    float acc = 0.0f;
    for (int blk = b0; blk < b1; ++blk) {
        const int cur = (blk - b0) & 1;
        const bool more = (blk + 1) < b1;
        if (more) { stage(blk + 1, buf[(blk - b0 + 1) & 1]); __pipeline_commit(); }
        __pipeline_wait_prior(more ? 1 : 0);
        __syncthreads();
        const uint8_t* up_s = buf[cur];
        const uint8_t* sh_s = buf[cur] + UB * 128;
        const int8_t*  sc_s = (const int8_t*)(buf[cur] + (UB + SB) * 128);
        const float scale = ms::e8m0_to_scale(sc_s[lane]);
        #pragma unroll
        for (int k = 0; k < BLOCK; ++k) {
            const int w = ms::unpack_ms_weight_elem(up_s, sh_s, 0, lane, k, 128, u, gs, UB, SB);
            acc += (static_cast<float>(w) * scale) * __bfloat162float(x[blk * BLOCK + k]);
        }
        __syncthreads();
    }
    if (o < OUT) partial[(long)sp * OUT + o] = acc;
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

    const char* e = getenv("MS_GEMV_CPASYNC");
    const bool cpasync = (OUT % 128 == 0) && !(e && atoi(e) == 0);
    if (cpasync) {                  // hide the unpack behind cp.async prefetch
        const size_t smem = (size_t)2 * ((UB + SB) * 128 + 128);
        wonly_gemv_cpasync_kernel<<<dim3(blocks, splitK), threads, smem>>>(
            reinterpret_cast<const __nv_bfloat16*>(x.data_ptr<at::BFloat16>()),
            scale_exp.data_ptr<int8_t>(), upper.data_ptr<uint8_t>(), shared.data_ptr<uint8_t>(),
            partial.data_ptr<float>(), (int)OUT, (int)NB, (int)u, (int)gs, UB, SB, splitK);
    } else {
        wonly_gemv_splitk_kernel<<<dim3(blocks, splitK), threads>>>(
            reinterpret_cast<const __nv_bfloat16*>(x.data_ptr<at::BFloat16>()),
            scale_exp.data_ptr<int8_t>(),
            upper.data_ptr<uint8_t>(),
            shared.data_ptr<uint8_t>(),
            partial.data_ptr<float>(),
            (int)OUT, (int)NB, (int)u, (int)gs, UB, SB, splitK);
    }

    gemv_combine_kernel<<<blocks, threads>>>(
        partial.data_ptr<float>(),
        reinterpret_cast<__nv_bfloat16*>(y.data_ptr<at::BFloat16>()),
        (int)OUT, splitK);
    return y;
}
