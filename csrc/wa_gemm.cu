// csrc/wa_gemm.cu  —  W+A GEMM  +  W-only prefill GEMM
//
// The doc routes these two GEMM scopes to CUTLASS for tensor-core throughput
// (doc §"CUTLASS 기반 커널 파이프라인"):
//   * W+A GEMM    : weight unpacked to INT8, activation quantized to MXINT8 on
//                   the fly, INT8 IMMA mainloop, (scale_w*scale_a) epilogue.
//   * Prefill GEMM: weight unpacked + dequantized to BF16, BF16 tensor-core
//                   mainloop with FP32 accumulation.
//
// STATUS: GPU-UNVALIDATED CORRECTNESS BASELINES (plain CUDA, NOT yet CUTLASS).
// Like w_gemv/kv_attention, these reuse the certified ms::unpack_ms_weight_elem
// and match the oracle (ms_lib.reference.wa_matmul / wonly_matmul), so the
// pytest gates can certify the *math + wiring* end-to-end today. They are NOT
// the optimized path — one thread per output element, FP32 accumulate, no
// tensor cores.
//
// THE CUTLASS OPTIMIZATION (next phase, replaces the bodies below):
//   Build a CUTLASS GEMM and inject the unpack into the Shared->Register
//   **custom iterator's load()** (NOT the mainloop), so the bit ops hide inside
//   the cp.async prologue (doc §"Custom Iterator 내 언패킹 주입"). W+A uses an
//   INT8 IMMA GEMM with a (scale_w*scale_a) epilogue; prefill uses a BF16 GEMM
//   with a dequant prologue. Requires the register-aligned + XOR-swizzled pack
//   (see ms_utils.cuh) for bank-conflict-free ldmatrix; re-certify that layout
//   through the roundtrip gate first. setup.py already accepts $CUTLASS_DIR.

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <math.h>
#include "core/ms_utils.cuh"

namespace {

constexpr int BLOCK = 32;

// ---- W-only prefill GEMM: Y[M,OUT] = X[M,K] @ dequant(W)^T (FP32 accumulate) -
//   one thread per (m, o). Correctness baseline for the BF16 tensor-core path.
__global__ void wonly_gemm_kernel(
        const __nv_bfloat16* __restrict__ X,     // [M, K]
        const int8_t*  __restrict__ scale_exp,   // [NB, OUT]
        const uint8_t* __restrict__ upper,       // [NB, UB, OUT]
        const uint8_t* __restrict__ shared,      // [NB, SB, OUT]
        __nv_bfloat16* __restrict__ Y,           // [M, OUT]
        int M, int OUT, int K, int NB, int u, int gs, int UB, int SB) {
    const int o = blockIdx.x * blockDim.x + threadIdx.x;
    const int m = blockIdx.y * blockDim.y + threadIdx.y;
    if (o >= OUT || m >= M) return;

    float acc = 0.0f;
    for (int blk = 0; blk < NB; ++blk) {
        const float scale = ms::e8m0_to_scale(scale_exp[blk * OUT + o]);
        #pragma unroll
        for (int k = 0; k < BLOCK; ++k) {
            const int w = ms::unpack_ms_weight_elem(upper, shared, blk, o, k,
                                                    OUT, u, gs, UB, SB);
            const float xv = __bfloat162float(X[m * K + blk * BLOCK + k]);
            acc += (static_cast<float>(w) * scale) * xv;
        }
    }
    Y[m * OUT + o] = __float2bfloat16(acc);
}

// ---- W+A GEMM: weight -> int8, activation -> MXINT8 on the fly, per-block
//   (scale_a*scale_w) * int8-dot. one thread per (m, o). Correctness baseline
//   for the INT8 IMMA path. Activation quant mirrors reference.quant_act
//   (share=False): s = 2^(floor(log2(max|x|))-6), q = clip(rint(x/s),-127,127).
__global__ void wa_gemm_kernel(
        const __nv_bfloat16* __restrict__ X,     // [M, K]
        const int8_t*  __restrict__ scale_exp,   // [NB, OUT]
        const uint8_t* __restrict__ upper,       // [NB, UB, OUT]
        const uint8_t* __restrict__ shared,      // [NB, SB, OUT]
        __nv_bfloat16* __restrict__ Y,           // [M, OUT]
        int M, int OUT, int K, int NB, int u, int gs, int UB, int SB) {
    const int o = blockIdx.x * blockDim.x + threadIdx.x;
    const int m = blockIdx.y * blockDim.y + threadIdx.y;
    if (o >= OUT || m >= M) return;

    float acc = 0.0f;
    for (int blk = 0; blk < NB; ++blk) {
        // on-the-fly MXINT8 activation quant for this row's 32-block
        float amax = 1e-30f;
        #pragma unroll
        for (int k = 0; k < BLOCK; ++k)
            amax = fmaxf(amax, fabsf(__bfloat162float(X[m * K + blk * BLOCK + k])));
        float ea = floorf(log2f(amax)) - (float)ms::E_MAX;        // E_MAX = 6
        ea = fmaxf(fminf(ea, 127.0f), -127.0f);
        const float sa = exp2f(ea);

        const float sw = ms::e8m0_to_scale(scale_exp[blk * OUT + o]);
        int idot = 0;
        #pragma unroll
        for (int k = 0; k < BLOCK; ++k) {
            const float xv = __bfloat162float(X[m * K + blk * BLOCK + k]);
            int qx = (int)rintf(xv / sa);
            qx = max(-127, min(127, qx));
            const int qw = ms::unpack_ms_weight_elem(upper, shared, blk, o, k,
                                                     OUT, u, gs, UB, SB);
            idot += qx * qw;
        }
        acc += (float)idot * sa * sw;
    }
    Y[m * OUT + o] = __float2bfloat16(acc);
}

} // namespace

// ---- host launchers (signatures match ms_lib.ops / the pybind schema) -------
static inline void gemm_dims(int64_t u, int64_t gs, int& UB, int& SB) {
    const int wbits = 8 - (int)u;
    UB = BLOCK * wbits / 8;
    SB = ((BLOCK / (int)gs) * (int)u + 7) / 8;
}

torch::Tensor wonly_gemm_cuda(
        torch::Tensor X, torch::Tensor scale_exp,
        torch::Tensor upper, torch::Tensor shared,
        int64_t M, int64_t OUT, int64_t K, int64_t NB, int64_t u, int64_t gs) {
    TORCH_CHECK(X.is_cuda() && X.scalar_type() == torch::kBFloat16, "X must be CUDA bf16");
    int UB, SB; gemm_dims(u, gs, UB, SB);
    auto Y = torch::empty({M, OUT}, X.options());
    const dim3 block(32, 8);
    const dim3 grid((OUT + block.x - 1) / block.x, (M + block.y - 1) / block.y);
    wonly_gemm_kernel<<<grid, block>>>(
        reinterpret_cast<const __nv_bfloat16*>(X.data_ptr<at::BFloat16>()),
        scale_exp.data_ptr<int8_t>(), upper.data_ptr<uint8_t>(), shared.data_ptr<uint8_t>(),
        reinterpret_cast<__nv_bfloat16*>(Y.data_ptr<at::BFloat16>()),
        (int)M, (int)OUT, (int)K, (int)NB, (int)u, (int)gs, UB, SB);
    return Y;
}

torch::Tensor wa_gemm_cuda(
        torch::Tensor X, torch::Tensor scale_exp,
        torch::Tensor upper, torch::Tensor shared,
        int64_t M, int64_t OUT, int64_t K, int64_t NB, int64_t u, int64_t gs) {
    TORCH_CHECK(X.is_cuda() && X.scalar_type() == torch::kBFloat16, "X must be CUDA bf16");
    int UB, SB; gemm_dims(u, gs, UB, SB);
    auto Y = torch::empty({M, OUT}, X.options());
    const dim3 block(32, 8);
    const dim3 grid((OUT + block.x - 1) / block.x, (M + block.y - 1) / block.y);
    wa_gemm_kernel<<<grid, block>>>(
        reinterpret_cast<const __nv_bfloat16*>(X.data_ptr<at::BFloat16>()),
        scale_exp.data_ptr<int8_t>(), upper.data_ptr<uint8_t>(), shared.data_ptr<uint8_t>(),
        reinterpret_cast<__nv_bfloat16*>(Y.data_ptr<at::BFloat16>()),
        (int)M, (int)OUT, (int)K, (int)NB, (int)u, (int)gs, UB, SB);
    return Y;
}
