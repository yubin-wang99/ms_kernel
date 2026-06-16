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
//   SHARED-MEMORY TILED so the weight is UNPACKED ONCE per block-tile and reused
//   by all TBM rows (the naive per-(m,o) kernel re-unpacked the weight M times ->
//   unpack cost scaled with batch). Each block computes a TBM x TBN output tile;
//   the K-loop streams TBK(=one MSAQ block) at a time into shared: As[k][m] from
//   X, Bs[k][o] = dequant(W) (the only place unpack happens). Each thread holds a
//   RTM x RTN register tile. Amortizes unpack by TBM(=64); large M -> FMA-bound.
constexpr int TBM = 64, TBN = 64, TBK = BLOCK;   // block tile (TBK = MSAQ block)
constexpr int RTM = 4,  RTN = 4;                 // per-thread register tile
// blockDim = (TBM/RTM)*(TBN/RTN) = 16*16 = 256 threads
__global__ void wonly_gemm_kernel(
        const __nv_bfloat16* __restrict__ X,     // [M, K]
        const int8_t*  __restrict__ scale_exp,   // [NB, OUT]
        const uint8_t* __restrict__ upper,       // [NB, UB, OUT]
        const uint8_t* __restrict__ shared,      // [NB, SB, OUT]
        __nv_bfloat16* __restrict__ Y,           // [M, OUT]
        int M, int OUT, int K, int NB, int u, int gs, int UB, int SB) {
    __shared__ float As[TBK][TBM];               // X tile (transposed: [k][m])
    __shared__ float Bs[TBK][TBN];               // dequant(W) tile [k][o]
    const int m0 = blockIdx.y * TBM, o0 = blockIdx.x * TBN;
    const int tid = threadIdx.x, tRow = tid / 16, tCol = tid % 16;

    float acc[RTM][RTN];
    #pragma unroll
    for (int i = 0; i < RTM; ++i)
        #pragma unroll
        for (int j = 0; j < RTN; ++j) acc[i][j] = 0.0f;

    for (int blk = 0; blk < NB; ++blk) {
        // --- stage X tile: As[k][m] ---
        for (int idx = tid; idx < TBM * TBK; idx += 256) {
            const int m = idx / TBK, k = idx % TBK;
            As[k][m] = (m0 + m < M)
                     ? __bfloat162float(X[(long)(m0 + m) * K + blk * TBK + k]) : 0.0f;
        }
        // --- stage dequant(W) tile: Bs[k][o] (the ONE unpack, reused by TBM rows) ---
        for (int idx = tid; idx < TBN * TBK; idx += 256) {
            const int oc = idx / TBK, k = idx % TBK, o = o0 + oc;
            float v = 0.0f;
            if (o < OUT) {
                const float sc = ms::e8m0_to_scale(scale_exp[blk * OUT + o]);
                v = (float)ms::unpack_ms_weight_elem(upper, shared, blk, o, k,
                                                     OUT, u, gs, UB, SB) * sc;
            }
            Bs[k][oc] = v;
        }
        __syncthreads();
        // --- register-tiled FMA ---
        #pragma unroll
        for (int k = 0; k < TBK; ++k) {
            float a[RTM], b[RTN];
            #pragma unroll
            for (int i = 0; i < RTM; ++i) a[i] = As[k][tRow * RTM + i];
            #pragma unroll
            for (int j = 0; j < RTN; ++j) b[j] = Bs[k][tCol * RTN + j];
            #pragma unroll
            for (int i = 0; i < RTM; ++i)
                #pragma unroll
                for (int j = 0; j < RTN; ++j) acc[i][j] += a[i] * b[j];
        }
        __syncthreads();
    }
    #pragma unroll
    for (int i = 0; i < RTM; ++i) {
        const int m = m0 + tRow * RTM + i;
        if (m >= M) continue;
        #pragma unroll
        for (int j = 0; j < RTN; ++j) {
            const int o = o0 + tCol * RTN + j;
            if (o < OUT) Y[(long)m * OUT + o] = __float2bfloat16(acc[i][j]);
        }
    }
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
    const dim3 block(256);
    const dim3 grid((OUT + TBN - 1) / TBN, (M + TBM - 1) / TBM);
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
