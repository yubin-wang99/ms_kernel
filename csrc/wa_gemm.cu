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

// ---- W+A GEMM — SHARED-MEMORY TILED. Like wonly_gemm but the activation is
//   MXINT8-quantized per (row, 32-block) on the fly. Since sa (act scale) and sw
//   (weight scale) are both powers of two, qx*qw*sa*sw is exact, so folding
//   As[k][m] = qx*sa and Bs[k][o] = qw*sw reduces W+A to the SAME accumulate as
//   wonly_gemm (= the oracle's per-block int-dot * sa*sw, bit-for-bit). Activation
//   quant mirrors reference.quant_act (share=False): sa = 2^(floor(log2 max|x|)-6),
//   qx = clip(rint(x/sa), -127, 127). The weight unpack (Bs) is staged once per
//   tile and reused by all TBM rows.
__global__ void wa_gemm_kernel(
        const __nv_bfloat16* __restrict__ X,     // [M, K]
        const int8_t*  __restrict__ scale_exp,   // [NB, OUT]
        const uint8_t* __restrict__ upper,       // [NB, UB, OUT]
        const uint8_t* __restrict__ shared,      // [NB, SB, OUT]
        __nv_bfloat16* __restrict__ Y,           // [M, OUT]
        int M, int OUT, int K, int NB, int u, int gs, int UB, int SB) {
    __shared__ float As[TBK][TBM];               // quant+dequant activation [k][m]
    __shared__ float Bs[TBK][TBN];               // dequant(W) [k][o]
    __shared__ float sa_s[TBM];                  // per-row activation scale (this block)
    const int m0 = blockIdx.y * TBM, o0 = blockIdx.x * TBN;
    const int tid = threadIdx.x, tRow = tid / 16, tCol = tid % 16;

    float acc[RTM][RTN];
    #pragma unroll
    for (int i = 0; i < RTM; ++i)
        #pragma unroll
        for (int j = 0; j < RTN; ++j) acc[i][j] = 0.0f;

    for (int blk = 0; blk < NB; ++blk) {
        // stage raw X tile
        for (int idx = tid; idx < TBM * TBK; idx += 256) {
            const int m = idx / TBK, k = idx % TBK;
            As[k][m] = (m0 + m < M)
                     ? __bfloat162float(X[(long)(m0 + m) * K + blk * TBK + k]) : 0.0f;
        }
        // stage dequant(W) tile (the ONE unpack, reused by TBM rows)
        for (int idx = tid; idx < TBN * TBK; idx += 256) {
            const int oc = idx / TBK, k = idx % TBK, o = o0 + oc;
            float v = 0.0f;
            if (o < OUT) {
                const float sw = ms::e8m0_to_scale(scale_exp[blk * OUT + o]);
                v = (float)ms::unpack_ms_weight_elem(upper, shared, blk, o, k,
                                                     OUT, u, gs, UB, SB) * sw;
            }
            Bs[k][oc] = v;
        }
        __syncthreads();
        // per-row activation scale: amax over the row's 32 values -> sa
        if (tid < TBM) {
            float amax = 1e-30f;
            #pragma unroll
            for (int k = 0; k < TBK; ++k) amax = fmaxf(amax, fabsf(As[k][tid]));
            float ea = floorf(log2f(amax)) - (float)ms::E_MAX;
            ea = fmaxf(fminf(ea, 127.0f), -127.0f);
            sa_s[tid] = exp2f(ea);
        }
        __syncthreads();
        // quant+dequant activation in place: As = clip(rint(x/sa),±127) * sa
        for (int idx = tid; idx < TBM * TBK; idx += 256) {
            const int m = idx / TBK, k = idx % TBK;
            const float sa = sa_s[m];
            int qx = (int)rintf(As[k][m] / sa);
            qx = max(-127, min(127, qx));
            As[k][m] = (float)qx * sa;
        }
        __syncthreads();
        // register-tiled accumulate (As folds sa, Bs folds sw -> exact int-dot*sa*sw)
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
    const dim3 block(256);
    const dim3 grid((OUT + TBN - 1) / TBN, (M + TBM - 1) / TBM);
    wa_gemm_kernel<<<grid, block>>>(
        reinterpret_cast<const __nv_bfloat16*>(X.data_ptr<at::BFloat16>()),
        scale_exp.data_ptr<int8_t>(), upper.data_ptr<uint8_t>(), shared.data_ptr<uint8_t>(),
        reinterpret_cast<__nv_bfloat16*>(Y.data_ptr<at::BFloat16>()),
        (int)M, (int)OUT, (int)K, (int)NB, (int)u, (int)gs, UB, SB);
    return Y;
}
