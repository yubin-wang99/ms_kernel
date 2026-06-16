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
#include <mma.h>
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
constexpr int TBK = BLOCK;   // K-tile = one MSAQ block (fixed)

// Templated shared-memory tiled W-only GEMM. <TBM,TBN> output tile, <RTM,RTN>
// per-thread register tile; blockDim = (TBM/RTM)*(TBN/RTN). Weight unpacked ONCE
// per tile (Bs stage) and reused by all TBM rows. Tile config swept via
// MS_TILE_CFG (host dispatch below) to find the optimum.
template<int TBM, int TBN, int RTM, int RTN>
__global__ void wonly_gemm_tiled(
        const __nv_bfloat16* __restrict__ X,
        const int8_t*  __restrict__ scale_exp,
        const uint8_t* __restrict__ upper,
        const uint8_t* __restrict__ shared,
        __nv_bfloat16* __restrict__ Y,
        int M, int OUT, int K, int NB, int u, int gs, int UB, int SB) {
    constexpr int TNT = TBN / RTN;               // threads along N
    constexpr int NT  = (TBM / RTM) * TNT;       // total threads
    __shared__ float As[TBK][TBM];
    __shared__ float Bs[TBK][TBN];
    const int m0 = blockIdx.y * TBM, o0 = blockIdx.x * TBN;
    const int tid = threadIdx.x, tRow = tid / TNT, tCol = tid % TNT;

    float acc[RTM][RTN];
    #pragma unroll
    for (int i = 0; i < RTM; ++i)
        #pragma unroll
        for (int j = 0; j < RTN; ++j) acc[i][j] = 0.0f;

    for (int blk = 0; blk < NB; ++blk) {
        for (int idx = tid; idx < TBM * TBK; idx += NT) {
            const int m = idx / TBK, k = idx % TBK;
            As[k][m] = (m0 + m < M)
                     ? __bfloat162float(X[(long)(m0 + m) * K + blk * TBK + k]) : 0.0f;
        }
        for (int idx = tid; idx < TBN * TBK; idx += NT) {
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

// Templated W+A GEMM — same tiling, plus per-(row,block) MXINT8 activation quant.
// sa, sw are powers of two so folding As=qx*sa, Bs=qw*sw is bit-exact to the
// per-block int-dot * sa*sw. Activation quant mirrors reference.quant_act.
template<int TBM, int TBN, int RTM, int RTN>
__global__ void wa_gemm_tiled(
        const __nv_bfloat16* __restrict__ X,
        const int8_t*  __restrict__ scale_exp,
        const uint8_t* __restrict__ upper,
        const uint8_t* __restrict__ shared,
        __nv_bfloat16* __restrict__ Y,
        int M, int OUT, int K, int NB, int u, int gs, int UB, int SB) {
    constexpr int TNT = TBN / RTN;
    constexpr int NT  = (TBM / RTM) * TNT;
    __shared__ float As[TBK][TBM];
    __shared__ float Bs[TBK][TBN];
    __shared__ float sa_s[TBM];
    const int m0 = blockIdx.y * TBM, o0 = blockIdx.x * TBN;
    const int tid = threadIdx.x, tRow = tid / TNT, tCol = tid % TNT;

    float acc[RTM][RTN];
    #pragma unroll
    for (int i = 0; i < RTM; ++i)
        #pragma unroll
        for (int j = 0; j < RTN; ++j) acc[i][j] = 0.0f;

    for (int blk = 0; blk < NB; ++blk) {
        for (int idx = tid; idx < TBM * TBK; idx += NT) {
            const int m = idx / TBK, k = idx % TBK;
            As[k][m] = (m0 + m < M)
                     ? __bfloat162float(X[(long)(m0 + m) * K + blk * TBK + k]) : 0.0f;
        }
        for (int idx = tid; idx < TBN * TBK; idx += NT) {
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
        for (int m = tid; m < TBM; m += NT) {
            float amax = 1e-30f;
            #pragma unroll
            for (int k = 0; k < TBK; ++k) amax = fmaxf(amax, fabsf(As[k][m]));
            float ea = floorf(log2f(amax)) - (float)ms::E_MAX;
            ea = fmaxf(fminf(ea, 127.0f), -127.0f);
            sa_s[m] = exp2f(ea);
        }
        __syncthreads();
        for (int idx = tid; idx < TBM * TBK; idx += NT) {
            const int m = idx / TBK, k = idx % TBK;
            const float sa = sa_s[m];
            int qx = (int)rintf(As[k][m] / sa);
            qx = max(-127, min(127, qx));
            As[k][m] = (float)qx * sa;
        }
        __syncthreads();
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

// ---- W-only prefill via BF16 TENSOR CORES (WMMA 16x16x16) -------------------
//   Same staging as wonly_gemm_tiled (Bs = dequant(W) bf16, As = X bf16) but the
//   inner product runs on tensor cores. Block tile 64x64, 4 warps (2x2) each
//   owning a 32x32 region (2x2 frags). Layout: C[m,o]=Sum_k X[m,k]*Wdq[o,k] ->
//   A=As[m][k] row_major, B=Bs[o][k] read as col_major K×O (== Wdq). 128 threads.
using namespace nvcuda;
constexpr int WSK = BLOCK + 8;          // shared K width (+pad to dodge bank conflicts)
__global__ void wonly_gemm_wmma(
        const __nv_bfloat16* __restrict__ X,
        const int8_t*  __restrict__ scale_exp,
        const uint8_t* __restrict__ upper,
        const uint8_t* __restrict__ shared,
        __nv_bfloat16* __restrict__ Y,
        int M, int OUT, int K, int NB, int u, int gs, int UB, int SB) {
    __shared__ __nv_bfloat16 As[64][WSK];   // X         [m][k]
    __shared__ __nv_bfloat16 Bs[64][WSK];   // dequant(W) [o][k]
    __shared__ float tmp[4][16][16];        // per-warp store scratch (float -> bf16)
    const int m0 = blockIdx.y * 64, o0 = blockIdx.x * 64;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int wM = warp >> 1, wN = warp & 1;          // 2x2 warp grid

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[2][2];
    #pragma unroll
    for (int i = 0; i < 2; ++i)
        #pragma unroll
        for (int j = 0; j < 2; ++j) wmma::fill_fragment(acc[i][j], 0.0f);

    for (int blk = 0; blk < NB; ++blk) {
        for (int idx = tid; idx < 64 * BLOCK; idx += 128) {
            const int m = idx / BLOCK, k = idx % BLOCK;
            As[m][k] = (m0 + m < M) ? X[(long)(m0 + m) * K + blk * BLOCK + k]
                                    : __float2bfloat16(0.0f);
        }
        for (int idx = tid; idx < 64 * BLOCK; idx += 128) {
            const int oc = idx / BLOCK, k = idx % BLOCK, o = o0 + oc;
            float v = 0.0f;
            if (o < OUT) {
                const float sc = ms::e8m0_to_scale(scale_exp[blk * OUT + o]);
                v = (float)ms::unpack_ms_weight_elem(upper, shared, blk, o, k,
                                                     OUT, u, gs, UB, SB) * sc;
            }
            Bs[oc][k] = __float2bfloat16(v);
        }
        __syncthreads();
        #pragma unroll
        for (int wk = 0; wk < BLOCK / 16; ++wk) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> a[2];
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> b[2];
            #pragma unroll
            for (int i = 0; i < 2; ++i) wmma::load_matrix_sync(a[i], &As[wM*32 + i*16][wk*16], WSK);
            #pragma unroll
            for (int j = 0; j < 2; ++j) wmma::load_matrix_sync(b[j], &Bs[wN*32 + j*16][wk*16], WSK);
            #pragma unroll
            for (int i = 0; i < 2; ++i)
                #pragma unroll
                for (int j = 0; j < 2; ++j) wmma::mma_sync(acc[i][j], a[i], b[j], acc[i][j]);
        }
        __syncthreads();
    }
    float* wt = &tmp[warp][0][0];
    #pragma unroll
    for (int i = 0; i < 2; ++i)
        #pragma unroll
        for (int j = 0; j < 2; ++j) {
            wmma::store_matrix_sync(wt, acc[i][j], 16, wmma::mem_row_major);
            __syncwarp();
            const int mb = m0 + wM*32 + i*16, ob = o0 + wN*32 + j*16;
            for (int e = lane; e < 256; e += 32) {
                const int m = mb + e / 16, o = ob + e % 16;
                if (m < M && o < OUT) Y[(long)m * OUT + o] = __float2bfloat16(wt[e]);
            }
            __syncwarp();
        }
}

// Tile-config dispatch. Swept optimum (RTX 3090, OUT=4096) is M-ADAPTIVE: a
// 128x128 tile reuses best but yields too few blocks (OUT/128=32 < 82 SMs) until
// M fills the second grid dim, so small M prefers 64x64 (more blocks). Crossover
// ~M=256. MS_TILE_CFG (env) forces a specific config (for sweeps).
//   cfg 1 = 64x64 r4x4 ;  cfg 5 = 128x128 r8x8
static inline int tile_cfg(int M) {
    if (const char* e = getenv("MS_TILE_CFG")) { int c = atoi(e); if (c >= 0) return c; }
    return (M >= 256) ? 5 : 1;
}

} // namespace

// ---- host launchers (signatures match ms_lib.ops / the pybind schema) -------
static inline void gemm_dims(int64_t u, int64_t gs, int& UB, int& SB) {
    const int wbits = 8 - (int)u;
    UB = BLOCK * wbits / 8;
    SB = ((BLOCK / (int)gs) * (int)u + 7) / 8;
}

// launch KERN<TBM,TBN,RTM,RTN> for the given tile; ARGS must be defined.
#define LAUNCH_TILE(KERN, BM, BN, RM, RN)                                        \
    KERN<BM, BN, RM, RN>                                                         \
        <<<dim3(((int)OUT + (BN) - 1) / (BN), ((int)M + (BM) - 1) / (BM)),       \
           dim3(((BM) / (RM)) * ((BN) / (RN)))>>>(ARGS)
#define DISPATCH_TILE(KERN) do {                                                 \
    switch (tile_cfg((int)M)) {                                                  \
        case 0: LAUNCH_TILE(KERN,  32,  32, 2, 2); break;                        \
        case 1: LAUNCH_TILE(KERN,  64,  64, 4, 4); break;                        \
        case 2: LAUNCH_TILE(KERN,  64,  64, 8, 8); break;                        \
        case 3: LAUNCH_TILE(KERN, 128,  64, 8, 4); break;                        \
        case 4: LAUNCH_TILE(KERN,  64, 128, 4, 8); break;                        \
        case 5: LAUNCH_TILE(KERN, 128, 128, 8, 8); break;                        \
        case 6: LAUNCH_TILE(KERN, 128, 128, 4, 4); break;                        \
        case 7: LAUNCH_TILE(KERN, 128, 128, 8, 4); break;                        \
        default: LAUNCH_TILE(KERN, 64, 64, 4, 4);                                \
    } } while (0)

torch::Tensor wonly_gemm_cuda(
        torch::Tensor X, torch::Tensor scale_exp,
        torch::Tensor upper, torch::Tensor shared,
        int64_t M, int64_t OUT, int64_t K, int64_t NB, int64_t u, int64_t gs) {
    TORCH_CHECK(X.is_cuda() && X.scalar_type() == torch::kBFloat16, "X must be CUDA bf16");
    int UB, SB; gemm_dims(u, gs, UB, SB);
    auto Y = torch::empty({M, OUT}, X.options());
#define ARGS \
    reinterpret_cast<const __nv_bfloat16*>(X.data_ptr<at::BFloat16>()), \
    scale_exp.data_ptr<int8_t>(), upper.data_ptr<uint8_t>(), shared.data_ptr<uint8_t>(), \
    reinterpret_cast<__nv_bfloat16*>(Y.data_ptr<at::BFloat16>()), \
    (int)M, (int)OUT, (int)K, (int)NB, (int)u, (int)gs, UB, SB
    if (tile_cfg((int)M) == 10)     // BF16 tensor-core (WMMA) path
        wonly_gemm_wmma<<<dim3(((int)OUT + 63) / 64, ((int)M + 63) / 64), 128>>>(ARGS);
    else
        DISPATCH_TILE(wonly_gemm_tiled);
#undef ARGS
    return Y;
}

torch::Tensor wa_gemm_cuda(
        torch::Tensor X, torch::Tensor scale_exp,
        torch::Tensor upper, torch::Tensor shared,
        int64_t M, int64_t OUT, int64_t K, int64_t NB, int64_t u, int64_t gs) {
    TORCH_CHECK(X.is_cuda() && X.scalar_type() == torch::kBFloat16, "X must be CUDA bf16");
    int UB, SB; gemm_dims(u, gs, UB, SB);
    auto Y = torch::empty({M, OUT}, X.options());
#define ARGS \
    reinterpret_cast<const __nv_bfloat16*>(X.data_ptr<at::BFloat16>()), \
    scale_exp.data_ptr<int8_t>(), upper.data_ptr<uint8_t>(), shared.data_ptr<uint8_t>(), \
    reinterpret_cast<__nv_bfloat16*>(Y.data_ptr<at::BFloat16>()), \
    (int)M, (int)OUT, (int)K, (int)NB, (int)u, (int)gs, UB, SB
    DISPATCH_TILE(wa_gemm_tiled);
#undef ARGS
    return Y;
}
