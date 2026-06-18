// csrc/mxint8.cu  —  plain-MXINT8 baseline kernels.
//
// The matched-optimization baseline for the MSAQ kernels: identical structure
// (thread mapping, FP32 accumulate, on-the-fly activation quant, online
// softmax), differing ONLY in the weight/KV read. MSAQ does the sub-byte unpack
// (ms::unpack_ms_weight_elem); MXINT8 reads the int8 mantissa directly. So the
// MSAQ-vs-MXINT8 latency delta isolates exactly the unpack overhead vs the
// fewer-bytes-read benefit — the decode-memory-bound question.
//
// Layout (out-innermost SoA, from ms_lib.pack.pack_weight_mxint8):
//   scale_exp [nb, OUT] int8           flat: blk*OUT + o
//   qweight   [nb, 32, OUT] int8       flat: (blk*32 + k)*OUT + o
// KV (pack_kv_mxint8): scale_exp [H,nb,L], qweight [H,nb,32,L].
//
// STATUS: GPU-UNVALIDATED until built; logic verified on CPU
// (tests/test_emulation.py MXINT8 cases). Same correctness-first level as the
// MSAQ kernels (no tensor cores / split-K yet) so the comparison is fair.

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <mma.h>
#include <math.h>
#include "core/ms_utils.cuh"

// Stage-0 activation quant launcher (defined in wa_gemm.cu) — shared so the MXINT8
// W+A path quantizes the activation identically to MSAQ (matched comparison).
void ms_launch_quant_act(const __nv_bfloat16* X, int8_t* qX, int8_t* sa_exp, int M, int K, int NB);

namespace {
constexpr int BLOCK = 32;

// ---- W-only decode GEMV (direct int8) — SPLIT-K, matched to w_gemv.cu --------
__global__ void mxint8_gemv_splitk_kernel(
        const __nv_bfloat16* __restrict__ x,
        const int8_t* __restrict__ scale_exp,    // [NB, OUT]
        const int8_t* __restrict__ qweight,      // [NB, 32, OUT]
        float* __restrict__ partial,             // [splitK, OUT]
        int OUT, int NB, int splitK) {
    const int o  = blockIdx.x * blockDim.x + threadIdx.x;
    const int sp = blockIdx.y;
    if (o >= OUT) return;
    const int per = (NB + splitK - 1) / splitK;
    const int b0 = sp * per, b1 = min(b0 + per, NB);
    float acc = 0.0f;
    for (int blk = b0; blk < b1; ++blk) {
        const float scale = ms::e8m0_to_scale(scale_exp[blk * OUT + o]);
        #pragma unroll
        for (int k = 0; k < BLOCK; ++k) {
            const int w = qweight[(blk * BLOCK + k) * OUT + o];   // direct int8
            const float xv = __bfloat162float(x[blk * BLOCK + k]);
            acc += (static_cast<float>(w) * scale) * xv;
        }
    }
    partial[(long)sp * OUT + o] = acc;
}

__global__ void mxint8_gemv_combine_kernel(
        const float* __restrict__ partial, __nv_bfloat16* __restrict__ y,
        int OUT, int splitK) {
    const int o = blockIdx.x * blockDim.x + threadIdx.x;
    if (o >= OUT) return;
    float acc = 0.0f;
    for (int sp = 0; sp < splitK; ++sp) acc += partial[(long)sp * OUT + o];
    y[o] = __float2bfloat16(acc);
}

// ---- W-only prefill GEMM — SHARED-MEMORY TILED, matched to wa_gemm.cu --------
//   Same tiling as the MSAQ kernel; only the B-tile load differs (direct int8 vs
//   unpack), keeping the comparison matched.
constexpr int TBK = BLOCK;   // K-tile = one MSAQ block
template<int TBM, int TBN, int RTM, int RTN>
__global__ void mxint8_gemm_tiled(
        const __nv_bfloat16* __restrict__ X,
        const int8_t* __restrict__ scale_exp,
        const int8_t* __restrict__ qweight,
        __nv_bfloat16* __restrict__ Y,
        int M, int OUT, int K, int NB) {
    constexpr int TNT = TBN / RTN;
    constexpr int NT  = (TBM / RTM) * TNT;
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
            if (o < OUT)
                v = (float)qweight[(blk * BLOCK + k) * OUT + o]
                  * ms::e8m0_to_scale(scale_exp[blk * OUT + o]);
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

// ---- W-only prefill via BF16 TENSOR CORES (WMMA) — matched to wonly_gemm_wmma -
//   Identical to csrc/wa_gemm.cu::wonly_gemm_wmma; only the B-tile load differs
//   (direct int8 -> bf16 here vs the MSAQ sub-byte unpack), keeping the tensor-
//   core comparison matched. Opt-in via MS_TILE_CFG=10.
using namespace nvcuda;
constexpr int WSK = BLOCK + 8;          // shared K width (+pad for bank conflicts)
__global__ void mxint8_gemm_wmma(
        const __nv_bfloat16* __restrict__ X,
        const int8_t* __restrict__ scale_exp,
        const int8_t* __restrict__ qweight,
        __nv_bfloat16* __restrict__ Y,
        int M, int OUT, int K, int NB) {
    __shared__ __nv_bfloat16 As[64][WSK];
    __shared__ __nv_bfloat16 Bs[64][WSK];
    __shared__ float tmp[4][16][16];
    const int m0 = blockIdx.y * 64, o0 = blockIdx.x * 64;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int wM = warp >> 1, wN = warp & 1;
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
            if (o < OUT) v = (float)qweight[(blk * BLOCK + k) * OUT + o]
                           * ms::e8m0_to_scale(scale_exp[blk * OUT + o]);
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

// ---- W-only WMMA, SOFTWARE-PIPELINED — matched to wonly_gemm_wmma_pipe --------
//   Same double-buffered overlap (next tile staged while current tile's MMA runs);
//   only the B-tile load differs (direct int8 vs MSAQ unpack). Opt-in MS_TILE_CFG=11.
__global__ void mxint8_gemm_wmma_pipe(
        const __nv_bfloat16* __restrict__ X, const int8_t* __restrict__ scale_exp,
        const int8_t* __restrict__ qweight, __nv_bfloat16* __restrict__ Y,
        int M, int OUT, int K, int NB) {
    __shared__ __nv_bfloat16 As[2][64][WSK];
    __shared__ __nv_bfloat16 Bs[2][64][WSK];
    __shared__ float tmp[4][16][16];
    const int m0 = blockIdx.y * 64, o0 = blockIdx.x * 64;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int wM = warp >> 1, wN = warp & 1;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[2][2];
    #pragma unroll
    for (int i = 0; i < 2; ++i)
        #pragma unroll
        for (int j = 0; j < 2; ++j) wmma::fill_fragment(acc[i][j], 0.0f);

    auto stage = [&](int b, int buf) {
        for (int idx = tid; idx < 64 * BLOCK; idx += 128) {
            const int m = idx / BLOCK, k = idx % BLOCK;
            As[buf][m][k] = (m0 + m < M) ? X[(long)(m0 + m) * K + b * BLOCK + k]
                                         : __float2bfloat16(0.0f);
        }
        for (int idx = tid; idx < 64 * BLOCK; idx += 128) {
            const int oc = idx / BLOCK, k = idx % BLOCK, o = o0 + oc;
            float v = 0.0f;
            if (o < OUT) v = (float)qweight[(b * BLOCK + k) * OUT + o]
                           * ms::e8m0_to_scale(scale_exp[b * OUT + o]);
            Bs[buf][oc][k] = __float2bfloat16(v);
        }
    };

    stage(0, 0);
    __syncthreads();
    for (int blk = 0; blk < NB; ++blk) {
        const int cur = blk & 1, nxt = cur ^ 1;
        #pragma unroll
        for (int wk = 0; wk < BLOCK / 16; ++wk) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> a[2];
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> b[2];
            #pragma unroll
            for (int i = 0; i < 2; ++i) wmma::load_matrix_sync(a[i], &As[cur][wM*32 + i*16][wk*16], WSK);
            #pragma unroll
            for (int j = 0; j < 2; ++j) wmma::load_matrix_sync(b[j], &Bs[cur][wN*32 + j*16][wk*16], WSK);
            #pragma unroll
            for (int i = 0; i < 2; ++i)
                #pragma unroll
                for (int j = 0; j < 2; ++j) wmma::mma_sync(acc[i][j], a[i], b[j], acc[i][j]);
        }
        if (blk + 1 < NB) stage(blk + 1, nxt);
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

// ---- W+A pure-int8 IMMA GEMM (activation pre-quantized) — matched to wa_imma --
//   Same 2-stage structure; B-stage reads int8 qweight DIRECTLY (no unpack), so
//   the MSAQ/MXINT8 delta is exactly the weight unpack. (change.md P26.)
__global__ void mxint8_wa_imma(
        const int8_t* __restrict__ qX, const int8_t* __restrict__ sa_exp,
        const int8_t* __restrict__ scale_exp, const int8_t* __restrict__ qweight,
        __nv_bfloat16* __restrict__ Y, int M, int OUT, int K, int NB) {
    __shared__ int8_t  As[2][64][32];
    __shared__ int8_t  Bs[2][64][32];
    __shared__ int32_t Cs[64][64];
    __shared__ float   sa_s[2][64], sw_s[2][64];
    const int m0 = blockIdx.y * 64, o0 = blockIdx.x * 64;
    const int tid = threadIdx.x, warp = tid >> 5, wM = warp >> 1, wN = warp & 1;
    float accf[32];
    #pragma unroll
    for (int j = 0; j < 32; ++j) accf[j] = 0.0f;

    auto stage = [&](int blk, int buf) {
        for (int idx = tid; idx < 64 * BLOCK; idx += 128) {
            const int m = idx >> 5, k = idx & 31;
            As[buf][m][k] = (m0 + m < M) ? qX[(long)(m0 + m) * K + blk * BLOCK + k] : (int8_t)0;
        }
        for (int idx = tid; idx < 64 * BLOCK; idx += 128) {       // B-stage: direct int8
            const int oc = idx >> 5, k = idx & 31, o = o0 + oc;
            Bs[buf][oc][k] = (o < OUT) ? qweight[(long)(blk * BLOCK + k) * OUT + o] : (int8_t)0;
        }
        if (tid < 64) {
            const int o = o0 + tid;
            sw_s[buf][tid] = (o < OUT) ? ms::e8m0_to_scale(scale_exp[blk * OUT + o]) : 0.0f;
            sa_s[buf][tid] = (m0 + tid < M) ? ms::e8m0_to_scale(sa_exp[(long)(m0 + tid) * NB + blk]) : 0.0f;
        }
    };

    stage(0, 0);
    __syncthreads();
    for (int blk = 0; blk < NB; ++blk) {
        const int cur = blk & 1, nxt = cur ^ 1;
        wmma::fragment<wmma::accumulator, 16, 16, 16, int32_t> c[2][2];
        #pragma unroll
        for (int i = 0; i < 2; ++i)
            #pragma unroll
            for (int j = 0; j < 2; ++j) wmma::fill_fragment(c[i][j], 0);
        #pragma unroll
        for (int wk = 0; wk < 2; ++wk) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, int8_t, wmma::row_major> a[2];
            wmma::fragment<wmma::matrix_b, 16, 16, 16, int8_t, wmma::col_major> b[2];
            #pragma unroll
            for (int i = 0; i < 2; ++i) wmma::load_matrix_sync(a[i], &As[cur][wM*32 + i*16][wk*16], 32);
            #pragma unroll
            for (int j = 0; j < 2; ++j) wmma::load_matrix_sync(b[j], &Bs[cur][wN*32 + j*16][wk*16], 32);
            #pragma unroll
            for (int i = 0; i < 2; ++i)
                #pragma unroll
                for (int j = 0; j < 2; ++j) wmma::mma_sync(c[i][j], a[i], b[j], c[i][j]);
        }
        if (blk + 1 < NB) stage(blk + 1, nxt);
        #pragma unroll
        for (int i = 0; i < 2; ++i)
            #pragma unroll
            for (int j = 0; j < 2; ++j)
                wmma::store_matrix_sync(&Cs[wM*32 + i*16][wN*32 + j*16], c[i][j], 64, wmma::mem_row_major);
        __syncthreads();
        #pragma unroll
        for (int j = 0; j < 32; ++j) {
            const int e = tid + 128 * j, m = e >> 6, o = e & 63;
            accf[j] += (float)Cs[m][o] * sa_s[cur][m] * sw_s[cur][o];
        }
        __syncthreads();
    }
    #pragma unroll
    for (int j = 0; j < 32; ++j) {
        const int e = tid + 128 * j, m = m0 + (e >> 6), o = o0 + (e & 63);
        if (m < M && o < OUT) Y[(long)m * OUT + o] = __float2bfloat16(accf[j]);
    }
}

// ---- W+A GEMM (int8 weight direct + on-the-fly MXINT8 activation, int dot) ---
// SHARED-MEMORY TILED W+A, matched to wa_gemm.cu (only the B-tile differs:
// direct int8 vs unpack). As folds sa (activation), Bs folds sw (weight).
template<int TBM, int TBN, int RTM, int RTN>
__global__ void mxint8_wa_gemm_tiled(
        const __nv_bfloat16* __restrict__ X,
        const int8_t* __restrict__ scale_exp,
        const int8_t* __restrict__ qweight,
        __nv_bfloat16* __restrict__ Y,
        int M, int OUT, int K, int NB) {
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
            if (o < OUT)
                v = (float)qweight[(blk * BLOCK + k) * OUT + o]
                  * ms::e8m0_to_scale(scale_exp[blk * OUT + o]);
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

// Tile dispatch — same M-adaptive optimum as wa_gemm.cu (matched).
static inline int tile_cfg(int M) {
    if (const char* e = getenv("MS_TILE_CFG")) { int c = atoi(e); if (c >= 0) return c; }
    return (M >= 256) ? 5 : 1;
}
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

// ---- KV-cache flash-decode (direct int8 K/V read) ---------------------------
// SPLIT-KV / FLASH-DECODING, identical structure to csrc/kv_attention.cu so the
// MSAQ-vs-MXINT8 comparison stays matched-optimization (only the K/V read
// differs: direct int8 here vs the sub-byte unpack in MSAQ). grid = (H, S), with
// S/key_tile chosen from the live SM count (ms::kv_split_count).

constexpr int KV_CHUNK = 128;   // keys per chunk (must match kv_attention.cu)

__device__ __forceinline__ float warp_reduce_sum(float v) {
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffffu, v, o);
    return v;
}

// Two-pass barrier-light flash-decode (방안 3), identical structure to
// csrc/kv_attention.cu::kv_decode_split_kernel; only the K/V read differs
// (direct int8 vs sub-byte unpack), keeping the comparison matched.
__global__ void mxint8_kv_split_kernel(
        const __nv_bfloat16* __restrict__ q,
        const int8_t* __restrict__ ks, const int8_t* __restrict__ kq,  // K: scale[H,nb,L], q[H,nb,L,32]
        const int8_t* __restrict__ vs, const int8_t* __restrict__ vq,  // V
        float* __restrict__ part_o, float* __restrict__ part_m, float* __restrict__ part_l,
        int H, int Lk, int D, int NB, int key_tile, int S, float sm_scale) {
    const int h = blockIdx.x;
    const int s = blockIdx.y;
    const int tid = threadIdx.x;
    const int lane = tid & 31, warpId = tid >> 5, nWarps = blockDim.x >> 5;
    const bool active = tid < D;

    extern __shared__ float smem[];
    float* q_sh = smem;                         // [D]
    float* sc   = smem + D;                      // [KV_CHUNK]
    if (active) q_sh[tid] = __bfloat162float(q[h * D + tid]);
    __syncthreads();

    const int j0 = s * key_tile;
    const int j1 = min(j0 + key_tile, Lk);
    float m_i = -INFINITY, l_i = 0.0f, acc = 0.0f;

    for (int cs = j0; cs < j1; cs += KV_CHUNK) {
        const int nC = min(KV_CHUNK, j1 - cs);

        // ---- Pass 1: scores (warp per key, __shfl reduction) ----
        for (int kk = warpId; kk < nC; kk += nWarps) {
            const int j = cs + kk;
            float part = 0.0f;
            for (int d = lane; d < D; d += 32) {
                const int blk = d / BLOCK, kd = d % BLOCK;
                const long qbase = (long)(h * NB + blk) * BLOCK * Lk;   // [H,nb,L,32]
                const long sbase = (long)(h * NB + blk) * Lk;
                part += q_sh[d] * (float)kq[qbase + (long)j * BLOCK + kd]
                                * ms::e8m0_to_scale(ks[sbase + j]);
            }
            part = warp_reduce_sum(part);
            if (lane == 0) sc[kk] = part * sm_scale;
        }
        __syncthreads();

        float m_chunk = -INFINITY;
        for (int kk = 0; kk < nC; ++kk) m_chunk = fmaxf(m_chunk, sc[kk]);
        const float m_new = fmaxf(m_i, m_chunk);
        const float alpha = expf(m_i - m_new);

        // ---- Pass 2: out[d] += Σ_kk p_kk·V[d,kk] ----
        const int blk = tid / BLOCK, kd = tid % BLOCK;
        const long qbase = (long)(h * NB + blk) * BLOCK * Lk;
        const long sbase = (long)(h * NB + blk) * Lk;
        float lsum = 0.0f, a = 0.0f;
        for (int kk = 0; kk < nC; ++kk) {
            const float p = expf(sc[kk] - m_new);
            lsum += p;
            if (active) {
                const int j = cs + kk;
                a += p * (float)vq[qbase + (long)j * BLOCK + kd]
                       * ms::e8m0_to_scale(vs[sbase + j]);
            }
        }
        l_i = l_i * alpha + lsum;
        acc = acc * alpha + a;
        m_i = m_new;
        __syncthreads();
    }
    if (active) part_o[((long)h * S + s) * D + tid] = acc;
    if (tid == 0) { part_m[h * S + s] = m_i; part_l[h * S + s] = l_i; }
}

__global__ void mxint8_kv_combine_kernel(
        const float* __restrict__ part_o, const float* __restrict__ part_m,
        const float* __restrict__ part_l, __nv_bfloat16* __restrict__ out,
        int H, int D, int S) {
    const int h = blockIdx.x;
    const int e = threadIdx.x;
    if (e >= D) return;
    float m_g = -INFINITY;
    for (int s = 0; s < S; ++s) m_g = fmaxf(m_g, part_m[h * S + s]);
    float l = 0.0f, acc = 0.0f;
    for (int s = 0; s < S; ++s) {
        const float w = expf(part_m[h * S + s] - m_g);
        l   += part_l[h * S + s] * w;
        acc += part_o[((long)h * S + s) * D + e] * w;
    }
    out[h * D + e] = __float2bfloat16(acc / l);
}

inline int next_pow2(int n) { int p = 1; while (p < n) p <<= 1; return p; }

// ---- KV WRITE (MXINT8 baseline), matched to kv_write_kernel ------------------
//   Same thread-per-token structure; quantize 32 head_dim to int8 directly (no
//   bit-pack), store to qweight [H,nb,L,32]. (change.md Phase 28.)
__global__ void mxint8_kv_write_kernel(
        const __nv_bfloat16* __restrict__ X,    // [H, L, D]
        int8_t* __restrict__ scale_exp,          // [H, nb, L]
        int8_t* __restrict__ qweight,            // [H, nb, L, 32]
        int H, int L, int D, int NB) {
    const int h = blockIdx.x;
    const int j = blockIdx.y * blockDim.x + threadIdx.x;
    if (j >= L) return;
    for (int blk = 0; blk < NB; ++blk) {
        const long xb = ((long)h * L + j) * D + (long)blk * BLOCK;
        float amax = 1e-30f;
        #pragma unroll
        for (int k = 0; k < BLOCK; ++k) amax = fmaxf(amax, fabsf(__bfloat162float(X[xb + k])));
        const int ea = ms::e8m0_exp_from_amax(amax);
        const float s = exp2f((float)ea);
        const long tok = (long)(h * NB + blk) * L + j;
        #pragma unroll
        for (int k = 0; k < BLOCK; ++k) {
            int q = (int)rintf(__bfloat162float(X[xb + k]) / s);
            qweight[tok * BLOCK + k] = (int8_t)max(-127, min(127, q));
        }
        scale_exp[tok] = (int8_t)ea;
    }
}

// Matched MXINT8 decode-append: quantize one new token X[H,D] into qweight cache
// at slot `pos` (stride Lcap), in place. thread = (h,blk). (change.md Phase 28.)
__global__ void mxint8_kv_append_kernel(
        const __nv_bfloat16* __restrict__ X,    // [H, D]
        int8_t* __restrict__ scale_exp,          // [H, nb, Lcap]
        int8_t* __restrict__ qweight,            // [H, nb, Lcap, 32]
        int H, int D, int NB, int pos, int Lcap) {
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= H * NB) return;
    const int h = t / NB, blk = t % NB;
    const long xb = (long)h * D + (long)blk * BLOCK;
    float amax = 1e-30f;
    #pragma unroll
    for (int k = 0; k < BLOCK; ++k) amax = fmaxf(amax, fabsf(__bfloat162float(X[xb + k])));
    const int ea = ms::e8m0_exp_from_amax(amax);
    const float s = exp2f((float)ea);
    const long slot = (long)(h * NB + blk) * Lcap + pos;
    #pragma unroll
    for (int k = 0; k < BLOCK; ++k) {
        int q = (int)rintf(__bfloat162float(X[xb + k]) / s);
        qweight[slot * BLOCK + k] = (int8_t)max(-127, min(127, q));
    }
    scale_exp[slot] = (int8_t)ea;
}
} // namespace

// ---- host launchers ---------------------------------------------------------
torch::Tensor mxint8_gemv_cuda(
        torch::Tensor x, torch::Tensor scale_exp, torch::Tensor qweight,
        int64_t OUT, int64_t NB) {
    auto y = torch::empty({OUT}, x.options());
    const int threads = 128, blocks = (int)((OUT + threads - 1) / threads);
    const int splitK = ms::gemv_splitk_count(blocks, (int)NB);
    auto partial = torch::empty({(int64_t)splitK, OUT},
                                x.options().dtype(torch::kFloat32));
    mxint8_gemv_splitk_kernel<<<dim3(blocks, splitK), threads>>>(
        reinterpret_cast<const __nv_bfloat16*>(x.data_ptr<at::BFloat16>()),
        scale_exp.data_ptr<int8_t>(), qweight.data_ptr<int8_t>(),
        partial.data_ptr<float>(), (int)OUT, (int)NB, splitK);
    mxint8_gemv_combine_kernel<<<blocks, threads>>>(
        partial.data_ptr<float>(),
        reinterpret_cast<__nv_bfloat16*>(y.data_ptr<at::BFloat16>()), (int)OUT, splitK);
    return y;
}

torch::Tensor mxint8_gemm_cuda(
        torch::Tensor X, torch::Tensor scale_exp, torch::Tensor qweight,
        int64_t M, int64_t OUT, int64_t K, int64_t NB) {
    auto Y = torch::empty({M, OUT}, X.options());
#define ARGS \
    reinterpret_cast<const __nv_bfloat16*>(X.data_ptr<at::BFloat16>()), \
    scale_exp.data_ptr<int8_t>(), qweight.data_ptr<int8_t>(), \
    reinterpret_cast<__nv_bfloat16*>(Y.data_ptr<at::BFloat16>()), \
    (int)M, (int)OUT, (int)K, (int)NB
    if (tile_cfg((int)M) == 11)     // BF16 WMMA software-pipelined, matched to MSAQ
        mxint8_gemm_wmma_pipe<<<dim3(((int)OUT + 63) / 64, ((int)M + 63) / 64), 128>>>(ARGS);
    else if (tile_cfg((int)M) == 10)  // BF16 tensor-core (WMMA), matched to MSAQ
        mxint8_gemm_wmma<<<dim3(((int)OUT + 63) / 64, ((int)M + 63) / 64), 128>>>(ARGS);
    else
        DISPATCH_TILE(mxint8_gemm_tiled);
#undef ARGS
    return Y;
}

// W+A MXINT8 = STAGE 0 (shared quant) + STAGE 1 (int8 IMMA), matched to wa_gemm.
torch::Tensor mxint8_wa_gemm_cuda(
        torch::Tensor X, torch::Tensor scale_exp, torch::Tensor qweight,
        int64_t M, int64_t OUT, int64_t K, int64_t NB) {
    auto Y = torch::empty({M, OUT}, X.options());
    const char* f = getenv("MS_WA_FOLD");
    if (f && atoi(f) == 1) {        // legacy fused FP32-fold path (A/B reference)
#define ARGS \
        reinterpret_cast<const __nv_bfloat16*>(X.data_ptr<at::BFloat16>()), \
        scale_exp.data_ptr<int8_t>(), qweight.data_ptr<int8_t>(), \
        reinterpret_cast<__nv_bfloat16*>(Y.data_ptr<at::BFloat16>()), \
        (int)M, (int)OUT, (int)K, (int)NB
        DISPATCH_TILE(mxint8_wa_gemm_tiled);
#undef ARGS
        return Y;
    }
    auto qX = torch::empty({M, K}, X.options().dtype(torch::kInt8));
    auto sa = torch::empty({M, NB}, X.options().dtype(torch::kInt8));
    ms_launch_quant_act(reinterpret_cast<const __nv_bfloat16*>(X.data_ptr<at::BFloat16>()),
                        qX.data_ptr<int8_t>(), sa.data_ptr<int8_t>(), (int)M, (int)K, (int)NB);
    mxint8_wa_imma<<<dim3(((int)OUT + 63) / 64, ((int)M + 63) / 64), 128>>>(
        qX.data_ptr<int8_t>(), sa.data_ptr<int8_t>(), scale_exp.data_ptr<int8_t>(),
        qweight.data_ptr<int8_t>(), reinterpret_cast<__nv_bfloat16*>(Y.data_ptr<at::BFloat16>()),
        (int)M, (int)OUT, (int)K, (int)NB);
    return Y;
}

torch::Tensor mxint8_kv_decode_cuda(
        torch::Tensor q, torch::Tensor ks, torch::Tensor kq,
        torch::Tensor vs, torch::Tensor vq,
        int64_t H, int64_t Lk, int64_t D, int64_t NB) {
    auto out = torch::empty({H, D}, q.options());
    const int threads = next_pow2((int)D);
    const size_t smem = (size_t)((int)D + KV_CHUNK) * sizeof(float);  // q_sh[D] + sc[CHUNK]
    const float sm = 1.0f / sqrtf((float)D);

    const int S = ms::kv_split_count((long)Lk, (int)H);
    const int key_tile = (int)((Lk + S - 1) / S);
    auto fopt = q.options().dtype(torch::kFloat32);
    auto part_o = torch::empty({H, (int64_t)S, D}, fopt);
    auto part_m = torch::empty({H, (int64_t)S}, fopt);
    auto part_l = torch::empty({H, (int64_t)S}, fopt);

    mxint8_kv_split_kernel<<<dim3((int)H, S), threads, smem>>>(
        reinterpret_cast<const __nv_bfloat16*>(q.data_ptr<at::BFloat16>()),
        ks.data_ptr<int8_t>(), kq.data_ptr<int8_t>(),
        vs.data_ptr<int8_t>(), vq.data_ptr<int8_t>(),
        part_o.data_ptr<float>(), part_m.data_ptr<float>(), part_l.data_ptr<float>(),
        (int)H, (int)Lk, (int)D, (int)NB, key_tile, S, sm);

    mxint8_kv_combine_kernel<<<(int)H, threads>>>(
        part_o.data_ptr<float>(), part_m.data_ptr<float>(), part_l.data_ptr<float>(),
        reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>()),
        (int)H, (int)D, S);
    return out;
}

// MXINT8 KV write launcher -> (scale_exp [H,nb,L], qweight [H,nb,L,32]).
std::vector<torch::Tensor> mxint8_kv_write_cuda(
        torch::Tensor X, int64_t H, int64_t L, int64_t D, int64_t NB) {
    TORCH_CHECK(X.is_cuda() && X.scalar_type() == torch::kBFloat16, "X must be CUDA bf16");
    auto i8 = X.options().dtype(torch::kInt8);
    auto scale_exp = torch::empty({H, NB, L}, i8);
    auto qweight   = torch::empty({H, NB, L, BLOCK}, i8);
    const int TPB = 256;
    mxint8_kv_write_kernel<<<dim3((int)H, ((int)L + TPB - 1) / TPB), TPB>>>(
        reinterpret_cast<const __nv_bfloat16*>(X.data_ptr<at::BFloat16>()),
        scale_exp.data_ptr<int8_t>(), qweight.data_ptr<int8_t>(), (int)H, (int)L, (int)D, (int)NB);
    return {scale_exp, qweight};
}

// Matched MXINT8 decode-append launcher (mutates scale_exp/qweight in place).
void mxint8_kv_append_cuda(
        torch::Tensor X, torch::Tensor scale_exp, torch::Tensor qweight,
        int64_t H, int64_t D, int64_t NB, int64_t pos, int64_t Lcap) {
    TORCH_CHECK(X.is_cuda() && X.scalar_type() == torch::kBFloat16, "X must be CUDA bf16");
    const int total = (int)(H * NB), TPB = 128;
    mxint8_kv_append_kernel<<<(total + TPB - 1) / TPB, TPB>>>(
        reinterpret_cast<const __nv_bfloat16*>(X.data_ptr<at::BFloat16>()),
        scale_exp.data_ptr<int8_t>(), qweight.data_ptr<int8_t>(),
        (int)H, (int)D, (int)NB, (int)pos, (int)Lcap);
}
