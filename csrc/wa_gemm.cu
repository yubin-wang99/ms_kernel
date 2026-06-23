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
#include <ATen/cuda/CUDAContext.h>

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

// ---- STAGE 0: runtime activation quantization (pre-pass, memory-bound) -------
//   X bf16 [M,K] -> qX int8 [M,K] + sa_exp int8 [M,nb]. ONE WARP per (m,blk):
//   lane k loads x, warp-reduce amax, E8M0 exp = floor(log2 amax) - E_MAX, then
//   qX = clip(round(x * 2^-exp), +-127). Each element quantized EXACTLY ONCE (vs
//   the fused kernel re-quantizing every X row OUT/TBN times). Shared by the MSAQ
//   and MXINT8 Stage-1 GEMMs so the comparison stays matched. (change.md P26.)
__global__ void quant_act_kernel(
        const __nv_bfloat16* __restrict__ X, int8_t* __restrict__ qX,
        int8_t* __restrict__ sa_exp, int M, int K, int NB) {
    const int wpb = blockDim.x >> 5;
    const int gw  = blockIdx.x * wpb + (threadIdx.x >> 5);   // global warp -> (m,blk)
    if (gw >= M * NB) return;
    const int m = gw / NB, blk = gw % NB, lane = threadIdx.x & 31;
    const long base = (long)m * K + (long)blk * BLOCK;       // BLOCK == warp width
    const float x = __bfloat162float(X[base + lane]);
    float a = fabsf(x);
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, o));
    int ea = (int)floorf(log2f(fmaxf(a, 1e-30f))) - (int)ms::E_MAX;
    ea = max(-127, min(127, ea));
    int q = (int)rintf(x * exp2f(-(float)ea));
    qX[base + lane] = (int8_t)max(-127, min(127, q));
    if (lane == 0) sa_exp[(long)m * NB + blk] = (int8_t)ea;
}

// ---- STAGE 0 (MSAQ-s): activation gets the SAME mantissa-sharing format as the
//   weight (NOT plain MXINT8). Per 32-block: E8M0 base sa; upper code q_upper =
//   clip(round(x / (sa*2^u))); residual r = x - q_upper*sa*2^u; shared code (one
//   per gs-group) r_shared = clip(round(mean_g(r) / sa)); int8 word qx =
//   q_upper*2^u + r_shared (gs elems share r_shared). Scale stays sa (base) — the
//   sa[m]*sw[o] combine is unchanged. Bit-exact to reference.quant_act(share=True)
//   / pack.decompose. The MXINT8 baseline keeps the plain-MXINT8 quant above:
//   this is a FORMAT DIFFERENCE (MSAQ-s vs MXINT8 activation), not an optimization,
//   so it is NOT mirrored into the baseline. (change.md P27.)
__global__ void quant_act_msaq_kernel(
        const __nv_bfloat16* __restrict__ X, int8_t* __restrict__ qX,
        int8_t* __restrict__ sa_exp, int M, int K, int NB, int u, int gs) {
    const int wpb = blockDim.x >> 5;
    const int gw  = blockIdx.x * wpb + (threadIdx.x >> 5);
    if (gw >= M * NB) return;
    const int m = gw / NB, blk = gw % NB, lane = threadIdx.x & 31;   // lane == k in [0,32)
    const long base = (long)m * K + (long)blk * BLOCK;
    const float x = __bfloat162float(X[base + lane]);
    float a = fabsf(x);                                          // 1. E8M0 base scale sa
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, o));
    int ea = (int)floorf(log2f(fmaxf(a, 1e-30f))) - (int)ms::E_MAX;
    ea = max(-127, min(127, ea));
    const float sa = exp2f((float)ea);
    const float s_unshared = sa * exp2f((float)u);              // 2. coarse scale sa*2^u
    const int qmax = (1 << (7 - u)) - 1;                        // 3. upper code (8-u bit signed)
    const int q_upper = max(-qmax, min(qmax, (int)rintf(x / s_unshared)));
    const float r = x - (float)q_upper * s_unshared;            // 4. residual
    float rs = r;                                              // 5. mean over the gs-group
    for (int o = 1; o < gs; o <<= 1) rs += __shfl_xor_sync(0xffffffffu, rs, o);
    const float res_avg = rs / (float)gs;
    const int smin = -(1 << (u - 1)), smax = (1 << (u - 1)) - 1; // 6. shared code (u bit signed)
    const int r_shared = max(smin, min(smax, (int)rintf(res_avg / sa)));
    qX[base + lane] = (int8_t)(q_upper * (1 << u) + r_shared);   // 7. int8 word
    if (lane == 0) sa_exp[(long)m * NB + blk] = (int8_t)ea;      // 8. scale = base sa
}

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

// COLUMN-MAJOR wide-load W-only prefill GEMM. Same tiling/GEMM as wonly_gemm_tiled,
// but the Bs (weight) stage reads the COLUMN-MAJOR planes upper_cm/shared_cm [nb,OUT,*]:
// each thread owns whole columns and wide-loads a column's UB bytes (contiguous; adjacent
// columns are UB apart -> the warp is coalesced), then STREAMING-unpacks all 32 codes from
// registers. Replaces the per-element random-access extract_code (sector-util ~12%, the
// L1-load bottleneck of the row-major path) with one coalesced wide load per column.
template<int TBM, int TBN, int RTM, int RTN>
__global__ void wonly_gemm_tiled_cm(
        const __nv_bfloat16* __restrict__ X,
        const int8_t*  __restrict__ scale_exp,
        const uint8_t* __restrict__ upper_cm,    // [nb, OUT, UB]
        const uint8_t* __restrict__ shared_cm,   // [nb, OUT, SB]
        __nv_bfloat16* __restrict__ Y,
        int M, int OUT, int K, int NB, int u, int gs, int UB, int SB) {
    constexpr int TNT = TBN / RTN;
    constexpr int NT  = (TBM / RTM) * TNT;
    __shared__ float As[TBK][TBM];
    __shared__ float Bs[TBK][TBN];
    const int m0 = blockIdx.y * TBM, o0 = blockIdx.x * TBN;
    const int tid = threadIdx.x, tRow = tid / TNT, tCol = tid % TNT;
    const int wbits = 8 - u, gsmask = gs - 1;
    const uint32_t umask = (1u << wbits) - 1u, usign = 1u << (wbits - 1);
    const uint32_t smask = (1u << u) - 1u, ssign = 1u << (u - 1);

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
        // per-column wide-load + streaming unpack -> Bs[0..31][oc]
        for (int oc = tid; oc < TBN; oc += NT) {
            const int o = o0 + oc;
            if (o >= OUT) {
                #pragma unroll
                for (int k = 0; k < TBK; ++k) Bs[k][oc] = 0.0f;
                continue;
            }
            const float sc = ms::e8m0_to_scale(scale_exp[blk * OUT + o]);
            const long base = (long)blk * OUT + o;
            uint32_t ureg[6];
            const uint32_t* src = reinterpret_cast<const uint32_t*>(upper_cm + base * UB);
            #pragma unroll
            for (int i = 0; i < 6; ++i) if (i < (UB >> 2)) ureg[i] = src[i];
            uint32_t sreg[3] = {0u, 0u, 0u};
            #pragma unroll
            for (int i = 0; i < 8; ++i) if (i < SB) sreg[i >> 2] |= (uint32_t)shared_cm[base * SB + i] << (8 * (i & 3));
            uint64_t ubuf = 0; int unb = 0, uwi = 0;
            uint64_t sbuf = 0; int snb = 0, swi = 0; int sh_code = 0;
            #pragma unroll
            for (int k = 0; k < TBK; ++k) {
                if (unb < wbits) { ubuf |= (uint64_t)ureg[uwi++] << unb; unb += 32; }
                const int up_code = (int)(((uint32_t)ubuf & umask) ^ usign) - (int)usign;
                ubuf >>= wbits; unb -= wbits;
                if ((k & gsmask) == 0) {
                    if (snb < u) { sbuf |= (uint64_t)sreg[swi++] << snb; snb += 32; }
                    sh_code = (int)(((uint32_t)sbuf & smask) ^ ssign) - (int)ssign;
                    sbuf >>= u; snb -= u;
                }
                Bs[k][oc] = (float)(up_code * (1 << u) + sh_code) * sc;
            }
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
        for (int m = tid; m < TBM; m += NT) {       // thread-per-row MSAQ-s activation quant
            float amax = 1e-30f;
            #pragma unroll
            for (int k = 0; k < TBK; ++k) amax = fmaxf(amax, fabsf(As[k][m]));
            int ea = (int)floorf(log2f(amax)) - (int)ms::E_MAX;
            ea = max(-127, min(127, ea));
            const float sa = exp2f((float)ea); sa_s[m] = sa;
            const float s_unshared = sa * exp2f((float)u);
            const int qmax = (1 << (7 - u)) - 1, smin = -(1 << (u - 1)), smax = (1 << (u - 1)) - 1;
            for (int g0 = 0; g0 < TBK; g0 += gs) {  // residual mean per gs-group -> shared code
                float ravg = 0.0f;
                for (int k = g0; k < g0 + gs; ++k) {
                    const int qu = max(-qmax, min(qmax, (int)rintf(As[k][m] / s_unshared)));
                    ravg += As[k][m] - (float)qu * s_unshared;
                }
                const int rsh = max(smin, min(smax, (int)rintf(ravg / (float)gs / sa)));
                for (int k = g0; k < g0 + gs; ++k) {
                    const int qu = max(-qmax, min(qmax, (int)rintf(As[k][m] / s_unshared)));
                    As[k][m] = (float)(qu * (1 << u) + rsh) * sa;   // qx_msaq * sa
                }
            }
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

// ---- W-only prefill WMMA, SOFTWARE-PIPELINED (cp.async-style overlap) --------
//   The plain WMMA stage is stage(unpack ALL) -> sync -> matmul -> sync, so the
//   per-element unpack ALU and the tensor-core MMA run SERIALLY -> staging-bound
//   (P22: the fast matmul exposes the u2/u3 unpack). Double-buffer As/Bs and, in
//   each K-step, issue the CURRENT tile's MMA first, then unpack the NEXT tile
//   into the other buffer: the unpack (ALU + its global byte loads) overlaps the
//   MMA (tensor-core pipe) -> the unpack hides behind the matmul (the lever P22
//   flagged). Matched mxint8_gemm_wmma_pipe has the same shape (int8 stage), so
//   the comparison stays fair. Opt-in via MS_TILE_CFG=11.
// Stream-dequant one weight column to a bf16 row (for the pipelined WMMA stage):
// gather the column's UB upper + SB shared bytes once, rolling-bit-buffer emit 32
// codes (shared advanced per group). Cheaper ALU than per-element straddle; uses
// 64 of 128 threads, but here the stage OVERLAPS the MMA, so its lower width is
// hidden — the win is the smaller total work. Same dense bytes -> bit-exact.
template<bool CM>
__device__ __forceinline__ void dequant_col_stream_bf16(
        const uint8_t* __restrict__ upper, const uint8_t* __restrict__ shared,
        int blk, int o, int OUT, int u, int gs, int UB, int SB,
        float scale, __nv_bfloat16* dst) {
    const int wbits = 8 - u;
    const uint32_t umask = (1u << wbits) - 1u, usign = 1u << (wbits - 1);
    const uint32_t smask = (1u << u) - 1u, ssign = 1u << (u - 1);
    const int gsmask = gs - 1;
    uint32_t ureg[6] = {0u,0u,0u,0u,0u,0u};
    uint32_t sreg[3] = {0u,0u,0u};
    if constexpr (CM) {
        // column-major [nb,OUT,UB]: the column's UB bytes are CONTIGUOUS -> wide-load
        // (adjacent columns UB apart -> the warp coalesces), vs the row-major *OUT stride.
        const long base = (long)blk * OUT + o;
        const uint32_t* src = reinterpret_cast<const uint32_t*>(upper + base * UB);
        #pragma unroll
        for (int i = 0; i < 6; ++i) if (i < (UB >> 2)) ureg[i] = src[i];
        #pragma unroll
        for (int bi = 0; bi < 8; ++bi)
            if (bi < SB) sreg[bi >> 2] |= (uint32_t)shared[base * SB + bi] << (8 * (bi & 3));
    } else {
        #pragma unroll
        for (int bi = 0; bi < 24; ++bi)
            if (bi < UB) ureg[bi >> 2] |= (uint32_t)upper[((long)blk*UB + bi)*OUT + o] << (8*(bi&3));
        #pragma unroll
        for (int bi = 0; bi < 8; ++bi)
            if (bi < SB) sreg[bi >> 2] |= (uint32_t)shared[((long)blk*SB + bi)*OUT + o] << (8*(bi&3));
    }
    uint64_t ubuf = 0; int unb = 0, uwi = 0;
    uint64_t sbuf = 0; int snb = 0, swi = 0; int sh_code = 0;
    #pragma unroll
    for (int k = 0; k < BLOCK; ++k) {
        if (unb < wbits) { ubuf |= (uint64_t)ureg[uwi++] << unb; unb += 32; }
        const int up_code = (int)(((uint32_t)ubuf & umask) ^ usign) - (int)usign;
        ubuf >>= wbits; unb -= wbits;
        if ((k & gsmask) == 0) {
            if (snb < u) { sbuf |= (uint64_t)sreg[swi++] << snb; snb += 32; }
            sh_code = (int)(((uint32_t)sbuf & smask) ^ ssign) - (int)ssign;
            sbuf >>= u; snb -= u;
        }
        dst[k] = __float2bfloat16((float)(up_code * (1 << u) + sh_code) * scale);
    }
}

template<bool CM>
__global__ void wonly_gemm_wmma_pipe(
        const __nv_bfloat16* __restrict__ X, const int8_t* __restrict__ scale_exp,
        const uint8_t* __restrict__ upper, const uint8_t* __restrict__ shared,
        __nv_bfloat16* __restrict__ Y,
        int M, int OUT, int K, int NB, int u, int gs, int UB, int SB) {
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

    // stage block `b` (As=X, Bs=dequant W) into shared buffer `buf`
    auto stage = [&](int b, int buf) {
        for (int idx = tid; idx < 64 * BLOCK; idx += 128) {
            const int m = idx / BLOCK, k = idx % BLOCK;
            As[buf][m][k] = (m0 + m < M) ? X[(long)(m0 + m) * K + b * BLOCK + k]
                                         : __float2bfloat16(0.0f);
        }
        for (int oc = tid; oc < 64; oc += 128) {     // column-streaming unpack (hidden by MMA)
            const int o = o0 + oc;
            if (o < OUT) {
                const float sc = ms::e8m0_to_scale(scale_exp[b * OUT + o]);
                dequant_col_stream_bf16<CM>(upper, shared, b, o, OUT, u, gs, UB, SB, sc, &Bs[buf][oc][0]);
            } else {
                #pragma unroll
                for (int k = 0; k < BLOCK; ++k) Bs[buf][oc][k] = __float2bfloat16(0.0f);
            }
        }
    };

    stage(0, 0);
    __syncthreads();
    for (int blk = 0; blk < NB; ++blk) {
        const int cur = blk & 1, nxt = cur ^ 1;
        #pragma unroll
        for (int wk = 0; wk < BLOCK / 16; ++wk) {        // CURRENT tile MMA (tensor core)
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
        if (blk + 1 < NB) stage(blk + 1, nxt);           // NEXT tile unpack — overlaps the MMA
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

// ---- DECODE TENSOR-CORE BATCHED GEMV (M=B small): split-K WMMA ----------------
//   wmma_pipe is built for prefill (64x64 tile, no split-K) -> at decode (M=8-32,
//   OUT=4096) it leaves the machine empty (~64 blocks) and runs 10-16x off. Here the
//   M-tile is just 16 (one WMMA m; M>16 uses grid.y) and the K axis is SPLIT for
//   occupancy (grid.z), so OUT/64 * ceil(M/16) * splitK blocks fill the SMs and the
//   kernel becomes weight-DRAM-bound -> MSAQ's ~0.4x weight bytes convert to time and
//   beat bf16 cuBLAS. 4 warps = a 16x64 output tile; weight unpacked to a bf16 tile
//   (dequant_col_stream_bf16) so the sub-byte read rides the tensor core. partial
//   [splitK, M, OUT] -> gemv_combine_tc.
template<bool CM>
__global__ void wonly_gemv_tc_kernel(
        const __nv_bfloat16* __restrict__ X, const int8_t* __restrict__ scale_exp,
        const uint8_t* __restrict__ upper, const uint8_t* __restrict__ shared,
        float* __restrict__ partial, int M, int OUT, int K, int NB, int u, int gs, int UB, int SB, int splitK) {
    __shared__ __nv_bfloat16 As[2][16][WSK];
    __shared__ __nv_bfloat16 Bs[2][64][WSK];
    __shared__ float wt[4][16][16];
    const int m0 = blockIdx.y * 16, o0 = blockIdx.x * 64, sp = blockIdx.z;
    const int per = (NB + splitK - 1) / splitK, b0 = sp * per, b1 = min(b0 + per, NB);
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
    wmma::fill_fragment(acc, 0.0f);
    auto stage = [&](int b, int buf) {
        for (int idx = tid; idx < 16 * BLOCK; idx += 128) {
            const int m = idx / BLOCK, k = idx % BLOCK;
            As[buf][m][k] = (m0 + m < M) ? X[(long)(m0 + m) * K + b * BLOCK + k] : __float2bfloat16(0.0f);
        }
        for (int oc = tid; oc < 64; oc += 128) {
            const int o = o0 + oc;
            if (o < OUT) {
                const float sc = ms::e8m0_to_scale(scale_exp[b * OUT + o]);
                dequant_col_stream_bf16<CM>(upper, shared, b, o, OUT, u, gs, UB, SB, sc, &Bs[buf][oc][0]);
            } else {
                #pragma unroll
                for (int k = 0; k < BLOCK; ++k) Bs[buf][oc][k] = __float2bfloat16(0.0f);
            }
        }
    };
    if (b0 < b1) stage(b0, 0);
    __syncthreads();
    for (int blk = b0; blk < b1; ++blk) {
        const int cur = (blk - b0) & 1, nxt = cur ^ 1;
        #pragma unroll
        for (int wk = 0; wk < BLOCK / 16; ++wk) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> a;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> b;
            wmma::load_matrix_sync(a, &As[cur][0][wk * 16], WSK);
            wmma::load_matrix_sync(b, &Bs[cur][warp * 16][wk * 16], WSK);
            wmma::mma_sync(acc, a, b, acc);
        }
        if (blk + 1 < b1) stage(blk + 1, nxt);
        __syncthreads();
    }
    wmma::store_matrix_sync(&wt[warp][0][0], acc, 16, wmma::mem_row_major);
    __syncwarp();
    for (int e = lane; e < 256; e += 32) {
        const int m = m0 + e / 16, o = o0 + warp * 16 + e % 16;
        if (m < M && o < OUT) partial[((long)sp * M + m) * OUT + o] = wt[warp][e / 16][e % 16];
    }
}

__global__ void gemv_combine_tc_kernel(
        const float* __restrict__ partial, __nv_bfloat16* __restrict__ y, int M, int OUT, int splitK) {
    const int o = blockIdx.x * blockDim.x + threadIdx.x, m = blockIdx.y;
    if (o >= OUT || m >= M) return;
    float acc = 0.0f;
    for (int sp = 0; sp < splitK; ++sp) acc += partial[((long)sp * M + m) * OUT + o];
    y[(long)m * OUT + o] = __float2bfloat16(acc);
}

// ---- STAGE 1: W+A as PURE INT8 GEMM (IMMA), activation PRE-QUANTIZED ----------
//   Reads qX int8 (Stage 0 output) directly — the in-kernel activation quant that
//   filled the prologue in P23/24 is GONE, so the only heavy stage is the weight
//   unpack (B-stage) = exactly W-only GEMM. INT8 MMA -> int32 per block; the
//   per-block epilogue applies (2^sa_exp[m,blk] * sw[o,blk]) and accumulates fp32.
//   Matched mxint8_wa_imma differs only in B-stage (direct int8). (change.md P26.)
//   DOUBLE-BUFFERED: the next block's A/B stage (the weight UNPACK is the only
//   heavy part) overlaps the current block's MMA -> the unpack hides behind the
//   tensor core, exactly the W-only WMMA-pipe lesson (P23) now that Stage 0 freed
//   the prologue of activation quant.
template<bool CM>
__global__ void wa_imma(
        const int8_t* __restrict__ qX, const int8_t* __restrict__ sa_exp,
        const int8_t* __restrict__ scale_exp, const uint8_t* __restrict__ upper,
        const uint8_t* __restrict__ shared, __nv_bfloat16* __restrict__ Y,
        int M, int OUT, int K, int NB, int u, int gs, int UB, int SB, int diag = 0) {
    // diag (MS_WA_DIAG): 1 = skip per-block epilogue (store+scale); 2 = skip the
    // B-stage weight unpack+read (Bs=0). full - diag1 = epilogue cost; full - diag2
    // = B-stage exposed cost (~0 if the unpack hides behind the MMA).
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
        if constexpr (CM) {
            // COLUMN-MAJOR wide-load: thread owns whole columns; wide-load the column's
            // UB bytes (contiguous, adjacent columns UB apart -> coalesced) then
            // streaming-unpack all 32 codes from registers (replaces the per-element
            // uncoalesced extract_code that was ~62% of wa_imma exposed).
            const int wbits = 8 - u, gsmask = gs - 1;
            const uint32_t umask = (1u << wbits) - 1u, usign = 1u << (wbits - 1);
            const uint32_t smask = (1u << u) - 1u, ssign = 1u << (u - 1);
            for (int oc = tid; oc < 64; oc += 128) {
                const int o = o0 + oc;
                if (diag == 2 || o >= OUT) {
                    #pragma unroll
                    for (int k = 0; k < BLOCK; ++k) Bs[buf][oc][k] = (int8_t)0;
                    continue;
                }
                const long base = (long)blk * OUT + o;
                uint32_t ureg[6];
                const uint32_t* src = reinterpret_cast<const uint32_t*>(upper + base * UB);
                #pragma unroll
                for (int i = 0; i < 6; ++i) if (i < (UB >> 2)) ureg[i] = src[i];
                uint32_t sreg[3] = {0u, 0u, 0u};
                #pragma unroll
                for (int i = 0; i < 8; ++i) if (i < SB) sreg[i >> 2] |= (uint32_t)shared[base * SB + i] << (8 * (i & 3));
                uint64_t ubuf = 0; int unb = 0, uwi = 0;
                uint64_t sbuf = 0; int snb = 0, swi = 0; int sh_code = 0;
                #pragma unroll
                for (int k = 0; k < BLOCK; ++k) {
                    if (unb < wbits) { ubuf |= (uint64_t)ureg[uwi++] << unb; unb += 32; }
                    const int up_code = (int)(((uint32_t)ubuf & umask) ^ usign) - (int)usign;
                    ubuf >>= wbits; unb -= wbits;
                    if ((k & gsmask) == 0) {
                        if (snb < u) { sbuf |= (uint64_t)sreg[swi++] << snb; snb += 32; }
                        sh_code = (int)(((uint32_t)sbuf & smask) ^ ssign) - (int)ssign;
                        sbuf >>= u; snb -= u;
                    }
                    Bs[buf][oc][k] = (int8_t)(up_code * (1 << u) + sh_code);
                }
            }
        } else {
            for (int idx = tid; idx < 64 * BLOCK; idx += 128) {       // weight unpack (heavy)
                const int oc = idx >> 5, k = idx & 31, o = o0 + oc;
                Bs[buf][oc][k] = (diag == 2) ? (int8_t)0 : ((o < OUT)
                    ? (int8_t)ms::unpack_ms_weight_elem(upper, shared, blk, o, k, OUT, u, gs, UB, SB)
                    : (int8_t)0);
            }
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
        for (int wk = 0; wk < 2; ++wk) {                          // MMA (cur, tensor core)
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
        if (blk + 1 < NB) stage(blk + 1, nxt);                    // next unpack — overlaps the MMA
        if (diag == 1) { __syncthreads(); continue; }             // skip epilogue (isolate its cost)
        #pragma unroll
        for (int i = 0; i < 2; ++i)
            #pragma unroll
            for (int j = 0; j < 2; ++j)
                wmma::store_matrix_sync(&Cs[wM*32 + i*16][wN*32 + j*16], c[i][j], 64, wmma::mem_row_major);
        __syncthreads();
        #pragma unroll
        for (int j = 0; j < 32; ++j) {                            // per-block scale + accumulate
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

// Stage-0 launchers. ms_launch_quant_act = plain MXINT8 (used by the MXINT8
// baseline, extern from mxint8.cu). ms_launch_quant_act_msaq = MSAQ-s decompose
// (used by the MSAQ W+A path — the activation gets the mantissa-sharing format).
void ms_launch_quant_act(const __nv_bfloat16* X, int8_t* qX, int8_t* sa_exp,
                         int M, int K, int NB) {
    const int tpb = 256, wpb = tpb >> 5;
    const int blocks = ((long)M * NB + wpb - 1) / wpb;
    quant_act_kernel<<<blocks, tpb, 0, at::cuda::getCurrentCUDAStream()>>>(X, qX, sa_exp, M, K, NB);
}
void ms_launch_quant_act_msaq(const __nv_bfloat16* X, int8_t* qX, int8_t* sa_exp,
                              int M, int K, int NB, int u, int gs) {
    const int tpb = 256, wpb = tpb >> 5;
    const int blocks = ((long)M * NB + wpb - 1) / wpb;
    quant_act_msaq_kernel<<<blocks, tpb, 0, at::cuda::getCurrentCUDAStream()>>>(X, qX, sa_exp, M, K, NB, u, gs);
}

// torch op: MSAQ-s activation quant -> (qX int8 [M,K], sa_exp int8 [M,nb]).
// Exposed so the benchmark can time Stage 0 alone (the pre-pass decomposition).
std::vector<torch::Tensor> quant_act_cuda(torch::Tensor X, int64_t M, int64_t K,
                                          int64_t NB, int64_t u, int64_t gs) {
    TORCH_CHECK(X.is_cuda() && X.scalar_type() == torch::kBFloat16, "X must be CUDA bf16");
    auto qX = torch::empty({M, K}, X.options().dtype(torch::kInt8));
    auto sa = torch::empty({M, NB}, X.options().dtype(torch::kInt8));
    ms_launch_quant_act_msaq(reinterpret_cast<const __nv_bfloat16*>(X.data_ptr<at::BFloat16>()),
                             qX.data_ptr<int8_t>(), sa.data_ptr<int8_t>(),
                             (int)M, (int)K, (int)NB, (int)u, (int)gs);
    return {qX, sa};
}

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
           dim3(((BM) / (RM)) * ((BN) / (RN))), 0, at::cuda::getCurrentCUDAStream()>>>(ARGS)
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
    if (tile_cfg((int)M) == 11)     // BF16 WMMA, software-pipelined (unpack hidden behind MMA)
        wonly_gemm_wmma_pipe<false><<<dim3(((int)OUT + 63) / 64, ((int)M + 63) / 64), 128, 0, at::cuda::getCurrentCUDAStream()>>>(ARGS);
    else if (tile_cfg((int)M) == 10)  // BF16 tensor-core (WMMA) path
        wonly_gemm_wmma<<<dim3(((int)OUT + 63) / 64, ((int)M + 63) / 64), 128, 0, at::cuda::getCurrentCUDAStream()>>>(ARGS);
    else
        DISPATCH_TILE(wonly_gemm_tiled);
#undef ARGS
    return Y;
}

// Column-major wide-load W-only prefill GEMM (coalesced unpack). Takes the
// COLUMN-MAJOR planes [nb,OUT,*]; same tile dispatch as wonly_gemm (default
// cfg 1/5). A/B vs the row-major wonly_gemm.
torch::Tensor wonly_gemm_cm_cuda(
        torch::Tensor X, torch::Tensor scale_exp,
        torch::Tensor upper_cm, torch::Tensor shared_cm,
        int64_t M, int64_t OUT, int64_t K, int64_t NB, int64_t u, int64_t gs) {
    TORCH_CHECK(X.is_cuda() && X.scalar_type() == torch::kBFloat16, "X must be CUDA bf16");
    int UB, SB; gemm_dims(u, gs, UB, SB);
    auto Y = torch::empty({M, OUT}, X.options());
#define ARGS \
    reinterpret_cast<const __nv_bfloat16*>(X.data_ptr<at::BFloat16>()), \
    scale_exp.data_ptr<int8_t>(), upper_cm.data_ptr<uint8_t>(), shared_cm.data_ptr<uint8_t>(), \
    reinterpret_cast<__nv_bfloat16*>(Y.data_ptr<at::BFloat16>()), \
    (int)M, (int)OUT, (int)K, (int)NB, (int)u, (int)gs, UB, SB
    DISPATCH_TILE(wonly_gemm_tiled_cm);
#undef ARGS
    return Y;
}

// TENSOR-CORE W-only prefill GEMM: BF16 WMMA (software-pipelined) with the weight
// dequant wide-loading the COLUMN-MAJOR planes (coalesced, overlaps the MMA). Combines
// tensor cores + coalesced unpack -> the fastest W-only prefill path. bf16 operands
// (fp32 accumulate); slight precision vs the fp32 scalar path, inherent to WMMA.
torch::Tensor wonly_gemm_tc_cuda(
        torch::Tensor X, torch::Tensor scale_exp,
        torch::Tensor upper_cm, torch::Tensor shared_cm,
        int64_t M, int64_t OUT, int64_t K, int64_t NB, int64_t u, int64_t gs) {
    TORCH_CHECK(X.is_cuda() && X.scalar_type() == torch::kBFloat16, "X must be CUDA bf16");
    int UB, SB; gemm_dims(u, gs, UB, SB);
    auto Y = torch::empty({M, OUT}, X.options());
    wonly_gemm_wmma_pipe<true><<<dim3(((int)OUT + 63) / 64, ((int)M + 63) / 64), 128, 0, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const __nv_bfloat16*>(X.data_ptr<at::BFloat16>()),
        scale_exp.data_ptr<int8_t>(), upper_cm.data_ptr<uint8_t>(), shared_cm.data_ptr<uint8_t>(),
        reinterpret_cast<__nv_bfloat16*>(Y.data_ptr<at::BFloat16>()),
        (int)M, (int)OUT, (int)K, (int)NB, (int)u, (int)gs, UB, SB);
    return Y;
}

// decode tensor-core batched GEMV: x [M,K] -> y [M,OUT]. split-K WMMA, weight-DRAM-bound.
torch::Tensor wonly_gemv_tc_cuda(
        torch::Tensor x, torch::Tensor scale_exp, torch::Tensor upper, torch::Tensor shared,
        int64_t M, int64_t OUT, int64_t NB, int64_t u, int64_t gs) {
    TORCH_CHECK(x.is_cuda() && x.scalar_type() == torch::kBFloat16, "x must be CUDA bf16");
    int UB, SB; gemm_dims(u, gs, UB, SB);
    auto y = torch::empty({M, OUT}, x.options());
    const int K = (int)NB * BLOCK;
    const int nO = ((int)OUT + 63) / 64, nM = ((int)M + 15) / 16;
    // split K so nO*nM*splitK ~ 4x SM count -> fill the machine (decode underfills otherwise)
    int splitK = (4 * 82) / (nO * nM); if (splitK < 1) splitK = 1; if (splitK > (int)NB) splitK = (int)NB;
    auto partial = torch::empty({(int64_t)splitK, M, OUT}, x.options().dtype(torch::kFloat32));
    dim3 grid(nO, nM, splitK);
    wonly_gemv_tc_kernel<false><<<grid, 128, 0, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const __nv_bfloat16*>(x.data_ptr<at::BFloat16>()),
        scale_exp.data_ptr<int8_t>(), upper.data_ptr<uint8_t>(), shared.data_ptr<uint8_t>(),
        partial.data_ptr<float>(), (int)M, (int)OUT, K, (int)NB, (int)u, (int)gs, UB, SB, splitK);
    gemv_combine_tc_kernel<<<dim3(((int)OUT + 127) / 128, (int)M), 128, 0, at::cuda::getCurrentCUDAStream()>>>(
        partial.data_ptr<float>(), reinterpret_cast<__nv_bfloat16*>(y.data_ptr<at::BFloat16>()),
        (int)M, (int)OUT, splitK);
    return y;
}

// W+A GEMM = STAGE 0 (runtime activation quant) + STAGE 1 (pure int8 IMMA GEMM).
//   MS_WA_FOLD=1 falls back to the old fused FP32-fold tiled path (for A/B checks).
torch::Tensor wa_gemm_cuda(
        torch::Tensor X, torch::Tensor scale_exp,
        torch::Tensor upper, torch::Tensor shared,
        int64_t M, int64_t OUT, int64_t K, int64_t NB, int64_t u, int64_t gs) {
    TORCH_CHECK(X.is_cuda() && X.scalar_type() == torch::kBFloat16, "X must be CUDA bf16");
    int UB, SB; gemm_dims(u, gs, UB, SB);
    auto Y = torch::empty({M, OUT}, X.options());
    const char* f = getenv("MS_WA_FOLD");
    if (f && atoi(f) == 1) {        // legacy fused FP32-fold path (A/B reference)
#define ARGS \
        reinterpret_cast<const __nv_bfloat16*>(X.data_ptr<at::BFloat16>()), \
        scale_exp.data_ptr<int8_t>(), upper.data_ptr<uint8_t>(), shared.data_ptr<uint8_t>(), \
        reinterpret_cast<__nv_bfloat16*>(Y.data_ptr<at::BFloat16>()), \
        (int)M, (int)OUT, (int)K, (int)NB, (int)u, (int)gs, UB, SB
        DISPATCH_TILE(wa_gemm_tiled);
#undef ARGS
        return Y;
    }
    // Stage 0: X -> qX int8 + sa_exp int8 (MSAQ-s activation: mantissa-sharing)
    auto qX = torch::empty({M, K}, X.options().dtype(torch::kInt8));
    auto sa = torch::empty({M, NB}, X.options().dtype(torch::kInt8));
    ms_launch_quant_act_msaq(reinterpret_cast<const __nv_bfloat16*>(X.data_ptr<at::BFloat16>()),
                             qX.data_ptr<int8_t>(), sa.data_ptr<int8_t>(),
                             (int)M, (int)K, (int)NB, (int)u, (int)gs);
    int diag = 0; if (const char* d = getenv("MS_WA_DIAG")) diag = atoi(d);
    // Stage 1: pure int8 IMMA GEMM (row-major weight unpack)
    wa_imma<false><<<dim3(((int)OUT + 63) / 64, ((int)M + 63) / 64), 128, 0, at::cuda::getCurrentCUDAStream()>>>(
        qX.data_ptr<int8_t>(), sa.data_ptr<int8_t>(), scale_exp.data_ptr<int8_t>(),
        upper.data_ptr<uint8_t>(), shared.data_ptr<uint8_t>(),
        reinterpret_cast<__nv_bfloat16*>(Y.data_ptr<at::BFloat16>()),
        (int)M, (int)OUT, (int)K, (int)NB, (int)u, (int)gs, UB, SB, diag);
    return Y;
}

// COLUMN-MAJOR wide-load W+A GEMM: Stage-0 activation quant + IMMA GEMM whose weight
// unpack wide-loads the COLUMN-MAJOR planes (coalesced). The wa_imma unpack was ~62%
// of total exposed (MS_WA_DIAG) -> not hidden behind the MMA, so coalescing it pays.
torch::Tensor wa_gemm_cm_cuda(
        torch::Tensor X, torch::Tensor scale_exp,
        torch::Tensor upper_cm, torch::Tensor shared_cm,
        int64_t M, int64_t OUT, int64_t K, int64_t NB, int64_t u, int64_t gs) {
    TORCH_CHECK(X.is_cuda() && X.scalar_type() == torch::kBFloat16, "X must be CUDA bf16");
    int UB, SB; gemm_dims(u, gs, UB, SB);
    auto Y = torch::empty({M, OUT}, X.options());
    auto qX = torch::empty({M, K}, X.options().dtype(torch::kInt8));
    auto sa = torch::empty({M, NB}, X.options().dtype(torch::kInt8));
    ms_launch_quant_act_msaq(reinterpret_cast<const __nv_bfloat16*>(X.data_ptr<at::BFloat16>()),
                             qX.data_ptr<int8_t>(), sa.data_ptr<int8_t>(),
                             (int)M, (int)K, (int)NB, (int)u, (int)gs);
    int diag = 0; if (const char* d = getenv("MS_WA_DIAG")) diag = atoi(d);
    wa_imma<true><<<dim3(((int)OUT + 63) / 64, ((int)M + 63) / 64), 128, 0, at::cuda::getCurrentCUDAStream()>>>(
        qX.data_ptr<int8_t>(), sa.data_ptr<int8_t>(), scale_exp.data_ptr<int8_t>(),
        upper_cm.data_ptr<uint8_t>(), shared_cm.data_ptr<uint8_t>(),
        reinterpret_cast<__nv_bfloat16*>(Y.data_ptr<at::BFloat16>()),
        (int)M, (int)OUT, (int)K, (int)NB, (int)u, (int)gs, UB, SB, diag);
    return Y;
}
