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
#include <type_traits>
#include "core/ms_utils.cuh"
#include <ATen/cuda/CUDAContext.h>

// MSAQ-s activation pre-pass (defined in wa_gemm.cu): bf16 X[M,K] -> int8 word
// qX[M,K] + base exp sa_exp[M,nb]. The W+A GEMV calls it with M=1.
void ms_launch_quant_act_msaq(const __nv_bfloat16* X, int8_t* qX, int8_t* sa_exp,
                              int M, int K, int NB, int u, int gs);

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

// ---- WIDE-LOAD split-K GEMV (column-major plane -> wide load + register extract)
//   The cp.async kernel hid the global load but stayed bound by NARROW per-byte
//   shared reads in the unpack (bfe didn't help -> not ALU-bound). This reads the
//   weight COLUMN-MAJOR [NB,OUT,UB]: thread o loads its column's whole UB-byte
//   block in ONE (u4: int4) or a few (u2/u3: 4-aligned uint32) wide coalesced
//   loads and extracts all 32 codes from registers -> ~UB narrow byte-loads
//   collapse to UB/16..UB/4. (Per-thread loads of consecutive columns are UB B
//   apart = warp-contiguous -> coalesced.) Templated on U4 with `if constexpr`
//   so the u4 (uint4+bfe, no straddle) and u2/u3 (word-load + general straddle
//   extract) paths don't share registers — same lesson as KV Phase 18. split-K +
//   the same combine.
template<bool U4>
__global__ void wonly_gemv_wide_kernel(
        const __nv_bfloat16* __restrict__ x,
        const int8_t*  __restrict__ scale_exp,   // [NB, OUT]
        const uint8_t* __restrict__ upper_cm,    // [NB, OUT, UB]   column-major
        const uint8_t* __restrict__ shared_cm,   // [NB, OUT, SB]
        float* __restrict__ partial,             // [splitK, OUT]
        int OUT, int NB, int u, int gs, int UB, int SB, int splitK, int sepsc) {
    const int tid = threadIdx.x;
    const int o   = blockIdx.x * blockDim.x + tid;
    const int sp  = blockIdx.y;
    const int per = (NB + splitK - 1) / splitK;
    const int b0  = sp * per, b1 = min(b0 + per, NB);
    const int gs_shift = __ffs(gs) - 1;                  // gs is a power of 2: k/gs == k>>shift
    float acc = 0.0f;

    if constexpr (U4) {
        // u4: column's 16 B == one int4 (stride 16 == width -> sector-coalesced).
        // Cheap nibble bfe, no straddle.
        if (o >= OUT) return;
        for (int blk = b0; blk < b1; ++blk) {
            const float scale = ms::e8m0_to_scale(scale_exp[blk * OUT + o]);
            uint8_t sb[8];
            const long sbase = ((long)blk * OUT + o) * SB;
            #pragma unroll
            for (int i = 0; i < 8; ++i) if (i < SB) sb[i] = shared_cm[sbase + i];
            const uint4 up4 = *reinterpret_cast<const uint4*>(
                                  upper_cm + ((long)blk * OUT + o) * UB);   // 16 B, one load
            const uint32_t uw[4] = { up4.x, up4.y, up4.z, up4.w };
            if (sepsc) {            // separated-scale: s·(2^u·Σx·up + Σ_g sh·xg), xg=group x-sum
                const int gsmask = gs - 1;
                float bup = 0.0f, bsh = 0.0f, xsum = 0.0f; int sh_code = 0;
                #pragma unroll
                for (int k = 0; k < BLOCK; ++k) {
                    if ((k & gsmask) == 0) {
                        if (k > 0) bsh += sh_code * xsum;
                        const int g = k >> gs_shift;
                        sh_code = ms::bfe_s32((int)sb[g >> 1], (g & 1) * 4, 4);
                        xsum = 0.0f;
                    }
                    const float xk = __bfloat162float(x[blk * BLOCK + k]);
                    bup += ms::bfe_s32((int)uw[k >> 3], (k & 7) * 4, 4) * xk;
                    xsum += xk;
                }
                bsh += sh_code * xsum;
                acc += scale * (16.0f * bup + bsh);
            } else {
                #pragma unroll
                for (int k = 0; k < BLOCK; ++k) {
                    const int up_code = ms::bfe_s32((int)uw[k >> 3], (k & 7) * 4, 4);
                    const int g = k >> gs_shift;
                    const int sh_code = ms::bfe_s32((int)sb[g >> 1], (g & 1) * 4, 4);
                    const int w = up_code * 16 + sh_code;
                    acc += (static_cast<float>(w) * scale) * __bfloat162float(x[blk * BLOCK + k]);
                }
            }
        }
    } else {
        // u2/u3 GEMV crossover (Phase 20). The limiter was EXTRACTION, not the load:
        // the byte-level straddle unpack (per-code shift/or + a conditional 2nd-byte
        // branch) was ~55% of the time, and u2 (25 B) and u3 (22 B) ran at the SAME
        // ~40 us -> byte-independent -> compute-bound, not memory-bound. (A plane-
        // split that perfectly coalesced the load was tried and gave ZERO speedup,
        // confirming load wasn't the limit.) Fix = a STREAMING bit-buffer unpack:
        // load the column's UB bytes as uint32 words, then a rolling 64-bit buffer
        // yields each code with ONE shift+mask (refill a word only when low) and the
        // shared code advances once per group -> ~64 random funnel-shifts collapse
        // to ~6 ORs + 32 shifts. That alone takes u2/u3 from ~1.45x to ~0.85x.
        if (o >= OUT) return;
        const int wbits = 8 - u;
        const uint32_t umask = (1u << wbits) - 1u, usign = 1u << (wbits - 1);
        const uint32_t smask = (1u << u) - 1u, ssign = 1u << (u - 1);
        for (int blk = b0; blk < b1; ++blk) {
            const float scale = ms::e8m0_to_scale(scale_exp[blk * OUT + o]);
            const long sbase = ((long)blk * OUT + o) * SB;
            uint32_t sreg[3] = {0u,0u,0u};               // shared codes as a bit stream (SB<=8 B)
            #pragma unroll
            for (int i = 0; i < 8; ++i)
                if (i < SB) sreg[i >> 2] |= (uint32_t)shared_cm[sbase + i] << (8 * (i & 3));
            uint32_t ureg[6] = {0u,0u,0u,0u,0u,0u};      // UB/4 <= 6 words (u2: 24 B)
            const uint32_t* src = reinterpret_cast<const uint32_t*>(
                                      upper_cm + ((long)blk * OUT + o) * UB);
            #pragma unroll
            for (int i = 0; i < 6; ++i) if (i < (UB >> 2)) ureg[i] = src[i];
            // STREAMING unpack: a rolling 64-bit bit-buffer yields each code with
            // ONE shift+mask (refill a 32-bit word only when it runs low) instead
            // of a random-access funnel-shift PER code -> ~32 funnelshifts collapse
            // to ~6 ORs. The shared code only changes once per group (gs elems), so
            // advance its own buffer on group boundaries, not every element.
            const int gsmask = gs - 1;
            uint64_t ubuf = 0; int unb = 0, uwi = 0;
            uint64_t sbuf = 0; int snb = 0, swi = 0;
            int sh_code = 0;
            float bup = 0.0f, bsh = 0.0f, xsum = 0.0f;    // sepsc accumulators
            for (int k = 0; k < BLOCK; ++k) {
                if (unb < wbits) { ubuf |= (uint64_t)ureg[uwi++] << unb; unb += 32; }
                const int up_code = (int)(((uint32_t)ubuf & umask) ^ usign) - (int)usign;
                ubuf >>= wbits; unb -= wbits;
                const float xk = __bfloat162float(x[blk * BLOCK + k]);
                if ((k & gsmask) == 0) {                  // new shared group
                    if (sepsc && k > 0) bsh += sh_code * xsum;
                    if (snb < u) { sbuf |= (uint64_t)sreg[swi++] << snb; snb += 32; }
                    sh_code = (int)(((uint32_t)sbuf & smask) ^ ssign) - (int)ssign;
                    sbuf >>= u; snb -= u; xsum = 0.0f;
                }
                if (sepsc) { bup += up_code * xk; xsum += xk; }
                else acc += (static_cast<float>(up_code * (1 << u) + sh_code) * scale) * xk;
            }
            if (sepsc) { bsh += sh_code * xsum; acc += scale * ((float)(1 << u) * bup + bsh); }
        }
    }
    partial[(long)sp * OUT + o] = acc;
}

// ---- BATCHED-DECODE W-only GEMV (M=B small): amortize the weight read over B rows --
//   The B=1 wide GEMV is memory-bound on the weight (the column read dominates; the
//   activation is tiny). At decode batch M=B the SAME weight column serves all B rows,
//   so reading it ONCE and applying it to MR activation rows held in registers
//   (acc[MR]) amortizes the weight read M-fold. MSAQ's ~0.5x weight bytes then convert
//   to time -> wins vs MXINT8 (1B) and bf16 GEMM until the compute crossover (~tens of
//   rows, §3.2). x [M,K] row-major; partial [splitK, M, OUT]; grid (OUT/thr, splitK, ceil(M/MR)).
template<bool U4, int MR>
__global__ void wonly_gemv_batched_kernel(
        const __nv_bfloat16* __restrict__ x,
        const int8_t*  __restrict__ scale_exp, const uint8_t* __restrict__ upper_cm,
        const uint8_t* __restrict__ shared_cm,
        float* __restrict__ partial, int M, int OUT, int NB, int u, int gs, int UB, int SB, int splitK) {
    const int o = blockIdx.x * blockDim.x + threadIdx.x;
    if (o >= OUT) return;
    const int sp = blockIdx.y, row0 = blockIdx.z * MR;
    const int per = (NB + splitK - 1) / splitK, b0 = sp * per, b1 = min(b0 + per, NB);
    const int K = NB * BLOCK, gs_shift = __ffs(gs) - 1;
    float acc[MR];
    #pragma unroll
    for (int j = 0; j < MR; ++j) acc[j] = 0.0f;
    for (int blk = b0; blk < b1; ++blk) {
        const float scale = ms::e8m0_to_scale(scale_exp[blk * OUT + o]);
        const long base = (long)blk * OUT + o;
        uint8_t sb[8];
        const long sbase = base * SB;
        #pragma unroll
        for (int i = 0; i < 8; ++i) if (i < SB) sb[i] = shared_cm[sbase + i];
        if constexpr (U4) {
            const uint4 up4 = *reinterpret_cast<const uint4*>(upper_cm + base * UB);
            const uint32_t uw[4] = { up4.x, up4.y, up4.z, up4.w };
            #pragma unroll
            for (int k = 0; k < BLOCK; ++k) {
                const int up_code = ms::bfe_s32((int)uw[k >> 3], (k & 7) * 4, 4);
                const int g = k >> gs_shift;
                const int sh_code = ms::bfe_s32((int)sb[g >> 1], (g & 1) * 4, 4);
                const float wf = (float)(up_code * 16 + sh_code) * scale;
                const int kk = blk * BLOCK + k;
                #pragma unroll
                for (int j = 0; j < MR; ++j) { const int m = row0 + j; if (m < M) acc[j] += wf * __bfloat162float(x[(long)m * K + kk]); }
            }
        } else {
            const int wbits = 8 - u;
            const uint32_t umask = (1u << wbits) - 1u, usign = 1u << (wbits - 1);
            const uint32_t smask = (1u << u) - 1u, ssign = 1u << (u - 1);
            uint32_t sreg[3] = {0u,0u,0u};
            #pragma unroll
            for (int i = 0; i < 8; ++i) if (i < SB) sreg[i >> 2] |= (uint32_t)sb[i] << (8 * (i & 3));
            uint32_t ureg[6] = {0u,0u,0u,0u,0u,0u};
            const uint32_t* src = reinterpret_cast<const uint32_t*>(upper_cm + base * UB);
            #pragma unroll
            for (int i = 0; i < 6; ++i) if (i < (UB >> 2)) ureg[i] = src[i];
            const int gsmask = gs - 1;
            uint64_t ubuf = 0; int unb = 0, uwi = 0; uint64_t sbuf = 0; int snb = 0, swi = 0; int sh_code = 0;
            for (int k = 0; k < BLOCK; ++k) {
                if (unb < wbits) { ubuf |= (uint64_t)ureg[uwi++] << unb; unb += 32; }
                const int up_code = (int)(((uint32_t)ubuf & umask) ^ usign) - (int)usign; ubuf >>= wbits; unb -= wbits;
                if ((k & gsmask) == 0) { if (snb < u) { sbuf |= (uint64_t)sreg[swi++] << snb; snb += 32; } sh_code = (int)(((uint32_t)sbuf & smask) ^ ssign) - (int)ssign; sbuf >>= u; snb -= u; }
                const float wf = (float)(up_code * (1 << u) + sh_code) * scale;
                const int kk = blk * BLOCK + k;
                #pragma unroll
                for (int j = 0; j < MR; ++j) { const int m = row0 + j; if (m < M) acc[j] += wf * __bfloat162float(x[(long)m * K + kk]); }
            }
        }
    }
    #pragma unroll
    for (int j = 0; j < MR; ++j) { const int m = row0 + j; if (m < M) partial[((long)sp * M + m) * OUT + o] = acc[j]; }
}

__global__ void gemv_combine_batched_kernel(
        const float* __restrict__ partial, __nv_bfloat16* __restrict__ y, int M, int OUT, int splitK) {
    const int o = blockIdx.x * blockDim.x + threadIdx.x, m = blockIdx.y;
    if (o >= OUT || m >= M) return;
    float acc = 0.0f;
    for (int sp = 0; sp < splitK; ++sp) acc += partial[((long)sp * M + m) * OUT + o];
    y[(long)m * OUT + o] = __float2bfloat16(acc);
}

// ---- W+A GEMV (decode): wide-load skeleton + int dot ------------------------
//   The W-only wide GEMV above with the activation ALSO quantized (MSAQ-s).
//   A separate pre-pass (ms_launch_quant_act_msaq, M=1) decomposes x[K] into the
//   int8 word qx[k] = q_upper*2^u + r_shared and the per-block base exp sa_exp[blk]
//   (the activation analog of the weight word). The unpack of qw is byte-identical
//   to W-only; the ONLY change is the accumulation: per block we run an INTEGER
//   dot idot = sum_k qw*qx (int8*int8 -> int32) and fold the two block scales ONCE
//   at the end (acc += idot * sw * sa) instead of a float madd per element. sw is
//   per (blk,o) (weight), sa is per blk (activation, shared across columns).
//   (change.md Phase 28.)
template<bool U4>
__global__ void wa_gemv_wide_kernel(
        const int8_t*  __restrict__ qx,          // [K]    activation int8 word (MSAQ-s)
        const int8_t*  __restrict__ sa_exp,      // [NB]   activation block base exp
        const int8_t*  __restrict__ scale_exp,   // [NB, OUT]    weight base exp
        const uint8_t* __restrict__ upper_cm,    // [NB, OUT, UB]   column-major
        const uint8_t* __restrict__ shared_cm,   // [NB, OUT, SB]
        float* __restrict__ partial,             // [splitK, OUT]
        int OUT, int NB, int u, int gs, int UB, int SB, int splitK) {
    const int o   = blockIdx.x * blockDim.x + threadIdx.x;
    const int sp  = blockIdx.y;
    const int per = (NB + splitK - 1) / splitK;
    const int b0  = sp * per, b1 = min(b0 + per, NB);
    const int gs_shift = __ffs(gs) - 1;

    // Stage this K-slice's activation into shared ONCE (the activation is the same
    // for every output column in the block). Otherwise each of the 128 columns
    // re-loads qx[k] from global, and those redundant loads contend with the weight
    // stream that the unpack is already latency-bound on. All threads (incl. o>=OUT)
    // cooperate, so the syncthreads precedes any return.
    extern __shared__ int8_t qx_sh[];                    // [(b1-b0)*BLOCK]
    const int slice = (b1 - b0) * BLOCK;
    for (int i = threadIdx.x; i < slice; i += blockDim.x) qx_sh[i] = qx[b0 * BLOCK + i];
    __syncthreads();
    float acc = 0.0f;

    if constexpr (U4) {
        if (o >= OUT) return;
        for (int blk = b0; blk < b1; ++blk) {
            const float sw = ms::e8m0_to_scale(scale_exp[blk * OUT + o]);
            const float sa = ms::e8m0_to_scale(sa_exp[blk]);
            uint8_t sb[8];
            const long sbase = ((long)blk * OUT + o) * SB;
            #pragma unroll
            for (int i = 0; i < 8; ++i) if (i < SB) sb[i] = shared_cm[sbase + i];
            const uint4 up4 = *reinterpret_cast<const uint4*>(
                                  upper_cm + ((long)blk * OUT + o) * UB);
            const uint32_t uw[4] = { up4.x, up4.y, up4.z, up4.w };
            const int8_t* qxb = qx_sh + (blk - b0) * BLOCK;
            int idot = 0;
            #pragma unroll
            for (int k = 0; k < BLOCK; ++k) {
                const int up_code = ms::bfe_s32((int)uw[k >> 3], (k & 7) * 4, 4);
                const int g = k >> gs_shift;
                const int sh_code = ms::bfe_s32((int)sb[g >> 1], (g & 1) * 4, 4);
                const int w = up_code * 16 + sh_code;
                idot += w * (int)qxb[k];
            }
            acc += (float)idot * sw * sa;
        }
    } else {
        if (o >= OUT) return;
        const int wbits = 8 - u;
        const uint32_t umask = (1u << wbits) - 1u, usign = 1u << (wbits - 1);
        const uint32_t smask = (1u << u) - 1u, ssign = 1u << (u - 1);
        for (int blk = b0; blk < b1; ++blk) {
            const float sw = ms::e8m0_to_scale(scale_exp[blk * OUT + o]);
            const float sa = ms::e8m0_to_scale(sa_exp[blk]);
            const long sbase = ((long)blk * OUT + o) * SB;
            uint32_t sreg[3] = {0u,0u,0u};
            #pragma unroll
            for (int i = 0; i < 8; ++i)
                if (i < SB) sreg[i >> 2] |= (uint32_t)shared_cm[sbase + i] << (8 * (i & 3));
            uint32_t ureg[6] = {0u,0u,0u,0u,0u,0u};
            const uint32_t* src = reinterpret_cast<const uint32_t*>(
                                      upper_cm + ((long)blk * OUT + o) * UB);
            #pragma unroll
            for (int i = 0; i < 6; ++i) if (i < (UB >> 2)) ureg[i] = src[i];
            const int gsmask = gs - 1;
            uint64_t ubuf = 0; int unb = 0, uwi = 0;
            uint64_t sbuf = 0; int snb = 0, swi = 0;
            int sh_code = 0;
            const int8_t* qxb = qx_sh + (blk - b0) * BLOCK;
            int idot = 0;
            for (int k = 0; k < BLOCK; ++k) {
                if (unb < wbits) { ubuf |= (uint64_t)ureg[uwi++] << unb; unb += 32; }
                const int up_code = (int)(((uint32_t)ubuf & umask) ^ usign) - (int)usign;
                ubuf >>= wbits; unb -= wbits;
                if ((k & gsmask) == 0) {
                    if (snb < u) { sbuf |= (uint64_t)sreg[swi++] << snb; snb += 32; }
                    sh_code = (int)(((uint32_t)sbuf & smask) ^ ssign) - (int)ssign;
                    sbuf >>= u; snb -= u;
                }
                const int w = up_code * (1 << u) + sh_code;
                idot += w * (int)qxb[k];
            }
            acc += (float)idot * sw * sa;
        }
    }
    partial[(long)sp * OUT + o] = acc;
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
        wonly_gemv_cpasync_kernel<<<dim3(blocks, splitK), threads, smem, at::cuda::getCurrentCUDAStream()>>>(
            reinterpret_cast<const __nv_bfloat16*>(x.data_ptr<at::BFloat16>()),
            scale_exp.data_ptr<int8_t>(), upper.data_ptr<uint8_t>(), shared.data_ptr<uint8_t>(),
            partial.data_ptr<float>(), (int)OUT, (int)NB, (int)u, (int)gs, UB, SB, splitK);
    } else {
        wonly_gemv_splitk_kernel<<<dim3(blocks, splitK), threads, 0, at::cuda::getCurrentCUDAStream()>>>(
            reinterpret_cast<const __nv_bfloat16*>(x.data_ptr<at::BFloat16>()),
            scale_exp.data_ptr<int8_t>(),
            upper.data_ptr<uint8_t>(),
            shared.data_ptr<uint8_t>(),
            partial.data_ptr<float>(),
            (int)OUT, (int)NB, (int)u, (int)gs, UB, SB, splitK);
    }

    gemv_combine_kernel<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        partial.data_ptr<float>(),
        reinterpret_cast<__nv_bfloat16*>(y.data_ptr<at::BFloat16>()),
        (int)OUT, splitK);
    return y;
}

// wide-load GEMV (column-major + streaming unpack), all u. split-K + combine.
torch::Tensor wonly_gemv_wide_cuda(
        torch::Tensor x, torch::Tensor scale_exp,
        torch::Tensor upper_cm, torch::Tensor shared_cm,
        int64_t OUT, int64_t NB, int64_t u, int64_t gs) {
    TORCH_CHECK(x.is_cuda() && x.scalar_type() == torch::kBFloat16, "x must be CUDA bf16");
    const int wbits = 8 - (int)u;
    const int UB = BLOCK * wbits / 8;
    const int SB = ((BLOCK / (int)gs) * (int)u + 7) / 8;
    auto y = torch::empty({OUT}, x.options());
    const int threads = 128;
    const int blocks = (int)((OUT + threads - 1) / threads);
    // wide kernel does few loads/block -> needs MORE splits for MLP (the
    // narrow-load kernels saturate at mult~3; wide wants ~16). See change.md Phase 16.
    const int splitK = ms::gemv_splitk_count(blocks, (int)NB, 16);
    auto partial = torch::empty({(int64_t)splitK, OUT}, x.options().dtype(torch::kFloat32));

    // separated-scale dot helps the extraction-bound u2/u3 paths; for u4 (memory-bound)
    // the group-sum bookkeeping adds unhidden latency -> default off there. MS_GEMV_SEPSC forces.
    int sepsc = ((int)u != 4) ? 1 : 0;
    if (const char* e = getenv("MS_GEMV_SEPSC")) sepsc = atoi(e) != 0 ? 1 : 0;
    auto launch = [&](auto U4tag) {
        wonly_gemv_wide_kernel<decltype(U4tag)::value><<<dim3(blocks, splitK), threads, 0, at::cuda::getCurrentCUDAStream()>>>(
            reinterpret_cast<const __nv_bfloat16*>(x.data_ptr<at::BFloat16>()),
            scale_exp.data_ptr<int8_t>(),
            upper_cm.data_ptr<uint8_t>(), shared_cm.data_ptr<uint8_t>(),
            partial.data_ptr<float>(), (int)OUT, (int)NB, (int)u, (int)gs, UB, SB, splitK, sepsc);
    };
    if ((int)u == 4) launch(std::true_type{});
    else             launch(std::false_type{});

    gemv_combine_kernel<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        partial.data_ptr<float>(),
        reinterpret_cast<__nv_bfloat16*>(y.data_ptr<at::BFloat16>()), (int)OUT, splitK);
    return y;
}

// batched-decode wide GEMV: x [M,K] -> y [M,OUT]. MR (compile-time rows/block) = next
// pow2 of M capped 32; M>MR tiles. Amortizes the weight read over M (the decode win).
torch::Tensor wonly_gemv_batched_cuda(
        torch::Tensor x, torch::Tensor scale_exp, torch::Tensor upper_cm, torch::Tensor shared_cm,
        int64_t M, int64_t OUT, int64_t NB, int64_t u, int64_t gs) {
    TORCH_CHECK(x.is_cuda() && x.scalar_type() == torch::kBFloat16, "x must be CUDA bf16");
    const int wbits = 8 - (int)u, UB = BLOCK * wbits / 8, SB = ((BLOCK / (int)gs) * (int)u + 7) / 8;
    auto y = torch::empty({M, OUT}, x.options());
    const int threads = 128, blocks = (int)((OUT + threads - 1) / threads);
    const int splitK = ms::gemv_splitk_count(blocks, (int)NB, 16);
    int MR = 1; while (MR < (int)M && MR < 32) MR <<= 1;
    const int nTiles = (int)((M + MR - 1) / MR);
    auto partial = torch::empty({(int64_t)splitK, M, OUT}, x.options().dtype(torch::kFloat32));
    dim3 grid(blocks, splitK, nTiles);
    auto launch = [&](auto U4tag, auto MRtag) {
        wonly_gemv_batched_kernel<decltype(U4tag)::value, decltype(MRtag)::value>
            <<<grid, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
            reinterpret_cast<const __nv_bfloat16*>(x.data_ptr<at::BFloat16>()),
            scale_exp.data_ptr<int8_t>(), upper_cm.data_ptr<uint8_t>(), shared_cm.data_ptr<uint8_t>(),
            partial.data_ptr<float>(), (int)M, (int)OUT, (int)NB, (int)u, (int)gs, UB, SB, splitK);
    };
#define MRDISP(U4) switch (MR) { \
        case 1:  launch(U4, std::integral_constant<int,1>{});  break; \
        case 2:  launch(U4, std::integral_constant<int,2>{});  break; \
        case 4:  launch(U4, std::integral_constant<int,4>{});  break; \
        case 8:  launch(U4, std::integral_constant<int,8>{});  break; \
        case 16: launch(U4, std::integral_constant<int,16>{}); break; \
        default: launch(U4, std::integral_constant<int,32>{}); break; }
    if ((int)u == 4) { MRDISP(std::true_type{}); } else { MRDISP(std::false_type{}); }
#undef MRDISP
    gemv_combine_batched_kernel<<<dim3(blocks, (int)M), threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        partial.data_ptr<float>(), reinterpret_cast<__nv_bfloat16*>(y.data_ptr<at::BFloat16>()),
        (int)M, (int)OUT, splitK);
    return y;
}

// W+A GEMV: MSAQ-s activation pre-pass (Stage 0) + wide-load int-dot GEMV. x is
// the bf16 decode vector [K]; weights are the column-major planes (same as wide).
torch::Tensor wa_gemv_cuda(
        torch::Tensor x, torch::Tensor scale_exp,
        torch::Tensor upper_cm, torch::Tensor shared_cm,
        int64_t OUT, int64_t NB, int64_t u, int64_t gs) {
    TORCH_CHECK(x.is_cuda() && x.scalar_type() == torch::kBFloat16, "x must be CUDA bf16");
    const int wbits = 8 - (int)u;
    const int UB = BLOCK * wbits / 8;
    const int SB = ((BLOCK / (int)gs) * (int)u + 7) / 8;
    const int K = (int)NB * BLOCK;

    // Stage 0: quantize the activation vector once (M=1) -> int8 word + base exp.
    auto qx = torch::empty({K}, x.options().dtype(torch::kInt8));
    auto sa_exp = torch::empty({NB}, x.options().dtype(torch::kInt8));
    ms_launch_quant_act_msaq(reinterpret_cast<const __nv_bfloat16*>(x.data_ptr<at::BFloat16>()),
                             qx.data_ptr<int8_t>(), sa_exp.data_ptr<int8_t>(),
                             1, K, (int)NB, (int)u, (int)gs);

    auto y = torch::empty({OUT}, x.options());
    const int threads = 128;
    const int blocks = (int)((OUT + threads - 1) / threads);
    const int splitK = ms::gemv_splitk_count(blocks, (int)NB, 16);
    auto partial = torch::empty({(int64_t)splitK, OUT}, x.options().dtype(torch::kFloat32));
    const int per = ((int)NB + splitK - 1) / splitK;
    const size_t smem = (size_t)per * BLOCK;             // staged qx slice (int8)

    auto launch = [&](auto U4tag) {
        wa_gemv_wide_kernel<decltype(U4tag)::value><<<dim3(blocks, splitK), threads, smem, at::cuda::getCurrentCUDAStream()>>>(
            qx.data_ptr<int8_t>(), sa_exp.data_ptr<int8_t>(), scale_exp.data_ptr<int8_t>(),
            upper_cm.data_ptr<uint8_t>(), shared_cm.data_ptr<uint8_t>(),
            partial.data_ptr<float>(), (int)OUT, (int)NB, (int)u, (int)gs, UB, SB, splitK);
    };
    if ((int)u == 4) launch(std::true_type{});
    else             launch(std::false_type{});

    gemv_combine_kernel<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        partial.data_ptr<float>(),
        reinterpret_cast<__nv_bfloat16*>(y.data_ptr<at::BFloat16>()), (int)OUT, splitK);
    return y;
}
