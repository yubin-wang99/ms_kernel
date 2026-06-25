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

// ---- (u,gs)-SPECIALIZED wide GEMV (u<4): same streaming unpack as the generic kernel
//   above, but U/GS/WBITS/UB/SB are COMPILE-TIME constants. That lets the compiler (a)
//   use constant masks + constant-width shifts (vs runtime variable shifts), (b) fully
//   unroll the UB/4 word-loads with no `i<SB`/`i<UB>>2` bounds branch, and (c) with the
//   k-loop unrolled, STATICALLY resolve the rolling bit-buffer refill schedule so `ureg`
//   is register-resident (the runtime-WBITS kernel can't — the refill cadence depends on
//   wbits). Measured lean-prototype evidence: this takes u2 from ~48 us to ~26-32 us
//   (BW-bound) on a 4096^2 GEMV. Bit-identical to wonly_gemv_wide_kernel<false>.
template<int U_, int GS_>
__global__ void wonly_gemv_wide_uspec(
        const __nv_bfloat16* __restrict__ x,
        const int8_t*  __restrict__ scale_exp,
        const uint8_t* __restrict__ upper_cm,
        const uint8_t* __restrict__ shared_cm,
        float* __restrict__ partial,
        int OUT, int NB, int splitK, int sepsc) {
    constexpr int WBITS = 8 - U_;
    constexpr int UBc = BLOCK * WBITS / 8;               // u2:24 u3:20
    constexpr int SBc = ((BLOCK / GS_) * U_ + 7) / 8;
    constexpr int NW  = UBc / 4;                          // upper words: u2:6 u3:5
    constexpr int NSW = (SBc + 3) / 4 > 0 ? (SBc + 3) / 4 : 1;
    constexpr uint32_t umask = (1u << WBITS) - 1u, usign = 1u << (WBITS - 1);
    constexpr uint32_t smask = (1u << U_) - 1u, ssign = 1u << (U_ - 1);
    constexpr int gsmask = GS_ - 1;
    const int o  = blockIdx.x * blockDim.x + threadIdx.x;
    if (o >= OUT) return;
    const int sp = blockIdx.y;
    const int per = (NB + splitK - 1) / splitK;
    const int b0 = sp * per, b1 = min(b0 + per, NB);
    float acc = 0.0f;
    for (int blk = b0; blk < b1; ++blk) {
        const float scale = ms::e8m0_to_scale(scale_exp[blk * OUT + o]);
        const long sbase = ((long)blk * OUT + o) * SBc;
        uint32_t sreg[NSW] = {0u};
        #pragma unroll
        for (int i = 0; i < SBc; ++i) sreg[i >> 2] |= (uint32_t)shared_cm[sbase + i] << (8 * (i & 3));
        const uint32_t* srcp = reinterpret_cast<const uint32_t*>(
                                   upper_cm + ((long)blk * OUT + o) * UBc);
        uint32_t ureg[NW];
        #pragma unroll
        for (int i = 0; i < NW; ++i) ureg[i] = srcp[i];
        uint64_t ubuf = 0; int unb = 0, uwi = 0;
        uint64_t sbuf = 0; int snb = 0, swi = 0;
        int sh_code = 0;
        float bup = 0.0f, bsh = 0.0f, xsum = 0.0f;
        #pragma unroll
        for (int k = 0; k < BLOCK; ++k) {                 // unrolled -> static buffer schedule
            if (unb < WBITS) { ubuf |= (uint64_t)ureg[uwi++] << unb; unb += 32; }
            const int up_code = (int)(((uint32_t)ubuf & umask) ^ usign) - (int)usign;
            ubuf >>= WBITS; unb -= WBITS;
            const float xk = __bfloat162float(x[blk * BLOCK + k]);
            if ((k & gsmask) == 0) {
                if (sepsc && k > 0) bsh += sh_code * xsum;
                if (snb < U_) { sbuf |= (uint64_t)sreg[swi++] << snb; snb += 32; }
                sh_code = (int)(((uint32_t)sbuf & smask) ^ ssign) - (int)ssign;
                sbuf >>= U_; snb -= U_; xsum = 0.0f;
            }
            if (sepsc) { bup += up_code * xk; xsum += xk; }
            else acc += (static_cast<float>(up_code * (1 << U_) + sh_code) * scale) * xk;
        }
        if (sepsc) { bsh += sh_code * xsum; acc += scale * ((float)(1 << U_) * bup + bsh); }
    }
    partial[(long)sp * OUT + o] = acc;
}

// MS-UNSIGNED wide B=1 GEMV (naive-ms): identical to wonly_gemv_wide_uspec but shared
// is UNSIGNED (no sign-extend) and the non-sepsc word is (up<<U_)|sh.
template<int U_, int GS_>
__global__ void wonly_gemv_wide_unsigned(
        const __nv_bfloat16* __restrict__ x,
        const int8_t*  __restrict__ scale_exp,
        const uint8_t* __restrict__ upper_cm,
        const uint8_t* __restrict__ shared_cm,
        float* __restrict__ partial,
        int OUT, int NB, int splitK, int sepsc) {
    constexpr int WBITS = 8 - U_;
    constexpr int UBc = BLOCK * WBITS / 8;
    constexpr int SBc = ((BLOCK / GS_) * U_ + 7) / 8;
    constexpr int NW  = UBc / 4;
    constexpr int NSW = (SBc + 3) / 4 > 0 ? (SBc + 3) / 4 : 1;
    constexpr uint32_t umask = (1u << WBITS) - 1u, usign = 1u << (WBITS - 1);
    constexpr uint32_t smask = (1u << U_) - 1u;
    constexpr int gsmask = GS_ - 1;
    const int o  = blockIdx.x * blockDim.x + threadIdx.x;
    if (o >= OUT) return;
    const int sp = blockIdx.y;
    const int per = (NB + splitK - 1) / splitK;
    const int b0 = sp * per, b1 = min(b0 + per, NB);
    float acc = 0.0f;
    for (int blk = b0; blk < b1; ++blk) {
        const float scale = ms::e8m0_to_scale(scale_exp[blk * OUT + o]);
        const long sbase = ((long)blk * OUT + o) * SBc;
        uint32_t sreg[NSW] = {0u};
        #pragma unroll
        for (int i = 0; i < SBc; ++i) sreg[i >> 2] |= (uint32_t)shared_cm[sbase + i] << (8 * (i & 3));
        const uint32_t* srcp = reinterpret_cast<const uint32_t*>(upper_cm + ((long)blk * OUT + o) * UBc);
        uint32_t ureg[NW];
        #pragma unroll
        for (int i = 0; i < NW; ++i) ureg[i] = srcp[i];
        uint64_t ubuf = 0; int unb = 0, uwi = 0;
        uint64_t sbuf = 0; int snb = 0, swi = 0;
        int sh_code = 0;
        float bup = 0.0f, bsh = 0.0f, xsum = 0.0f;
        #pragma unroll
        for (int k = 0; k < BLOCK; ++k) {
            if (unb < WBITS) { ubuf |= (uint64_t)ureg[uwi++] << unb; unb += 32; }
            const int up_code = (int)(((uint32_t)ubuf & umask) ^ usign) - (int)usign;
            ubuf >>= WBITS; unb -= WBITS;
            const float xk = __bfloat162float(x[blk * BLOCK + k]);
            if ((k & gsmask) == 0) {
                if (sepsc && k > 0) bsh += sh_code * xsum;
                if (snb < U_) { sbuf |= (uint64_t)sreg[swi++] << snb; snb += 32; }
                sh_code = (int)((uint32_t)sbuf & smask);   // UNSIGNED
                sbuf >>= U_; snb -= U_; xsum = 0.0f;
            }
            if (sepsc) { bup += up_code * xk; xsum += xk; }
            else acc += (static_cast<float>((up_code << U_) | sh_code) * scale) * xk;
        }
        if (sepsc) { bsh += sh_code * xsum; acc += scale * ((float)(1 << U_) * bup + bsh); }
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
    // SHARED-ACTIVATION (see wonly_gemv_batched_uspec): stage [MR][BLOCK] activation in
    // shared once per K-block; the 128 output-column threads broadcast-read it.
    __shared__ float As[MR][BLOCK];
    const int o = blockIdx.x * blockDim.x + threadIdx.x;
    const int sp = blockIdx.y, row0 = blockIdx.z * MR, tid = threadIdx.x;
    const int per = (NB + splitK - 1) / splitK, b0 = sp * per, b1 = min(b0 + per, NB);
    const int K = NB * BLOCK, gs_shift = __ffs(gs) - 1;
    const bool active = (o < OUT);
    float acc[MR];
    #pragma unroll
    for (int j = 0; j < MR; ++j) acc[j] = 0.0f;
    for (int blk = b0; blk < b1; ++blk) {
        for (int idx = tid; idx < MR * BLOCK; idx += blockDim.x) {
            const int m = idx / BLOCK, k = idx % BLOCK;
            As[m][k] = (row0 + m < M) ? __bfloat162float(x[(long)(row0 + m) * K + blk * BLOCK + k]) : 0.0f;
        }
        __syncthreads();
        if (active) {
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
                #pragma unroll
                for (int j = 0; j < MR; ++j) acc[j] += wf * As[j][k];
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
                #pragma unroll
                for (int j = 0; j < MR; ++j) acc[j] += wf * As[j][k];
            }
        }
        }
        __syncthreads();
    }
    if (active) {
        #pragma unroll
        for (int j = 0; j < MR; ++j) { const int m = row0 + j; if (m < M) partial[((long)sp * M + m) * OUT + o] = acc[j]; }
    }
}

// (u,gs)-specialized batched W-only (u<4). Same as wonly_gemv_batched_kernel<false,MR>
// but U/GS are compile-time -> register-resident streaming unpack via the helper.
template<int U_, int GS_, int MR>
__global__ void wonly_gemv_batched_uspec(
        const __nv_bfloat16* __restrict__ x,
        const int8_t*  __restrict__ scale_exp, const uint8_t* __restrict__ upper_cm,
        const uint8_t* __restrict__ shared_cm,
        float* __restrict__ partial, int M, int OUT, int NB, int splitK) {
    // SHARED-ACTIVATION decode GEMV. ncu showed the per-column global reload of x[m,kk]
    // pegged L1 at 87% (DRAM only 17%) -> the activation, not the weight, bounded it.
    // Stage the [MR][BLOCK] activation tile into shared once per K-block; all threads in
    // the block (= 128 output columns) then broadcast-read it -> L1 traffic collapses,
    // leaving the unpack ALU + weight DRAM as the (much smaller) bound.
    __shared__ float As[MR][BLOCK];
    const int o = blockIdx.x * blockDim.x + threadIdx.x;
    const int sp = blockIdx.y, row0 = blockIdx.z * MR, tid = threadIdx.x;
    const int per = (NB + splitK - 1) / splitK, b0 = sp * per, b1 = min(b0 + per, NB);
    const int K = NB * BLOCK;
    const bool active = (o < OUT);
    float acc[MR];
    #pragma unroll
    for (int j = 0; j < MR; ++j) acc[j] = 0.0f;
    for (int blk = b0; blk < b1; ++blk) {
        for (int idx = tid; idx < MR * BLOCK; idx += blockDim.x) {
            const int m = idx / BLOCK, k = idx % BLOCK;
            As[m][k] = (row0 + m < M) ? __bfloat162float(x[(long)(row0 + m) * K + blk * BLOCK + k]) : 0.0f;
        }
        __syncthreads();
        if (active) {
            const float scale = ms::e8m0_to_scale(scale_exp[blk * OUT + o]);
            const long base = (long)blk * OUT + o;
            ms::stream_block_uspec<U_, GS_>(upper_cm, shared_cm, base, [&](int k, int word) {
                const float wf = (float)word * scale;
                #pragma unroll
                for (int j = 0; j < MR; ++j) acc[j] += wf * As[j][k];
            });
        }
        __syncthreads();
    }
    if (active) {
        #pragma unroll
        for (int j = 0; j < MR; ++j) { const int m = row0 + j; if (m < M) partial[((long)sp * M + m) * OUT + o] = acc[j]; }
    }
}

// MS-UNSIGNED batched W-only GEMV: identical to wonly_gemv_batched_uspec but the
// weight word comes from the unsigned (OR-concat) unpack. Same upper_cm/shared_cm
// planes (from pack_weight_unsigned). Tests whether unsigned-shared cuts kernel time.
template<int U_, int GS_, int MR>
__global__ void wonly_gemv_batched_unsigned(
        const __nv_bfloat16* __restrict__ x,
        const int8_t*  __restrict__ scale_exp, const uint8_t* __restrict__ upper_cm,
        const uint8_t* __restrict__ shared_cm,
        float* __restrict__ partial, int M, int OUT, int NB, int splitK) {
    __shared__ float As[MR][BLOCK];
    const int o = blockIdx.x * blockDim.x + threadIdx.x;
    const int sp = blockIdx.y, row0 = blockIdx.z * MR, tid = threadIdx.x;
    const int per = (NB + splitK - 1) / splitK, b0 = sp * per, b1 = min(b0 + per, NB);
    const int K = NB * BLOCK;
    const bool active = (o < OUT);
    float acc[MR];
    #pragma unroll
    for (int j = 0; j < MR; ++j) acc[j] = 0.0f;
    for (int blk = b0; blk < b1; ++blk) {
        for (int idx = tid; idx < MR * BLOCK; idx += blockDim.x) {
            const int m = idx / BLOCK, k = idx % BLOCK;
            As[m][k] = (row0 + m < M) ? __bfloat162float(x[(long)(row0 + m) * K + blk * BLOCK + k]) : 0.0f;
        }
        __syncthreads();
        if (active) {
            const float scale = ms::e8m0_to_scale(scale_exp[blk * OUT + o]);
            const long base = (long)blk * OUT + o;
            ms::stream_block_uspec_unsigned<U_, GS_>(upper_cm, shared_cm, base, [&](int k, int word) {
                const float wf = (float)word * scale;
                #pragma unroll
                for (int j = 0; j < MR; ++j) acc[j] += wf * As[j][k];
            });
        }
        __syncthreads();
    }
    if (active) {
        #pragma unroll
        for (int j = 0; j < MR; ++j) { const int m = row0 + j; if (m < M) partial[((long)sp * M + m) * OUT + o] = acc[j]; }
    }
}

// NIBBLE-RELAYOUT batched W-only GEMV. Same shared-activation structure as
// wonly_gemv_batched_uspec, but the weight comes from the 3 re-layout planes
// (hi4/lowun/shared) and is unpacked via stream_block_relayout (high nibble = bfe,
// HW sign-extend, no straddle). Prototype to test the user's nibble-align idea.
template<int U_, int GS_, int MR>
__global__ void wonly_gemv_batched_relayout(
        const __nv_bfloat16* __restrict__ x,
        const int8_t*  __restrict__ scale_exp, const uint8_t* __restrict__ hi4_cm,
        const uint8_t* __restrict__ lowun_cm, const uint8_t* __restrict__ shared_cm,
        float* __restrict__ partial, int M, int OUT, int NB, int splitK) {
    __shared__ float As[MR][BLOCK];
    const int o = blockIdx.x * blockDim.x + threadIdx.x;
    const int sp = blockIdx.y, row0 = blockIdx.z * MR, tid = threadIdx.x;
    const int per = (NB + splitK - 1) / splitK, b0 = sp * per, b1 = min(b0 + per, NB);
    const int K = NB * BLOCK;
    const bool active = (o < OUT);
    float acc[MR];
    #pragma unroll
    for (int j = 0; j < MR; ++j) acc[j] = 0.0f;
    for (int blk = b0; blk < b1; ++blk) {
        for (int idx = tid; idx < MR * BLOCK; idx += blockDim.x) {
            const int m = idx / BLOCK, k = idx % BLOCK;
            As[m][k] = (row0 + m < M) ? __bfloat162float(x[(long)(row0 + m) * K + blk * BLOCK + k]) : 0.0f;
        }
        __syncthreads();
        if (active) {
            const float scale = ms::e8m0_to_scale(scale_exp[blk * OUT + o]);
            const long base = (long)blk * OUT + o;
            ms::stream_block_relayout<U_, GS_>(hi4_cm, lowun_cm, shared_cm, base, [&](int k, int word) {
                const float wf = (float)word * scale;
                #pragma unroll
                for (int j = 0; j < MR; ++j) acc[j] += wf * As[j][k];
            });
        }
        __syncthreads();
    }
    if (active) {
        #pragma unroll
        for (int j = 0; j < MR; ++j) { const int m = row0 + j; if (m < M) partial[((long)sp * M + m) * OUT + o] = acc[j]; }
    }
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

// ---- BATCHED-DECODE W+A GEMV (M=B small): weight word unpacked once, MR int-dots ---
//   The W+A wide GEMV extended to M rows: per block unpack the weight word ONCE and run
//   MR integer dots (one per activation row), folding the two block scales once per row.
//   qx [M,K] int8 word, sa_exp [M,NB]. Amortizes the weight read over B (the decode win).
template<bool U4, int MR>
__global__ void wa_gemv_batched_kernel(
        const int8_t* __restrict__ qx, const int8_t* __restrict__ sa_exp,
        const int8_t* __restrict__ scale_exp, const uint8_t* __restrict__ upper_cm,
        const uint8_t* __restrict__ shared_cm,
        float* __restrict__ partial, int M, int OUT, int NB, int u, int gs, int UB, int SB, int splitK) {
    // SHARED-ACTIVATION (see wonly_gemv_batched_uspec): stage int8 [MR][BLOCK] activation
    // in shared once per K-block; the 128 output-column threads broadcast-read it.
    __shared__ int8_t As[MR][BLOCK];
    const int o = blockIdx.x * blockDim.x + threadIdx.x;
    const int sp = blockIdx.y, row0 = blockIdx.z * MR, tid = threadIdx.x;
    const int per = (NB + splitK - 1) / splitK, b0 = sp * per, b1 = min(b0 + per, NB);
    const int K = NB * BLOCK, gs_shift = __ffs(gs) - 1;
    const bool active = (o < OUT);
    float acc[MR];
    #pragma unroll
    for (int j = 0; j < MR; ++j) acc[j] = 0.0f;
    for (int blk = b0; blk < b1; ++blk) {
        for (int idx = tid; idx < MR * BLOCK; idx += blockDim.x) {
            const int m = idx / BLOCK, k = idx % BLOCK;
            As[m][k] = (row0 + m < M) ? qx[(long)(row0 + m) * K + blk * BLOCK + k] : (int8_t)0;
        }
        __syncthreads();
        if (active) {
        const float sw = ms::e8m0_to_scale(scale_exp[blk * OUT + o]);
        const long base = (long)blk * OUT + o;
        uint8_t sb[8];
        const long sbase = base * SB;
        #pragma unroll
        for (int i = 0; i < 8; ++i) if (i < SB) sb[i] = shared_cm[sbase + i];
        int idot[MR];
        #pragma unroll
        for (int j = 0; j < MR; ++j) idot[j] = 0;
        if constexpr (U4) {
            const uint4 up4 = *reinterpret_cast<const uint4*>(upper_cm + base * UB);
            const uint32_t uw[4] = { up4.x, up4.y, up4.z, up4.w };
            #pragma unroll
            for (int k = 0; k < BLOCK; ++k) {
                const int up_code = ms::bfe_s32((int)uw[k >> 3], (k & 7) * 4, 4);
                const int g = k >> gs_shift;
                const int sh_code = ms::bfe_s32((int)sb[g >> 1], (g & 1) * 4, 4);
                const int w = up_code * 16 + sh_code;
                #pragma unroll
                for (int j = 0; j < MR; ++j) idot[j] += w * (int)As[j][k];
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
                const int w = up_code * (1 << u) + sh_code;
                #pragma unroll
                for (int j = 0; j < MR; ++j) idot[j] += w * (int)As[j][k];
            }
        }
        #pragma unroll
        for (int j = 0; j < MR; ++j) { const int m = row0 + j; if (m < M) acc[j] += (float)idot[j] * sw * ms::e8m0_to_scale(sa_exp[m * NB + blk]); }
        }
        __syncthreads();
    }
    if (active) {
        #pragma unroll
        for (int j = 0; j < MR; ++j) { const int m = row0 + j; if (m < M) partial[((long)sp * M + m) * OUT + o] = acc[j]; }
    }
}

// (u,gs)-specialized W+A decode (u<4): same as wa_gemv_wide_kernel<false> with a
// compile-time register-resident unpack feeding the int dot.
template<int U_, int GS_>
__global__ void wa_gemv_wide_uspec(
        const int8_t*  __restrict__ qx, const int8_t* __restrict__ sa_exp,
        const int8_t*  __restrict__ scale_exp, const uint8_t* __restrict__ upper_cm,
        const uint8_t* __restrict__ shared_cm,
        float* __restrict__ partial, int OUT, int NB, int splitK) {
    const int o   = blockIdx.x * blockDim.x + threadIdx.x;
    const int sp  = blockIdx.y;
    const int per = (NB + splitK - 1) / splitK;
    const int b0  = sp * per, b1 = min(b0 + per, NB);
    extern __shared__ int8_t qx_sh[];
    const int slice = (b1 - b0) * BLOCK;
    for (int i = threadIdx.x; i < slice; i += blockDim.x) qx_sh[i] = qx[b0 * BLOCK + i];
    __syncthreads();
    if (o >= OUT) return;
    float acc = 0.0f;
    for (int blk = b0; blk < b1; ++blk) {
        const float sw = ms::e8m0_to_scale(scale_exp[blk * OUT + o]);
        const float sa = ms::e8m0_to_scale(sa_exp[blk]);
        const long base = (long)blk * OUT + o;
        const int8_t* qxb = qx_sh + (blk - b0) * BLOCK;
        int idot = 0;
        ms::stream_block_uspec<U_, GS_>(upper_cm, shared_cm, base, [&](int k, int word) {
            idot += word * (int)qxb[k];
        });
        acc += (float)idot * sw * sa;
    }
    partial[(long)sp * OUT + o] = acc;
}

// (u,gs)-specialized batched W+A (u<4): compile-time unpack + MR int dots.
template<int U_, int GS_, int MR>
__global__ void wa_gemv_batched_uspec(
        const int8_t* __restrict__ qx, const int8_t* __restrict__ sa_exp,
        const int8_t* __restrict__ scale_exp, const uint8_t* __restrict__ upper_cm,
        const uint8_t* __restrict__ shared_cm,
        float* __restrict__ partial, int M, int OUT, int NB, int splitK) {
    // DEQUANT-IN-STAGING decode W+A. At decode (memory-bound) the int8 dot has no
    // compute advantage -- (qa.qw).sa.sw == (qa.sa).(qw.sw) -- and the int path was 70us
    // (vs W-only's float 40us): IMAD < FFMA on Ampere, int8 shared reads, and idot[MR]+
    // acc[MR] = 2x accumulators capping occupancy. So fold sa into the staged activation
    // (As = qx * sa, in float) and run the W-only float MAC: one FFMA per element, single
    // acc[MR]. Numerically identical W+A, at W-only speed. Sas staged first (one sync),
    // then As, then the MAC -> three light syncs per block.
    __shared__ float As[MR][BLOCK];
    __shared__ float Sas[MR];
    const int o = blockIdx.x * blockDim.x + threadIdx.x;
    const int sp = blockIdx.y, row0 = blockIdx.z * MR, tid = threadIdx.x;
    const int per = (NB + splitK - 1) / splitK, b0 = sp * per, b1 = min(b0 + per, NB);
    const int K = NB * BLOCK;
    const bool active = (o < OUT);
    float acc[MR];
    #pragma unroll
    for (int j = 0; j < MR; ++j) acc[j] = 0.0f;
    for (int blk = b0; blk < b1; ++blk) {
        if (tid < MR) Sas[tid] = (row0 + tid < M) ? ms::e8m0_to_scale(sa_exp[(row0 + tid) * NB + blk]) : 0.0f;
        __syncthreads();
        for (int idx = tid; idx < MR * BLOCK; idx += blockDim.x) {
            const int m = idx / BLOCK, k = idx % BLOCK;
            As[m][k] = (row0 + m < M) ? (float)qx[(long)(row0 + m) * K + blk * BLOCK + k] * Sas[m] : 0.0f;
        }
        __syncthreads();
        if (active) {
            const float sw = ms::e8m0_to_scale(scale_exp[blk * OUT + o]);
            const long base = (long)blk * OUT + o;
            ms::stream_block_uspec<U_, GS_>(upper_cm, shared_cm, base, [&](int k, int word) {
                const float wf = (float)word * sw;
                #pragma unroll
                for (int j = 0; j < MR; ++j) acc[j] += wf * As[j][k];
            });
        }
        __syncthreads();
    }
    if (active) {
        #pragma unroll
        for (int j = 0; j < MR; ++j) { const int m = row0 + j; if (m < M) partial[((long)sp * M + m) * OUT + o] = acc[j]; }
    }
}

// MS-UNSIGNED W+A wide B=1 (naive-ms): wa_gemv_wide_uspec with unsigned weight unpack.
template<int U_, int GS_>
__global__ void wa_gemv_wide_unsigned(
        const int8_t*  __restrict__ qx, const int8_t* __restrict__ sa_exp,
        const int8_t*  __restrict__ scale_exp, const uint8_t* __restrict__ upper_cm,
        const uint8_t* __restrict__ shared_cm,
        float* __restrict__ partial, int OUT, int NB, int splitK) {
    const int o   = blockIdx.x * blockDim.x + threadIdx.x;
    const int sp  = blockIdx.y;
    const int per = (NB + splitK - 1) / splitK;
    const int b0  = sp * per, b1 = min(b0 + per, NB);
    extern __shared__ int8_t qx_sh[];
    const int slice = (b1 - b0) * BLOCK;
    for (int i = threadIdx.x; i < slice; i += blockDim.x) qx_sh[i] = qx[b0 * BLOCK + i];
    __syncthreads();
    if (o >= OUT) return;
    float acc = 0.0f;
    for (int blk = b0; blk < b1; ++blk) {
        const float sw = ms::e8m0_to_scale(scale_exp[blk * OUT + o]);
        const float sa = ms::e8m0_to_scale(sa_exp[blk]);
        const long base = (long)blk * OUT + o;
        const int8_t* qxb = qx_sh + (blk - b0) * BLOCK;
        int idot = 0;
        ms::stream_block_uspec_unsigned<U_, GS_>(upper_cm, shared_cm, base, [&](int k, int word) {
            idot += word * (int)qxb[k];
        });
        acc += (float)idot * sw * sa;
    }
    partial[(long)sp * OUT + o] = acc;
}

// MS-UNSIGNED W+A batched (naive-ms): wa_gemv_batched_uspec with unsigned weight unpack.
template<int U_, int GS_, int MR>
__global__ void wa_gemv_batched_unsigned(
        const int8_t* __restrict__ qx, const int8_t* __restrict__ sa_exp,
        const int8_t* __restrict__ scale_exp, const uint8_t* __restrict__ upper_cm,
        const uint8_t* __restrict__ shared_cm,
        float* __restrict__ partial, int M, int OUT, int NB, int splitK) {
    __shared__ float As[MR][BLOCK];
    __shared__ float Sas[MR];
    const int o = blockIdx.x * blockDim.x + threadIdx.x;
    const int sp = blockIdx.y, row0 = blockIdx.z * MR, tid = threadIdx.x;
    const int per = (NB + splitK - 1) / splitK, b0 = sp * per, b1 = min(b0 + per, NB);
    const int K = NB * BLOCK;
    const bool active = (o < OUT);
    float acc[MR];
    #pragma unroll
    for (int j = 0; j < MR; ++j) acc[j] = 0.0f;
    for (int blk = b0; blk < b1; ++blk) {
        if (tid < MR) Sas[tid] = (row0 + tid < M) ? ms::e8m0_to_scale(sa_exp[(row0 + tid) * NB + blk]) : 0.0f;
        __syncthreads();
        for (int idx = tid; idx < MR * BLOCK; idx += blockDim.x) {
            const int m = idx / BLOCK, k = idx % BLOCK;
            As[m][k] = (row0 + m < M) ? (float)qx[(long)(row0 + m) * K + blk * BLOCK + k] * Sas[m] : 0.0f;
        }
        __syncthreads();
        if (active) {
            const float sw = ms::e8m0_to_scale(scale_exp[blk * OUT + o]);
            const long base = (long)blk * OUT + o;
            ms::stream_block_uspec_unsigned<U_, GS_>(upper_cm, shared_cm, base, [&](int k, int word) {
                const float wf = (float)word * sw;
                #pragma unroll
                for (int j = 0; j < MR; ++j) acc[j] += wf * As[j][k];
            });
        }
        __syncthreads();
    }
    if (active) {
        #pragma unroll
        for (int j = 0; j < MR; ++j) { const int m = row0 + j; if (m < M) partial[((long)sp * M + m) * OUT + o] = acc[j]; }
    }
}

// FULLY-FUSED decode W+A: fake-quant the activation INSIDE the staging (no separate
// quant_act kernel, no int8 qx global round-trip). One thread per row 2-passes its 32-elem
// block (amax -> E8M0 scale; round+clamp to int8 -> As = q*sa, float), then the W-only
// float MAC runs. The activation is 8-bit (not sub-byte) and consumed immediately (no
// storage), so plain int8 fake-quant is the right model (>= MSAQ-shared accuracy). 2-pass
// avoids a 32-elem register array (occupancy). Cost: the fake-quant is redone per
// output-column block (~OUT/128 x), but it hides under the l1tex-bound MAC (measured).
template<int U_, int GS_, int MR>
__global__ void wa_gemv_batched_fused_uspec(
        const __nv_bfloat16* __restrict__ x,         // bf16 [M,K] RAW activation (not pre-quantized)
        const int8_t* __restrict__ scale_exp, const uint8_t* __restrict__ upper_cm,
        const uint8_t* __restrict__ shared_cm,
        float* __restrict__ partial, int M, int OUT, int NB, int splitK) {
    __shared__ float As[MR][BLOCK];
    const int o = blockIdx.x * blockDim.x + threadIdx.x;
    const int sp = blockIdx.y, row0 = blockIdx.z * MR, tid = threadIdx.x;
    const int per = (NB + splitK - 1) / splitK, b0 = sp * per, b1 = min(b0 + per, NB);
    const int K = NB * BLOCK;
    const bool active = (o < OUT);
    float acc[MR];
    #pragma unroll
    for (int j = 0; j < MR; ++j) acc[j] = 0.0f;
    for (int blk = b0; blk < b1; ++blk) {
        if (tid < MR) {
            const int m = row0 + tid;
            if (m < M) {
                const long xb = (long)m * K + blk * BLOCK;
                float amax = 1e-30f;
                #pragma unroll
                for (int k = 0; k < BLOCK; ++k) amax = fmaxf(amax, fabsf(__bfloat162float(x[xb + k])));
                const int ea = ms::e8m0_exp_from_amax(amax);
                const float sa = ms::e8m0_to_scale((int8_t)ea), inv = exp2f(-(float)ea);
                #pragma unroll
                for (int k = 0; k < BLOCK; ++k) {
                    const int q = max(-128, min(127, (int)rintf(__bfloat162float(x[xb + k]) * inv)));
                    As[tid][k] = (float)q * sa;
                }
            } else {
                #pragma unroll
                for (int k = 0; k < BLOCK; ++k) As[tid][k] = 0.0f;
            }
        }
        __syncthreads();
        if (active) {
            const float sw = ms::e8m0_to_scale(scale_exp[blk * OUT + o]);
            const long base = (long)blk * OUT + o;
            ms::stream_block_uspec<U_, GS_>(upper_cm, shared_cm, base, [&](int k, int word) {
                const float wf = (float)word * sw;
                #pragma unroll
                for (int j = 0; j < MR; ++j) acc[j] += wf * As[j][k];
            });
        }
        __syncthreads();
    }
    if (active) {
        #pragma unroll
        for (int j = 0; j < MR; ++j) { const int m = row0 + j; if (m < M) partial[((long)sp * M + m) * OUT + o] = acc[j]; }
    }
}

// ---- DEQUANT the whole weight to a bf16 [OUT,K] buffer (column-major wide-load) -----
//   For PREFILL the fused per-tile dequant runs the tensor cores at ~11% (it starves the
//   MMA and re-dequants the weight M/TBM times). Instead dequant ONCE here (memory-bound,
//   amortized over the whole GEMM) and let cuBLAS do the bf16 GEMM at full speed. Thread o
//   owns a column: wide-load its UB bytes/block, streaming-unpack 32 codes, write bf16.
//   Output is COLUMN-MAJOR-of-the-GEMM: Wbf[K,OUT] (= X[M,K] @ Wbf[K,OUT] with no
//   transpose). Thread (o, blk=blockIdx.y) writes Wbf[(blk*32+k)*OUT + o] -> consecutive
//   `o` across the warp = coalesced stores; grid.y over NB gives OUT*NB threads (full
//   occupancy). Memory-bound (~2.7x weight bytes), one-time before the cuBLAS GEMM.
__global__ void ms_dequant_bf16_kernel(
        const int8_t* __restrict__ scale_exp, const uint8_t* __restrict__ upper_cm,
        const uint8_t* __restrict__ shared_cm, __nv_bfloat16* __restrict__ Wbf,
        int OUT, int K, int NB, int u, int gs, int UB, int SB) {
    const int o = blockIdx.x * blockDim.x + threadIdx.x;
    const int blk = blockIdx.y;
    if (o >= OUT) return;
    const int wbits = 8 - u, gsmask = gs - 1;
    const uint32_t umask = (1u << wbits) - 1u, usign = 1u << (wbits - 1);
    const uint32_t smask = (1u << u) - 1u, ssign = 1u << (u - 1);
    const float scale = ms::e8m0_to_scale(scale_exp[blk * OUT + o]);
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
    for (int k = 0; k < BLOCK; ++k) {
        if (unb < wbits) { ubuf |= (uint64_t)ureg[uwi++] << unb; unb += 32; }
        const int up_code = (int)(((uint32_t)ubuf & umask) ^ usign) - (int)usign;
        ubuf >>= wbits; unb -= wbits;
        if ((k & gsmask) == 0) {
            if (snb < u) { sbuf |= (uint64_t)sreg[swi++] << snb; snb += 32; }
            sh_code = (int)(((uint32_t)sbuf & smask) ^ ssign) - (int)ssign;
            sbuf >>= u; snb -= u;
        }
        Wbf[((long)(blk * BLOCK + k)) * OUT + o] = __float2bfloat16((float)(up_code * (1 << u) + sh_code) * scale);
    }
}

// MS-UNSIGNED dequant: shared UNSIGNED (no sign-extend), word = (up<<u)|sh. For naive-ms.
__global__ void ms_dequant_bf16_unsigned_kernel(
        const int8_t* __restrict__ scale_exp, const uint8_t* __restrict__ upper_cm,
        const uint8_t* __restrict__ shared_cm, __nv_bfloat16* __restrict__ Wbf,
        int OUT, int K, int NB, int u, int gs, int UB, int SB) {
    const int o = blockIdx.x * blockDim.x + threadIdx.x;
    const int blk = blockIdx.y;
    if (o >= OUT) return;
    const int wbits = 8 - u, gsmask = gs - 1;
    const uint32_t umask = (1u << wbits) - 1u, usign = 1u << (wbits - 1);
    const uint32_t smask = (1u << u) - 1u;
    const float scale = ms::e8m0_to_scale(scale_exp[blk * OUT + o]);
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
    for (int k = 0; k < BLOCK; ++k) {
        if (unb < wbits) { ubuf |= (uint64_t)ureg[uwi++] << unb; unb += 32; }
        const int up_code = (int)(((uint32_t)ubuf & umask) ^ usign) - (int)usign;
        ubuf >>= wbits; unb -= wbits;
        if ((k & gsmask) == 0) {
            if (snb < u) { sbuf |= (uint64_t)sreg[swi++] << snb; snb += 32; }
            sh_code = (int)((uint32_t)sbuf & smask);     // UNSIGNED
            sbuf >>= u; snb -= u;
        }
        Wbf[((long)(blk * BLOCK + k)) * OUT + o] = __float2bfloat16((float)((up_code << u) | sh_code) * scale);
    }
}

} // namespace

// Dequant the MSAQ weight to a bf16 [OUT,K] buffer (for prefill: dequant-once + cuBLAS).
torch::Tensor ms_dequant_bf16_cuda(
        torch::Tensor scale_exp, torch::Tensor upper_cm, torch::Tensor shared_cm,
        int64_t OUT, int64_t K, int64_t NB, int64_t u, int64_t gs) {
    const int UB = 32 * (8 - (int)u) / 8, SB = ((32 / (int)gs) * (int)u + 7) / 8;
    auto W = torch::empty({K, OUT}, torch::dtype(torch::kBFloat16).device(scale_exp.device()));  // [K,OUT]
    const int threads = 256, blocks = (int)((OUT + threads - 1) / threads);
    ms_dequant_bf16_kernel<<<dim3(blocks, (int)NB), threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        scale_exp.data_ptr<int8_t>(), upper_cm.data_ptr<uint8_t>(), shared_cm.data_ptr<uint8_t>(),
        reinterpret_cast<__nv_bfloat16*>(W.data_ptr<at::BFloat16>()),
        (int)OUT, (int)K, (int)NB, (int)u, (int)gs, UB, SB);
    return W;
}

torch::Tensor ms_dequant_bf16_unsigned_cuda(
        torch::Tensor scale_exp, torch::Tensor upper_cm, torch::Tensor shared_cm,
        int64_t OUT, int64_t K, int64_t NB, int64_t u, int64_t gs) {
    const int UB = 32 * (8 - (int)u) / 8, SB = ((32 / (int)gs) * (int)u + 7) / 8;
    auto W = torch::empty({K, OUT}, torch::dtype(torch::kBFloat16).device(scale_exp.device()));
    const int threads = 256, blocks = (int)((OUT + threads - 1) / threads);
    ms_dequant_bf16_unsigned_kernel<<<dim3(blocks, (int)NB), threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        scale_exp.data_ptr<int8_t>(), upper_cm.data_ptr<uint8_t>(), shared_cm.data_ptr<uint8_t>(),
        reinterpret_cast<__nv_bfloat16*>(W.data_ptr<at::BFloat16>()),
        (int)OUT, (int)K, (int)NB, (int)u, (int)gs, UB, SB);
    return W;
}

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
    // (u,gs)-specialized fast path for u<4 (compile-time constants); the generic kernel
    // is the fallback for any combo not instantiated below. Bit-identical either way.
    // MS_GEMV_NOSPEC=1 forces the generic path (for A/B timing).
    auto launch_spec = [&](auto Ut, auto Gt) {
        wonly_gemv_wide_uspec<decltype(Ut)::value, decltype(Gt)::value><<<dim3(blocks, splitK), threads, 0, at::cuda::getCurrentCUDAStream()>>>(
            reinterpret_cast<const __nv_bfloat16*>(x.data_ptr<at::BFloat16>()),
            scale_exp.data_ptr<int8_t>(),
            upper_cm.data_ptr<uint8_t>(), shared_cm.data_ptr<uint8_t>(),
            partial.data_ptr<float>(), (int)OUT, (int)NB, splitK, sepsc);
    };
    const bool nospec = getenv("MS_GEMV_NOSPEC") && atoi(getenv("MS_GEMV_NOSPEC")) != 0;
    bool launched = false;
    if ((int)u == 4) { launch(std::true_type{}); launched = true; }
    else if (!nospec) {
        #define SPEC(UU, GG) if (!launched && (int)u == UU && (int)gs == GG) { \
            launch_spec(std::integral_constant<int,UU>{}, std::integral_constant<int,GG>{}); launched = true; }
        SPEC(2, 8) SPEC(2, 16) SPEC(2, 4) SPEC(2, 32)
        SPEC(3, 8) SPEC(3, 16) SPEC(3, 4) SPEC(3, 32)
        #undef SPEC
    }
    if (!launched) launch(std::false_type{});   // generic u<4 fallback

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
    auto launch_spec = [&](auto Ut, auto Gt, auto MRtag) {
        wonly_gemv_batched_uspec<decltype(Ut)::value, decltype(Gt)::value, decltype(MRtag)::value>
            <<<grid, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
            reinterpret_cast<const __nv_bfloat16*>(x.data_ptr<at::BFloat16>()),
            scale_exp.data_ptr<int8_t>(), upper_cm.data_ptr<uint8_t>(), shared_cm.data_ptr<uint8_t>(),
            partial.data_ptr<float>(), (int)M, (int)OUT, (int)NB, splitK);
    };
#define MRDISP(U4) switch (MR) { \
        case 1:  launch(U4, std::integral_constant<int,1>{});  break; \
        case 2:  launch(U4, std::integral_constant<int,2>{});  break; \
        case 4:  launch(U4, std::integral_constant<int,4>{});  break; \
        case 8:  launch(U4, std::integral_constant<int,8>{});  break; \
        case 16: launch(U4, std::integral_constant<int,16>{}); break; \
        default: launch(U4, std::integral_constant<int,32>{}); break; }
#define MRDISP_SPEC(UU,GG) switch (MR) { \
        case 1:  launch_spec(std::integral_constant<int,UU>{}, std::integral_constant<int,GG>{}, std::integral_constant<int,1>{});  break; \
        case 2:  launch_spec(std::integral_constant<int,UU>{}, std::integral_constant<int,GG>{}, std::integral_constant<int,2>{});  break; \
        case 4:  launch_spec(std::integral_constant<int,UU>{}, std::integral_constant<int,GG>{}, std::integral_constant<int,4>{});  break; \
        case 8:  launch_spec(std::integral_constant<int,UU>{}, std::integral_constant<int,GG>{}, std::integral_constant<int,8>{});  break; \
        case 16: launch_spec(std::integral_constant<int,UU>{}, std::integral_constant<int,GG>{}, std::integral_constant<int,16>{}); break; \
        default: launch_spec(std::integral_constant<int,UU>{}, std::integral_constant<int,GG>{}, std::integral_constant<int,32>{}); break; }
    const bool nospec = getenv("MS_GEMV_NOSPEC") && atoi(getenv("MS_GEMV_NOSPEC")) != 0;
    bool launched = false;
    if ((int)u == 4) { MRDISP(std::true_type{}); launched = true; }
    else if (!nospec) {
        if      ((int)u==2 && (int)gs==8)  { MRDISP_SPEC(2,8);  launched = true; }
        else if ((int)u==3 && (int)gs==8)  { MRDISP_SPEC(3,8);  launched = true; }
        else if ((int)u==2 && (int)gs==16) { MRDISP_SPEC(2,16); launched = true; }
        else if ((int)u==3 && (int)gs==16) { MRDISP_SPEC(3,16); launched = true; }
    }
    if (!launched) { MRDISP(std::false_type{}); }
#undef MRDISP
#undef MRDISP_SPEC
    gemv_combine_batched_kernel<<<dim3(blocks, (int)M), threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        partial.data_ptr<float>(), reinterpret_cast<__nv_bfloat16*>(y.data_ptr<at::BFloat16>()),
        (int)M, (int)OUT, splitK);
    return y;
}

// VECTORIZED nibble-relayout GEMV (u3/gs16 only): high nibble -> float via
// __byte_perm magic (8 at a time), low_un(1b) folded as a float select. Tests
// whether Marlin-style byte_perm beats the funnel unpack on Blackwell.
template<int MR>
__global__ void wonly_gemv_batched_relayout_vec_u3(
        const __nv_bfloat16* __restrict__ x,
        const int8_t*  __restrict__ scale_exp, const uint8_t* __restrict__ hi4_cm,
        const uint8_t* __restrict__ lowun_cm, const uint8_t* __restrict__ shared_cm,
        float* __restrict__ partial, int M, int OUT, int NB, int splitK) {
    __shared__ float As[MR][BLOCK];
    const int o = blockIdx.x * blockDim.x + threadIdx.x;
    const int sp = blockIdx.y, row0 = blockIdx.z * MR, tid = threadIdx.x;
    const int per = (NB + splitK - 1) / splitK, b0 = sp * per, b1 = min(b0 + per, NB);
    const int K = NB * BLOCK;
    const bool active = (o < OUT);
    float acc[MR];
    #pragma unroll
    for (int j = 0; j < MR; ++j) acc[j] = 0.0f;
    for (int blk = b0; blk < b1; ++blk) {
        for (int idx = tid; idx < MR * BLOCK; idx += blockDim.x) {
            const int m = idx / BLOCK, kk = idx % BLOCK;
            As[m][kk] = (row0 + m < M) ? __bfloat162float(x[(long)(row0 + m) * K + blk * BLOCK + kk]) : 0.0f;
        }
        __syncthreads();
        if (active) {
            const float scale = ms::e8m0_to_scale(scale_exp[blk * OUT + o]);
            const long base = (long)blk * OUT + o;
            const uint32_t* hp = reinterpret_cast<const uint32_t*>(hi4_cm + base * 16);
            uint32_t hw0 = hp[0], hw1 = hp[1], hw2 = hp[2], hw3 = hp[3];
            const uint32_t lwd = *reinterpret_cast<const uint32_t*>(lowun_cm + base * 4);  // 1 bit/elem
            const uint32_t sreg = shared_cm[base * 1];
            const int sh0 = (int)((sreg & 7u) ^ 4u) - 4;          // signed 3-bit, group 0 (k<16)
            const int sh1 = (int)(((sreg >> 3) & 7u) ^ 4u) - 4;   // group 1 (k>=16)
            const float s16 = 16.0f * scale, s8 = 8.0f * scale;
            const float base0 = scale * (float)sh0 - 128.0f * scale;
            const float base1 = scale * (float)sh1 - 128.0f * scale;
            const float MAGIC = 8388608.0f;
            // process 8 elements per hi-word; even nibbles -> k0,k0+2,..; odd -> k0+1,..
            #pragma unroll
            for (int wi = 0; wi < 4; ++wi) {
                const uint32_t hw = (wi==0)?hw0:(wi==1)?hw1:(wi==2)?hw2:hw3;
                const uint32_t ev = hw & 0x0F0F0F0Fu;             // hi_u for even elems
                const uint32_t od = (hw >> 4) & 0x0F0F0F0Fu;      // hi_u for odd elems
                const int k0 = wi * 8;
                const float bb = (k0 < 16) ? base0 : base1;
                #pragma unroll
                for (int q = 0; q < 4; ++q) {
                    const int ke = k0 + 2*q, ko = k0 + 2*q + 1;
                    const uint32_t sele = 0x7440u | (uint32_t)q;  // byte q -> pos0, 0x4B -> pos3, 0 elsewhere
                    const uint32_t selo = 0x7440u | (uint32_t)q;
                    const float hfe = __uint_as_float(__byte_perm(ev, 0x4B000000u, sele)) - MAGIC;
                    const float hfo = __uint_as_float(__byte_perm(od, 0x4B000000u, selo)) - MAGIC;
                    float wfe = hfe * s16 + bb;
                    float wfo = hfo * s16 + bb;
                    if ((lwd >> ke) & 1u) wfe += s8;
                    if ((lwd >> ko) & 1u) wfo += s8;
                    #pragma unroll
                    for (int j = 0; j < MR; ++j) { acc[j] += wfe * As[j][ke]; acc[j] += wfo * As[j][ko]; }
                }
            }
        }
        __syncthreads();
    }
    if (active) {
        #pragma unroll
        for (int j = 0; j < MR; ++j) { const int m = row0 + j; if (m < M) partial[((long)sp * M + m) * OUT + o] = acc[j]; }
    }
}

// MS-UNSIGNED batched GEMV host: planes from pack_weight_unsigned (u3/gs16, u2/gs8).
torch::Tensor wonly_gemv_batched_unsigned_cuda(
        torch::Tensor x, torch::Tensor scale_exp, torch::Tensor upper_cm, torch::Tensor shared_cm,
        int64_t M, int64_t OUT, int64_t NB, int64_t u, int64_t gs) {
    TORCH_CHECK(x.is_cuda() && x.scalar_type() == torch::kBFloat16, "x must be CUDA bf16");
    auto y = torch::empty({M, OUT}, x.options());
    const int threads = 128, blocks = (int)((OUT + threads - 1) / threads);
    const int splitK = ms::gemv_splitk_count(blocks, (int)NB, 16);
    int MR = 1; while (MR < (int)M && MR < 32) MR <<= 1;
    const int nTiles = (int)((M + MR - 1) / MR);
    auto partial = torch::empty({(int64_t)splitK, M, OUT}, x.options().dtype(torch::kFloat32));
    dim3 grid(blocks, splitK, nTiles);
    auto L = [&](auto Ut, auto Gt, auto MRt) {
        wonly_gemv_batched_unsigned<decltype(Ut)::value, decltype(Gt)::value, decltype(MRt)::value>
            <<<grid, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
            reinterpret_cast<const __nv_bfloat16*>(x.data_ptr<at::BFloat16>()),
            scale_exp.data_ptr<int8_t>(), upper_cm.data_ptr<uint8_t>(), shared_cm.data_ptr<uint8_t>(),
            partial.data_ptr<float>(), (int)M, (int)OUT, (int)NB, splitK);
    };
#define UNMR(UU,GG) switch (MR) { \
        case 1:  L(std::integral_constant<int,UU>{},std::integral_constant<int,GG>{},std::integral_constant<int,1>{});  break; \
        case 2:  L(std::integral_constant<int,UU>{},std::integral_constant<int,GG>{},std::integral_constant<int,2>{});  break; \
        case 4:  L(std::integral_constant<int,UU>{},std::integral_constant<int,GG>{},std::integral_constant<int,4>{});  break; \
        case 8:  L(std::integral_constant<int,UU>{},std::integral_constant<int,GG>{},std::integral_constant<int,8>{});  break; \
        case 16: L(std::integral_constant<int,UU>{},std::integral_constant<int,GG>{},std::integral_constant<int,16>{}); break; \
        default: L(std::integral_constant<int,UU>{},std::integral_constant<int,GG>{},std::integral_constant<int,32>{}); break; }
    if      ((int)u==3 && (int)gs==16) { UNMR(3,16); }
    else if ((int)u==2 && (int)gs==8)  { UNMR(2,8); }
    else TORCH_CHECK(false, "unsigned proto: only u3/gs16 and u2/gs8");
#undef UNMR
    gemv_combine_batched_kernel<<<dim3(blocks, (int)M), threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        partial.data_ptr<float>(), reinterpret_cast<__nv_bfloat16*>(y.data_ptr<at::BFloat16>()),
        (int)M, (int)OUT, splitK);
    return y;
}

// NIBBLE-RELAYOUT batched GEMV host (prototype): planes from pack_weight_relayout.
torch::Tensor wonly_gemv_batched_relayout_cuda(
        torch::Tensor x, torch::Tensor scale_exp, torch::Tensor hi4_cm,
        torch::Tensor lowun_cm, torch::Tensor shared_cm,
        int64_t M, int64_t OUT, int64_t NB, int64_t u, int64_t gs) {
    TORCH_CHECK(x.is_cuda() && x.scalar_type() == torch::kBFloat16, "x must be CUDA bf16");
    auto y = torch::empty({M, OUT}, x.options());
    const int threads = 128, blocks = (int)((OUT + threads - 1) / threads);
    const int splitK = ms::gemv_splitk_count(blocks, (int)NB, 16);
    int MR = 1; while (MR < (int)M && MR < 32) MR <<= 1;
    const int nTiles = (int)((M + MR - 1) / MR);
    auto partial = torch::empty({(int64_t)splitK, M, OUT}, x.options().dtype(torch::kFloat32));
    dim3 grid(blocks, splitK, nTiles);
    const bool vec = getenv("MS_RELAYOUT_VEC") && atoi(getenv("MS_RELAYOUT_VEC")) != 0;
    auto L = [&](auto Ut, auto Gt, auto MRt) {
        if (vec && decltype(Ut)::value == 3 && decltype(Gt)::value == 16) {
            wonly_gemv_batched_relayout_vec_u3<decltype(MRt)::value>
                <<<grid, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
                reinterpret_cast<const __nv_bfloat16*>(x.data_ptr<at::BFloat16>()),
                scale_exp.data_ptr<int8_t>(), hi4_cm.data_ptr<uint8_t>(),
                lowun_cm.data_ptr<uint8_t>(), shared_cm.data_ptr<uint8_t>(),
                partial.data_ptr<float>(), (int)M, (int)OUT, (int)NB, splitK);
            return;
        }
        wonly_gemv_batched_relayout<decltype(Ut)::value, decltype(Gt)::value, decltype(MRt)::value>
            <<<grid, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
            reinterpret_cast<const __nv_bfloat16*>(x.data_ptr<at::BFloat16>()),
            scale_exp.data_ptr<int8_t>(), hi4_cm.data_ptr<uint8_t>(),
            lowun_cm.data_ptr<uint8_t>(), shared_cm.data_ptr<uint8_t>(),
            partial.data_ptr<float>(), (int)M, (int)OUT, (int)NB, splitK);
    };
#define RLMR(UU,GG) switch (MR) { \
        case 1:  L(std::integral_constant<int,UU>{},std::integral_constant<int,GG>{},std::integral_constant<int,1>{});  break; \
        case 2:  L(std::integral_constant<int,UU>{},std::integral_constant<int,GG>{},std::integral_constant<int,2>{});  break; \
        case 4:  L(std::integral_constant<int,UU>{},std::integral_constant<int,GG>{},std::integral_constant<int,4>{});  break; \
        case 8:  L(std::integral_constant<int,UU>{},std::integral_constant<int,GG>{},std::integral_constant<int,8>{});  break; \
        case 16: L(std::integral_constant<int,UU>{},std::integral_constant<int,GG>{},std::integral_constant<int,16>{}); break; \
        default: L(std::integral_constant<int,UU>{},std::integral_constant<int,GG>{},std::integral_constant<int,32>{}); break; }
    if      ((int)u==3 && (int)gs==16) { RLMR(3,16); }
    else if ((int)u==2 && (int)gs==8)  { RLMR(2,8); }
    else TORCH_CHECK(false, "relayout proto: only u3/gs16 and u2/gs8");
#undef RLMR
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
    auto launch_spec = [&](auto Ut, auto Gt) {
        wa_gemv_wide_uspec<decltype(Ut)::value, decltype(Gt)::value><<<dim3(blocks, splitK), threads, smem, at::cuda::getCurrentCUDAStream()>>>(
            qx.data_ptr<int8_t>(), sa_exp.data_ptr<int8_t>(), scale_exp.data_ptr<int8_t>(),
            upper_cm.data_ptr<uint8_t>(), shared_cm.data_ptr<uint8_t>(),
            partial.data_ptr<float>(), (int)OUT, (int)NB, splitK);
    };
    const bool nospec = getenv("MS_GEMV_NOSPEC") && atoi(getenv("MS_GEMV_NOSPEC")) != 0;
    bool launched = false;
    if ((int)u == 4) { launch(std::true_type{}); launched = true; }
    else if (!nospec) {
        #define SPEC(UU,GG) if (!launched && (int)u==UU && (int)gs==GG) { \
            launch_spec(std::integral_constant<int,UU>{}, std::integral_constant<int,GG>{}); launched = true; }
        SPEC(2,8) SPEC(2,16) SPEC(2,4) SPEC(2,32)
        SPEC(3,8) SPEC(3,16) SPEC(3,4) SPEC(3,32)
        #undef SPEC
    }
    if (!launched) launch(std::false_type{});

    gemv_combine_kernel<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        partial.data_ptr<float>(),
        reinterpret_cast<__nv_bfloat16*>(y.data_ptr<at::BFloat16>()), (int)OUT, splitK);
    return y;
}

// ===== MS-UNSIGNED (naive-ms) hosts: u3/gs16 + u2/gs8 only =====
torch::Tensor wonly_gemv_wide_unsigned_cuda(
        torch::Tensor x, torch::Tensor scale_exp, torch::Tensor upper_cm, torch::Tensor shared_cm,
        int64_t OUT, int64_t NB, int64_t u, int64_t gs) {
    auto y = torch::empty({OUT}, x.options());
    const int threads = 128, blocks = (int)((OUT + threads - 1) / threads);
    const int splitK = ms::gemv_splitk_count(blocks, (int)NB, 16);
    auto partial = torch::empty({(int64_t)splitK, OUT}, x.options().dtype(torch::kFloat32));
    int sepsc = ((int)u != 4) ? 1 : 0;
    if (const char* e = getenv("MS_GEMV_SEPSC")) sepsc = atoi(e) != 0 ? 1 : 0;
    auto L = [&](auto Ut, auto Gt) {
        wonly_gemv_wide_unsigned<decltype(Ut)::value, decltype(Gt)::value><<<dim3(blocks, splitK), threads, 0, at::cuda::getCurrentCUDAStream()>>>(
            reinterpret_cast<const __nv_bfloat16*>(x.data_ptr<at::BFloat16>()), scale_exp.data_ptr<int8_t>(),
            upper_cm.data_ptr<uint8_t>(), shared_cm.data_ptr<uint8_t>(), partial.data_ptr<float>(),
            (int)OUT, (int)NB, splitK, sepsc); };
    if ((int)u==3 && (int)gs==16) L(std::integral_constant<int,3>{}, std::integral_constant<int,16>{});
    else if ((int)u==2 && (int)gs==8) L(std::integral_constant<int,2>{}, std::integral_constant<int,8>{});
    else TORCH_CHECK(false, "unsigned: u3/gs16,u2/gs8");
    gemv_combine_kernel<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        partial.data_ptr<float>(), reinterpret_cast<__nv_bfloat16*>(y.data_ptr<at::BFloat16>()), (int)OUT, splitK);
    return y;
}
torch::Tensor wa_gemv_unsigned_cuda(
        torch::Tensor x, torch::Tensor scale_exp, torch::Tensor upper_cm, torch::Tensor shared_cm,
        int64_t OUT, int64_t NB, int64_t u, int64_t gs) {
    const int K = (int)NB * BLOCK;
    auto qx = torch::empty({K}, x.options().dtype(torch::kInt8));
    auto sa_exp = torch::empty({NB}, x.options().dtype(torch::kInt8));
    ms_launch_quant_act_msaq(reinterpret_cast<const __nv_bfloat16*>(x.data_ptr<at::BFloat16>()),
                             qx.data_ptr<int8_t>(), sa_exp.data_ptr<int8_t>(), 1, K, (int)NB, (int)u, (int)gs);
    auto y = torch::empty({OUT}, x.options());
    const int threads = 128, blocks = (int)((OUT + threads - 1) / threads);
    const int splitK = ms::gemv_splitk_count(blocks, (int)NB, 16);
    auto partial = torch::empty({(int64_t)splitK, OUT}, x.options().dtype(torch::kFloat32));
    const int per = ((int)NB + splitK - 1) / splitK; const size_t smem = (size_t)per * BLOCK;
    auto L = [&](auto Ut, auto Gt) {
        wa_gemv_wide_unsigned<decltype(Ut)::value, decltype(Gt)::value><<<dim3(blocks, splitK), threads, smem, at::cuda::getCurrentCUDAStream()>>>(
            qx.data_ptr<int8_t>(), sa_exp.data_ptr<int8_t>(), scale_exp.data_ptr<int8_t>(),
            upper_cm.data_ptr<uint8_t>(), shared_cm.data_ptr<uint8_t>(), partial.data_ptr<float>(),
            (int)OUT, (int)NB, splitK); };
    if ((int)u==3 && (int)gs==16) L(std::integral_constant<int,3>{}, std::integral_constant<int,16>{});
    else if ((int)u==2 && (int)gs==8) L(std::integral_constant<int,2>{}, std::integral_constant<int,8>{});
    else TORCH_CHECK(false, "unsigned: u3/gs16,u2/gs8");
    gemv_combine_kernel<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        partial.data_ptr<float>(), reinterpret_cast<__nv_bfloat16*>(y.data_ptr<at::BFloat16>()), (int)OUT, splitK);
    return y;
}

// batched-decode W+A GEMV: x [M,K] -> y [M,OUT]. Stage-0 MSAQ-s activation quant (M rows)
// then batched int-dot GEMV (weight word read once, amortized over B).
torch::Tensor wa_gemv_batched_cuda(
        torch::Tensor x, torch::Tensor scale_exp, torch::Tensor upper_cm, torch::Tensor shared_cm,
        int64_t M, int64_t OUT, int64_t NB, int64_t u, int64_t gs) {
    TORCH_CHECK(x.is_cuda() && x.scalar_type() == torch::kBFloat16, "x must be CUDA bf16");
    const int wbits = 8 - (int)u, UB = BLOCK * wbits / 8, SB = ((BLOCK / (int)gs) * (int)u + 7) / 8;
    const int K = (int)NB * BLOCK;
    auto qx = torch::empty({M, K}, x.options().dtype(torch::kInt8));
    auto sa_exp = torch::empty({M, NB}, x.options().dtype(torch::kInt8));
    auto y = torch::empty({M, OUT}, x.options());
    const int threads = 128, blocks = (int)((OUT + threads - 1) / threads);
    const int splitK = ms::gemv_splitk_count(blocks, (int)NB, 16);
    // W+A holds idot[MR] (int) + acc[MR] (float) -> 2x the accumulators of W-only, so MR=32
    // spills; cap MR (tile M) to stay in registers. Default 8 (env MS_WA_MR to tune).
    int mrcap = 8; if (const char* e = getenv("MS_WA_MR")) mrcap = atoi(e);
    int MR = 1; while (MR < (int)M && MR < mrcap) MR <<= 1;
    const int nTiles = (int)((M + MR - 1) / MR);
    auto partial = torch::empty({(int64_t)splitK, M, OUT}, x.options().dtype(torch::kFloat32));
    dim3 grid(blocks, splitK, nTiles);
    const bool nospec = getenv("MS_GEMV_NOSPEC") && atoi(getenv("MS_GEMV_NOSPEC")) != 0;
    bool launched = false;
    // MS_WA_FUSED=1: fake-quant the activation INSIDE the GEMV staging (no quant_act pre-pass,
    // no int8 qx round-trip). DOCUMENTED-NEGATIVE, default OFF: it's 4-9% SLOWER than the split
    // path. quant_act is a once-per-[M,K] op; fusing it into the per-output-column-block GEMV
    // redoes the fake-quant ~OUT/128 (~32x) times, and that redundant compute costs more than
    // the single 13us quant_act it removes. The split path (quant_act once + dequant-in-staging
    // float MAC, = wa_gemv_batched_uspec) stays the default at 44us@M8.
    const bool fused = getenv("MS_WA_FUSED") && atoi(getenv("MS_WA_FUSED")) != 0;
    if (fused && !nospec && (int)u != 4) {
        auto lf = [&](auto Ut, auto Gt, auto MRtag) {
            wa_gemv_batched_fused_uspec<decltype(Ut)::value, decltype(Gt)::value, decltype(MRtag)::value>
                <<<grid, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
                reinterpret_cast<const __nv_bfloat16*>(x.data_ptr<at::BFloat16>()),
                scale_exp.data_ptr<int8_t>(), upper_cm.data_ptr<uint8_t>(), shared_cm.data_ptr<uint8_t>(),
                partial.data_ptr<float>(), (int)M, (int)OUT, (int)NB, splitK);
        };
#define MRDISP_FUSED(UU,GG) switch (MR) { \
        case 1:  lf(std::integral_constant<int,UU>{}, std::integral_constant<int,GG>{}, std::integral_constant<int,1>{});  break; \
        case 2:  lf(std::integral_constant<int,UU>{}, std::integral_constant<int,GG>{}, std::integral_constant<int,2>{});  break; \
        case 4:  lf(std::integral_constant<int,UU>{}, std::integral_constant<int,GG>{}, std::integral_constant<int,4>{});  break; \
        case 8:  lf(std::integral_constant<int,UU>{}, std::integral_constant<int,GG>{}, std::integral_constant<int,8>{});  break; \
        case 16: lf(std::integral_constant<int,UU>{}, std::integral_constant<int,GG>{}, std::integral_constant<int,16>{}); break; \
        default: lf(std::integral_constant<int,UU>{}, std::integral_constant<int,GG>{}, std::integral_constant<int,32>{}); break; }
        if      ((int)u==2 && (int)gs==8)  { MRDISP_FUSED(2,8);  launched = true; }
        else if ((int)u==3 && (int)gs==8)  { MRDISP_FUSED(3,8);  launched = true; }
        else if ((int)u==2 && (int)gs==16) { MRDISP_FUSED(2,16); launched = true; }
        else if ((int)u==3 && (int)gs==16) { MRDISP_FUSED(3,16); launched = true; }
#undef MRDISP_FUSED
    }
    // OLD path (u=4, non-specialized, or MS_WA_FUSED=0): quant_act pre-pass + int dot.
    if (!launched) {
        ms_launch_quant_act_msaq(reinterpret_cast<const __nv_bfloat16*>(x.data_ptr<at::BFloat16>()),
                                 qx.data_ptr<int8_t>(), sa_exp.data_ptr<int8_t>(), (int)M, K, (int)NB, (int)u, (int)gs);
        auto launch = [&](auto U4tag, auto MRtag) {
            wa_gemv_batched_kernel<decltype(U4tag)::value, decltype(MRtag)::value>
                <<<grid, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
                qx.data_ptr<int8_t>(), sa_exp.data_ptr<int8_t>(), scale_exp.data_ptr<int8_t>(),
                upper_cm.data_ptr<uint8_t>(), shared_cm.data_ptr<uint8_t>(),
                partial.data_ptr<float>(), (int)M, (int)OUT, (int)NB, (int)u, (int)gs, UB, SB, splitK);
        };
        auto launch_spec = [&](auto Ut, auto Gt, auto MRtag) {
            wa_gemv_batched_uspec<decltype(Ut)::value, decltype(Gt)::value, decltype(MRtag)::value>
                <<<grid, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
                qx.data_ptr<int8_t>(), sa_exp.data_ptr<int8_t>(), scale_exp.data_ptr<int8_t>(),
                upper_cm.data_ptr<uint8_t>(), shared_cm.data_ptr<uint8_t>(),
                partial.data_ptr<float>(), (int)M, (int)OUT, (int)NB, splitK);
        };
#define MRDISP(U4) switch (MR) { \
        case 1:  launch(U4, std::integral_constant<int,1>{});  break; \
        case 2:  launch(U4, std::integral_constant<int,2>{});  break; \
        case 4:  launch(U4, std::integral_constant<int,4>{});  break; \
        case 8:  launch(U4, std::integral_constant<int,8>{});  break; \
        case 16: launch(U4, std::integral_constant<int,16>{}); break; \
        default: launch(U4, std::integral_constant<int,32>{}); break; }
#define MRDISP_SPEC(UU,GG) switch (MR) { \
        case 1:  launch_spec(std::integral_constant<int,UU>{}, std::integral_constant<int,GG>{}, std::integral_constant<int,1>{});  break; \
        case 2:  launch_spec(std::integral_constant<int,UU>{}, std::integral_constant<int,GG>{}, std::integral_constant<int,2>{});  break; \
        case 4:  launch_spec(std::integral_constant<int,UU>{}, std::integral_constant<int,GG>{}, std::integral_constant<int,4>{});  break; \
        case 8:  launch_spec(std::integral_constant<int,UU>{}, std::integral_constant<int,GG>{}, std::integral_constant<int,8>{});  break; \
        case 16: launch_spec(std::integral_constant<int,UU>{}, std::integral_constant<int,GG>{}, std::integral_constant<int,16>{}); break; \
        default: launch_spec(std::integral_constant<int,UU>{}, std::integral_constant<int,GG>{}, std::integral_constant<int,32>{}); break; }
        if ((int)u == 4) { MRDISP(std::true_type{}); }
        else if (!nospec) {
            if      ((int)u==2 && (int)gs==8)  { MRDISP_SPEC(2,8);  }
            else if ((int)u==3 && (int)gs==8)  { MRDISP_SPEC(3,8);  }
            else if ((int)u==2 && (int)gs==16) { MRDISP_SPEC(2,16); }
            else if ((int)u==3 && (int)gs==16) { MRDISP_SPEC(3,16); }
            else { MRDISP(std::false_type{}); }
        }
        else { MRDISP(std::false_type{}); }
#undef MRDISP
#undef MRDISP_SPEC
    }
    gemv_combine_batched_kernel<<<dim3(blocks, (int)M), threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        partial.data_ptr<float>(), reinterpret_cast<__nv_bfloat16*>(y.data_ptr<at::BFloat16>()),
        (int)M, (int)OUT, splitK);
    return y;
}

torch::Tensor wa_gemv_batched_unsigned_cuda(
        torch::Tensor x, torch::Tensor scale_exp, torch::Tensor upper_cm, torch::Tensor shared_cm,
        int64_t M, int64_t OUT, int64_t NB, int64_t u, int64_t gs) {
    const int K = (int)NB * BLOCK;
    auto qx = torch::empty({M, K}, x.options().dtype(torch::kInt8));
    auto sa_exp = torch::empty({M, NB}, x.options().dtype(torch::kInt8));
    ms_launch_quant_act_msaq(reinterpret_cast<const __nv_bfloat16*>(x.data_ptr<at::BFloat16>()),
                             qx.data_ptr<int8_t>(), sa_exp.data_ptr<int8_t>(), (int)M, K, (int)NB, (int)u, (int)gs);
    auto y = torch::empty({M, OUT}, x.options());
    const int threads = 128, blocks = (int)((OUT + threads - 1) / threads);
    const int splitK = ms::gemv_splitk_count(blocks, (int)NB, 16);
    int MR = 1; while (MR < (int)M && MR < 32) MR <<= 1;
    const int nTiles = (int)((M + MR - 1) / MR);
    auto partial = torch::empty({(int64_t)splitK, M, OUT}, x.options().dtype(torch::kFloat32));
    dim3 grid(blocks, splitK, nTiles);
    auto L = [&](auto Ut, auto Gt, auto MRt) {
        wa_gemv_batched_unsigned<decltype(Ut)::value, decltype(Gt)::value, decltype(MRt)::value>
            <<<grid, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
            qx.data_ptr<int8_t>(), sa_exp.data_ptr<int8_t>(), scale_exp.data_ptr<int8_t>(),
            upper_cm.data_ptr<uint8_t>(), shared_cm.data_ptr<uint8_t>(),
            partial.data_ptr<float>(), (int)M, (int)OUT, (int)NB, splitK); };
#define WAUMR(UU,GG) switch (MR) { \
        case 1:  L(std::integral_constant<int,UU>{},std::integral_constant<int,GG>{},std::integral_constant<int,1>{});  break; \
        case 2:  L(std::integral_constant<int,UU>{},std::integral_constant<int,GG>{},std::integral_constant<int,2>{});  break; \
        case 4:  L(std::integral_constant<int,UU>{},std::integral_constant<int,GG>{},std::integral_constant<int,4>{});  break; \
        case 8:  L(std::integral_constant<int,UU>{},std::integral_constant<int,GG>{},std::integral_constant<int,8>{});  break; \
        case 16: L(std::integral_constant<int,UU>{},std::integral_constant<int,GG>{},std::integral_constant<int,16>{}); break; \
        default: L(std::integral_constant<int,UU>{},std::integral_constant<int,GG>{},std::integral_constant<int,32>{}); break; }
    if      ((int)u==3 && (int)gs==16) { WAUMR(3,16); }
    else if ((int)u==2 && (int)gs==8)  { WAUMR(2,8); }
    else TORCH_CHECK(false, "unsigned: u3/gs16,u2/gs8");
#undef WAUMR
    gemv_combine_batched_kernel<<<dim3(blocks, (int)M), threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        partial.data_ptr<float>(), reinterpret_cast<__nv_bfloat16*>(y.data_ptr<at::BFloat16>()),
        (int)M, (int)OUT, splitK);
    return y;
}
