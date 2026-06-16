// csrc/core/ms_utils.cuh
//
// Shared device-side primitives for the MSAQ-signed kernels. The doc's
// centerpiece: "extract `upper` and `shared` from the bitstream and synthesize
// an INT8 word" is needed identically by W-only GEMV, W+A GEMM and KV
// attention, so it lives here as __device__ __forceinline__ and is #included by
// every .cu. Edit the unpack in ONE place.
//
// =============================================================================
//  LAYOUT THIS HEADER DECODES  (the CURRENT certified pack — ms_lib/pack.py)
// =============================================================================
//  Out-innermost SoA, dense LSB-first. Per 32-element block `blk`, column `o`:
//    upper plane  [nb, UB, OUT]  UB = 32*(8-u)/8       flat: (blk*UB+bi)*OUT+o
//    shared plane [nb, SB, OUT]  SB = ceil((32/gs)*u/8) flat: (blk*SB+sbi)*OUT+o
//    scale_exp    [nb, OUT] int8                        flat: blk*OUT+o
//  Within-block element k:  upper code at bit k*(8-u);  shared code (group
//  g=k/gs) at bit g*u. Codes are dense and CAN straddle a byte boundary, and
//  the two bytes of a straddling code are OUT elements apart in memory.
//  unpack_ms_weight_elem() handles the straddle with a 2-byte load + shift/or.
//  This is bit-exact vs ms_lib.pack.dequant_weight (certified via the pytest
//  roundtrip gate); the CUDA kernels inherit that correctness for free.
//
// =============================================================================
//  bfe.s32 / 128-bit loads  (the doc's optimization — NOT enabled here)
// =============================================================================
//  Single-op `bfe.s32` extraction and int4/float4 vectorized loads require the
//  register-aligned packing the doc proposes (pad/swizzle so N whole codes fall
//  inside one 32-bit word, byte-contiguous). The dense layout above is NOT
//  byte-contiguous per code, so bfe cannot replace the shift/or path until the
//  packing is changed. When that lands:
//    (1) re-lay-out ms_lib.pack to register-aligned codes,
//    (2) RE-CERTIFY the new layout through the pack<->unpack roundtrip test,
//    (3) flip MSAQ_USE_BFE and implement the bfe variant below.
//  bfe_s32() is provided now so the eventual switch is mechanical.

#pragma once
#include <cstdint>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

#ifndef MSAQ_USE_BFE
#define MSAQ_USE_BFE 0   // 0 = portable shift/mask (current layout); 1 = bfe.s32 (target layout)
#endif

namespace ms {

constexpr int E_MAX = 6;   // MXINT8 (elem_bitwidth=8): E_max = n - 2 = 6

// --- HOST: choose # of key-splits so grid blocks ~ mult * #SM (occupancy 방안1)
//   The decode attention launches H*S blocks; with H tiny (8) a single split
//   leaves the 82-SM RTX 3090 mostly idle. Derive S from the live SM count so
//   the grid lands at ~mult x #SM (Little's-law in-flight target).
//   MS_KV_SPLIT_MULT (env, default 3) sweeps the multiplier; MIN_TILE keeps a
//   tiny Lk from over-splitting into trivial tiles. Returns S (#tiles per head).
inline int kv_split_count(long Lk, int H) {
    static int sm = -1;
    if (sm < 0) cudaDeviceGetAttribute(&sm, cudaDevAttrMultiProcessorCount, 0);
    int mult = 3;
    if (const char* e = getenv("MS_KV_SPLIT_MULT")) { int m = atoi(e); if (m > 0) mult = m; }
    constexpr int MIN_TILE = 32;                       // floor on keys/tile
    int S    = (int)(((long)mult * sm + H - 1) / H);   // splits to hit mult*#SM blocks
    int maxS = (int)((Lk + MIN_TILE - 1) / MIN_TILE);  // cap so tile >= MIN_TILE
    if (S > maxS) S = maxS;
    if (S < 1)    S = 1;
    return S;
}

// --- HOST: GEMV split-K factor. The W-only GEMV launches base_blocks =
//   ceil(OUT/128) blocks (one thread per output column); for OUT=4096 that is 32
//   blocks << 82 SMs. Split the K-reduction into `splitK` partial sums so the
//   grid becomes base_blocks*splitK ~ mult*#SM and fills the machine; a cheap 2nd
//   kernel sums the partials (no atomics). Capped at NB (each split needs >=1
//   32-block). MS_GEMV_SPLITK_MULT (env, default 3) sweeps the multiplier.
inline int gemv_splitk_count(int base_blocks, int NB) {
    static int sm = -1;
    if (sm < 0) cudaDeviceGetAttribute(&sm, cudaDevAttrMultiProcessorCount, 0);
    int mult = 3;
    if (const char* e = getenv("MS_GEMV_SPLITK_MULT")) { int m = atoi(e); if (m > 0) mult = m; }
    int sp = (int)(((long)mult * sm + base_blocks - 1) / base_blocks);
    if (sp > NB) sp = NB;
    if (sp < 1)  sp = 1;
    return sp;
}

// --- E8M0 shared scale: stored exponent (int8) -> 2^exp ----------------------
__device__ __forceinline__ float e8m0_to_scale(int8_t exp) {
    return exp2f(static_cast<float>(exp));
}

// --- sign-extend the low `bits` of v (v already masked to those bits) --------
__device__ __forceinline__ int sign_extend(uint32_t v, int bits) {
    const int s = 1 << (bits - 1);
    return (static_cast<int>(v) ^ s) - s;
}

// --- PTX bit-field-extract (sign-extended); for the TARGET register-aligned
//     layout. `pos` = start bit, `len` = width. Single instruction (~4 cyc). ---
__device__ __forceinline__ int bfe_s32(int32_t word, int pos, int len) {
    int d;
    asm("bfe.s32 %0, %1, %2, %3;" : "=r"(d) : "r"(word), "r"(pos), "r"(len));
    return d;
}

// --- extract one dense-packed signed code that may straddle two bytes --------
//   plane[(blk*BYTES + byteIdx)*OUT + o] is the byte stream; the code of width
//   `width` starts at global bit `bit0` within this block's BYTES bytes.
__device__ __forceinline__ int extract_code(const uint8_t* plane, int base_byte,
                                            int OUT, int o, int bit0, int width,
                                            int n_bytes) {
    const int bi  = bit0 >> 3;
    const int off = bit0 & 7;
    uint32_t b0 = plane[(base_byte + bi) * OUT + o];
    uint32_t code = b0 >> off;
    if (off + width > 8 && (bi + 1) < n_bytes) {
        uint32_t b1 = plane[(base_byte + bi + 1) * OUT + o];
        code |= b1 << (8 - off);
    }
    code &= (1u << width) - 1u;
    return sign_extend(code, width);
}

// --- reconstruct the INT8 MXINT8 word for element k of block blk, column o ---
//   word = upper*2^u + shared_expanded  (a valid MXINT8 integer; the W+A path
//   feeds this straight to IMMA, the W-only path casts to bf16 and scales).
__device__ __forceinline__ int unpack_ms_weight_elem(
        const uint8_t* upper, const uint8_t* shared,
        int blk, int o, int k, int OUT,
        int u, int gs, int UB, int SB) {
    const int wbits = 8 - u;
#if MSAQ_USE_BFE
    // TODO(target layout): with register-aligned packing, replace the two
    // extract_code() calls with a 32-bit vectorized load + two bfe_s32() ops.
    // Not valid against the current dense layout — guarded off by default.
    static_assert(MSAQ_USE_BFE == 0, "bfe path needs the register-aligned pack; see header");
    return 0;
#else
    const int up_code = extract_code(upper, blk * UB, OUT, o, k * wbits, wbits, UB);
    const int g       = k / gs;
    const int sh_code = extract_code(shared, blk * SB, OUT, o, g * u, u, SB);
    return up_code * (1 << u) + sh_code;
#endif
}

// --- KV variant: TOKEN-MAJOR planes [H, nb, L, BYTES] (BYTES innermost; Stage
//     4a). base_u/base_h fold in the per-head + per-block offset (= same value as
//     the old [.,BYTES,L] base since UB*L == L*UB); a block's BYTES bytes for a
//     fixed `key` are CONTIGUOUS (stride 1), so the per-element address is
//     base + key*BYTES + byteIdx and the straddle byte is the adjacent one. This
//     makes a warp's 32 head_dim reads at a fixed key coalesce. Same straddle
//     handling, same codes. (Stage 4b's word-aligned bfe variant was tried and
//     reverted — slower; the kernel is load-latency/MLP-bound, not extraction-
//     instruction-bound. See change.md Phase 5.) ------------------------------
__device__ __forceinline__ int unpack_ms_kv_elem(
        const uint8_t* upper, const uint8_t* shared,
        long base_u, long base_h, int L, int key, int k,
        int u, int gs, int UB, int SB) {
    const int wbits = 8 - u;
    const int g = k / gs;
    const long ku = base_u + (long)key * UB;     // this key's contiguous upper bytes
    const long kh = base_h + (long)key * SB;     // this key's contiguous shared bytes
    // upper
    const int ub0 = (k * wbits) >> 3, uoff = (k * wbits) & 7;
    uint32_t code = (uint32_t)upper[ku + ub0] >> uoff;
    if (uoff + wbits > 8 && (ub0 + 1) < UB)
        code |= (uint32_t)upper[ku + ub0 + 1] << (8 - uoff);
    code &= (1u << wbits) - 1u;
    const int up_code = sign_extend(code, wbits);
    // shared
    const int sb0 = (g * u) >> 3, soff = (g * u) & 7;
    uint32_t sc = (uint32_t)shared[kh + sb0] >> soff;
    if (soff + u > 8 && (sb0 + 1) < SB)
        sc |= (uint32_t)shared[kh + sb0 + 1] << (8 - soff);
    sc &= (1u << u) - 1u;
    const int sh_code = sign_extend(sc, u);
    return up_code * (1 << u) + sh_code;
}

}  // namespace ms

// =============================================================================
//  Micro-profiling (Macro §A): clock64() spans, compiled out unless requested.
//  Build with  -DENABLE_PROFILING  and pass a global cycle buffer to the kernel.
//  WARNING: inserting clock64() perturbs the scheduler/register allocation —
//  use only to direction-find, never as the reported number (use cuda.Event +
//  ncu for that). See README "Profiling".
// =============================================================================
#ifdef ENABLE_PROFILING
  #define MS_PROF_DECL  unsigned long long _ms_t0 = 0ull, _ms_dt = 0ull;
  #define MS_PROF_START() do { _ms_t0 = clock64(); } while (0)
  #define MS_PROF_STOP(buf, idx) do { _ms_dt = clock64() - _ms_t0; (buf)[(idx)] = _ms_dt; } while (0)
#else
  #define MS_PROF_DECL
  #define MS_PROF_START() do {} while (0)
  #define MS_PROF_STOP(buf, idx) do {} while (0)
#endif
