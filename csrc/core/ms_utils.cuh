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
inline int gemv_splitk_count(int base_blocks, int NB, int default_mult = 3) {
    static int sm = -1;
    if (sm < 0) cudaDeviceGetAttribute(&sm, cudaDevAttrMultiProcessorCount, 0);
    int mult = default_mult;
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
    const int g       = k >> (__ffs(gs) - 1);   // gs is a power of 2: avoid runtime divide
    const int sh_code = extract_code(shared, blk * SB, OUT, o, g * u, u, SB);
    return up_code * (1 << u) + sh_code;
#endif
}

// --- (u,gs)-SPECIALIZED streaming block unpack (u<4), shared by the wide-load GEMV
//   and the KV-decode K read. One 32-element block whose UBc bytes are CONTIGUOUS at
//   `upper_cm + base*UBc` (column-major weight: base=blk*OUT+o; token-major KV:
//   base=(hk*NB+blk)*Lcap+key); per_elem(k, word) called for each reconstructed word.
//   With U_/GS_ compile-time, WBITS/UBc/SBc and the masks/shifts are constants and the
//   SEPARATED variant: yields (kd, up_code, sh_code) separately (not the combined word) so the
//   caller can do separated-scale (sepsc) accumulation: dot = scale*(2^u*Σq·up + Σ_g qg·sh).
//   Same rolling-buffer unpack as stream_block_uspec; sh_code is the current group's code.
template<int U_, int GS_, typename F>
__device__ __forceinline__ void stream_block_uspec_sep(
        const uint8_t* __restrict__ upper_cm, const uint8_t* __restrict__ shared_cm,
        long base, F per_elem) {
    constexpr int MXB = 32;
    constexpr int Up = U_ > 0 ? U_ : 1, Gp = GS_ > 0 ? GS_ : 1;   // guards: this template is also
    constexpr int UBc = MXB * (8 - Up) / 8, SBc = ((MXB / Gp) * Up + 7) / 8;  // INSTANTIATED (dead) for
    constexpr int NW = UBc / 4, NSW = (SBc + 3) / 4 > 0 ? (SBc + 3) / 4 : 1;  // U_=-1 (u4) -> avoid <<(-)
    constexpr uint32_t umask = (1u << (8 - Up)) - 1u, usign = 1u << (8 - Up - 1);
    constexpr uint32_t smask = (1u << Up) - 1u, ssign = 1u << (Up - 1);
    constexpr int gsmask = Gp - 1, WBITS = 8 - Up;
    uint32_t sreg[NSW] = {0u};
    #pragma unroll
    for (int i = 0; i < SBc; ++i) sreg[i >> 2] |= (uint32_t)shared_cm[base * SBc + i] << (8 * (i & 3));
    const uint32_t* src = reinterpret_cast<const uint32_t*>(upper_cm + base * UBc);
    uint32_t ureg[NW];
    #pragma unroll
    for (int i = 0; i < NW; ++i) ureg[i] = src[i];
    uint64_t ubuf = 0; int unb = 0, uwi = 0;
    uint64_t sbuf = 0; int snb = 0, swi = 0; int sh_code = 0;
    #pragma unroll
    for (int k = 0; k < MXB; ++k) {
        if (unb < WBITS) { ubuf |= (uint64_t)ureg[uwi++] << unb; unb += 32; }
        const int up_code = (int)(((uint32_t)ubuf & umask) ^ usign) - (int)usign;
        ubuf >>= WBITS; unb -= WBITS;
        if ((k & gsmask) == 0) {
            if (snb < U_) { sbuf |= (uint64_t)sreg[swi++] << snb; snb += 32; }
            sh_code = (int)(((uint32_t)sbuf & smask) ^ ssign) - (int)ssign;
            sbuf >>= U_; snb -= U_;
        }
        per_elem(k, up_code, sh_code);
    }
}

//   k-loop unroll makes the rolling-buffer schedule static -> ureg register-resident.
//   Bit-exact to the runtime streaming unpack. (kernel_ver3.md / compile_time_optimization.md.)
template<int U_, int GS_, typename F>
__device__ __forceinline__ void stream_block_uspec(
        const uint8_t* __restrict__ upper_cm, const uint8_t* __restrict__ shared_cm,
        long base, F per_elem) {
    constexpr int MXB = 32;
    constexpr int WBITS = 8 - U_;
    constexpr int UBc = MXB * WBITS / 8;
    constexpr int SBc = ((MXB / GS_) * U_ + 7) / 8;
    constexpr int NW  = UBc / 4;
    constexpr int NSW = (SBc + 3) / 4 > 0 ? (SBc + 3) / 4 : 1;
    constexpr uint32_t umask = (1u << WBITS) - 1u, usign = 1u << (WBITS - 1);
    constexpr uint32_t smask = (1u << U_) - 1u, ssign = 1u << (U_ - 1);
    constexpr int gsmask = GS_ - 1;
    uint32_t sreg[NSW] = {0u};
    #pragma unroll
    for (int i = 0; i < SBc; ++i) sreg[i >> 2] |= (uint32_t)shared_cm[base * SBc + i] << (8 * (i & 3));
    const uint32_t* src = reinterpret_cast<const uint32_t*>(upper_cm + base * UBc);
    uint32_t ureg[NW];
    #pragma unroll
    for (int i = 0; i < NW; ++i) ureg[i] = src[i];
    uint64_t ubuf = 0; int unb = 0, uwi = 0;
    uint64_t sbuf = 0; int snb = 0, swi = 0; int sh_code = 0;
    #pragma unroll
    for (int k = 0; k < MXB; ++k) {
        if (unb < WBITS) { ubuf |= (uint64_t)ureg[uwi++] << unb; unb += 32; }
        const int up_code = (int)(((uint32_t)ubuf & umask) ^ usign) - (int)usign;
        ubuf >>= WBITS; unb -= WBITS;
        if ((k & gsmask) == 0) {
            if (snb < U_) { sbuf |= (uint64_t)sreg[swi++] << snb; snb += 32; }
            sh_code = (int)(((uint32_t)sbuf & smask) ^ ssign) - (int)ssign;
            sbuf >>= U_; snb -= U_;
        }
        per_elem(k, up_code * (1 << U_) + sh_code);
    }
}

// --- MS-UNSIGNED streaming unpack (ms_lib.pack.pack_weight_unsigned). Same dense
//   plane layout as stream_block_uspec, but the shared code is UNSIGNED (floor-upper
//   design) so it needs NO sign-extend and the int8 word is a pure bit-concat
//   (up_code<<U_)|sh: the shared bits slot into the upper word's zeroed low U_ bits.
//   Saves the shared xor/sub sign-extend; reconstruction is OR not signed-add.
template<int U_, int GS_, typename F>
__device__ __forceinline__ void stream_block_uspec_unsigned(
        const uint8_t* __restrict__ upper_cm, const uint8_t* __restrict__ shared_cm,
        long base, F per_elem) {
    constexpr int MXB = 32;
    constexpr int WBITS = 8 - U_;
    constexpr int UBc = MXB * WBITS / 8;
    constexpr int SBc = ((MXB / GS_) * U_ + 7) / 8;
    constexpr int NW  = UBc / 4;
    constexpr int NSW = (SBc + 3) / 4 > 0 ? (SBc + 3) / 4 : 1;
    constexpr uint32_t umask = (1u << WBITS) - 1u, usign = 1u << (WBITS - 1);
    constexpr uint32_t smask = (1u << U_) - 1u;
    constexpr int gsmask = GS_ - 1;
    uint32_t sreg[NSW] = {0u};
    #pragma unroll
    for (int i = 0; i < SBc; ++i) sreg[i >> 2] |= (uint32_t)shared_cm[base * SBc + i] << (8 * (i & 3));
    const uint32_t* src = reinterpret_cast<const uint32_t*>(upper_cm + base * UBc);
    uint32_t ureg[NW];
    #pragma unroll
    for (int i = 0; i < NW; ++i) ureg[i] = src[i];
    uint64_t ubuf = 0; int unb = 0, uwi = 0;
    uint64_t sbuf = 0; int snb = 0, swi = 0; int sh_code = 0;
    #pragma unroll
    for (int k = 0; k < MXB; ++k) {
        if (unb < WBITS) { ubuf |= (uint64_t)ureg[uwi++] << unb; unb += 32; }
        const int up_code = (int)(((uint32_t)ubuf & umask) ^ usign) - (int)usign;
        ubuf >>= WBITS; unb -= WBITS;
        if ((k & gsmask) == 0) {
            if (snb < U_) { sbuf |= (uint64_t)sreg[swi++] << snb; snb += 32; }
            sh_code = (int)((uint32_t)sbuf & smask);     // UNSIGNED: no sign-extend
            sbuf >>= U_; snb -= U_;
        }
        per_elem(k, (up_code << U_) | sh_code);           // OR-concat (low U_ bits of up_code<<U_ are 0)
    }
}

// --- MXFP8-MSAQ (E3M4) streaming unpack. SAME dense plane layout as stream_block_uspec
//   (upper field width 8-U_, shared U_-bit signed), but each upper field is an FP8 E3M4
//   element [sign:1|exp:3|upmant:(MB-U_)] -> yields the reconstructed FP8 VALUE (float, before
//   the block scale): val = sign * 2^(ee-MB) * (m_up*2^U_ + sh). Bit-exact to msaq_mxfp8(e3m4)
//   / msfp8_e3m4_dequant_bf16_kernel. The ONLY difference vs the INT path is per-element ALU
//   (exp/mantissa split + ldexpf); memory traffic is identical -> isolates fused decode ALU.
template<int U_, int GS_, typename F>
__device__ __forceinline__ void stream_block_uspec_fp8_e3m4(
        const uint8_t* __restrict__ upper_cm, const uint8_t* __restrict__ shared_cm,
        long base, F per_elem) {
    constexpr int MXB = 32, EB = 3, MB = 4, BIAS = 3, EMIN = -2;
    constexpr int WBITS = 8 - U_, MBUP = MB - U_;
    constexpr int UBc = MXB * WBITS / 8;
    constexpr int SBc = ((MXB / GS_) * U_ + 7) / 8;
    constexpr int NW  = UBc / 4;
    constexpr int NSW = (SBc + 3) / 4 > 0 ? (SBc + 3) / 4 : 1;
    constexpr uint32_t umask = (1u << WBITS) - 1u;
    constexpr uint32_t expmask = (1u << EB) - 1u, upmask = (1u << MBUP) - 1u, lead = 1u << MBUP;
    constexpr uint32_t smask = (1u << U_) - 1u, ssign = 1u << (U_ - 1);
    constexpr int gsmask = GS_ - 1;
    uint32_t sreg[NSW] = {0u};
    #pragma unroll
    for (int i = 0; i < SBc; ++i) sreg[i >> 2] |= (uint32_t)shared_cm[base * SBc + i] << (8 * (i & 3));
    const uint32_t* src = reinterpret_cast<const uint32_t*>(upper_cm + base * UBc);
    uint32_t ureg[NW];
    #pragma unroll
    for (int i = 0; i < NW; ++i) ureg[i] = src[i];
    uint64_t ubuf = 0; int unb = 0, uwi = 0;
    uint64_t sbuf = 0; int snb = 0, swi = 0; int sh_code = 0;
    #pragma unroll
    for (int k = 0; k < MXB; ++k) {
        if (unb < WBITS) { ubuf |= (uint64_t)ureg[uwi++] << unb; unb += 32; }
        const uint32_t field = (uint32_t)ubuf & umask;
        ubuf >>= WBITS; unb -= WBITS;
        const uint32_t sgn = field >> (WBITS - 1);
        const uint32_t expf = (field >> MBUP) & expmask;
        const int m_up = (int)((expf ? lead : 0u) | (field & upmask));
        const int ee = expf ? (int)expf - BIAS : EMIN;
        if ((k & gsmask) == 0) {
            if (snb < U_) { sbuf |= (uint64_t)sreg[swi++] << snb; snb += 32; }
            sh_code = (int)(((uint32_t)sbuf & smask) ^ ssign) - (int)ssign;
            sbuf >>= U_; snb -= U_;
        }
        int mag = m_up << U_;
        if (sgn) mag = -mag;
        per_elem(k, ldexpf((float)(mag + sh_code), ee - MB));   // FP8 value (pre block-scale)
    }
}

// --- DENSE + SELECTIVE-bfe unpack: reads the EXISTING dense upper_cm plane (pack_weight,
//   0.66x bytes, NO repack) but extracts each code with a SINGLE bfe.s32 when it does not
//   cross a 32-bit register boundary (compile-time known), and funnelshift+bfe only for the
//   few straddlers. Keeps dense bytes AND ~1 op/code. Bit-exact to stream_block_uspec.
template<int U_, int GS_, typename F>
__device__ __forceinline__ void stream_block_dense_bfe(
        const uint8_t* __restrict__ upper_cm, const uint8_t* __restrict__ shared_cm,
        long base, F per_elem) {
    constexpr int MXB = 32;
    constexpr int WBITS = 8 - U_;
    constexpr int UBc = MXB * WBITS / 8;
    constexpr int NW  = UBc / 4;
    constexpr int SBc = ((MXB / GS_) * U_ + 7) / 8;
    constexpr int gs_shift = (GS_>=2?1:0)+(GS_>=4?1:0)+(GS_>=8?1:0)+(GS_>=16?1:0)+(GS_>=32?1:0);
    const uint32_t* src = reinterpret_cast<const uint32_t*>(upper_cm + base * UBc);
    uint32_t ureg[NW];
    #pragma unroll
    for (int i = 0; i < NW; ++i) ureg[i] = src[i];
    uint32_t sreg = 0u;
    #pragma unroll
    for (int i = 0; i < SBc; ++i) sreg |= (uint32_t)shared_cm[base * SBc + i] << (8 * i);
    #pragma unroll
    for (int k = 0; k < MXB; ++k) {
        const int gb = k * WBITS, wi = gb >> 5, bo = gb & 31;
        int up;
        if (bo + WBITS <= 32) {                                  // no straddle -> 1 bfe
            up = bfe_s32((int)ureg[wi], bo, WBITS);
        } else {                                                 // straddle -> funnel window + bfe
            const uint32_t hi = (wi + 1 < NW) ? ureg[wi + 1] : 0u;
            up = bfe_s32((int)__funnelshift_r(ureg[wi], hi, bo), 0, WBITS);
        }
        const int g  = k >> gs_shift;
        const int sh = bfe_s32((int)sreg, g * U_, U_);
        per_elem(k, up * (1 << U_) + sh);
    }
}

// --- REGISTER-ALIGNED unpack (ms_lib.pack.pack_weight_ra). Upper codes padded so each
//   (8-U_)-bit code lies wholly inside one 32-bit word (CPW codes/word, NW words) ->
//   ONE bfe.s32 per code (HW sign-extend), no straddle/rolling-buffer/mask/sign-extend.
//   shared dense in 1 word (n_group*U_<=8), also bfe. Reconstructs the SAME signed int8
//   word as stream_block_uspec. base = blk*OUT+o (column-major); a column's NW words are
//   contiguous at upper_ra_cm + base*NW*4.
template<int U_, int GS_, typename F>
__device__ __forceinline__ void stream_block_ra(
        const uint8_t* __restrict__ upper_ra_cm, const uint8_t* __restrict__ shared_cm,
        long base, F per_elem) {
    constexpr int MXB = 32;
    constexpr int WBITS = 8 - U_;
    constexpr int CPW = 32 / WBITS;
    constexpr int NW  = (MXB + CPW - 1) / CPW;
    constexpr int SBc = ((MXB / GS_) * U_ + 7) / 8;
    constexpr int gs_shift = (GS_>=2?1:0)+(GS_>=4?1:0)+(GS_>=8?1:0)+(GS_>=16?1:0)+(GS_>=32?1:0);
    const uint32_t* src = reinterpret_cast<const uint32_t*>(upper_ra_cm + base * (NW * 4));
    uint32_t ureg[NW];
    #pragma unroll
    for (int i = 0; i < NW; ++i) ureg[i] = src[i];
    uint32_t sreg = 0u;
    #pragma unroll
    for (int i = 0; i < SBc; ++i) sreg |= (uint32_t)shared_cm[base * SBc + i] << (8 * i);
    #pragma unroll
    for (int k = 0; k < MXB; ++k) {
        const int up = bfe_s32((int)ureg[k / CPW], (k % CPW) * WBITS, WBITS);   // 1 op, signed
        const int g  = k >> gs_shift;
        const int sh = bfe_s32((int)sreg, g * U_, U_);                          // 1 op, signed
        per_elem(k, up * (1 << U_) + sh);
    }
}

// --- NIBBLE-ALIGNED re-layout streaming unpack (ms_lib.pack.pack_weight_relayout).
//   The (8-u)-bit per-element upper code is split into a high nibble (signed 4-bit,
//   16B plane, bfe-extractable with HW sign-extend, no straddle) + a small dense
//   low_un ((4-u)-bit) plane. Reconstructs the SAME int8 word as stream_block_uspec:
//     w = hi4*16 + low_un*2^u + shared.   hi4_cm: 16B/block; lowun_cm: ceil((4-u)*32/8)
//   B/block (u3:4, u2:8); shared_cm: SBc B/block. Bit-exact to pack_weight_relayout.
template<int U_, int GS_, typename F>
__device__ __forceinline__ void stream_block_relayout(
        const uint8_t* __restrict__ hi4_cm, const uint8_t* __restrict__ lowun_cm,
        const uint8_t* __restrict__ shared_cm, long base, F per_elem) {
    constexpr int MXB = 32;
    constexpr int LU = 4 - U_;                          // low_un bits/elem (u3:1 u2:2)
    constexpr int LUB = (LU * MXB + 7) / 8;             // u3:4 u2:8 bytes
    constexpr int NLU = (LUB + 3) / 4;
    constexpr int SBc = ((MXB / GS_) * U_ + 7) / 8;
    constexpr int gs_shift = (GS_>=2?1:0)+(GS_>=4?1:0)+(GS_>=8?1:0)+(GS_>=16?1:0)+(GS_>=32?1:0);
    constexpr uint32_t lumask = (1u << LU) - 1u;
    constexpr uint32_t smask = (1u << U_) - 1u, ssign = 1u << (U_ - 1);
    const uint32_t* hp = reinterpret_cast<const uint32_t*>(hi4_cm + base * 16);
    uint32_t hw[4];
    #pragma unroll
    for (int i = 0; i < 4; ++i) hw[i] = hp[i];          // 16B = 32 signed nibbles
    uint32_t lw[NLU] = {0u};
    #pragma unroll
    for (int i = 0; i < LUB; ++i) lw[i >> 2] |= (uint32_t)lowun_cm[base * LUB + i] << (8 * (i & 3));
    uint32_t sreg = 0u;
    #pragma unroll
    for (int i = 0; i < SBc; ++i) sreg |= (uint32_t)shared_cm[base * SBc + i] << (8 * (i & 3));
    #pragma unroll
    for (int k = 0; k < MXB; ++k) {
        const int hi = (int)((hw[k >> 3] >> ((k & 7) * 4)) & 0xF) - 8;    // biased nibble -> signed
        const int bit = k * LU;                                          // (4-u)-bit field, no straddle
        const int lu = (int)((lw[bit >> 5] >> (bit & 31)) & lumask);     // unsigned
        const int g = k >> gs_shift;
        const int sh = (int)(((sreg >> (g * U_)) & smask) ^ ssign) - (int)ssign;
        per_elem(k, hi * 16 + (lu << U_) + sh);
    }
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
    const int g = k >> (__ffs(gs) - 1);          // gs is a power of 2: avoid runtime divide
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

// --- KV variant, u==4 FAST PATH: upper(4b) and shared(4b) are both nibble-
//   aligned, so NO straddle ever happens. Replaces the general path's
//   conditional 2-byte load + variable mask + xor-sign-extend with a single
//   byte load + one bfe.s32 per plane (sign-extended in HW). u4 is the
//   fewest-bytes config (16B upper/block) and the KV benchmark default; the
//   general unpack_ms_kv_elem stays for u in {2,3}. Bit-exact: up*16+sh with
//   both codes 4-bit two's-complement == the general reconstruction at u=4. ----
__device__ __forceinline__ int unpack_ms_kv_elem_u4(
        const uint8_t* upper, const uint8_t* shared,
        long base_u, long base_h, int key, int k,
        int gs, int UB, int SB) {
    const long ku = base_u + (long)key * UB;
    const long kh = base_h + (long)key * SB;
    const int up_code = bfe_s32((int)upper[ku + (k >> 1)], (k & 1) * 4, 4);
    const int g       = k >> (__ffs(gs) - 1);
    const int sh_code = bfe_s32((int)shared[kh + (g >> 1)], (g & 1) * 4, 4);
    return up_code * 16 + sh_code;
}

// --- KV variant, MXFP8-MSAQ (E3M4): single-element FP8 VALUE (pre-block-scale).
//   FP8 analog of unpack_ms_kv_elem (token-major planes [.,L,BYTES]). Extracts the
//   (8-u)-bit FP8 E3M4 field [sign|exp:3|upmant:(4-u)] for element k and the u-bit
//   SIGNED group-shared code, returns the reconstructed FP8 value
//     sign * 2^(ee-MB) * (m_up*2^u + sh_code)
//   (the caller multiplies the E8M0 block scale). Bit-exact to
//   stream_block_uspec_fp8_e3m4 / msaq_mxfp8(e3m4); same straddle handling as the
//   INT unpack_ms_kv_elem (the ONLY change is the per-element FP decode). ---------
__device__ __forceinline__ float unpack_ms_kv_elem_fp8(
        const uint8_t* upper, const uint8_t* shared,
        long base_u, long base_h, int key, int k,
        int u, int gs, int UB, int SB) {
    constexpr int EB = 3, MB = 4, BIAS = 3, EMIN = -2;   // E3M4
    const int wbits = 8 - u, mbup = MB - u;
    const int g = k >> (__ffs(gs) - 1);
    const long ku = base_u + (long)key * UB;
    const long kh = base_h + (long)key * SB;
    // upper FP8 field (may straddle two bytes)
    const int ub0 = (k * wbits) >> 3, uoff = (k * wbits) & 7;
    uint32_t code = (uint32_t)upper[ku + ub0] >> uoff;
    if (uoff + wbits > 8 && (ub0 + 1) < UB)
        code |= (uint32_t)upper[ku + ub0 + 1] << (8 - uoff);
    code &= (1u << wbits) - 1u;
    const uint32_t sgn  = code >> (wbits - 1);
    const uint32_t expf = (code >> mbup) & ((1u << EB) - 1u);
    const uint32_t upm  = code & ((1u << mbup) - 1u);
    const int m_up = (int)((expf ? (1u << mbup) : 0u) | upm);   // implicit leading 1 for normals
    const int ee   = expf ? (int)expf - BIAS : EMIN;
    // shared SIGNED u-bit code for group g (may straddle)
    const int sb0 = (g * u) >> 3, soff = (g * u) & 7;
    uint32_t sc = (uint32_t)shared[kh + sb0] >> soff;
    if (soff + u > 8 && (sb0 + 1) < SB)
        sc |= (uint32_t)shared[kh + sb0 + 1] << (8 - soff);
    sc &= (1u << u) - 1u;
    const int sh_code = sign_extend(sc, u);
    int mag = m_up << u;
    if (sgn) mag = -mag;
    return ldexpf((float)(mag + sh_code), ee - MB);
}

// =============================================================================
//  PACK / QUANTIZE primitives (device counterpart of ms_lib.pack.decompose) —
//  shared by the KV write/append kernels (bit-pack tail) and the W+A activation
//  pre-pass (int8-word combine tail). New numerics: none — bit-exact to
//  pack.decompose / reference.quant_act(share=True). (change.md Phase 28.)
// =============================================================================

// E8M0 exponent for a block's max-abs:  clamp(floor(log2 amax) - E_MAX, +-127).
__device__ __forceinline__ int e8m0_exp_from_amax(float amax) {
    int e = (int)floorf(log2f(fmaxf(amax, 1e-30f))) - E_MAX;
    return max(-127, min(127, e));
}

// Decompose one 32-element block (thread holds x[32]) -> (8-u)-bit q_upper[32] +
// u-bit r_shared[32/gs] (one per gs-group) + E8M0 exponent (return). Identical to
// pack.decompose: q_upper = clip(round(x/(sa*2^u))); r_shared from the group-mean
// residual. gs is a power of two. q_upper/r_shared buffers sized 32 / 32/gs.
__device__ __forceinline__ int decompose_ms_block(
        const float* __restrict__ x, int u, int gs, int* q_upper, int* r_shared) {
    float amax = 1e-30f;
    #pragma unroll
    for (int k = 0; k < 32; ++k) amax = fmaxf(amax, fabsf(x[k]));
    const int ea = e8m0_exp_from_amax(amax);
    const float sa = exp2f((float)ea), s_unshared = sa * exp2f((float)u);
    const int qmax = (1 << (7 - u)) - 1, smin = -(1 << (u - 1)), smax = (1 << (u - 1)) - 1;
    const int ng = 32 / gs;
    for (int g = 0; g < ng; ++g) {
        float ravg = 0.0f;
        for (int kk = 0; kk < gs; ++kk) {
            const int k = g * gs + kk;
            q_upper[k] = max(-qmax, min(qmax, (int)rintf(x[k] / s_unshared)));
            ravg += x[k] - (float)q_upper[k] * s_unshared;
        }
        r_shared[g] = max(smin, min(smax, (int)rintf(ravg / (float)gs / sa)));
    }
    return ea;
}

// MXFP8-MSAQ (E3M4) decompose: device counterpart of ms_lib.pack.decompose_msfp8
// (efb_iters=2, wshare). One thread holds x[32]; produces (8-u)-bit FP8 E3M4 fields
// field[32] = [sign|exp:3|upmant:(4-u)] and SIGNED u-bit group-shared r_shared[32/gs],
// returns the E8M0 exponent (ONE exponent of headroom so the bit-storable encoder never
// saturates). Numerically faithful (fp32) to decompose_msfp8 — the decode is then bit-exact.
__device__ __forceinline__ int decompose_msfp8_block(
        const float* __restrict__ x, int u, int gs, int* field, int* r_shared) {
    constexpr int EB = 3, MB = 4, BIAS = 3, EMIN = -2, MAXEXP = 4;   // E3M4
    const int mb_up = MB - u;
    const float lead = (float)(1 << mb_up), mmax = (float)((1 << (mb_up + 1)) - 1);
    const int ng = 32 / gs, half = 1 << (u - 1), efb_iters = 2;
    float amax = 1e-30f;
    #pragma unroll
    for (int k = 0; k < 32; ++k) amax = fmaxf(amax, fabsf(x[k]));
    int s_exp = (int)floorf(log2f(amax)) - (MAXEXP - 1);            // headroom = 1
    s_exp = max(-127, min(127, s_exp));
    const float s_base = exp2f((float)s_exp);
    float y[32], up[32], ee[32];
    #pragma unroll
    for (int k = 0; k < 32; ++k) y[k] = x[k] / s_base;
    // round_storable(t): nearest FP8 with (mb_up) stored mantissa bits; rounding PROMOTES
    // the exponent at the top instead of saturating. Writes up (value) and ee (stored exp).
    auto rstore = [&](float t, float& outv, float& oute) {
        float et = floorf(log2f(fmaxf(fabsf(t), 1e-30f)));
        et = fmaxf((float)EMIN, fminf((float)MAXEXP, et));
        float m = rintf(fabsf(t) / exp2f(et - (float)mb_up));
        if (m >= 2.0f * lead && et < (float)MAXEXP) { et += 1.0f; m = lead; }
        m = fminf(m, mmax);
        outv = copysignf(m * exp2f(et - (float)mb_up), t);
        oute = et;
    };
    #pragma unroll
    for (int k = 0; k < 32; ++k) rstore(y[k], up[k], ee[k]);
    int sh_g[32];
    for (int it = 0; it <= efb_iters; ++it) {
        // (1) optimal shared | fixed upper: wshare = step_up^2 (=4^ee) weighted group mean.
        for (int g = 0; g < ng; ++g) {
            float num = 0.0f, den = 0.0f;
            for (int t = 0; t < gs; ++t) {
                const int k = g * gs + t;
                const float step_up = exp2f(ee[k] - (float)mb_up);
                num += ((y[k] - up[k]) / step_up) * (step_up * step_up);
                den += step_up * step_up;
            }
            int sv = (int)rintf((num / fmaxf(den, 1e-30f)) * (float)(1 << u));
            sh_g[g] = max(-half, min(half - 1, sv));
        }
        if (it == efb_iters) break;
        // (2) optimal upper | fixed shared: re-round each element absorbing the sharing error.
        for (int g = 0; g < ng; ++g) {
            const float shared_elem = (float)sh_g[g] / (float)(1 << u);
            for (int t = 0; t < gs; ++t) {
                const int k = g * gs + t;
                const float step_up = exp2f(ee[k] - (float)mb_up);
                rstore(y[k] - shared_elem * step_up, up[k], ee[k]);
            }
        }
    }
    // encode each upper element as the FP8 E3M4 field [sign|exp|upmant]
    const int wbits = 8 - u;
    #pragma unroll
    for (int k = 0; k < 32; ++k) {
        const float step_up = exp2f(ee[k] - (float)mb_up);
        const int m_up_abs = (int)rintf(fabsf(up[k]) / step_up);
        const int sgn = (up[k] < 0.0f) ? 1 : 0;
        const int normal = (m_up_abs >= (int)lead) ? 1 : 0;
        const int expb = normal ? ((int)ee[k] + BIAS) : 0;
        const int upm  = normal ? (m_up_abs - (int)lead) : m_up_abs;
        field[k] = (sgn << (wbits - 1)) | (expb << mb_up) | upm;
    }
    for (int g = 0; g < ng; ++g) r_shared[g] = sh_g[g];
    return s_exp;
}

// light-MS decompose: same stored format as decompose_ms_block, but INTEGER-friendly.
// One FP round per element (q8 = round(x/sa)); everything after is integer — the
// per-element FP residual (x - q_upper*s_unshared) and the FP group-mean are replaced
// by integer subtract + integer round-to-nearest mean. q_upper = round(q8 / 2^u) and
// shared = round(mean(q8 - q_upper*2^u)) in pure int. (Validated ~= decompose_ms_block
// in QSNR/PPL; a double-rounding variant of MSAQ. change.md Phase 41.)
__device__ __forceinline__ int decompose_lightms_block(
        const float* __restrict__ x, int u, int gs, int* q_upper, int* r_shared) {
    float amax = 1e-30f;
    #pragma unroll
    for (int k = 0; k < 32; ++k) amax = fmaxf(amax, fabsf(x[k]));
    const int ea = e8m0_exp_from_amax(amax);
    const float inv_sa = exp2f(-(float)ea);
    const int qmax = (1 << (7 - u)) - 1, smin = -(1 << (u - 1)), smax = (1 << (u - 1)) - 1;
    const int half_u = 1 << (u - 1), lg = __ffs(gs) - 1, half_g = gs >> 1;
    int q8[32];
    #pragma unroll
    for (int k = 0; k < 32; ++k) q8[k] = (int)rintf(x[k] * inv_sa);   // only FP round / elem
    const int ng = 32 / gs;
    for (int g = 0; g < ng; ++g) {
        int ssum = 0;
        for (int kk = 0; kk < gs; ++kk) {
            const int k = g * gs + kk;
            const int q = q8[k];
            int up = (q >= 0) ? ((q + half_u) >> u) : -(((-q) + half_u) >> u);  // round(q/2^u), signed
            up = max(-qmax, min(qmax, up));
            q_upper[k] = up;
            int res = q - (up << u);
            res = max(smin, min(smax, res));
            ssum += res;
        }
        const int m = (ssum >= 0) ? ((ssum + half_g) >> lg) : -(((-ssum) + half_g) >> lg);  // int round-mean
        r_shared[g] = max(smin, min(smax, m));
    }
    return ea;
}

// Dense LSB bit-pack tail (inverse of extract_code): n codes of `width` bits,
// LSB-first, into a contiguous `nbytes`-byte buffer. A code straddles at most two
// bytes (width<=7). Used by the KV write/append kernels to produce the certified
// token-major planes that kv_decode_* reads back.
__device__ __forceinline__ void pack_codes_lsb(
        const int* __restrict__ codes, int n, int width, uint8_t* __restrict__ buf, int nbytes) {
    for (int i = 0; i < nbytes; ++i) buf[i] = 0;
    for (int i = 0; i < n; ++i) {
        const uint32_t code = (uint32_t)codes[i] & ((1u << width) - 1u);
        const int bit = i * width, by = bit >> 3, off = bit & 7;
        buf[by] |= (uint8_t)(code << off);
        if (off + width > 8 && by + 1 < nbytes) buf[by + 1] |= (uint8_t)(code >> (8 - off));
    }
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
