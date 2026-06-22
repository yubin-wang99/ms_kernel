// csrc/kv_attention.cu  —  [pure CUDA]  KV-cache flash-decode attention
//
// Decode-step attention with K/V stored MSAQ-signed and dequantized on the fly
// inside the FlashAttention load path (doc §"KV Cache ... Decode (Fused
// Attention)"). Separating dequant into a standalone pass would double KV
// traffic, so it is fused here.
//
// STATUS: dequant reuses the certified ms::unpack_ms_kv_elem, so the math
// matches ms_lib.reference.kv_attention; certify with
// tests/test_kv.py::test_kv_decode_attention_vs_oracle.
//
// OCCUPANCY: SPLIT-KV / FLASH-DECODING. The original kernel launched one block
// per head (H=8 blocks) -> on a 82-SM RTX 3090 that leaves >90% of the SMs idle,
// a hard launch-config ceiling no per-SM tuning can lift. We now split the key
// axis into tiles of KV_TILE keys: grid = (H, S) with S = ceil(Lk/KV_TILE), so
// the block count grows 8 -> 8*S and fills the machine. Each (h,s) block runs
// online softmax over only its key tile and writes a PARTIAL (acc, m, l); a
// second combine kernel merges the S partials per head with the standard
// online-softmax rescale (m_g = max_s m_s; weight each partial by exp(m_s-m_g)).
// This is the memory-parallelism fix, not the compute fix.
//
// WHAT IS STILL DELIBERATELY UNOPTIMIZED HERE (the next phase):
//   * Q·K^T is a per-key block reduction over head_dim, and P·V is a per-thread
//     scalar accumulate. FlashAttention does both as TILED TENSOR-CORE matmuls;
//     swap those in once correct.
//   * Single decode step (Lq=1): the query is the newest token, so it attends
//     to ALL Lk cached keys -> no causal mask needed (matches the oracle with
//     Lq=1). Multi-step / chunked decode adds masking.
//
// Layout: K/V planes are TOKEN-MAJOR [H, nb, L, UB|SB] (BYTES innermost; Stage
// 4a) so a warp's 32 head_dim reads at a fixed key coalesce. One block per
// (head, key-tile); one thread per head_dim element e (blockDim padded to pow2).

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_pipeline.h>
#include <cstdlib>
#include <math.h>
#include <type_traits>
#include "core/ms_utils.cuh"
#include <ATen/cuda/CUDAContext.h>
#include <mma.h>

namespace {
using namespace nvcuda;

constexpr int BLOCK = 32;
constexpr int WSK = BLOCK + 8;   // shared K-tile width (+pad) — matches wa_gemm.cu
constexpr int KV_CHUNK = 128;   // keys processed per chunk (bounds shared mem)
constexpr int CP_CHUNK = 64;    // keys per cp.async double-buffer chunk (smaller smem)

// cp.async-copy n bytes global->shared in 4-byte chunks (UB,SB multiples of 4 keep
// the offsets 4-aligned); a <4-byte tail (small shared planes) is copied directly.
__device__ __forceinline__ void cpa(unsigned char* dst, const unsigned char* src,
                                     int n, int tid, int nt) {
    // Pick the WIDEST aligned cp.async width: 16B (LDGSTS.128) is ~4x fewer
    // transactions than 4B and is what makes the upper-plane staging hit full
    // bandwidth. u4 (UB=16) is 16B-aligned end-to-end; u3/u2 (UB=20/24) fall to
    // 8/4B. The (dst|src|n) test is uniform across the warp -> branch is free.
    const unsigned amask = (unsigned)(((uintptr_t)src | (uintptr_t)dst) | (unsigned)n);
    if ((amask & 15) == 0) {
        for (int off = tid * 16; off < n; off += nt * 16)
            __pipeline_memcpy_async(dst + off, src + off, 16);
    } else if ((amask & 7) == 0) {
        for (int off = tid * 8; off < n; off += nt * 8)
            __pipeline_memcpy_async(dst + off, src + off, 8);
    } else {
        for (int off = tid * 4; off < n; off += nt * 4)   // n is a multiple of 4 (UB*nC)
            __pipeline_memcpy_async(dst + off, src + off, 4);
    }
}
// synchronous byte copy (for the tiny shared-code planes whose SB=2 key stride can
// leave the source 2-mod-4 -> cp.async would misalign; they're small, no need to hide).
__device__ __forceinline__ void sync_copy(unsigned char* dst, const unsigned char* src,
                                          int n, int tid, int nt) {
    for (int off = tid; off < n; off += nt) dst[off] = src[off];
}
// keys per split (key_tile) and #tiles (S) are chosen at launch from the live SM
// count (ms::kv_split_count) so grid = (H, S) lands at ~2-4x #SM blocks.

// warp-wide sum reduction (no __syncthreads); result valid in lane 0.
__device__ __forceinline__ float warp_reduce_sum(float v) {
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffffu, v, o);
    return v;
}

// ---- Phase 1: per (head, key-tile) partial flash-attention (방안 3: barrier-light)
//   The OLD kernel did a block-wide tree reduction PER KEY (~8 __syncthreads each
//   key) -> the inner loop was barrier-bound, no memory-level parallelism. This
//   two-pass form removes the per-key barrier:
//     Pass 1 (scores): one WARP computes one key's q·K dot via __shfl reduction;
//       warps run different keys concurrently -> loads overlap. No block barrier.
//     Pass 2 (output): thread d accumulates out[d] = Σ_kk p_kk·V[d,kk] by looping
//       keys -> no cross-thread reduction. No block barrier.
//   Only ~2 __syncthreads PER CHUNK (not per key). Scores buffered in shared in
//   KV_CHUNK-key chunks (bounds shared mem); chunks combined by online softmax.
//   Writes the UNnormalized partial (acc, m, l) for the combine kernel.
__global__ void kv_decode_split_kernel(
        const __nv_bfloat16* __restrict__ q,    // [H, D]
        const int8_t*  __restrict__ ks, const uint8_t* __restrict__ ku,
        const uint8_t* __restrict__ kh,         // K planes [H,nb,L,(UB|SB)]
        const int8_t*  __restrict__ vs, const uint8_t* __restrict__ vu,
        const uint8_t* __restrict__ vh,         // V planes
        float* __restrict__ part_o,             // [H, S, D]  partial acc
        float* __restrict__ part_m,             // [H, S]     partial max
        float* __restrict__ part_l,             // [H, S]     partial denom
        int H, int Hkv, int Lk, int Lcap, int D, int NB, int u, int gs, int UB, int SB,
        int key_tile, int S, float sm_scale) {

    const int h = blockIdx.x;                   // q head (grid.x == Hq)
    const int hk = h / (H / Hkv);               // GQA: q head -> kv head (group=Hq/Hkv)
    const int s = blockIdx.y;                   // key-tile index
    const int tid = threadIdx.x;
    const int lane = tid & 31, warpId = tid >> 5, nWarps = blockDim.x >> 5;
    const bool active = tid < D;                // thread owns head_dim d=tid (pass 2)

    extern __shared__ float smem[];
    float* q_sh = smem;                         // [D]
    float* sc   = smem + D;                      // [KV_CHUNK]  this chunk's scores
    if (active) q_sh[tid] = __bfloat162float(q[h * D + tid]);
    __syncthreads();

    const int j0 = s * key_tile;
    const int j1 = min(j0 + key_tile, Lk);
    float m_i = -INFINITY, l_i = 0.0f, acc = 0.0f;   // acc = this thread's out[d]

    for (int cs = j0; cs < j1; cs += KV_CHUNK) {
        const int nC = min(KV_CHUNK, j1 - cs);

        // ---- Pass 1: scores (warp per key; __shfl reduction; no block barrier) ----
        for (int kk = warpId; kk < nC; kk += nWarps) {
            const int j = cs + kk;
            float part = 0.0f;
            for (int d = lane; d < D; d += 32) {     // this warp's lanes cover head_dim
                const int blk = d / BLOCK, kd = d % BLOCK;
                const long base_u = (long)(hk * NB + blk) * UB * Lcap;
                const long base_h = (long)(hk * NB + blk) * SB * Lcap;
                const float ksc = ms::e8m0_to_scale(ks[(hk * NB + blk) * Lcap + j]);
                part += q_sh[d] * (float)ms::unpack_ms_kv_elem(ku, kh, base_u, base_h,
                                                Lk, j, kd, u, gs, UB, SB) * ksc;
            }
            part = warp_reduce_sum(part);
            if (lane == 0) sc[kk] = part * sm_scale;
        }
        __syncthreads();

        // ---- chunk max (recomputed per thread from shared; cheap, no barrier) ----
        float m_chunk = -INFINITY;
        for (int kk = 0; kk < nC; ++kk) m_chunk = fmaxf(m_chunk, sc[kk]);
        const float m_new = fmaxf(m_i, m_chunk);
        const float alpha = expf(m_i - m_new);

        // ---- Pass 2: out[d] += Σ_kk p_kk·V[d,kk]  (thread d; no block barrier) ----
        const int blk = tid / BLOCK, kd = tid % BLOCK;       // d = tid
        const long base_u = (long)(hk * NB + blk) * UB * Lcap;
        const long base_h = (long)(hk * NB + blk) * SB * Lcap;
        float lsum = 0.0f, a = 0.0f;
        for (int kk = 0; kk < nC; ++kk) {
            const float p = expf(sc[kk] - m_new);
            lsum += p;                            // identical across threads -> l_i
            if (active) {
                const int j = cs + kk;
                const float vsc = ms::e8m0_to_scale(vs[(hk * NB + blk) * Lcap + j]);
                a += p * (float)ms::unpack_ms_kv_elem(vu, vh, base_u, base_h,
                                                Lk, j, kd, u, gs, UB, SB) * vsc;
            }
        }
        l_i = l_i * alpha + lsum;
        acc = acc * alpha + a;
        m_i = m_new;
        __syncthreads();                          // protect sc[] before next chunk
    }

    // ---- write partials (UNnormalized: combine merges across tiles) ----
    if (active) part_o[((long)h * S + s) * D + tid] = acc;
    if (tid == 0) { part_m[h * S + s] = m_i; part_l[h * S + s] = l_i; }
}

// ---- cp.async variant: same two-pass, but each CP_CHUNK of keys' K/V packed
//   bytes (upper+shared planes, all NB blocks) are PREFETCHED to shared via
//   cp.async while the previous chunk is being unpacked -> the loads overlap the
//   unpack (hide it behind memory). Staged tile is [NB][CP_CHUNK][BYTES]; unpack
//   reads it with base_u=blk*CP_CHUNK*UB, key=local (reusing unpack_ms_kv_elem).
//   Scales are tiny -> read from global. -------------------------------------
__global__ void kv_decode_cpasync_kernel(
        const __nv_bfloat16* __restrict__ q,
        const int8_t*  __restrict__ ks, const uint8_t* __restrict__ ku,
        const uint8_t* __restrict__ kh,
        const int8_t*  __restrict__ vs, const uint8_t* __restrict__ vu,
        const uint8_t* __restrict__ vh,
        float* __restrict__ part_o, float* __restrict__ part_m, float* __restrict__ part_l,
        int H, int Hkv, int Lk, int Lcap, int D, int NB, int u, int gs, int UB, int SB,
        int key_tile, int S, float sm_scale, int diag) {
    const int h = blockIdx.x, s = blockIdx.y, tid = threadIdx.x;
    const int hk = h / (H / Hkv);                  // GQA: q head -> kv head
    const int lane = tid & 31, warpId = tid >> 5, nWarps = blockDim.x >> 5, NT = blockDim.x;
    const bool active = tid < D;
    const int wbits = 8 - u;                       // diag: raw-byte index for memory ceiling

    extern __shared__ unsigned char smem_cp[];
    float* q_sh = (float*)smem_cp;
    float* sc   = q_sh + D;                       // [CP_CHUNK]
    unsigned char* base = (unsigned char*)(sc + CP_CHUNK);
    const int kuB = NB * CP_CHUNK * UB, khB = NB * CP_CHUNK * SB;
    const int slotB = 2 * (kuB + khB);            // K(up,sh) + V(up,sh)
    unsigned char* buf[2] = { base, base + slotB };
    if (active) q_sh[tid] = __bfloat162float(q[h * D + tid]);

    auto stage = [&](int cs, int nC, unsigned char* b) {
        unsigned char* pKu = b, *pKh = b + kuB, *pVu = b + kuB + khB, *pVh = b + 2*kuB + khB;
        for (int blk = 0; blk < NB; ++blk) {
            const long bu = (long)(hk * NB + blk) * Lcap * UB + (long)cs * UB;
            const long bh = (long)(hk * NB + blk) * Lcap * SB + (long)cs * SB;
            cpa(pKu + blk*CP_CHUNK*UB, ku + bu, nC*UB, tid, NT);       // big: async
            cpa(pVu + blk*CP_CHUNK*UB, vu + bu, nC*UB, tid, NT);
            if (diag != 3) {  // diag3: skip shared-plane sync_copy to isolate its cost
                sync_copy(pKh + blk*CP_CHUNK*SB, kh + bh, nC*SB, tid, NT); // tiny: sync
                sync_copy(pVh + blk*CP_CHUNK*SB, vh + bh, nC*SB, tid, NT);
            }
        }
    };

    const int j0 = s * key_tile, j1 = min(j0 + key_tile, Lk);
    float m_i = -INFINITY, l_i = 0.0f, acc = 0.0f;
    if (j0 >= j1) { if (active) part_o[((long)h*S+s)*D+tid]=0.0f;
                    if (tid==0){part_m[h*S+s]=-INFINITY; part_l[h*S+s]=0.0f;} return; }

    stage(j0, min(CP_CHUNK, j1 - j0), buf[0]);
    __pipeline_commit();

    int ci = 0;
    for (int cs = j0; cs < j1; cs += CP_CHUNK, ++ci) {
        const int nC = min(CP_CHUNK, j1 - cs);
        const int cur = ci & 1;
        const bool more = (cs + CP_CHUNK) < j1;
        if (more) { stage(cs + CP_CHUNK, min(CP_CHUNK, j1 - cs - CP_CHUNK), buf[(ci+1)&1]);
                    __pipeline_commit(); }
        __pipeline_wait_prior(more ? 1 : 0);
        __syncthreads();
        unsigned char* b = buf[cur];
        const unsigned char* pKu = b, *pKh = b + kuB, *pVu = b + kuB + khB, *pVh = b + 2*kuB + khB;

        // Pass 1: scores (warp per key, __shfl) — K from staged shared
        for (int kk = warpId; kk < nC; kk += nWarps) {
            const int j = cs + kk;
            float part = 0.0f;
            for (int d = lane; d < D; d += 32) {
                const int blk = d / BLOCK, kd = d % BLOCK;
                const float ksc = ms::e8m0_to_scale(ks[(hk * NB + blk) * Lcap + j]);
                // diag==1: memory ceiling -> read 1 staged upper byte, skip unpack
                float kv;
                if (diag == 1 || diag == 3)  // memory ceiling: 1 staged upper byte, no unpack
                    kv = (float)pKu[(long)blk*CP_CHUNK*UB + (long)kk*UB + ((kd*wbits)>>3)];
                else if (u == 4)      // fast path: nibble-aligned bfe, no straddle
                    kv = (float)ms::unpack_ms_kv_elem_u4(pKu, pKh,
                            (long)blk*CP_CHUNK*UB, (long)blk*CP_CHUNK*SB, kk, kd, gs, UB, SB);
                else
                    kv = (float)ms::unpack_ms_kv_elem(pKu, pKh,
                            (long)blk*CP_CHUNK*UB, (long)blk*CP_CHUNK*SB, 0, kk, kd,
                            u, gs, UB, SB);
                part += q_sh[d] * kv * ksc;
            }
            part = warp_reduce_sum(part);
            if (lane == 0) sc[kk] = part * sm_scale;
        }
        __syncthreads();
        float m_chunk = -INFINITY;
        for (int kk = 0; kk < nC; ++kk) m_chunk = fmaxf(m_chunk, sc[kk]);
        const float m_new = fmaxf(m_i, m_chunk);
        const float alpha = expf(m_i - m_new);

        // Pass 2: output (thread d) — V from staged shared
        const int blk = tid / BLOCK, kd = tid % BLOCK;
        float lsum = 0.0f, a = 0.0f;
        for (int kk = 0; kk < nC; ++kk) {
            // diag!=0: skip exp (linear weight) to isolate softmax-exp cost
            const float p = (diag == 0) ? expf(sc[kk] - m_new) : (sc[kk] - m_new);
            lsum += p;
            if (active) {
                const int j = cs + kk;
                const float vsc = ms::e8m0_to_scale(vs[(hk * NB + blk) * Lcap + j]);
                float vv;
                if (diag == 1)
                    vv = (float)pVu[(long)blk*CP_CHUNK*UB + (long)kk*UB + ((kd*wbits)>>3)];
                else if (u == 4)
                    vv = (float)ms::unpack_ms_kv_elem_u4(pVu, pVh,
                            (long)blk*CP_CHUNK*UB, (long)blk*CP_CHUNK*SB, kk, kd, gs, UB, SB);
                else
                    vv = (float)ms::unpack_ms_kv_elem(pVu, pVh,
                            (long)blk*CP_CHUNK*UB, (long)blk*CP_CHUNK*SB, 0, kk, kd,
                            u, gs, UB, SB);
                a += p * vv * vsc;
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

// ---- u==4 KEY-PER-THREAD WIDE-READ kernel (change.md Phase 18, fix plan A) ---
//   DIAGNOSIS (Phase 17): the warp-per-key (Pass1) / thread-per-d (Pass2) mapping
//   reads only HALF a memory sector of USEFUL bytes — a warp's 32 lanes want 32
//   u4 codes = 16 bytes, but a DRAM/L2 sector is 32 bytes, so MSAQ's effective BW
//   is ~38% of MXINT8 (which packs 1 byte/elem -> a warp's 32 int8 fill a full
//   sector). That, not unpack or exp, is why MSAQ KV stays ~1.5x slower despite
//   reading 0.56x the bytes. This is the GEMV-wide win (change.md Phase 14/16)
//   ported to KV: assign one THREAD per key. The token-major plane [H,NB,Lk,UB]
//   already stores a key's UB=16 bytes contiguously with consecutive keys 16 B
//   apart, so consecutive threads -> consecutive keys -> a warp reads 512 B
//   fully-contiguous, fully-useful (100% sector util) in ONE uint4 load/block.
//   No repack (identical bytes) -> bit-exact at u=4.
//     Pass 1 (scores): thread t owns key cs+t. It uint4-loads K[key,block] for
//       each of NB blocks, bfe-extracts all 32 codes from registers (GEMV style),
//       and accumulates the FULL q·K dot in-thread (no warp reduction) -> sc[t].
//     Pass 2 (output): V can't be thread-per-key (the output is reduced OVER
//       keys, so a key-owning thread would have to scatter into all D outputs).
//       Instead STAGE this chunk's V coalesced into shared (a contiguous copy is
//       already 100%-util) then run the easy thread-per-d accumulate from shared
//       -> the global V read is wide/coalesced, the narrow part is on-chip. (The
//       cp.async kernel staged the same way but its double-buffer barriers cost
//       more than they saved in this BW-bound regime — here it's one sync copy.)
//   All u: u4 uses one uint4 load + bfe (UB=16); u2/u3 (UB=20/24, not int4-
//   aligned) load the key's UB bytes as 4-aligned uint32 words into registers
//   and extract with the general straddle path — the coalescing win comes from
//   the thread-per-key mapping, not the vector width, so no repack is needed.
//   Templated on U4 with `if constexpr` so each instantiation drops the other
//   path's registers (a merged runtime branch bloated u4 regs -> killed its
//   occupancy). -----------------------------------------------------------------
template<bool U4, bool VPACK>
__global__ void kv_decode_wide_kernel(
        const __nv_bfloat16* __restrict__ q,
        const int8_t*  __restrict__ ks, const uint8_t* __restrict__ ku,
        const uint8_t* __restrict__ kh,
        const int8_t*  __restrict__ vs, const uint8_t* __restrict__ vu,
        const uint8_t* __restrict__ vh,
        float* __restrict__ part_o, float* __restrict__ part_m, float* __restrict__ part_l,
        int H, int Hkv, int Lk, int Lcap, int D, int NB, int u, int gs, int UB, int SB,
        int key_tile, int S, int chunk, float sm_scale, int v8, int sepsc, int vt, int qrot) {
    const int h = blockIdx.x, s = blockIdx.y, tid = threadIdx.x, NT = blockDim.x;
    const int hk = h / (H / Hkv);                  // GQA: q head -> kv head
    const int gs_shift = __ffs(gs) - 1;            // gs is pow2: k/gs == k>>shift
    const bool active = tid < D;                   // pass-2 owns head_dim d=tid

    // batched decode: grid.z = batch. b==0 (grid.z==1) leaves the single-token
    // path byte-identical; per-batch planes are contiguous so offset the pointers.
    const long b = blockIdx.z;
    q += b * (long)H * D;
    ks += b * (long)Hkv * NB * Lcap; ku += b * (long)Hkv * NB * Lcap * UB; kh += b * (long)Hkv * NB * Lcap * SB;
    vs += b * (long)Hkv * NB * Lcap; vu += b * (long)Hkv * NB * Lcap * UB; vh += b * (long)Hkv * NB * Lcap * SB;
    part_o += b * (long)H * S * D; part_m += b * (long)H * S; part_l += b * (long)H * S;

    extern __shared__ unsigned char smem_w[];
    float* q_sh  = (float*)smem_w;                 // [D]
    float* qg_sh = q_sh + D;                        // [NB*BLOCK] q group-sums (separated-scale)
    float* sc    = qg_sh + (long)NB * BLOCK;        // [chunk] scores
    unsigned char* pVu = (unsigned char*)(sc + chunk);  // [NB*chunk*UB] staged V upper
    unsigned char* pVh = pVu + (long)NB * chunk * UB;   // [NB*chunk*SB] staged V shared
    if (active) q_sh[tid] = __bfloat162float(q[h * D + tid]);
    __syncthreads();
    // ---- FUSED ONLINE Q-ROTATION (post-RoPE H_D, orthonormal) ----------------
    //   Mirrors the KV-Key rotation done at append (kv_append_rot) so QK^T is
    //   preserved: (Q·H)(K·H)^T = Q·K^T. Done here in the prologue (once/block,
    //   all D threads active) so it costs ZERO extra launch — its marginal cost
    //   is what MS_KV_QROT=1 adds over =0. FWHT: 7 butterfly stages, 1/sqrt(D).
    if (qrot) {
        for (int hh = 1; hh < D; hh <<= 1) {
            float nv = 0.0f;
            if (active) {
                const float ve = q_sh[tid], vp = q_sh[tid ^ hh];
                nv = (tid & hh) ? (vp - ve) : (ve + vp);
            }
            __syncthreads();
            if (active) q_sh[tid] = nv;
            __syncthreads();
        }
        if (active) q_sh[tid] *= rsqrtf((float)D);
        __syncthreads();
    }
    // separated-scale: per (blk,g) query group-sum qg = Σ_{d∈group} q[d] (key-independent,
    // computed once). The K/V dot then factors: Σ q·(up·2^u+sh)·s = s·(2^u·Σq·up + Σ_g sh·qg).
    if (sepsc) {
        const int ng = BLOCK >> gs_shift;
        for (int idx = tid; idx < NB * ng; idx += NT) {
            const int blk = idx / ng, g = idx % ng, base = g << gs_shift;
            float acc = 0.0f;
            for (int t = 0; t < gs; ++t) acc += q_sh[blk * BLOCK + base + t];
            qg_sh[blk * BLOCK + g] = acc;
        }
        __syncthreads();
    }

    const int j0 = s * key_tile, j1 = min(j0 + key_tile, Lk);
    float m_i = -INFINITY, l_i = 0.0f, acc = 0.0f;

    for (int cs = j0; cs < j1; cs += chunk) {
        const int nC = min(chunk, j1 - cs);

        // ---- Pass 1: thread t == key cs+t; wide K load; in-thread dot ----
        //   UB==16 (u4): one uint4 + bfe. u2/u3 (UB=20/24, not int4-aligned): load
        //   the key's UB bytes as 4-aligned uint32 words into registers, extract
        //   with the general straddle path (bit-exact). Coalescing comes from the
        //   thread-per-key mapping (consecutive keys are UB-contiguous), NOT from
        //   the vector width, so it does not need int4 alignment / a repack.
        if (tid < nC) {
            const int key = cs + tid;
            float dot = 0.0f;
            for (int blk = 0; blk < NB; ++blk) {
                const float ksc = ms::e8m0_to_scale(ks[(hk * NB + blk) * Lcap + key]);
                const long kbase = (long)(hk * NB + blk) * Lcap + key;
                uint8_t sb[8];                              // SB <= 8 shared-code bytes
                const long sbase = kbase * SB;
                #pragma unroll
                for (int i = 0; i < 8; ++i) if (i < SB) sb[i] = kh[sbase + i];
                if constexpr (U4) {                         // u4 fast path: uint4 + bfe
                    const uint4 up4 = *reinterpret_cast<const uint4*>(ku + kbase * UB);
                    const uint32_t uw[4] = { up4.x, up4.y, up4.z, up4.w };
                    if (sepsc) {                            // separated-scale: up per-elem, sh per-group
                        float bup = 0.0f, bsh = 0.0f;
                        #pragma unroll
                        for (int kd = 0; kd < BLOCK; ++kd)
                            bup += q_sh[blk * BLOCK + kd] * ms::bfe_s32((int)uw[kd >> 3], (kd & 7) * 4, 4);
                        const int ng = BLOCK >> gs_shift;
                        for (int g = 0; g < ng; ++g)
                            bsh += ms::bfe_s32((int)sb[g >> 1], (g & 1) * 4, 4) * qg_sh[blk * BLOCK + g];
                        dot += ksc * (16.0f * bup + bsh);
                    } else {
                        #pragma unroll
                        for (int kd = 0; kd < BLOCK; ++kd) {
                            const int up_code = ms::bfe_s32((int)uw[kd >> 3], (kd & 7) * 4, 4);
                            const int g       = kd >> gs_shift;
                            const int sh_code = ms::bfe_s32((int)sb[g >> 1], (g & 1) * 4, 4);
                            dot += q_sh[blk * BLOCK + kd] * ((up_code * 16 + sh_code) * ksc);
                        }
                    }
                } else {                                    // u2/u3: STREAMING bit-buffer unpack
                    // thread-per-key unpacks ALL 32 codes of its key sequentially, so the
                    // rolling 64-bit buffer (one shift+mask per code, refill a word only when
                    // low) replaces the per-code straddle funnel-shift -> ~the W-only GEMV
                    // Phase-20 win ported to the KV read (the u2/u3 path was left on the heavy
                    // general unpack). Bit-exact (same codes). MXINT8 reads int8 directly so
                    // no matched change -> this is a mantissa-sharing-only optimization.
                    uint32_t ureg[6];                       // UB/4 <= 6 words (u2: 24B)
                    const uint32_t* src = reinterpret_cast<const uint32_t*>(ku + kbase * UB);
                    #pragma unroll
                    for (int i = 0; i < 6; ++i) if (i < (UB >> 2)) ureg[i] = src[i];
                    uint32_t sreg[3] = {0u,0u,0u};
                    #pragma unroll
                    for (int i = 0; i < 8; ++i) if (i < SB) sreg[i >> 2] |= (uint32_t)sb[i] << (8 * (i & 3));
                    const int wbits = 8 - u, gsmask = gs - 1;
                    const uint32_t umask = (1u << wbits) - 1u, usign = 1u << (wbits - 1);
                    const uint32_t smask = (1u << u) - 1u, ssign = 1u << (u - 1);
                    uint64_t ubuf = 0; int unb = 0, uwi = 0;
                    uint64_t sbuf = 0; int snb = 0, swi = 0; int sh_code = 0;
                    for (int kd = 0; kd < BLOCK; ++kd) {
                        if (unb < wbits) { ubuf |= (uint64_t)ureg[uwi++] << unb; unb += 32; }
                        const int up_code = (int)(((uint32_t)ubuf & umask) ^ usign) - (int)usign;
                        ubuf >>= wbits; unb -= wbits;
                        if ((kd & gsmask) == 0) {
                            if (snb < u) { sbuf |= (uint64_t)sreg[swi++] << snb; snb += 32; }
                            sh_code = (int)(((uint32_t)sbuf & smask) ^ ssign) - (int)ssign;
                            sbuf >>= u; snb -= u;
                        }
                        const int w = up_code * (1 << u) + sh_code;
                        dot += q_sh[blk * BLOCK + kd] * (w * ksc);
                    }
                }
            }
            sc[tid] = dot * sm_scale;
        }

        // ---- stage this chunk's V into shared, coalesced (100%-util global read) ----
        //   u4: 1 uint4 == 1 key. u2/u3: copy the contiguous nC*UB bytes as uint32
        //   words (UB is a multiple of 4) -> still fully coalesced.
        for (int blk = 0; blk < NB; ++blk) {
            if constexpr (VPACK) {
                // L1+L3 staging: 2 keys/thread, write PACKED+TRANSPOSED nibbles (no write-race).
                //   vp_up[blk][kd][pi] = up(2pi)@kd | up(2pi+1)@kd<<4 ; vp_sh[blk][g][pi] = sh pair.
                const int ng_v = BLOCK >> gs_shift;
                int CHP = (chunk + 1) >> 1; CHP = (CHP + 3) & ~3; if (((CHP >> 2) & 1) == 0) CHP += 4;
                const long VPUP = (long)NB * BLOCK * CHP;
                const uint4* vsrc = reinterpret_cast<const uint4*>(
                        vu + ((long)(hk * NB + blk) * Lcap + cs) * UB);
                const unsigned char* vsh = vh + ((long)(hk * NB + blk) * Lcap + cs) * SB;
                uint8_t* up_dst = (uint8_t*)pVu + (long)blk * BLOCK * CHP;
                uint8_t* sh_dst = (uint8_t*)pVu + VPUP + (long)blk * ng_v * CHP;
                const int NP = (nC + 1) >> 1;
                for (int pi = tid; pi < NP; pi += NT) {
                    const int ka = 2 * pi, kb = 2 * pi + 1;
                    const uint4 ua = vsrc[ka];
                    uint4 ub; if (kb < nC) ub = vsrc[kb]; else { ub.x = ub.y = ub.z = ub.w = 0; }
                    const uint32_t uwa[4] = { ua.x, ua.y, ua.z, ua.w };
                    const uint32_t uwb[4] = { ub.x, ub.y, ub.z, ub.w };
                    #pragma unroll
                    for (int kd2 = 0; kd2 < BLOCK; ++kd2) {
                        const uint32_t na = (uwa[kd2 >> 3] >> ((kd2 & 7) * 4)) & 0xF;
                        const uint32_t nb = (uwb[kd2 >> 3] >> ((kd2 & 7) * 4)) & 0xF;
                        up_dst[(long)kd2 * CHP + pi] = (uint8_t)(na | (nb << 4));
                    }
                    uint8_t sa[8], sb[8];
                    #pragma unroll
                    for (int t = 0; t < 8; ++t) {
                        sa[t] = (t < SB) ? vsh[ka * SB + t] : 0;
                        sb[t] = (kb < nC && t < SB) ? vsh[kb * SB + t] : 0;
                    }
                    for (int gg = 0; gg < ng_v; ++gg) {
                        const uint32_t na = (sa[gg >> 1] >> ((gg & 1) * 4)) & 0xF;
                        const uint32_t nb = (sb[gg >> 1] >> ((gg & 1) * 4)) & 0xF;
                        sh_dst[(long)gg * CHP + pi] = (uint8_t)(na | (nb << 4));
                    }
                }
            } else if (U4 && v8) {
                // u4 fast Pass-2: stage V already reconstructed as int8 codes (up*16+sh,
                // in [-120,119]). Pass-2 then reads ONE int8/elem (no bfe, no gs shared
                // plane) -> identical to MXINT8's Pass-2, halving Pass-2 shared traffic.
                const uint4* vsrc = reinterpret_cast<const uint4*>(
                        vu + ((long)(hk * NB + blk) * Lcap + cs) * UB);
                const unsigned char* vsh = vh + ((long)(hk * NB + blk) * Lcap + cs) * SB;
                int8_t* d8 = (int8_t*)pVu;
                const int CH = chunk + 4;            // vt: pad so CH/4 is odd -> conflict-free int32 reads
                for (int i = tid; i < nC; i += NT) {
                    const uint4 up4 = vsrc[i];
                    const uint32_t uw[4] = { up4.x, up4.y, up4.z, up4.w };
                    uint8_t sb[8];
                    #pragma unroll
                    for (int t = 0; t < 8; ++t) if (t < SB) sb[t] = vsh[i * SB + t];
                    #pragma unroll
                    for (int kd = 0; kd < BLOCK; ++kd) {
                        const int up_code = ms::bfe_s32((int)uw[kd >> 3], (kd & 7) * 4, 4);
                        const int g       = kd >> gs_shift;
                        const int sh_code = ms::bfe_s32((int)sb[g >> 1], (g & 1) * 4, 4);
                        const int8_t code = (int8_t)(up_code * 16 + sh_code);
                        if (vt) d8[(long)blk * BLOCK * CH + kd * CH + i] = code;   // [blk][kd][kk]
                        else    d8[(long)blk * chunk * BLOCK + i * BLOCK + kd] = code;
                    }
                }
            } else if constexpr (U4) {
                const uint4* src4 = reinterpret_cast<const uint4*>(
                        vu + ((long)(hk * NB + blk) * Lcap + cs) * UB);
                uint4* dst4 = reinterpret_cast<uint4*>(pVu + (long)blk * chunk * UB);
                for (int i = tid; i < nC; i += NT) dst4[i] = src4[i];
            } else {
                const uint32_t* s32 = reinterpret_cast<const uint32_t*>(
                        vu + ((long)(hk * NB + blk) * Lcap + cs) * UB);
                uint32_t* d32 = reinterpret_cast<uint32_t*>(pVu + (long)blk * chunk * UB);
                const int nw = (nC * UB) >> 2;
                for (int i = tid; i < nw; i += NT) d32[i] = s32[i];
            }
            if constexpr (!VPACK) {        // v8/vpack consumed the shared plane during reconstruct/pack
                if (!(U4 && v8)) {
                    const unsigned char* srch = vh + ((long)(hk * NB + blk) * Lcap + cs) * SB;
                    unsigned char* dsth = pVh + (long)blk * chunk * SB;
                    for (int i = tid; i < nC * SB; i += NT) dsth[i] = srch[i];  // tiny shared plane
                }
            }
        }
        __syncthreads();   // Pass-1 sc + staged V visible before Pass 2

        // ---- chunk max (per thread from shared) ----
        float m_chunk = -INFINITY;
        for (int kk = 0; kk < nC; ++kk) m_chunk = fmaxf(m_chunk, sc[kk]);
        const float m_new = fmaxf(m_i, m_chunk);
        const float alpha = expf(m_i - m_new);

        // Scores are per-KEY (q·k, d-independent) so m_new is identical across the 32 d-lanes.
        // Convert sc -> probabilities ONCE per kk (cooperative) instead of 32x redundantly per d
        // (the per-thread expf was ~half the SM compute at full occupancy / high batch).
        __syncthreads();
        for (int kk = tid; kk < nC; kk += NT) sc[kk] = __expf(sc[kk] - m_new);
        __syncthreads();

        // ---- Pass 2: out[d] += Σ_kk p_kk·V[d,kk]  (thread d, V from staged shared) ----
        const int blk = tid / BLOCK, kd = tid % BLOCK;
        float lsum = 0.0f, a = 0.0f;
        if constexpr (VPACK) { if (active) {
            // L1+L3: V staged PACKED & TRANSPOSED at nibble density (vp[blk][kd][kk], 2 codes/byte).
            // This lane's kk-nibbles are CONTIGUOUS → one int32 = 8 codes (vs v8's 4), conflict-free
            // (CHP/4 odd), decoded in registers. Shared traffic 1.0→~0.75× (gs2) on the bottleneck.
            const int g = kd >> gs_shift, ng_v = BLOCK >> gs_shift;
            int CHP = (chunk + 1) >> 1; CHP = (CHP + 3) & ~3; if (((CHP >> 2) & 1) == 0) CHP += 4;
            const long VPUP = (long)NB * BLOCK * CHP;
            const uint8_t* up_row = (const uint8_t*)pVu + (long)(blk * BLOCK + kd) * CHP;
            const uint8_t* sh_row = (const uint8_t*)pVu + VPUP + (long)(blk * ng_v + g) * CHP;
            const long vscb = (long)(hk * NB + blk) * Lcap + cs;
            int kk = 0;
            for (; kk + 7 < nC; kk += 8) {
                const uint32_t uw = *reinterpret_cast<const uint32_t*>(up_row + (kk >> 1));   // 8 up-nibbles
                const uint32_t sw = *reinterpret_cast<const uint32_t*>(sh_row + (kk >> 1));   // 8 sh-nibbles
                #pragma unroll
                for (int t = 0; t < 8; ++t) {
                    const float p = sc[kk + t]; lsum += p;   // sc already = probability
                    const int V = ms::bfe_s32((int)uw, t * 4, 4) * 16 + ms::bfe_s32((int)sw, t * 4, 4);
                    a += (p * ms::e8m0_to_scale(vs[vscb + kk + t])) * (float)V;
                }
            }
            for (; kk < nC; ++kk) {
                const float p = sc[kk]; lsum += p;
                const int sft = (kk & 1) * 4;
                const int V = ms::bfe_s32((int)(up_row[kk >> 1] >> sft), 0, 4) * 16
                            + ms::bfe_s32((int)(sh_row[kk >> 1] >> sft), 0, 4);
                a += (p * ms::e8m0_to_scale(vs[vscb + kk])) * (float)V;
            }
        }} else {                          // close if(active) + if constexpr(VPACK)
        if (active && U4 && v8 && vt) {
            // transposed+padded staging: this thread's kk-codes are CONTIGUOUS -> read 4 at a
            // time via one int32 (conflict-free, 4x fewer shared transactions). p/exp unchanged.
            const int CH = chunk + 4;
            const int8_t* vbase = (const int8_t*)pVu + (long)blk * BLOCK * CH + (long)kd * CH;
            const long vscb = (long)(hk * NB + blk) * Lcap + cs;
            int kk = 0;
            for (; kk + 3 < nC; kk += 4) {
                const int32_t w4 = *reinterpret_cast<const int32_t*>(vbase + kk);
                #pragma unroll
                for (int t = 0; t < 4; ++t) {
                    const float p = sc[kk + t]; lsum += p;
                    const float vsc = ms::e8m0_to_scale(vs[vscb + kk + t]);
                    a += p * (float)((int8_t)((w4 >> (8 * t)) & 0xff)) * vsc;
                }
            }
            for (; kk < nC; ++kk) {
                const float p = sc[kk]; lsum += p;
                a += p * (float)vbase[kk] * ms::e8m0_to_scale(vs[vscb + kk]);
            }
        } else
        for (int kk = 0; kk < nC; ++kk) {
            const float p = sc[kk];   // sc already = probability
            lsum += p;
            if (active) {
                const int j = cs + kk;
                const float vsc = ms::e8m0_to_scale(vs[(hk * NB + blk) * Lcap + j]);
                float vv;
                if (U4 && v8)
                    vv = (float)((const int8_t*)pVu)[(long)blk * chunk * BLOCK + kk * BLOCK + kd];
                else if constexpr (U4)
                    vv = (float)ms::unpack_ms_kv_elem_u4(pVu, pVh,
                        (long)blk * chunk * UB, (long)blk * chunk * SB, kk, kd, gs, UB, SB);
                else
                    vv = (float)ms::unpack_ms_kv_elem(pVu, pVh,
                        (long)blk * chunk * UB, (long)blk * chunk * SB, 0, kk, kd, u, gs, UB, SB);
                a += p * vv * vsc;
            }
        }
        }   // close else (!VPACK)
        l_i = l_i * alpha + lsum;
        acc = acc * alpha + a;
        m_i = m_new;
        __syncthreads();   // protect sc[] + staged V before next chunk
    }
    if (active) part_o[((long)h * S + s) * D + tid] = acc;
    if (tid == 0) { part_m[h * S + s] = m_i; part_l[h * S + s] = l_i; }
}

// ---- WARP-TRANSPOSE P·V kernel (design B) — staging removed entirely --------
//   PROBLEM the wide kernel hit (for_fair_comparison.md): Pass-2 wants out[d] =
//   Σ_k p_k·V[k,d] with a thread PER head-dim d, but a coalesced V read is
//   thread PER key (a key's UB=16 B are contiguous, consecutive keys 16 B
//   apart). The wide kernel bridged the two mappings by STAGING V into shared
//   (full-sector global read), but that staging is what neutralized MSAQ:
//   ncu shows it caps occupancy (11 KB smem), adds a __syncthreads barrier
//   (~15% issue stall) and turns MSAQ's lower DRAM traffic (14% vs MXINT8 24%)
//   into L1TEX/shared traffic (41% — MSAQ's top resource). So MSAQ reads fewer
//   sectors (0.73×) yet ties on time.
//   FIX (design B): never touch shared for V. Read V thread-per-key (coalesced,
//   full-sector, 0.56× DRAM bytes), unpack in registers, then do the
//   key→head-dim transpose IN REGISTERS with a 32-lane warp all-reduce. One
//   warp owns one block (NB warps == D/32): lane k loads key (sub*32+k)'s 16 B
//   record for that block, forms row[kd] = p_k·vsc_k·V[k,kd] (kd = 0..31), and a
//   __shfl_xor butterfly all-reduces across the 32 lanes so lane d ends up with
//   Σ_k row_k[d] = the contribution to out[(warp,d)]. No staged V, no barrier in
//   the chunk loop, smem = q_sh[D]+sc[chunk] (~1 KB) → occupancy is freed and
//   MSAQ's fewer-bytes/-sectors advantage finally converts to time.
//   Bit-exact with the wide path (same codes, same up·16+sh arithmetic). U4 only
//   (the 0.56× target); D must be 128 so NB(=4) warps == D/32. Other u / D fall
//   back to the wide kernel. (change.md Phase 34.)
//   MEASURED RESULT (kept as a documented negative, opt-in MS_KV_WARPT): removing
//   the staging barrier/shared did NOT beat the wide kernel — it was ~5-10% SLOWER
//   (Lk4680: 39.5µs vs wide 35.9µs). ncu: shared dropped 11→2 KB but occupancy
//   stayed ~23% because the grid is only ~0.5-0.76 waves (too few blocks to fill
//   the SMs, and the per-SM block cap was never the limiter), while the broadcast
//   transpose's ~192 shfl/subchunk add issue pressure (L1TEX 57%). The wide
//   kernel's bottleneck is not the staging tax alone but the irreducible sub-byte-V
//   half-sector reduction at one-wave occupancy — see for_fair_comparison.md.
template<bool U4>
__global__ void kv_decode_warpT_kernel(
        const __nv_bfloat16* __restrict__ q,
        const int8_t*  __restrict__ ks, const uint8_t* __restrict__ ku,
        const uint8_t* __restrict__ kh,
        const int8_t*  __restrict__ vs, const uint8_t* __restrict__ vu,
        const uint8_t* __restrict__ vh,
        float* __restrict__ part_o, float* __restrict__ part_m, float* __restrict__ part_l,
        int H, int Hkv, int Lk, int Lcap, int D, int NB, int u, int gs, int UB, int SB,
        int key_tile, int S, int chunk, float sm_scale) {
    const int h = blockIdx.x, s = blockIdx.y, tid = threadIdx.x;
    const int hk = h / (H / Hkv);
    const int gs_shift = __ffs(gs) - 1;
    const int warp = tid >> 5, lane = tid & 31;    // warp == block (0..NB-1); lane owns out-dim
    const int d_out = warp * BLOCK + lane;         // this thread's head-dim output index

    extern __shared__ unsigned char smem_t[];
    float* q_sh = (float*)smem_t;                  // [D]
    float* sc   = q_sh + D;                        // [chunk] scores
    if (tid < D) q_sh[tid] = __bfloat162float(q[h * D + tid]);
    __syncthreads();

    const int j0 = s * key_tile, j1 = min(j0 + key_tile, Lk);
    float m_i = -INFINITY, l_i = 0.0f, acc = 0.0f;

    for (int cs = j0; cs < j1; cs += chunk) {
        const int nC = min(chunk, j1 - cs);

        // ---- Pass 1: thread-per-key full q·K dot -> sc[tid] (identical to wide) ----
        if (tid < nC) {
            const int key = cs + tid;
            float dot = 0.0f;
            for (int blk = 0; blk < NB; ++blk) {
                const float ksc = ms::e8m0_to_scale(ks[(hk * NB + blk) * Lcap + key]);
                const long kbase = (long)(hk * NB + blk) * Lcap + key;
                uint8_t sb[8];
                const long sbase = kbase * SB;
                #pragma unroll
                for (int i = 0; i < 8; ++i) if (i < SB) sb[i] = kh[sbase + i];
                const uint4 up4 = *reinterpret_cast<const uint4*>(ku + kbase * UB);
                const uint32_t uw[4] = { up4.x, up4.y, up4.z, up4.w };
                #pragma unroll
                for (int kd = 0; kd < BLOCK; ++kd) {
                    const int up_code = ms::bfe_s32((int)uw[kd >> 3], (kd & 7) * 4, 4);
                    const int g       = kd >> gs_shift;
                    const int sh_code = ms::bfe_s32((int)sb[g >> 1], (g & 1) * 4, 4);
                    dot += q_sh[blk * BLOCK + kd] * ((up_code * 16 + sh_code) * ksc);
                }
            }
            sc[tid] = dot * sm_scale;
        }
        __syncthreads();   // sc[] visible to all lanes before Pass-2

        // ---- online-softmax max + rescale (each thread, all see same sc[]) ----
        float m_chunk = -INFINITY;
        for (int kk = 0; kk < nC; ++kk) m_chunk = fmaxf(m_chunk, sc[kk]);
        const float m_new = fmaxf(m_i, m_chunk);
        const float alpha = expf(m_i - m_new);
        float lsum = 0.0f;
        for (int kk = 0; kk < nC; ++kk) lsum += expf(sc[kk] - m_new);

        // ---- Pass 2: broadcast-transpose P·V. warp owns block 'warp'; 32 keys per
        //   subchunk. Lane k holds key (sub*32+k)'s 16 B record + p·vsc. For each
        //   source key k, its record/scalar are broadcast (__shfl) and THIS lane
        //   extracts only ITS OWN out-dim d=lane → scalar accumulator (no 32-reg
        //   row[] array → no spill → occupancy freed). a[lane] = Σ_k p_k·vsc_k·V[k][lane].
        float a = 0.0f;
        for (int sub = 0; sub * BLOCK < nC; ++sub) {
            const int koff = sub * BLOCK + lane;       // key this lane loads
            float pv = 0.0f; uint32_t w0 = 0, w1 = 0, w2 = 0, w3 = 0, shw = 0;
            if (koff < nC) {
                const int key = cs + koff;
                const float vsc = ms::e8m0_to_scale(vs[(hk * NB + warp) * Lcap + key]);
                pv = expf(sc[koff] - m_new) * vsc;
                const long vbase = (long)(hk * NB + warp) * Lcap + key;
                const uint4 up4 = *reinterpret_cast<const uint4*>(vu + vbase * UB);
                w0 = up4.x; w1 = up4.y; w2 = up4.z; w3 = up4.w;
                #pragma unroll
                for (int i = 0; i < 8; ++i) if (i < SB) shw |= (uint32_t)vh[vbase * SB + i] << (8 * i);
            }
            const int g = lane >> gs_shift;            // this lane's shared-code group
            #pragma unroll
            for (int k = 0; k < BLOCK; ++k) {
                const float    pvk = __shfl_sync(0xffffffffu, pv, k);
                const uint32_t b0  = __shfl_sync(0xffffffffu, w0, k);
                const uint32_t b1  = __shfl_sync(0xffffffffu, w1, k);
                const uint32_t b2  = __shfl_sync(0xffffffffu, w2, k);
                const uint32_t b3  = __shfl_sync(0xffffffffu, w3, k);
                const uint32_t bs  = __shfl_sync(0xffffffffu, shw, k);
                const uint32_t wd  = (lane < 8) ? b0 : (lane < 16) ? b1 : (lane < 24) ? b2 : b3;
                const int up_code  = ms::bfe_s32((int)wd, (lane & 7) * 4, 4);
                const int sh_code  = ms::bfe_s32((int)((bs >> (8 * (g >> 1))) & 0xff), (g & 1) * 4, 4);
                a += pvk * (float)(up_code * 16 + sh_code);
            }
        }
        acc = acc * alpha + a;
        l_i = l_i * alpha + lsum;
        m_i = m_new;
        // no per-chunk barrier needed: sc[] is re-synced at the top of Pass-1.
        __syncthreads();   // protect sc[] before next chunk overwrites it
    }
    if (d_out < D) part_o[((long)h * S + s) * D + d_out] = acc;
    if (tid == 0) { part_m[h * S + s] = m_i; part_l[h * S + s] = l_i; }
}

// ---- GQA-BATCHED flash-decode (design A): one block per (kv_head, key-tile) ---
//   Processes ALL G = Hq/Hkv query heads of this kv head TOGETHER. K and V are
//   read+unpacked ONCE per key and reused across the G query rows -> the KV memory
//   traffic is amortized G-fold (the per-q-head kernel re-read each kv head's V G
//   times). Roofline: P.V at GQA G has AI = G*D / (V bytes/key) >> nothing, but
//   far BELOW the BF16 ridge (~75 FLOP/byte) -> memory-bound, so MSAQ's 0.56x
//   bytes reduce time ~0.56x once the (amortized) unpack hides. Pass-1 thread-per-
//   key computes G in-thread dots from one unpacked K; V staged full-sector once,
//   reused for G accumulators in Pass-2. Partials written at the G q-head slots so
//   the existing per-q-head combine is reused. (change.md Phase 33.)
constexpr int MAX_G = 8;
template<bool U4>
__global__ void kv_decode_gqa_kernel(
        const __nv_bfloat16* __restrict__ q,    // [Hq, D]
        const int8_t*  __restrict__ ks, const uint8_t* __restrict__ ku,
        const uint8_t* __restrict__ kh,
        const int8_t*  __restrict__ vs, const uint8_t* __restrict__ vu,
        const uint8_t* __restrict__ vh,
        float* __restrict__ part_o,             // [Hq, S, D]
        float* __restrict__ part_m, float* __restrict__ part_l,   // [Hq, S]
        int Hq, int Hkv, int G, int Lk, int Lcap, int D, int NB, int u, int gs,
        int UB, int SB, int key_tile, int S, int chunk, float sm_scale, int v8, int sepsc, int vt) {
    const int hk = blockIdx.x;                  // kv head (grid.x = Hkv)
    const int s  = blockIdx.y;
    const int tid = threadIdx.x, NT = blockDim.x;
    const int gs_shift = __ffs(gs) - 1;
    const bool active = tid < D;

    extern __shared__ unsigned char smem_g[];
    float* q_sh  = (float*)smem_g;                      // [G][D]
    float* qg_sh = q_sh + (long)G * D;                  // [G][NB*BLOCK] q group-sums (sepsc only)
    float* sc    = qg_sh + (sepsc ? (long)G * NB * BLOCK : 0);  // [G][chunk]
    unsigned char* pVu = (unsigned char*)(sc + (long)G * chunk);  // [NB*chunk*UB] (or int8 if v8)
    unsigned char* pVh = pVu + (long)NB * chunk * UB;            // [NB*chunk*SB]

    for (int g = 0; g < G; ++g)
        if (active) q_sh[g * D + tid] = __bfloat162float(q[(hk * G + g) * D + tid]);
    __syncthreads();
    // separated-scale: per (g,blk,grp) query group-sum, computed once, reused for every key.
    if (sepsc) {
        const int ng = BLOCK >> gs_shift;
        for (int idx = tid; idx < G * NB * ng; idx += NT) {
            const int gg = idx / (NB * ng), rem = idx % (NB * ng);
            const int blk = rem / ng, grp = rem % ng, base = grp << gs_shift;
            float acc = 0.0f;
            for (int t = 0; t < gs; ++t) acc += q_sh[gg * D + blk * BLOCK + base + t];
            qg_sh[gg * NB * BLOCK + blk * BLOCK + grp] = acc;
        }
        __syncthreads();
    }

    const int j0 = s * key_tile, j1 = min(j0 + key_tile, Lk);
    float m_i[MAX_G], l_i[MAX_G], acc[MAX_G];
    #pragma unroll
    for (int g = 0; g < MAX_G; ++g) { m_i[g] = -INFINITY; l_i[g] = 0.0f; acc[g] = 0.0f; }

    for (int cs = j0; cs < j1; cs += chunk) {
        const int nC = min(chunk, j1 - cs);

        // ---- Pass 1: thread-per-key; unpack K once, feed G in-thread dots ----
        if (tid < nC) {
            const int key = cs + tid;
            float dot[MAX_G];
            #pragma unroll
            for (int g = 0; g < MAX_G; ++g) dot[g] = 0.0f;
            for (int blk = 0; blk < NB; ++blk) {
                const float ksc = ms::e8m0_to_scale(ks[(hk * NB + blk) * Lcap + key]);
                const long kbase = (long)(hk * NB + blk) * Lcap + key;
                uint8_t sb[8];
                const long sbase = kbase * SB;
                #pragma unroll
                for (int i = 0; i < 8; ++i) if (i < SB) sb[i] = kh[sbase + i];
                if constexpr (U4) {
                    const uint4 up4 = *reinterpret_cast<const uint4*>(ku + kbase * UB);
                    const uint32_t uw[4] = { up4.x, up4.y, up4.z, up4.w };
                    if (sepsc) {                        // separated-scale (low-reg: fan up to G like the combined path)
                        float bup[MAX_G];
                        #pragma unroll
                        for (int g = 0; g < MAX_G; ++g) bup[g] = 0.0f;
                        #pragma unroll
                        for (int kd = 0; kd < BLOCK; ++kd) {
                            const int up = ms::bfe_s32((int)uw[kd >> 3], (kd & 7) * 4, 4);
                            for (int g = 0; g < G; ++g) bup[g] += q_sh[g * D + blk * BLOCK + kd] * up;
                        }
                        const int ng = BLOCK >> gs_shift;
                        for (int g = 0; g < G; ++g) {
                            float bsh = 0.0f;
                            for (int grp = 0; grp < ng; ++grp)
                                bsh += ms::bfe_s32((int)sb[grp >> 1], (grp & 1) * 4, 4)
                                     * qg_sh[g * NB * BLOCK + blk * BLOCK + grp];
                            dot[g] += ksc * (16.0f * bup[g] + bsh);
                        }
                    } else {
                        #pragma unroll
                        for (int kd = 0; kd < BLOCK; ++kd) {
                            const int up_code = ms::bfe_s32((int)uw[kd >> 3], (kd & 7) * 4, 4);
                            const int gg = kd >> gs_shift;
                            const int sh_code = ms::bfe_s32((int)sb[gg >> 1], (gg & 1) * 4, 4);
                            const float w = (up_code * 16 + sh_code) * ksc;
                            const int d = blk * BLOCK + kd;
                            for (int g = 0; g < G; ++g) dot[g] += q_sh[g * D + d] * w;
                        }
                    }
                } else {
                    uint32_t ureg[6];
                    const uint32_t* src = reinterpret_cast<const uint32_t*>(ku + kbase * UB);
                    #pragma unroll
                    for (int i = 0; i < 6; ++i) if (i < (UB >> 2)) ureg[i] = src[i];
                    uint32_t sreg[3] = {0u,0u,0u};
                    #pragma unroll
                    for (int i = 0; i < 8; ++i) if (i < SB) sreg[i >> 2] |= (uint32_t)sb[i] << (8 * (i & 3));
                    const int wbits = 8 - u, gsmask = gs - 1;
                    const uint32_t umask = (1u << wbits) - 1u, usign = 1u << (wbits - 1);
                    const uint32_t smask = (1u << u) - 1u, ssign = 1u << (u - 1);
                    uint64_t ubuf = 0; int unb = 0, uwi = 0;
                    uint64_t sbuf = 0; int snb = 0, swi = 0; int sh_code = 0;
                    for (int kd = 0; kd < BLOCK; ++kd) {
                        if (unb < wbits) { ubuf |= (uint64_t)ureg[uwi++] << unb; unb += 32; }
                        const int up_code = (int)(((uint32_t)ubuf & umask) ^ usign) - (int)usign;
                        ubuf >>= wbits; unb -= wbits;
                        if ((kd & gsmask) == 0) {
                            if (snb < u) { sbuf |= (uint64_t)sreg[swi++] << snb; snb += 32; }
                            sh_code = (int)(((uint32_t)sbuf & smask) ^ ssign) - (int)ssign;
                            sbuf >>= u; snb -= u;
                        }
                        const float w = (up_code * (1 << u) + sh_code) * ksc;
                        const int d = blk * BLOCK + kd;
                        for (int g = 0; g < G; ++g) dot[g] += q_sh[g * D + d] * w;
                    }
                }
            }
            for (int g = 0; g < G; ++g) sc[g * chunk + tid] = dot[g] * sm_scale;
        }

        // ---- stage V full-sector (read 0.56x packed bytes once, reuse for G) ----
        for (int blk = 0; blk < NB; ++blk) {
            if (U4 && v8) {                             // stage reconstructed int8 codes
                const uint4* vsrc = reinterpret_cast<const uint4*>(
                        vu + ((long)(hk * NB + blk) * Lcap + cs) * UB);
                const unsigned char* vsh = vh + ((long)(hk * NB + blk) * Lcap + cs) * SB;
                int8_t* d8 = (int8_t*)pVu;
                const int CH = chunk + 4;            // vt: CH/4 odd -> conflict-free int32 reads
                for (int i = tid; i < nC; i += NT) {
                    const uint4 up4 = vsrc[i];
                    const uint32_t uw[4] = { up4.x, up4.y, up4.z, up4.w };
                    uint8_t sb2[8];
                    #pragma unroll
                    for (int t = 0; t < 8; ++t) if (t < SB) sb2[t] = vsh[i * SB + t];
                    #pragma unroll
                    for (int kd = 0; kd < BLOCK; ++kd) {
                        const int up_code = ms::bfe_s32((int)uw[kd >> 3], (kd & 7) * 4, 4);
                        const int g2 = kd >> gs_shift;
                        const int sh_code = ms::bfe_s32((int)sb2[g2 >> 1], (g2 & 1) * 4, 4);
                        const int8_t code = (int8_t)(up_code * 16 + sh_code);
                        if (vt) d8[(long)blk * BLOCK * CH + kd * CH + i] = code;   // [blk][kd][kk]
                        else    d8[(long)blk * chunk * BLOCK + i * BLOCK + kd] = code;
                    }
                }
            } else if constexpr (U4) {
                const uint4* src4 = reinterpret_cast<const uint4*>(
                        vu + ((long)(hk * NB + blk) * Lcap + cs) * UB);
                uint4* dst4 = reinterpret_cast<uint4*>(pVu + (long)blk * chunk * UB);
                for (int i = tid; i < nC; i += NT) dst4[i] = src4[i];
            } else {
                const uint32_t* s32 = reinterpret_cast<const uint32_t*>(
                        vu + ((long)(hk * NB + blk) * Lcap + cs) * UB);
                uint32_t* d32 = reinterpret_cast<uint32_t*>(pVu + (long)blk * chunk * UB);
                const int nw = (nC * UB) >> 2;
                for (int i = tid; i < nw; i += NT) d32[i] = s32[i];
            }
            if (!(U4 && v8)) {
                const unsigned char* srch = vh + ((long)(hk * NB + blk) * Lcap + cs) * SB;
                unsigned char* dsth = pVh + (long)blk * chunk * SB;
                for (int i = tid; i < nC * SB; i += NT) dsth[i] = srch[i];
            }
        }
        __syncthreads();

        // ---- Pass 2: thread-per-d; per-query softmax rescale, then unpack each V
        //   element ONCE and fan out to all G accumulators ----
        const int blk = tid / BLOCK, kd = tid % BLOCK;
        float m_new[MAX_G];
        for (int g = 0; g < G; ++g) {
            float m_chunk = -INFINITY;
            for (int kk = 0; kk < nC; ++kk) m_chunk = fmaxf(m_chunk, sc[g * chunk + kk]);
            m_new[g] = fmaxf(m_i[g], m_chunk);
            const float alpha = expf(m_i[g] - m_new[g]);
            acc[g] *= alpha; l_i[g] *= alpha; m_i[g] = m_new[g];
        }
        __syncthreads();
        // convert scores -> probabilities IN-PLACE in shared: exp computed ONCE per
        // (g,kk) (cooperative) instead of redundantly per thread-d -> kills the
        // transcendental blow-up that made the naive GQA kernel compute-bound.
        for (int idx = tid; idx < G * nC; idx += NT) {
            const int g = idx / nC, kk = idx % nC;
            sc[g * chunk + kk] = __expf(sc[g * chunk + kk] - m_new[g]);
        }
        __syncthreads();
        for (int g = 0; g < G; ++g) {                 // l denom (d-independent sum, no exp)
            float ls = 0.0f;
            for (int kk = 0; kk < nC; ++kk) ls += sc[g * chunk + kk];
            l_i[g] += ls;
        }
        // Pass-2: unpack each V element ONCE, fan out p (from shared) to G accumulators
        if (active && U4 && v8 && vt) {
            // transposed+padded staging: this thread's kk-codes are contiguous -> int32 = 4 codes
            // (conflict-free), each fanned to G accumulators.
            const int CH = chunk + 4;
            const int8_t* vbase = (const int8_t*)pVu + (long)blk * BLOCK * CH + (long)kd * CH;
            const long vscb = (long)(hk * NB + blk) * Lcap + cs;
            int kk = 0;
            for (; kk + 3 < nC; kk += 4) {
                const int32_t w4 = *reinterpret_cast<const int32_t*>(vbase + kk);
                #pragma unroll
                for (int t = 0; t < 4; ++t) {
                    const float vv = (float)((int8_t)((w4 >> (8 * t)) & 0xff))
                                   * ms::e8m0_to_scale(vs[vscb + kk + t]);
                    for (int g = 0; g < G; ++g) acc[g] += sc[g * chunk + kk + t] * vv;
                }
            }
            for (; kk < nC; ++kk) {
                const float vv = (float)vbase[kk] * ms::e8m0_to_scale(vs[vscb + kk]);
                for (int g = 0; g < G; ++g) acc[g] += sc[g * chunk + kk] * vv;
            }
        } else
        for (int kk = 0; kk < nC; ++kk) {
            if (!active) break;
            const int j = cs + kk;
            const float vsc = ms::e8m0_to_scale(vs[(hk * NB + blk) * Lcap + j]);
            float vv;
            if (U4 && v8)
                vv = (float)((const int8_t*)pVu)[(long)blk * chunk * BLOCK + kk * BLOCK + kd];
            else if constexpr (U4)
                vv = (float)ms::unpack_ms_kv_elem_u4(pVu, pVh,
                    (long)blk * chunk * UB, (long)blk * chunk * SB, kk, kd, gs, UB, SB);
            else
                vv = (float)ms::unpack_ms_kv_elem(pVu, pVh,
                    (long)blk * chunk * UB, (long)blk * chunk * SB, 0, kk, kd, u, gs, UB, SB);
            vv *= vsc;
            for (int g = 0; g < G; ++g) acc[g] += sc[g * chunk + kk] * vv;
        }
        __syncthreads();
    }
    for (int g = 0; g < G; ++g) {
        const int qh = hk * G + g;
        if (active) part_o[((long)qh * S + s) * D + tid] = acc[g];
        if (tid == 0) { part_m[qh * S + s] = m_i[g]; part_l[qh * S + s] = l_i[g]; }
    }
}

// ---- Phase 2: merge the S partials per head into the final output ----------
//   Standard flash-decoding combine: global max m_g = max_s m_s, then each
//   tile's contribution is rescaled by exp(m_s - m_g). One block per head;
//   thread e merges its own head_dim element across all S tiles.
__global__ void kv_decode_combine_kernel(
        const float* __restrict__ part_o,       // [H, S, D]
        const float* __restrict__ part_m,       // [H, S]
        const float* __restrict__ part_l,       // [H, S]
        __nv_bfloat16* __restrict__ out,        // [H, D]
        int H, int D, int S) {

    const int h = blockIdx.x;
    const int e = threadIdx.x;
    if (e >= D) return;
    const long b = blockIdx.y;                      // batched: grid.y = batch (b==0 default)
    part_o += b * (long)H * S * D; part_m += b * (long)H * S; part_l += b * (long)H * S;
    out += b * (long)H * D;

    float m_g = -INFINITY;
    for (int s = 0; s < S; ++s) m_g = fmaxf(m_g, part_m[h * S + s]);

    float l = 0.0f, acc = 0.0f;
    for (int s = 0; s < S; ++s) {
        const float w = expf(part_m[h * S + s] - m_g);   // empty/lesser tile -> ~0
        l   += part_l[h * S + s] * w;
        acc += part_o[((long)h * S + s) * D + e] * w;
    }
    out[h * D + e] = __float2bfloat16(acc / l);
}

// ---- TENSOR-CORE P·V (FlashDecoding Pass-2) — the win-track --------------------
//   P·V at (batched) decode is a GEMM:  O[M, D] = P[M, Lk] @ V[D, Lk]   (contract Lk),
//   M = batch x GQA-group rows. The scalar kernel was latency/dequant-bound (MSAQ
//   BW ~0.5x MXINT8); a bf16 WMMA does the key-reduction in the tensor-core pipe so
//   the kernel becomes memory-bound -> MSAQ reading 0.58x V bytes from DRAM wins.
//   KEY: V stays **token-major** in DRAM (per-token grouping = accurate, KIVI-aligned,
//   fair) — we read it coalesced/full-sector and UNPACK into a d-major bf16 shared
//   tile (the transpose is on-chip, no DRAM sector penalty). So accuracy AND the
//   byte-advantage are both kept. Structure mirrors wa_gemm.cu::wonly_gemm_wmma; only
//   the B-tile source differs (V kv-unpack vs weight-unpack). U4 templated; u2/u3 use
//   the general kv-unpack. Single kv-head per (grid.z); P pre-softmaxed (2-pass).
// IMPROVED: coalesced thread-per-key B-load (each thread reads a key's full record
// = full-sector 0.58x DRAM, then transposes 32 d-values into the d-major bf16 tile
// on-chip) + double-buffered software pipeline (unpack(next) overlaps MMA(current),
// the wonly_gemm_wmma_pipe lever). Aims to lift MSAQ's dequant throughput past the
// ~0.5x BW ceiling. (change.md Phase 38.)
template<bool U4>
__global__ void pv_wmma_kernel(
        const __nv_bfloat16* __restrict__ P,    // [Hkv, M, Lk]
        const int8_t* __restrict__ vs, const uint8_t* __restrict__ vu,
        const uint8_t* __restrict__ vh,         // V planes [Hkv, NBd, Lk, *] (token-major)
        float* __restrict__ partial,            // [Hkv, S, M, D] fp32 (split-K partials)
        int M, int D, int Lk, int NBd, int u, int gs, int UB, int SB, int S) {
    __shared__ __nv_bfloat16 As[2][64][WSK];     // P         [m][k] (double-buffered)
    __shared__ __nv_bfloat16 Bs[2][64][WSK];     // dequant V [d][k]
    __shared__ float tmp[4][16][16];
    const int hk = blockIdx.z / S, sp = blockIdx.z % S;   // split-K: kv head + Lk-slice
    const int m0 = blockIdx.y * 64, d0 = blockIdx.x * 64;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int wM = warp >> 1, wN = warp & 1;
    const int gs_shift = __ffs(gs) - 1;
    const int ndb = d0 >> 5;                      // first d-block of this 64-d tile (even)
    P += (long)hk * M * Lk;  partial += (long)(hk * S + sp) * M * D;
    vs += (long)hk * NBd * Lk; vu += (long)hk * NBd * Lk * UB; vh += (long)hk * NBd * Lk * SB;

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[2][2];
    #pragma unroll
    for (int i = 0; i < 2; ++i)
        #pragma unroll
        for (int j = 0; j < 2; ++j) wmma::fill_fragment(acc[i][j], 0.0f);

    const int nLk = (Lk + BLOCK - 1) / BLOCK;
    const int kt = (nLk + S - 1) / S, kc0 = sp * kt, kc1 = min(kc0 + kt, nLk);

    auto stage = [&](int kc, int buf) {
        const int k0 = kc * BLOCK;
        for (int idx = tid; idx < 64 * BLOCK; idx += 128) {       // A-tile: P[m, lk]
            const int m = idx / BLOCK, k = idx % BLOCK, lk = k0 + k;
            As[buf][m][k] = (m0 + m < M && lk < Lk) ? P[(long)(m0 + m) * Lk + lk]
                                                    : __float2bfloat16(0.0f);
        }
        // B-tile: thread = (key j, d-block dbl, dd-half). Reads key j's record ONCE
        // (coalesced over j -> full sector), unpacks 16 d's, writes d-major Bs[d][j].
        const int j = tid & 31, dbl = (tid >> 5) & 1, half = tid >> 6;
        const int db = ndb + dbl, lk = k0 + j, d_base = dbl * 32 + half * 16;
        if (lk < Lk) {
            const long vbase = (long)db * Lk + lk;
            const float vsc = ms::e8m0_to_scale(vs[vbase]);
            if constexpr (U4) {
                const uint4 up4 = *reinterpret_cast<const uint4*>(vu + vbase * UB);
                const uint32_t uw[4] = { up4.x, up4.y, up4.z, up4.w };
                uint8_t sb[8];
                #pragma unroll
                for (int i = 0; i < 8; ++i) if (i < SB) sb[i] = vh[vbase * SB + i];
                #pragma unroll
                for (int t = 0; t < 16; ++t) {
                    const int dd = d_base - dbl * 32 + t;   // dd within block = half*16 + t
                    const int up_code = ms::bfe_s32((int)uw[dd >> 3], (dd & 7) * 4, 4);
                    const int g = dd >> gs_shift;
                    const int sh_code = ms::bfe_s32((int)sb[g >> 1], (g & 1) * 4, 4);
                    Bs[buf][dbl * 32 + dd][j] = __float2bfloat16((float)(up_code * 16 + sh_code) * vsc);
                }
            } else {
                const long bu = (long)db * Lk * UB, bh = (long)db * Lk * SB;
                #pragma unroll
                for (int t = 0; t < 16; ++t) {
                    const int dd = half * 16 + t;
                    const int code = ms::unpack_ms_kv_elem(vu, vh, bu, bh, Lk, lk, dd, u, gs, UB, SB);
                    Bs[buf][dbl * 32 + dd][j] = __float2bfloat16((float)code * vsc);
                }
            }
        } else {
            #pragma unroll
            for (int t = 0; t < 16; ++t) Bs[buf][dbl * 32 + half * 16 + t][j] = __float2bfloat16(0.0f);
        }
    };

    stage(kc0, 0);
    __syncthreads();
    for (int kc = kc0; kc < kc1; ++kc) {
        const int cur = (kc - kc0) & 1, nxt = cur ^ 1;
        #pragma unroll
        for (int wk = 0; wk < BLOCK / 16; ++wk) {                 // MMA on the CURRENT tile
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
        if (kc + 1 < kc1) stage(kc + 1, nxt);                     // unpack NEXT — overlaps the MMA
        __syncthreads();
    }
    float* wt = &tmp[warp][0][0];
    #pragma unroll
    for (int i = 0; i < 2; ++i)
        #pragma unroll
        for (int j = 0; j < 2; ++j) {
            wmma::store_matrix_sync(wt, acc[i][j], 16, wmma::mem_row_major);
            __syncwarp();
            const int mb = m0 + wM*32 + i*16, db = d0 + wN*32 + j*16;
            for (int e = lane; e < 256; e += 32) {
                const int m = mb + e / 16, d = db + e % 16;
                if (m < M && d < D) partial[(long)m * D + d] = wt[e];   // fp32 partial
            }
            __syncwarp();
        }
}

// combine split-K partials: O[hk,m,d] = sum_s partial[hk,s,m,d]. grid (Hkv, M), thr=D.
__global__ void pv_wmma_combine_kernel(const float* __restrict__ partial,
        __nv_bfloat16* __restrict__ O, int Hkv, int S, int M, int D) {
    const int hk = blockIdx.x, m = blockIdx.y, d = threadIdx.x;
    if (d >= D) return;
    float acc = 0.0f;
    for (int s = 0; s < S; ++s) acc += partial[(((long)hk * S + s) * M + m) * D + d];
    O[((long)hk * M + m) * D + d] = __float2bfloat16(acc);
}

// ---- TENSOR-CORE Q·K (FlashDecoding Pass-1): scores[M,Lk] = Q[M,D] @ K[Lk,D]^T --
//   Shared-KV: one K per kv-head, M = N*G query rows. Contract over D (token-major K
//   read coalesced thread-per-key, same as P·V's V load). bf16 WMMA; output fp32
//   scores for the softmax. NBd = D/32 contraction chunks (small -> no split-K, the
//   Lk/64 N-tiles fill the grid). (change.md Phase 39.)
template<bool U4>
__global__ void qk_wmma_kernel(
        const __nv_bfloat16* __restrict__ Q,    // [Hkv, M, D]
        const int8_t* __restrict__ ks, const uint8_t* __restrict__ ku,
        const uint8_t* __restrict__ kh,         // K planes [Hkv, NBd, Lk, *]
        float* __restrict__ scores,             // [Hkv, M, Lk]
        int M, int D, int Lk, int NBd, int u, int gs, int UB, int SB, float sm_scale) {
    __shared__ __nv_bfloat16 As[64][WSK];        // Q [m][d]
    __shared__ __nv_bfloat16 Bs[64][WSK];        // dequant K [lk][d]
    __shared__ float tmp[4][16][16];
    const int hk = blockIdx.z;
    const int m0 = blockIdx.y * 64, n0 = blockIdx.x * 64;   // M-tile, Lk-tile
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int wM = warp >> 1, wN = warp & 1;
    const int gs_shift = __ffs(gs) - 1;
    Q += (long)hk * M * D;  scores += (long)hk * M * Lk;
    ks += (long)hk * NBd * Lk; ku += (long)hk * NBd * Lk * UB; kh += (long)hk * NBd * Lk * SB;

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[2][2];
    #pragma unroll
    for (int i = 0; i < 2; ++i)
        #pragma unroll
        for (int j = 0; j < 2; ++j) wmma::fill_fragment(acc[i][j], 0.0f);

    for (int db = 0; db < NBd; ++db) {              // contract over D in 32-wide chunks
        for (int idx = tid; idx < 64 * BLOCK; idx += 128) {     // A-tile: Q[m, db*32+d]
            const int m = idx / BLOCK, d = idx % BLOCK;
            As[m][d] = (m0 + m < M) ? Q[(long)(m0 + m) * D + db * BLOCK + d]
                                    : __float2bfloat16(0.0f);
        }
        // B-tile: thread-per-key coalesced. thread=(key lkl, dd-half); unpack 16 d's.
        const int lkl = tid >> 1, half = tid & 1;
        const int key = n0 + lkl;
        if (key < Lk) {
            const long kbase = (long)db * Lk + key;
            const float ksc = ms::e8m0_to_scale(ks[kbase]);
            if constexpr (U4) {
                const uint4 up4 = *reinterpret_cast<const uint4*>(ku + kbase * UB);
                const uint32_t uw[4] = { up4.x, up4.y, up4.z, up4.w };
                uint8_t sb[8];
                #pragma unroll
                for (int i = 0; i < 8; ++i) if (i < SB) sb[i] = kh[kbase * SB + i];
                #pragma unroll
                for (int t = 0; t < 16; ++t) {
                    const int dd = half * 16 + t;
                    const int up_code = ms::bfe_s32((int)uw[dd >> 3], (dd & 7) * 4, 4);
                    const int g = dd >> gs_shift;
                    const int sh_code = ms::bfe_s32((int)sb[g >> 1], (g & 1) * 4, 4);
                    Bs[lkl][dd] = __float2bfloat16((float)(up_code * 16 + sh_code) * ksc);
                }
            } else {
                const long bu = (long)db * Lk * UB, bh = (long)db * Lk * SB;
                #pragma unroll
                for (int t = 0; t < 16; ++t) {
                    const int dd = half * 16 + t;
                    const int code = ms::unpack_ms_kv_elem(ku, kh, bu, bh, Lk, key, dd, u, gs, UB, SB);
                    Bs[lkl][dd] = __float2bfloat16((float)code * ksc);
                }
            }
        } else {
            #pragma unroll
            for (int t = 0; t < 16; ++t) Bs[lkl][half * 16 + t] = __float2bfloat16(0.0f);
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
            const int mb = m0 + wM*32 + i*16, nb = n0 + wN*32 + j*16;
            for (int e = lane; e < 256; e += 32) {
                const int m = mb + e / 16, lk = nb + e % 16;
                if (m < M && lk < Lk) scores[(long)m * Lk + lk] = wt[e] * sm_scale;
            }
            __syncwarp();
        }
}

// matched MXINT8 Q·K WMMA (K read direct int8) ---------------------------------
__global__ void qk_wmma_mx_kernel(
        const __nv_bfloat16* __restrict__ Q,
        const int8_t* __restrict__ ks, const int8_t* __restrict__ kq,  // K [Hkv, NBd, Lk, 32]
        float* __restrict__ scores,
        int M, int D, int Lk, int NBd, float sm_scale) {
    __shared__ __nv_bfloat16 As[64][WSK];
    __shared__ __nv_bfloat16 Bs[64][WSK];
    __shared__ float tmp[4][16][16];
    const int hk = blockIdx.z;
    const int m0 = blockIdx.y * 64, n0 = blockIdx.x * 64;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int wM = warp >> 1, wN = warp & 1;
    Q += (long)hk * M * D;  scores += (long)hk * M * Lk;
    ks += (long)hk * NBd * Lk; kq += (long)hk * NBd * Lk * BLOCK;

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[2][2];
    #pragma unroll
    for (int i = 0; i < 2; ++i)
        #pragma unroll
        for (int j = 0; j < 2; ++j) wmma::fill_fragment(acc[i][j], 0.0f);

    for (int db = 0; db < NBd; ++db) {
        for (int idx = tid; idx < 64 * BLOCK; idx += 128) {
            const int m = idx / BLOCK, d = idx % BLOCK;
            As[m][d] = (m0 + m < M) ? Q[(long)(m0 + m) * D + db * BLOCK + d]
                                    : __float2bfloat16(0.0f);
        }
        const int lkl = tid >> 1, half = tid & 1;
        const int key = n0 + lkl;
        if (key < Lk) {
            const long kbase = (long)db * Lk + key;
            const float ksc = ms::e8m0_to_scale(ks[kbase]);
            const int8_t* rec = kq + kbase * BLOCK;
            #pragma unroll
            for (int t = 0; t < 16; ++t) {
                const int dd = half * 16 + t;
                Bs[lkl][dd] = __float2bfloat16((float)rec[dd] * ksc);
            }
        } else {
            #pragma unroll
            for (int t = 0; t < 16; ++t) Bs[lkl][half * 16 + t] = __float2bfloat16(0.0f);
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
            const int mb = m0 + wM*32 + i*16, nb = n0 + wN*32 + j*16;
            for (int e = lane; e < 256; e += 32) {
                const int m = mb + e / 16, lk = nb + e % 16;
                if (m < M && lk < Lk) scores[(long)m * Lk + lk] = wt[e] * sm_scale;
            }
            __syncwarp();
        }
}

// ---- SCALAR Q·K (no bf16 staging): thread-per-key (coalesced full-sector K read),
//   dequant K[key] per d-block into registers, accumulate M-tile dots. Avoids the
//   WMMA bf16-staging that made the tensor-core Q·K unpack-bound at small M. M is
//   tiled at MQK=32 accumulators (single tile when M<=32 -> no K re-dequant).
//   (change.md Phase 40.)
constexpr int MQK = 32;
template<bool U4>
__global__ void qk_scalar_kernel(
        const __nv_bfloat16* __restrict__ Q,    // [Hkv, M, D]
        const int8_t* __restrict__ ks, const uint8_t* __restrict__ ku,
        const uint8_t* __restrict__ kh,         // K planes [Hkv, NBd, Lk, *]
        float* __restrict__ scores,             // [Hkv, M, Lk]
        int M, int D, int Lk, int NBd, int u, int gs, int UB, int SB, float sm_scale) {
    const int hk = blockIdx.y;
    const int key = blockIdx.x * blockDim.x + threadIdx.x;
    if (key >= Lk) return;
    const int gs_shift = __ffs(gs) - 1;
    Q += (long)hk * M * D; scores += (long)hk * M * Lk;
    ks += (long)hk * NBd * Lk; ku += (long)hk * NBd * Lk * UB; kh += (long)hk * NBd * Lk * SB;

    for (int mt0 = 0; mt0 < M; mt0 += MQK) {
        const int mcnt = min(MQK, M - mt0);
        float dot[MQK];
        #pragma unroll
        for (int i = 0; i < MQK; ++i) dot[i] = 0.0f;
        for (int db = 0; db < NBd; ++db) {
            const long kbase = (long)db * Lk + key;
            const float ksc = ms::e8m0_to_scale(ks[kbase]);
            float Kval[32];
            if constexpr (U4) {
                const uint4 up4 = *reinterpret_cast<const uint4*>(ku + kbase * UB);
                const uint32_t uw[4] = { up4.x, up4.y, up4.z, up4.w };
                uint8_t sb[8];
                #pragma unroll
                for (int i = 0; i < 8; ++i) if (i < SB) sb[i] = kh[kbase * SB + i];
                #pragma unroll
                for (int dd = 0; dd < 32; ++dd) {
                    const int up_code = ms::bfe_s32((int)uw[dd >> 3], (dd & 7) * 4, 4);
                    const int g = dd >> gs_shift;
                    const int sh_code = ms::bfe_s32((int)sb[g >> 1], (g & 1) * 4, 4);
                    Kval[dd] = (float)(up_code * 16 + sh_code) * ksc;
                }
            } else {
                const long bu = (long)db * Lk * UB, bh = (long)db * Lk * SB;
                #pragma unroll
                for (int dd = 0; dd < 32; ++dd)
                    Kval[dd] = (float)ms::unpack_ms_kv_elem(ku, kh, bu, bh, Lk, key, dd, u, gs, UB, SB) * ksc;
            }
            for (int mm = 0; mm < mcnt; ++mm) {
                const __nv_bfloat16* qrow = Q + (long)(mt0 + mm) * D + db * BLOCK;
                float a = 0.0f;
                #pragma unroll
                for (int dd = 0; dd < 32; ++dd) a += __bfloat162float(qrow[dd]) * Kval[dd];
                dot[mm] += a;
            }
        }
        for (int mm = 0; mm < mcnt; ++mm)
            scores[(long)(mt0 + mm) * Lk + key] = dot[mm] * sm_scale;
    }
}

__global__ void qk_scalar_mx_kernel(
        const __nv_bfloat16* __restrict__ Q,
        const int8_t* __restrict__ ks, const int8_t* __restrict__ kq,  // K [Hkv, NBd, Lk, 32]
        float* __restrict__ scores,
        int M, int D, int Lk, int NBd, float sm_scale) {
    const int hk = blockIdx.y;
    const int key = blockIdx.x * blockDim.x + threadIdx.x;
    if (key >= Lk) return;
    Q += (long)hk * M * D; scores += (long)hk * M * Lk;
    ks += (long)hk * NBd * Lk; kq += (long)hk * NBd * Lk * BLOCK;

    for (int mt0 = 0; mt0 < M; mt0 += MQK) {
        const int mcnt = min(MQK, M - mt0);
        float dot[MQK];
        #pragma unroll
        for (int i = 0; i < MQK; ++i) dot[i] = 0.0f;
        for (int db = 0; db < NBd; ++db) {
            const long kbase = (long)db * Lk + key;
            const float ksc = ms::e8m0_to_scale(ks[kbase]);
            const int8_t* rec = kq + kbase * BLOCK;
            float Kval[32];
            #pragma unroll
            for (int dd = 0; dd < 32; ++dd) Kval[dd] = (float)rec[dd] * ksc;
            for (int mm = 0; mm < mcnt; ++mm) {
                const __nv_bfloat16* qrow = Q + (long)(mt0 + mm) * D + db * BLOCK;
                float a = 0.0f;
                #pragma unroll
                for (int dd = 0; dd < 32; ++dd) a += __bfloat162float(qrow[dd]) * Kval[dd];
                dot[mm] += a;
            }
        }
        for (int mm = 0; mm < mcnt; ++mm)
            scores[(long)(mt0 + mm) * Lk + key] = dot[mm] * sm_scale;
    }
}

// matched MXINT8 P·V WMMA (B-tile = int8 V read direct; same tensor-core path) -----
__global__ void pv_wmma_mx_kernel(
        const __nv_bfloat16* __restrict__ P,
        const int8_t* __restrict__ vs, const int8_t* __restrict__ vq,  // V [Hkv, NBd, Lk, 32]
        float* __restrict__ partial,            // [Hkv, S, M, D] fp32
        int M, int D, int Lk, int NBd, int S) {
    __shared__ __nv_bfloat16 As[2][64][WSK];
    __shared__ __nv_bfloat16 Bs[2][64][WSK];
    __shared__ float tmp[4][16][16];
    const int hk = blockIdx.z / S, sp = blockIdx.z % S;
    const int m0 = blockIdx.y * 64, d0 = blockIdx.x * 64;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int wM = warp >> 1, wN = warp & 1;
    const int ndb = d0 >> 5;
    P += (long)hk * M * Lk;  partial += (long)(hk * S + sp) * M * D;
    vs += (long)hk * NBd * Lk; vq += (long)hk * NBd * Lk * BLOCK;

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[2][2];
    #pragma unroll
    for (int i = 0; i < 2; ++i)
        #pragma unroll
        for (int j = 0; j < 2; ++j) wmma::fill_fragment(acc[i][j], 0.0f);

    const int nLk = (Lk + BLOCK - 1) / BLOCK;
    const int kt = (nLk + S - 1) / S, kc0 = sp * kt, kc1 = min(kc0 + kt, nLk);

    auto stage = [&](int kc, int buf) {
        const int k0 = kc * BLOCK;
        for (int idx = tid; idx < 64 * BLOCK; idx += 128) {
            const int m = idx / BLOCK, k = idx % BLOCK, lk = k0 + k;
            As[buf][m][k] = (m0 + m < M && lk < Lk) ? P[(long)(m0 + m) * Lk + lk]
                                                    : __float2bfloat16(0.0f);
        }
        const int j = tid & 31, dbl = (tid >> 5) & 1, half = tid >> 6;
        const int db = ndb + dbl, lk = k0 + j;
        if (lk < Lk) {
            const long vbase = (long)db * Lk + lk;
            const float vsc = ms::e8m0_to_scale(vs[vbase]);
            const int8_t* rec = vq + vbase * BLOCK;
            #pragma unroll
            for (int t = 0; t < 16; ++t) {
                const int dd = half * 16 + t;
                Bs[buf][dbl * 32 + dd][j] = __float2bfloat16((float)rec[dd] * vsc);
            }
        } else {
            #pragma unroll
            for (int t = 0; t < 16; ++t) Bs[buf][dbl * 32 + half * 16 + t][j] = __float2bfloat16(0.0f);
        }
    };

    stage(kc0, 0);
    __syncthreads();
    for (int kc = kc0; kc < kc1; ++kc) {
        const int cur = (kc - kc0) & 1, nxt = cur ^ 1;
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
        if (kc + 1 < kc1) stage(kc + 1, nxt);
        __syncthreads();
    }
    float* wt = &tmp[warp][0][0];
    #pragma unroll
    for (int i = 0; i < 2; ++i)
        #pragma unroll
        for (int j = 0; j < 2; ++j) {
            wmma::store_matrix_sync(wt, acc[i][j], 16, wmma::mem_row_major);
            __syncwarp();
            const int mb = m0 + wM*32 + i*16, db = d0 + wN*32 + j*16;
            for (int e = lane; e < 256; e += 32) {
                const int m = mb + e / 16, d = db + e % 16;
                if (m < M && d < D) partial[(long)m * D + d] = wt[e];   // fp32 partial
            }
            __syncwarp();
        }
}

inline int next_pow2(int n) { int p = 1; while (p < n) p <<= 1; return p; }

// split the P·V Lk-loop so base_blocks*S ~= mult*#SM fills the machine; cap at nLk
// (each split needs >=1 32-key chunk). MS_PV_SPLIT_MULT (env, default 4).
inline int pv_split_count(int base_blocks, int nLk) {
    static int sm = -1;
    if (sm < 0) cudaDeviceGetAttribute(&sm, cudaDevAttrMultiProcessorCount, 0);
    int mult = 4;
    if (const char* e = getenv("MS_PV_SPLIT_MULT")) { int m = atoi(e); if (m > 0) mult = m; }
    int S = (mult * sm + base_blocks - 1) / base_blocks;
    if (S > nLk) S = nLk;
    if (S < 1) S = 1;
    return S;
}

// ---- KV WRITE (prefill): pack_kv as a CUDA kernel ---------------------------
//   bf16 X [H,L,D] (post-projection+RoPE) -> token-major MSAQ planes that
//   kv_decode_* reads back unchanged. THREAD-PER-TOKEN (the write mirror of the
//   Phase-18 read): thread (h,j) loops the nb head_dim blocks; for a fixed
//   (h,blk) consecutive threads = consecutive tokens write UB-contiguous bytes
//   -> coalesced store (u4: 512 B/warp). Blocks = H*ceil(L/TPB), occupancy free
//   (no split needed). decompose_ms_block + dense LSB bit-pack (ms_utils). One
//   plane set (K or V) per call. (change.md Phase 28.)
// thread-per-token, one head_dim BLOCK per block-grid-z. Lifting nb into the grid
// (vs an in-thread blk loop) is the occupancy fix for GQA: at H=8 the old grid was
// only H*ceil(L/256)=32 blocks -> <0.5 block/SM on the 82-SM 3090, so the kernel ran
// ~10x off its (BW-light) ceiling. grid (H, ceil(L/TPB), nb) -> H*ceil(L/128)*nb
// (~224) blocks fills the machine. Consecutive threads = consecutive tokens at a
// fixed (h,blk) -> UB-contiguous coalesced store (the Phase-18 write mirror, kept).
__global__ void kv_write_kernel(
        const __nv_bfloat16* __restrict__ X,    // [H, L, D]
        int8_t*  __restrict__ scale_exp,         // [H, nb, L]
        uint8_t* __restrict__ upper,             // [H, nb, L, UB]
        uint8_t* __restrict__ shared,            // [H, nb, L, SB]
        int H, int L, int D, int NB, int u, int gs, int UB, int SB, int lightms) {
    const int h = blockIdx.x;
    const int blk = blockIdx.z;                            // one head_dim block per grid-z
    const int j = blockIdx.y * blockDim.x + threadIdx.x;
    if (j >= L) return;
    const int ng = 32 / gs, wbits = 8 - u;
    float x[32];
    const long xb = ((long)h * L + j) * D + (long)blk * 32;
    #pragma unroll
    for (int k = 0; k < 32; ++k) x[k] = __bfloat162float(X[xb + k]);
    int q_upper[32], r_shared[16];
    const int ea = lightms ? ms::decompose_lightms_block(x, u, gs, q_upper, r_shared)
                           : ms::decompose_ms_block(x, u, gs, q_upper, r_shared);
    const long tok = (long)(h * NB + blk) * L + j;
    uint32_t ureg[8] = {0u,0u,0u,0u,0u,0u,0u,0u};         // <=24 packed upper bytes
    ms::pack_codes_lsb(q_upper, 32, wbits, (uint8_t*)ureg, UB);
    uint32_t* dU = reinterpret_cast<uint32_t*>(upper + tok * UB);  // UB,tok*UB 4-aligned
    #pragma unroll
    for (int w = 0; w < 8; ++w) if (w < (UB >> 2)) dU[w] = ureg[w];
    uint8_t sbuf[8];
    ms::pack_codes_lsb(r_shared, ng, u, sbuf, SB);
    for (int bi = 0; bi < SB; ++bi) shared[tok * SB + bi] = sbuf[bi];
    scale_exp[tok] = (int8_t)ea;
}

// ---- KV QUANTIZE (decode append): the L=1, in-place specialization of write ---
//   Each decode step quantizes the new token's K (or V) [H,D] and writes it into
//   the pre-allocated cache at token slot `pos` (stride Lcap). thread = (h,blk)
//   (work = H*nb blocks, tiny) -> launch-latency dominated; fuse into the
//   projection/RoPE epilogue or attention prologue in deployment. Same
//   decompose+bit-pack primitive and token-major slot as the write kernel, so the
//   read path sees one consistent format. (change.md Phase 28.)
__global__ void kv_append_kernel(
        const __nv_bfloat16* __restrict__ X,    // [H, D] (new token's K or V)
        int8_t*  __restrict__ scale_exp,         // [H, nb, Lcap]
        uint8_t* __restrict__ upper,             // [H, nb, Lcap, UB]
        uint8_t* __restrict__ shared,            // [H, nb, Lcap, SB]
        int H, int D, int NB, int pos, int Lcap, int u, int gs, int UB, int SB, int lightms) {
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= H * NB) return;
    const int h = t / NB, blk = t % NB, ng = 32 / gs, wbits = 8 - u;
    float x[32];
    const long xb = (long)h * D + (long)blk * 32;
    #pragma unroll
    for (int k = 0; k < 32; ++k) x[k] = __bfloat162float(X[xb + k]);
    int q_upper[32], r_shared[16];
    const int ea = lightms ? ms::decompose_lightms_block(x, u, gs, q_upper, r_shared)
                           : ms::decompose_ms_block(x, u, gs, q_upper, r_shared);
    const long slot = (long)(h * NB + blk) * Lcap + pos;
    uint8_t ubuf[32], sbuf[8];
    ms::pack_codes_lsb(q_upper, 32, wbits, ubuf, UB);
    ms::pack_codes_lsb(r_shared, ng, u, sbuf, SB);
    for (int bi = 0; bi < UB; ++bi) upper[slot * UB + bi] = ubuf[bi];
    for (int bi = 0; bi < SB; ++bi) shared[slot * SB + bi] = sbuf[bi];
    scale_exp[slot] = (int8_t)ea;
}

// ---- KV QUANTIZE + FUSED ONLINE K-ROTATION (decode append) -------------------
//   The Hadamard rotation is over the FULL head_dim (H_D), not per-32-block, so
//   one block owns one head: D threads load the row, do the FWHT in shared
//   (orthonormal, 1/sqrt(D)), then NB threads each decompose+pack their 32-elem
//   block to the same token-major slot as kv_append. Because the rotation rides
//   the append's existing launch, its MARGINAL cost vs kv_append (no rotation)
//   is the true online K-rotation tax. grid=(H), block=D.
__global__ void kv_append_rot_kernel(
        const __nv_bfloat16* __restrict__ X,    // [H, D] (new token's K)
        int8_t*  __restrict__ scale_exp,         // [H, nb, Lcap]
        uint8_t* __restrict__ upper,             // [H, nb, Lcap, UB]
        uint8_t* __restrict__ shared,            // [H, nb, Lcap, SB]
        int H, int D, int NB, int pos, int Lcap, int u, int gs, int UB, int SB, int lightms) {
    extern __shared__ float sr[];                          // [D] rotated row
    const int h = blockIdx.x, e = threadIdx.x;
    sr[e] = __bfloat162float(X[(long)h * D + e]);
    __syncthreads();
    for (int hh = 1; hh < D; hh <<= 1) {                   // FWHT, 7 stages for D=128
        const float ve = sr[e], vp = sr[e ^ hh];
        const float nv = (e & hh) ? (vp - ve) : (ve + vp);
        __syncthreads();
        sr[e] = nv;
        __syncthreads();
    }
    sr[e] *= rsqrtf((float)D);                             // orthonormal H
    __syncthreads();

    if (e >= NB) return;                                   // NB threads do the quant+pack
    const int blk = e, ng = 32 / gs, wbits = 8 - u;
    float x[32];
    #pragma unroll
    for (int k = 0; k < 32; ++k) x[k] = sr[blk * 32 + k];
    int q_upper[32], r_shared[16];
    const int ea = lightms ? ms::decompose_lightms_block(x, u, gs, q_upper, r_shared)
                           : ms::decompose_ms_block(x, u, gs, q_upper, r_shared);
    const long slot = (long)(h * NB + blk) * Lcap + pos;
    uint8_t ubuf[32], sbuf[8];
    ms::pack_codes_lsb(q_upper, 32, wbits, ubuf, UB);
    ms::pack_codes_lsb(r_shared, ng, u, sbuf, SB);
    for (int bi = 0; bi < UB; ++bi) upper[slot * UB + bi] = ubuf[bi];
    for (int bi = 0; bi < SB; ++bi) shared[slot * SB + bi] = sbuf[bi];
    scale_exp[slot] = (int8_t)ea;
}

} // namespace

// Host launcher. Signature matches ms_lib.ops.kv_decode_attention / the schema.
torch::Tensor kv_decode_attention_cuda(
        torch::Tensor q,                                   // bf16 [H, D]
        torch::Tensor ks, torch::Tensor ku, torch::Tensor kh,
        torch::Tensor vs, torch::Tensor vu, torch::Tensor vh,
        int64_t H, int64_t Hkv, int64_t Lk, int64_t D, int64_t NB, int64_t u, int64_t gs,
        int64_t Lcap) {
    TORCH_CHECK(q.is_cuda(), "q must be a CUDA tensor");
    TORCH_CHECK(q.scalar_type() == torch::kBFloat16, "q must be bf16");
    if (Lcap < 0) Lcap = Lk;                       // default: cache exactly sized (stride==Lk)
    const int wbits = 8 - (int)u;
    const int UB = BLOCK * wbits / 8;
    const int SB = ((BLOCK / (int)gs) * (int)u + 7) / 8;

    auto out = torch::empty({H, D}, q.options());
    const int threads = next_pow2((int)D);             // 1 thread / head_dim elem
    const size_t smem = (size_t)((int)D + KV_CHUNK) * sizeof(float);  // q_sh[D] + sc[CHUNK]
    const float sm_scale = 1.0f / sqrtf((float)D);

    // ---- GQA-batched flash-decode (design A): default when G=H/Hkv >= 2 ----
    //   One block per kv head processes all G queries -> KV read+unpacked once,
    //   amortized G-fold. STATUS: structurally correct (gated) and the right lever,
    //   but this scalar + full-chunk-staging form is occupancy/latency-bound (~25x
    //   off the ~0.64x roofline) -> realizing the roofline needs cp.async double-
    //   buffer + small MMA tiles. OPT-IN via MS_KV_GQA=1 (default off; wide wins for
    //   now). Splits the key axis by the KV-head count (grid = Hkv x Sg).
    const int Gq = (int)(H / Hkv);
    if (const char* g0 = getenv("MS_KV_GQA"); Gq >= 2 && Gq <= 8 && (g0 && atoi(g0) == 1)) {
        const int chunk = threads;
        const int Sg = ms::kv_split_count((long)Lk, (int)Hkv);
        const int key_tile_g = (int)((Lk + Sg - 1) / Sg);
        auto fopt = q.options().dtype(torch::kFloat32);
        auto p_o = torch::empty({H, (int64_t)Sg, D}, fopt);
        auto p_m = torch::empty({H, (int64_t)Sg}, fopt);
        auto p_l = torch::empty({H, (int64_t)Sg}, fopt);
        int v8 = ((int)u == 4 && (int)gs <= 2) ? 1 : 0;
        if (const char* e = getenv("MS_KV_V8")) v8 = ((int)u == 4 && atoi(e) != 0) ? 1 : 0;
        // sepsc breaks the GQA combined-w amortization over G -> default off here (env can force).
        int sepsc = 0;
        if (const char* e = getenv("MS_KV_SEPSC")) sepsc = ((int)u == 4 && atoi(e) != 0) ? 1 : 0;
        int vt = v8 ? 1 : 0;
        if (const char* e = getenv("MS_KV_VT")) vt = (v8 && atoi(e) != 0) ? 1 : 0;
        const size_t stageBg = vt ? (size_t)NB * BLOCK * (chunk + 4)
                             : (v8 ? (size_t)NB * chunk * BLOCK : (size_t)NB * chunk * (UB + SB));
        const size_t qgB = sepsc ? (size_t)Gq * NB * BLOCK * sizeof(float) : 0;
        const size_t smem_g = ((size_t)Gq * (int)D + (size_t)Gq * chunk) * sizeof(float) + qgB + stageBg;
        auto launch = [&](auto U4tag) {
            auto kg = kv_decode_gqa_kernel<decltype(U4tag)::value>;
            if (smem_g > 48 * 1024)
                cudaFuncSetAttribute(kg, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_g);
            kg<<<dim3((int)Hkv, Sg), threads, smem_g, at::cuda::getCurrentCUDAStream()>>>(
                reinterpret_cast<const __nv_bfloat16*>(q.data_ptr<at::BFloat16>()),
                ks.data_ptr<int8_t>(), ku.data_ptr<uint8_t>(), kh.data_ptr<uint8_t>(),
                vs.data_ptr<int8_t>(), vu.data_ptr<uint8_t>(), vh.data_ptr<uint8_t>(),
                p_o.data_ptr<float>(), p_m.data_ptr<float>(), p_l.data_ptr<float>(),
                (int)H, (int)Hkv, Gq, (int)Lk, (int)Lcap, (int)D, (int)NB, (int)u, (int)gs,
                UB, SB, key_tile_g, Sg, chunk, sm_scale, v8, sepsc, vt);
        };
        if ((int)u == 4) launch(std::true_type{}); else launch(std::false_type{});
        kv_decode_combine_kernel<<<(int)H, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
            p_o.data_ptr<float>(), p_m.data_ptr<float>(), p_l.data_ptr<float>(),
            reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>()), (int)H, (int)D, Sg);
        return out;
    }

    // split the key axis -> grid (H, S) sized from the live SM count so the block
    // count (H*S) lands at ~2-4x #SM and fills the machine (occupancy 방안1).
    const int S = ms::kv_split_count((long)Lk, (int)H);
    const int key_tile = (int)((Lk + S - 1) / S);      // keys per tile
    auto fopt = q.options().dtype(torch::kFloat32);
    auto part_o = torch::empty({H, (int64_t)S, D}, fopt);
    auto part_m = torch::empty({H, (int64_t)S}, fopt);
    auto part_l = torch::empty({H, (int64_t)S}, fopt);

    const char* e = getenv("MS_KV_CPASYNC");
    const bool cpasync = !(e && atoi(e) == 0);
    int diag = 0;                   // MS_KV_DIAG: 1=mem ceiling, 2=+unpack(no exp), 0=full
    if (const char* d = getenv("MS_KV_DIAG")) diag = atoi(d);
    // key-per-thread WIDE read (fix plan A): default on for all u (the KV bench
    // default), bit-exact (same bytes — coalescing comes from thread-per-key, not
    // from int4 alignment, so u2/u3 need no repack). Disabled when diag!=0 so the
    // cp.async diagnostic path stays measurable. MS_KV_WIDE=0 falls back for A/B.
    const char* wenv = getenv("MS_KV_WIDE");
    const bool wide = (diag == 0) && !(wenv && atoi(wenv) == 0);
    // design B: warp-transpose P·V (no V staging). u4 + D==128 only (NB warps ==
    // D/32); other u/D fall through to wide. Opt-in via MS_KV_WARPT=1.
    const char* tenv = getenv("MS_KV_WARPT");
    const bool warpT = (diag == 0) && (tenv && atoi(tenv) == 1)
                       && ((int)u == 4) && ((int)D == 128) && (threads == 128);
    if (warpT) {
        const int chunk = threads;
        const size_t smem_t = (size_t)((int)D + chunk) * sizeof(float);  // q_sh + sc only
        kv_decode_warpT_kernel<true><<<dim3((int)H, S), threads, smem_t, at::cuda::getCurrentCUDAStream()>>>(
            reinterpret_cast<const __nv_bfloat16*>(q.data_ptr<at::BFloat16>()),
            ks.data_ptr<int8_t>(), ku.data_ptr<uint8_t>(), kh.data_ptr<uint8_t>(),
            vs.data_ptr<int8_t>(), vu.data_ptr<uint8_t>(), vh.data_ptr<uint8_t>(),
            part_o.data_ptr<float>(), part_m.data_ptr<float>(), part_l.data_ptr<float>(),
            (int)H, (int)Hkv, (int)Lk, (int)Lcap, (int)D, (int)NB, (int)u, (int)gs, UB, SB, key_tile, S, chunk, sm_scale);
    } else if (wide) {
        const int chunk = threads;  // pass1: thread/key, pass2: thread/head_dim
        // u4-only: stage V as reconstructed int8 (one byte/elem) -> Pass-2 == MXINT8.
        // Helps when the gs shared plane is large (gs<=2); for gs>=8 the extra staging
        // smem costs occupancy, so default off there. MS_KV_V8 forces (0/1).
        // vpack (transposed-packed nibble V staging) is DEFAULT for u4/gs<=2 -- beats v8+vt ~10%
        // (smaller smem 13 vs 16.5 KB -> higher occupancy). MS_KV_VPACK=0 falls back to v8+vt.
        int vpack = ((int)u == 4 && (int)gs <= 2) ? 1 : 0;
        if (const char* e = getenv("MS_KV_VPACK")) vpack = ((int)u == 4 && atoi(e) != 0) ? 1 : 0;
        int v8 = (!vpack && (int)u == 4 && (int)gs <= 2) ? 1 : 0;
        if (const char* e = getenv("MS_KV_V8")) v8 = (!vpack && (int)u == 4 && atoi(e) != 0) ? 1 : 0;
        // separated-scale K dot (u4): factor scales to block level, shared term per-group.
        int sepsc = ((int)u == 4) ? 1 : 0;
        if (const char* e = getenv("MS_KV_SEPSC")) sepsc = ((int)u == 4 && atoi(e) != 0) ? 1 : 0;
        // vt: transposed+padded int8 V staging -> conflict-free vectorized Pass-2 reads.
        int vt = v8 ? 1 : 0;
        if (const char* e = getenv("MS_KV_VT")) vt = (v8 && atoi(e) != 0) ? 1 : 0;
        int CHPv = (chunk + 1) >> 1; CHPv = (CHPv + 3) & ~3; if (((CHPv >> 2) & 1) == 0) CHPv += 4;
        const int ngv = BLOCK / (int)gs;
        const size_t stageB = vpack ? (size_t)NB * CHPv * (BLOCK + ngv)
                            : (vt ? (size_t)NB * BLOCK * (chunk + 4)
                            : (v8 ? (size_t)NB * chunk * BLOCK : (size_t)NB * chunk * (UB + SB)));
        const size_t smem_w = (size_t)((int)D + chunk) * sizeof(float)
                            + (size_t)NB * BLOCK * sizeof(float) + stageB;   // +qg_sh
        auto launch = [&](auto U4tag) {
          auto run = [&](auto VPtag) {
            auto k = kv_decode_wide_kernel<decltype(U4tag)::value, decltype(VPtag)::value>;
            if (smem_w > 48 * 1024)   // opt in to >48KB dynamic smem (e.g. D=256: staged V)
                cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_w);
            k<<<dim3((int)H, S), threads, smem_w, at::cuda::getCurrentCUDAStream()>>>(
                reinterpret_cast<const __nv_bfloat16*>(q.data_ptr<at::BFloat16>()),
                ks.data_ptr<int8_t>(), ku.data_ptr<uint8_t>(), kh.data_ptr<uint8_t>(),
                vs.data_ptr<int8_t>(), vu.data_ptr<uint8_t>(), vh.data_ptr<uint8_t>(),
                part_o.data_ptr<float>(), part_m.data_ptr<float>(), part_l.data_ptr<float>(),
                (int)H, (int)Hkv, (int)Lk, (int)Lcap, (int)D, (int)NB, (int)u, (int)gs, UB, SB, key_tile, S, chunk, sm_scale, v8, sepsc, vt, 0);
          };
          if (vpack) run(std::true_type{}); else run(std::false_type{});
        };
        if ((int)u == 4) launch(std::true_type{});
        else             launch(std::false_type{});
    } else if (cpasync) {           // hide K/V unpack behind cp.async prefetch
        const size_t smem_cp = (size_t)((int)D + CP_CHUNK) * sizeof(float)
                             + (size_t)2 * 2 * (NB * CP_CHUNK * (UB + SB));   // 2 buf x (K+V)(up+sh)
        if (smem_cp > 48 * 1024)
            cudaFuncSetAttribute(kv_decode_cpasync_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_cp);
        kv_decode_cpasync_kernel<<<dim3((int)H, S), threads, smem_cp, at::cuda::getCurrentCUDAStream()>>>(
            reinterpret_cast<const __nv_bfloat16*>(q.data_ptr<at::BFloat16>()),
            ks.data_ptr<int8_t>(), ku.data_ptr<uint8_t>(), kh.data_ptr<uint8_t>(),
            vs.data_ptr<int8_t>(), vu.data_ptr<uint8_t>(), vh.data_ptr<uint8_t>(),
            part_o.data_ptr<float>(), part_m.data_ptr<float>(), part_l.data_ptr<float>(),
            (int)H, (int)Hkv, (int)Lk, (int)Lcap, (int)D, (int)NB, (int)u, (int)gs, UB, SB, key_tile, S, sm_scale, diag);
    } else {
        kv_decode_split_kernel<<<dim3((int)H, S), threads, smem, at::cuda::getCurrentCUDAStream()>>>(
            reinterpret_cast<const __nv_bfloat16*>(q.data_ptr<at::BFloat16>()),
            ks.data_ptr<int8_t>(), ku.data_ptr<uint8_t>(), kh.data_ptr<uint8_t>(),
            vs.data_ptr<int8_t>(), vu.data_ptr<uint8_t>(), vh.data_ptr<uint8_t>(),
            part_o.data_ptr<float>(), part_m.data_ptr<float>(), part_l.data_ptr<float>(),
            (int)H, (int)Hkv, (int)Lk, (int)Lcap, (int)D, (int)NB, (int)u, (int)gs, UB, SB, key_tile, S, sm_scale);
    }

    kv_decode_combine_kernel<<<(int)H, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        part_o.data_ptr<float>(), part_m.data_ptr<float>(), part_l.data_ptr<float>(),
        reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>()),
        (int)H, (int)D, S);
    return out;
}

// ---- BATCHED decode (KV-only batch/seqlen sweep): grid.z = batch -------------
//   q [B,H,D]; KV planes [B,Hkv,NB,Lcap,*] contiguous; out [B,H,D]. Forces the
//   wide kernel. Batch supplies the occupancy the single-token decode lacks
//   (one-wave) → pushes KV read toward BW-bound where MSAQ's 0.58x bytes win.
torch::Tensor kv_decode_attention_batched_cuda(
        torch::Tensor q, torch::Tensor ks, torch::Tensor ku, torch::Tensor kh,
        torch::Tensor vs, torch::Tensor vu, torch::Tensor vh,
        int64_t B, int64_t H, int64_t Hkv, int64_t Lk, int64_t D, int64_t NB, int64_t u, int64_t gs,
        int64_t Lcap) {
    TORCH_CHECK(q.is_cuda() && q.scalar_type() == torch::kBFloat16, "q must be CUDA bf16");
    if (Lcap < 0) Lcap = Lk;
    const int wbits = 8 - (int)u;
    const int UB = BLOCK * wbits / 8;
    const int SB = ((BLOCK / (int)gs) * (int)u + 7) / 8;
    const int threads = next_pow2((int)D), chunk = threads;
    const float sm_scale = 1.0f / sqrtf((float)D);
    const int S = ms::kv_split_count((long)Lk, (int)(H * B));   // split for B*H*S blocks
    const int key_tile = (int)((Lk + S - 1) / S);
    auto fopt = q.options().dtype(torch::kFloat32);
    auto out = torch::empty({B, H, D}, q.options());
    auto part_o = torch::empty({B, H, (int64_t)S, D}, fopt);
    auto part_m = torch::empty({B, H, (int64_t)S}, fopt);
    auto part_l = torch::empty({B, H, (int64_t)S}, fopt);
    int vpack = ((int)u == 4 && (int)gs <= 2) ? 1 : 0;
    if (const char* e = getenv("MS_KV_VPACK")) vpack = ((int)u == 4 && atoi(e) != 0) ? 1 : 0;
    int v8 = (!vpack && (int)u == 4 && (int)gs <= 2) ? 1 : 0;
    if (const char* e = getenv("MS_KV_V8")) v8 = (!vpack && (int)u == 4 && atoi(e) != 0) ? 1 : 0;
    int sepsc = ((int)u == 4) ? 1 : 0;
    if (const char* e = getenv("MS_KV_SEPSC")) sepsc = ((int)u == 4 && atoi(e) != 0) ? 1 : 0;
    int vt = v8 ? 1 : 0;
    if (const char* e = getenv("MS_KV_VT")) vt = (v8 && atoi(e) != 0) ? 1 : 0;
    int qrot = 0;                                  // fused online Q-rotation (H_D)
    if (const char* e = getenv("MS_KV_QROT")) qrot = atoi(e) != 0 ? 1 : 0;
    int CHPv = (chunk + 1) >> 1; CHPv = (CHPv + 3) & ~3; if (((CHPv >> 2) & 1) == 0) CHPv += 4;
    const int ngv = BLOCK / (int)gs;
    const size_t stageB = vpack ? (size_t)NB * CHPv * (BLOCK + ngv)
                        : (vt ? (size_t)NB * BLOCK * (chunk + 4)
                        : (v8 ? (size_t)NB * chunk * BLOCK : (size_t)NB * chunk * (UB + SB)));
    const size_t smem_w = (size_t)((int)D + chunk) * sizeof(float)
                        + (size_t)NB * BLOCK * sizeof(float) + stageB;   // +qg_sh
    auto launch = [&](auto U4tag) {
      auto run = [&](auto VPtag) {
        auto k = kv_decode_wide_kernel<decltype(U4tag)::value, decltype(VPtag)::value>;
        if (smem_w > 48 * 1024)
            cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_w);
        k<<<dim3((int)H, S, (int)B), threads, smem_w, at::cuda::getCurrentCUDAStream()>>>(
            reinterpret_cast<const __nv_bfloat16*>(q.data_ptr<at::BFloat16>()),
            ks.data_ptr<int8_t>(), ku.data_ptr<uint8_t>(), kh.data_ptr<uint8_t>(),
            vs.data_ptr<int8_t>(), vu.data_ptr<uint8_t>(), vh.data_ptr<uint8_t>(),
            part_o.data_ptr<float>(), part_m.data_ptr<float>(), part_l.data_ptr<float>(),
            (int)H, (int)Hkv, (int)Lk, (int)Lcap, (int)D, (int)NB, (int)u, (int)gs, UB, SB, key_tile, S, chunk, sm_scale, v8, sepsc, vt, qrot);
      };
      if (vpack) run(std::true_type{}); else run(std::false_type{});
    };
    if ((int)u == 4) launch(std::true_type{}); else launch(std::false_type{});
    kv_decode_combine_kernel<<<dim3((int)H, (int)B), threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        part_o.data_ptr<float>(), part_m.data_ptr<float>(), part_l.data_ptr<float>(),
        reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>()), (int)H, (int)D, S);
    return out;
}

// ---- TENSOR-CORE P·V launcher (win-track proxy). P [Hkv,M,Lk] pre-softmaxed,
//   V token-major planes [Hkv,NBd,Lk,*] -> O [Hkv,M,D]. grid (D/64, M/64, Hkv). -----
torch::Tensor pv_wmma_cuda(
        torch::Tensor P, torch::Tensor vs, torch::Tensor vu, torch::Tensor vh,
        int64_t Hkv, int64_t M, int64_t D, int64_t Lk, int64_t NBd, int64_t u, int64_t gs) {
    TORCH_CHECK(P.is_cuda() && P.scalar_type() == torch::kBFloat16, "P must be CUDA bf16");
    const int wbits = 8 - (int)u;
    const int UB = BLOCK * wbits / 8, SB = ((BLOCK / (int)gs) * (int)u + 7) / 8;
    // split-K over Lk so grid = (D/64)*(M/64)*Hkv*S fills the SMs (decode tile is tiny).
    const int mt = ((int)M + 63) / 64, dt = ((int)D + 63) / 64;
    int S = pv_split_count(dt * mt * (int)Hkv, (int)((Lk + BLOCK - 1) / BLOCK));
    auto fopt = P.options().dtype(torch::kFloat32);
    auto partial = torch::empty({Hkv, (int64_t)S, M, D}, fopt);
    auto O = torch::empty({Hkv, M, D}, P.options());
    dim3 grid(dt, mt, (int)Hkv * S);
    auto launch = [&](auto U4tag) {
        pv_wmma_kernel<decltype(U4tag)::value><<<grid, 128, 0, at::cuda::getCurrentCUDAStream()>>>(
            reinterpret_cast<const __nv_bfloat16*>(P.data_ptr<at::BFloat16>()),
            vs.data_ptr<int8_t>(), vu.data_ptr<uint8_t>(), vh.data_ptr<uint8_t>(),
            partial.data_ptr<float>(),
            (int)M, (int)D, (int)Lk, (int)NBd, (int)u, (int)gs, UB, SB, S);
    };
    if ((int)u == 4) launch(std::true_type{}); else launch(std::false_type{});
    pv_wmma_combine_kernel<<<dim3((int)Hkv, (int)M), next_pow2((int)D), 0, at::cuda::getCurrentCUDAStream()>>>(
        partial.data_ptr<float>(), reinterpret_cast<__nv_bfloat16*>(O.data_ptr<at::BFloat16>()),
        (int)Hkv, S, (int)M, (int)D);
    return O;
}

torch::Tensor pv_wmma_mx_cuda(
        torch::Tensor P, torch::Tensor vs, torch::Tensor vq,
        int64_t Hkv, int64_t M, int64_t D, int64_t Lk, int64_t NBd) {
    TORCH_CHECK(P.is_cuda() && P.scalar_type() == torch::kBFloat16, "P must be CUDA bf16");
    const int mt = ((int)M + 63) / 64, dt = ((int)D + 63) / 64;
    int S = pv_split_count(dt * mt * (int)Hkv, (int)((Lk + BLOCK - 1) / BLOCK));
    auto fopt = P.options().dtype(torch::kFloat32);
    auto partial = torch::empty({Hkv, (int64_t)S, M, D}, fopt);
    auto O = torch::empty({Hkv, M, D}, P.options());
    dim3 grid(dt, mt, (int)Hkv * S);
    pv_wmma_mx_kernel<<<grid, 128, 0, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const __nv_bfloat16*>(P.data_ptr<at::BFloat16>()),
        vs.data_ptr<int8_t>(), vq.data_ptr<int8_t>(),
        partial.data_ptr<float>(),
        (int)M, (int)D, (int)Lk, (int)NBd, S);
    pv_wmma_combine_kernel<<<dim3((int)Hkv, (int)M), next_pow2((int)D), 0, at::cuda::getCurrentCUDAStream()>>>(
        partial.data_ptr<float>(), reinterpret_cast<__nv_bfloat16*>(O.data_ptr<at::BFloat16>()),
        (int)Hkv, S, (int)M, (int)D);
    return O;
}

// ---- TENSOR-CORE Q·K launcher: Q[Hkv,M,D], K planes -> scores[Hkv,M,Lk] fp32 ----
torch::Tensor qk_wmma_cuda(
        torch::Tensor Q, torch::Tensor ks, torch::Tensor ku, torch::Tensor kh,
        int64_t Hkv, int64_t M, int64_t D, int64_t Lk, int64_t NBd, int64_t u, int64_t gs) {
    TORCH_CHECK(Q.is_cuda() && Q.scalar_type() == torch::kBFloat16, "Q must be CUDA bf16");
    const int wbits = 8 - (int)u;
    const int UB = BLOCK * wbits / 8, SB = ((BLOCK / (int)gs) * (int)u + 7) / 8;
    const float sm = 1.0f / sqrtf((float)D);
    auto scores = torch::empty({Hkv, M, Lk}, Q.options().dtype(torch::kFloat32));
    const char* sc = getenv("MS_QK_SCALAR");
    const bool scalar = sc && atoi(sc) == 1;
    if (scalar) {                          // thread-per-key scalar Q·K (no bf16 staging)
        dim3 grid(((int)Lk + 127) / 128, (int)Hkv);
        auto launch = [&](auto U4tag) {
            qk_scalar_kernel<decltype(U4tag)::value><<<grid, 128, 0, at::cuda::getCurrentCUDAStream()>>>(
                reinterpret_cast<const __nv_bfloat16*>(Q.data_ptr<at::BFloat16>()),
                ks.data_ptr<int8_t>(), ku.data_ptr<uint8_t>(), kh.data_ptr<uint8_t>(),
                scores.data_ptr<float>(), (int)M, (int)D, (int)Lk, (int)NBd, (int)u, (int)gs, UB, SB, sm);
        };
        if ((int)u == 4) launch(std::true_type{}); else launch(std::false_type{});
        return scores;
    }
    dim3 grid(((int)Lk + 63) / 64, ((int)M + 63) / 64, (int)Hkv);
    auto launch = [&](auto U4tag) {
        qk_wmma_kernel<decltype(U4tag)::value><<<grid, 128, 0, at::cuda::getCurrentCUDAStream()>>>(
            reinterpret_cast<const __nv_bfloat16*>(Q.data_ptr<at::BFloat16>()),
            ks.data_ptr<int8_t>(), ku.data_ptr<uint8_t>(), kh.data_ptr<uint8_t>(),
            scores.data_ptr<float>(), (int)M, (int)D, (int)Lk, (int)NBd, (int)u, (int)gs, UB, SB, sm);
    };
    if ((int)u == 4) launch(std::true_type{}); else launch(std::false_type{});
    return scores;
}

torch::Tensor qk_wmma_mx_cuda(
        torch::Tensor Q, torch::Tensor ks, torch::Tensor kq,
        int64_t Hkv, int64_t M, int64_t D, int64_t Lk, int64_t NBd) {
    TORCH_CHECK(Q.is_cuda() && Q.scalar_type() == torch::kBFloat16, "Q must be CUDA bf16");
    const float sm = 1.0f / sqrtf((float)D);
    auto scores = torch::empty({Hkv, M, Lk}, Q.options().dtype(torch::kFloat32));
    const char* sc = getenv("MS_QK_SCALAR");
    if (sc && atoi(sc) == 1) {
        dim3 grid(((int)Lk + 127) / 128, (int)Hkv);
        qk_scalar_mx_kernel<<<grid, 128, 0, at::cuda::getCurrentCUDAStream()>>>(
            reinterpret_cast<const __nv_bfloat16*>(Q.data_ptr<at::BFloat16>()),
            ks.data_ptr<int8_t>(), kq.data_ptr<int8_t>(),
            scores.data_ptr<float>(), (int)M, (int)D, (int)Lk, (int)NBd, sm);
        return scores;
    }
    dim3 grid(((int)Lk + 63) / 64, ((int)M + 63) / 64, (int)Hkv);
    qk_wmma_mx_kernel<<<grid, 128, 0, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const __nv_bfloat16*>(Q.data_ptr<at::BFloat16>()),
        ks.data_ptr<int8_t>(), kq.data_ptr<int8_t>(),
        scores.data_ptr<float>(), (int)M, (int)D, (int)Lk, (int)NBd, sm);
    return scores;
}
// KV write launcher: bf16 X [H,L,D] -> (scale_exp [H,nb,L], upper [H,nb,L,UB],
// shared [H,nb,L,SB]) — the certified token-major planes. UB/SB from u,gs.
std::vector<torch::Tensor> kv_write_cuda(
        torch::Tensor X, int64_t H, int64_t L, int64_t D, int64_t NB, int64_t u, int64_t gs) {
    TORCH_CHECK(X.is_cuda() && X.scalar_type() == torch::kBFloat16, "X must be CUDA bf16");
    const int wbits = 8 - (int)u;
    const int UB = BLOCK * wbits / 8;
    const int SB = ((BLOCK / (int)gs) * (int)u + 7) / 8;
    auto i8 = X.options().dtype(torch::kInt8);
    auto u8 = X.options().dtype(torch::kUInt8);
    auto scale_exp = torch::empty({H, NB, L}, i8);
    auto upper     = torch::empty({H, NB, L, UB}, u8);
    auto shared    = torch::empty({H, NB, L, SB}, u8);
    const int TPB = 128;                                  // smaller block + grid-z=nb -> more blocks
    const char* lm = getenv("MS_LIGHTMS"); const int lightms = (lm && atoi(lm) == 1) ? 1 : 0;
    kv_write_kernel<<<dim3((int)H, ((int)L + TPB - 1) / TPB, (int)NB), TPB, 0,
                      at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const __nv_bfloat16*>(X.data_ptr<at::BFloat16>()),
        scale_exp.data_ptr<int8_t>(), upper.data_ptr<uint8_t>(), shared.data_ptr<uint8_t>(),
        (int)H, (int)L, (int)D, (int)NB, (int)u, (int)gs, UB, SB, lightms);
    return {scale_exp, upper, shared};
}

// Decode-append launcher: quantize one new token X[H,D] into the cache planes
// (scale_exp[H,nb,Lcap], upper[H,nb,Lcap,UB], shared[H,nb,Lcap,SB]) at slot `pos`,
// in place. Mutates the three cache tensors (no allocation, no return).
void kv_append_cuda(
        torch::Tensor X, torch::Tensor scale_exp, torch::Tensor upper, torch::Tensor shared,
        int64_t H, int64_t D, int64_t NB, int64_t pos, int64_t Lcap, int64_t u, int64_t gs) {
    TORCH_CHECK(X.is_cuda() && X.scalar_type() == torch::kBFloat16, "X must be CUDA bf16");
    const int wbits = 8 - (int)u;
    const int UB = BLOCK * wbits / 8;
    const int SB = ((BLOCK / (int)gs) * (int)u + 7) / 8;
    const int total = (int)(H * NB), TPB = 128;
    const char* lm = getenv("MS_LIGHTMS"); const int lightms = (lm && atoi(lm) == 1) ? 1 : 0;
    kv_append_kernel<<<(total + TPB - 1) / TPB, TPB, 0, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const __nv_bfloat16*>(X.data_ptr<at::BFloat16>()),
        scale_exp.data_ptr<int8_t>(), upper.data_ptr<uint8_t>(), shared.data_ptr<uint8_t>(),
        (int)H, (int)D, (int)NB, (int)pos, (int)Lcap, (int)u, (int)gs, UB, SB, lightms);
}

// Fused K-rotation append: rotate each head's D-row by H_D, then quantize+append.
// grid=(H), block=D threads; D*sizeof(float) dynamic smem for the row.
void kv_append_rot_cuda(
        torch::Tensor X, torch::Tensor scale_exp, torch::Tensor upper, torch::Tensor shared,
        int64_t H, int64_t D, int64_t NB, int64_t pos, int64_t Lcap, int64_t u, int64_t gs) {
    TORCH_CHECK(X.is_cuda() && X.scalar_type() == torch::kBFloat16, "X must be CUDA bf16");
    const int wbits = 8 - (int)u;
    const int UB = BLOCK * wbits / 8;
    const int SB = ((BLOCK / (int)gs) * (int)u + 7) / 8;
    const char* lm = getenv("MS_LIGHTMS"); const int lightms = (lm && atoi(lm) == 1) ? 1 : 0;
    const size_t smem = (size_t)D * sizeof(float);
    kv_append_rot_kernel<<<(int)H, (int)D, smem, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const __nv_bfloat16*>(X.data_ptr<at::BFloat16>()),
        scale_exp.data_ptr<int8_t>(), upper.data_ptr<uint8_t>(), shared.data_ptr<uint8_t>(),
        (int)H, (int)D, (int)NB, (int)pos, (int)Lcap, (int)u, (int)gs, UB, SB, lightms);
}
