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
#include "core/ms_utils.cuh"

namespace {

constexpr int BLOCK = 32;
constexpr int KV_CHUNK = 128;   // keys processed per chunk (bounds shared mem)
constexpr int CP_CHUNK = 64;    // keys per cp.async double-buffer chunk (smaller smem)

// cp.async-copy n bytes global->shared in 4-byte chunks (UB,SB multiples of 4 keep
// the offsets 4-aligned); a <4-byte tail (small shared planes) is copied directly.
__device__ __forceinline__ void cpa(unsigned char* dst, const unsigned char* src,
                                     int n, int tid, int nt) {
    for (int off = tid * 4; off < n; off += nt * 4)   // n is a multiple of 4 (UB*nC)
        __pipeline_memcpy_async(dst + off, src + off, 4);
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
        int H, int Lk, int D, int NB, int u, int gs, int UB, int SB,
        int key_tile, int S, float sm_scale) {

    const int h = blockIdx.x;
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
                const long base_u = (long)(h * NB + blk) * UB * Lk;
                const long base_h = (long)(h * NB + blk) * SB * Lk;
                const float ksc = ms::e8m0_to_scale(ks[(h * NB + blk) * Lk + j]);
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
        const long base_u = (long)(h * NB + blk) * UB * Lk;
        const long base_h = (long)(h * NB + blk) * SB * Lk;
        float lsum = 0.0f, a = 0.0f;
        for (int kk = 0; kk < nC; ++kk) {
            const float p = expf(sc[kk] - m_new);
            lsum += p;                            // identical across threads -> l_i
            if (active) {
                const int j = cs + kk;
                const float vsc = ms::e8m0_to_scale(vs[(h * NB + blk) * Lk + j]);
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
        int H, int Lk, int D, int NB, int u, int gs, int UB, int SB,
        int key_tile, int S, float sm_scale) {
    const int h = blockIdx.x, s = blockIdx.y, tid = threadIdx.x;
    const int lane = tid & 31, warpId = tid >> 5, nWarps = blockDim.x >> 5, NT = blockDim.x;
    const bool active = tid < D;

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
            const long bu = (long)(h * NB + blk) * Lk * UB + (long)cs * UB;
            const long bh = (long)(h * NB + blk) * Lk * SB + (long)cs * SB;
            cpa(pKu + blk*CP_CHUNK*UB, ku + bu, nC*UB, tid, NT);       // big: async
            cpa(pVu + blk*CP_CHUNK*UB, vu + bu, nC*UB, tid, NT);
            sync_copy(pKh + blk*CP_CHUNK*SB, kh + bh, nC*SB, tid, NT); // tiny: sync
            sync_copy(pVh + blk*CP_CHUNK*SB, vh + bh, nC*SB, tid, NT);
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
                const float ksc = ms::e8m0_to_scale(ks[(h * NB + blk) * Lk + j]);
                part += q_sh[d] * (float)ms::unpack_ms_kv_elem(pKu, pKh,
                            (long)blk*CP_CHUNK*UB, (long)blk*CP_CHUNK*SB, 0, kk, kd,
                            u, gs, UB, SB) * ksc;
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
            const float p = expf(sc[kk] - m_new);
            lsum += p;
            if (active) {
                const int j = cs + kk;
                const float vsc = ms::e8m0_to_scale(vs[(h * NB + blk) * Lk + j]);
                a += p * (float)ms::unpack_ms_kv_elem(pVu, pVh,
                            (long)blk*CP_CHUNK*UB, (long)blk*CP_CHUNK*SB, 0, kk, kd,
                            u, gs, UB, SB) * vsc;
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

inline int next_pow2(int n) { int p = 1; while (p < n) p <<= 1; return p; }

} // namespace

// Host launcher. Signature matches ms_lib.ops.kv_decode_attention / the schema.
torch::Tensor kv_decode_attention_cuda(
        torch::Tensor q,                                   // bf16 [H, D]
        torch::Tensor ks, torch::Tensor ku, torch::Tensor kh,
        torch::Tensor vs, torch::Tensor vu, torch::Tensor vh,
        int64_t H, int64_t Lk, int64_t D, int64_t NB, int64_t u, int64_t gs) {
    TORCH_CHECK(q.is_cuda(), "q must be a CUDA tensor");
    TORCH_CHECK(q.scalar_type() == torch::kBFloat16, "q must be bf16");
    const int wbits = 8 - (int)u;
    const int UB = BLOCK * wbits / 8;
    const int SB = ((BLOCK / (int)gs) * (int)u + 7) / 8;

    auto out = torch::empty({H, D}, q.options());
    const int threads = next_pow2((int)D);             // 1 thread / head_dim elem
    const size_t smem = (size_t)((int)D + KV_CHUNK) * sizeof(float);  // q_sh[D] + sc[CHUNK]
    const float sm_scale = 1.0f / sqrtf((float)D);

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
    if (cpasync) {                  // hide K/V unpack behind cp.async prefetch
        const size_t smem_cp = (size_t)((int)D + CP_CHUNK) * sizeof(float)
                             + (size_t)2 * 2 * (NB * CP_CHUNK * (UB + SB));   // 2 buf x (K+V)(up+sh)
        kv_decode_cpasync_kernel<<<dim3((int)H, S), threads, smem_cp>>>(
            reinterpret_cast<const __nv_bfloat16*>(q.data_ptr<at::BFloat16>()),
            ks.data_ptr<int8_t>(), ku.data_ptr<uint8_t>(), kh.data_ptr<uint8_t>(),
            vs.data_ptr<int8_t>(), vu.data_ptr<uint8_t>(), vh.data_ptr<uint8_t>(),
            part_o.data_ptr<float>(), part_m.data_ptr<float>(), part_l.data_ptr<float>(),
            (int)H, (int)Lk, (int)D, (int)NB, (int)u, (int)gs, UB, SB, key_tile, S, sm_scale);
    } else {
        kv_decode_split_kernel<<<dim3((int)H, S), threads, smem>>>(
            reinterpret_cast<const __nv_bfloat16*>(q.data_ptr<at::BFloat16>()),
            ks.data_ptr<int8_t>(), ku.data_ptr<uint8_t>(), kh.data_ptr<uint8_t>(),
            vs.data_ptr<int8_t>(), vu.data_ptr<uint8_t>(), vh.data_ptr<uint8_t>(),
            part_o.data_ptr<float>(), part_m.data_ptr<float>(), part_l.data_ptr<float>(),
            (int)H, (int)Lk, (int)D, (int)NB, (int)u, (int)gs, UB, SB, key_tile, S, sm_scale);
    }

    kv_decode_combine_kernel<<<(int)H, threads>>>(
        part_o.data_ptr<float>(), part_m.data_ptr<float>(), part_l.data_ptr<float>(),
        reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>()),
        (int)H, (int)D, S);
    return out;
}