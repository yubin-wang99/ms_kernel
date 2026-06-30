// C2 LUT-fold cost microbench for the FP4 + vector-VQ KV residual.
//
// Question (vq_kv_global_results.md): KV decode is a byte-roofline GEMV; does folding the per-element
// VQ residual LUT into the dequant add measurable latency, or hide under the DRAM read?
//
// Replicates the kv_attention.cu kdot pattern (one thread per key computes q.K over D=NB*32; q staged
// in shared; quantized K streamed + dequantized on the fly + MAC). COALESCED key-minor (plane) layout:
// field f of key k at f*NKEYS+k, so consecutive threads read consecutive bytes (like the real kernel's
// token-major KV). Three kernels, identical geometry, differing only in per-element decode + bytes:
//
//   kdot_fp4   4.25 b/elem : FP4 base only          (64B codes + 4B E8M0 / key)
//   kdot_fp4vq 5.25 b/elem : FP4 base + VQ residual (+16B group-index/key; gather codeword from smem, add)
//   kdot_iso   5.25 b/elem : same bytes as fp4vq, NO gather (reads index bytes, arithmetic-only)
//
//   fp4 vs fp4vq -> total cost of the residual (extra index bytes + LUT gather).
//   fp4vq vs iso -> isolates the LUT-gather ALU at iso-bytes == the C2 "is the correction free?" test.

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdint>

#define BLOCK 32
#define NB 4                 // D = NB*BLOCK = 128 (head_dim)
#define D (NB*BLOCK)
#define G 8                  // VQ group size
#define GPB (BLOCK/G)        // groups per 32-block = 4
#define NG (D/G)             // group-indices per key = 16
#define NBYTE (D/2)          // FP4 code bytes per key = 64
#define KCB 256              // codebook entries
#define NKEYS (4*1024*1024)  // 4M keys -> 256M elems
#define THREADS 256
#define ITERS 60

__constant__ float FP4MAG[8] = {0.f, 0.5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f};

// codes plane: byte field f (0..63) of key k at codes[(long)f*NKEYS + k]   (coalesced)
// sexp plane:  block scale blk (0..3) at sexp[(long)blk*NKEYS + k]
// idx plane:   group g (0..15)        at idx[(long)g*NKEYS + k]
__global__ void kdot_fp4(const float* __restrict__ q, const uint8_t* __restrict__ codes,
                         const int8_t* __restrict__ sexp, float* __restrict__ out) {
    __shared__ float qsh[D];
    for (int i = threadIdx.x; i < D; i += blockDim.x) qsh[i] = q[i];
    __syncthreads();
    const long key = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (key >= NKEYS) return;
    float dot = 0.f;
    #pragma unroll
    for (int blk = 0; blk < NB; ++blk) {
        float sc = exp2f((float)sexp[(long)blk * NKEYS + key]);
        #pragma unroll
        for (int p = 0; p < BLOCK / 2; ++p) {
            uint8_t byte = codes[(long)(blk * 16 + p) * NKEYS + key];
            int c0 = byte & 0xF, c1 = byte >> 4;
            float w0 = (c0 & 8 ? -1.f : 1.f) * FP4MAG[c0 & 7];
            float w1 = (c1 & 8 ? -1.f : 1.f) * FP4MAG[c1 & 7];
            dot += qsh[blk * BLOCK + 2 * p]     * (w0 * sc);
            dot += qsh[blk * BLOCK + 2 * p + 1] * (w1 * sc);
        }
    }
    out[key] = dot;
}

__global__ void kdot_fp4vq(const float* __restrict__ q, const uint8_t* __restrict__ codes,
                           const int8_t* __restrict__ sexp, const uint8_t* __restrict__ idx,
                           const __half* __restrict__ cbk, float* __restrict__ out) {
    __shared__ float qsh[D];
    __shared__ __half cbsh[KCB * G];                     // 4KB codebook in smem
    for (int i = threadIdx.x; i < D; i += blockDim.x) qsh[i] = q[i];
    for (int i = threadIdx.x; i < KCB * G; i += blockDim.x) cbsh[i] = cbk[i];
    __syncthreads();
    const long key = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (key >= NKEYS) return;
    float dot = 0.f;
    #pragma unroll
    for (int blk = 0; blk < NB; ++blk) {
        float sc = exp2f((float)sexp[(long)blk * NKEYS + key]);
        #pragma unroll
        for (int p = 0; p < BLOCK / 2; ++p) {
            uint8_t byte = codes[(long)(blk * 16 + p) * NKEYS + key];
            int e0 = 2 * p, e1 = 2 * p + 1;
            int c0 = byte & 0xF, c1 = byte >> 4;
            float w0 = (c0 & 8 ? -1.f : 1.f) * FP4MAG[c0 & 7];
            float w1 = (c1 & 8 ? -1.f : 1.f) * FP4MAG[c1 & 7];
            int g0 = blk * GPB + e0 / G, g1 = blk * GPB + e1 / G;
            uint8_t i0 = idx[(long)g0 * NKEYS + key], i1 = idx[(long)g1 * NKEYS + key];
            float r0 = __half2float(cbsh[i0 * G + (e0 % G)]);
            float r1 = __half2float(cbsh[i1 * G + (e1 % G)]);
            dot += qsh[blk * BLOCK + e0] * ((w0 + r0) * sc);
            dot += qsh[blk * BLOCK + e1] * ((w1 + r1) * sc);
        }
    }
    out[key] = dot;
}

__global__ void kdot_iso(const float* __restrict__ q, const uint8_t* __restrict__ codes,
                         const int8_t* __restrict__ sexp, const uint8_t* __restrict__ idx,
                         float* __restrict__ out) {
    __shared__ float qsh[D];
    for (int i = threadIdx.x; i < D; i += blockDim.x) qsh[i] = q[i];
    __syncthreads();
    const long key = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (key >= NKEYS) return;
    float dot = 0.f;
    #pragma unroll
    for (int blk = 0; blk < NB; ++blk) {
        float sc = exp2f((float)sexp[(long)blk * NKEYS + key]);
        #pragma unroll
        for (int p = 0; p < BLOCK / 2; ++p) {
            uint8_t byte = codes[(long)(blk * 16 + p) * NKEYS + key];
            int e0 = 2 * p, e1 = 2 * p + 1;
            int c0 = byte & 0xF, c1 = byte >> 4;
            float w0 = (c0 & 8 ? -1.f : 1.f) * FP4MAG[c0 & 7];
            float w1 = (c1 & 8 ? -1.f : 1.f) * FP4MAG[c1 & 7];
            float r0 = (float)idx[(long)(blk * GPB + e0 / G) * NKEYS + key] * (1.f / 256.f);
            float r1 = (float)idx[(long)(blk * GPB + e1 / G) * NKEYS + key] * (1.f / 256.f);
            dot += qsh[blk * BLOCK + e0] * ((w0 + r0) * sc);
            dot += qsh[blk * BLOCK + e1] * ((w1 + r1) * sc);
        }
    }
    out[key] = dot;
}

static void* dalloc(size_t n) { void* p; cudaMalloc(&p, n); cudaMemset(p, 1, n); return p; }

int main() {
    float* q       = (float*)dalloc(D * sizeof(float));
    uint8_t* codes = (uint8_t*)dalloc((long)NKEYS * NBYTE);
    int8_t* sexp   = (int8_t*)dalloc((long)NKEYS * NB);
    uint8_t* idx   = (uint8_t*)dalloc((long)NKEYS * NG);
    __half* cbk    = (__half*)dalloc((long)KCB * G * sizeof(__half));
    float* out     = (float*)dalloc((long)NKEYS * sizeof(float));
    int grid = (NKEYS + THREADS - 1) / THREADS;

    auto bench = [&](const char* nm, double bpk, auto launch) {
        for (int i = 0; i < 5; ++i) launch();
        cudaDeviceSynchronize();
        cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
        cudaEventRecord(a);
        for (int i = 0; i < ITERS; ++i) launch();
        cudaEventRecord(b); cudaEventSynchronize(b);
        float ms; cudaEventElapsedTime(&ms, a, b); ms /= ITERS;
        double gb = (double)NKEYS * bpk / 1e9;
        printf("  %-8s %.3f b/elem | %.4f ms | %.1f GB/s\n", nm, bpk * 8.0 / D, ms, gb / (ms / 1e3));
    };
    printf("C2 microbench (coalesced): %d keys, D=%d, codebook %dx%d in smem; peak ~555 GB/s\n",
           NKEYS, D, KCB, G);
    bench("fp4",   (double)(NBYTE + NB),       [&]{ kdot_fp4<<<grid, THREADS>>>(q, codes, sexp, out); });
    bench("fp4vq", (double)(NBYTE + NB + NG),  [&]{ kdot_fp4vq<<<grid, THREADS>>>(q, codes, sexp, idx, cbk, out); });
    bench("iso",   (double)(NBYTE + NB + NG),  [&]{ kdot_iso<<<grid, THREADS>>>(q, codes, sexp, idx, out); });
    cudaError_t e = cudaGetLastError();
    if (e) printf("CUDA error: %s\n", cudaGetErrorString(e));
    return 0;
}
