// csrc/rotate.cu  —  [pure CUDA]  online Hadamard K/Q-rotation (decode hot path)
//
// The accuracy study (precision/rot_results.md, commit 7a37da2) established that
// a full head_dim Hadamard rotation of the KV-KEY is the structural win for
// MSAQ: it kills the persistent channel outliers (QuaRot mechanism) and makes
// the fast nibble u4 config robust. To realize that win at inference we must
// rotate K ONLINE — every decode step the new token's K (Hkv rows of head_dim)
// must be rotated post-RoPE before it is quantized + appended, and Q must be
// rotated to match so (Q·H)(K·H)^T = Q·H·H^T·K^T = Q·K^T is preserved.
//
// COST MODEL: this is added to the LATENCY-BOUND decode hot path, so it is a
// tax to be weighed against the accuracy/aggressiveness gain (no-fake-win).
// We implement the rotation as a FAST WALSH-HADAMARD TRANSFORM (FWHT): a
// D=128 rotation is log2(128)=7 butterfly stages = 128*7 fused-add ops per row,
// vs 128*128 for a naive H-matmul. The 1/sqrt(D) normalization is folded into
// the final stage so H is ORTHONORMAL (H·H^T = I) -> no attention-score rescale.
//
// One block handles ROWS_PER_BLOCK rows; 128 threads per row do the butterfly in
// shared memory (7 __syncthreads). bf16 in / fp32 compute / bf16 out. The decode
// step rotates Q [Hq,128] and K [Hkv,128] (two launches) or any [...,128] tensor.
//
// STANDALONE here = an UPPER BOUND on the added latency (separate kernel launch).
// In production the Q-rotation folds into the QK^T epilogue and the K-rotation
// into kv_append, hiding the launch; this kernel measures the un-fused tax.

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <math.h>
#include <ATen/cuda/CUDAContext.h>

namespace {

constexpr int D = 128;                 // head_dim (Llama-3.1) — FWHT size, pow2
constexpr int ROWS_PER_BLOCK = 8;      // rows handled per block (occupancy lever)

// In-place FWHT over the last dim (D=128). grid = ceil(N/ROWS_PER_BLOCK),
// block = (D, ROWS_PER_BLOCK). Each row r is owned by threadIdx.y; the D threads
// in threadIdx.x cooperatively butterfly that row through shared memory.
__global__ void hadamard_rotate_kernel(
        const __nv_bfloat16* __restrict__ x,   // [N, D]
        __nv_bfloat16* __restrict__ out,        // [N, D]
        int N) {
    __shared__ float s[ROWS_PER_BLOCK][D];

    const int row = blockIdx.x * ROWS_PER_BLOCK + threadIdx.y;
    const int e   = threadIdx.x;                 // 0..D-1, this thread's element
    float* sr = s[threadIdx.y];

    if (row < N) sr[e] = __bfloat162float(x[row * D + e]);
    __syncthreads();

    // 7 butterfly stages: pair (e, e^h); low element (e&h)==0 gets sum, high gets
    // diff (= low - high). syncthreads brackets each stage's read/write.
    #pragma unroll
    for (int h = 1; h < D; h <<= 1) {
        const int partner = e ^ h;
        const float ve = sr[e];
        const float vp = sr[partner];
        const float nv = (e & h) ? (vp - ve) : (ve + vp);
        __syncthreads();
        sr[e] = nv;
        __syncthreads();
    }

    // orthonormal Hadamard: divide by sqrt(D) so H·H^T = I (no score rescale).
    if (row < N) out[row * D + e] = __float2bfloat16(sr[e] * rsqrtf((float)D));
}

} // namespace

// out-of-place (returns a new tensor). x: [..., 128] bf16, contiguous.
torch::Tensor hadamard_rotate_cuda(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda() && x.scalar_type() == torch::kBFloat16, "x must be cuda bf16");
    TORCH_CHECK(x.size(-1) == D, "last dim must be 128 (head_dim)");
    auto xc = x.contiguous();
    auto out = torch::empty_like(xc);
    const int N = xc.numel() / D;
    dim3 block(D, ROWS_PER_BLOCK);
    dim3 grid((N + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK);
    auto stream = at::cuda::getCurrentCUDAStream();
    hadamard_rotate_kernel<<<grid, block, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(xc.data_ptr<at::BFloat16>()),
        reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>()), N);
    return out;
}
