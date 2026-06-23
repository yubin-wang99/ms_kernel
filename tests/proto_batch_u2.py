"""Batched u2 GEMV: x[B,K] -> y[B,OUT]. Unpack each weight ONCE per block and reuse it
across all B tokens -> amortize weight-DRAM read + unpack-ALU over B (arithmetic
intensity rises with B). Sweeps B and compares to a dense bf16 matmul baseline.

Hypothesis (packing_explained §9): B=1 decode is SM/latency-bound below the BW roofline;
as B grows the fixed 13.6 MB weight read amortizes -> time/token falls until the kernel
becomes compute-bound (rides the roofline). bf16 matmul reads 32 MB weights (no quant)
so MSAQ should keep a per-token edge to larger B.

Run: python tests/proto_batch_u2.py
"""
import os, sys; sys.path.insert(0, ".")
import numpy as np, torch
from torch.utils.cpp_extension import load_inline
from ms_lib.pack import pack_weight, decompose, dequant_weight, BLOCK

U, GS = 2, 8
OUT, K = 4096, 4096
NB = K // BLOCK
N_GROUP = BLOCK // GS
SB = (N_GROUP * U + 7) // 8
SPLITK = int(os.environ.get("SPLITK", "8"))
BS = [1, 2, 4, 8, 16, 32]

CUDA = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#define BLK 32
#define OUTW 256

// thread o owns output column o; accumulates B partial dots. Weight bytes loaded +
// unpacked ONCE per block, reused across all B tokens (acc[B] in registers).
template<int B>
__global__ void gemv_u2_batch(
    const __nv_bfloat16* __restrict__ x,          // [B, K]
    const signed char* __restrict__ scale_exp,
    const unsigned char* __restrict__ a16, const unsigned char* __restrict__ b8,
    const unsigned char* __restrict__ shared_cm,
    float* __restrict__ partial,                  // [splitK, B, OUT]
    int OUT, int NB, int SB, int splitK, int K) {
    const int o  = blockIdx.x * OUTW + threadIdx.x;
    const int sp = blockIdx.y;
    if (o >= OUT) return;
    const int per = (NB + splitK - 1) / splitK;
    const int b0 = sp * per, b1 = min(b0 + per, NB);
    float acc[B];
    #pragma unroll
    for (int b = 0; b < B; ++b) acc[b] = 0.f;
    for (int blk = b0; blk < b1; ++blk) {
        const long base = (long)blk * OUT + o;
        const float scale = exp2f((float)scale_exp[base]);
        const unsigned char sb = shared_cm[base * SB];
        const uint4 a = *reinterpret_cast<const uint4*>(a16 + base * 16);
        const uint2 bb = *reinterpret_cast<const uint2*>(b8 + base * 8);
        unsigned int ureg[6] = { a.x, a.y, a.z, a.w, bb.x, bb.y };
        unsigned long long ubuf = 0ull; int unb = 0, uwi = 0;
        const int kbase = blk * BLK;
        #pragma unroll
        for (int k = 0; k < BLK; ++k) {
            if (unb < 6) { ubuf |= (unsigned long long)ureg[uwi++] << unb; unb += 32; }
            const unsigned int c6 = (unsigned int)(ubuf & 63u); ubuf >>= 6; unb -= 6;
            const int q = (int)(c6 ^ 32u) - 32;
            const int g = k >> 3;
            const int sh = (int)(((sb >> (g * 2)) & 3u) ^ 2u) - 2;
            const float wval = (float)(q * 4 + sh) * scale;   // unpack ONCE, reuse x B
            #pragma unroll
            for (int b = 0; b < B; ++b)
                acc[b] += wval * __bfloat162float(x[(long)b * K + kbase + k]);
        }
    }
    #pragma unroll
    for (int b = 0; b < B; ++b) partial[((long)sp * B + b) * OUT + o] = acc[b];
}

template<int B>
static torch::Tensor launch(torch::Tensor x, torch::Tensor se, torch::Tensor a16,
                            torch::Tensor b8, torch::Tensor sh, int OUT, int NB, int SB,
                            int splitK, int K) {
    auto partial = torch::zeros({splitK, B, OUT}, torch::dtype(torch::kFloat32).device(x.device()));
    dim3 grid((OUT + OUTW - 1) / OUTW, splitK);
    gemv_u2_batch<B><<<grid, OUTW>>>(
        (const __nv_bfloat16*)x.data_ptr(), (const signed char*)se.data_ptr(),
        (const unsigned char*)a16.data_ptr(), (const unsigned char*)b8.data_ptr(),
        (const unsigned char*)sh.data_ptr(), partial.data_ptr<float>(), OUT, NB, SB, splitK, K);
    return partial.sum(0);  // [B, OUT]
}

torch::Tensor run_batch(torch::Tensor x, torch::Tensor se, torch::Tensor a16, torch::Tensor b8,
                        torch::Tensor sh, int OUT, int NB, int SB, int splitK, int K, int B) {
    switch (B) {
        case 1:  return launch<1>(x, se, a16, b8, sh, OUT, NB, SB, splitK, K);
        case 2:  return launch<2>(x, se, a16, b8, sh, OUT, NB, SB, splitK, K);
        case 4:  return launch<4>(x, se, a16, b8, sh, OUT, NB, SB, splitK, K);
        case 8:  return launch<8>(x, se, a16, b8, sh, OUT, NB, SB, splitK, K);
        case 16: return launch<16>(x, se, a16, b8, sh, OUT, NB, SB, splitK, K);
        case 32: return launch<32>(x, se, a16, b8, sh, OUT, NB, SB, splitK, K);
        default: TORCH_CHECK(false, "unsupported B");
    }
}
"""
CPP = "#include <torch/extension.h>\ntorch::Tensor run_batch(torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,int,int,int,int,int,int);\n"
m = load_inline(name="proto_batch", cpp_sources=CPP, cuda_sources=CUDA, functions=["run_batch"], verbose=False)

# ---- pack (bytesplit planes) ----
rng = np.random.default_rng(0)
W = (rng.standard_normal((OUT, K)) * rng.uniform(0.2, 4.0, (OUT, 1))).astype(np.float32)
p = pack_weight(W, U, GS)
A16 = np.ascontiguousarray(p["upper_cm"][:, :, :16])
B8  = np.ascontiguousarray(p["upper_cm"][:, :, 16:24])
dev = "cuda"
se  = torch.from_numpy(p["scale_exp"]).to(dev)
a16 = torch.from_numpy(A16).to(dev)
b8  = torch.from_numpy(B8).to(dev)
shc = torch.from_numpy(p["shared_cm"]).to(dev)
Wdq = torch.from_numpy(dequant_weight(p)).to(torch.bfloat16).to(dev)   # [OUT,K] for bf16 baseline
Wt  = Wdq.t().contiguous()                                             # [K,OUT]

# production planes (dense, for wonly_gemm / wa_gemm — tensor-core int-dot path), hoisted
from ms_lib import ops as _ops
up_d = torch.from_numpy(p["upper"]).to(dev)
sh_d = torch.from_numpy(p["shared"]).to(dev)
OUTn, Kn, nbn = int(p["OUT"]), int(p["K"]), int(p["nb"])
prod_wa = lambda X: torch.ops.msaq.wa_gemm(X, se, up_d, sh_d, X.shape[0], OUTn, Kn, nbn, U, GS)

def bench(fn, iters=200):
    for _ in range(30): fn()
    torch.cuda.synchronize()
    e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
    e0.record()
    for _ in range(iters): fn()
    e1.record(); torch.cuda.synchronize()
    return e0.elapsed_time(e1) / iters * 1e3  # us

W_BYTES = OUT * K * (24 + SB + 1) / 32          # MSAQ weight footprint
BF16_BYTES = OUT * K * 2
PEAK = 936e9
print(f"weight: MSAQ u2 = {W_BYTES/1e6:.1f} MB,  bf16 = {BF16_BYTES/1e6:.1f} MB,  roofline(MSAQ) = {W_BYTES/PEAK*1e6:.1f} us")
print(f"{'B':>3} | {'naive us':>8} {'us/tok':>7} | {'WA-IMMA us':>10} {'us/tok':>7} | {'bf16 us':>8} {'us/tok':>7}")
print("-" * 72)
for B in BS:
    x = torch.randn(B, K, device=dev, dtype=torch.bfloat16)
    if B == 1:
        y = m.run_batch(x, se, a16, b8, shc, OUT, NB, SB, SPLITK, K, B)
        ref = (x.float() @ Wt.float())
        assert (y - ref).abs().max() < 1e-1, (y - ref).abs().max().item()
    t_ms = bench(lambda: m.run_batch(x, se, a16, b8, shc, OUT, NB, SB, SPLITK, K, B))
    t_wa = bench(lambda: prod_wa(x))
    t_bf = bench(lambda: torch.matmul(x, Wt))
    print(f"{B:>3} | {t_ms:8.2f} {t_ms/B:7.2f} | {t_wa:10.2f} {t_wa/B:7.2f} | {t_bf:8.2f} {t_bf/B:7.2f}")
