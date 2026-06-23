"""Prototype: 4-plane u2 repack vs 3-plane streaming u2 — isolate the unpack cost.

Hypothesis (from packing_explained §7 profiling): u2 GEMV decode is SM/issue-bound,
not DRAM-bound, so removing the 6-bit byte straddle (streaming shift/mask/OR) by
splitting the unshared upper code into a 4-bit nibble plane + a 2-bit plane should
cut unpack instructions and lower SM% / time.

Two JIT kernels share an IDENTICAL launch config (split-K GEMV, one thread per output
column); the ONLY difference is how the (8-u)=6-bit upper code is unpacked:
  * gemv_u2_streaming : 3-plane, upper_cm[24B] loaded as 6 uint32, rolling 6-bit
                        straddle extract (mirrors csrc/w_gemv.cu u2 path).
  * gemv_u2_split4    : 4-plane, nibble[16B] uint4 + low2[8B] uint2, both straddle-free
                        shift+mask, recombine to a 6-bit signed q_upper.
Bytes touched per block: streaming 24+1+1, split4 16+8+1+1 — SAME total (so any delta
is unpack, not memory traffic).

Run:        python tests/proto_split4_u2.py            # correctness + timing
ncu (SM%):  NCU=1 KERN=streaming|split4 python tests/proto_split4_u2.py   # one measured launch
"""
import os, sys; sys.path.insert(0, ".")
import numpy as np, torch
from torch.utils.cpp_extension import load_inline
from ms_lib.pack import pack_weight, decompose, dequant_weight, BLOCK

U, GS = 2, 8
OUT, K = 4096, 4096
NB = K // BLOCK
N_GROUP = BLOCK // GS          # 4
SB = (N_GROUP * U + 7) // 8    # 1
SPLITK = int(os.environ.get("SPLITK", "16"))

CUDA = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

#define BLK 32
#define OUTW 256

__global__ void gemv_u2_streaming(
    const __nv_bfloat16* __restrict__ x, const signed char* __restrict__ scale_exp,
    const unsigned char* __restrict__ upper_cm, const unsigned char* __restrict__ shared_cm,
    float* __restrict__ partial, int OUT, int NB, int SB, int splitK) {
    const int o  = blockIdx.x * OUTW + threadIdx.x;
    const int sp = blockIdx.y;
    if (o >= OUT) return;
    const int per = (NB + splitK - 1) / splitK;
    const int b0 = sp * per, b1 = min(b0 + per, NB);
    float acc = 0.f;
    for (int blk = b0; blk < b1; ++blk) {
        const long base = (long)blk * OUT + o;
        const float scale = exp2f((float)scale_exp[base]);
        const unsigned char sb = shared_cm[base * SB];
        const unsigned int* src = reinterpret_cast<const unsigned int*>(upper_cm + base * 24);
        unsigned int ureg[6];
        #pragma unroll
        for (int i = 0; i < 6; ++i) ureg[i] = src[i];
        unsigned long long ubuf = 0ull; int unb = 0, uwi = 0;
        #pragma unroll
        for (int k = 0; k < BLK; ++k) {
            if (unb < 6) { ubuf |= (unsigned long long)ureg[uwi++] << unb; unb += 32; }
            const unsigned int c6 = (unsigned int)(ubuf & 63u); ubuf >>= 6; unb -= 6;
            const int q  = (int)(c6 ^ 32u) - 32;                 // sign-extend 6
            const int g  = k >> 3;
            const int sh = (int)(((sb >> (g * 2)) & 3u) ^ 2u) - 2; // sign-extend 2
            const int word = q * 4 + sh;
            acc += (float)word * scale * __bfloat162float(x[blk * BLK + k]);
        }
    }
    partial[(long)sp * OUT + o] = acc;
}

__global__ void gemv_u2_split4(
    const __nv_bfloat16* __restrict__ x, const signed char* __restrict__ scale_exp,
    const unsigned char* __restrict__ nib, const unsigned char* __restrict__ low2,
    const unsigned char* __restrict__ shared_cm,
    float* __restrict__ partial, int OUT, int NB, int SB, int splitK) {
    const int o  = blockIdx.x * OUTW + threadIdx.x;
    const int sp = blockIdx.y;
    if (o >= OUT) return;
    const int per = (NB + splitK - 1) / splitK;
    const int b0 = sp * per, b1 = min(b0 + per, NB);
    float acc = 0.f;
    for (int blk = b0; blk < b1; ++blk) {
        const long base = (long)blk * OUT + o;
        const float scale = exp2f((float)scale_exp[base]);
        const unsigned char sb = shared_cm[base * SB];
        const uint4 n4 = *reinterpret_cast<const uint4*>(nib  + base * 16);  // 16 B, no straddle
        const uint2 l2 = *reinterpret_cast<const uint2*>(low2 + base * 8);   //  8 B, no straddle
        const unsigned int uw[4] = { n4.x, n4.y, n4.z, n4.w };
        const unsigned int lw[2] = { l2.x, l2.y };
        #pragma unroll
        for (int k = 0; k < BLK; ++k) {
            const unsigned int top4 = (uw[k >> 3] >> ((k & 7) * 4)) & 15u;
            const unsigned int lo2  = (lw[k >> 4] >> ((k & 15) * 2)) & 3u;
            const unsigned int c6 = (top4 << 2) | lo2;
            const int q  = (int)(c6 ^ 32u) - 32;                 // sign-extend 6
            const int g  = k >> 3;
            const int sh = (int)(((sb >> (g * 2)) & 3u) ^ 2u) - 2;
            const int word = q * 4 + sh;
            acc += (float)word * scale * __bfloat162float(x[blk * BLK + k]);
        }
    }
    partial[(long)sp * OUT + o] = acc;
}

// byte-split: SAME dense 6-bit stream as streaming, but the 24 bytes live in a 16 B
// plane + an 8 B plane so each plane's per-thread stride == its load width (uint4=16,
// uint2=8) -> coalesced (high sector-util). Unpack is the SAME rolling 6-bit buffer
// over 6 regs {a.x,a.y,a.z,a.w,b.x,b.y} (codes straddle the 16/24 boundary naturally),
// so NO per-element reassembly (unlike split4). Bit-identical to streaming.
__global__ void gemv_u2_bytesplit(
    const __nv_bfloat16* __restrict__ x, const signed char* __restrict__ scale_exp,
    const unsigned char* __restrict__ a16, const unsigned char* __restrict__ b8,
    const unsigned char* __restrict__ shared_cm,
    float* __restrict__ partial, int OUT, int NB, int SB, int splitK) {
    const int o  = blockIdx.x * OUTW + threadIdx.x;
    const int sp = blockIdx.y;
    if (o >= OUT) return;
    const int per = (NB + splitK - 1) / splitK;
    const int b0 = sp * per, b1 = min(b0 + per, NB);
    float acc = 0.f;
    for (int blk = b0; blk < b1; ++blk) {
        const long base = (long)blk * OUT + o;
        const float scale = exp2f((float)scale_exp[base]);
        const unsigned char sb = shared_cm[base * SB];
        const uint4 a = *reinterpret_cast<const uint4*>(a16 + base * 16);   // 16 B, stride==width
        const uint2 b = *reinterpret_cast<const uint2*>(b8  + base * 8);    //  8 B, stride==width
        unsigned int ureg[6] = { a.x, a.y, a.z, a.w, b.x, b.y };
        unsigned long long ubuf = 0ull; int unb = 0, uwi = 0;
        #pragma unroll
        for (int k = 0; k < BLK; ++k) {
            if (unb < 6) { ubuf |= (unsigned long long)ureg[uwi++] << unb; unb += 32; }
            const unsigned int c6 = (unsigned int)(ubuf & 63u); ubuf >>= 6; unb -= 6;
            const int q  = (int)(c6 ^ 32u) - 32;
            const int g  = k >> 3;
            const int sh = (int)(((sb >> (g * 2)) & 3u) ^ 2u) - 2;
            const int word = q * 4 + sh;
            acc += (float)word * scale * __bfloat162float(x[blk * BLK + k]);
        }
    }
    partial[(long)sp * OUT + o] = acc;
}

// bytesplit + SEPARATED-SCALE accumulate (the production sepsc trick): instead of
// per-element word*scale*x, factor scale out of the block:
//   acc += scale * ( 4*Σ_k q_k*x_k  +  Σ_g sh_g * Σ_{k in g} x_k )
// fewer scale-mults but adds per-group bookkeeping (xsum, group-boundary branch).
// Tests whether sepsc is the production SM-bound bloat vs the lean per-elem path.
__global__ void gemv_u2_sepsc(
    const __nv_bfloat16* __restrict__ x, const signed char* __restrict__ scale_exp,
    const unsigned char* __restrict__ a16, const unsigned char* __restrict__ b8,
    const unsigned char* __restrict__ shared_cm,
    float* __restrict__ partial, int OUT, int NB, int SB, int splitK) {
    const int o  = blockIdx.x * OUTW + threadIdx.x;
    const int sp = blockIdx.y;
    if (o >= OUT) return;
    const int per = (NB + splitK - 1) / splitK;
    const int b0 = sp * per, b1 = min(b0 + per, NB);
    float acc = 0.f;
    for (int blk = b0; blk < b1; ++blk) {
        const long base = (long)blk * OUT + o;
        const float scale = exp2f((float)scale_exp[base]);
        const unsigned char sb = shared_cm[base * SB];
        const uint4 a = *reinterpret_cast<const uint4*>(a16 + base * 16);
        const uint2 b = *reinterpret_cast<const uint2*>(b8  + base * 8);
        unsigned int ureg[6] = { a.x, a.y, a.z, a.w, b.x, b.y };
        unsigned long long ubuf = 0ull; int unb = 0, uwi = 0;
        float bup = 0.f, bsh = 0.f, xsum = 0.f; int sh_code = 0;
        #pragma unroll
        for (int k = 0; k < BLK; ++k) {
            if ((k & 7) == 0) {                       // gs=8 group boundary
                if (k > 0) bsh += sh_code * xsum;
                const int g = k >> 3;
                sh_code = (int)(((sb >> (g * 2)) & 3u) ^ 2u) - 2;
                xsum = 0.f;
            }
            if (unb < 6) { ubuf |= (unsigned long long)ureg[uwi++] << unb; unb += 32; }
            const unsigned int c6 = (unsigned int)(ubuf & 63u); ubuf >>= 6; unb -= 6;
            const int q = (int)(c6 ^ 32u) - 32;
            const float xk = __bfloat162float(x[blk * BLK + k]);
            bup += q * xk; xsum += xk;
        }
        bsh += sh_code * xsum;
        acc += scale * (4.f * bup + bsh);
    }
    partial[(long)sp * OUT + o] = acc;
}

torch::Tensor run_streaming(torch::Tensor x, torch::Tensor se, torch::Tensor up, torch::Tensor sh,
                            int OUT, int NB, int SB, int splitK) {
    auto partial = torch::zeros({splitK, OUT}, torch::dtype(torch::kFloat32).device(x.device()));
    dim3 grid((OUT + OUTW - 1) / OUTW, splitK);
    gemv_u2_streaming<<<grid, OUTW>>>(
        (const __nv_bfloat16*)x.data_ptr(), (const signed char*)se.data_ptr(),
        (const unsigned char*)up.data_ptr(), (const unsigned char*)sh.data_ptr(),
        partial.data_ptr<float>(), OUT, NB, SB, splitK);
    return partial.sum(0);
}
torch::Tensor run_split4(torch::Tensor x, torch::Tensor se, torch::Tensor nib, torch::Tensor low2,
                         torch::Tensor sh, int OUT, int NB, int SB, int splitK) {
    auto partial = torch::zeros({splitK, OUT}, torch::dtype(torch::kFloat32).device(x.device()));
    dim3 grid((OUT + OUTW - 1) / OUTW, splitK);
    gemv_u2_split4<<<grid, OUTW>>>(
        (const __nv_bfloat16*)x.data_ptr(), (const signed char*)se.data_ptr(),
        (const unsigned char*)nib.data_ptr(), (const unsigned char*)low2.data_ptr(),
        (const unsigned char*)sh.data_ptr(),
        partial.data_ptr<float>(), OUT, NB, SB, splitK);
    return partial.sum(0);
}
torch::Tensor run_bytesplit(torch::Tensor x, torch::Tensor se, torch::Tensor a16, torch::Tensor b8,
                            torch::Tensor sh, int OUT, int NB, int SB, int splitK) {
    auto partial = torch::zeros({splitK, OUT}, torch::dtype(torch::kFloat32).device(x.device()));
    dim3 grid((OUT + OUTW - 1) / OUTW, splitK);
    gemv_u2_bytesplit<<<grid, OUTW>>>(
        (const __nv_bfloat16*)x.data_ptr(), (const signed char*)se.data_ptr(),
        (const unsigned char*)a16.data_ptr(), (const unsigned char*)b8.data_ptr(),
        (const unsigned char*)sh.data_ptr(),
        partial.data_ptr<float>(), OUT, NB, SB, splitK);
    return partial.sum(0);
}
torch::Tensor run_sepsc(torch::Tensor x, torch::Tensor se, torch::Tensor a16, torch::Tensor b8,
                        torch::Tensor sh, int OUT, int NB, int SB, int splitK) {
    auto partial = torch::zeros({splitK, OUT}, torch::dtype(torch::kFloat32).device(x.device()));
    dim3 grid((OUT + OUTW - 1) / OUTW, splitK);
    gemv_u2_sepsc<<<grid, OUTW>>>(
        (const __nv_bfloat16*)x.data_ptr(), (const signed char*)se.data_ptr(),
        (const unsigned char*)a16.data_ptr(), (const unsigned char*)b8.data_ptr(),
        (const unsigned char*)sh.data_ptr(),
        partial.data_ptr<float>(), OUT, NB, SB, splitK);
    return partial.sum(0);
}
"""

CPP = r"""
#include <torch/extension.h>
torch::Tensor run_streaming(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, int, int, int, int);
torch::Tensor run_split4(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, int, int, int, int);
torch::Tensor run_bytesplit(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, int, int, int, int);
torch::Tensor run_sepsc(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, int, int, int, int);
"""
m = load_inline(name="proto_split4", cpp_sources=CPP, cuda_sources=CUDA,
                functions=["run_streaming", "run_split4", "run_bytesplit", "run_sepsc"], verbose=False)

# ---- pack ----
rng = np.random.default_rng(0)
W = (rng.standard_normal((OUT, K)) * rng.uniform(0.2, 4.0, (OUT, 1))).astype(np.float32)
p = pack_weight(W, U, GS)                                   # 3-plane (real format)

# 4-plane: split the SAME q_upper (6-bit signed) into a 4-bit nibble + 2-bit plane.
blocks = W.reshape(OUT, NB, BLOCK).reshape(OUT * NB, BLOCK)
_, q_upper, _ = decompose(blocks, U, GS)                    # [OUT*NB, 32], same as pack_weight uses
q6 = (q_upper & 63).reshape(OUT, NB, BLOCK)                 # 6-bit two's complement
top4 = ((q6 >> 2) & 15).astype(np.uint8)                   # bits[5:2]
low2 = (q6 & 3).astype(np.uint8)                            # bits[1:0]

def nib_pack_cm(c):    # [OUT,NB,32] 0..15 -> [NB,OUT,16] LSB-first dense (2 nibbles/byte)
    c = c.reshape(OUT, NB, 16, 2)
    byt = (c[..., 0] | (c[..., 1] << 4)).astype(np.uint8)
    return np.ascontiguousarray(byt.transpose(1, 0, 2))
def two_pack_cm(c):    # [OUT,NB,32] 0..3 -> [NB,OUT,8] LSB-first dense (4 codes/byte)
    c = c.reshape(OUT, NB, 8, 4)
    byt = (c[..., 0] | (c[..., 1] << 2) | (c[..., 2] << 4) | (c[..., 3] << 6)).astype(np.uint8)
    return np.ascontiguousarray(byt.transpose(1, 0, 2))

nib_cm = nib_pack_cm(top4)                                  # [NB,OUT,16]
low_cm = two_pack_cm(low2)                                  # [NB,OUT,8]

# byte-split: just slice the dense 24-byte upper_cm into a 16 B + 8 B plane (no recompute,
# bit-identical stream). Each plane's per-thread stride becomes its load width.
A16 = np.ascontiguousarray(p["upper_cm"][:, :, :16])        # [NB,OUT,16]
B8  = np.ascontiguousarray(p["upper_cm"][:, :, 16:24])      # [NB,OUT,8]

dev = "cuda"
x   = torch.from_numpy(rng.standard_normal(K).astype(np.float32)).to(torch.bfloat16).to(dev)
se  = torch.from_numpy(p["scale_exp"]).to(dev)
up  = torch.from_numpy(p["upper_cm"]).to(dev)
shc = torch.from_numpy(p["shared_cm"]).to(dev)
nibt= torch.from_numpy(nib_cm).to(dev)
lowt= torch.from_numpy(low_cm).to(dev)
a16t= torch.from_numpy(A16).to(dev)
b8t = torch.from_numpy(B8).to(dev)

CALLS = {
    "streaming": lambda: m.run_streaming(x, se, up, shc, OUT, NB, SB, SPLITK),
    "split4":    lambda: m.run_split4(x, se, nibt, lowt, shc, OUT, NB, SB, SPLITK),
    "bytesplit": lambda: m.run_bytesplit(x, se, a16t, b8t, shc, OUT, NB, SB, SPLITK),
    "sepsc":     lambda: m.run_sepsc(x, se, a16t, b8t, shc, OUT, NB, SB, SPLITK),
}

# ---- ncu single-launch mode ----
if os.environ.get("NCU"):
    call = CALLS[os.environ.get("KERN", "bytesplit")]
    for _ in range(5): call()
    torch.cuda.synchronize(); call(); torch.cuda.synchronize()
    sys.exit(0)

# ---- correctness ----
y_s = CALLS["streaming"]()
y_4 = CALLS["split4"]()
y_b = CALLS["bytesplit"]()
y_e = CALLS["sepsc"]()
ref = torch.from_numpy(dequant_weight(p)).float().to(dev) @ x.float()
print(f"max|streaming-split4|   = {(y_s - y_4).abs().max().item():.3e}  (must be ~0)")
print(f"max|streaming-bytesplit|= {(y_s - y_b).abs().max().item():.3e}  (must be ~0)")
print(f"max|sepsc-ref|          = {(y_e - ref).abs().max().item():.3e}  (sepsc refactor, ~ref)")
print(f"max|streaming-ref|      = {(y_s - ref).abs().max().item():.3e}  rel={((y_s-ref).abs().max()/ref.abs().max()).item():.2e}")

# ---- timing ----
def bench(fn, iters=300):
    for _ in range(30): fn()
    torch.cuda.synchronize()
    e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
    e0.record()
    for _ in range(iters): fn()
    e1.record(); torch.cuda.synchronize()
    return e0.elapsed_time(e1) / iters * 1e3  # us

t_s = bench(CALLS["streaming"])
t_4 = bench(CALLS["split4"])
t_b = bench(CALLS["bytesplit"])
t_e = bench(CALLS["sepsc"])
print(f"\nstreaming (3-plane, 6xu32 load)        : {t_s:7.2f} us")
print(f"split4    (4-plane, u4+u2, reassemble) : {t_4:7.2f} us  ({t_s/t_4:.3f}x)")
print(f"bytesplit (2-plane, u4+u2, per-elem)   : {t_b:7.2f} us  ({t_s/t_b:.3f}x)")
print(f"sepsc     (bytesplit + separated-scale): {t_e:7.2f} us  ({t_s/t_e:.3f}x)")
