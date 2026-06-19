# End-to-End Harness Results — BF16-normalized total time

RTX 3090, CUDA-graph decode (prefill=800 / decode=3880). Timing harness: random reused weights, glue (RMSNorm/RoPE/SwiGLU/SDPA) in bf16 common to all paths. **Each cell = total inference time normalized to that (model, scope)'s BF16 = 1.000; lower = faster.** Scopes apply quantization to: S1 weights (W-only GEMM/GEMV), S2 weights+activations (INT8 IMMA / int-dot), S3 KV cache only, S4 weights+KV. MSAQ swept over u∈{2,3,4} × gs∈{2,8,32}; MXINT8 and BF16 have no u/gs. See [kernel_ver2.md](kernel_ver2.md), [for_fair_comparison.md].


## Llama-3.1-8B  (BF16 = 1.000, abs 145.8s)

| config | S1 W-only | S2 W+A | S3 KV-only | S4 W+KV |
|---|---|---|---|---|
| **BF16** | 1.000 | 1.000 | 1.000 | 1.000 |
| **MXINT8** | 0.999 | 0.885 | 0.631 | 0.632 |
| MSAQ u2 gs2 | 0.974 | 0.941 | 0.684 | 0.666 |
| MSAQ u2 gs8 | 0.933 | 0.906 | 0.671 | 0.611 |
| MSAQ u2 gs32 | 0.924 | 0.895 | 0.670 | 0.600 |
| MSAQ u3 gs2 | 0.977 | 0.939 | 0.686 | 0.670 |
| MSAQ u3 gs8 | 0.924 | 0.893 | 0.670 | 0.600 |
| MSAQ u3 gs32 | 0.912 | 0.878 | 0.665 | 0.584 |
| MSAQ u4 gs2 | 0.812 | 0.817 | 0.653 | 0.474 |
| MSAQ u4 gs8 | 0.794 | 0.799 | 0.638 | 0.440 |
| **MSAQ u4 gs32** | 0.788 | 0.796 | 0.636 | 0.429 |

## Gemma-2-9B  (BF16 = 1.000, abs 187.5s)

| config | S1 W-only | S2 W+A | S3 KV-only | S4 W+KV |
|---|---|---|---|---|
| **BF16** | 1.000 | 1.000 | 1.000 | 1.000 |
| **MXINT8** | 0.997 | 0.884 | 0.618 | 0.613 |
| MSAQ u2 gs2 | 0.980 | 0.940 | 0.770 | 0.748 |
| MSAQ u2 gs8 | 0.941 | 0.907 | 0.746 | 0.685 |
| MSAQ u2 gs32 | 0.931 | 0.896 | 0.743 | 0.673 |
| MSAQ u3 gs2 | 0.980 | 0.937 | 0.775 | 0.753 |
| MSAQ u3 gs8 | 0.931 | 0.893 | 0.672 | 0.604 |
| MSAQ u3 gs32 | 0.920 | 0.881 | 0.668 | 0.588 |
| MSAQ u4 gs2 | 0.824 | 0.823 | 0.699 | 0.528 |
| MSAQ u4 gs8 | 0.807 | 0.806 | 0.633 | 0.444 |
| **MSAQ u4 gs32** | 0.801 | 0.802 | 0.630 | 0.433 |

## Mistral-7B  (BF16 = 1.000, abs 142.3s)

| config | S1 W-only | S2 W+A | S3 KV-only | S4 W+KV |
|---|---|---|---|---|
| **BF16** | 1.000 | 1.000 | 1.000 | 1.000 |
| **MXINT8** | 0.999 | 0.883 | 0.622 | 0.623 |
| MSAQ u2 gs2 | 0.975 | 0.941 | 0.677 | 0.659 |
| MSAQ u2 gs8 | 0.934 | 0.906 | 0.663 | 0.602 |
| MSAQ u2 gs32 | 0.924 | 0.893 | 0.662 | 0.590 |
| MSAQ u3 gs2 | 0.975 | 0.940 | 0.679 | 0.661 |
| MSAQ u3 gs8 | 0.923 | 0.890 | 0.663 | 0.591 |
| MSAQ u3 gs32 | 0.911 | 0.876 | 0.659 | 0.574 |
| MSAQ u4 gs2 | 0.808 | 0.813 | 0.645 | 0.462 |
| MSAQ u4 gs8 | 0.790 | 0.794 | 0.630 | 0.426 |
| **MSAQ u4 gs32** | 0.782 | 0.790 | 0.627 | 0.416 |

## Best MSAQ (min over u,gs) — normalized to BF16 and to MXINT8

| model | scope | best cfg | /bf16 | /mxint8 |
|---|---|---|---|---|
| Llama-3.1-8B | S1 W-only | u4 gs32 | 0.788 | 0.788 |
| Llama-3.1-8B | S2 W+A | u4 gs32 | 0.796 | 0.899 |
| Llama-3.1-8B | S3 KV-only | u4 gs32 | 0.636 | 1.008 |
| Llama-3.1-8B | S4 W+KV | u4 gs32 | 0.429 | 0.679 |
| Gemma-2-9B | S1 W-only | u4 gs32 | 0.801 | 0.803 |
| Gemma-2-9B | S2 W+A | u4 gs32 | 0.802 | 0.907 |
| Gemma-2-9B | S3 KV-only | u4 gs32 | 0.630 | 1.020 |
| Gemma-2-9B | S4 W+KV | u4 gs32 | 0.433 | 0.707 |
| Mistral-7B | S1 W-only | u4 gs32 | 0.782 | 0.782 |
| Mistral-7B | S2 W+A | u4 gs32 | 0.790 | 0.895 |
| Mistral-7B | S3 KV-only | u4 gs32 | 0.627 | 1.008 |
| Mistral-7B | S4 W+KV | u4 gs32 | 0.416 | 0.667 |
