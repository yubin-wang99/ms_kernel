# C2 — is the VQ residual LUT-fold free on KV decode? (kernel sketch + NCU)

Code: `vq_c2_microbench.cu`. Closes the last gap of `vq_kv_global_results.md`: FP4+vector-VQ passes the
KV PPL gate and generalizes, but the claim that the per-element VQ correction is *free* (caveat C2) was
argued from KV decode being byte-roofline, not measured. This measures it.

## Kernel sketch — LUT-fold into the dequant GEMV

KV decode score = `qᵀK` is a GEMV: one thread per key streams that key's quantized K over D=128
(NB=4 blocks of 32), dequantizes on the fly, MACs against q staged in shared (the `kv_attention.cu`
kdot pattern). The VQ residual folds directly into the per-element dequant — no separate pass, no second
GEMM:

```
codebook (K×g halves, e.g. 256×8 = 4 KB) staged once in shared per block;
per element e in the key:
    nib  = unpack FP4 code           # base, 4 b
    w    = sign · FP4MAG[nib]        # base magnitude (LUT, 8 entries in const)
    g    = e / g_size                # group id
    idx  = group_index[g]            # +log2(K)/g b/elem  (the only stored cost)
    w   += codebook_smem[idx*g + e%g]   # <-- the fold: one smem gather + one add
    dot += q[e] * (w * scale)
```

So decode cost = base unpack + **one shared-memory gather + one add per element**. The correction is
purely additive in the dequant; the index is the only extra DRAM traffic.

## Measurement (RTX PRO 4000 Blackwell, sm_120; copy peak ≈ 555 GB/s)

Three kernels, identical geometry + coalesced key-minor layout (consecutive threads → consecutive
bytes, like token-major KV), differing only in per-element decode and bytes streamed. 4M keys, D=128.

| kernel | b/elem | time | GB/s | DRAM %peak | SM(compute) %peak | occupancy | smem bank-conflicts |
|---|--:|--:|--:|--:|--:|--:|--:|
| `fp4` (base only) | 4.250 | 0.501 ms | 569 | 93.3% | 55.9% | 94% | 41 K |
| **`fp4vq` (base+VQ)** | 5.250 | **0.620 ms** | 568 | **92.6%** | 68.2% | 95% | 307 K |
| `iso` (same bytes, no gather) | 5.250 | 0.628 ms | 561 | 91.7% | 50.4% | 95% | 37 K |

## Verdict — the LUT-fold is FREE; the only cost is the index bytes

1. **All three are DRAM-bound** (DRAM 92–93% of peak; compute only 50–68%) — including the VQ kernel.
   The byte-roofline assumption holds for the KV-decode GEMV.
2. **At iso-bytes, `fp4vq` ≈ `iso`** (0.620 vs 0.628 ms — VQ is even marginally faster), both at ~peak
   bandwidth. The codebook gather + add **hides entirely under the DRAM read**: it raises compute util
   (68% vs 50%) and shared-memory bank conflicts (307 K vs 37 K), but stays well below the DRAM ceiling,
   so it adds **zero** wall-clock. **C2 escape confirmed.**
3. **Cost scales purely with bytes:** `fp4vq`/`fp4` = 0.620/0.501 = **+23.7% time ≈ +23.5% bytes**
   (5.25/4.25). The VQ residual costs exactly its stored index bits and nothing more — no second GEMM,
   no decode penalty.

So on KV decode the FP4+vector-VQ correction is free in compute: decode time tracks bytes/token (the
quantity the fractional rung already optimizes), not decode complexity. The full bits→quality→
bytes/token→capacity→batch→RPS chain now has every link except the end-to-end RPS measurement:
- precision: FP4+VQ owns the 4.75–6.25b KV band, beats mantissa-share iso-bit (`vq_kv_results.md`);
- generalization: fixed/global calibrated codebook reproduces it (`vq_kv_global_results.md`);
- **compute: the correction is free (this file).**

## Caveats / scope
- This is a faithful **microbench of the kdot inner loop** (same pattern as `kv_attention.cu`), not the
  full fused attention kernel; integrating into the real `kv_decode_*` kernels is the production step.
- The gather adds real shared-memory pressure (307 K conflicts, 68% compute): at K=256/g=8 it is hidden,
  but a much larger codebook or a conflict-hostile layout could erode the margin — keep the codebook
  small and smem-resident (4 KB here), or broadcast-friendly.
- Free *compute* ≠ free *capacity*: the +1 b/elem index is real KV bytes; the win is that those bytes
  buy more quality per byte than mantissa-share, at no extra decode time.
