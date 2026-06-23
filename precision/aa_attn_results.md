# W+A + attention activation×activation (AA): robust (u,gs) shift

Does quantizing the **prefill-attention internal matmuls** — `Q·Kᵀ` and `P·V`, where **both operands are
activations** — on top of W+A (weight + linear-input activation) change the max-aggressive robust sharing
config? Plain MSAQ (block=32, no rotation/two-level), wikitext-2, Llama-3.1-8B, **BF16 PPL 6.5684**, 30
windows, criterion ≤3.5%. `aa_attn_ppl.py` / `aa_attn_ppl.txt`.

**AA** = a manual SDPA that MSAQ-quantizes **Q, K** (→ `Q·Kᵀ`) and **P=softmax(scores), V** (→ `P·V`),
each along its last dim (P's key axis zero-padded to a multiple of 32 for the block reshape); GQA KV is
repeated to the query heads first. Baseline **W+A** keeps attention in full precision (fused SDPA).
PPL is teacher-forced (sliding-window NLL; quant applied inside the full-context forward).

## Result
| config | W+A (attn fp) | W+A+AA (attn act×act quant) | Δ |
|---|---|---|---|
| u2/gs2  | +0.79% | +1.38% | +0.59 |
| u2/gs4  | +1.47% | +1.96% | +0.49 |
| **u2/gs8**  | +1.59% | **+2.47%** | +0.88 |
| u2/gs16 | +1.77% | +2.72% | +0.95 |
| u2/gs32 | +1.73% | +2.76% | +1.03 |
| u3/gs2  | +3.71% ❌ | **+5.48% ❌** | +1.77 |
| **max-aggressive robust** | **u2/gs8 (6.50 b/elem)** | **u2/gs8 (6.50 b/elem)** | — |

## Findings
1. **The fewest-bits robust config does NOT change — still `u2/gs8` (6.50 bits/elem).** Attention
   activation×activation quantization is **survivable at u2** (the W+A robust regime): u2/gs8 +2.47%,
   u2/gs32 +2.76%, all within the 3.5% bar.
2. **But AA consumes ~+0.9–1.0 pp of the PPL budget** across u2 configs (u2/gs8 1.59→2.47, u2/gs32
   1.73→2.76) — the margin to 3.5% shrinks from ~1.9pp to ~0.7–1.0pp.
3. **u3 goes from borderline-fail to firmly out of reach**: u3/gs2 +3.71% (W+A, just over) → **+5.48%**
   (W+A+AA). So with attention AA quant there is **zero headroom for u3** — the robust frontier is hard-
   pinned at u2.
4. **Interpretation.** Attention activations (Q,K,V and the softmax probs P) are individually quant-
   tolerant at u2 (consistent with KV's tolerance), so adding the two AA matmuls keeps W+A at its u2/gs8
   robust point; the cost is the ~1pp margin and the loss of any u3 option. The dominant accuracy bottleneck
   remains the linear-activation × weight path (which already capped W+A at u2), not the attention AA matmuls.

## Note
This is an **accuracy** study (does the quant format survive). It is not wired into the latency harness:
the prefill attention in `harness_batchsweep` is bf16 SDPA (the AA matmuls would need quantized-attention
kernels to realize any latency effect). The point here is the robust-config shift, not speed.
