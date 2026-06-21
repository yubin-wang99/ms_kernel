# Hadamard rotation × MSAQ — accuracy verification (QSNR + small-sample PPL)

Does block-Hadamard rotation (Gaussianize before E8M0 scale) improve MSAQ accuracy? Rotation is a
pair op: `(W·H)(X·H)ᵀ = W·H·Hᵀ·Xᵀ = c·W·Xᵀ` (unnormalized ±1 H, `H·Hᵀ = n·I`; `n=2^k` folds into
E8M0 exactly → no extra rounding). Tested with the **effective-dequant fold** (`x_deq = msaq(x@H)@Hᵀ/n`)
so the result is bit-faithful to "rotate both sides" without online un-rotation in the accuracy test.
wikitext-2, Llama-3.1-8B, BF16 PPL=6.57. (`rot_qsnr.py`, `rot_ppl.py`, `rot_kv_ppl.py`.)

## 1. Weight scope (H₃₂ per 32-block) — modest gain, hurts u4
QSNR (224 Linear tensors): **+0.27–0.33 dB** across configs (small — MXINT8's 8-bit base is already
precise; no FP4-level drama). PPL:
| config | MSAQ | +rot | gain |
|---|---|---|---|
| u3/mg8 (robust) | +2.92% | **+2.26%** | +0.66pp |
| u3/mg4 | +2.77% | +1.93% | +0.85pp |
| u2/mg8 | +0.79% | +0.54% | +0.25pp |
| u4/mg8 | +10.0% | **+13.5%** | **−3.49pp (HURTS)** |
Helps the robust u2/u3, but **hurts u4**: 3-unshared-bit base is too coarse — the original "1 outlier +
31 small" block was actually easier to code than a rotation-flattened uniform block. → only worth it
for u2/u3 weights, and the gain is small.

## 2. KV scope (H₁₂₈ full head-dim) — substantial, and structurally important
Key has persistent CHANNEL outliers → full head_dim rotation mixes them (32-block can't). Result:
| scope·config | MSAQ | +rot | gain |
|---|---|---|---|
| **K** u3/mg8 | +1.07% | +0.42% | +0.65pp |
| **K** u4/mg8 | +4.63% | **+1.77%** | **+2.85pp** |
| **K** u4/mg2 | +2.49% | +1.08% | +1.41pp |
| **V** (all) | +0.06..+0.28% | ~same | **~0** |
| **KV** u4/mg8 | +5.14% (FAIL) | **+2.04% (robust)** | **+3.09pp** |
| KV u3/mg8 | +1.25% | +0.35% | +0.90pp |

- **Key rotation is a big win (+0.65..+2.85pp)** — kills channel outliers (QuaRot mechanism).
- **Value rotation ≈ 0** — V is already quant-tolerant → rotate only K (V is also free via W_o fold,
  but accuracy doesn't need it).
- **Structural payoff:** rotation makes the **fast nibble u4 config robust for KV** — u4/mg8 +5.14→
  +2.04%, u4/mg2 +2.49→+1.08%. u4/gs2 is exactly the packing-friendly config the vpack KV-decode
  kernel wins with → rotation **expands the robust frontier toward the fastest config**.

## Verdict
Rotation's accuracy value for MSAQ is **concentrated in KV-Key**, not weights. Weight gain is marginal
(+0.3 dB, hurts u4); **Key gain is large and unlocks the aggressive nibble config**. Next: a kernel
that rotates K online (post-RoPE H₁₂₈, Q mirrored) — but that adds cost to the latency-bound decode
hot path, so the TPOT delta must be weighed against the accuracy/aggressiveness gain (per the no-fake-win
rule). V-rotation via W_o fold is free but accuracy-irrelevant. Matched baseline: compare MSAQ+rot vs
MXINT8+rot on the accuracy axis; on latency, rotating only MSAQ-K is a format difference (note in
for_fair_comparison.md), like quant_act_msaq.
