# two-tier MSAQ — mechanism, positioning, and where robust accuracy appears (Steps 1-3 synthesis)

## 1. What it is — "native low-bit base + a cheap fractional correction"

Three parts:

| part | what | role |
|---|---|---|
| **native MX base** | E2M1 (FP4) or E2M3 (FP6), hardware MXFP | carries most bits on the fast native path |
| **shared residual** | one u-bit value per contraction-axis group gs, **additive on its own per-block E8M0** | fills BETWEEN native rungs, continuously (the fractional knob) |
| **MX+** (optional) | block-max element's exponent bits reused as mantissa | rescues the 1 outlier (orthogonal to the residual's DC) |

**Defining property** (vs MXFP-MSAQ): the shared value is applied **additively on a uniform per-block
scale — NOT multiplied by each element's exponent**. Consequences: (1) u is **independent of the base
mantissa** (E2M1+u4 is expressible; MXFP-MSAQ caps at u≤mb); (2) the correction **factorizes** as
`Y = AŴ + (ĀR̄)·d` — a contraction-aligned **1/32-FLOP correction GEMM** over a native base, instead
of a per-element sub-byte unpack. Cost: the residual needs its own scale (+~0.25 b/elem).

Two knobs: **u** (shared bits), **gs** (group size). Smaller gs = finer = more accurate, +u/gs bits.

## 2. Positioning vs existing formats (same harness, Llama-3.1-8B, BF16 PPL 5.6877, gate ≤3%)

**KV scope** (two-tier's home — K rotated H128 + D-block, V T-block):

| format | b/elem | ΔPPL | robust? | note |
|---|--:|--:|:--:|---|
| E2M1 native (FP4) | 4.25 | +3.46% | ✗ | even rotated, not alone |
| **two-tier MX+ E2M1+u3 gs32** | **4.75** | **+2.71%** | **✓** | robust floor (uniform) |
| **two-tier allocation (mix)** | **5.50** | **+1.30%** | ✓ | per-layer, beats uniform |
| two-tier allocation (mix) | 5.125 | +1.98% | ✓ | +22% KV capacity |
| two-tier MX+ E2M1+u3 gs2 | 6.156 | +1.85% | ✓ | |
| E2M3 native (FP6) | 6.25 | +0.30% | ✓ | quality anchor |

**Weight scope** (contrast — the fractional band is worthless here, crushed by the E2M3 wall):

| format | b/elem | ΔPPL | robust? |
|---|--:|--:|:--:|
| E2M1 native | 4.25 | +12.15% | ✗ |
| two-tier MX+ E2M1 (best sub-6b) | 5.4–5.7 | +3.1% | ✗ (borderline) |
| MXINT8-MSAQ.efb | 6.00 | +2.70% | ✓ |
| E2M3 native | 6.25 | +0.64% | ✓ |

**vs MXFP-MSAQ** (matched per-element E2M1, weight): two-tier wins every cell by ~4pp; MXFP-MSAQ's
exponent-tied shared collapses on real eb=2 outlier blocks (one config diverged to +3.7e7%). MXFP-MSAQ
remains correct for its native FP8 (eb≥3) regime. Detail: `precision/two_tier_vs_msaq_results.md`.

## 3. Where robust accuracy appears

| scope | robust (≤3%) floor | why |
|---|---|---|
| **KV** | **≈ 4.75 b/elem** (uniform); allocation gives **+1.3% at 5.5b** | no native E2M3 wall + KV is the most quant-tolerant scope → the fractional band lives |
| weight | ≈ 6.0–6.25 b/elem | nothing usable below E2M3; the band is dominated |

**One line:** two-tier's robust KV accuracy starts at **~4.75 b/elem — about 1.5 bits below the native
FP6 (E2M3) robust point (6.25b)**. It populates the ~4.75–5.5b gap between the hardware rungs {4,6,8},
buying the same quality at fewer KV bytes → +14–22% capacity → batch → RPS. On weight the same band is
worthless (E2M3 is too cheap/good at 6.25b); on KV there is no such wall, so the band converts to
capacity.

## 4. Allocation (Step 2) — turning the ladder into per-layer mixed precision

Not all layers need the same KV precision. Method: (1) probe each layer's sensitivity (only layer L on
the cheap 4.75b rung, the rest E2M3; ΔPPL); (2) rank least→most sensitive; (3) greedily put the K
least-sensitive layers on cheap, the rest on E2M3, and measure the ACTUAL joint PPL.

Sensitivity is strongly heterogeneous (Llama-3.1-8B, 32 layers):
- **6 layers (8,11,15,26,25,28) are ~free to cheapen** (ΔPPL ≤ 0, noise-level).
- early layers are the most sensitive (**L1 +0.20, L7 +0.19, L3 +0.17**); middle layers tolerant.

Allocation curve vs the best uniform rung at matched bytes:

| avg b/elem | allocation ΔPPL | uniform @ ~bytes | winner |
|--:|--:|---|:--:|
| 6.250 | +0.30% | E2M3 +0.30% | = |
| 5.875 | +0.68% | (none) | alloc |
| **5.500** | **+1.30%** | uniform 5.66b +1.88% | **alloc −0.58pp** |
| **5.125** | **+1.98%** | uniform 5.16b +2.56% | **alloc −0.58pp** |
| 4.938 | +2.28% | uniform 4.91b +2.95% | **alloc −0.67pp** |
| 4.750 | +2.71% | = all-cheap | = |

**The mix beats any single uniform rung by ~0.6pp across the whole range** — classic rate-distortion:
spend bits on sensitive layers (E2M3), save on tolerant ones (cheap). Capacity: +1.30% (well usable)
at 0.88× E2M3 bytes = **+14% KV capacity**; +1.98% at 0.82× = **+22%**. Borrows KVTuner (offline
layer sensitivity) / RateQuant (rate-distortion) skeletons but expands each cell's alphabet from the
discrete native rungs to the continuous two-tier ladder.

Simplifications (refinement targets): layer-wise only (head-wise is finer — 8 KV heads/layer); 2 rungs
only (cheap 4.75 / E2M3 — the full ladder has more); K and V allocated together (they differ in
sensitivity); greedy from independent first-order probes (final numbers use real joint PPL, so they
hold, but the *allocation* may be improvable by a knapsack over the full rung set under a hard ceiling).

## Provenance
Code/results in `precision/two_tier_*`, `precision/kv_ladder_*`; design `kv_ladder_design.md`;
feasibility `kv_ladder_step3_feasibility.md`; gate verdicts `precision/two_tier_results.md`.
