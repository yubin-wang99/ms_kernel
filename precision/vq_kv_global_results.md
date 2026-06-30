# FP4 + vector-VQ residual on KV — generalization with a FIXED (calibration-learned) codebook

Code: `vq_kv_global_ppl.py`, raw `vq_kv_global_llama31_8b.txt`. Closes the main caveat of
`vq_kv_results.md`: there the VQ codebook was learned in-distribution on the eval data (upper bound;
C3 + calibration asymmetry). Here the codebook is learned on wikitext-2 **TRAIN** (disjoint) and applied
**FIXED** to wikitext-2 **TEST**.

**Setup.** Llama-3.1-8B (NousResearch), BF16 TEST PPL = 5.6877 (30w), KV-only, FP4 base (K: H128-rot +
MX D-block + VQ residual; V: MX T-block + VQ residual). Calibration = 8 TRAIN windows; codebooks fit
offline (k-means), then frozen. Three regimes:
- **indist** — fit on the eval tensor (per layer, cached) — the prior upper bound.
- **perlayer** — fixed per-(layer, K/V) codebook learned on TRAIN — tests data generalization.
- **global** — fixed SINGLE codebook per K/V, shared across all 32 layers, learned on TRAIN — tests
  whether one table suffices (C3).

## Results (ΔPPL% vs BF16; gate ≤ 3%)

| config | b/elem | indist | **perlayer (fixed, calib)** | global (fixed, calib) |
|---|--:|--:|--:|--:|
| FP4+VQ g8/K16 | 4.750 | +2.33% | **+2.42%** | +2.62% |
| FP4+VQ g8/K256 | 5.250 | +1.69% | **+1.66%** | +1.81% |
| FP4+VQ g4/K256 | 6.250 | +0.39% | **+0.30%** | +0.37% |

Reference (same harness, `kv_ladder_step1`): mantissa-share MX+ E2M1+u2 gs32 (4.719b) **+3.07%**, gs2
(5.656b) **+1.88%**; native FP6 (6.25b) **+0.30%**.

## Verdict — generalization CONFIRMED; the win is real, not in-distribution overfit

- **The fixed calibration-learned codebook matches in-distribution within ±0.1 pp at every budget**
  (perlayer is even *better* than indist at 5.25b and 6.25b — the in-distribution "advantage" was noise,
  not a real edge). So FP4+VQ's KV win is **not** an artifact of fitting the codebook on eval data: it
  survives a clean train→test split.
- **A single GLOBAL codebook (one table per K/V across all 32 layers) costs only +0.1…+0.29 pp** vs
  per-layer. C3's "one table wobbles" is largely refuted on KV — a single calibrated table holds up;
  per-layer banks buy a fraction of a point. Decode can carry one tiny LUT, not 32.
- **The win stands against the incumbent under a fair (now-calibrated) comparison.** Fixed FP4+VQ:
  - 4.75b **+2.42%** beats mantissa-share MX+ iso-bit (4.719b, +3.07%) by ~0.65 pp.
  - 5.25b **+1.66%** beats the best incumbent fractional point (5.656b, +1.88%) at **fewer bits**.
  - 6.25b **+0.30% = native FP6** — the E2M3 wall that capped weight is **closed on KV** with a
    deployable, fixed codebook.

## Net

FP4 + vector-VQ residual is a **generalizing** KV format: a codebook calibrated once on disjoint data
(even a single global table) reproduces the in-distribution result and owns the 4.75–6.25b fractional
band, reaching FP6 quality at FP4+2b. The residual-correction line — dead on weights (EM_sharing,
free-index, VQ-locked-behind-C2) — lands cleanly here.

**Remaining honest gaps (deployment, not precision):**
- VQ still requires an offline calibration step that data-independent mantissa-share does not; the ~0.65 pp
  iso-bit edge is the price/benefit of that calibration. A calibrated mantissa-share baseline would tighten
  the comparison.
- The C2 LUT-fold into the KV-dequant GEMV is argued from decode being byte-roofline (kv_ladder §0), not
  yet measured — kernel sketch + NCU is the next step.
- The capacity→batch→RPS payoff (the actual KV thesis) is downstream of the bytes/token these fractional
  rungs buy, not shown here.
