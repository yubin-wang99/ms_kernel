# Free-index two-level residual — recovery-rate (ρ) gate

Code: `index_residual.py` (sweep), `index_residual_qsnr.txt` (full output),
`index_residual_washout_diag.txt` (mechanism diagnostic).

## 1. Mechanism — "free index two-level residual"

On a native MX base (FP4/FP6/INT4/INT6), approximate the residual `R = W − Ŵ` with a **global scalar
codebook**, where each element's codebook **index is DERIVED FROM the base encoding** (free, no storage)
instead of from the residual. Then `correction[elem] = codebook[ index(base_elem) ] · ulp[elem]`,
making the element a two-level type: base (1st) + free-indexed residual correction (2nd). Only the index
"costs" anything, and for the free candidates it costs nothing (already in the base datapath); codebook
entry values are learned offline.

**Index candidates (swept, not fixed):**
- *Free / no-storage* — ① exponent shift τ, ② base mantissa low bits (ulp-normalized; meant to carry
  cell-curvature bias), ③ magnitude rank, ④ τ×②.
- *Stored / reference* — ⑤ residual sign (1 b), ⑥ residual high bits, ⑦ offline k-means VQ index.

**Hypotheses.** (H1) some free index carries residual info — info-theoretically only via base
cell-curvature bias in mantissa low bits (②④), since τ carries only scale and the RTN residual value is
~independent of kept bits. (H2) coarser base ⇒ more recoverable (ρ FP4/INT4 > FP6/INT6).

## 2. Decisive experiment — recovery rate ρ

Judge **recovery rate**, not absolute accuracy. Two baselines bracket every candidate:
- `zero`  = base only, no correction (lower bound).
- `oracle` = best K-entry scalar VQ of the residual (Lloyd-Max, nearest assignment; upper bound).

`ρ = (QSNR(cand) − QSNR(zero)) / (QSNR(oracle) − QSNR(zero))`.  ρ≈0 ⇒ index carries no residual info
(H1 rejected); ρ→1 ⇒ free index ≈ oracle.

**Grid:** base ∈ {FP4,INT4,FP6,INT6} × index ∈ {①–⑦} × K ∈ {2,4,8,16} × sign-guard ∈ {off, A, B}.
Codebook = per-bucket mean of the residual after bucketing by the candidate index (index logic ⟂ entry
values). Metric: per-layer QSNR (q_proj, down_proj) on Llama-3.1-8B. **Weight scope first; KV only if the
weight gate passes.**

**Design decisions fixed in code (per caveats):**
- **C6 normalization:** codebook operates on the **ulp-normalized** residual `rn = r/ulp` (exact local
  grid step for FP, 1 for INT in the scaled domain) — so ②/⑥ are cell-relative, not scale-mixed.
- Codebook learned **in-distribution per (tensor, base)** ⇒ ρ measures the index's *pure information
  content* (an upper bound on any real global/per-scope table; C3 deferred).
- Free = ①②③④ at **sign-guard OFF** (no stored bits beyond the index). Sign-guard **B adds a stored
  residual-sign bit**, so B numbers are NOT free and are reported separately. ⑦ VQ = oracle assignment
  (stored) ⇒ ρ≈1 by construction (sanity).

## 3. Results — ρ on Llama-3.1-8B weights (q_proj shown; down_proj identical)

**Pure-free ρ (sign-guard OFF) = 0.00 for every free index, every base, every K.** Stored references
behave as expected. (oracle Δ = dB the K-entry oracle adds over `zero`.)

| FP4 (zero 17.88 dB) | K=2 | K=4 | K=8 | K=16 |  | INT4 (zero 15.70) | K=2 | K=4 | K=8 | K=16 |
|---|---|---|---|---|---|---|---|---|---|---|
| oracle (ΔdB) | +4.39 | +9.34 | +13.16 | +15.33 |  | oracle (ΔdB) | +3.84 | +10.51 | +16.69 | +22.35 |
| ① τ / ② mant / ③ rank / ④ τ×m | **0.00** | **0.00** | **0.00** | **0.00** |  | ①②③④ | **0.00** | **0.00** | **0.00** | **0.00** |
| ⑤ r-sign* | 1.00 | 0.47 | 0.33 | 0.29 |  | ⑤ r-sign* | 0.97 | 0.36 | 0.22 | 0.17 |
| ⑥ r-hi* | 1.00 | 0.94 | 0.96 | 0.98 |  | ⑥ r-hi* | 0.97 | 0.94 | 0.95 | 0.98 |
| ⑦ VQ* | 1.00 | 1.00 | 1.00 | 1.00 |  | ⑦ VQ* | 1.00 | 1.00 | 1.00 | 1.00 |

FP6 / INT6: identical pattern — free ①②③④ = 0.00 at all K; ⑤ 1.0@K2, ⑥ ≈0.98–1.0, ⑦ =1.0
(full table in `index_residual_qsnr.txt`). `*` = stored.

**Sign-guard B (adds +1 stored residual-sign bit), K=2 ρ** — looks high but is NOT the free index:

| base | ① τ | ② mant | ③ rank | ④ τ×m |
|---|---|---|---|---|
| FP4 | 1.02 | 1.09 | 1.04 | 1.04 |
| INT4 | 1.19 | 1.08 | 1.33 | 1.19 |
| down_proj FP4 | 3.86 | 4.04 | 3.87 | 3.35 |

The four *different* free indices give ~identical ρ under B (and it equals ⑤ r-sign alone) ⇒ the recovery
is **entirely the stored sign bit**, nothing from the index. (ρ>1 on down_proj: B at K=2 = 2 buckets × 2
signs = 4 entries vs the K=2 oracle's 2 — it beats the *2*-entry oracle purely by having more entries.)

### Mechanism diagnostic (`index_residual_washout_diag.txt`) — real washout, not a code artifact

For every free index the buckets are genuinely varied and populated, but each bucket's residual is
zero-mean: |bucket-mean| ≈ 1e-4 vs within-bucket std ≈ 0.26 (|mean|/std ≈ 0.001).

| FP4 q_proj, K=4 | bucket pop % | \|bucket-mean\| | within-bucket std | (rn std = 0.259) |
|---|---|---|---|---|
| ② mant-lo | [39,17,25,18] | [1e-4, 1e-4, 1e-4, 3e-4] | [0.20, 0.30, 0.26, 0.32] | |
| ① τ | [30,32,16,22] | ≈ 0 | [0.11, 0.32, 0.27, 0.30] | |

The free index captures the residual's **scale** (within-bucket std varies 0.11↔0.32) but **not its
value/sign** — the residual is zero-mean inside every base-derived bucket, so an additive bucket-mean
codebook reconstructs ≈0. Even ② mant-lo (H1's only theoretical escape via cell-curvature bias) has
|mean| ≈ 1e-4 ⇒ that bias is negligible for these weights.

## 4. Verdict

- **H1 REJECTED, decisively.** No free index (τ, mantissa-low-bits, rank, τ×mant) recovers any of the
  oracle residual correction — ρ = 0.00 across all bases and all K. The cell-curvature-bias escape (②④)
  is empirically null. Confirms C1: the RTN residual value is independent of the base-derived index; the
  index carries only scale, and the value is zero-mean within every bucket → an additive free-indexed
  codebook is expected-0.
- **H2 is moot.** There is no "coarser base ⇒ more free-indexable" signal — FP4/INT4 free ρ = 0 exactly
  like FP6/INT6. Base coarseness does not make the residual free-indexable.
- **Only STORED bits carry the residual:** ⑤ sign is the single most valuable bit (ρ=1.0 @K=2), ⑥ r-hi
  ≈ the residual itself (ρ≈0.95–1.0), ⑦ VQ = oracle. All require storage and a per-element add.

**Three-way branch (spec §2) → branch 2: "stored-only high, free pegged at zero."** H1 rejection is
confirmed; the center of gravity moves to a deliberately STORED second level — either the §5 shared
scalar (= the mantissa-sharing already validated in `two_tier_results.md` / `em_sharing_results.md`) or
a full ⑦ VQ. The free-index, no-storage premise is dead on weights.

**Gate NOT passed ⇒ KV extension not run** (spec: "offline weight scope 먼저, 통과 후 KV로 확장").
The failure mode is scope-independent (zero-mean residual, already shown identical on KV in
`em_sharing_results.md`: mean/std ≈ 0.15 on both W and KV), so a free index would wash out on KV for the
same reason; pursuing it there is not warranted without first beating `zero` on weights, which it cannot.

**Caveat status:** C1 (info-theoretic null) — **realized, this is the result**. C5 (τ cardinality) —
visible: ρ never rises with K for free indices (no saturation point because it starts at 0). C2/C4
(compute factorization, sign-guard divergence) — moot, since no free candidate survives to a deployment
discussion. C3/C6 — handled in setup (in-distribution codebook; ulp-normalized), and C6 specifically
gave ② its best shot (cell-relative bits) yet it still read 0.
