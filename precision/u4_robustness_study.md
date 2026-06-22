# u=4 robustness study — block size · scale format · rotation · MX two-level (MSAQ-signed)

**Question.** With the OCP-standard 32-element block, can MSAQ-signed reach **u=4** (the most
aggressive nibble base) *robustly* — within **3% of BF16 wikitext PPL** — across the 4 quantization
scopes (weight / weight+activation / KV / weight+KV)? If not at block=32/E8M0, which lever fixes it?

**Setup.** Llama-3.1-8B-Instruct, wikitext-2-raw test, sliding window (2048/1024, 30 windows).
**BF16 PPL = 6.5684**; criterion **≤ +3.00%**. Quantizer = MSAQ-signed (`mg` = shared group). Each lever
toggled via SDPA / Linear.forward / in-place-weight patches; all scripts cross-check `d2=0`/`e8m0` ==
the deployed `msaq_signed`. Scripts: `block16_ppl.py`, `scale_fmt_ppl.py`, `weight_wide_rot_ppl.py`,
`mx_2lvl_ppl.py`, `wa_rot_2lvl_ppl.py` (+ `.txt` logs).

## Baseline — E8M0, block=32 (the starting point): u=4 fails almost everywhere
| scope | u4/mg8 | u4/mg4 | u4/mg2 |
|---|---|---|---|
| weight | +10.02% | +9.03% | +6.04% |
| weight+act | +26.17% | +21.94% | +13.43% |
| KV | +5.14% | +4.26% | +2.89% ✅ |
| weight+KV | +17.55% | +15.27% | +9.32% |
Only **KV u4/mg2** clears 3%. Everything else fails — W+A worst (activation × weight error compounds).

## Lever 1 — block size 32 → 16 → 8
Monotonic improvement (finer E8M0 scale snaps less). **Only KV becomes fully robust, and only at block=8.**
| scope (u4/mg8) | b32 | b16 | b8 |
|---|---|---|---|
| weight | +10.02% | +7.82% | +6.22% ❌ |
| weight+act | +26.17% | +17.29% | +12.23% ❌ |
| KV | +5.14% | +3.94% | **+2.50% ✅** |
| weight+KV | +17.55% | +13.40% | +9.20% ❌ |
KV: block=8 robust for **all** mg (mg8 +2.50, mg4 +2.16, mg2 +1.29). Cost: scale overhead 0.25→0.5→**1.0 bit/elem**.

## Lever 2 — scale FORMAT E8M0 → E4M3 / UE5M3 (block=32, 3 mantissa bits in the scale)
A mantissa'd block scale fits absmax tighter than pow2-only E8M0 → ~20–40% lower error.
| scope (u4/mg8) | E8M0 | E4M3 | UE5M3 |
|---|---|---|---|
| weight | +10.02% | +6.18% | +6.17% ❌ |
| weight+act | +26.17% | **+163.10%** 💥 | +15.80% ❌ |
| KV | +5.14% | +3.05% | +3.08% ❌ (경계) |
| weight+KV | +17.55% | +11.00% | +10.76% ❌ |
- **E4M3 collapses on activations (+121–163%)**: its 4-bit exponent can't span the activation block-scale
  spread (even with a per-tensor global). **UE5M3 (5-bit exp) is the safe mantissa'd scale.**
- KV u4/mg4·mg2 become robust; **u4/mg8 lands right at +3.08% (still FAIL)**. No scope's mg8 cleared.

## Lever 3 — per-tensor FP32 scale on top of per-block E8M0 (block=32): NEGATIVE
rmse −0.8% only (vs −23% for block=16). **A per-tensor (coarser-than-block) scale is absorbed by the
per-block E8M0**, which already fits each block with a full 8-bit exponent. *Accuracy needs FINER-than-block
scaling, not coarser.* (inline rmse check, no PPL run.)

## Lever 4 — weight rotation (Hadamard): NEGATIVE for u=4 weight
| weight | none | block-H32 | full input-dim H |
|---|---|---|---|
| u4/mg8 | +10.02% | +13.85% | +11.83% ❌ |
| u3/mg8 | +2.92% | +2.03% | +2.07% ✅ |
Rotation **hurts weight u4** (both H32 and full H4096) — weight outliers are **in-block** ("1 big + 31
small"), which MSAQ already codes well; flattening them overwhelms the coarse 4-bit base. It *helps* the
already-robust u3 (mechanism works, wrong regime). (Contrast: KV-Key outliers are **channel**-type →
rotation kills them, the deployed online-K-rotation win, `rot_results.md`.)

## Lever 5 — MX6/MX9 TWO-LEVEL scaling (block=32): the winner
L1 = 8-bit E8M0 / 32 + **L2 = d2-bit microexponent / 2 elements** (Rouhani et al. shared microexponents).
A local sub-block scale at only **d2/2 bit/elem**. `d2=0` == baseline.
| scope (u4/mg8) | d2=0 | d2=1 (+0.5 b/elem) | d2=2 (+1.0 b/elem) |
|---|---|---|---|
| weight | +10.02% | +4.72% | +3.44% (mg4/mg2 ✅) |
| weight+act | +26.17% | +10.03% | +6.44% ❌ |
| KV | +5.14% | **+2.06% ✅** | **+1.45% ✅** |
| weight+KV | +17.55% | +7.68% | +5.48% ❌ |
- **KV: d2=1 (0.5 bit/elem) makes ALL u4 configs robust** incl. mg8 (+2.06%); d2=2 (+1.45%) beats rotation.
- **weight: d2=2 makes u4/mg4 (+2.83%)·mg2 (+2.09%) robust — first lever to crack weight u4.**
- Best bit-efficiency of any lever (block=8 needs 0.75 b/elem for KV only; d2=1 needs 0.5 b/elem and also helps weight).

## Lever 6 — two-level d2=2 + rotation: cracks W+A (the last holdout)
| scope (d2=2) | mode | u4/mg8 | u4/mg4 | u4/mg2 |
|---|---|---|---|---|
| weight | +rot | +3.12% | +2.52% ✅ | +1.53% ✅ |
| act | +rot | +1.95% ✅ | +1.90% ✅ | +1.30% ✅ |
| **weight+act** | no-rot | +6.44% | +5.37% | +3.87% ❌ |
| **weight+act** | **+rot** | +5.51% | +4.49% | **+2.75% ✅** |
- **W+A u4/mg2 becomes robust only with d2=2 + rotation (+2.75%)** — the rotation's ~1pp (mostly from
  *activation*-rotation: act-only 2.80→1.95) pushes it over. mg4/mg8 still fail.
- Note the **reversal**: rotation *hurt* weight at d2=0 but *helps* at d2=2 — the per-2 microexponent
  already captures the in-block outlier, so rotation's flattening is no longer destructive (levers go
  conflict → complementary).

## Final recipe — minimum lever for u=4 robustness, per scope
| scope | u4 robust 최소 레시피 | robust mg range | best PPL |
|---|---|---|---|
| **KV** | two-level **d2=1** | all (incl. mg8) | mg8 +2.06% |
| **act** | two-level **d2=2** | all | mg8 +1.95% (+rot) |
| **weight** | two-level **d2=2** | mg ≤ 4 | mg2 +1.53% (+rot) |
| **weight+act** | two-level **d2=2 + rotation** | mg2 only | +2.75% |

(weight+KV: d2=2 reaches +5.48% (mg8) / ~+3.0% (mg2, borderline); a d2=2+rot pass at mg2 is likely but
untested.)

## Verdict
At block=32, plain E8M0 cannot make u=4 robust outside KV/mg2. The decisive lever is **MX-style
two-level scaling** (a cheap per-sub-block microexponent): **d2=1** robustifies all of KV-u4, **d2=2**
robustifies weight-u4 (mg≤4) and activation-u4, and **d2=2 + QuaRot rotation** finally robustifies
**W+A-u4** (mg2). What does *not* work: per-tensor scaling (absorbed by per-block E8M0), E4M3 scale on
activations (range collapse → use UE5M3), and weight rotation alone (in-block outliers). Block-size
reduction works for KV but is less bit-efficient than two-level. The hardest scope throughout is **W+A**
(compounded activation error), robust only at the least-aggressive sharing (mg2). Rotation's value is
**scope-specific** — large for channel-outlier KV/activations, negative-then-neutral for in-block weights.
