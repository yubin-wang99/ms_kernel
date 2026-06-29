# §7-first gate — two-tier MSAQ (native MX base + shared residual), weight iso-bit PPL

**Question (spec §7-first):** does a native E2M1 base + contraction-axis shared residual push the
bits-vs-PPL frontier BELOW the native rungs (MXFP4=4.25b, E2M3=6.25b) into usable (<3% ΔPPL)
territory — E2M3-class quality cheaper than 6.25b by riding the FP4 tier + a fractional correction?

**Setup:** Llama-3.1-8B (NousResearch), wikitext-2, BF16 PPL = 5.6877. Weight-only on all
q/k/v/o/gate/up/down Linears, MX block=32 (contraction-axis aligned). A-weighted / MX+ runs calibrate
the within-group 32×32 Gram H_g = AᵀA on wikitext-2 **train** (16×2048 tok, no test leakage). Gate:
ΔPPL ≤ 3%. Code: `two_tier_ppl.py`, `two_tier_aware_ppl.py`, `two_tier_mxplus_ppl.py`,
`two_tier_gs_sweep_ppl.py`.

## Three levers, then the gs sweep that corrected the first verdict

1. **Residual objective (gs=32):** recon-L2 efb E2M1+u4 = +10.93%; **A-weighted (§4)** E2M1+u4 =
   +7.92% (−3pp; downstream-output optimum >> weight-recon). H=I reduces A-weighted exactly to the DC
   mean; with channel-importance heterogeneity it cuts output error ~20%.
2. **MX+ (block-max outlier rescue):** repurposes the outlier's exponent bits as mantissa (1→3 bits at
   E2M1). +7.19% alone @ 4.406b — more bit-efficient than the residual — and **stacks orthogonally**:
   MX+ E2M1+u4 +u2-resid = +5.28% (outlier vs DC are different error dimensions, §3 confirmed).
3. **gs sweep (THE correction).** The "u4≈u2 saturation → weight done" read was a **gs=32 artifact**
   (one shared value per block). gs = sub-group within the block; smaller gs = finer residual (+u/gs
   bits, one E8M0/block). Across ALL u, finer gs improves monotonically — saturation broken.

## Full (u,gs) Pareto frontier — best ΔPPL at each bit budget

| bits | ΔPPL% | config | owner |
|--:|--:|---|---|
| 4.250 | +12.15% | E2M1 native | native |
| 4.562 | +7.92% | E2M1+u2 gs32 | two-tier |
| 4.719 | +5.28% | MX+ E2M1+u2 gs32 | two-tier |
| 4.781 | +4.54% | MX+ E2M1+u2 gs16 | two-tier |
| 4.906 | +4.38% | MX+ E2M1+u2 gs8 | two-tier |
| 5.156 | +4.02% | MX+ E2M1+u2 gs4 | two-tier |
| 5.406 | +3.16% | MX+ E2M1+u3 gs4 | two-tier |
| 5.656 | +3.15% | MX+ E2M1+u2 gs2 | two-tier |
| **6.000** | **+2.70%** | **MXINT8-MSAQ.efb** | existing |
| 6.156 | +1.90% | MX+ E2M1+u3 gs2 | two-tier (first usable ✓) |
| **6.250** | **+0.64%** | **E2M3 native** | HW-native wall |

(Full grid in `two_tier_gs_sweep_llama31_8b.txt` + `_u23_*`. u2/u3 dominate the low-bit frontier; u4
is never Pareto below 6b — too many residual bits for the accuracy.)

## Verdict — weight gate CLOSED (negative, but fully characterized)

- **The mechanism works and is now completely mapped:** A-weighted residual + MX+ + fine gs builds a
  real Pareto frontier from 4.25b→6.25b. **two-tier OWNS the frontier in ~4.56–5.66b** — a band where
  native and MXINT8-MSAQ have NO option at all.
- **But it never reaches usable (<3%) strictly below the existing anchors.** Best sub-6b point is
  +3.15% @ 5.656b (just over the gate). The first usable two-tier point is ≥6.0b, where MXINT8-MSAQ.efb
  (6.0b, +2.70%) and **hardware-native E2M3 (6.25b, +0.64%, zero residual machinery)** already live and
  dominate. The **E2M3 wall** is decisive — confirmed robust across the full (u,gs) grid, not an
  artifact.
- So on weight there is **no usable Pareto win**: two-tier's exclusive territory is the sub-usable
  fractional band; usable accuracy means E2M3, which it cannot beat per-bit.

## Consequence (spec §7 fallback, now maximally data-justified): pivot to the KV ladder (§5)

The Pareto-owned **4.6–5.7b fractional band is the asset** — worthless on weight (crushed by the
cheap hardware-native E2M3 wall) but **exactly the KV allocator's habitat**:
- KV native also offers only discrete {4,6,8} rungs, but **has no cheap E2M3-class wall**, and KV is
  the **most quant-tolerant scope** (`scope_uvgs_results.md`: S3 KV alone tolerates u4, u4/gs2 +2.89%;
  weight caps at u3, W+A at u2).
- On KV, bytes/token = capacity → batch → RPS (S3 mq/mx 1.27× @ B32 L512, byte-roofline). A fractional
  rung between native 4 and 6 directly buys capacity; the same toolkit (A-weighted residual + MX+, now
  with the u2/u3 + fine-gs configuration proven here) lands where it could not on weight.
- KV/activations are also genuinely outlier-dominated (KVQuant/KIVI) — unlike weights — so MX+ (which
  still helped on mild weight outliers) has more leverage there.

Design: `kv_ladder_design.md`. Next: KV ladder Step 1 (rung PPL on S3), carrying the proven
configuration — A-weighted (or its online-realizable static form) + MX+, u2/u3, gs swept {2..32}.
