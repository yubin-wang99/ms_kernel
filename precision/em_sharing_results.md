# EM_sharing — precision gate (dual hypothesis), weight calibration-free QSNR

**Idea (spec).** Two-level micro-format: native MX base (gb=32, E8M0) + contraction-axis residual
`R_hat[k,n] = m[n,⌊k/gm⌋]·2^{e[n,⌊k/ge⌋]}` with DECOUPLED granularities — u-bit signed mantissa per
`gm`, be-bit residual exponent per `ge`. Free vars {u, gm, ge, be}, storage splits `B_res = u/gm + be/ge`.
Generalizes `two_tier` (which fixed ge=32, be=8: mantissa & exponent shared one granularity).

**Dual hypothesis (spec §4).** INT base (uniform grid) → residual envelope flat → fine on MANTISSA
(gm small, ge=32, be=0). FP base (non-uniform grid) → envelope ∝ 2^elem-exp → fine on EXPONENT
(ge=8, be>0; gm=32, "normalize then coarse mantissa").

**Setup.** Llama-3.1-8B (NousResearch) layer-0 `q_proj` (4096²) + `down_proj` (4096×14336), real
weights via safetensors. Calibration-free reconstruction QSNR (spec §5 option 1), efb=2 coordinate
descent. Bit-accounting does NOT affect QSNR (only shifts the bits axis), so the verdict is
accounting-independent. Code: `em_sharing.py` (`--selftest` validates the math).

## Math validated (selftest)
- u=0 reduces EXACTLY to each native base (FP4/FP6 == two_tier u0; INT4/INT6 == native int snap), max|Δ|=0.
- efb recon-MSE monotone non-increasing in iters (5.21e-6 → 4.94e-6 over 0..3).

## Verdict — dual hypothesis FALSIFIED on weights; mantissa-fine dominates ALL bases

Iso-bit QSNR (q_proj; same pattern on down_proj):

| base | mant-fine gm8 u2 (0.25b) | exp-fine gm32 ge8 be2 u2 (0.31b) | winner |
|--|--:|--:|--|
| FP4  | **18.65** | 18.58 | mant-fine (fewer bits) |
| FP6  | **30.95** | 30.90 | mant-fine |
| INT4 | **16.63** | 16.38 | mant-fine |
| INT6 | **29.61** | 29.53 | mant-fine |

- **The residual exponent earns zero bits.** be2 ≡ be3 give *identical* QSNR (FP4: 18.79=18.79). On the
  predicted-FP-winning axis, exp-fine loses iso-bit to mantissa-fine for FP4 AND FP6.
- **No exponent allocation ever wins iso-bit**, even a "fair" per-subgroup mantissa (gm=ge): at 0.75b,
  pure mant-fine gm4 u3 = **20.07** vs balanced gm4/ge8/be2 u2 = 18.61.

## Mechanism (airtight — diagnostic, FP4 q_proj)

The dual hypothesis's *premise is true* but its *conclusion fails*, and we know exactly why:

1. **Premise holds:** FP4 residual envelope across ge=8 subgroups genuinely varies — CoV 0.534,
   median 3.2× max/min within a block. So exp-fine *should* have signal.
2. **But the primitive can't use it:** the shared mantissa is a **DC mean**. With gm=32 the single
   per-block mantissa rounds to **0 in 100% of blocks (u2), 96% (u3)** → reconstructs ≈0 residual →
   the fine exponent scales zero. That is why be2≡be3.
3. **Root cause:** the rounding residual is **zero-mean** (mean/std = 0.145). A coarse-group DC term
   captures ≈ nothing; the only way to capture a zero-mean residual is a *small* group (fine gm). The
   exponent axis is the wrong place to spend bits — there is no nonzero coarse-DC for it to scale.

So the spec's "normalize-then-coarse-mantissa" reasoning conflated *uniform magnitude* with
*capturable by a coarse DC*; normalizing fixes scale but the residual stays zero-mean, so a coarse
mantissa is dead regardless of exponent fineness.

## Consequence

- EM_sharing's NEW degree of freedom (decoupled exponent granularity, be>0/ge<32) **unlocks nothing on
  weights** — it confirms & extends `two_tier_results.md`: mantissa-sharing fineness was the only lever,
  and fixing ge=32 was correct. The weight precision gate stays CLOSED (E2M3 wall, per two_tier).
- **Where exp-fine could still pay off: KV/activations.** There the residual is plausibly NOT zero-mean
  in the same way (genuine per-token outliers, KVQuant/KIVI), and envelope variation is much stronger —
  the regime two_tier already pivoted to (`kv_ladder_design.md`, `scope_uvgs_results.md`: KV tolerates
  u4). The next EM_sharing precision test, if pursued, belongs on KV, not weights.

## Follow-up — is the DC-mean failure fixable by a richer primitive? (`em_primitives.py`)

The DC-mean washout is the root cause; we tested whether a different SHARED primitive can revive the
exponent axis on weights. Four primitives, calibration-free QSNR on FP4/INT4 q_proj:

| primitive | what's shared / per-element | result |
|--|--|--|
| **P0** DC-mean (EM_sharing) | one signed value per gm; exp per ge | exponent earns nothing (washout) |
| **P1** DC-mean ÷ free per-elem envelope | one signed value per gm, be=0 | **worse** (14.0 dB @0.75b vs P0 20.1) — still zero-mean |
| **P2** per-elem sign + shared mag × free env | sign per-elem, magnitude per gm | **dead** (12 dB) — free envelope mis-scales |
| **P3** per-elem sign + shared mag × be-exp/ge | sign per-elem, magnitude+exp shared | **revives exp-fine**, finer ge raises QSNR |

- **Mechanism confirmed by inversion:** only when the SIGN is un-shared (per-element) is the shared
  quantity (|R| envelope) non-zero-mean — and *then* the be-bit exponent finally earns bits (FP4 P3:
  ge16→ge8→ge4 = 18.51→20.50→22.83). So the spec's "fine exponent" intuition is correct *about a
  sign-separated magnitude*, not about the DC-mean primitive it was attached to.
- **But P3 only TIES plain mantissa-fine iso-bit** (~1.59b: P3 22.83 vs P0-interp 22.86) and pays
  **+1 b/elem** for the per-element sign — which destroys the sub-byte storage + 1/32-FLOP correction-GEMM
  economics that are EM_sharing's entire reason to exist. Finer ge also helps INT4 under P3
  (18.10→19.45), so even the revived primitive yields **no clean INT→mantissa / FP→exponent split**.

**Net:** the negative weight verdict is **robust across primitive choice**. Reviving the exponent axis
requires un-sharing the sign (≥1 b/elem), at which point you've abandoned sharing and still only match
the trivial mantissa-fine baseline. EM_sharing adds nothing over plain mantissa-sharing on weights.

## Extension — same primitive re-examination on KV (`em_kv_capture.py` + `em_kv_primitives.py`)

Hypothesis for the pivot: KV residual might be **non-zero-mean / strongly outlier-enveloped** (KVQuant/
KIVI), which is exactly what the DC-mean primitive needs to make the exponent axis pay. Tested on REAL
Llama-3.1-8B K/V (layer 0 + 16, 256 tok): K rotated by H128 along D (kv_ladder q_K), V along token T.

**Diagnostic — the hypothesis is FALSE. KV residual is zero-mean just like weights:**

| tensor | envelope CoV (ge8) | mean/std | gm32 washout (u2) |
|--|--:|--:|--:|
| L0 K_rot | 0.423 | 0.146 | 100% |
| L0 V_tok | 0.470 | 0.165 | 99.9% |
| L0 K_raw | 0.464 | 0.144 | 99.9% |
| (weight ref) | 0.53 | 0.145 | 100% |

The rounding residual is zero-mean by construction (mean/std ≈ 0.15, identical to weights), and KV
envelope variation is if anything **slightly weaker** than weights (CoV 0.42–0.47 < 0.53) — even raw V
with its per-token outliers. The DC washout is 100%, same as weights.

**Primitive ordering — same as weights, with one small twist.** P0 mantissa-fine dominates at low bits;
P1/P2 (free-envelope) dead; P3 (sign-separated + stored be-exp) revives the exponent axis. Iso-bit
(FP4 base, K_rot; same on V_tok and L16):

| residual budget | P0 mant-fine | P3 sign+exp-fine | winner |
|--:|--:|--:|--|
| ~1.34b | ≈22.7 (interp) | 22.45 | P0 |
| ~1.59b | ≈23.5 (interp) | **24.06** | **P3 (+~0.5 dB)** |

Unlike weights (where P3 only *tied*), on KV P3 **crosses over** mantissa-fine at the high-bit end
(≳1.5b residual), consistently across tensors/layers. BUT: (1) the driver is **not** stronger envelope
variation (KV CoV ≤ weight); the edge is small (~0.5 dB); (2) P3 still pays **+1 b/elem** for the
per-element sign, so the residual is a ~1.5–2-bit per-element quantizer, **not** a shared 1/32 term —
it abandons the sub-byte storage + correction-GEMM economics that are the whole point. At ≈5.8 b/elem
with per-element machinery you would simply use native E2M3 (6.25b) or MXINT6, which the KV ladder
already carries.

## Final verdict (weight + KV)

The EM_sharing exponent-granularity DOF does **not** pay within the sharing constraint, on **either**
scope. Root cause is scope-independent: the nearest-rounding residual is **zero-mean**, so a shared
(coarse-group) value — DC mantissa or DC×exponent — washes to ≈0, and only un-sharing the sign (≥1
b/elem, i.e. giving up sharing) lets the exponent matter, at which point EM_sharing is just an
expensive per-element format that ties (weights) or marginally beats (KV, ~0.5 dB) plain
mantissa-fine. The KV crossover is real but architecturally moot. **Plain mantissa-sharing (two_tier,
ge=32) remains the right configuration; EM_sharing's new axis is closed on weights and KV alike.**
