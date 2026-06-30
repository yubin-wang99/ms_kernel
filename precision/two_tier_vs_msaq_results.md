# two-tier (additive shared) vs MXFP-MSAQ (exponent-scaled shared) — which is better?

Both methods = **per-element E{eb}M{P} + a u-bit value shared over a group of gs**. They differ in
ONE thing: how the shared value is applied.

| | reconstruction | shared scale | u vs mb | base+correction split? |
|---|---|---|---|---|
| **MXFP-MSAQ** (`msaq_mxfp8`) | `rec_i = upper_i + sh_g · 2^(ee_i − P)` | **per-element exponent ee_i** (a sub-LSB mantissa extension) | **u ≤ mb** (shares mantissa bits) | **No** — sh entangled with each ee_i |
| **two-tier** (`quant`) | `rec_i = base_i + sh_g · d_blk` | **one per-block E8M0 d_blk** (uniform additive) | u **independent** of mb | **Yes** — `Y = AŴ + (ĀR̄)·d`, a 1/32 GEMM |

The user's framing is exact: *one multiplies the shared value by the (per-element) exponent, the
other does not.* That single choice cascades into three consequences:

1. **Bit cost.** MXFP-MSAQ rides existing per-element exponents → the shared needs **no separate
   scale**. two-tier's additive shared needs its **own E8M0** (+8/32 b/elem). So at matched
   (per-element format, u, gs) two-tier costs **0.25 b/elem MORE**.
2. **Reach.** MXFP-MSAQ can only share what the mantissa holds (**u ≤ mb**): it cannot add u2/u3/u4
   to an E2M1 base (1 mantissa bit). two-tier's additive u is **unbounded** — the entire point that
   made E2M1+u4 expressible.
3. **Kernel decomposability.** two-tier's uniform per-block scale lets the shared term factor OUT of
   the GEMM as a contraction-aligned **1/32-FLOP correction** (native base on tensor cores, no
   per-element unpack). MXFP-MSAQ's shared, multiplied by each element's exponent, **cannot** be
   pulled out — it stays inside the per-element dequant (the sub-byte unpack bottleneck of §2).

## The one root difference, made concrete — relative vs absolute correction

Both store `base_i + shared_g · (scale)`. The ONLY difference is the scale the shared rides:

```
MXFP-MSAQ:  x̂_i = base_i + sh_g · 2^(e_i − m)     scale = each element's OWN exponent e_i
two-tier :  x̂_i = base_i + sh_g · d_blk           scale = ONE per-block E8M0 d_blk
```

So MXFP-MSAQ's shared is a **relative** (per-element, magnitude-proportional) mantissa refinement;
two-tier's is an **absolute** (single-scale, same for all) additive offset.

**Worked example — an outlier block** `[16.0, 0.5, 0.4, 0.3]` (one outlier + a ~0.4 bulk), E2M1
(narrow exponent range):
1. the block E8M0 is set by the **max (16)** → the bulk (0.3–0.5) is ~2⁻⁵ of the max → its exponent
   floors at e_min → the bulk is flushed toward 0.
2. **MXFP-MSAQ correction:** the bulk elements' e_i is at the floor → their shared correction
   `sh · 2^(e_min − m)` is ~0 → **the bulk cannot be lifted. Collapse** (the +3.7e7% row below).
3. **two-tier correction:** residual `r = [0, 0.5, 0.4, 0.3]`; its own `d_blk` is set to the **bulk
   scale (~0.4)**, independent of the block max → adds ~0.4 back to the bulk → **recovered**. The
   outlier is already handled by base+MX+. **Survives.**

A relative correction couples the whole block to its largest element (one outlier inflates the scale
and starves the rest); an absolute offset carries its own scale and breaks that coupling.

### The three consequences cascade from this one choice

| | MXFP-MSAQ (relative) | two-tier (absolute) |
|---|---|---|
| **① reach (u)** | shares mantissa bits → **u ≤ mb** (no E2M1+u2…) | offset independent of base → **any u** (E2M1+u4) |
| **② bit cost** | reuses the element's exponent → **no separate scale** | needs its own E8M0 → **+0.25 b/elem** |
| **③ kernel split** | `Σ A_i·sh·2^(e_i−m)` — 2^(e_i) entangled with data → **cannot factor** → per-element unpack | `Σ A_i·(sh·d) = (Σ A_i)·sh·d = ĀR̄` → **1/32-FLOP correction GEMM**, native base, no unpack |

③ is the decisive one: only a **uniform** shared scale lets `Σ A_i·const` collapse to `(Σ A_i)·const`.
MXFP-MSAQ's per-element scale never collapses → it keeps the sub-byte unpack bottleneck two-tier
exists to escape.

## Experiment — matched per-element E2M1, weight scope, Llama-3.1-8B (BF16 PPL 5.6877)

two-tier = `quant(2,1,u,gs)` (additive, DC=recon-L2). MXFP-MSAQ = `msaq_mxfp8(u,gs,2,1+u)` (efb,
recon-L2). Both per-element E2M1; two-tier carries the +0.25b scale handicap.

| u | gs | TT bits | TT ΔPPL | MS bits | MS ΔPPL | winner |
|--:|--:|--:|--:|--:|--:|:--:|
| 1 | 32 | 4.531 | +11.45% | 4.281 | +15.85% | TT |
| 1 |  8 | 4.625 | +10.92% | 4.375 | +15.64% | TT |
| 1 |  2 | 5.000 |  +9.61% | 4.750 | **+3.7e7%** | TT (MS diverged) |
| 2 | 32 | 4.562 | +10.87% | 4.312 | +15.55% | TT |
| 2 |  8 | 4.750 |  +9.26% | 4.500 | +14.32% | TT |
| 2 |  2 | 5.500 |  +5.53% | 5.250 |  +9.54% | TT |
| 3 | 32 | 4.594 | +11.01% | 4.344 | +15.11% | TT |
| 3 |  8 | 4.875 |  +8.58% | 4.625 | +12.84% | TT |
| 3 |  2 | 6.000 |  +4.09% | 5.750 |  +8.01% | TT |

**two-tier wins every cell by ~4–4.7pp — despite spending 0.25b MORE** (iso-bit the gap is larger).

Synthetic-weight QSNR (no model) tells the *why* — it is regime-dependent:
- **Gaussian / heavy-tail** (elements at similar exponents): two-tier +1.5–2.5 dB (≈ its 0.25b edge).
- **Smooth high-dynamic-range** (log-uniform): **MXFP-MSAQ wins +0.3–0.8 dB** — exponent-scaling pays
  off when a block genuinely spans many exponents. This is its intended FP8 use case.

The PPL gap is far larger than QSNR because real weight blocks are not smooth-log-uniform: they have
**a few extreme outliers over a small bulk**. There the block E8M0 inflates, every bulk element's
stored exponent ee_i collapses to the floor, and MXFP-MSAQ's exponent-tied shared **collapses with
them — it cannot correct the bulk**. two-tier's independent residual E8M0 sets itself to the bulk
scale and survives (the u1/gs2 +3.7e7% is the extreme failure of exactly this). The quantizer emits
no inf/nan; the format is simply unable to represent these blocks at eb=2.

## Verdict

**For the MXFP4/MXFP6 (sub-fp8) regime this project targets, two-tier is the better method** — on
both axes:
- **Accuracy:** more robust on real low-eb weight distributions (independent residual scale survives
  outlier blocks where MXFP-MSAQ's exponent-tied shared collapses); wins the weight PPL across the
  matched grid even with a +0.25b handicap.
- **Kernel:** the decisive, independent reason — two-tier decomposes into a native MX base (tensor-
  core, no per-element unpack) + a 1/32-FLOP correction GEMM. MXFP-MSAQ cannot, so it keeps the
  sub-byte unpack bottleneck (`subbyte_unpack_analysis_260625.md`) the whole reformulation exists to
  escape.

**MXFP-MSAQ is not worse in general** — its exponent-scaled mantissa-sharing is the *right* tool for
its native regime (FP8 elements, eb≥3, smooth high-dynamic-range data), where it beat MXINT8
(`mxfp6_results.md`). It is simply the wrong tool for the low-eb fractional-rung goal: it needs eb≥3
to be stable, can't exceed u≤mb, and won't factor into a cheap correction GEMM. The two-tier
reformulation was the correct move for "native MXFP4/6 base + cheap fractional residual."
