# Full VQ as the stored 2nd-level residual (free-index successor)

Code: `vq_residual.py`, raw output `vq_residual_qsnr.txt`. Successor to `index_residual.md`, whose
verdict (H1 rejected; only STORED bits recover the residual) pointed here.

## 1. Mechanism

The free-index gate showed no base-derived index recovers the residual, and only stored bits do — the
strongest being VQ. Two flavors:
- **Scalar VQ** (per-element, g=1): store one index/element selecting a scalar correction. Cost
  log2(K) b/elem + a full per-element add. This *is* the `index_residual` oracle — no cheaper than just
  adding mantissa bits, and it triggers caveat C2 (full second GEMM).
- **Vector VQ** (per-group, g>1): store one index per group of g elements selecting a g-dim residual
  codeword. Cost **log2(K)/g b/elem** — the cheap sub-bit regime scalar VQ can't reach. This only pays
  beyond scalar if the residual VECTOR has exploitable intra-group **structure** (correlation/low rank).

Codebook entries learned offline (k-means); the element becomes base (1st) + VQ correction (2nd).

## 2. Experiment

On real Llama-3.1-8B weights (q_proj, down_proj), in the **ulp-normalized** residual domain (C6),
base ∈ {FP4,INT4,FP6,INT6}:
1. **Structure diagnostic** — PCA energy concentration of the g-dim residual (top-1/2/4 PC vs the white
   reference 1/g), effective rank/g (participation ratio), adjacent-element correlation. White residual
   ⇒ vector VQ cannot beat scalar via structure, and the correction cannot be low-rank-folded (C2).
2. **Recovery** — vector-VQ QSNR gain over `zero` (base only) across a (g,K) grid → storage =
   log2(K)/g ∈ [0.06, 2.0] b/elem; vs per-element scalar VQ (≥1 b/elem) and vs the §5 mantissa-sharing
   incumbent (from `em_sharing_results.md`) at iso-storage. Codebook learned in-distribution per tensor
   (⇒ an upper bound on a real global/per-scope table; C3 deferred).

## 3. Results

### 3a. Structure — the residual is WHITE (no exploitable group structure)

Identical across all bases and both layers:

| base | top-1 / top-2 / top-4 PC energy (g=8) | white ref (top-1) | eff_rank/g | adj-corr |
|--|--|--|--|--|
| FP4 | 12.8% / 25.5% / 50.6% | 12.5% | 1.00 | +0.000 |
| INT4 | 12.9% / 25.8% / 50.8% | 12.5% | 1.00 | +0.001 |
| FP6 / INT6 | ~12.6% / ~25.3% / ~50.3% | 12.5% | 1.00 | ~0.000 |

Every PC carries ≈ 1/g of the energy, effective rank = g, neighbor correlation = 0. The residual vector
is **white noise** — no dominant direction, no low-rank, no correlation. (Consistent with the zero-mean,
~white residual seen throughout EM_sharing / free-index.)

### 3b. Recovery — vector VQ BEATS both incumbents iso-storage (FP4 q_proj; zero = 17.88 dB)

| storage b/elem | mantissa-share Δ (§5) | scalar VQ Δ | **vector VQ Δ** | best VQ cfg |
|--:|--:|--:|--:|--|
| 0.25 | +0.77 | — (min 1b) | **+1.29** | g32/K256 |
| 0.75 | +2.19 | — | **+3.71** | g8/K64 |
| 1.00 | — | +4.39 | **+5.09** | g8/K256 |
| 1.50 | +4.59 | — | **+7.42** | g4/K64 |
| 2.00 | — | +9.31 | **+10.23** | g4/K256 |

- **Vector VQ > mantissa-sharing** at every iso-bit (+0.5 dB @0.25b → +2.8 dB @1.5b). Unlike the DC-mean
  / free-index approaches (doomed by the zero-mean residual), VQ tiles the residual distribution
  *directly*, so zero-mean is irrelevant to it. **This is the first residual scheme to clearly pass.**
- **Vector VQ > scalar VQ** at iso-bit (+0.70 dB @1b, +0.92 @2b) — but this is exactly the universal
  **iid space-filling VQ gain** (~0.5–1.5 dB for white sources), NOT structure exploitation (3a says
  there is none). All bases/layers show the same (INT4: +1.36 @1b; down_proj near-identical).

### 3c. The E2M3 wall still caps weight

FP4(4.25) + 2.0b vector VQ = 6.25 b/elem → 17.88+10.23 = **28.1 dB** < native E2M3 (6.25b, **30.0 dB**).
So on weight, the same wall as `two_tier_results.md` holds: VQ is the best *fractional-band* (4.25–6.25b)
filler but cannot beat the hardware-native E2M3 rung per-bit.

## 4. Verdict — precision PASS, but the win is locked behind compute (C2)

- **Precision gate PASSES.** Vector VQ is a genuinely better residual quantizer than both the §5
  mantissa-sharing incumbent (+0.5…+2.8 dB iso-bit) and scalar VQ (+~0.8 dB iid VQ gain), with monotone
  recovery across all fractional storage. It is immune to the zero-mean failure that sank EM_sharing and
  the free index.
- **But the structure diagnostic is the catch.** The residual is perfectly white (eff_rank/g = 1.00,
  adj-corr = 0). Consequences:
  1. VQ's edge over scalar is *only* the iid space-filling gain — there is no structural free lunch.
  2. White + full-rank ⇒ the correction is **irreducibly per-element with NO low-rank factorization**:
     it cannot fold into a 1/g-FLOP correction GEMM. It is a full per-element second GEMM / per-element
     LUT decode — **caveat C2 realized, ~2× compute.** The precision win relocates the entire problem to
     compute; it does not avoid it.
- **C2 escape (decides where VQ lives).** A per-element LUT *can* fold into the **dequant** as a
  non-uniform remap — but only for **INT4/INT6** (software dequant); FP4/FP6 native tensorcores do
  uniform dequant, so folding there forfeits the native path. So VQ's precision win is realizable cheaply
  **only on INT bases.**

**Strategic consequence.** Vector VQ is the best fractional-band residual tool found so far, but on
*weight* it is double-capped (E2M3 wall per-bit; C2 ~2× compute on FP native). It survives cleanly only
where both caps lift: **(i) INT4/INT6 base** (LUT folds into software dequant, C2 escaped) **(ii) KV
scope** (no cheap E2M3 wall; the fractional 4.6–5.7b band is the KV allocator's habitat per
`kv_ladder_design.md`). That intersection — **INT-base vector-VQ residual in the KV fractional band** —
is the configuration to take forward; it is the first place the residual idea passes precision *without*
the win being eaten by compute or the native wall. (Next, if pursued: PPL gate there + the C2 dequant-fold
kernel sketch.)
