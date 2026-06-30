# INT-base vector-VQ residual on KV — PPL gate

Code: `vq_kv_ppl.py`, raw `vq_kv_ppl_llama31_8b.txt`. Follows `vq_residual.md` (vector VQ passes weight
precision but is locked behind compute C2; the proposed survivor was INT-base × KV).

**Setup.** Llama-3.1-8B (NousResearch), wikitext-2, BF16 PPL = 5.6877 (30 windows), KV-only (K: H128-rot
+ MX D-block + VQ residual, fold back; V: MX T-block + VQ residual along tokens). SDPA patched to quantize
K/V on the fly (mirrors `kv_ladder_step1_ppl.py`). VQ codebook learned **in-distribution per (layer,K/V)**
on the first window and cached (upper bound on a calibrated/global table; C3). Gate: ΔPPL ≤ 3%.

## Results

| config | b/elem | PPL | ΔPPL% | gate |
|---|--:|--:|--:|:--:|
| FP4 native | 4.250 | 5.8845 | +3.46% | ✗ |
| **INT4 zero** | 4.250 | 6.1759 | **+8.58%** | ✗ |
| INT4+VQ g8/K16 | 4.750 | 6.1856 | +8.75% | ✗ |
| INT4+VQ g8/K256 | 5.250 | 6.0596 | +6.54% | ✗ |
| INT4+VQ g4/K256 | 6.250 | 5.7466 | +1.04% | ✓ |
| **FP4+VQ g8/K16** | 4.750 | 5.8264 | **+2.44%** | ✓ |
| **FP4+VQ g8/K256** | 5.250 | 5.7894 | **+1.79%** | ✓ |
| **FP4+VQ g4/K256** | 6.250 | 5.7187 | **+0.55%** | ✓ |
| FP6 native | 6.250 | 5.7049 | +0.30% | ✓ |
| INT6 zero | 6.250 | 5.7184 | +0.54% | ✓ |
| INT6+VQ g8/K256 | 7.250 | 5.7040 | +0.29% | ✓ |
| FP6+VQ g8/K16 | 6.750 | 5.6988 | +0.19% | ✓ |

Incumbent reference (`kv_ladder_step1_llama31_8b.txt`, same harness): FP4 native +3.46%; mantissa-share
MX+ E2M1+u2 gs32 (4.719b) **+3.07%**, gs2 (5.656b) **+1.88%**.

## Verdict — the asked-for INT base FAILS; FP-base VQ is the real KV survivor

**1. INT-base vector-VQ does NOT work as a fractional KV format.**
- **INT4 base is too weak on KV** (+8.58% at zero vs FP4's +3.46%): a uniform grid wastes resolution on
  the outlier dynamic range that KV has and FP's exponent absorbs. At low/fractional VQ budgets the
  residual **cannot rescue it** (g8/K16 +8.75% — no better than zero); only a 2-bit VQ at **6.25b**
  (g4/K256, +1.04%) drags it under the gate — i.e. only once it has stopped being a cheap fractional rung.
- **INT6 base is already saturated** (+0.54% at 6.25b, lossless-class), so VQ adds almost nothing
  (+0.29% at 7.25b) — not worth the bit. There is no fractional band where INT-base+VQ is the right tool:
  INT4 too weak for VQ to fix cheaply, INT6 already done.

**2. The C2 premise was scope-wrong, and fixing it hands the win to FP base.** `vq_residual.md` chose INT
because the per-element VQ LUT folds only into INT *software* dequant (FP native = uniform *tensorcore*
dequant). But **KV decode is a byte-roofline GEMV on CUDA cores, not a tensorcore MMA**
(`kv_ladder_design.md` §0) — so dequant is software for **every** base on KV, and the VQ LUT folds for FP
too. FP base is both foldable AND accuracy-strong on KV.

**3. FP4 + vector-VQ residual passes the gate and OWNS the fractional band (4.25–6.25b):**
- **4.75b: +2.44%** — beats mantissa-share MX+ at iso-bit (4.719b, +3.07%) by 0.6 pp, and beats FP4
  native (+3.46%).
- **5.25b: +1.79%** — beats the best incumbent fractional point (5.656b, +1.88%) at **fewer bits**.
- **6.25b: +0.55%** — essentially reaches native FP6 (+0.30%); on KV the E2M3 wall that capped weight is
  nearly closed by FP4+VQ.

This is the **first residual scheme in this line to pass a PPL gate, beat the incumbents iso-bit, AND
(on KV) escape C2** — but on **FP base**, not the INT base proposed. The clean takeaway: *vector-VQ
residual is a real KV win; pair it with FP4 (not INT4).*

## Caveats
- **C3 / calibration asymmetry (the main one to close next).** The VQ codebook is *learned*
  (in-distribution, cached per layer) — the numbers are an upper bound. The mantissa-share incumbent it
  beats is *data-independent* (online per-block DC), so part of FP4+VQ's edge is the calibration it
  requires. Next gate: a **fixed global/per-scope codebook** (no per-tensor fit) to test generalization,
  and a calibrated mantissa-share baseline for a fair fight.
- VQ recovery on KV is real but modest in absolute pp (the fractional band is already mild, +1–3%); the
  value is **bytes/token at fixed quality → capacity → batch → RPS** (the `kv_ladder_design.md` thesis),
  not a PPL rescue.
- C2 fold (VQ LUT into the KV-dequant GEMV) is asserted from the GEMV/byte-roofline nature of decode, not
  yet measured — a kernel sketch + NCU check is the deployment follow-up.
