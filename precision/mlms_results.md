# Multi-level mantissa sharing — robustness verification + multi-level-MS (MSAQ-multi)

Two questions:
1. Does **multi-level** sharing (hierarchical: different granularity per bit-significance) stay
   within the 3% BF16-PPL bar at a **lower bits/elem** than **single-level** can?
2. Does lifting our **MSAQ** idea onto the multi-level structure (**MSAQ-multi**) beat naive-multi?

Defs in `mlms_quant.py`; PPL sweep in `mlms_boundary.py` (wikitext-2, Llama-3.1-8B).

## Mechanism (MSAQ-multi, designed for minimal inference time)
Decode is fixed to the **cheapest** form `(q_upper·2^Σb + residual_shared)·s` — identical cost to
naive-multi (integer shift-adds + 1 FP mul). Only the **encode** carries the MSAQ idea:
1. Quantize the unshared upper with its own scale, **round-to-nearest** (not truncate): `q_upper`.
2. Centered signed residual `q_r = q − q_upper·2^Σb`, quantized to a clean Σb-bit signed int
   (full range → no coverage gap).
3. Run the **same proven bit-plane hierarchical sharing** on the residual.

## 1. Does multi-level reach lower bpe robustly? — NO (all 3 scopes)
The bits/elem floor is set by the number of **unshared** bits (sum_ml): the minimum bpe for sum_ml
shared bits is `8.25 − sum_ml·(31/32)` (all shared bits at mg=32). Going below single-level's floor
**requires sum_ml ≥ 4 (≤4 unshared bits)**, and 4 unshared bits is too coarse — independent of how
the granularity is distributed across levels.

| scope | single-level frontier | best multi-level below it | result |
|---|---|---|---|
| **KV**  (most tolerant) | u3/mg32 = 5.344 (+1.24%) | [2,2]/[32,2] 5.31 → **naive +3.45%, MSAQ +14.4%** | FAIL |
| **weight** | u3/mg8 = 5.625 (+3.00%) | [2,2]/[8,4] 5.00 → **naive +74%** | FAIL (catastrophic) |
| **activation** (strictest) | u3/mg4 = 6.000 (+2.43%) | [2,1]/[16,16] 5.44 → naive +7.05%, **MSAQ +4.69%** | FAIL |

- **MSB-fine helps but isn't enough**: keeping high shared bits fine (small mg) + LSBs coarse is the
  best aggressive shape ([2,2]/[32,2] = KV +3.45%, closest), but it only reaches 5.31 (0.03 below the
  floor) and still fails. The mg2 level costs ~1.0 bpe, so a fine high bit is expensive — the only way
  to lower bpe is to share a 4th bit coarsely, which breaks accuracy.
- **At ISO-bpe, multi-level is slightly WORSE than single-level**: KV 5.344 single u3/mg32 +1.24% vs
  multi [2,1]/[32,32] naive +2.19% / MSAQ +1.86%. Splitting the shared bits into levels **fragments**
  the sharing (separate quantization boundaries per level) → loses to single-level's joint sharing.

**⇒ Multi-level does not extend the robust frontier. The most aggressive robust config remains
single-level u3 (KV 5.344, the result we already have a kernel for).**

## 2. MSAQ-multi vs naive-multi — MIXED, structure-dependent
- **Helps** when the top shared level is **1 bit** ([3,1], [1,2,1]) or on **activations** (outliers):
  act [2,1]/[16,16] MSAQ +4.69% vs naive +7.05% (**+2.36pp**); KV [2,1]/[32,32] +0.33pp; weight [2,1] +0.1dB QSNR.
- **Hurts** when the top shared level is **2 bits** ([2,2]): the residual's sign bit gets shared over a
  large group → sign flips. KV [2,2]/[32,2] MSAQ +14.4% vs naive +3.45% (**−10.9pp**); act [2,2] −23pp.
- So MSAQ-multi's centered-residual benefit (real for activations, as in single-level) is **negated by
  residual-sign sharing** in 2-bit top levels. It is not a uniform win.

## Conclusion
Hierarchical multi-level sharing gives **finer granularity control within a fixed unshared-bit budget**
but **cannot push below the unshared-bit-count floor**, and at equal bpe it loses to single-level's
joint sharing. MSAQ-multi helps only in narrow cases (1-bit top level / activations). **Net: no
lower-bpe robust win over single-level MSAQ — a documented negative.** The aggressive-robust operating
point stays single-level u3 (KV 5.344). Kernel/inference-time exploration therefore targets single-level
configs, with attention to packing-friendliness (next phase).

## Reproduce
- `mlms_quant.py` — naive_ml (ssnf bit-plane) / msaq_ml (residual successive signed sharing) + weight QSNR.
- `mlms_boundary.py [weight|activation|kv]` — per-scope PPL sweep vs the 3% bar (`MLMS_WIN` windows).
- Raw: `mlms_kv*.txt`, `mlms_act.txt`, `mlms_weight.txt`.
