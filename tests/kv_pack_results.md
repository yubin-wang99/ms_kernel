# Packing-friendliness vs aggressiveness — actual kernel inference time

Precision verification fixed the aggressive-robust operating point at **single-level u3** (multi-level
gave no lower-bpe win — see `precision/mlms_results.md`). The kernels (`wonly_gemv_wide`,
`kv_decode_attention`) already parametrize `(u, gs)`, so this phase just drives them across the robust
configs to answer the user's question: **does a less-aggressive but packing-friendly config beat the
most-aggressive robust config on real time?** Answer: **yes — footprint does not predict speed; nibble
alignment does.** (RTX 3090, `tests/gemv_u_bench.py`, `tests/kv_pack_bench.py`.)

## Why u4 is special
Unshared bytes/block = `32·(8−u)/8`. Only **u4 → 4 bits/elem = a clean nibble** (2 per byte). u3 (5
bits/elem) and u2 (6) pack fields that **straddle byte boundaries** → per-element shift/mask on the
critical path. So u4 unpack is essentially free (memory-bound); u3/u2 are extraction-bound.

## Weight GEMV (W-only, gs8)
| config | footprint | time vs MX | effective BW | bound |
|---|---|---|---|---|
| u4 (nibble) | 0.58× | **0.58–0.60×** | ~440 GB/s (= MXINT8) | memory-bound — FULL conversion |
| u3 (robust for weight) | 0.70× | 0.83–0.90× | ~346 GB/s | extraction-bound — partial |
| u2 | 0.79× | 0.83–0.88× | ~400 GB/s | extraction-bound |

u4 fully converts bytes→time, but **u4 is not robust for weight** (3 unshared bits too coarse), so the
robust weight config (u3) only yields ~10–17% speedup (extraction-bound).

## KV decode (H8/16, Lk16384, D128)
| config | footprint | time vs MX | accuracy (wikitext-2, 3% bar) |
|---|---|---|---|
| MXINT8 | 1.00× | 1.00× | exact |
| u2/gs32 | 0.79× | 1.64–1.67× | robust ~+0.5% |
| **u3/gs32 (most-aggressive robust)** | **0.67× (smallest)** | **1.59–1.61× (slow)** | +1.24% |
| u3/gs8 | 0.70× | 1.64–1.68× | ~+2% |
| **u4/gs2 (nibble, packing-friendly)** | 0.76× (larger) | **1.32× (fast)** | +2.72% |
| u4/gs8 (nibble) | 0.58× | 1.02× (≈MXINT8) | NOT robust (+5%) |

- **Footprint ⊥ speed**: u3/gs32 has the *smallest* robust footprint (0.67×) yet is among the
  *slowest* (1.59×); the larger but **nibble u4/gs2 is much faster (1.32×)**. u3's 5-bit extraction
  throttles to ~140 GB/s vs u4's ~190–210 GB/s.
- All MSAQ KV configs remain **slower than MXINT8** (extraction overhead a plain-int8 read avoids;
  consistent with the Phase-32 KV-read tie/loss). Among MSAQ, the packing-friendly **u4/gs2 is the
  best robust operating point**, not the bpe-minimal u3/gs32.

## Takeaway
The inference-time lever is **nibble alignment (u4), not minimal bits/elem**. The most-aggressive
robust config (u3) is extraction-bound and squanders its footprint advantage. A less-aggressive,
packing-friendly config (u4/gs2 for KV) delivers the clearer speedup — exactly the user's hypothesis.
The remaining gap to a true KV-read win over MXINT8 is the per-element bit-extraction itself, which
only vanishes at nibble alignment (u4), where accuracy is robust *only* with small groups (gs2).
