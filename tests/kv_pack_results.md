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

## Optimization 1: int8-staged V Pass-2 (`MS_KV_V8`, default on for u4/gs≤2)
Diag showed KV-decode is latency/occupancy-bound (both MSAQ and MXINT8 sustain only ~326 GB/s,
~35% of peak), and that the u4/gs2 gap over u4/gs8 is the **V-staging shared plane**: the wide
kernel stages raw packed V (upper+shared) into smem and re-unpacks per (kk,kd) in Pass-2; for gs2
the shared plane is 4× bigger (SB=8 vs 2) → more smem (caps occupancy) + 2 shared bytes/elem read.

Fix: for u4, stage V **already reconstructed as int8 codes** (`up·16+sh`, in [-120,119] since the
encoder clamps q_upper∈[-7,7]) during Pass-1. Pass-2 then reads **one int8/elem** — identical to
MXINT8's Pass-2, no bfe, no gs shared-plane. Bit-exact (test_kv 72/72).

| config | before | after (v8) | speedup | vs MXINT8 |
|---|---|---|---|---|
| u4/gs2 H8 Lk16k  | 1.45× | **1.25×** | 1.16× | still >1 |
| u4/gs2 H16 Lk16k | 1.49× | **1.15×** | 1.29× | still >1 |
| u4/gs8 (SB=2) | 1.17× | 1.13× (hurts) | — | gated OFF (extra smem not worth it) |

Cuts the u4/gs2 gap from ~1.45–1.49× to ~1.15–1.25×; gated to gs≤2 (helps only when the shared
plane is large). Not yet under MXINT8 — the remaining gap is the Pass-1 K extraction/compute
(separated-scale dequant, next).

## Optimization 2: separated-scale K dot (`MS_KV_SEPSC`, default on for u4)
Instead of reconstructing a single code `(up·2^u+sh)` per element then scaling, factor the dot:
`Σ_d q·(up·2^u+sh)·s = s·(2^u·Σ_d q·up + Σ_g sh·qg)` where `qg[g]=Σ_{d∈group} q[d]` is the
query group-sum — **key-independent, precomputed once** in smem. The shared term becomes per-GROUP
(16 ops for gs2 vs 32) and both scales fold to the block level. Within rel-Frobenius tol (test_kv 72/72).

Effect on u4/gs2 (on top of v8): a further ~4%.
| config | v8 only | v8+sepsc | MXINT8 |
|---|---|---|---|
| u4/gs2 H8 Lk16k  | 1.14× | **1.09×** | 1.00× |
| u4/gs2 H16 Lk16k | 1.06× | **1.02×** (parity) | 1.00× |

## Where u4/gs2 KV-read lands (honest result)
v8 + separated-scale took u4/gs2 from the original **1.45–1.50× down to ~1.02–1.13× MXINT8** — a
**near-tie** (parity at H16), not a strict win. Why no strict win:
- KV-decode is **latency/occupancy-bound, not bandwidth-bound** at every size and batch tested (both
  MSAQ and MXINT8 sustain only 65–330 GB/s, ≤35% of the 3090's ~900 GB/s peak; batched B=32 still
  ~130 GB/s). So MSAQ's 0.76× **byte savings never convert to time**.
- The residual gap is the **irreducible bit-extraction** (bfe per element for K, plus the int8 V
  reconstruct) that a plain-int8 MXINT8 read does not pay. More splits help MXINT8 *more* (it scales
  better with occupancy), so tuning `MS_KV_SPLIT_MULT` doesn't flip it (best H8 = mult4, 1.05×).
- GQA (Hq32/Hkv8) makes it worse: the wide path reads each KV head once **per query head** (4×),
  amplifying MSAQ's per-read cost (batched ratio 1.07→1.26× as B grows).
This matches the project's documented KV-read history (fundamental obstacle → fair tie). The one
untried avenue is the design-A GQA kernel (reads+unpacks KV **once** per key, reused across the G
query rows) — there MSAQ's extraction would amortize over G; the v8/sepsc levers were applied to the
wide kernel, not that path.

## Optimization 3 (attempted): port v8+sepsc into the design-A GQA kernel
The design-A GQA kernel (`MS_KV_GQA=1`) reads+unpacks each KV once per key and reuses it across the
G=4 query heads — the regime where MSAQ's extraction should amortize. Ported both levers into it.

| Lk | MXINT8 | wide (v8+sepsc) | gqaA base | gqaA +v8 | gqaA +v8+sepsc |
|---|---|---|---|---|---|
| 4096  | 106µs | **0.97× (WIN)** | 1.55× | 1.50× | 1.45× |
| 16384 | 344µs | 1.15× | 1.42× | 1.35× | 1.38× |

- v8 helps design-A (1.55→1.50, 1.42→1.35) and sepsc helps only at short Lk (1.50→1.45); at long Lk
  sepsc **hurts** design-A (1.35→1.38) — as expected, it breaks design-A's combined-w amortization
  over G (the shared term becomes per-(g,group) = G·ng work). So sepsc is default-OFF for GQA.
- **But design-A stays far behind the wide kernel** (1.45× vs 0.97× at Lk4096). Its scalar +
  full-chunk-staging form is occupancy/latency-bound (~25× off roofline per the launcher note);
  v8/sepsc don't fix that structural cost. Realizing design-A's KV-reuse roofline needs the deeper
  cp.async double-buffer + small-MMA rewrite — out of scope for these two levers.

**Net of the whole effort:** the **wide kernel with v8+separated-scale** is the winner — it reaches
**0.97× MXINT8 (a real win) at GQA Lk4096**, and ~1.15× at Lk16384. The original u4/gs2 was 1.45–1.50×.

## Profiling the optimized wide kernel (ncu) + occupancy attempt
ncu on u4/gs2 wide (v8+sepsc, H8/Lk16384, 148µs) pinned the limiter:
- **DRAM throughput 20.7%** → NOT bandwidth-bound (so the 0.76× byte savings can't convert here).
- **L1/TEX cache 63.5%** → shared-memory traffic is the top resource (the staged-V Pass-2 reads).
- **Achieved occupancy 24% / theoretical 33%, limited by REGISTERS** (4 blocks/SM; smem allows 5,
  warps allow 12). The kernel needs ~128 regs for its accumulators.

Tried `__launch_bounds__(256,3)` to cap registers (~85) and raise occupancy → **backfired**
(GQA Lk4096 0.97→1.09×, Lk16384 1.15→1.28×, H16 1.02→1.15×): capping forced **spills to local
memory**, which add exactly the L1/TEX/DRAM traffic that is already the bottleneck. Reverted.

**Conclusion:** the kernel is **L1/TEX-shared-bound and register-heavy**, not bandwidth- or
occupancy-bound in a way reg-capping can fix. The v8+separated-scale state is the practical optimum
(0.97× MX at GQA Lk4096; ~1.03–1.15× at long context). Further gains would need a structural change
that cuts shared-memory traffic (e.g. transposed/padded V staging for conflict-free vectorized
Pass-2 reads, or cp.async pipelining) — larger, higher-risk, uncertain payoff.

## Takeaway
The inference-time lever is **nibble alignment (u4), not minimal bits/elem**. The most-aggressive
robust config (u3) is extraction-bound and squanders its footprint advantage. A less-aggressive,
packing-friendly config (u4/gs2 for KV) delivers the clearer speedup — exactly the user's hypothesis.
The remaining gap to a true KV-read win over MXINT8 is the per-element bit-extraction itself, which
only vanishes at nibble alignment (u4), where accuracy is robust *only* with small groups (gs2).
