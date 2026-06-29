# KV ladder — Step 3 (feasibility): can a fused kernel realize the two-tier rung without an unpack
# penalty that eats the capacity win?

Gate (kv_ladder_design §5/§4): the fused correction must add **< 2% decode latency** — hide under the
base KV-read roofline. Measured with the existing harness (`tests/kv_decode_bw_bench.py`, ncu) + the
already-quantified unpack analysis (`subbyte_unpack_analysis_260625.md`), BEFORE committing to the
full native-base decode kernel. RTX PRO 4000 Blackwell, peak BW **672 GB/s**.

## 1. The regime — decode is latency-bound at small batch; the win is at the BW-bound tail

`kv_decode_bw_bench.py` (GQA Hq32/Hkv8, D128), achieved bandwidth vs the 672 GB/s peak:

| Lk | B | MXINT8 (8.25b) | MSAQ u3/gs16 (5.5b) | MSAQ u4/gs16 (4.5b) |
|--:|--:|--|--|--|
| 4096 | 1 | 127µs / 68 GB/s (10%) | 55µs / 104 GB/s (15%) | 57µs / 82 GB/s (12%) |
| 4096 | 8 | 1384µs / 50 GB/s (7%) | 356µs / 130 GB/s (19%) | 367µs / 103 GB/s (15%) |
| 16384 | 8 | 5478µs / 51 GB/s | 1665µs / 111 GB/s | 1666µs / 91 GB/s |

- At B=1–8 decode runs at **7–19% of peak BW → deeply latency-bound**, NOT byte-roofline-bound. So a
  byte saving does not convert 1:1 to speed here; it converts to **capacity** (more KV blocks fit the
  pool → larger batch). The KV-ladder value (Steps 1–2) was established exactly at the large-batch
  end (S3 B32 L512, where §8 shows decode approaches BW-bound and the byte advantage is realized).
- **Telling:** MSAQ u3/gs16 (5.5b) is FASTER and higher-BW than u4/gs16 (4.5b) despite MORE bytes —
  the u4 nibble's straddle-free unpack is cheap, but u3's better gs/layout wins here. Unpack cost,
  not bytes, gates the small-batch regime (§9: "the only lever left is making unpack cheaper").

## 2. The correction is a SUBSET of work the current kernel already ships

The §4 correction in the decode passes is:
- Pass 1 (scores): `+ Σ_γ Q̄[γ]·K̄[γ]`, Q̄[γ]=Σ_{d∈γ}Q[d] — a **D/gs = 4-term** reduction (D=128,
  gs=32) of the already-loaded Q, against the **per-block shared residual K̄** — which the current
  kernel ALREADY loads (the `kh`/`vh` "shared plane") and ALREADY consumes per element.
- Pass 2: `+ Σ P̄·V̄`, analogous, T/gs terms.

Crucially, two-tier's shared is applied **per group (4 corrections/key/head)**, whereas the current
sub-byte MSAQ does a shared-plane combine **per element (128/key/head)**. So two-tier's correction is
**strictly less work than the per-element sharing the deployed kernel already performs** at 104–130
GB/s. The correction's marginal cost is therefore bounded above by something already shipped and fast
→ **comfortably under the 2% gate**. The residual bytes add **u/(gs·base) ≈ 3·4/(32·... ) ≈ 1.6%** of
the base byte count — negligible, and they ride the existing shared-plane prefetch (no new
transaction shape).

## 3. Native nibble base can only improve MLP vs the current sub-byte format

The bottleneck (`subbyte_unpack_analysis` §10): per-element unpack swaps memory-wait cycles for
compute and **steals memory-level parallelism** — memory-waiting warps/issue drop 18 (MXINT8) → 5.6
(MSAQ), so DRAM can't fill (Little's law). The only fix is cheaper unpack (§9).

Two-tier's cheap rung is **E2M1 base = a 4-bit nibble** → byte-aligned, **no straddle** (the §3/appendix
cheap case, ~3 ALU/elem vs ~5–7 for straddling u2/u3), PLUS the shared moves from per-element to
per-group. So two-tier's per-element unpack ALU is **≤ current sub-byte MSAQ** → MLP ≥ current → it
realizes **at least** the current byte→speed conversion, plausibly more (§11 already showed cheaper
unpack lifts MLP 5.6→8.53, +52%).

## Verdict — FEASIBLE; the full native-base decode kernel is justified

- The correction's marginal cost is **bounded by the per-element sharing the deployed kernel already
  does** (and which runs at 104–130 GB/s) → **< 2% gate met by construction**; residual bytes ≈ 1.6%.
- The native nibble base removes per-element straddle + per-element shared-combine → unpack ALU can
  only go DOWN vs the current sub-byte format → MLP up → more of the byte-floor advantage realized.
- The capacity→RPS payoff lives at the large-batch / long-context BW-bound tail (S3 B32), exactly
  where Steps 1–2 measured the ladder's value.

**Caveat (honest):** decode is GEMV (not tensor-core MMA), so even native MXFP needs *software* field
extraction — "native base = zero unpack" is a weight/prefill-GEMM claim, not a decode one. The decode
win is narrower: nibble base (no straddle) + per-group (not per-element) correction → **less** unpack
than today's format, not none. The exact speedup needs the built kernel to measure; this feasibility
pass establishes the correction fits under the gate and the direction is strictly favorable.

Next (full build, when prioritized): native-E2M1-nibble + per-group residual + MX+ decode kernel
(extend `kv_decode_cpasync_kernel`), fold rotation into `kv_append`, NCU the achieved MLP/BW vs this
projection.
