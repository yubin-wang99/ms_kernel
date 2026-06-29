# KV ladder design — two-tier MSAQ as the fractional alphabet for head/layer KV allocation (§5)

**Why we are here.** The §7-first weight gate is negative (`precision/two_tier_results.md`): native
E2M1 + shared residual + MX+ recovers ~7pp of FP4's 12pp gap but floors at +5.3% PPL — the 31
non-outlier elements' 1-bit mantissa is irreducible, so usable accuracy means E2M3 (6.25b), where a
residual adds ≈0. **The mechanisms all work** (A-weighted shared residual: −3pp vs recon-L2; MX+:
bit-efficient, stacks orthogonally with the residual) — they just can't beat E2M1's bulk-mantissa
floor *on weights*. This doc moves the center of gravity to **KV**, where the data says they should
land. Spec §5/§7 prescribed exactly this fallback.

**Why KV is the right home (data-justified).**
- `scope_uvgs_results.md`: **S3 KV is the most quant-tolerant scope** — the *only* one that tolerates
  u=4 (u4/gs2 = +2.89%, u3/gs16 = +1.35%, 5.50 b/elem). Weight caps at u3, W+A at u2. Maximum headroom.
- `RPS_results_forPPT.md`: S3 KV-only mq/mx = **1.27× at B32 L512** (kernel byte-roofline ≈1.9×),
  mq/bf = **2.41×**. The KV win scales with batch and L_out — it is a **capacity → batch → RPS** lever.
- The problem shape is favorable: native KV offers only the discrete rungs {4, 6, 8}. We are not
  fighting a catastrophe (as at E2M1 weight) — we are **filling between rungs**, where even modest
  fractional gains let the allocator pack heads tighter under a fixed memory ceiling.

---

## 0. Honest framing of the win — capacity, not tensor cores

Decode attention is **two GEMVs** (Q·Kᵀ over head_dim D, then P·V over tokens T), M=1 per sequence,
each sequence attending its own KV → **memory/byte-roofline-bound, not tensor-core MMA**
(`kv_attention.cu` is hand-written CUDA-core GEMV; the RPS note calls the kernel win "byte-roofline").
So the §2 "native base rides the tensor core with no unpack" argument is a **weight/prefill-GEMM**
claim and does **not** transfer to decode. The KV ladder's value is:

1. **Bytes/token granularity.** Per-head allocation over a fractional alphabet hits a tighter *average*
   bytes/token at fixed quality than discrete native rungs → more KV blocks fit the pool → larger
   batch → higher RPS (the S3 mechanism).
2. **Base-format unpack is still cheaper native.** Even in GEMV, a clean native E2M3 6-bit element
   decodes more cheaply than custom MSAQ's funnel-shift across 3 sub-byte planes
   (`subbyte_unpack_analysis_260625.md`). The residual adds only u/32 bytes and a D/32- (or T/32-)
   contraction correction, which in a memory-bound kernel is hidden if **fused** (see §4).

The correction is therefore nearly free in *bytes*; the real risk is **kernel latency** from a
separate correction GEMV (launch + sequential dependency). Decode is latency-bound (~14–23% DRAM at
small batch), so the correction MUST fuse into the existing passes — the decode analog of §7-second's
"hides in the MMA shadow" (here: hides under the base KV-read roofline). NCU-verify.

---

## 1. The ladder — output alphabet for the allocator

| rung | base (native MX) | residual | MX+ | bytes/token·head (D=128) | vs fp8 (128B) |
|---|---|---|--:|--:|--:|
| R0 | E2M1 (FP4) | — | — | 4.25·128/8 = 68 | 0.53× |
| R1 | E2M1 | +u2 / gs32 | opt | ~75 | 0.59× |
| R2 | E2M1 | +u4 / gs32 | opt | ~76 | 0.59× |
| R3 | E2M3 (FP6) | — | — | 6.25·128/8 = 100 | 0.78× |
| R4 | E2M3 | +u4 / gs32 | — | ~106 | 0.83× |

(bytes ≈ base + outlier-index(5/32) where MX+ on + (u+E8M0)/32 residual; exact field order is an impl
detail, but block-aligned for paging — see `vllm_phase2_design.md` §3.) The allocator's job: assign
each (layer, head) a rung to **minimize ΣΔPPL s.t. Σbytes ≤ ceiling**. Discrete native gives it only
{R0, R3, fp8}; the ladder gives it 5+ rungs → finer Pareto packing.

**De-risk first:** the rungs must be accuracy-monotone and *usefully spaced* on real KV. Step 1 (§5)
measures each rung's S3 PPL before any allocation logic.

---

## 2. K/V axis split — the structural crux

The two GEMVs contract on **different axes**, so K and V get different treatment.

### K — contraction = head_dim D (Q·Kᵀ)
- Native MX base needs the MX block along **D**. But **Key has per-channel (D) outliers** (KVQuant,
  KIVI both quantize Key per-channel) → naive D-blocking puts the outlier in the block and inflates
  the E8M0 scale, starving the rest.
- **Fix (already built): online head-wise Hadamard rotation H_D** on Q and K post-RoPE
  (`csrc/rotate.cu`, `torch.ops.msaq.hadamard_rotate`). `(Q·H)(K·H)ᵀ = Q·Kᵀ` preserved (verified
  3.3e-3). It spreads the D-channel outlier so native E2M3 D-blocking is robust (`rot_results.md`).
  **Cost when fused into `kv_append` (K) and the QK epilogue (Q): ≈ 0** — the standalone 9 µs is pure
  launch overhead, flat in (B, Lk) (`rot_kv_latency.md`).
- Residual shared along **D**, group gs=32 → correction **Q·K̄ᵀ** has contraction **D/32 = 4**
  (Llama D=128). MX+ rescues K's surviving outlier — and **KV is genuinely outlier-dominated** (unlike
  weights), so MX+ has *more* leverage here than the +0.7pp it gave on weight.

### V — contraction = token axis T (P·V)
- MX block along **T**; residual shared along **T** → correction **P̄·V̄** with contraction **T/32**.
- **V has weak channel outliers → no rotation needed** (KVQuant/KIVI both keep Value per-token). One
  fewer moving part than K.

---

## 3. Residual objective for KV — the genuine fork (online-write caveat)

The §7-first weight win used an **A-weighted** shared residual (output-error optimum via within-group
Gram H_g = AᵀA, calibration-offline). For KV this does **not** transfer cleanly, because **K is
written once but attended by many future Q** — there is no single "A" to optimize against at write
time. Three options, increasing fidelity/cost:

- **(a) DC mean** (data-independent). Simplest, online, zero calibration. The §7-first floor for
  DC-only was weak *on weights*, but KV's larger headroom (S3 tolerates u4) may make it sufficient.
- **(b) Calibration-static channel importance.** Offline-estimate per-(layer,head,channel) Q-energy
  (the K-side analog of H_g, but averaged over calibration Q instead of a specific one); bake it into
  the write-time residual snap. Carries the proven A-weighted machinery in its realizable static form.
- **(c) MX+ ⊕ (a or b).** The orthogonal stack that won §7-first. Recommended baseline: **(a)+MX+**,
  escalate to **(b)+MX+** only if Step-1 PPL shows DC is the bottleneck.

Decision deferred to Step-1 data. V-side is easier: P (attention probs) at the V-GEMV *is* available
when V is consumed, but V is also written once — same caveat, same options.

---

## 4. Kernel — fuse the correction into the decode passes

Entry: `csrc/kv_attention.cu` already has `kv_decode_split_kernel` (flash-decoding, grid (Hq,S)) and
`kv_decode_cpasync_kernel` (prefetches upper+shared planes via cp.async). The correction folds in with
**no extra launch**:

- **Pass 1 (scores, Q·Kᵀ):** after the base GEMV over D, add `Q̄·K̄ᵀ` where `Q̄[γ] = Σ_{d∈γ} Q[d]`
  (a D/32=4-term reduction of the already-loaded Q) and K̄ is the per-block shared residual (one u-bit
  value/group, already in the prefetched "shared" plane). 4 extra MACs/key — trivial vs the D=128 base.
- **Pass 2 (P·V):** add `P̄·V̄` with `P̄[γ] = Σ_{t∈γ} P[t]` (T/32 reduction of the scores).
- The residual bytes ride the **same cp.async prefetch** as today's shared plane → no new memory
  transaction shape. NCU gate (the §7-second analog): fused-kernel latency increase over base-only KV
  read must sit under the byte-read roofline (decode is memory-bound, so the extra MACs are free *iff*
  they don't add a sync/launch). Target: <2% TPOT at S3 B32.

Connection to **`vllm_phase2_design.md`**: the ladder is the *precision alphabet* the Phase-2 packed
paged layout must encode. Per-head rung → variable bytes/block; `get_kv_cache_shape` must size to the
allocator's chosen per-layer rung. The packed single-tensor block layout (§3 of that doc) already
plans upper/shared/scale planes — the residual is one more (u/32-byte) plane, block-aligned.

---

## 5. De-risked build ladder (what to implement, cheapest first)

1. **Rung PPL (no kernel).** Extend the teacher-forced KV-quant harness (the `F.scaled_dot_product_
   attention` patch in `scope_uvgs_sweep.py` / `rot_kv_ppl.py`) to apply two-tier to K (rotated, via
   `hadamard_rotate`) and V. **Carry the weight-gate-proven configuration** (`two_tier_results.md`):
   **MX+ base + A-weighted residual, u∈{2,3} (NOT u4 — never Pareto), gs swept {2,4,8,16,32}** (one
   E8M0/block). On weight this exact family OWNED the 4.6–5.7b Pareto band; KV is where that band is
   worth something. Measure each rung's S3 wikitext PPL. **Gate:** rungs accuracy-monotone and spaced
   enough that the allocator has real choices, AND at least one fractional rung is *usable* below the
   nearest native rung (the win weight could not get because of its E2M3 wall — KV has none).
2. **Allocation.** Offline per-(layer,head) sensitivity (KVTuner-style sweep; RateQuant rate-distortion
   for head importance) → assign rungs to minimize ΣΔPPL s.t. Σbytes ≤ ceiling. **Gate (§7-third):**
   head/layer {R0,R1,R2,R4} allocation beats **uniform E2M3** at the *same memory ceiling* on S3 PPL.
3. **Fused kernel.** Add the §4 correction to `kv_decode_*` + fold rotation into `kv_append`. **Gate:**
   NCU shows the fused correction adds <2% decode latency (hides under base KV-read roofline).
4. **E2E RPS.** Measure S3 (B32, L512) RPS of the allocated ladder vs uniform E2M3 at matched memory.
   **Gate:** ≥ the 1.27× mq/mx baseline at equal or better PPL — i.e. the fractional allocation buys
   either more batch (RPS) or better quality at the same bytes.

**Fallback within KV:** if (a)-DC rungs are too coarse, escalate to (b)-calibration-static before
adding kernel complexity. If even R1/R2 don't beat uniform E2M3 per byte, the contribution narrows to
"native-base KV (unpack-cheap) + MX+ rotation" without the fractional residual — still a real result
(cheaper unpack than custom MSAQ vpack at matched quality).

---

## Open questions to settle before Step 1
- **gs for KV residual:** the weight gs sweep proved finer gs breaks saturation and that **u2/u3 (not
  u4)** sit on the Pareto frontier. Sweep the full **gs∈{2,4,8,16,32} × u∈{2,3}** on KV too — note K's
  contraction is D=128 (4 sub-groups at gs=32, 64 at gs=2), so gs has a different cost/structure than
  the K=4096 weight case; let the data pick.
- **Residual objective (a/b/c):** weight's win needed the A-weighted (§4) objective (DC alone floored
  ~3pp higher). For KV the online-write caveat (§3) blocks per-Q optimization — start **(b)
  calibration-static channel importance + MX+** (the realizable A-weighted form), fall back to (a) DC
  only if Step-1 shows it suffices given KV's larger headroom.
- **Does MX+ on rotated K still help?** Rotation spreads outliers (reduces MX+'s target) — measure
  whether MX+ is additive *after* rotation, or whether rotation already captures that error dimension.
  (On weight MX+ was clearly Pareto-additive; rotated-K may differ.)
