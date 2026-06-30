# Offline calibration for two-tier MSAQ — KV and weight

All of two-tier's data-dependent choices are made **once per model, offline**, on a small calibration
set, and baked into a **static config** that runtime simply applies (no decisions on the hot path —
standard PTQ). This doc collects the KV-scope and weight-scope calibration into one place.

## What calibration produces (the artifacts)

| artifact | KV scope | weight scope | why |
|---|---|---|---|
| **residual values** | **DC mean** — data-independent, **no calibration** | **A-weighted** — needs per-Linear within-group Gram `H_g = AᵀA` | KV's large headroom makes DC enough (Step 1); weight has less headroom and needs the downstream-error-optimal shared (the §4 objective, calibrated) |
| **rotation** | fixed **H128 Hadamard** on K/Q (not data-dependent) | n/a | kills Key's per-channel outliers; decisive (+6.6pp), folds ~free into `kv_append` |
| **precision map** | per-layer (or per-head) rung assignment | per-layer (or per-Linear) rung assignment | the allocation — spend bits where sensitivity is high |

**Key distinction:** KV calibration is *only* the allocation probe (residual is DC). Weight
calibration is *Hessian + allocation* (residual is A-weighted). Both are offline; neither touches the
runtime hot path.

## The calibration pipeline

1. **Calibration data** — wikitext-2 **train**, 16×2048 tokens (disjoint from eval; PTQ assumes it is
   representative). Model-specific, run once.
2. **[weight only] within-group Gram** `H_g` per Linear — one forward pass with hooks accumulating
   the 32×32 diagonal Gram blocks of `AᵀA` along the contraction axis (`collect_hessian`). ~0.5–1.8 MB
   per Linear. Used by the A-weighted residual (`two_tier_aware` / `two_tier_mxplus`).
3. **Sensitivity probe** — put one unit (layer / head / Linear) on the cheap rung, the rest on the
   quality rung; measure ΔPPL. Yields a sensitivity profile. (Ranking-only, ~12 windows.)
4. **Allocation** — rank least→most sensitive; assign rungs to minimize ΣΔPPL s.t. Σbytes ≤ ceiling
   (greedy here; a knapsack over the full rung set is the refinement). Validate with the ACTUAL joint
   PPL of the chosen map (interactions, not summed first-order).
5. **Output** — a static map `{layer (or head) → rung}` (+ the per-Linear `H_g` for weight). Shipped
   with the model config, like KVTuner's offline table.

## Allocation results — both scopes beat uniform; KV converts to capacity, weight to size

Llama-3.1-8B, wikitext-2, BF16 PPL 5.6877, gate ≤3%. Two rungs: cheap = MX+ E2M1+u3 gs32 (4.75b),
quality = E2M3 (6.25b).

**KV (per-layer)** — allocation beats any uniform rung at matched bytes by ~0.6pp. Using the minimal
cheap rung (4.406b = E2M1+MX+, no residual) the curve shifts ~0.4b cheaper:

| avg b/elem | allocation ΔPPL | capacity vs E2M3 |
|--:|--:|--|
| 5.098 | **+1.30%** | 0.816× bytes → **+22.5%** |
| 4.867 | +1.65% | 0.78× → **+28%** |

A **3-rung hybrid** (adding an INT8-MSAQ mid rung between two-tier-low and E2M3-high, motivated by the
regime-split where INT8 wins above ~5.3b) does NOT help: at matched bytes it ties or loses to the
2-rung (e.g. 5.1b: 2-rung +1.30% vs 3-rung +1.57%). The allocation is **bimodal** — sensitive layers
want full E2M3, tolerant layers want the cheap rung, nobody wants the middle (E2M3 at 6.25b already
beats INT8 at 6.3b). So two-tier-low + E2M3-high is optimal; the design stays a clean 2-rung.

**KV (head-wise)** beats per-layer by a further ~0.15pp at matched bytes (head 6,7 sensitive, 0,1
tolerant — ~3× spread). Modest; needs per-head byte-offset packing.

| avg b/elem | per-layer | head-wise |
|--:|--:|--:|
| 5.875 | +0.77% | **+0.62%** |
| 5.500 | +1.51% | **+1.34%** |
| 5.125 | +2.04% | **+1.94%** |

**Weight (per-layer)** — allocation also beats uniform, and crucially makes **sub-E2M3 usable**, which
the uniform weight gate could not:

| avg b/elem | allocation ΔPPL | uniform @ bytes | usable? |
|--:|--:|--|:--:|
| 5.875 | +1.48% | ~+2.3% | ✓ |
| 5.500 | **+2.41%** | +3.16% | **✓ (uniform was not)** |
| 5.125 | +3.52% | +4.02% | ✗ |

→ allocation lowers the weight robust floor from ~6.0–6.25b (uniform) to **~5.3–5.5b** = ~12% smaller
weights / weight-bandwidth at usable accuracy. (Weight sensitivity: last layer 31 most sensitive,
mid-layers tolerant. KV sensitivity: early layers 1/3/7 most sensitive.)

## Granularity guidance

- **Per-layer** is the default — simple, uniform bytes/block per layer, paging is trivial.
- **Head-wise (KV)** adds ~0.15pp at matched bytes — take it only if the per-head byte-offset packing
  is worth the gain. **Per-Linear (weight)** is the finer weight analog (untested here).
- The probe here used a **separable** (layer + head) prior for head-wise; a full per-cell probe or a
  knapsack over the full two-tier ladder (not just 2 rungs) would extract more.

## Runtime (where the static config lives in serving)

Offline calibration → `{unit → rung}` map (+ weight `H_g`). At **engine init**, each layer's KV cache
backend is configured with its rung; `get_kv_cache_shape` returns per-layer packed bytes; the pool
sizes `num_gpu_blocks` from the **sum** of per-layer bytes/token → lower average bytes = more blocks =
more concurrent sequences = higher RPS. At **runtime**, prefill encodes K (rotated)/V to each unit's
rung and writes the paged cache; decode reads + attends at that rung (the fused kernel). **No
calibration, no precision decision, on the hot path** — only a static dispatch. Integration point:
`vllm_phase2_design.md` (vLLM V1 backend, paged packed layout).

Provenance: `precision/two_tier_aware_ppl.py` (Hessian), `precision/kv_ladder_step2_alloc_ppl.py`
(KV per-layer), `precision/kv_ladder_headwise_ppl.py` (KV head-wise),
`precision/two_tier_weight_alloc_ppl.py` (weight), `two_tier_summary.md` (synthesis).
