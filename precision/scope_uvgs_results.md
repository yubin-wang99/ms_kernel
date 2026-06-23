# Per-scope max-aggressive robust (u, gs) — plain MSAQ, ≤3.5% wikitext PPL

For each E2E scope, the **most aggressive `(u, gs)`** (fewest bits/elem) that keeps wikitext-2 PPL within
**3.5%** of BF16, using **plain MSAQ-signed (block=32, NO rotation / NO two-level)**. Sweep `u∈{2,3,4}`
(S2/S5: `{2,3}` — W+A decides at u≤3), `gs∈{2,4,8,16,32}`, gs ascending with early-stop (aggressiveness
is monotonic in gs and u). Llama-3.1-8B, **BF16 PPL = 6.5684**, 30 windows. Script `scope_uvgs_sweep.py`,
log `scope_uvgs_sweep.txt`.

`u` = per-element *unshared* upper bits (↑u ⇒ fewer bytes), `gs` = shared-code group size (↑gs ⇒ coarser
shared scale, fewer bytes). bits/elem = `(UB + SB + 1)·8/32` (upper + shared + E8M0 scale).

## Result — most-aggressive robust per scope
| scope | max-agg robust `(u,gs)` | PPL Δ | bits/elem | u=4 robust? | first FAIL |
|---|---|---|---|---|---|
| **S1 weight** | **u3/gs16** | +3.36% | **5.50** | ✗ (u4/gs2 +6.04) | u3/gs32 +3.53 |
| **S2 weight+act** | **u2/gs8** | +1.59% | 6.50 | ✗ (u3/gs4 +4.22) | u3/gs4 +4.22 |
| **S3 KV** | **u3/gs16** | +1.35% | **5.50** | ✓ (**u4/gs2 +2.89%**) | u4/gs4 +4.26 |
| **S4 weight+KV** | **u2/gs8** | +1.12% | 6.50 | ✗ (u3/gs4 +4.22) | u3/gs4 +4.22 |
| **S5 weight+act+KV** | **u2/gs8** | +1.93% | 6.50 | ✗ (u3/gs2 +4.68) | u3/gs2 +4.68 |

(S1/S3 reach 5.50 b/elem at u3; S2/S4/S5 are weight/activation-bound at u2 → 6.50 b/elem. Only **KV (S3)
tolerates u=4** — the nibble config, u4/gs2 +2.89%.)

## Notes
- **Aggressiveness ladder.** S3 KV is the most quant-tolerant (u4/gs2 OK; u3 to gs32 all ≤+1.35%). The
  weight scopes cap at u3 (S1) and the W+A scopes at u2 (S2/S5) — activation error compounds, so adding A
  drops the robust ceiling from u3 (weight-only) to u2.
- **Why u4 fails outside KV (plain MSAQ).** Weight/activation u4 (3 unshared bits + shared) is too coarse
  for block=32 E8M0 — exactly what the rotation / MX two-level levers fix (see `u4_robustness_study.md`,
  `rot_results.md`); those are **not** applied here (this is the plain-format ceiling).
- **E2E latency config (per the u=4 rule):** run each scope at its robust config, but **where u=4 is
  robust use the u=4 nibble** (kernel-optimal, the vpack/batched kernels win with it). → S1 `u3/gs16`,
  S2 `u2/gs8`, **S3 `u4/gs2`**, S4 `u2/gs8`, S5 `u2/gs8`.

## PPL measurement method = teacher forcing (standard HF sliding-window NLL)
PPL here is the **teacher-forced** language-model perplexity, NOT free-running/autoregressive generation:
- One forward per window (`window=2048`, `stride=1024`, ~30 windows of wikitext-2 test); each position's
  next-token cross-entropy conditions on the **ground-truth previous tokens** (`model(inp, labels=tgt)`).
  Errors do **not** compound across steps (unlike autoregressive decode).
- The sliding-window overlap is masked (`tgt[:, :-trg] = -100`) so only the new (non-overlap) tokens are
  scored; `PPL = exp(Σ NLL / Σ tokens)`. This is the canonical, literature-comparable recipe.
- **Quantization is applied inside that teacher-forced forward**: weights in-place, activations via a
  `Linear.forward` patch, KV via an `F.scaled_dot_product_attention` patch. So KV quant is measured over
  the **full-context (prefill-style) forward of all positions at once**, not a step-by-step decode
  trajectory — the standard way to score a quantization format's accuracy.
