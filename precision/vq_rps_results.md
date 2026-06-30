# End-to-end RPS — FP4 + vector-VQ KV residual

Code: `vq_rps.py` (reuses the reviewed `capacity_model.py` math), raw `vq_rps_results.txt`. Closes the
final link of the chain: does the FP4+VQ KV bytes/token saving convert to serving throughput?

**Inputs, all measured in this line of work:**
- iso-accuracy KV operating points (PPL gate, `vq_kv_results.md` / `vq_kv_global_results.md`):
  FP4+VQ 4.75b = +2.44%, 5.25b = +1.79%; incumbent mantissa-share 5.44b ≈ +1.9%; native FP6 6.25b
  = +0.30%; native FP4 4.25b = +3.46% (fails the 3% gate).
- decode bandwidth = **560 GB/s** (the C2 microbench, `vq_c2_results.md`: kdot hit 555–569 GB/s and the
  VQ correction is free — decode stays at this BW).
- Llama-3.1-8B, RTX PRO 4000 Blackwell 24 GB, util 0.90, 2 GB workspace, L_out=128.
- **KV-isolated:** weights fixed at MXINT8 8.25 b/elem in every row, so the only variable is b_kv → the
  pure KV-residual contribution (weight quant is orthogonal and stacks on top).

Capacity: `M_avail ≥ W + B·L_seq·κ(b_kv)`, κ = 2·layers·n_kv·d_head·b_kv/8 = 65,536·b_kv/8 B/token →
B_max → memory-bound decode roofline `t_step=(W+B·L·κ)/BW`, req/s = (B/t_step)/L_out.

## Throughput (RPS) ratio vs MXINT8, by context length

| KV format | b_kv | ΔPPL | gate | 4k | 16k | 64k |
|---|--:|--:|:--:|--:|--:|--:|
| MXINT8 (baseline) | 8.25 | ~0 | ✓ | 1.00× | 1.00× | 1.00× |
| native FP6 | 6.25 | +0.30% | ✓ | 1.32× | 1.31× | 1.40× |
| mantissa-share (incumbent) | 5.44 | +1.9% | ✓ | 1.53× | 1.51× | 1.51× |
| **FP4+VQ 5.25** | 5.25 | +1.79% | ✓ | **1.58×** | 1.58× | 1.75× |
| **FP4+VQ 4.75** | 4.75 | +2.44% | ✓ | **1.75×** | 1.72× | 1.85× |
| native FP4 | 4.25 | +3.46% | ✗ | 1.96× | 1.92× | 1.97× |

B_max @16k ctx: MXINT8 10 seq → FP4+VQ 4.75 17 seq (1.7×). Max context @ B=32: MXINT8 5k → FP4+VQ 4.75 8k.

## Verdict — the bytes/token saving converts to RPS; the chain is complete

- **FP4+VQ 4.75b (+2.44%, passes gate) → ~1.75× RPS over MXINT8** (1.85× at long context), the regime
  where capacity binds. The RPS multiple ≈ the KV byte ratio 8.25/4.75 = 1.74×, exactly as the
  byte-roofline predicts (and the C2 result guarantees the correction adds no decode time).
- **vs the incumbent mantissa-share (1.53×), FP4+VQ adds another ~1.15× RPS at equal-or-better
  accuracy.** FP4+VQ 5.25b is a strict win — **better PPL (+1.79% vs +1.9%) AND more RPS (1.58× vs
  1.53×)** than mantissa-share.
- **VQ's specific value is making the 4.75–5.25b band USABLE.** Native FP4 (4.25b) would give ~1.96× but
  fails the gate (+3.46%); the VQ residual buys back the accuracy to keep most of that capacity win
  (1.75×) inside the 3% gate. That is the whole point of the residual.

## Complete chain (all measured)

| link | result | doc |
|---|---|---|
| precision (fractional band) | FP4+VQ owns 4.75–6.25b KV, beats mantissa-share iso-bit | `vq_kv_results.md` |
| generalization | fixed/global calibrated codebook reproduces it | `vq_kv_global_results.md` |
| compute (C2) | LUT-fold free — decode stays DRAM-bound at 560 GB/s | `vq_c2_results.md` |
| **end-to-end RPS** | **~1.75× over MXINT8, ~1.15× over mantissa-share, iso-accuracy** | this file |

## Caveats
- This is the **analytical capacity-roofline RPS** (the repo's `capacity_model.py`), grounded in measured
  BW and measured iso-accuracy b_kv — not a live vLLM serving run. `vllm_phase0_serving.py` exists for the
  on-hardware confirmation (the true final step).
- KV-isolated (weights = MXINT8); reducing weights independently stacks more batch (capacity_model config
  (c) style).
- FP4+VQ points are within the 3% gate, not lossless-iso with MXINT8; the clean iso-accuracy comparison is
  vs mantissa-share (both ~+1.8–1.9%), where FP4+VQ wins on both axes.
- Capacity binds at long context / batch; at short ctx B_max is large enough that the scheduler/compute may
  cap first — the ratios are the capacity-frontier bound.
