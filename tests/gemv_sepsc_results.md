# separated-scale (sepsc) generalized to W-only GEMV

Applied the KV-decode separated-scale dequant to `wonly_gemv_wide` (`MS_GEMV_SEPSC`). The dot
factors `Σ_k x_k·(up_k·2^u + sh_{g(k)})·s = s·(2^u·Σ_k x_k·up_k + Σ_g sh_g·xg_g)`, where the input
group-sum `xg_g = Σ_{k∈group} x_k` is computed incrementally (running group-sum, no register array).

## Result (RTX 3090, `tests/gemv_u_bench.py` style, gs8)
| config | sepsc off | sepsc on | Δ | bound |
|---|---|---|---|---|
| **u3** (robust weight) | 0.89–0.90× MX | **0.86–0.87× MX** | **+2.7 / +5.1%** | extraction-bound |
| u2 | 0.90–0.93× | 0.89× | +2.6 / +4.5% | extraction-bound |
| u4 | 0.66× | 0.75–0.77× | **−12 / −14%** | memory-bound |

## Reading
- sepsc **helps the extraction-bound u2/u3 paths** — exactly the robust weight operating point
  (weight needs u3; u4 isn't robust). The streaming bit-buffer unpack is the limiter, and folding
  the scale to block level + making the shared term per-group trims the dot work in its shadow.
- sepsc **hurts memory-bound u4**: the kernel already runs at HBM BW (0.66× bytes → 0.66× time), so
  the extra group-sum bookkeeping adds latency that isn't hidden. → default **off for u4**, on for u2/u3.
- This confirms the general rule from KV-decode: **sepsc pays off when the kernel is compute/
  extraction-bound, not when it is memory-bound.**

## Status
`MS_GEMV_SEPSC` default on for u!=4 (the robust-weight u3 path gains ~3–5%), off for u4. Bit-exact
within GEMV tolerance (test_w 45/45). KV-decode wide kernel keeps its own v8+sepsc.
