# Serving-metrics report — MSAQ vs MXINT8 vs bf16, by phase

Our results re-labeled in the standard LLM-serving metric frame (TTFT / TPOT / throughput / E2E),
split by **prefill** (compute-bound) and **decode** (memory-bound) phase. Llama-3.1-8B 32L,
L_in=1024, RTX PRO 4000 Blackwell. Deployed quant configs: S1 weight u3/gs16, S3 KV u4/gs16,
S5 W+A+KV u2/gs8 (KV u4-equiv). Source: `harness_perscope_results_260625*.jsonl`.

---

## 1. Metric frame & mapping to our harness

| phase | axis | standard metric | our measurement |
|---|---|---|---|
| **prefill** | latency | **TTFT** (Time To First Token) | `ttft` — full prompt forward, cuda.Event min-of-N |
| prefill | throughput | **prompt throughput** (input tok/s) | `B·L_in·1000 / ttft` |
| **decode** | latency | **TPOT** = ITL = TBT (Time Per Output Token) | `decode_ms / L_out` (CUDA-graph per-step curve) |
| decode | throughput | **output throughput** (output tok/s) | `B·L_out·1000 / decode_ms` |
| request | latency | **E2E latency** = TTFT + (N_out−1)·TPOT | `total_ms = ttft + decode_ms` |
| system | throughput | **RPS** (requests/s) | `B·1000 / total_ms` |

**Phase principle:** prefill is **compute-bound** (GEMM + causal SDPA); decode is **memory-bound**
(streams weights + KV per token). Quantization shrinks *bytes moved*, so its gains land on the
**decode** metrics (TPOT, output-tok/s), while **prefill (TTFT) is a format-tie** (weights dequant to
bf16 for cuBLAS; KV not read in prefill). This is why KV-quant must be read in decode-phase metrics.

---

## 2. Showcase — S3 KV-only (the KV-quant result), B=32

**Decode-heavy (L_in=1024, L_out=512):**

| metric | phase | bf16 | MXINT8 | **MSAQ** | MSAQ vs MX | MSAQ vs bf16 |
|---|---|--:|--:|--:|--:|--:|
| **TTFT** (ms) | prefill·lat | 7685 | 7775 | 7739 | 1.00× (tie) | 1.01× (tie) |
| prompt tok/s | prefill·thru | 4264 | 4214 | 4234 | tie | tie |
| **TPOT** (ms/tok) | decode·lat | 126.9 | 59.3 | **43.8** | **0.74×** (1.36× faster) | 0.35× (2.9× faster) |
| **output tok/s** | decode·thru | 252 | 539 | **731** | **1.36×** | 2.90× |
| **E2E latency** (ms) | request·lat | 72679 | 38155 | **30156** | **0.79×** (−21%) | 0.41× (−59%) |
| **RPS** (req/s) | system·thru | 0.440 | 0.839 | **1.061** | **1.26×** | 2.41× |

**Reading:** TTFT/prompt-throughput are a **dead tie** (KV-quant doesn't touch prefill). The win is
entirely in the **decode** column: **TPOT 1.36× lower** and **output throughput 1.36× higher** than
MXINT8 — that's the honest KV-quant headline. It propagates to **E2E latency −21%** and **RPS 1.26×**
because at L_out=512 the decode phase dominates the request.

**Prefill-heavy (L_out=128)** — same TTFT tie, decode win intact, but request-level metrics diluted:

| metric | bf16 | MXINT8 | **MSAQ** | MSAQ vs MX |
|---|--:|--:|--:|--:|
| TPOT (ms/tok) | 112.5 | 55.0 | **41.7** | **0.76×** |
| output tok/s | 284 | 582 | **768** | **1.32×** |
| RPS (req/s) | 1.452 | 2.169 | **2.458** | **1.13×** |
| E2E latency (ms) | 22042 | 14753 | **13021** | **0.88×** |

The decode metrics (TPOT, output-tok/s) barely move with L_out — they're the *true* KV-quant signal
(~1.3× over MX). RPS shrinks 1.32→1.13× only because prefill (a tie) is 57% of this shorter request.

---

## 3. Decode-phase win across scopes (B=32, L_out=512)

The metrics that isolate each quant axis — **TPOT** and **output throughput** (MSAQ/MXINT8):

| scope | quant axis | TPOT mq / mx (ms) | TPOT ratio | output tok/s mq / mx | ratio |
|---|---|--:|--:|--:|--:|
| S1 W-only | weight | 119.4 / 127.6 | 0.94× | 268 / 251 | 1.07× |
| **S3 KV-only** | **KV** | **43.8 / 59.3** | **0.74×** | **731 / 539** | **1.36×** |
| S5 W+A+KV | full | 51.4 / 59.7 | 0.86× | 623 / 536 | 1.16× |

KV quant gives the largest decode-phase win (1.36×); weight quant is modest (1.07×, the bf16-weight
GEMV byte saving); full-quant is KV-dominated (1.16×). **All three are TTFT-tied** (~7.6 s prefill).

---

## 4. Latency vs throughput — and the SLO caveat (goodput)

- **Latency knobs**: TTFT (responsiveness) and TPOT/ITL (streaming speed, must beat ~5–10 tok/s human
  reading). At **B=1** (latency-optimal), S3 L512: TPOT msaq 27.2 / mx 28.3 / bf16 28.9 ms — nearly
  tied (decode is launch/latency-bound at B=1, so KV-quant's byte win doesn't convert; cf. nibble
  analysis). KV-quant's decode win **appears only at B≥8**, where decode becomes bandwidth-bound.
- **Throughput knobs**: output-tok/s and RPS rise with batch B (more sequences amortize the weight
  load) — at the cost of higher TTFT/TPOT (queueing). The standard presentation is a throughput-vs-
  latency curve.
- **Goodput** (throughput meeting an SLO) is the honest serving metric: our numbers are **offline /
  saturated static-batch** (no continuous batching, no queueing) — an *upper bound* on goodput, not
  SLO-constrained online throughput. Stated as such in `RPS_results.md`.

---

## 5. Takeaways

1. **Report quant gains on decode-phase metrics (TPOT, output tok/s).** That's where memory-bound
   quantization lands. S3 KV: **TPOT 1.36× / output-tok/s 1.36×** over MXINT8 — the clean headline.
2. **TTFT (prefill) is a format-tie** — compute-bound, weights dequant to bf16, KV unused. Don't
   expect (or report) prefill-phase quant wins.
3. **RPS/E2E dilute the decode win by the prefill fraction.** They look smaller (1.13–1.26×) the more
   prefill-heavy the workload (short generation). Use them for request-level framing, but pair with
   the decode-phase metrics that carry the real signal.
4. **B=1 ties on TPOT** (launch-bound); the decode-throughput win is a **B≥8 / bandwidth-bound**
   phenomenon — exactly the batched-serving regime where it matters.
