# KV-cache quantization — analysis (MSAQ on Blackwell)

Consolidated analysis of MSAQ KV-cache quantization: **where the time goes (encode vs read)**,
the read-kernel optimization, the accuracy budget, the RPS impact, and the scale-block-size study.
GPU: NVIDIA RTX PRO 4000 Blackwell (sm_120), 24 GB. Model: Llama-3.1-8B 32L. Measured 2026-06-28.

`mq`=MSAQ, `mx`=MXINT8, `bf`=bf16. Deployed KV config (S3): **u4/gs16, vpack**.

---

## TL;DR

1. **Runtime KV-quant ENCODE overhead is negligible.** prefill `kv_write` = 0.62% of prefill;
   decode `kv_append` = ~3–7% of `kv_read`. The KV-quant time cost is almost entirely the **READ**
   (dequant during attention), which is the kernel we optimized — encode is not an RPS drag.
2. **KV-read kernel is at the byte-roofline** after opening the vpack gate to all u4 gs:
   B32 KV-read mq/mx **0.52** at gs16 (was 0.83 on the slow fallback; gs2 = 0.63).
3. **RPS win dilutes** from the 1.92× kernel win to 1.27× request-RPS (L512) — prefill tie +
   bf16 weight-GEMVs are format-neutral. Bigger RPS win needs decode-heavy/long-context workloads.
4. **Scale block-size 64 does NOT help speed** (KV is per-element L1TEX-bound, not per-block/scale
   bound) and costs accuracy (u4 +2.41pp PPL). Not worth it.

---

## 1. Cost structure — where KV-quant time goes (encode vs read)

Runtime quantization = the extra passes that *encode* bf16 → MSAQ at inference time (weights are
offline; KV is not). Measured per-op, isolated, B=32, ctx=1024, u4/gs16:

| op | what | absolute | share |
|---|---|---|---|
| **prefill `kv_write`** | encode Lp=1024 tokens → cache | 735 µs/plane → **47.1 ms** (K+V × 32L) | **0.62%** of prefill (7629 ms) |
| **decode `kv_append`** | encode 1 new token → cache | 9.9 µs/plane → 631 µs/step (K+V × 32L) | **~3–7%** of `kv_read` |
| **decode `kv_read`** (attend) | dequant + attention, full ctx | 293.8 µs/layer → **9.40 ms/step** (× 32L) | the KV-quant cost that matters |

**Q: is prefill `kv_write` hidden behind other ops?** No — it runs serially on the default stream
(no overlap). But at 0.62% it is dwarfed by the compute-bound prefill (GEMM + causal SDPA); an A/B
(no-op `write_prefill`) couldn't even resolve it above the ±1.4% prefill run-to-run noise. Note the
prefill attention uses bf16 SDPA on the *un-quantized* projected K/V, so `kv_write` exists purely to
populate the cache for decode. The fraction *shrinks* further at long context (attention grows ∝ Lp²,
write ∝ Lp).

**Conclusion:** the runtime KV-quant ENCODE is not a meaningful RPS drag (prefill 0.62%, decode
~1–2% of step). Optimization effort is correctly spent on the READ (293.8 µs/layer), not the encode.
Driver: `scratchpad/kv_quant_overhead.py`.

---

## 2. KV-read kernel — vpack gate → byte-roofline

`vpack` (fast transposed-packed nibble V staging) was gated to `u4 && gs<=2`; gs>2 fell to a slow
plain-staging fallback. Opening the gate to **all u4 gs** (`csrc/kv_attention.cu`, single-token +
batched: `vpack = ((int)u==4)`) is a free win. Isolated KV-read microbench (`tests/kv_batch_bench.py`,
B32, Lk4096, bit-exact, MXINT8 & gs2 unchanged):

| config | B32 mq/mx (before) | B32 mq/mx (after) | byte ratio |
|---|--:|--:|--:|
| gs2 (vpack, old default) | 0.63 | 0.63 | 0.76× |
| gs8 | 0.85 (fallback) | **0.529** | 0.58× |
| **gs16** | 0.83 (fallback) | **0.517** | 0.545× |

At gs16 MSAQ reads at **higher** useful BW than MXINT8 (98 vs 93 GB/s) AND fewer bytes → 0.52 ≈ the
0.545 byte-roofline. **ncu** (gs16, B16): DRAM **17%**, L1TEX **70%** (bottleneck), occ 37% (shared-mem
caps at 6 blk). The kernel is L1TEX/shared-bound, not DRAM-bound.

### 2b. Does nibble alignment matter? — modest (~4%), not decisive

Measured u4 (nibble) vs u3/u2 (straddle) at the SAME gs16 (only the unshared-field width differs),
KV decode, Lk4096:

| | u2/gs16 (26 B/blk) | u3/gs16 (22 B/blk) | u4/gs16 (18 B/blk, nibble) |
|---|--:|--:|--:|
| B=1  | 57.4 µs (0.45) | 57.4 µs (0.45) | 59.5 µs (0.46) |
| B=16 | 898.7 µs (0.59) | 859.8 µs (0.57) | **823.9 µs (0.55)** |
| B=32 | 1622.6 µs (0.55) | 1596.7 µs (0.54) | **1536.0 µs (0.52)** |

Unlike weight read (byte-flat — see `nibble_analysis.md`), KV decode **is** read/byte-sensitive: at
B≥16 the time is **monotonic in bytes** (u4 18B < u3 22B < u2 26B), so the read is on the critical
path. **But nibble alignment per se is only a ~4% edge**: u4 vs u3 at B32 = 1536 vs 1597 µs (3.8%),
and the 18% byte saving converts only ~20% (kernel is L1TEX/staging-bound, not pure-byte-bound). The
**v8/vt int8-staging closed most of the straddle penalty**, so u3 (straddle) nearly matches u4
(nibble/vpack). At B=1 (launch-bound) nibble is even marginally *slower*. So the real KV levers are
**byte count + staging quality (vpack/v8/vt)**; nibble unpack is a ~4% bonus, not decisive.

---

## 3. Accuracy budget (gs sweep)

Wikitext-2 PPL, Llama-3.1-8B base, KV K+V quantized u4 + H128 rotation (`scratchpad/kv_gs_ppl.py`):

| gs | byte ratio vs MXINT8 | ΔPPL vs bf16 |
|--:|--:|--:|
| 2 | 0.758 | +1.39% |
| 4 | 0.636 | +1.99% |
| 8 | 0.576 | +2.64% |
| 16 | 0.545 | +2.58% |
| 32 | 0.545 | +2.91% |

gs16 (SB=1) is the low-byte sweet spot. **Deployed: gs16** — trades +1.2pp PPL (vs gs2) for the
0.63→0.52 read win. PERSCOPE_CFG["S3 KV-only"] = (4,16).

---

## 4. RPS impact and dilution

The isolated KV-read kernel win (1.92× = 1/0.52) dilutes through the full pipeline. E2E per-scope
(`tests/e2e_perscope_260625.py`, gs16), S3 KV-only RPS mq/mx:

| workload | B=8 | B=16 | B=32 | decode-tok/s B32 |
|---|--:|--:|--:|--:|
| L_out=128 (prefill-heavy) | 1.17× | 1.10× | 1.13× | — |
| L_out=512 (decode-heavy) | 1.27× | 1.17× | **1.27×** | **1.36×** |

Dilution chain (B32): **kernel 1.92×** → (bf16 weight-GEMVs neutral) **decode-tok/s 1.36×** →
(prefill tie) **RPS 1.27× (L512) / 1.13× (L128)**. KV-quant only shows in RPS when decode-heavy /
long-context; the lever for a bigger RPS win is the *workload*, not the kernel (kernel is at roofline).
Full tables: `RPS_results.md`.

---

## 5. Scale block-size 32 vs 64 — does NOT speed up the kernel

Hypothesis: block=64 (E8M0 scale per 64 elems, not 32) could speed the kernel via halved scale plane /
cache-line / per-block overhead. **Disproven for both KV and weight:**

- **Byte saving is only the scale plane**: ~2.8% (KV u4/gs16) / ~2.3% (weight). UB/SB are per-element
  identical; only the scale halves (−0.0156 B/elem).
- **KV is per-element L1TEX-bound, not per-block bound.** ncu (KV u4/gs16, B16): DRAM 15%, L1TEX 73%,
  **shared_ld 21.6M > global_ld 13.4M**. The bottleneck = per-element shared loads (q_sh + staged V),
  which block size does NOT change (same element count). The only per-block saving (scale reads) lives
  in the idle DRAM (15%) → block=64 KV speedup ≈ **0**.
- **Weight GEMV** is per-element too (weight bytes + straddle unpack) → block=64 ≤ ~2%.
- **Accuracy cost** grows with aggressiveness: weight u2 +0.13pp (≈free), u3 +0.8–0.9pp, u4 +2.41pp.
  KV uses u4 → +2.4pp for ~0% speed. **Not worth it.**

Block-size only pays off if a kernel is per-block/scale-bound or DRAM-bound (e.g. a future tensor-core
flash-decode). Current kernels are not. Drivers: `scratchpad/kv_prof.py`, `scratchpad/block64_weight_ppl.py`.

---

## Reproduce

```bash
cd ~/ms_kernel && source .venv/bin/activate
# cost structure (encode vs read):
CUDA_VISIBLE_DEVICES=N PYTHONPATH=. python scratchpad/kv_quant_overhead.py
# read-kernel ratio sweep:
CUDA_VISIBLE_DEVICES=N PYTHONPATH=. python tests/kv_batch_bench.py
# accuracy (gs): scratchpad/kv_gs_ppl.py   |   block32-vs-64: scratchpad/block64_weight_ppl.py
# E2E RPS: tests/e2e_perscope_260625.py --Bs 1,8,16,32 --lout {128,512}; then tests/rps_report.py
```
