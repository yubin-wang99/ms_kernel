# Phase 2 design — sub-byte MSAQ KV backend for vLLM

Goal: push vLLM's KV cache **below fp8** (u4 4.5b / u3·gs16 5.44b) so the capacity→RPS win of
Phase 0 (fp8 = 2× pool) extends further — the "better than fp8" contribution. This doc is the
**port analysis + layout spec + de-risked ladder**, written before any kernel code.

## 1. Can we port all our MSAQ kernels 1:1?  — No.

**Inventory** (from `csrc/`, `ms_lib/ops.py`):
- **KV production** (Phase 2 candidates): `kv_decode_attention_batched` (decode), `kv_append` (write),
  `kv_write` (pack/encode). Plus variants `msfp8_kv_*` (E3M4), `mxint8_kv_*` (baseline).
- **Not the KV backend**: `wonly_gemm`/`wa_gemm`/`mxint8_gemm`/`msfp8_gemm`/`quant_act` = weight/activation
  quant → **Phase 1** (LinearMethod). `kv_kdot_*`, `qk_wmma`/`pv_wmma` = benchmark probes / TC primitives.

**4 structural mismatches** (measured against vLLM 0.23 V1 backend contract):

| # | ours | vLLM requires | port work |
|---|---|---|---|
| 1 | dense per-seq: `ks += b*Hkv*NB*Lcap`, token by `pos` stride `Lcap` | **paged blocks** gathered via `block_table` [n_seqs, max_blocks] | rewrite kernel inner loop to walk block_table (biggest change) |
| 2 | **3 planes** (upper / shared / scale_exp) as separate tensors | **one tensor/layer**, shape from `get_kv_cache_shape(num_blocks, block_size, n_kv_heads, head_size)` | pack 3 planes into one **block-aligned byte tensor**; unpack in-kernel |
| 3 | `kv_append` writes dense `pos` (stride `Lcap`) | `reshape_and_cache` using **`slot_mapping`** (physical slot per new token) | rewrite: per-token quantize → scatter into paged slot |
| 4 | **decode only** (M=1 GEMV, non-causal "attend all Lk") | **prefill** too (chunked, causal, varlen via `query_start_loc`/`seq_lens`) | new paged-quant prefill, or hybrid (current chunk bf16) |

**Reusable**: the compute core (q·K, softmax, P·V math; the sub-byte packing bit-tricks).
**Must re-implement**: the entire memory layer (paging, single-tensor packing, slot write) + a prefill path.
→ Not "all kernels, 1:1." Pick **one** production config and re-implement the memory layer around it.

## 2. vLLM backend contract (what we must satisfy) — vLLM 0.23 V1

- `AttentionBackend` subclass: `get_name`, `get_impl_cls`, `get_metadata_cls`, `get_builder_cls`,
  `get_kv_cache_shape(num_blocks, block_size, n_kv_heads, head_size)`, `swap_blocks`, `copy_blocks`.
- Must declare our dtype in `supported_kv_cache_dtypes` (currently only auto/fp16/bf16/fp8 per backend).
- Decode/prefill metadata handed to the kernel: `block_table`, `slot_mapping`, `seq_lens`,
  `query_start_loc`, `max_seq_len`. CUDA-graph compatibility expected for decode.
- `cache_engine` allocates the per-layer tensor from `get_kv_cache_shape` → our packed byte layout
  defines bytes/token, hence num_gpu_blocks (the capacity lever Phase 0 measured).

## 3. Packed single-tensor block layout (resolves mismatch #2)

One **uint8** tensor per layer, block-granular so paging/`copy_blocks` work on whole blocks:
`[num_blocks, 2 (K,V), block_size, n_kv_heads, bytes_per_token_per_head]`
(or a flat byte span per block; exact field order is an impl detail, but it MUST be block-aligned).

Per token·head bytes (head_dim=128), for u-bit shared + (8−u)-bit upper + E8M0 scale per group `gs`:
- `upper`  = `head_dim * (8−u) / 8` bytes
- `shared` = `(head_dim/gs) * u / 8` bytes  (sub-byte, packed across the group dim)
- `scale`  = `(head_dim/gs) * 1` byte (E8M0 per group)

| config | upper | shared | scale | **bytes/tok/head** | vs fp8 (128B) |
|---|--:|--:|--:|--:|--:|
| u4 / gs16 | 64 | 4 | 8 | **76** | 0.59× (pool 1.68×) |
| u3 / gs16 | 80 | 6 | 8 | **94**† | 0.73× (pool 1.36×) |

†approx; u3 shared is 3 bits × 8 groups = 3 bytes → re-derive exactly at impl. Point: **clearly < fp8's
128B/tok/head → bigger pool than fp8** (Phase 0 fp8 was 2× bf16; u4 ≈ 3.4× bf16 ≈ 1.7× fp8).

Favorable fact: head_dim is already grouped in `gs`/32 in our encoder, and vLLM `block_size` (default 16)
maps cleanly to a token-block — so block-alignment is natural. (Mismatch is the TOKEN paging, not grouping.)

## 4. De-risked ladder (correctness gate before throughput)

1. **Layout + `get_kv_cache_shape` + pack/unpack roundtrip test** (fork-independent foundation).
2. **`reshape_and_cache`** (paged quantized write via `slot_mapping`) — correctness vs `kv_write` reference.
3. **Decode kernel** adapted to `block_table` gather — **correctness gate**: greedy generation matches
   bf16/fp8 within tolerance on a short prompt (the plan's "decode kernel must be correct end-to-end").
4. **Prefill** (the fork — see §5).
5. **Register backend**, rerun the Pareto vs fp8 (extend `vllm_phase0_serving.py` with `--kv-dtype msaq-u4`).
   Pair with the **accuracy-vs-bits Pareto** (else "just use fp8/INT4").

## 5. Open forks (decide before kernel code)

- **Which config first**: `u4/gs16` (simplest, nibble-aligned upper, biggest pool 1.68× fp8) vs
  `u3/gs16` (better accuracy, smaller pool). Recommend u4/gs16 for the first correctness pass.
- **Prefill strategy**:
  (a) **decode-only first** — keep prefill KV in bf16, quantize-on-evict to paged store; simplest, gets a
      correctness + Pareto result fastest; (b) **hybrid** — current chunk bf16, past blocks quantized;
  (c) **full paged-quant prefill** — most work, needed for the long-context capacity win at prefill.
  Recommend (a) → (c).

## 6. Effort / risk
2–4 weeks; backend memory layer + prefill are the risk. Phase 0 already proved the capacity mechanism
(fp8 lower bound); Phase 2 quantifies MSAQ beyond fp8. Correctness gate (§4.3) before any perf claim.
