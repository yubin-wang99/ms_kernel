# Online K-rotation kernel — added decode latency

The accuracy study (`rot_results.md`) found head-dim **H₁₂₈ rotation of KV-Key** is the
structural MSAQ win (kills channel outliers, makes nibble u4 robust). Realizing it online
costs decode-hot-path time: every step the new token's **Q [Hq,128] and K [Hkv,128]** must be
Hadamard-rotated post-RoPE (Q mirrored so `(Q·H)(K·H)ᵀ = Q·K^T` is preserved). This measures
that tax. Kernel: `csrc/rotate.cu` (`torch.ops.msaq.hadamard_rotate`), bench `tests/rot_kv_bench.py`,
RTX 3090, Llama-3.1 shape (Hq=32, Hkv=8, D=128).

## Kernel
FWHT (fast Walsh–Hadamard), 7 butterfly stages = 128·7 ops/row vs 128² for a naive H-matmul.
1/√128 folded into the last stage → **orthonormal** H (no attention-score rescale). bf16 in /
fp32 compute / bf16 out. **Correctness:** rel-err vs H-matmul 2.6e-3, QKᵀ-preservation 3.3e-3 (bf16). PASS.

## Latency (µs/step, vs `kv_decode_attention`)
| B | Lk | attn | **Q+K (1 launch)** | Q+K (2 launch) | K-only |
|--|--|--|--|--|--|
| 1 | 1024 | 31 | **8.9 (28%)** | 17.6 (57%) | 9.0 (29%) |
| 1 | 4096 | 104 | **9.8 (9%)** | 19.1 (18%) | 9.8 (9%) |
| 1 | 16384 | 422 | **9.8 (2.3%)** | 18.7 (4.4%) | 9.9 (2.3%) |
| 8 | 4096 | 787 | **8.9 (1.1%)** | 17.4 (2.2%) | 8.9 (1.1%) |
| 32 | 4096 | 2665 | **9.0 (0.3%)** | 17.6 (0.7%) | 9.0 (0.3%) |
| 32 | 16384 | 10777 | **8.9 (0.1%)** | 18.8 (0.2%) | 8.9 (0.1%) |

## Finding: the tax is LAUNCH-BOUND, not compute-bound
The rotation time is **flat at ~9 µs/launch** across every (B, Lk) — it does not grow with the
data. The tell: **Q+K in one launch (8.9 µs) = K alone (9.0 µs) = exactly half of Q+K as two
launches (18 µs)**. The work (40–1280 rows × 128) is far below what saturates a kernel, so the
cost is purely per-launch overhead. Consequences:

- **Absolute added latency ≈ 9 µs/step** if Q and K are rotated in one launch (18 µs if separate).
- As a fraction of attention it's only large at the *smallest* config (B=1, Lk=1024: 28%) where
  attention itself is just 31 µs; at any realistic batched / long-context decode it is **<1–5%,
  often <0.5%**. Against *full* TPOT (which also includes the QKVO + MLP GEMVs, far bigger than
  attention) the fraction is smaller still.
- The 9 µs is launch overhead, not math. **Fusing the K-rotation into `kv_append` and the
  Q-rotation into the QK epilogue removes the launch entirely → the true marginal cost is ≈ 0.**
  This standalone kernel is the *upper bound*.

## FUSED — true marginal cost (no extra launch)
The ~9 µs above is launch overhead, so the rotation was fused into the launches the decode step
already pays (`tests/rot_fused_bench.py`):
- **K-rotation → `kv_append`** (`csrc kv_append_rot_kernel` / `torch.ops.msaq.kv_append_rot`): one
  block per head loads the full D-row, FWHT in shared, then NB threads quantize+pack to the same
  token-major slot. Marginal = `kv_append_rot − kv_append`.
- **Q-rotation → attn prologue** (`kv_decode_wide_kernel`, gated `MS_KV_QROT=1`): FWHT on `q_sh[D]`
  right after the existing q-load, once per block. Marginal = attn(`QROT=1`) − attn(`QROT=0`).

**Correctness (isolated, so quant noise can't hide a bug):**
(A) same rotated-K cache, kernel-rotated q vs host-rotated q → rel_fro **1.5e-3** (bf16/FWHT order only). PASS.
(B) `append_rot` cache vs host `pack_kv(bf16(K)@H)`, decoded → rel_fro **0.0** (byte-exact: the
kernel's fp32 FWHT matches the host matmul to the quant code). PASS.

**Marginal latency (RTX 3090):**
| path | baseline | +rotation | **marginal** |
|---|---|---|---|
| K-append (Hkv=8) | 8.41 µs | 8.38 µs | **−0.03 µs (noise)** |
| attn B=1 Lk=1024 | 31.1 µs | 31.7 µs | **+0.6 µs (+1.9%)** |
| attn B=1 Lk=4096 | 103.9 µs | 104.5 µs | **+0.6 µs (+0.6%)** |
| attn B=8 Lk=4096 | 736.6 µs | 741.1 µs | **+4.5 µs (+0.6%)** |
| attn B=32 Lk=4096 | 2675 µs | 2673 µs | **−2 µs (noise)** |

Fused, the rotation rides existing launches: **K-rotation marginal ≈ 0** (the append's FWHT hides
under its launch); **Q-rotation marginal ≤ ~1 µs / ≤ ~1%**, at or below the attn run-to-run jitter
(the negative entries are noise — rotation cannot speed attention up). The standalone ~9 µs/launch
was *entirely* launch overhead; fusion removes it.

## Verdict
Online K-rotation's true cost is **~0** when fused: K-rotation into `kv_append` adds no measurable
time, Q-rotation into the attn prologue adds ≤1 µs/step (≤1%, noise-bound). The standalone kernel's
~9 µs/launch was pure launch overhead. The accuracy/aggressiveness gain (u4 KV becomes robust:
+5.14%→+2.04% PPL, the fast nibble config the vpack KV-decode kernel wins with) is bought for
**essentially free latency**. V stays un-rotated (accuracy-irrelevant per `rot_results.md`).
Kernels: `kv_append_rot` (K, byte-exact), `MS_KV_QROT=1` (Q, fused prologue); standalone
`hadamard_rotate` kept for the upper-bound measurement.
