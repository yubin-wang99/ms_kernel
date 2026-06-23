# MSAQ-signed packing / unpacking — explained

How one MXINT8 value is split into three planes, how the layout differs at **u=4 (nibble)** vs **u<4**,
and exactly what reads / shifts / masks / sign-extends happen when a kernel unpacks one element.
Ground truth: `ms_lib/pack.py` (offline pack, NumPy) and `csrc/core/ms_utils.cuh` (device unpack). Block
size is the OCP MX block **`BLOCK = 32`**; a weight `[OUT, K]` has `nb = K/32` blocks per output row
(KV reuses this with `[L, D]` → blocks along `head_dim`).

## 1. The value model — one E8M0 scale + two integer codes per element

MSAQ-signed stores an int8 "word" `q = q_upper·2^u + r_shared` and one E8M0 scale per block; the value is
`x ≈ q · scale`. Two knobs:
- **`u`** = how many low bits of the 8-bit mantissa are **shared** across a group. The per-element
  **upper** code is then `wbits = 8 − u` bits (↑u ⇒ fewer per-element bits ⇒ fewer bytes, more aggressive).
- **`gs`** = group size for the shared code: one `u`-bit **shared** code per `gs` elements
  (`n_group = 32/gs` shared codes per block). ↑gs ⇒ coarser shared scale, fewer bytes.

### Decompose (`pack.decompose`, single FP rounding)
```
s_base      = 2^(floor(log2(max|x_block|)) − 6)          # E8M0 (8-bit exponent), 1 per 32-block
s_unshared  = s_base · 2^u
q_upper[k]  = clip( round(x[k] / s_unshared), ±q_max ),   q_max = 2^(7−u) − 1     # (8−u)-bit SIGNED, per element
residual    = x − q_upper · s_unshared
r_shared[g] = clip( round( mean_{k∈group g}(residual) / s_base ), ±2^(u−1) )      # u-bit SIGNED, per group
```
### Reconstruct (`pack.reconstruct` / `dequant_weight`)
```
q = q_upper[k] · 2^u + r_shared[g(k)]      # a valid MXINT8 integer word
x ≈ q · s_base                             # W-only: ×scale to float ;  W+A: feed q as int8 to IMMA
```
So per element the information lives in **3 places**: the block's E8M0 exponent, the element's `(8−u)`-bit
upper code, and its group's `u`-bit shared code.

## 2. The three planes (what gets stored where)

`pack_weight` emits three SoA planes (out-innermost; the per-block byte index is the middle axis):

| plane | dtype | shape (row-major) | holds | bytes per 32-block |
|---|---|---|---|---|
| **`scale_exp`** | int8 | `[nb, OUT]` | E8M0 exponent (`scale = 2^exp`) | **1** |
| **`upper`** | uint8 | `[nb, UB, OUT]` | 32 upper codes (`wbits` bits each), dense LSB-first | **`UB = 32·(8−u)/8`** |
| **`shared`** | uint8 | `[nb, SB, OUT]` | `n_group` shared codes (`u` bits each), dense LSB-first | **`SB = ceil(n_group·u/8)`** |

Codes are packed **dense, LSB-first** (`_pack_codes_lsb`): code `i` occupies bits `[i·width, i·width+width)`
of the block's byte run, low bit first; a code may **straddle** two bytes. In the row-major plane the two
bytes of a straddling code are **`OUT` elements apart** in memory (because OUT is innermost).

**Column-major twins** `upper_cm [nb, OUT, UB]`, `shared_cm [nb, OUT, SB]` (just a transpose, identical
bytes) put a column's `UB`/`SB` bytes **contiguous**, so the wide-load GEMV reads a whole block's upper in
one `uint4` (u4) / a few `uint32` (u2/u3) coalesced loads. **KV** uses token-major `[H, nb, L, BYTES]`
(BYTES innermost) so a key's bytes are contiguous and a warp's 32 head-dim reads at a fixed key coalesce.

### Plane sizes
`UB` depends only on `u`; `SB` on `u` and `gs` (`n_group = 32/gs`, `SB = ceil(n_group·u/8)`):

| u | wbits=8−u | **UB** = 32·wbits/8 |
|--|--|--|
| 2 | 6 | 24 |
| 3 | 5 | 20 |
| 4 | 4 | 16 |

| representative config | UB | SB | +scale | bits/elem = (UB+SB+1)·8/32 |
|--|--|--|--|--|
| u4/gs2  | 16 | ceil(16·4/8)=8 | 1 | **6.25** |
| u4/gs8  | 16 | ceil(4·4/8)=2  | 1 | **4.75** |
| u3/gs16 | 20 | ceil(2·3/8)=1  | 1 | **5.50** |
| u2/gs8  | 24 | ceil(4·2/8)=1  | 1 | **6.50** |
(MXINT8 baseline = 32 mantissa + 1 scale = 33 B/block = 8.25 bits/elem.)

## 3. u = 4 (nibble) vs u < 4 (straddle) — the central layout difference

**u = 4** → upper code = 4 bits, shared code = 4 bits → both are **nibble-aligned**: exactly 2 codes per
byte, **a code NEVER crosses a byte boundary**.
- upper byte for element `k` is `k>>1`; the nibble is `(k&1)*4`. shared byte for group `g` is `g>>1`.
- `UB = 16` (one `uint4` = 16 B covers a whole block's upper) — the fewest-bytes config and the KV/GEMV
  default. Unpack is a **single byte load + one `bfe.s32`** per plane (HW does mask+sign):
  `unpack_ms_kv_elem_u4` / the `uint4`+`bfe` path in `w_gemv.cu`, `kv_attention.cu`.

**u < 4 (u = 2, 3)** → wbits = 6 or 5 (not a power-of-2 sub-byte width) → codes are dense and **straddle
byte boundaries**. e.g. u=3 (5-bit upper): k=0 → bits 0–4 (byte 0); **k=1 → bits 5–9 → byte 0 high 3 bits +
byte 1 low 2 bits (straddle)**; etc. So unpack needs a conditional 2-byte load + shift/OR before masking.
`UB = 20` (u3) / `24` (u2). Handled by `extract_code` (random access) or the rolling bit-buffer
("streaming unpack") that yields each code with one shift+mask, refilling a 32-bit word only when low.

## 4. Unpacking one element — reads, slice, mask, sign

For element `k` of block `blk`, output column `o` (`g = k >> log2(gs)` is its group):

**(a) scale** — read **1 byte** `scale_exp[blk·OUT + o]`; `scale = exp2f(exp)`.

**(b) upper code** (`wbits` bits, from `upper`):
```
bit0 = k·wbits ;  byte = bit0>>3 ;  off = bit0&7
code = upper[base_u + byte] >> off                       # SLICE: drop the low `off` bits
if (off + wbits > 8):  code |= upper[base_u + byte+1] << (8−off)   # STRADDLE: pull high bits from next byte
code &= (1<<wbits) − 1                                    # MASK to wbits bits
up_code = (code ^ (1<<(wbits−1))) − (1<<(wbits−1))        # SIGN-EXTEND (two's complement)
```
Reads **1 byte, or 2 if it straddles** (only possible for u<4). In the row-major plane the straddle byte
is `+OUT` away; in CM/KV planes it is the adjacent byte (`+1`).

**(c) shared code** (`u` bits, from `shared`): identical pattern at `bit0 = g·u` over `SB` bytes → `sh_code`.

**(d) combine**: `word = up_code·2^u + sh_code`. **W-only**: `value = word · scale` (→ bf16). **W+A**:
`word` is fed straight to the INT8 IMMA; the two block scales fold once in the epilogue.

### u = 4 fast path (no straddle, no manual mask/sign)
```
up_code = bfe_s32( upper[base_u + (k>>1)], (k&1)*4, 4 )   # 1 byte load + 1 bfe.s32 (mask+sign in HW)
sh_code = bfe_s32( shared[base_h + (g>>1)], (g&1)*4, 4 )
word    = up_code*16 + sh_code
```
`bfe.s32 d, word, pos, len` is a single PTX bit-field-extract that masks the `len` bits at `pos` and
sign-extends — replacing the shift / conditional 2nd-byte load / mask / xor-sign of the general path.

## 5. How much is read per block (at unpack)
Over a full 32-block, a kernel touches: **`scale` 1 byte** + **`upper` UB bytes** (16 / 20 / 24 for
u4 / u3 / u2) + **`shared` SB bytes** (≤8). That `UB+SB+1` bytes per 32 elements is the MSAQ footprint —
e.g. u4/gs2 = 25 B/block = 6.25 bits/elem vs MXINT8's 33 B/block (32 mantissa + 1 scale = 8.25 bits/elem).
The fewer bytes is exactly the memory-traffic win the decode kernels convert to time.

## 6. Device pack tail (write path, inverse)
KV write/append re-pack on the GPU: `decompose_ms_block` (thread holds `x[32]`) produces `q_upper[32]`
(`(8−u)`-bit) + `r_shared[32/gs]` (`u`-bit) + E8M0 exponent (same numerics as `pack.decompose`), then
`pack_codes_lsb` writes them dense-LSB-first into the `UB`/`SB`-byte planes (the same straddle handling in
reverse: `buf[by] |= code<<off; if(off+width>8) buf[by+1] |= code>>(8−off)`). The read path
(`unpack_ms_*`) is bit-exact to this.

— files: `ms_lib/pack.py` (`decompose`, `pack_weight`, `_pack_codes_lsb`, `dequant_weight`, `weight_int8`),
`csrc/core/ms_utils.cuh` (`unpack_ms_weight_elem`, `unpack_ms_kv_elem`, `unpack_ms_kv_elem_u4`,
`extract_code`, `bfe_s32`, `sign_extend`, `decompose_ms_block`, `pack_codes_lsb`).

## 7. Profiling — is u2/u3 decode actually BW-bound? (measured)

We measured the wide-load decode kernels with Nsight Compute to test the common assumption that
"decode is DRAM-bandwidth-bound, so simplifying the unpack (e.g. removing byte straddle) cannot help."
**The assumption is false at single-request decode sizes.**

Setup: RTX 3090 (Ampere sm_86), Nsight Compute 2022.1, split-K wide-load kernels, single-request decode
(GEMV: OUT=K=4096 / KV: Lk=4680, H=8). Repro: `tests/ncu_uprobe.py` (GEMV), `tests/kv_ncu_driver.py` (KV).
Metrics are each unit's throughput as % of peak; `sector-util` = useful bytes per global-load sector
(lower ⇒ more wasted sector traffic).

| path | u | **SM%** | DRAM% | L2% | L1% | sector-util% | time(µs) |
|---|---|---|---|---|---|---|---|
| GEMV | u2/gs8 | **57.7** | 30.9 | 14.3 | 28.5 | 15.7 | 50.9 |
| GEMV | u3/gs8 | **54.6** | 27.1 | 12.6 | 28.2 | 18.3 | 50.1 |
| GEMV | u4/gs8 | 68.3 | 31.7 | 20.6 | **68.3** | 39.6 | **37.5** |
| KV   | u3   | **33.0** | 10.2 | 5.0 | 29.5 | 27.2 | 77.0 |
| KV   | u4   | 37.7 | 13.3 | 7.5 | **35.3** | 49.6 | **50.0** |

Reading:
1. **Not DRAM-bandwidth-bound.** DRAM% never exceeds ~32% (KV: 10–13%). All SOL units sit <70% → this is the
   **latency/occupancy-bound** regime (the problem is too small to saturate any unit).
2. **u2/u3 GEMV is SM-bound** (SM 54–58% is the top unit; DRAM/L1 ~28%). The streaming-unpack shift/mask/OR
   (unpack ALU+LSU) is plausibly on the critical path — ALU-leaning, **not** memory-leaning.
3. **u4 is faster everywhere** (GEMV 1.34×, KV 1.54×) with much higher sector-util. But u4's edge bundles three
   things — fewer bytes (UB=16<20/24), no straddle, AND a single-`bfe` unpack — so it over-states the value of
   straddle removal alone.
4. KV's "BW-bound" (Phase 17/18 comments) is really **effective-BW / sector-utilization** bound, not DRAM
   saturation: DRAM% is low; the limiter is the L1TEX pipe moving half-wasted sectors (sector-util 27% → 50% u4).

**Implication for the 4-plane idea (split the straddling (8−u)-bit upper into a 4-bit nibble plane + a (4−u)-bit
plane).** Because decode is SM/issue-bound (not DRAM-bound), cutting unpack instruction count is a *real* lever —
the earlier "coalesced-load split gave ZERO speedup" experiment only optimized the memory side, which was never
the wall. But the upside is bounded: the 4-plane repack keeps u2's byte count, so it captures only the
unpack-ALU share of u4's 1.34× gap (not the byte/sector share), and adds plane-count overhead. See §8 for the
prototype that measures this directly.

## 8. Prototype — 4-plane u2 repack, measured (the idea does NOT pay off)

We built the 4-plane u2 repack and benchmarked it against the 3-plane streaming u2, in one JIT harness with an
**identical** launch config so the only difference is the unpack (`tests/proto_split4_u2.py`):
- **streaming** (3-plane): `upper_cm` 24 B → 6× `uint32`, rolling 6-bit straddle extract.
- **split4** (4-plane): nibble `[16 B]` `uint4` + low-2 `[8 B]` `uint2` (both straddle-free shift+mask), then
  **reassemble** the 6-bit signed `q_upper = sign6((top4<<2)|lo2)`. Same total bytes/block.

Bit-exactness: `max|streaming−split4| = 0` (identical math); both match the fp reference to 1.6e-7 rel.

| metric (RTX 3090, ncu single launch) | streaming (3-plane) | split4 (4-plane) |
|---|---|---|
| **instructions executed** | 5.35 M | **6.37 M (+19%)** |
| sector-util% (global ld) | 15.7 | **48.3** |
| DRAM% | 61.8 | 51.4 |
| L1% | 49.1 | 36.7 |
| SM% | 50.0 | 50.9 |
| registers/thread | 39 | 40 |
| time — 300-iter bench | 33.1 µs | 32.5 µs (**1.018×**) |

**Verdict: the 4-plane split does not speed up decode (~±2%, a wash).** The hypothesis's premise — that byte
straddle is expensive — is wrong: the rolling-buffer straddle is already ~1 shift+mask per code (refill
amortized). Splitting a 6-bit code into 4+2 aligned fields forces a **per-element reassembly**
(`(top4<<2)|lo2`) from two separately-loaded registers, which costs *more* ALU than the straddle it removes —
hence **+19% instructions**, not fewer. The split's only real win is **sector utilization (15.7→48.3%)** from the
cleanly-aligned `uint4`+`uint2` loads, but at single-request decode that doesn't dominate. This is structural,
not an implementation artifact: any non-power-of-2 code split into aligned sub-fields must be reassembled, and
the reassembly is the cost. The genuine lever remains **u=4** (the upper code IS 4 bits, so no reassembly:
single `bfe`, fewest bytes, best sector-util — the 1.34× winner in §7).

## 9. "Not BW-bound" ⇒ can we drive it INTO the BW-bound region? (measured)

If single-request decode is latency/occupancy-bound (§7), the natural idea is: push it toward the BW roofline by
fetching more. The subtlety: **you can't fetch *more* — the byte count is fixed** (each weight is read exactly
once: 13.6 MB for this u2 GEMV). The real lever is not bigger fetches but **more concurrent in-flight requests**
(occupancy / split-K / MLP / batching) to hide DRAM latency so the fixed bytes stream faster. Sweeping split-K on
the §8 prototype (concurrency knob; same kernel, same bytes):

| split-K | time | DRAM% | SM% | warp-occ% | DRAM bytes |
|---|---|---|---|---|---|
| 1  | 136 µs | 11.1 | 8.5 | 16.7 | 13.64 MB |
| 8  | 27.2 µs | 59.1 | 46.5 | 25.7 | 13.70 MB |
| 32 | **25.8 µs** | **68.0** | 55.7 | 73.7 | 13.82 MB |
| — BW roofline (13.6 MB / 936 GB/s) | **14.6 µs** | 100 | — | — | — |

Reading:
1. **The intuition is right (in direction).** More concurrency drove DRAM 11→68% and time 136→26 µs (5.3×):
   latency-bound → toward BW-bound, exactly as hoped.
2. **But "fetch more" is the wrong mental model — bytes are FIXED.** `dram__bytes` is ~constant (13.64→13.82 MB)
   across all split-K; the win came from occupancy (16.7→73.7%), i.e. more *concurrent* fetches of the *same*
   bytes, not larger ones. (Per-thread width is already maxed at `uint4`=16 B; going wider isn't one instruction
   and would cut occupancy.)
3. **It plateaus at ~68% DRAM / 26 µs — ~1.8× short of the 14.6 µs roofline.** The wall is **sector utilization**:
   u2-streaming delivers only 15.7% useful bytes per 32 B sector, so the L1TEX/L2 pipe saturates moving
   half-wasted sectors before DRAM hits 100%. *This* is where "more useful bytes per transaction" (alignment —
   what split4's `uint4`+`uint2` raised to 48%) actually lifts the ceiling; it just didn't matter at B=1 where the
   kernel re-hit an instruction wall (§8).

So the levers that genuinely move the roofline (vs. just reaching it): (a) **sector-utilization** (alignment / less
waste per fetch) raises the plateau; (b) **batch B>1** reuses each weight across tokens → arithmetic intensity up →
ride the roofline until compute-bound (the repo's batched-decode path); (c) **fewer bits/elem (u4)** lowers the
roofline itself (14.6 → ~9 µs). Note the production `wonly_gemv` measured 50 µs / DRAM 31% (§7) vs this lean
prototype's 26 µs / DRAM 68% at split-K 32 — the production kernel (sepsc path, 96 regs, lower occupancy) leaves
BW on the table, so raising its occupancy / trimming per-thread ALU is real, untapped headroom.

## 10. Occupancy bump + sector-util max — why neither helps the u2 GEMV (measured)

Three follow-ups, all measured on the production/prototype u2 GEMV.

**(a) Why is per-sector useful-bytes so low (15.7%)?** Column-major `upper_cm[nb,OUT,UB]`: thread `o` owns
column `o`, its `UB` bytes contiguous. But `UB=24` is not a power-of-2 vector width, so the block loads as
6× `uint32`. Within *one* word-load instruction, lane `o` addresses `base + o*24 + i*4` → the warp's 32 lanes
are **strided by 24 bytes** (4 useful bytes each), spanning ~24 sectors of 32 B → 128 useful / 768 fetched =
**16.7%** per instruction. u4 escapes this: `UB=16` = one `uint4`, lane stride 16 = load width → 32 lanes cover
512 contiguous bytes → high util.

**(b) Maximize sector-util — `bytesplit` (the "load the block, split in registers" idea, done right).** A literal
single fat plane can't coalesce (stride 24/26 ≠ power-of-2). The fix is to split the 24 **bytes** (not the codes)
into a 16 B + 8 B plane so each plane's per-thread stride equals its load width (`uint4`=16, `uint2`=8), then run
the **same** rolling 6-bit unpack over the 6 registers — codes straddle the 16/24 boundary naturally, so **no
reassembly** (unlike split4). Measured at split-K 32:

| kernel | sector-util% | DRAM% | DRAM bytes | inst | regs | time |
|---|---|---|---|---|---|---|
| streaming (6× uint32) | 15.7 | 67.2 | 13.76 MB | 5.47 M | 39 | 26.3 µs |
| **bytesplit (uint4+uint2)** | **48.3** | 68.4 | 13.67 MB | **5.41 M** | 38 | 26.0 µs |
| split4 (semantic split) | 48.3 | 58.4 | 13.80 MB | 6.49 M | 40 | 26.1 µs |

`bytesplit` gets split4's coalescing at streaming's instruction count (bit-identical output) — mechanically the
best of both. **But time is unchanged.** Why: `dram__bytes` is the *same* for all three. The "wasted" sector bytes
in streaming's strided loads are immediately reused by the next of the 6 word-loads (all 6 words of all 32 columns
live in one 768 B run → L2 serves them once). So low sector-util inflated only L1 *request* count, not DRAM
traffic, and the kernel isn't L1-request-bound. **Sector-util is a red herring for the column-major weight GEMV**
(it bit because L2 absorbs the redundancy) — unlike KV token-major, where each thread reads a distinct key once and
the half-wasted sector is never reused (there raising sector-util is a real win).

**(c) Bump production occupancy + re-measure.** Knobs are env-only (`MS_GEMV_SPLITK_MULT`, `MS_GEMV_SEPSC`; no
rebuild). Full-op u2 timing (kernel+combine): default `MULT=16`→44.9 µs, `32`→43.8 µs, `64`→48.0 µs (over-split
hurts). ncu of the wide kernel (`MULT=32`): **warps-active 71.7%** (already high, reg-limited at 48 regs),
**SM 61.5% vs DRAM 33%**, sector-util 15.7%. So production u2 is **SM/issue-bound at high occupancy — not
occupancy-bound and not memory-bound.** Bumping split-K therefore can't help; the limiter is per-element ALU
(unpack + separated-scale accumulation). `sepsc=1` (48.8 µs) does beat `sepsc=0` (53.8 µs), as its comment claims.

**Synthesis across §7–10.** u2/u3 GEMV decode is fundamentally **SM/ALU-bound** at single-request sizes. None of
{more occupancy, better coalescing/sector-util, straddle removal} helps, because none reduces per-element ALU. The
only levers that move it are **fewer bytes + fewer unpack ops simultaneously** — i.e. **u=4** (one `bfe`, 16 B, no
reassembly: 33–38 µs vs u2's 44–50 µs) — or **batch B>1** to amortize. (A *lean* u2 kernel can reach 26 µs /
68% DRAM, so production's u2 wide kernel — 2× that, SM-bound — has headroom, but in ALU/registers, not occupancy.)

## 11. Is sepsc the bloat? + batch scaling (measured)

**(a) sepsc is NOT the production bloat.** Added the separated-scale accumulate to the *lean* u2 kernel
(`gemv_u2_sepsc`, bytesplit loads) and compared to the per-element path, same config:

| lean variant (split-K 32) | time | vs streaming |
|---|---|---|
| streaming (per-elem) | 32.5 µs | 1.00× |
| bytesplit (per-elem) | 32.8 µs | 0.99× |
| sepsc (separated-scale) | 31.9 µs | 1.02× |

All within noise → **sepsc neither helps nor hurts in the lean kernel.** So the production 48 µs vs lean 26–32 µs gap
is *not* sepsc. The remaining suspect is **runtime-generality**: production passes `u,gs,UB,SB` as runtime ints
(variable-width shifts, extra registers, no clean unroll), while the lean kernel hardcodes u2/gs8 at compile time.
Specializing the production kernel per-(u,gs) is the likely headroom — not occupancy, not sepsc.

**(b) Batch scaling.** Batched kernel unpacks each weight once per block and reuses it across B tokens (amortizes
weight-DRAM + unpack-ALU). Sweep vs dense bf16 matmul (cuBLAS, tensor cores):

| B | naive-MSAQ µs | µs/tok | bf16 µs | bf16 µs/tok |
|---|---|---|---|---|
| 1  | 35.5 | 35.5 | 44.0 | 44.0 |
| 2  | 35.5 | 17.8 | 44.1 | 22.1 |
| 4  | 59.8 | 15.0 | 44.2 | 11.0 |
| 8  | 111  | 13.9 | 44.2 | 5.5 |
| 16 | 179  | 11.2 | 44.8 | 2.8 |
| 32 | 369  | 11.5 | 44.7 | 1.4 |

Reading: **MSAQ µs/token does fall with B** (35→11.5: weight-read amortization works, as §9 predicted). **But total
time grows ~linearly past B=2, and MSAQ loses to bf16 per-token beyond B≈2.** Why: the naive kernel reuses the
*weight* but does the B dot-products as **scalar fp32 madds on CUDA cores**, and the `x[b*K+…]` activation reads
stride by K (uncoalesced) — so compute, not weight-BW, dominates as B grows. bf16 matmul is flat ~44 µs (B≈free)
because cuBLAS rides **tensor cores**. The lesson: amortizing the weight read is necessary but not sufficient — a
competitive batched path must feed **tensor cores** (INT8 IMMA), i.e. the repo's `wa_gemm`/batched-decode path, not
a scalar reuse loop. (`wa_gemm` was verified correct here (rel ~3%) but showed a flat ~600 µs per-call fixed
overhead in this isolated microbench, unrepresentative of its tuned regime — proper batched eval belongs in the
repo's batched harness.) Net: single-token decode favors the byte/ALU-lean MSAQ kernel; once B grows past ~2,
tensor-core paths win and the right comparison is INT8-IMMA-MSAQ vs bf16, not scalar MSAQ.

## 12. Landed: (u,gs)-specialized production kernel — 1.8× (rebuilt, tests pass)

§11(a) pointed at runtime-generality as the production bloat. Confirmed and fixed: added
`wonly_gemv_wide_uspec<int U, int GS>` (compile-time `U/GS/WBITS/UB/SB`) next to the generic
`wonly_gemv_wide_kernel<false>`, with `wonly_gemv_wide_cuda` dispatching known combos to it and falling back to
the generic kernel otherwise (`MS_GEMV_NOSPEC=1` forces generic for A/B). Bit-identical (`max|spec−generic| = 0`
across u2/u3 × gs8/16); all 45 `tests/test_w.py` pass.

| config | generic (full op) | **specialized** | speedup |
|---|---|---|---|
| u2/gs8 | 45.1 µs | **25.7 µs** | **1.76×** |
| u3/gs8 | 43.9 µs | **23.6 µs** | **1.86×** |

ncu of the specialized u2 kernel vs the generic (§10c):

| | generic | specialized |
|---|---|---|
| time (kernel) | 48.8 µs | **25.3 µs** (1.93×) |
| SM% | 61.5 | 52.6 |
| DRAM% | 33.3 | **65.6** |
| instructions | 5.47 M | 5.33 M |
| regs/thread | 48 | 56 |

The kernel **flipped from SM-bound (61% SM / 33% DRAM) to BW-bound (53% SM / 66% DRAM)** — i.e. it now reaches the
~68% DRAM plateau the lean prototype hit (§9), confirming the gap was compile-time-resolvable overhead, not
algorithm. Why it works: constant `WBITS`/`U`/`GS` give constant masks + **constant-width shifts** (vs runtime
variable shifts), fully-unrolled `UB/4` word-loads with no `i<SB`/`i<UB>>2` bounds branch, and — with the k-loop
`#pragma unroll`ed — a **statically-resolved rolling-buffer schedule** so `ureg[uwi]` is register-resident (the
runtime-WBITS kernel can't: its refill cadence is data-dependent). Costs +8 regs (full unroll) but occupancy stays
~56% warps-active. (Only the W-only decode GEMV is specialized; the batched/`wa` paths still use the generic
kernel and could get the same treatment.)
