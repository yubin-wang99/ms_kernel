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
