# MSAQ-unsigned (floor) variant — sign-free shared residual, OR-combine

A variant of MSAQ-signed that keeps the **shared residual code sign-free**. The trick is to make the
residual one-sided by quantizing the upper with **round toward −∞** (mathematical floor `⌊·⌋`, = the
quotient of *Euclidean* division — **not** truncation toward zero): round-to-nearest leaves a `±` residual
(which needs a sign), but round-toward-−∞ leaves a `≥ 0` residual. The group mean is then a plain unsigned
code, and the recombine degenerates from an arithmetic **ADD (with carry)** into a pure **bit-OR
(concatenation)**.

> **Precision note on "floor".** Round-toward-−∞ makes the *upper* code `q_un` go negative for `x < 0` —
> that is fine, `q_un` is the *signed* upper and is stored signed. What is stored unsigned is the
> **residual** `res = x − q_un·s_un`, and `res ≥ 0` holds *because* `q_un = ⌊x/s_un⌋ ⇒ q_un·s_un ≤ x`. On
> the integer grid this is Euclidean division by `2^u`: quotient `q_un = ⌊q8/2^u⌋` (can be negative),
> remainder `res = q8 mod 2^u ∈ [0, 2^u)` (the *least non-negative residue*, ≥ 0 by definition). It must be
> Euclidean/floor, **not** C-style truncation toward zero — truncated division gives a remainder with the
> sign of the dividend (`res ≤ 0` for `x < 0`), which breaks unsignedness. The unambiguous bit form on a
> two's-complement byte: `q_un = q8 >> u` (**arithmetic**, sign-extending right shift) and
> `res = q8 & (2^u − 1)` (mask) — the mask yields the non-negative remainder automatically.

Notation:

```
usat_u(v) = clamp(round(v), 0, 2^u - 1)     # "unsigned u-bit saturating round-to-nearest"
                                            # equivalently  ⌊v⌉_0^{2^u-1}
```

Per block of `BLOCK = 32` elements, group size `gs = mg`. `u` = shared lower-bit width; the unshared upper
keeps `8-u` signed bits.

## Encode

```
s_base = E8M0 = 2^(floor(log2(max|x|)) - 6)            # one 8-bit scale exponent per block
s_un   = s_base * 2^u                                  # unshared quantum

q_un   = rtni(x / s_un).clamp(-2^(7-u), 2^(7-u)-1)     # signed (8-u)-bit UPPER; rtni = round toward -inf
                                                      #   (Euclidean quotient = arith. q8 >> u), NOT trunc
res    = x - q_un * s_un                               # in [0, s_un)  ->  ALWAYS >= 0 (incl. negatives;
                                                      #   = q8 mod 2^u, the non-negative remainder)
shared = usat_u( mean_over_mg(res) / s_base )          # group mean -> UNSIGNED u-bit, one code per group
```

## Decode / recombine

```
x_hat = ( (q_un << u) | shared ) * s_base              # OR-fill the low u bits, then * scale
      = ( q_un * 2^u + shared ) * s_base               # (an ADD, but no carry: low u bits are 0)
```

The OR is valid because `q_un << u` has its low `u` bits zero and `shared ∈ [0, 2^u-1]` fits exactly into
them — so `(q_un << u) | shared == (q_un << u) + shared`, with no carry into the upper code, even for a
negative (two's-complement) `q_un`.

## Why each of the 4 parts matters

The naive one-liner "subtract MXINT(8-u) and store the difference unsigned" misses four operations:

1. **Round the upper toward −∞** (`⌊·⌋`, Euclidean quotient — **not** truncation toward zero) → the
   residual is the non-negative remainder (`res = q8 mod 2^u ∈ [0, 2^u)`), so the share can be *genuinely*
   sign-free. (The quotient `q_un` itself still goes negative for `x < 0`; that is the *signed* upper and is
   stored signed — it is the *remainder* that is unsigned.) Round-to-nearest would give a signed `±`
   residual, which **cannot** be stored unsigned without an offset/bias (offset-binary) or a separate sign
   bit. Truncation toward zero would give a remainder with the dividend's sign — also not unsigned.
   (See MSAQ-signed.)
2. **`usat_u(res / s_base)`** → the continuous FP residual must be **quantized to an integer**; the
   `clamp(0, 2^u-1)` handles the round-up-to-`2^u` edge (otherwise it overflows `u` bits / carries into
   the upper code).
3. **`mean_over_mg`** → this is the **sharing / compression itself**: one `u`-bit code per group of `mg`,
   so `SB = u * ceil(BLOCK / mg)`. Without the mean it is per-element `(8-u) + u = 8` bits/elem = plain
   MXINT8, i.e. **zero saving**.
4. **`* s_base`** → the E8M0 block scale, applied once on dequant; stored once (8 bits) per block.

## Tradeoff vs MSAQ-signed

| | combine | carry | sign of share | accuracy |
|---|---|---|---|---|
| **MSAQ-signed** | `(q_un<<u) + shared` arithmetic ADD | yes (low→high) | signed, in the `u` bits | better |
| **MSAQ-unsigned (rtni)** | `(q_un<<u) \| shared` bit-OR | none | none (rtni residual ≥0) | worse |

The unsigned/rtni combine is cheaper (OR-concat, no carry) and the share needs no sign plane. But
round-toward-−∞ drops round-to-nearest on the dominant upper bits (a systematic `+s_un/2` bias), **and** the unsigned
residual can only correct **upward** within the rounded-down cell — it cannot borrow downward the way the signed
two's-complement share does. That one-directional, biased correction is exactly the accuracy gap observed
as `naive_ms < msaq` in the weight-QSNR table (`precision/lightms_qsnr.py`).

See also: `change.md` (MSAQ-signed format reference + this variant), `precision/lightms_qsnr.py`
(`msaq_signed`, `naive_ms`), `precision/single_level_mantissa_sharing.py` (bit accounting, encodings).
