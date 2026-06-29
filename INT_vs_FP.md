# MSAQ: MXINT8 vs MXFP8 — packing / quantization / dequantization 메커니즘 비교

MSAQ(Mantissa-Sharing Quantization)가 **INT 원소(MXINT8)** 에 적용될 때와 **FP 원소(MXFP8-E3M4)** 에
적용될 때 어떻게 다른지 정리한다. 결론부터:

> MSAQ는 **"upper(원소별 `8-u`비트) + shared(그룹 공유 `u`비트) + E8M0 블록 스케일"** 이라는
> **하나의 동일한 비트 컨테이너**다. MXINT은 upper를 *선형 정수*로, MXFP는 *FP8 E3M4 비트필드*로
> 해석할 뿐이다. 따라서 **packing 레이아웃·메모리 트래픽은 글자 그대로 같고**, 차이는
> (a) encoder(quant) 수식과 (b) dequant 커널의 **per-element ALU**, 그리고 (c) **어디에 커널이
> 구현되어 있는가**(weight에는 둘 다, **KV에는 INT만**)에 있다.

소스: `ms_lib/pack.py`, `ms_lib/reference.py`, `csrc/core/ms_utils.cuh`,
`csrc/kv_attention.cu`, `precision/msaq_mxfp8_ppl.py`.

---

## 0. 공통 골격 (둘이 똑같은 부분)

두 경우 모두 OCP-MX 구조 (`pack.py:22-24`):

- **BLOCK = 32**: 32개 원소가 한 블록
- **E8M0 블록 스케일**: 블록당 1개의 2의 거듭제곱 스케일 (`scale_exp`, int8 저장)
- **MSAQ 분해**: 각 원소를 `upper`(원소별 `8-u`비트) + `shared`(그룹 `gs`개 공유 `u`비트)로 분리
- **plane 레이아웃**: `scale_exp [nb,OUT]`, `upper [nb,UB,OUT]`, `shared [nb,SB,OUT]`
  (out-innermost SoA, LSB-first dense). `UB = BLOCK*(8-u)/8`, `SB = ceil(n_group*u/8)`

`pack_weight`(INT)와 `pack_weight_msfp8`(FP)를 비교하면 plane shape, `_pack_codes_lsb` 호출,
transpose, `UB`/`SB` 계산이 **글자 그대로 동일**하다 (`pack.py:101-137` vs `199-223`).
재구성 공식의 형태도 동일: `value = (m_up·2^u + shared) · scale`. 차이는 `m_up`과 `scale`의 *의미*.

비트 회계도 같다 — E3M4는 `1+3+4=8`이라 폭이 INT8과 일치:
- MXINT8: `(8-u) + u/gs + 8/32` (`bits_mxint8`)
- MXFP8 : `1 + eb + (mb-u) + u/gs + 8/32` = `(8-u) + u/gs + 8/32` (`bits_mxfp8`)

→ **같은 비트 예산**에서 INT은 균일 선형 그리드, FP는 원소별 지수를 제공.

---

## 1. Quantization 로직 (encoder) — 본질적 차이

### MXINT8-MSAQ: `decompose` (`pack.py:37-60`)

원소가 **선형 정수**. 한 블록이 하나의 선형 그리드를 공유한다.

```
s_base     = E8M0(max_abs)                 # 2^(floor(log2 amax) - 6)
s_unshared = s_base * 2^u                   # coarse 그리드
q_upper    = clip(round(x / s_unshared), -(2^(7-u)-1), 2^(7-u)-1)   # (8-u)비트 부호정수
residual   = x - q_upper * s_unshared
res_avg    = mean_over_group(residual)      # gs개 평균
r_shared   = clip(round(res_avg / s_base), -2^(u-1), 2^(u-1)-1)     # u비트 부호정수
```

재구성: `(q_upper·2^u + r_shared)·s_base` → **유효한 MXINT8 정수 워드**. sharing을 풀면 평범한
int8 한 개가 되므로 W+A 경로에서 int8 텐서코어 GEMM에 그대로 재활용된다 (`weight_int8`).

### MXFP8-MSAQ: `decompose_msfp8` (`pack.py:148-196`)

원소가 **FP8 E3M4**(sign 1 / exp 3 / mantissa 4). 각 원소가 **자기 지수**를 가지고, sharing은
지수가 아니라 *저(low) 만티사 비트만* 공유한다. INT과 다른 4가지:

1. **Per-element 지수**: `round_storable(t)`가 원소마다 지수 `ee`·만티사 `m`을 따로 구함
   (`pack.py:166-172`). 블록 내 dynamic range가 클 때(작은 값이 죽지 않음) 유리.
2. **"Bit-storable" 헤드룸 트릭**: upper가 `mb-u`비트만 들고 있어 블록 스케일에 지수 1칸 헤드룸을
   준다 (`s_exp = floor(log2 amax) - (maxexp-1)`, line 162). 맨 위에서 saturate 대신 **지수를
   promote** (`promote = m >= 2*lead`, line 170) → `[sign|exp|(mb-u)mant]` 필드로 모든 값이 정확히
   표현되고 커널이 bit-exact 디코드.
3. **wshare (지수 가중 평균)**: shared를 `step_up² = 4^ee` 가중 평균으로 구함 (line 181-182).
   FP는 원소마다 양자화 스텝(`step_up = 2^(ee-mb_up)`)이 달라 큰 지수 원소가 L2 오차를 지배 →
   shared가 그 원소에 맞춰져야 최적. INT은 모든 원소가 같은 스텝이라 단순 평균이 곧 최적.
4. **shared 양자가 원소별 지수에 묶임**: `shared_elem*step_up`의 `step_up`이 각 원소의 *저장된*
   지수 `ee` 기준 (line 186). 그룹이 여러 지수에 걸쳐도 공유 보정이 올바르게 적용.

두 경우 모두 **EFB(error-feedback) 좌표하강**(`efb_iters=2`)을 쓴다: "shared 고정 → upper
재라운딩" 반복으로 sharing 오차를 upper가 흡수. INT판(`msaq_mxint8_efb`)은 균일 가중, FP판은
wshare 가중. EFB는 encoder-only라 추론 시 무료.

마지막 encoding: FP는 `field = (sgn<<(wbits-1)) | (exp<<mb_up) | upm`으로 FP8 비트필드 패킹
(`pack.py:188-195`). INT은 `q_upper`를 그냥 부호정수로 패킹.

| | MXINT8-MSAQ | MXFP8-MSAQ (E3M4) |
|---|---|---|
| upper 필드 (`8-u` bit) | 선형 부호정수 `q_upper` | FP8 비트필드 `[sign\|exp:3\|upmant:(4-u)]` |
| shared 필드 (`u` bit) | 부호정수 (선형 잔차) | 부호정수 (만티사 LSB 분율) |
| 블록 스케일 | `E8M0(amax)`, 헤드룸 0 | `E8M0`, **헤드룸 1칸** (promote용) |
| shared 평균 | 단순 평균 | `4^ee` 가중 평균 (wshare) |
| plane 모양/바이트수 | — | **INT과 동일** |

---

## 2. Dequantization 로직 (런타임 CUDA, weight 경로)

`ms_utils.cuh`의 두 streaming-unpack 템플릿이 차이를 압축해서 보여준다.

**공통 전반부**: `upper_cm`/`shared_cm`를 uint32 레지스터로 wide-load → rolling bit-buffer로
`8-u`비트 `field`, `u`비트 `sh_code` 추출. **이 부분은 두 커널이 동일.**

**INT8 — `stream_block_uspec` (`ms_utils.cuh:189-223`)**:
```cpp
up_code = sign_extend(field, 8-u);          // 부호정수
sh_code = sign_extend(sbuf, u);
per_elem(k, up_code * (1<<u) + sh_code);    // 정수 워드 1개 (shift + add)
```
결과는 정수. 블록 스케일은 호출부에서 `* 2^exp` 곱.

**FP8 — `stream_block_uspec_fp8_e3m4` (`ms_utils.cuh:272-313`)**:
```cpp
sgn  = field >> (wbits-1);
expf = (field >> mb_up) & expmask;            // 3비트 지수
m_up = (expf ? lead : 0) | (field & upmask);  // 암묵 leading-1 복원 (normal/subnormal)
ee   = expf ? expf - BIAS : EMIN;
sh_code = sign_extend(...);
mag = m_up << u; if (sgn) mag = -mag;
per_elem(k, ldexpf((float)(mag + sh_code), ee - MB));   // FP 값 (ldexpf로 지수 적용)
```

dequant에서 FP가 추가로 도는 것: **per-element 지수/만티사 분리 + 암묵 leading-1 복원 +
`ldexpf`**. INT은 정수 `shift+add` 한 번이면 끝. 메모리 트래픽(로드 바이트)은 완전히 동일 →
*"isolates fused decode ALU"* (주석). 이게 FP쪽 decode ALU가 더 무거운 트레이드오프의 정체.

---

## 3. KV 경로 — INT8 vs MXFP8

### 3a. 원래 상태: weight에는 MXFP8 커널이 있었지만 KV에는 없었다

분석 시점(이 문서 작성 시)에 KV에 등록된 커널은 INT 계열 두 가지뿐이었다:

- **INT-기반 MSAQ (배포)**: `kv_decode_attention(_batched)`, `kv_kdot_uspec`,
  `kv_kdot_unsigned`, `kv_kdot_relayout`, `kv_write`/`kv_append`(_rot)
- **MXINT8 baseline**: `mxint8_kv_decode(_batched)`, `mxint8_kv_write`, `mxint8_kv_append`,
  `kv_kdot_mxint8`

`msfp8_*` 함수(`pybind.cpp`)는 **전부 weight 경로**(`msfp8_dequant_bf16`, `msfp8_gemv_wide`,
`msfp8_gemv_batched`, `msfp8_gemm`)였고, **KV용 msfp8 커널은 없었다**. MXFP8 KV는
`precision/msaq_mxfp8_ppl.py`의 SDPA fake-quant(PPL 측정)에만 존재했다.

> **업데이트(이후 구현됨)**: weight 경로와 1:1 대응이 되도록 **MXFP8-MSAQ KV 커널을 추가했다** —
> `msfp8_kv_decode_attention(_batched)`, `msfp8_kv_kdot`(probe), `msfp8_kv_write`/`kv_append`
> (`csrc/kv_attention.cu`), Python측 `pack_kv_msfp8`/`dequant_weight_msfp8`/`kv_attention_msfp8`/
> `ops.msfp8_kv_decode`, 그리고 bit-exact 검증 `tests/test_msfp8_kv.py`. 저장 plane 레이아웃은
> INT과 동일하고, K-dot은 `stream_block_uspec_fp8_e3m4`로, **P·V는 float로** 디코드한다(아래 3c의
> int8-staging이 FP8에 이식 불가하므로). `sepsc`도 빠지고 `qrot`은 유지된다. 즉 아래 분석(왜 비용만
> 늘고 정확도 이득이 없는지)은 그대로 유효하며, 커널은 **포맷 대칭성/완전성**을 위해 추가된 것이다.

### 3b. KV 저장 포맷은 (INT 안에서) weight와 동형

KV 캐시도 동일한 MSAQ 컨테이너를 쓴다. `pack_kv`(`pack.py:479-504`)는 머리(head)마다
`pack_weight`를 `[L, D]`(블록은 head_dim, 토큰 innermost)에 적용하고, **바이트 축을 innermost로**
배치(`[H,nb,L,UB]`)해 토큰 32개가 연속 바이트 → coalesced load. MXINT8 baseline은
`pack_kv_mxint8`로 `qweight` int8 plane을 만든다 (`pack.py:581-591`).

### 3c. KV 융합 어텐션 — 디코드가 INT으로 유지되는 이유

`kv_decode_attention` 커널(`kv_attention.cu`)은 K와 V를 attention 안에서 dequant한다:

- **K-dot (QKᵀ)**: 키를 thread-per-key로 wide-load, MSAQ 코드를 `up·2^u + sh` **정수**로 복원해
  E8M0 스케일과 곱해 dot 누적 (`kv_attention.cu:551-570`). `sepsc`(separated-scale)로
  `Σ q·(up·2^u+sh)·s = s·(2^u·Σq·up + Σ_g sh·qg)`로 분해해 스케일 곱을 키-독립으로 빼낸다
  (line 526-537). 전부 정수 도트 + 소수 스칼라.
- **V-dot (P·V)**: `v8`(int8-staged V) + `vt`(transposed 레이아웃)으로, MSAQ 코드를 **int8 워드로
  재스테이징**해 P·V를 int8로 돌린다 (`kv_attention.cu:674-714`). 즉 배포 방향은
  *"FP를 끌어들이기는커녕 V까지 int8로 환원"* 하는 쪽이다.

여기에 FP8 원소를 넣으면 K/V 디코드마다 §2의 `ldexpf` per-element ALU가 추가된다 — **정확히
배포가 피하려는 비용**. 그런데 KV-read는 이미 **byte-roofline / L1TEX-bound**다
(`KV_cache_analysis.md`: gs16 B32에서 mq/mx 0.52 ≈ 0.545 byte-roofline; ncu DRAM 17% /
L1TEX 70%). 메모리가 병목인 커널에 ALU만 더해봐야 손해다.

### 3d. 정확도: KV scope에서 FP의 per-element 지수 이점이 **미미**

per-element 지수는 **outlier가 많은 scope(activation)** 에서 값을 한다. KV는 그렇지 않아 INT과
사실상 동률이다 (Llama-3.1-8B, PPL Δ%, `msaq_mxfp8_results.md` / `..._llama31_8b.txt`):

| bits | scope | MXFP8-E3M4 | MXINT8 | 승자 |
|---|---|--:|--:|---|
| 7.38 | KV | +0.23 | **+0.12** | INT |
| 6.75 | KV | +0.32 | **+0.23** | INT |
| 6.00 | KV | **+0.96** | +0.99 | FP (미세) |
| 6.00 | weight+act | **+4.27** | +6.02 | FP (뚜렷) |

KV에서는 7.38/6.75b INT 미세 우세, 6.0b FP 미세 우세로 **무승부**. FP가 분명히 이기는 곳은
activation(weight+act)이다. 즉 KV는 FP로 바꿀 **정확도 유인도 약하다**.

### 3e. 결론 (KV)

| 항목 | MXINT8-MSAQ (KV, 배포) | MXFP8-MSAQ (KV) |
|---|---|---|
| 융합 어텐션 커널 | **있음** (`kv_decode_attention` 등) | **있음** (`msfp8_kv_decode_attention`; 대칭성 위해 추가) |
| K/V 디코드 | 정수 `up·2^u+sh` × E8M0 | (가정 시) `ldexpf` per-element ALU 추가 |
| V 처리 방향 | **int8-staged** (`v8`/`vt`) → P·V int8 | int8 재스테이징과 충돌 |
| 병목 | byte-roofline / L1TEX-bound | ALU만 늘려 손해 |
| KV 정확도 (vs INT) | 기준 | **사실상 동률** (6.0b만 미세 우세) |

→ **KV에서 FP8는 비용(ALU)만 늘고 정확도 이득이 없어 커널화되지 않았다.** weight 경로와 정반대로,
KV의 배포 방향은 V까지 int8로 환원하는 **"더 INT스럽게"** 쪽이다.

---

## 4. 한눈 요약

| 축 | MXINT8-MSAQ | MXFP8-MSAQ (E3M4) |
|---|---|---|
| upper 원소 의미 | 선형 정수 | FP8 `[sign\|exp:3\|mant:(4-u)]` |
| 저장 레이아웃·바이트 | — | **완전히 동일** (`8-u` upper + `u` shared + E8M0) |
| encoder(quant) | 단일 선형 그리드, 단순평균 잔차 | 원소별 지수 + 헤드룸 promote + `4^ee` wshare |
| dequant ALU | 정수 `shift+add` | + 지수/만티사 분리 + `ldexpf` |
| weight 커널 | 있음 | **있음** (`msfp8_gemv/gemm/dequant`) |
| KV 커널 | **있음** (배포; V는 int8-staged) | **없음** (PPL 실험만) |
| 정확도 강점 | KV·저-outlier scope에서 동률~우세 | **activation 등 outlier-heavy scope** |
| 비용 | 가벼운 decode ALU | 무거운 decode ALU (memory-bound 커널엔 독) |

**큰 그림**: ~6bit대 *정밀도*에서는 outlier가 많은 activation/weight에서 HW-native MXFP가
유리하지만, *디코드 ALU*는 FP가 더 무겁다. 그래서 메모리 병목인 **KV 경로는 INT(심지어 V를 int8로
환원)으로 가고**, 정확도 유인이 큰 **weight/activation 경로에만 MXFP 커널을 둔다.**
