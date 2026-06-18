# 공정 비교 감사 — MSAQ 커널이 MXINT8과 "element 취급" 외에 다른 점

**원칙:** MSAQ 커널은 경쟁 상대 MXINT8 커널과 **오직 element를 꺼내는 방식**(sub-byte 언팩
vs int8 직읽기)에서만 달라야 한다. scale 처리, thread mapping, split, 누적기, 타일링, 레이아웃
등 나머지는 같아야 비교가 "mantissa-sharing의 순수 효과"를 격리한다. 아래는 전 커널을 짝지어
**그 외 모든 차이**를 적출한 결과다.

기호: ✅ matched · ⚠️ 차이(아래 분류) · 🟥 공정성에 영향 줄 수 있는 비대칭.

분류:
- **(N) 포맷이 강제** — packed plane 때문에 불가피(각자 자기 포맷의 최적 coalesce). 방어 가능.
- **(T) 커널별 튜닝(best-vs-best)** — 같은 lever를 양쪽에 강제하면 한쪽이 손해라 각자 최적값. 방어 가능하나 비대칭.
- **(A) 최적화 수준 비대칭** — 한쪽에만 적용된 format-무관 최적화 → **결과를 편향시킬 수 있음**.
- **(D) 의도된 포맷 차이** — 이미 문서화된 규약(미러 안 함).

---

## 스코프별 적출

### ✅ scale 처리 — 전 커널 동일
양쪽 다 **E8M0 base scale**를 `ms::e8m0_to_scale`로 복원하고, 블록당 1개를 동일 위치에서 읽어
동일하게 fold(GEMV: `w·scale·x` / W+A: `idot·sw·sa` / KV: `·ksc`,`·vsc`). MSAQ의 shared 코드는
**element 값 안**(`up·2^u+sh`)에만 들어가고 scale 경로는 MXINT8과 글자 그대로 같다. → scale은
element 취급의 일부이며 별도 비대칭 없음.

### ✅ W-only GEMM / W+A GEMM — 잘 matched
- `wonly_gemm_*` ↔ `mxint8_gemm_*`: **같은** 타일 구성(`<TBM,TBN,RTM,RTN>`), 같은 WMMA/
  `_wmma_pipe`, 같은 grid(`(OUT/64, M/64)`), 같은 `DISPATCH_TILE`. B-타일 적재만 다름(언팩 vs int8).
- `wa_gemm_cuda` ↔ `mxint8_wa_gemm_cuda`: **같은** 2-stage(활성화 prepass + INT8 IMMA), 같은
  IMMA grid. 활성화 prepass도 양쪽 **warp-per-block** 동일 구조(`quant_act_msaq_kernel` ↔
  `quant_act_kernel`).
  - ⚠️(D) **활성화 포맷**: MSAQ는 MSAQ-s decompose, MXINT8은 plain MXINT8. 이미 문서화된
    의도적 포맷 차이(미러 안 함; `wa_gemm.cu` 주석/change.md P27). scale·IMMA·grid는 동일.

### ✅ KV write / KV append — matched
- KV write: 양쪽 **thread-per-token + grid.z=nb**(occupancy fix를 generic이라 둘 다 적용),
  token-major plane. element 취급만 다름.
- KV append: 양쪽 **thread=(h,blk), 1-block**. 구조 동일.
  - ⚠️(N·불가피) MSAQ append는 본질적으로 **일을 더 함**: decompose **+ bit-pack** vs
    decompose + int8 store. 이는 "element를 packed로 만드는" 작업 자체라 제거 불가(포맷의 정의).
    공정성 위배가 아니라 mantissa-sharing의 내재 비용.

### ⚠️ W-only GEMV / W+A GEMV — 차이 3~4개
| 항목 | MSAQ | MXINT8 | 분류 |
|------|------|--------|------|
| 가중치 레이아웃 | column-major `[NB,OUT,UB]` | out-innermost `[NB,32,OUT]` | (N) 각자 포맷의 coalesced 레이아웃 |
| 적재 폭 | wide `uint4`(u4)/word-stream(u2u3) | scalar int8 | (N) sub-byte sector-util 때문 |
| **split-K mult** | **16** (`gemv_splitk_count(...,16)`, w_gemv.cu:428/469) | **3**(기본, mxint8.cu:649/675) | (T) wide 커널은 블록당 적재가 적어 더 쪼개야 saturate; narrow는 mult~3에 saturate |
| 누적기 | fp32(W-only)/int dot(W+A) | fp32 / int dot | ✅ 동일 |
| **qx shared-staging**(W+A) | 있음(MSAQ) | **없음** | (T) staging이 MXINT8엔 **해**(39.6→54µs)라 미적용 — 각자 최적 |

- (N)·(T)는 "각 포맷을 각자의 최적 구성으로"라는 best-vs-best 원칙상 방어 가능. 단 **split-K mult
  16 vs 3**과 **qx-staging 유무**는 *수치적으로 다른 설정*임을 명시. (qx-staging은 MSAQ의
  unpack-stall을 풀어주는 것이라 mantissa-sharing-유도 lever로 정당화. change.md P31 참조.)

### 🟥 KV read (dequant attention) — 최적화 수준 비대칭 (가장 주의)
| 항목 | MSAQ `kv_decode_wide_kernel` | MXINT8 `mxint8_kv_split_kernel` |
|------|------|------|
| Pass-1 매핑 | **thread-per-key**(키별 in-thread dot, 워프 reduction **없음**), kv_attention.cu:367 | **warp-per-key**(lane=d, `__shfl` reduction), mxint8.cu:526 |
| 적재 | wide `uint4`/word | scalar int8 |
| split-K over keys + combine | ✅ 있음 | ✅ 있음 |
| online-softmax, scale fold | ✅ 동일 | ✅ 동일 |

- 두 KV plane은 **둘 다 token-major**(`[H,nb,L,UB]` vs `[H,nb,L,32]`)다(mxint8.cu:13 주석
  `[H,nb,32,L]`은 **stale**; 실제 write/read는 `[H,nb,L,32]`).
- **thread-per-key**(Phase 18)는 두 가지를 동시에 한다: (1) sub-byte의 **half-sector** 문제 해결
  — 이건 MSAQ 전용 정당화. (2) **워프 reduction 제거**(키 dot를 in-thread로) — 이건 **format
  무관**이라 MXINT8도 채택하면 빨라질 수 있다. 그런데 **MXINT8 baseline은 (2)를 못 받았다**:
  과거 MSAQ의 구버전(`kv_decode_split_kernel`, warp-per-key)에 matched된 뒤, MSAQ만 wide로
  업그레이드되고 MXINT8 KV read는 warp-per-key에 남았다.
- **영향:** MXINT8 KV read는 자기 최적이 아닐 수 있음 → MSAQ의 KV-read 우위(커널 0.41~0.79,
  S3 KV-only end-to-end 0.92~0.97)가 **일부는 mantissa-sharing이 아니라 이 매핑 비대칭에서 올 수
  있다.** MXINT8 KV read도 thread-per-key(in-thread dot)로 올려 다시 비교해야 완전 공정.
  (단 sub-byte half-sector 이득 부분은 진짜 mantissa-sharing 효과로 남는다.)

---

## 요약 — 무엇이 진짜 "element 취급 외 차이"인가
1. **🟥 KV read 매핑(thread-per-key vs warp-per-key)** — 유일하게 결과를 편향시킬 수 있는 비대칭.
   thread-per-key의 "워프 reduction 제거" 부분이 format-무관인데 MXINT8엔 미적용. **권장: MXINT8
   KV read를 thread-per-key로 맞춘 뒤 S3/S4 재측정.**
2. **⚠️ GEMV split-K mult(16 vs 3), qx-staging 유무** — 각자 최적(best-vs-best)이라 방어 가능하나
   "동일 설정"은 아님. 명시 필요.
3. **⚠️ GEMV 레이아웃/적재폭, KV read 적재폭** — packed 포맷이 강제하는 불가피한 차이(각자 coalesced).
4. **⚠️ KV append이 일을 더 함**(decompose+pack) — 포맷의 내재 비용, 위배 아님.
5. **✅ scale 처리, GEMM 전부, KV write/append 구조, 누적기** — matched.
6. **(D) W+A 활성화 포맷(MSAQ-s vs MXINT8)** — 의도된 문서화 차이.

**결론:** scale을 포함한 대부분은 element 취급에만 차이가 있고 공정하다. 단 **KV read의 thread-per-key
vs warp-per-key**는 format-무관 최적화가 MSAQ에만 들어간 비대칭이라, KV를 포함한 시나리오(S3·S4)의
MSAQ 우위를 다소 과대평가할 수 있다 — 이것만 MXINT8쪽도 올리면 비교가 완전히 깨끗해진다.
