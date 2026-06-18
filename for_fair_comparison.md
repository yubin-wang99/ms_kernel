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

### ✅🟥 KV read (dequant attention) — 비대칭 **해결**, 그리고 공정 결과의 진실
**해결:** MXINT8 KV read를 MSAQ와 동일하게 **thread-per-key**(in-thread q·K dot, 워프
reduction 제거)로 올렸다(`mxint8_kv_split_kernel`, 정확도 GQA rel_fro 1.7e-3). 이제 두 KV read는
Pass-1 매핑까지 동일하고, **element 취급(int8 직읽 vs sub-byte 언팩)만** 다르다.

**비대칭의 실제 크기(정량화):** thread-per-key로 올리니 **MXINT8 KV read가 ~2× 빨라졌다**
(Lk=4680 254→**120µs**, Lk=2848 137→**76µs**). 즉 이전 MSAQ의 KV-read "압승"(u4 0.41~0.77)은
**거의 전부 이 최적화 비대칭의 산물**이었고 mantissa-sharing 효과가 아니었다.

**공정 비교 결과(둘 다 thread-per-key):**
| Lk | u4 /mx | u2 /mx |
|----|--------|--------|
| 1056 | 1.04 | 1.83 |
| 2848 | 1.03 | 1.46 |
| 4680 | 1.01 | 1.64 |
→ **u4는 거의 정확히 tie(1.01~1.04, 미세 손해), u2/u3는 명확히 손해.** MSAQ KV read는 공정
비교에서 **이기지 못한다.**

**왜 못 이기나(근본):** wide 커널은 u4 언팩을 이미 ~2µs 안으로 숨긴다(MSAQ 225µs vs MXINT8
223µs). 그런데 이 커널은 **~66 GB/s**로 돌아 **BW-bound가 아니라 latency/compute-bound**다(스칼라
flash-decode, 텐서코어 미사용). BW-bound가 아니면 "바이트를 덜 읽는" 이점이 시간에 반영되지 않는다
→ u4는 정확히 tie, u2/u3은 추가 언팩만큼 손해. (split-K를 늘리면 per-block 오버헤드가 적은 MXINT8이
더 이득 → 오히려 MSAQ가 더 짐.)

**MSAQ가 이기는 유일한 국면 = BW-bound attention.** decode attention은 KV 캐시를 1회만 읽어
(재사용 없음) 본질적으로 memory-bound여야 한다. 제대로 된 FlashDecoding(텐서코어 MMA로 Q·K^T,
P·V) 커널이면 BW-bound가 되고, 거기서 MSAQ u4는 바이트 비율 그대로 **~0.56×**로 이긴다. 현재
스칼라 커널은 66 GB/s(BW의 7~8× 부족)라 그 국면에 도달하지 못한다. → **공정한 MSAQ KV 우위는
BW-bound(텐서코어) flash-decode 재작성이 전제**이며, 이는 양 포맷 모두에 대한 대규모 작업이다.

**BW-bound 재작성 시도와 근본 장애물(시도 결과).** "MSAQ가 이기는 국면 = BW-bound" 가설로
flash-decode를 BW-bound로 끌어올리려 했으나 다음 장애물을 확인:
- 커널은 BW 시간의 **~20×** 느리게 돈다(MSAQ V 2.7MB는 500GB/s면 ~5µs인데 커널은 100+µs).
  즉 per-key 스칼라 compute + V-staging copy + softmax + combine **오버헤드가 지배**하고, 전역
  BW는 병목이 아니다 → 바이트를 덜 읽어도 시간이 안 준다.
- **근본: sub-byte V의 Pass-2(per-d, 키들에 대한 reduction).** thread-per-d로 V를 직접 읽으면 한
  워프가 한 키의 32 nibble=16B만 만져 **half-sector**(바이트 이점 0). 이를 피하려 **staging**(전역
  V를 coalesced로 shared에 복사)하면 full-sector·0.56× 바이트지만 **shared(~10KB)가 occupancy를
  캡**하고 staging copy+sync 오버헤드가 split-K 확장을 막는다(mult>12서 MSAQ가 *더 느려짐*). 실제로
  staging 제거(direct-V) 실험은 MSAQ를 **tie→1.40 손해**로 악화시켰다(half-sector V). 텐서코어 P·V
  도 V를 shared에 unpack해야 해(=staging) 이 장애물을 못 피한다.
- **결론(시도 후):** 현 구조에서 MSAQ KV read의 공정 최선은 **tie(u4 ~1.0)**. 깨끗한 win은 오버헤드를
  전부 걷어낸 **완전한 FlashDecoding 재설계**(BW 포화)를 요구하며 — sub-byte V의 per-d reduction
  특성상 그조차 V 바이트 이점을 보장하지 못한다. → 본 라운드에선 **공정 최선(tie) 상태로 end-to-end
  재측정**(아래)하고, KV read의 win은 미해결 과제로 명시.

<details><summary>(원래 적출된 비대칭 — 기록용)</summary>

원래 문제: MSAQ는 thread-per-key(wide, Phase 18)인데 MXINT8 KV read만 구버전 warp-per-key+
`__shfl` reduction에 남아 있었음. thread-per-key의 "워프 reduction 제거"는 format-무관 최적화라
MXINT8에도 적용돼야 공정 → 위에서 적용·정량화 완료.
</details>

### (구) 🟥 KV read 비대칭 — 상세(해결 전)
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
1. **✅ KV read 매핑 — 해결됨.** MXINT8을 thread-per-key로 올려 Pass-1까지 동일하게 맞춤.
   정량화 결과 그 비대칭이 **~2× MXINT8 KV read를 가렸었고**, 공정 비교에선 **MSAQ KV read가
   u4에서 tie(1.01~1.04)·u2/u3에서 손해** → 이전 KV "압승"은 산물. 공정한 MSAQ KV 우위는
   **BW-bound(텐서코어) flash-decode**에서만 성립(스칼라 커널은 latency-bound라 바이트 절약이
   시간에 안 나타남). 상세는 위 KV read 절.
2. **⚠️ GEMV split-K mult(16 vs 3), qx-staging 유무** — 각자 최적(best-vs-best)이라 방어 가능하나
   "동일 설정"은 아님. 명시 필요.
3. **⚠️ GEMV 레이아웃/적재폭, KV read 적재폭** — packed 포맷이 강제하는 불가피한 차이(각자 coalesced).
4. **⚠️ KV append이 일을 더 함**(decompose+pack) — 포맷의 내재 비용, 위배 아님.
5. **✅ scale 처리, GEMM 전부, KV write/append 구조, 누적기** — matched.
6. **(D) W+A 활성화 포맷(MSAQ-s vs MXINT8)** — 의도된 문서화 차이.

**결론:** scale을 포함한 대부분은 element 취급에만 차이가 있고 공정하다. **KV read의 매핑 비대칭은
해결**(MXINT8도 thread-per-key)했고, 그 결과 **이전 KV "압승"은 ~2× MXINT8 under-optimization
산물**이었음이 드러났다 — 공정 비교에선 MSAQ KV read가 u4 tie·u2/u3 손해다. 따라서 **KV를 포함한
시나리오(S3·S4)의 종전 수치는 MSAQ를 과대평가**했으며, 공정 커널로 재측정이 필요하다(아래 진행 중).
MSAQ가 KV read를 공정하게 이기려면 스칼라 flash-decode를 **BW-bound(텐서코어 MMA)**로 재작성해야
한다 — 그때 비로소 "바이트를 덜 읽는" 이점이 시간으로 환원된다(u4 ~0.56× 기대).
