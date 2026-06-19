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

### Phase 34 — design B(warp-transpose) + 점유율 lever 시도, tie 재확인 (2026-06-19)
이전 절의 "66 GB/s, BW 미포화" 수치는 **stale**다. MXINT8 KV read를 thread-per-key로 올린 뒤
다시 측정하니 MXINT8는 **이미 memory-bound**로 **~300–480 GB/s**(Lk↑일수록 peak 936의 51%까지)
스케일한다. MSAQ wide는 **~140–220 GB/s**에서 saturate. 둘 다 점유율 ~23–24%.

**ncu 병목 확정(MS_KV_WIDE=1, u4, Lk4680):** DRAM 14% / **L1TEX 41%(MSAQ 최상위 자원)** /
sectors 0.73× MXINT8 / **regs 121 → 4 blk/SM** / **waves 0.76**(머신을 한 wave도 못 채움) /
지배 stall = long_scoreboard(메모리 latency)+barrier 15%, **math_pipe 3.9%(=ALU bound 아님)**.
→ MSAQ는 *바이트가 아니라* (a) sub-byte-V의 staging 오버헤드(L1TEX)와 (b) one-wave 저점유율
latency에 묶여 있다. "MSAQ tie면 바이트만 0.58×니까 점유율만 올리면 0.58××480=278 GB/s로 win"
가설로 다음을 모두 시도:

- **design B (warp-transpose P·V, staging 완전 제거).** 신규 `kv_decode_warpT_kernel`(u4·D128,
  opt-in `MS_KV_WARPT`). V를 thread-per-key coalesced 적재 → 레지스터 unpack → 32-lane
  broadcast all-reduce로 key→d 전치(스칼라 누적기, row[] 배열 회피). **bit-exact**(wide 대비
  rel_fro 0~6e-5). 결과: shared 11→2 KB로 줄었지만 **점유율 그대로 ~23%**(grid가 0.5 wave로
  *더* 작아짐 — per-SM block 캡이 한계가 아니었음), shfl issue가 L1TEX 57%로 상승 → **wide보다
  ~5–10% 느림**(Lk4680 39.5 vs 35.9µs). 즉 병목은 staging tax 단독이 아니다.
- **점유율 lever.** split mult 3→24 sweep: MSAQ는 flat(combine 오버헤드가 상쇄), **MXINT8만
  38.7→32.7µs로 개선**(per-block 일이 적어 split 이득이 큼) → 격차 *확대*. `__launch_bounds__`로
  regs 121→80 강제: spill로 **오히려 느려짐**(35.6→41.2µs), 점유율은 grid 부족(0.5 wave)이라 불변.
  기존 cp.async 커널: 2× 느림.
- **per-kernel 분해(ncu, Lk4680 mult3):** decode MSAQ 48.5µs vs MXINT8 45.6µs(**6% 차, 사실상
  tie**), combine 8.2µs(포맷 무관·동일). MSAQ가 0.58× 바이트를 읽고도 decode가 tie인 것이 핵심.

**근본 원인(정밀화).** P·V는 *키에 대한 reduction*이라 출력은 thread-per-d로 잡힌다. **int8 V는 이
매핑에서 자연히 full-sector**(워프 32 d-lane = 32 contiguous int8 = 정확히 1 섹터, staging 불필요).
**sub-byte V는 half-sector**(32 nibble=16 B)라 MSAQ는 staging(→L1TEX) 또는 transpose(→shfl)
오버헤드를 *반드시* 지불하고, 그 비용이 0.58× 바이트 절감을 거의 정확히 상쇄한다 — one-wave 저점유율
에서 latency로 숨길 여지도 없다. 이것이 int8 대비 sub-byte의 **구조적 비대칭**이다.

**남은 유일한 정공법 = GQA amortization.** staging/unpack 비용을 G개 query에 분산하면(KV 1회 read,
G개 출력 재사용) per-output 오버헤드가 1/G로 줄어 memory-bound 진입 → 그때 0.58× 바이트가 시간으로
환원. 단 **공정하려면 MXINT8에도 동일한 GQA 커널**이 필요하고(MX도 per-head 커널은 KV를 G회 재독),
GQA-MX 역시 memory win을 받으므로 **win이 보장되진 않는다.** 기존 `kv_decode_gqa_kernel`은
점유율 붕괴(40 GB/s)로 미달 — 제대로 된 재작성은 별도 대규모 작업. **현 라운드 공정 결론: u4 KV read는
tie(best-vs-best에서 MX로 미세하게 기움), design B/C/D로는 못 뒤집음. (change.md Phase 34.)**

### Phase 35 — batch sweep: latency-wall 가설의 반증 (2026-06-19)
"KV read tie의 근본은 one-wave 저점유율 latency-bound다 → batch로 점유율을 (combine 오버헤드
없이) 채우면 BW-bound가 되어 0.58× 바이트가 0.58× 시간으로 환원될 것"이라는 가설을 직접 검증.
배치 flash-decode 커널을 추가(`blockIdx.z=batch`, MSAQ wide + MXINT8 짝, single-token 경로는
b==0으로 byte-identical, `kv_decode_attention_batched`/`mxint8_kv_decode_batched`), B∈{1,4,8,16,32}
× GQA 32:8 × Lk4096 × u4 측정(`tests/kv_batch_bench.py`, 배치 slice == single-token 검증 완료).

| B | MXINT8 (useful GB/s) | MSAQ (useful GB/s) | MSAQ/MX |
|---|---|---|---|
| 1 | 95µs (91) | 90µs (55) | **0.94 win** |
| 4 | 350µs (99) | 375µs (53) | 1.07 |
| 8 | 689µs (100) | 753µs (53) | 1.09 |
| 16 | 1162µs (119) | 1317µs (60) | 1.13 |
| 32 | 2082µs (133) | 2329µs (68) | 1.12 (gs32: 1.07) |

**결과: batch는 MSAQ를 살리는 게 아니라 *더 지게* 만든다(가설 반증).** 메커니즘:
- per-q-head 커널은 GQA 그룹(=4) 안에서 KV를 **4× 재독**하므로 실제 DRAM 트래픽은 useful의 ~4×.
  즉 B=32에서 **MXINT8 실효 ~530 GB/s(피크의 57%, 진짜 BW-bound 도달)** vs **MSAQ ~270 GB/s(29%)**.
- batch가 머신을 채워 BW-bound로 끌어올린 것은 맞다. **그런데 MSAQ의 달성 가능 BW 천장이 MXINT8의
  ~절반**이다(dequant unpack + staging barrier가 throughput을 throttle). → MSAQ는 0.58× 바이트를
  0.51× BW로 읽어 **0.58/0.51 ≈ 1.14× 시간** → 관측된 ~1.1× 손해와 정확히 일치.
- B=1 win(0.94)은 *둘 다 BW-bound가 아닐 때*(one-wave latency + launch 고정비 지배) 적은 바이트가
  미세하게 유리한 것일 뿐. 점유율이 차는 순간 MXINT8의 2× 높은 BW 천장이 지배한다.

**확정 결론(3개 lever 종합: split-K / warp-transpose / batch).** KV read의 binding constraint는
점유율이 아니라 **MSAQ의 dequant-throughput BW 천장(MXINT8 대비 ~0.5×)**이다. 병렬도를 더 줘봐야
대역폭을 실제로 소비할 수 있는 쪽(MXINT8)만 이득. 공정 win은 *dequant 자체를 int8 read와 throughput
동급*으로 만들어야(근본적으로 더 싼 unpack, 또는 텐서코어 dequant 파이프라인) 가능하며, 단순 병렬화로는
불가능함이 batch로 입증됨. (change.md Phase 35.)

### Phase 36 — channel-major V(KIVI식) 탐색: 속도 win, 그러나 정확도 trade로 기각 (2026-06-19)
P·V half-sector의 *원인*인 token-major layout을 고치는 시도. V를 **channel-major `[d,token]`**로 깔면
P·V가 coalesced GEMV가 되어 staging/transpose가 사라짐. proxy(`tests/pv_gemv_proxy.py`): token-major에서
MSAQ를 0.5× BW로 묶던 천장이 사라져 **BW-bound 영역(head fuse OUT=1024+Lk≥8192)에서 MSAQ가 MXINT8과
동일 BW 도달 → ratio 0.54~0.58 WIN**(weight GEMV 0.56과 동일). 즉 layout이 원인이었음을 확인.

**그러나 fair·정당성 점검에서 기각.** dense block은 32원소 연속이라 channel-major는 **grouping을 32-token
블록=reduction축으로 강제**(layout↔grouping 분리불가) → KIVI가 V에 권장하는 *per-token grouping*의 반대.
정확도 probe(`tests/v_grouping_accuracy.py`, 현실적 token_var=3): dequant rel_fro가 **token-major 대비
×1.31(u4)~×1.76(MXINT8) 악화**. **양쪽 포맷 동일 적용이라 fair는 유지되나 둘 다 정확도 저하** —
"공짜 win"이 아니라 속도↔정확도 trade이고 관례(KIVI)에 역행. dense packing에선 *per-token(정확도)↔
channel-major(속도)*를 동시 만족 불가. **결론: 공정·정확도 둘 다 지키는 KV-read win은 per-token V를
유지한 채 staging+텐서코어 P·V로 흡수하는 길(=BW-bound FlashDecoding 재작성)뿐.** (change.md Phase 36.)

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
