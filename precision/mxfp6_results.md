# MXFP6-E2M3 (하드웨어-native) vs 커스텀 MSAQ — ~6비트 양자화의 답

`precision/mxfp6_ppl.py`. Blackwell MXFP8 하드웨어 조사("E3M4·공유mantissa는 텐서코어 경로 없음, 표준 MX 원소를
직접 내보내야 가속 가능")의 후속으로, **하드웨어-native MXFP6**(E3M2/E2M3, 6-bit 원소 + UE8M0 블록 스케일 = 6.25 b/elem)가
커스텀 6.0b MSAQ와 정확도가 대등한지 측정했다.

## 결과 — 결정적: MXFP6-E2M3가 모든 ~6비트 MSAQ를 압도

**Llama-3.1-8B** (wikitext-2, BF16 PPL=5.6877). BF16 대비 PPL 증가율(%), within 3% 기준. `*` = 표준 MX 원소(텐서코어 native).

| scope | MXFP6-E3M2* 6.25b | **MXFP6-E2M3* 6.25b** | E3M4-MSAQ 6.0b | MXINT8-MSAQ 6.0b | MXFP8-E4M3* 8.25b |
|---|--:|--:|--:|--:|--:|
| weight | +1.68 ✓ | **+0.54 ✓** | +2.57 ✓ | +2.98 ✓ | +0.48 ✓ |
| weight+act | +3.54 ✗ | **+1.17 ✓** | +4.27 ✗ | +6.02 ✗ | +0.88 ✓ |
| KV | +1.52 ✓ | **+0.36 ✓** | +0.96 ✓ | +0.99 ✓ | +0.36 ✓ |
| weight+KV | +3.08 ✗ | **+0.87 ✓** | +3.48 ✗ | +4.29 ✗ | +0.79 ✓ |

(proxy SmolLM2-1.7B도 동일 순위, 격차 더 큼: E2M3 weight+act +2.11 vs E3M4 +6.59 vs INT8 +10.68. `mxfp6_ppl_smollm2.txt`.)

- **MXFP6-E2M3가 ~6비트에서 유일하게 전 scope within-3%.** E3M4-MSAQ·MXINT8-MSAQ 대비 큰 차로 우세.
  ⚠️ 단 위 표의 MSAQ는 **6.0b**라 0.25b 불리하고, INT8-MSAQ는 **naive(mean)** 였다 — **보정 MSAQ를 같은/조금 더 많은
  비트로 주면 격차가 크게 좁혀진다**(아래 "검증" Pareto frontier 참조). E2M3의 진짜 우위는 "frontier에서 ~30dB를 가장
  싸게 + 하드웨어-native"이지, "MSAQ가 못 쓴다"가 아니다.
- **8.25b E4M3에 근접**(weight+act +1.17 vs +0.88)하면서 **2비트 저렴**.
- **E3M2(지수3/만티사2)는 나쁘다**(+3.5%) — 지수에 비트를 낭비. **E2M3(지수2/만티사3)가 정답**.

## 검증 — confound 아님, 그리고 더 일반적인 진실 (`mxfp6_verify.py`)

"6.25b가 6.0b를 4~6배 이기는 게 이상하다, per-tensor scale이나 다른 요소가 침투한 것 아니냐"는 의심을 직접 점검:

- **(A) 구현 버그 없음**: `msaq_mxfp8(u=0,2,3)` == 독립 재구현(E2M3 FP6 그리드를 직접 enumerate해 nearest-snap)
  이 max|Δ|=3.9e-3(bf16 수준). 진짜 표준 MXFP6-E2M3다.
- **(B) per-tensor scale 침투 없음**: 스케일이 순수 E8M0 power-of-2(`frac(log2 s)=0`), 블록 수 = numel/32 정확
  → **per-32-block E8M0뿐, per-tensor 아님**. 모든 비교 포맷이 동일한 블록 스케일을 쓴다.
- **(C) ⚠️ 초기 framing 정정 — "sharing 자체가 손해"는 틀렸다 (비트 불공정에서 온 착시)**: 처음엔 plain MXINT6
  6.25b가 MSAQ 6.0b를 이기는 걸 보고 "공유가 역효과"라 했으나, 이는 **6.25b vs 6.0b**의 0.25b 차이가 섞인 것이었다.
  공정한 bits-vs-QSNR Pareto(`mxfp6_frontier.py`, 보정 INT-MSAQ.efb/UBC 포함)로 보면 **보정 MSAQ는 MXINT6를 이긴다**.
- **(D) 진짜 결론은 Pareto frontier** (실제 Llama q_proj QSNR; ★=frontier, 합성·down_proj 동일 경향):

  | bits | 최고 포맷(★) | QSNR | 비고 |
  |--:|---|--:|---|
  | 6.00 | E3M4-MSAQ.efb / INT8-MSAQ.efb | 25.7 / 23.8 | **보정 MSAQ가 6.0b 최고** |
  | 6.25 | **MXFP6-E2M3*** | **29.98** | **~30dB를 가장 싸게; frontier** |
  | 6.25 | (MXINT6) | 27.65 | E2M3 > MXINT6 |
  | 6.75 | E3M4-MSAQ.efb u2/4 | 30.38 | 보정 MSAQ가 +0.5b로 E2M3 추월 |
  | 7.25 | MXINT7(plain) | 33.41 | 고비트에선 plain INT이 MSAQ 압도 |
  | 8.25 | MXINT8(plain) | 39.12 | |

  - **보정 MSAQ는 정당한 Pareto 경쟁자다** — MXINT6를 이기고(사용자 지적이 맞음), 6.0b에선 INT8-MSAQ.efb가 최고,
    6.75b에선 E3M4-MSAQ.efb가 E2M3를 근소 추월. **"sharing은 항상 손해"는 명백히 틀림.**
  - **그러나 E2M3는 6.25b로 ~30dB에 도달하는 frontier상 가장 싼 점**이고, MSAQ는 같은 품질에 +0.5b가 필요하다.
    게다가 E2M3만 **하드웨어-native**.
  - **고비트(≥7b)에선 plain MXINT7/8이 모든 MSAQ를 압도** — 공유는 저비트에서만 의미가 있다.
  - 보정 자체도 만능 아님: q_proj에서 INT8-MSAQ.**UBC**(21.8)는 mean(23.4)보다 나빴고, **efb**(23.8)가 가장 안정적.
    전체 표 `mxfp6_frontier.txt`.

## 왜 — 6비트에선 mantissa, 그리고 E2M3의 자리

per-block E8M0 스케일이 dynamic range를 공급하므로 6비트는 **지수가 아니라 mantissa에 써야** 한다(E2M3: 3/6이
mantissa; E3M2는 지수 낭비라 나쁨). E3M4-MSAQ는 8-bit FP8 중 mantissa 최다지만 6.0b로 줄이려 mantissa를 공유해
per-element 1비트만 남겼다 — **mantissa 비중 높은 6-bit native(E2M3)를 공유 없이 쓰는 게 더 직접적**이다. 보정 MSAQ는
그 손실을 상당 부분 만회하지만(위 frontier), E2M3와 같은 품질엔 ~0.5b 더 든다.

## 함의 — 6비트 타깃이면 MXFP6-E2M3

E2M3가 **정확도×하드웨어 결합**에서 최적:
1. **정확도**: bits-vs-QSNR frontier에서 ~30dB를 가장 싸게(6.25b). 보정 MSAQ는 정당한 경쟁자이나 같은 품질에 +0.5b.
2. **하드웨어**: 표준 MXFP6 원소라 Blackwell block-scaled 텐서코어(`tcgen05.mma`/`mma.sync ... block_scale`)에
   **per-element 언팩 없이** 올라간다(커스텀 dequant·`ldexpf`·funnel-unpack 불필요). MSAQ는 어떤 보정을 쓰든 HW 경로 없음.

→ **권장: 6비트 타깃이면 native MXFP6-E2M3** (정확도 frontier + 유일한 HW-native). 커스텀 E3M4-MSAQ 커널은
   이 조합 우위로 superseded. **단 MSAQ가 "나쁜 아이디어"였던 건 아니다** — 보정 MSAQ는 frontier 경쟁자이고, 저비트
   weight 압축에선 여전히 유효; E2M3는 단지 더 싼 frontier 점이자 HW-native라 이긴다.

### 남은 검증 / 다음
- **속도**: 이 박스는 소비자 Blackwell(sm_120). MXFP6 텐서코어 GEMM은 FP32 누산 시 절반 속도(=bf16) — 따라서
  *throughput* 이득은 datacenter(sm_100)에서 크고, sm_120에선 "dequant 제거 + 정확도↑"가 주 이득. 실제 CUTLASS/
  cuBLASLt block-scaled MXFP6 GEMM 경로로 E2M3를 돌려 bf16/MXINT8 대비 측정 필요(CUDA≥12.8, CUTLASS≥3.8,
  예제 `79_blackwell_geforce_*`).
- **E2M3 인코더/패커**: 현재는 `msaq_mxfp8(u=0, eb=2, mb=3)`로 수치 검증만. 표준 MXFP6 비트 레이아웃 패커 +
  텐서코어 GEMM 래퍼가 실제 배포에 필요.
- E2M3가 activation·KV에서도 이기므로 W·A·KV 전 scope에 적용 가능.
