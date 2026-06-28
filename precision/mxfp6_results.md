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

- **MXFP6-E2M3가 ~6비트에서 유일하게 전 scope within-3%.** E3M4-MSAQ·MXINT8-MSAQ 대비 **4~6배 낮은 PPL 열화**, 그것도
  단 **+0.25b**(6.25 vs 6.0)로. 0.25b로 설명 안 되는 구조적 격차다.
- **8.25b E4M3에 근접**(weight+act +1.17 vs +0.88)하면서 **2비트 저렴**.
- **E3M2(지수3/만티사2)는 나쁘다**(+3.5%) — 지수에 비트를 낭비. **E2M3(지수2/만티사3)가 정답**.

## 검증 — confound 아님, 그리고 더 일반적인 진실 (`mxfp6_verify.py`)

"6.25b가 6.0b를 4~6배 이기는 게 이상하다, per-tensor scale이나 다른 요소가 침투한 것 아니냐"는 의심을 직접 점검:

- **(A) 구현 버그 없음**: `msaq_mxfp8(u=0,2,3)` == 독립 재구현(E2M3 FP6 그리드를 직접 enumerate해 nearest-snap)
  이 max|Δ|=3.9e-3(bf16 수준). 진짜 표준 MXFP6-E2M3다.
- **(B) per-tensor scale 침투 없음**: 스케일이 순수 E8M0 power-of-2(`frac(log2 s)=0`), 블록 수 = numel/32 정확
  → **per-32-block E8M0뿐, per-tensor 아님**. 모든 비교 포맷이 동일한 블록 스케일을 쓴다.
- **(C) 신뢰성 있는 새 사실 — "E2M3 마법"이 아니라 sharing 자체가 손해**: 공유 없는 **MXINT6**(plain 6-bit 정수
  + 같은 E8M0, 6.25b)조차 MSAQ 6.0b를 5~7 dB 이긴다. 즉 핵심은 E2M3가 아니라 **비트-공유(MSAQ) vs 비공유**다.
- **(D) 비트 예산 차이 아님**: MSAQ에 0.25b를 더 줘도(E3M4-MSAQ 6.5b = 28.8 dB) E2M3 6.25b(30.0 dB)를 못 따라온다.

QSNR(dB), 실제 Llama q_proj 가중치 (전체 표 `mxfp6_verify.txt`, 합성·down_proj도 동일 순위):

| 포맷 | bits | QSNR | 공유? |
|---|--:|--:|---|
| **MXFP6-E2M3** | 6.25 | **29.98** | 비공유(native FP6) |
| MXINT6 (plain) | 6.25 | 27.65 | 비공유(native INT6) |
| E3M4-MSAQ 6.5b | 6.50 | 28.75 | 공유 |
| E3M4-MSAQ 6.0b | 6.00 | 25.72 | 공유 |
| MXINT8-MSAQ 6.0b | 6.00 | 23.42 | 공유 |
| MXFP8-E4M3 8.25b | 8.25 | 31.37 | 비공유 |

→ **비공유 6.25b(E2M3>INT6)가 공유 6.0b(E3M4·MXINT8 MSAQ)를 일관되게 능가.** MSAQ의 mantissa/bit 공유가
  ~6비트에선 역효과. E2M3는 그 중 최고(mantissa 중심 FP6) + 유일한 하드웨어 native.

## 왜 — MSAQ가 푼 문제가 애초에 잘못됐다

per-block E8M0 스케일이 이미 dynamic range를 공급하므로, 6비트는 **지수가 아니라 mantissa에 써야** 한다(E2M3: 3/6이
mantissa). 그리고 핵심: **~6비트에선 공유(MSAQ)할 이유가 없다** — native FP6 원소가 per-element mantissa 3비트를 전부
쓰는데, E3M4-MSAQ 6.0b는 4 mantissa 중 3을 그룹 공유해 **per-element 1비트만** 남긴다. 즉 MSAQ의 mantissa-sharing은
sub-byte를 만들려고 정밀도를 깎았는데, **6비트짜리 native float은 깎을 필요 없이 그냥 6비트를 쓴다.**

이전 결론("mantissa > 지수, 블록 스케일이 있으면")은 옳았으나, 그걸 8-bit FP8 중 mantissa 최다인 E3M4로 실현한 게
차선이었다. **올바른 실현은 mantissa 비중이 높은 6-bit native 포맷(E2M3)을 공유 없이 쓰는 것**이다.

## 함의 — 커스텀 MSAQ-E3M4 경로는 MXFP6-E2M3로 대체된다

E2M3는 **두 축 모두**에서 커스텀 MSAQ를 이긴다:
1. **정확도**: 6.25b로 6.0b MSAQ를 4~6배 차이로 능가(위 표).
2. **하드웨어**: 표준 MXFP6 원소라 Blackwell block-scaled 텐서코어(`tcgen05.mma`/`mma.sync ... block_scale`)에
   **per-element 언팩 없이** 올라간다. 커스텀 dequant 커널·`ldexpf`·sub-byte funnel-unpack 전부 불필요. 하드웨어가
   UE8M0 스케일을 블록당 자동 적용.

→ **권장: 6비트 타깃이면 커스텀 MSAQ 대신 native MXFP6-E2M3.** 우리가 만든 E3M4-MSAQ 커널(dequant/GEMV/GEMM)은
   이 결과로 **상위 호환에 의해 폐기(superseded)**된다 — 단, "block scale이 있으면 mantissa가 지수보다 중요"를
   극단까지 밀어붙인 끝에 E2M3에 도달한 것이므로 그 여정 자체가 결론을 낳았다.

### 남은 검증 / 다음
- **속도**: 이 박스는 소비자 Blackwell(sm_120). MXFP6 텐서코어 GEMM은 FP32 누산 시 절반 속도(=bf16) — 따라서
  *throughput* 이득은 datacenter(sm_100)에서 크고, sm_120에선 "dequant 제거 + 정확도↑"가 주 이득. 실제 CUTLASS/
  cuBLASLt block-scaled MXFP6 GEMM 경로로 E2M3를 돌려 bf16/MXINT8 대비 측정 필요(CUDA≥12.8, CUTLASS≥3.8,
  예제 `79_blackwell_geforce_*`).
- **E2M3 인코더/패커**: 현재는 `msaq_mxfp8(u=0, eb=2, mb=3)`로 수치 검증만. 표준 MXFP6 비트 레이아웃 패커 +
  텐서코어 GEMM 래퍼가 실제 배포에 필요.
- E2M3가 activation·KV에서도 이기므로 W·A·KV 전 scope에 적용 가능.
