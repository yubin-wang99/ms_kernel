# light-MS — INTEGER-residual mantissa sharing (정확도 검증 + robust-aggressive 경계)

MSAQ의 FP-residual 평균을 **INT-residual 평균**으로 바꾼 경량 변형. online-quant 경로
(KV append / W+A activation pre-pass)의 FP group-mean을 INT 연산으로 대체해 커널 overhead를
줄이되 정확도는 MSAQ와 동등하게 유지하는 것이 목표. 함수: `error_correction_mechanism.py::lightMS_signed`.

## 0. 정의 (vs MSAQ_signed)
저장 포맷·dequant 동일((7−u)-bit unshared + u-bit signed shared, 복원 `upper·2^u+shared`).
차이는 **평균 순서**뿐:
- **MSAQ_signed**: `residual_fp = x − x_unshared` → **FP group-mean** → u-bit 정수 양자화.
- **light-MS**:    `residual_fp` → **per-elem u-bit 정수 양자화 먼저** → **INT round-to-nearest mean**.

## 1. 정확도 — light-MS ≈ MSAQ, ≫ naive (wikitext-2 PPL, BF16=6.57, 기준 3%)
세 scope(weight / activation / KV) 실측. `light_ms`/`MSAQ_signed` 재구현은 프레임워크 함수와 bit-exact.

| scope | config | naive | **light** | MSAQ |
|---|---|---|---|---|
| weight | u3/mg4 | +3.36 ❌ | **+2.79 ✓** | +2.81 |
| | u4/mg8 | +31.1 | +10.4 | +10.3 |
| activation | u3/mg4 | +3.88 ❌ | **+2.54 ✓** | +2.89 |
| | u4/mg8 | +24.9 | **+12.1** | +12.7 |
| KV | u4/mg2 | +3.07 | +2.72 ✓ | +2.89 |
| | u4/mg8 | +6.37 | +5.19 | +5.14 |

- **light-MS ≈ MSAQ**: 모든 scope·config에서 ≤0.6%p, aggressive 구간선 light이 미세 우위.
- **light-MS ≫ naive**: 특히 **activation에서 2×**(u4mg8 naive +24.9% vs light +12.1%) — activation outlier에서
  naive의 unsigned low-bit 공유가 만드는 *상관 오차*가 치명적; light의 signed-residual 구조가 회피.
- **⚠️ QSNR 신뢰 불가**: naive는 QSNR상 light과 동급(0.06 dB 내)인데 PPL은 2× 나쁨 → **PPL이 결정적.**

## 2. robust-aggressive 경계 (light-MS, bits/elem = (8−u)+u/mg+0.25, 최소-bits robust)

| scope | **min-bits robust** | bits/elem | PPL | vs MXINT8(8.25b) |
|---|---|---|---|---|
| **KV** | **u3 / mg32** | **5.344** | +1.24% | **0.65×** |
| weight | u3 / mg8 | 5.625 | +3.00% | 0.68× |
| activation | u3 / mg4 | 6.000 | +2.54% | 0.73× |

**핵심:**
- **u4는 전 scope fail** — unshared가 3-bit뿐이라 mg를 줄여 bits를 늘려도(u4/mg2=6.25b) 거칠어 fail(+6%~).
  → 공격성 knob은 **u3 고정 + mg를 scope가 견디는 만큼 ↑**.
- **KV 가장 관대**(u3/mg32, 블록 32개 통째 공유, +1.24%) → 0.65×까지 압축. **Act 가장 엄격**(u3/mg4).
  Weight 중간(u3/mg8). → **scope 차등 적용**.

## 3. 결론
light-MS = **MSAQ 동등 정확도 + naive보다 명백히 우수(특히 activation) + INT-averaging으로 online-quant
커널 overhead 절감.** robust-aggressive: KV u3/mg32(0.65×) · Weight u3/mg8(0.68×) · Act u3/mg4(0.73×).

## 재현 스크립트
- `lightms_qsnr.py` — weight QSNR (naive/light/MSAQ), 자체 완결 양자화 정의.
- `lightms_output_qsnr.py` — output-QSNR + 반복-텍스트 PPL.
- `lightms_wikitext_ppl.py` — weight-only wikitext PPL.
- `lightms_act_kv_ppl.py` — activation/KV scope PPL (SDPA 패치로 KV 양자화).
- `lightms_boundary.py` — scope별 bits-정렬 robust 경계 sweep.
