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

## 4. 커널 시간 (GPU, RTX 3090) — light-MS는 quant time을 줄이지 않음 (정직한 negative)
정수-친화 decompose(`decompose_lightms_block`: q8=round(x/sa) 1회 후 전부 정수)를 KV write/append에
env-gated(`MS_LIGHTMS=1`)로 넣고 측정.

**(i) quant-kernel time** (`tests/quant_time_bench.py`):
| | MXINT8 | MSAQ(FP-avg) | light-MS(INT) |
|---|---|---|---|
| KV append (launch-bound) | 7.05µs | 8.45µs | **8.43µs (=MSAQ)** |
| KV write (decompose 노출) | 50µs | 62µs (1.23×) | **73µs (1.18× MSAQ)** |
- append는 launch-bound라 light-MS = MSAQ(증가 없음). **write는 light-MS가 1.18× 느림** — 정수
  decompose의 signed-rounding 분기 + q8[32] 레지스터 압박이 손해. **GPU는 FP FMA가 싸서**(MSAQ의
  residual은 1 FMA) 정수화가 이득이 아님 — 순수 정수 ASIC 가정에서만 유효.
- 둘 다 MXINT8보다 1.2~1.45× 느림 = decompose+bit-pack의 내재 포맷 비용(light-MS 무관).
- 정확도(정수 double-rounding 변형 `light_ms_int`): QSNR이 MSAQ 대비 −0.07~−0.30 dB(u3/u4), u2mg2서 −1.4 dB.

**(ii) footprint → GEMV speed** (`tests/gemv_u_bench.py`, W-only 4096²):
| config | bytes | GEMV time | BW |
|---|---|---|---|
| MXINT8 | 1.00× | 1.00× | 356 GB/s |
| u4 (0.58×) | 0.58× | **0.58×** | 355 (memory-bound, 완전 환원) |
| u3 (0.70×, robust) | 0.70× | **0.84×** | 295 (extraction-bound) |
- u4만 byte→time 완전 환원, robust u3는 streaming-unpack이 extraction-bound라 부분적(0.84×). light-MS=MSAQ.

**결론:** light-MS의 가치는 **정확도(≈MSAQ ≫naive) + 정수-친화(ASIC)**이지 **GPU 커널 speedup이 아님**.
quant는 launch/메모리-bound라 averaging이 병목이 아니고, 정수화는 GPU에서 오히려 손해.
→ default는 MSAQ(`decompose_ms_block`) 불변, light-MS는 `MS_LIGHTMS` gated 실험/documented-negative로 보존.

## 재현 스크립트
- `lightms_qsnr.py` — weight QSNR (naive/light/MSAQ), 자체 완결 양자화 정의.
- `lightms_output_qsnr.py` — output-QSNR + 반복-텍스트 PPL.
- `lightms_wikitext_ppl.py` — weight-only wikitext PPL.
- `lightms_act_kv_ppl.py` — activation/KV scope PPL (SDPA 패치로 KV 양자화).
- `lightms_boundary.py` — scope별 bits-정렬 robust 경계 sweep.
- `lightms_qsnr.py::light_ms_int` — 커널 정수 decompose(double-rounding) 정확도 변형.
- `../tests/quant_time_bench.py` — (i) KV write/append quant time (MSAQ/light/MXINT8).
- `../tests/gemv_u_bench.py` — (ii) footprint→GEMV speed.

커널: `csrc/core/ms_utils.cuh::decompose_lightms_block`(정수-친화), KV write/append에 `MS_LIGHTMS=1` gated.
