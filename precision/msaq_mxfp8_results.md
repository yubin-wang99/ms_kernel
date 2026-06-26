# MSAQ on MXFP8 — mantissa-sharing applied to FP8 elements (E4M3 / E5M2 / E3M4)

`precision/msaq_mxfp8_ppl.py`. MSAQ까지 INT8(MXINT8)에 적용해온 mantissa-sharing을 **FP8 원소**에 적용한다.
block=32 + per-block E8M0 scale(=MX)는 그대로 두고, 각 원소를 FP8(sign + eb지수 + mb만티사)로 두되
**만티사 하위 u비트를 mg개 원소가 공유**(원소별 지수 단위로 정규화해 적용 → 서로 다른 지수에도 올바르게 더해짐).

- 원소 포맷: E4M3(eb4/mb3), E5M2(eb5/mb2), E3M4(eb3/mb4).
- 유효 bits/elem = `1(sign) + eb + (mb−u) + u/mg + 8/32(E8M0)`.
- cross-check (selftest, 정확히 성립): **u=0 → 순수 MXFP8** (Δ=0.0), **mg=1 → full FP8**.

## 결과 — power-weighted QSNR (dB), block=32

3개 합성 분포: `W`=Gaussian 가중치, `Wt`=약한 heavy-tail, `Ws`=블록 내부 dynamic range 큼(log-uniform, 2^10 범위).

| 포맷 | cfg | bits | QSNR_W | QSNR_Wt | QSNR_Ws |
|---|---|--:|--:|--:|--:|
| E3M4 | MXFP8(u0) | 8.25 | 37.50 | 37.49 | 37.80 |
| E3M4 | u2/mg8 | 6.50 | 25.86 | 25.87 | 26.68 |
| E3M4 | u3/mg4 | 6.00 | 20.84 | 20.86 | 21.56 |
| E4M3 | MXFP8(u0) | 8.25 | 31.45 | 31.44 | 31.39 |
| E4M3 | u2/mg8 | 6.50 | 20.02 | 20.04 | 20.94 |
| E5M2 | MXFP8(u0) | 8.25 | 25.38 | 25.35 | 24.67 |
| **MXINT8** | **MXINT8(u0)** | 8.25 | **42.37** | **41.98** | **41.15** |
| **MXINT8** | **u2/mg8** | 6.50 | **30.67** | **30.29** | **30.57** |
| **MXINT8** | **u3/mg4** | 6.00 | **25.51** | **25.13** | **25.89** |

(전체 표: `msaq_mxfp8_selftest.txt`.)

## 결론

1. **MSAQ는 MXFP8에 깨끗하게 적용된다** — 구현·검증 완료(u0=MXFP8, mg1=full FP8 cross-check 정확). 만티사 하위
   비트를 mg개 원소가 공유하고, 원소별 지수 단위로 정규화해 적용하므로 서로 다른 지수에도 올바르게 동작.
2. **그러나 동일 bits에서 MXFP8-MSAQ는 MXINT8-MSAQ를 못 이긴다** — 전 포맷·전 bit·전 분포에서 INT8이 **~4–6 dB 우위**.
   - 8.25b: INT8 42.4 ≫ E3M4 37.5 ≫ E4M3 31.5 ≫ E5M2 25.4 (이미 baseline부터 INT8 승; MSAQ가 순위를 못 바꿈).
   - 6.5b: INT8 u2/mg8 30.6 vs E3M4 u2/mg8 25.9.
3. **이유**: per-block-32 E8M0 scale이 **이미 dynamic range를 공급**하므로, FP8가 지수에 쓰는 비트가 (블록 내부에선)
   잉여가 된다. QSNR은 power-weighted라 **큰 원소가 지배**하는데, 큰 원소는 block-scale+INT8가 잘 처리하고, FP8의
   지수 비트는 power 기여가 미미한 작은 원소에만 쓰여 mantissa 정밀도로 전환되지 못한다. mantissa가 가장 많은
   **E3M4가 FP8 중 최선**(가장 INT8에 근접)이지만 여전히 ~4–5 dB 뒤진다. dynamic range를 키운 `Ws`에서도 INT8 우위
   유지 — block scale이 블록당 적응하기 때문.
4. **outlier 견고성**: FP8는 W↔Ws QSNR 변화가 작아(원소별 지수) outlier에 강하지만, INT8-MSAQ도 per-block scale
   덕에 W↔Ws 변화가 작다(≤1 dB). 즉 per-block scaling이 있는 한 FP8의 range 이점이 QSNR로 드러나지 않는다.

**함의**: weight 양자화에서 MXINT8이 MXFP8보다 선호되는 통설(블록 스케일이 range를 주므로 uniform INT가 비트
효율적)이 MSAQ에도 그대로 적용된다 — MSAQ는 두 base 포맷 위에서 **동일하게** 작동하지만, base의 우열(INT8>FP8)을
뒤집지 못한다.

## PPL 측정 (wikitext-2, proxy 모델 SmolLM2-1.7B, BF16 PPL=6.9955)

⚠️ **proxy**: 이 머신엔 gated Llama-3.1-8B(+HF토큰)가 없어 ungated Llama-arch **SmolLM2-1.7B**로 측정. 절대값은
8B와 다르나 **포맷 간 상대 순위·scope 거동**은 유효(8B 정식 수치는 precision 환경에서 동일 스크립트로 실행).
표는 BF16 대비 PPL 증가율(%), within 3% 기준. (`msaq_mxfp8_ppl_smollm2.txt`)

**bit-matched 직접 비교 — E3M4(최선 FP8) vs MXINT8:**

| bits | scope | E3M4-MSAQ | MXINT8-MSAQ |
|--:|---|--:|--:|
| 7.38 | weight | +0.48 | **+0.30** |
| 7.38 | weight+act | +1.38 | **+0.90** |
| 7.38 | KV | +0.64 | **+0.14** |
| 7.38 | weight+KV | +1.23 | **+0.46** |
| 6.75 | weight | +2.55 | **+1.18** |
| 6.75 | weight+act | +5.35 ❌ | **+2.76 ✓** |
| 6.75 | KV | +2.31 | **+0.57** |
| 6.75 | weight+KV | +5.13 ❌ | **+1.71 ✓** |
| 6.00 | KV | +10.52 | **+1.82 ✓** |
| 6.00 | weight+act | +46.06 | **+10.68** |

- **MXINT8-MSAQ가 전 bit·전 scope(weight/act/KV)에서 E3M4-MSAQ를 이긴다** — QSNR 결론과 완전 일치. 격차는 저비트
  일수록 벌어짐(6.0b KV: E3M4 +10.5% vs INT8 +1.8%).
- **KV가 양쪽 모두 가장 robust한 scope**이고, INT8 KV는 특히 강함(+1.82%@6.0b도 OK). FP8의 per-element 지수가 KV
  outlier에 유리할 것이란 가설은 이 모델에선 **반증**(INT8이 더 좋음). 단 8B의 더 극단적 outlier에선 다를 여지는 남음.
- **activation(weight+act)이 가장 어려운 scope**(양쪽 다 %가 가장 큼).
- **E5M2(만티사 2비트)는 +17~770%로 사용 불가**; **E4M3 < E3M4**; 포맷 순위 **INT8 > E3M4 > E4M3 ≫ E5M2** (PPL=QSNR 일치).
- 그래도 **E3M4-MSAQ @7.38b는 전 scope within 3%**(weight 0.48/act 1.38/KV 0.64/wkv 1.23) → MXFP8-MSAQ도 ~7.4비트면 동작은 함.

## 한계 / 다음

- 위는 **합성 분포 + power-weighted QSNR**이다. 최종 판정은 **wikitext-2 PPL**(모델 출력 가중 오차)이며, 작은
  값/outlier가 모델 정확도에 미치는 영향은 QSNR과 다를 수 있다(특히 활성화·KV). PPL 스윕은 동일 파일에 구현돼 있어
  precision 환경(transformers + Llama-3.1-8B)에서 바로 실행 가능:
  ```bash
  CUDA_VISIBLE_DEVICES=0 python precision/msaq_mxfp8_ppl.py > precision/msaq_mxfp8_ppl.txt 2>&1
  ```
  scope(weight / weight+act / KV / weight+KV) × 포맷(E4M3/E5M2/E3M4) × (u,mg)로 BF16 대비 PPL %를 출력(within 3% 기준).
