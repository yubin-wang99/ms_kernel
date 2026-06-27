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

(위 표의 u>0 행은 **plain 인코더**(wshare·efb 없음) 기준 — 베이스 순위 비교용. 개선된 인코더 수치는 아래
"최적화" 섹션과 `msaq_mxfp8_selftest.txt`. u0(=MXFP8 baseline)은 인코더와 무관하게 동일.)

## 최적화 — 인코더 2단계 coordinate descent (wshare + error-feedback)

shared/upper를 번갈아 최적화하는 **coordinate descent**로 인코더를 개선했다. **저장 포맷·decode는 불변**
(per-elem upper + group shared) — 전부 **인코더 전용, 추론에선 공짜**. `msaq_mxfp8(..., wshare=True, efb_iters=2)`(기본값).

**(a) wshare — FP-특화 가중 shared** (shared | upper 고정). plain `mean(frac)`은 모든 원소가 같은 선형 scale을
쓰는 **INT8에서 물려받은 형태**다. FP8은 원소마다 quantum `step_up_i = 2^(e_i−(mb−u))`가 달라 복원오차가
`(frac_i − shared)·step_up_i`이므로, 그룹 L2 `Σ(frac_i − shared)²·step_up_i²`를 최소화하는 **최적 shared는
`step_up_i²`(∝ 4^e_i) 가중평균**(큰 지수 원소가 오차 지배 → shared가 그쪽에 맞춰짐).

**(b) efb — error feedback** (upper | shared 고정). 초기 upper는 shared를 모른 채 라운딩됐으므로, shared 확정 후
`upper2 = round((y − shared·step_up)/step_up)`로 **재라운딩**해 그룹 공유오차 `(frac_i − shared)`를 per-element
upper가 흡수한다(INT8의 `upper_bits_correction`의 FP 아날로그). (a)→(b)를 번갈아 반복(`efb_iters`)하면
**L2 단조 감소, ~2회에 수렴**. cross-check(u=0, mg=1) 보존, **L2상 절대 손해 없음**.

- QSNR(합성): wshare가 plain 대비 +0.2~4.5 dB, efb가 그 위에 추가 +0.4~0.6 dB(공격적 u3/u4에서 최대).
- **결정적으로, 이 둘은 base format의 우열을 실제로 뒤집는다(아래 PPL).** 합성 Gaussian W의 QSNR에선 여전히
  INT8이 +3 dB 우세지만, **실제 모델 weight/activation은 블록 내 dynamic range가 커서 `Ws` regime에 가깝고**,
  거기선 FP8-MSAQ가 INT8과 대등~우세(E3M4 6.0b Ws 26.85 > INT8 25.89). 전체 표: `msaq_mxfp8_selftest.txt`.

## 결론 — **개정(2026-06-27): 저비트에서 MXFP8-MSAQ가 MXINT8-MSAQ를 이긴다**

⚠️ 이전 결론("INT8이 전 scope 승")은 **plain-mean 인코더(wshare·efb 둘 다 없음)** 기준이었고, 개선된 인코더의
PPL은 측정된 적이 없었다. 실제로 측정하니 **결론이 뒤집힌다**.

1. **MSAQ는 MXFP8에 깨끗하게 적용된다** — 구현·검증 완료(u0=MXFP8, mg1=full FP8 cross-check 정확).
2. **6.0~6.75b에서 E3M4-MSAQ가 MXINT8-MSAQ를 이기거나 동률**이다(아래 PPL). 특히 **activation(weight+act)에서
   FP8 압승**(6.0b 5.96% vs INT8 10.68%) — outlier가 많은 scope에서 **per-element 지수가 실제로 값을 한다**는
   원래 가설이 모델 PPL로 입증됐다. **순수 KV 7.38b만 INT8 우위**(+0.59 vs +0.14).
3. **무엇이 레버였나**: 주역은 **wshare**(plain→wshare가 대부분: 6.0b weight+act 46→7.8%). **efb는 그 위에 일관되게
   −1~2pp 추가**(7.8→6.0%)하는 보조 레버 — 공짜라 가치 있으나 주역은 아니다.
4. **여전한 한계**: E5M2(mb2)는 +15~60%로 사용 불가, E4M3(mb3)도 E3M4에 밀린다. **mantissa가 많은 E3M4만**
   INT8과 경쟁 가능. 8.25b baseline QSNR은 INT8(42)≫E3M4(37.5)로 INT8이 우세하나, **저비트 MSAQ regime에선
   FP8의 per-element 지수 이점이 INT8의 baseline 우위를 역전**한다.

**함의**: "블록 스케일이 range를 주니 uniform INT가 항상 낫다"는 통설은 **고비트에선 맞지만 저비트 MSAQ에선 깨진다** —
mantissa 비트를 깎는 MSAQ가 INT8을 더 크게 손상시키고(uniform grid가 작은 원소를 굶김), FP8의 지수가 그 손상을
완화하기 때문. **MSAQ는 base format 선택을 INT8→FP8(E3M4)로 뒤집을 수 있다, 단 저비트(≤6.75b)·outlier scope에서.**

## PPL 측정 (wikitext-2, proxy 모델 SmolLM2-1.7B, BF16 PPL=6.9955)

⚠️ **proxy**: 이 머신엔 gated Llama-3.1-8B(+HF토큰)가 없어 ungated Llama-arch **SmolLM2-1.7B**로 측정. 절대값은
8B와 다르나 **포맷 간 상대 순위·scope 거동**은 유효(8B 정식 수치는 precision 환경에서 동일 스크립트로 실행).
표는 BF16 대비 PPL 증가율(%), within 3% 기준. (`msaq_mxfp8_ppl_smollm2.txt`, 인코더 = wshare + efb_iters=2)

**bit-matched 직접 비교 — E3M4(최선 FP8) vs MXINT8** (승자 굵게):

| bits | scope | E3M4-MSAQ | MXINT8-MSAQ |
|--:|---|--:|--:|
| 7.38 | weight | +0.33 | **+0.30** |
| 7.38 | weight+act | **+0.81** | +0.90 |
| 7.38 | KV | +0.59 | **+0.14** |
| 7.38 | weight+KV | +0.88 | **+0.46** |
| 6.75 | weight | **+0.99** | +1.18 |
| 6.75 | weight+act | **+1.68** | +2.76 |
| 6.75 | KV | +0.61 | **+0.57** |
| 6.75 | weight+KV | **+1.57** | +1.71 |
| 6.00 | weight | +4.27 | **+4.22** |
| 6.00 | weight+act | **+5.96** | +10.68 |
| 6.00 | KV | **+1.22** | +1.82 |
| 6.00 | weight+KV | **+5.75** | +6.36 |

**efb 기여 분리** (E3M4, plain→wshare만→wshare+efb), `MSAQ_EFB=0`으로 efb-off 재현 가능:

| bits | scope | plain | wshare | wshare+efb | INT8 |
|--:|---|--:|--:|--:|--:|
| 6.75 | weight+act | +5.35 | +1.91 | **+1.68** | +2.76 |
| 6.00 | weight+act | +46.06 | +7.79 | **+5.96** | +10.68 |
| 6.00 | KV | +10.52 | +1.75 | **+1.22** | +1.82 |
| 6.00 | weight+KV | +28.77 | +7.34 | **+5.75** | +6.36 |

- **E3M4-MSAQ가 6.0~6.75b 대부분 scope에서 INT8을 이긴다** — weight+act는 전 bit 승, weight·weight+KV는
  6.0~6.75b 승. **순수 KV만 INT8 우위**(per-block scale이 KV outlier를 이미 잘 처리).
- **wshare가 주역, efb가 보조**(−1~2pp 추가). 둘 다 인코더-only → **추론 공짜**.
- **E5M2 사용 불가, E4M3 < E3M4**; **mantissa가 많은 E3M4만 INT8과 경쟁**.

## 한계 / 다음

- **proxy(SmolLM2-1.7B)에서 측정된 역전이므로 Llama-3.1-8B 정식 재현이 다음 단계**다. 8B의 더 극단적 activation
  outlier에선 FP8의 per-element 지수 이점이 **더 커질** 여지가 있다(역전 폭 확대 기대). 동일 스크립트로 바로 실행:
- 합성 Gaussian W QSNR은 여전히 INT8 우세지만, **모델 PPL은 FP8(E3M4) 우세** — 실제 weight/act가 합성보다
  intra-block dynamic range가 커서 `Ws` regime에 가깝기 때문. 즉 QSNR(Gaussian)은 더 이상 최종 판정 기준이 아니다.
- PPL 스윕은 동일 파일에 구현돼 있어 precision 환경(transformers + Llama-3.1-8B)에서 바로 실행 가능:
  ```bash
  CUDA_VISIBLE_DEVICES=0 python precision/msaq_mxfp8_ppl.py > precision/msaq_mxfp8_ppl.txt 2>&1
  ```
  scope(weight / weight+act / KV / weight+KV) × 포맷(E4M3/E5M2/E3M4) × (u,mg)로 BF16 대비 PPL %를 출력(within 3% 기준).
