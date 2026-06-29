# 중간 비트 sharing vs LSB sharing — weight QSNR 검증

MSAQ/naive-MS의 mantissa sharing은 항상 **최하위 `sb`비트(LSB 윈도우)**를 그룹 공유한다.
이 실험은 공유 윈도우를 비트 offset `lo`만큼 위로 밀어 **중간 비트**를 공유했을 때
정밀도가 LSB 공유 대비 우위인지를 QSNR로 검증한다.

스크립트: `precision/midbit_qsnr.py` · 모델: Llama-3.1-8B base weights (224 Linear 텐서) · block=32.

## 비트 레이아웃 (iso-bit 비교)

```
8-bit code = [ top (8-sb-lo) | shared sb | low lo ]
               per-element      SHARED       per-element
  lo = 0 → 하위 sb비트 공유               == 현행 LSB scheme (baseline)
  lo > 0 → 하위 lo비트 per-elem 유지, 그 위 sb비트를 공유 (중간 윈도우)
```

`(sb, mg)` 고정 시 per-block 비트 예산은 모든 `lo`에서 **동일**
(`(8-sb)` unshared bits/elem + `sb·⌈BLOCK/mg⌉` shared bits). `lo`는 *어느* sb-윈도우를
공유하느냐만 바꾸므로 lo=0 vs lo>0는 공정한 iso-bit 비교다.

두 recombination 계열 모두 측정 (lightms_qsnr.py에서 일반화):
- `naive_win` — unsigned bit-field 치환(OR-style). **lo=0은 배포 `naive_ms`와 bit-exact.**
- `msaq_win`  — signed residual INT-mean(ADD-style). lo=0은 커널 정수 decompose(`light_ms_int`)
  경로와 동일, 배포 `light_ms`/MSAQ와는 double-rounding 차이로 0.06–0.82 dB 이내(u2mg2 최대).

## 결과 — QSNR(dB), lo가 커질수록 단조 감소 (전 config, 두 계열 공통)

### naive (OR-style unsigned)
```
sb  mg | lo=0      lo=1      lo=2      lo=3      lo=4    | best
 2   2 | 31.767   26.076   20.143   14.124    7.964     | lo=0
 2   4 | 30.498   24.726   18.767   12.739    6.564     | lo=0
 2   8 | 29.948   24.141   18.173   12.143    5.967     | lo=0
 3   2 | 26.447   20.509   14.481    8.268    1.209     | lo=0
 3   4 | 24.825   18.858   12.821    6.601   -0.478     | lo=0
 3   8 | 24.190   18.214   12.174    5.952   -1.131     | lo=0
 4   2 | 20.607   14.574    8.345    1.225     --       | lo=0
 4   4 | 18.882   12.841    6.608   -0.517     --       | lo=0
 4   8 | 18.221   12.177    5.944   -1.183     --       | lo=0
```

### msaq (signed INT-mean)
```
sb  mg | lo=0      lo=1      lo=2      lo=3      lo=4    | best
 2   2 | 30.554   24.392   18.625   12.827    6.959     | lo=0
 2   4 | 29.433   23.252   17.455   11.628    5.767     | lo=0
 2   8 | 28.973   22.780   16.973   11.142    5.289     | lo=0
 3   2 | 26.135   20.095   14.115    8.172    2.471     | lo=0
 3   4 | 24.553   18.483   12.507    6.582    0.982     | lo=0
 3   8 | 23.927   17.850   11.872    5.952    0.384     | lo=0
 4   2 | 20.536   14.484    8.491    2.746     --       | lo=0
 4   4 | 18.821   12.763    6.791    1.139     --       | lo=0
 4   8 | 18.163   12.105    6.138    0.513     --       | lo=0
```

**모든 9개 (sb,mg) × 2계열에서 lo=0(LSB)이 최선.** 중간 비트 공유는 예외 없이 열세.

## 왜? — offset 1비트당 정확히 −6.02 dB

복원 오차는 `(shared − mid)·s_base·2^lo` 꼴이다 (`mid`=per-elem 윈도우 코드, `shared`=그룹 평균).
공유 필드가 유효 가중치 `2^lo`를 가지므로, 그룹 평균에서의 per-element 편차가 진폭에서 `2^lo`,
**잡음 전력에서 `2^(2lo)`** 만큼 증폭된다 → QSNR이 `lo`당 `20·log10(2) = 6.02 dB` 하락.

실측 step이 정확히 이 값에 수렴한다 (예: naive sb2mg2: −5.69 / −5.93 / −6.02 / −6.16 dB).
첫 step이 6.02보다 약간 작은 것은 새로 per-elem로 보존되는 하위 `lo`비트가 오차를 일부 상쇄하기
때문이며, 그 이득은 공유 필드의 `2^(2lo)` 증폭에 전혀 미치지 못한다.

직관: LSB 공유는 **가장 덜 중요한 비트**를 그룹 공통값으로 대체하므로 mismatch 비용이 최소.
중간 비트를 공유하면 더 무거운 비트를 평균값으로 뭉개고, 대신 per-elem로 지키는 것은 가장 가벼운
하위 비트 — 정밀도 trade가 명백히 손해다. weight 블록 내 중간 비트가 그룹 내에서 상관(거의 상수)이라
평균이 잘 대표한다면 역전 가능하나, Llama weight에선 그런 상관이 관측되지 않는다.

## 결론

- **중간 비트 sharing은 LSB sharing 대비 정밀도 우위가 없다.** iso-bit에서 offset 1비트당 ~6 dB QSNR
  손해이며, 시도한 모든 (sb, mg)에서 lo=0이 최선. → 현행 LSB-타겟 MSAQ 유지가 옳다.
- **검증 한계(QSNR)**: light_ms_results.md의 기존 발견대로 QSNR은 sharing 변형 간 *순위*를 항상
  보장하진 않는다(naive vs light가 QSNR 동급이나 PPL 2× 차이). 다만 본 실험의 신호는 ~6 dB/bit로
  그 노이즈 폭(≤0.3 dB)을 압도하는 *체계적·이론정합적* 차이라 QSNR만으로 결론이 확정적이다.
  PPL 추가 검증은 불필요(가설이 큰 마진으로 기각됨).

재현: `CUDA_VISIBLE_DEVICES=1 python precision/midbit_qsnr.py`
