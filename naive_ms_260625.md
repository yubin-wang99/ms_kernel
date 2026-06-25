# naive-ms — 병목 분리용 단순 mantissa-sharing (260625)

현재 MSAQ(ms)의 **병목이 양자화 알고리즘(decompose)인지 저장 포맷(서브바이트 unpack)인지**를
분리하기 위한 레퍼런스 포맷. **최종 저장 포맷은 ms와 동일**(upper (8-u)-bit 평면 + shared u-bit
평면 + E8M0 scale)이되, 코드를 **가장 단순하게** 유도한다.

## 1. 설계

```
q8       = clip(round(x / s), -127, 127)        # 그냥 MXINT8 int8
upper[k] = q8[k] >> u                            # 산술 시프트 → signed (8-u)-bit (상위 비트)
low[k]   = q8[k] & (2^u - 1)                     # 하위 u-bit (unsigned, [0,2^u))
shared[g]= clip(round(mean_g(low)), 0, 2^u-1)    # 그룹의 하위비트 평균 → 하나의 unsigned INT_u
복원     : w = (upper << u) | shared = upper·2^u + shared      # ADD 아닌 bit-concat
```

- ms와의 차이는 **인코딩(offline)뿐**: ms는 `x`에서 float 도메인 decompose(별도 coarse 라운딩 +
  잔차 평균)를 하지만, naive-ms는 **이미 양자화된 int8 q8의 하위 비트를 정수 평균**해 공유한다.
- shared가 **unsigned**라 복원이 sign-extend·add 없는 **OR-concat** → ms-unsigned 언팩 커널을 그대로 사용.
- 구현: `ms_lib/pack.py`의 `decompose_naive`/`pack_weight_naive`/`dequant_weight_naive`.
  커널(공정 비교용, "동일 환경"): unsigned concat 언팩의 **신규 커널** —
  `wonly_gemv_wide_unsigned`/`wonly_gemv_batched_unsigned`/`wa_gemv(_batched)_unsigned`/
  `ms_dequant_bf16_unsigned`(weight), `kv_kdot_unsigned`(KV). MXINT8·ms와 같은 splitK/occupancy/
  와이드로드 구조를 공유한다(format 축만 차이). harness에 `naive_*` 경로 추가.

## 2. 정확도 (QSNR, dB ↑ 높을수록 정확) — `decompose_*` 비교

| 분포 | u/gs | MXINT8 | ms(signed) | ms-unsigned | **naive-ms** |
|---|---|--:|--:|--:|--:|
| normal | 3/16 | 41.9 | 24.0 | 24.0 | **24.0** |
| normal | 2/8  | 41.9 | 30.2 | 30.2 | **30.1** |
| heavy  | 3/16 | 37.9 | 21.4 | 16.6 | **17.9** |
| heavy  | 2/8  | 37.7 | 26.7 | 23.0 | **24.6** |

- **normal: naive-ms ≈ ms**(동등). 모든 sharing 포맷은 MXINT8(전체 int8 저장)보다 12~18 dB 낮다 —
  이게 sharing의 **정확도↔바이트 트레이드**(naive-ms·ms는 MXINT8의 ~0.66~0.78× 바이트).
- **heavy-tail: ms(signed) > naive-ms > ms-unsigned**. naive-ms는 중간 — 단순 정수평균이라 ms의
  최적화된 잔차 decompose보다 약간 손해지만, 같은 floor 계열인 ms-unsigned보다는 낫다.

## 3. 커널 속도 (공정 비교, ratio <1 = 더 빠름)

### 3a. KV K-dot (커널레벨, Lk=2048, GPU1 idle, wide-load MXINT8 baseline)

| u/gs · B | ms(sig) µs | naive µs | MXINT8 µs | **nv/ms** | nv/mx | ms/mx |
|---|--:|--:|--:|--:|--:|--:|
| 3/16 B8  | 44.7 | 44.0 | 40.0 | **0.98** | 1.10 | 1.12 |
| 3/16 B32 | 184.0 | 181.3 | 171.4 | **0.99** | 1.06 | 1.07 |
| 2/8  B8  | 44.7 | 43.5 | 39.9 | **0.97** | 1.09 | 1.12 |
| 2/8  B32 | 191.3 | 184.5 | 168.2 | **0.96** | 1.10 | 1.14 |

→ **naive-ms ≈ ms (0.96~0.99)**. isolated K-dot은 integer-unpack-bound라 ms·naive-ms 둘 다 wide-load
MXINT8에 ~6~14% 패배(언팩 ALU > 바이트 절감). **공정성 주의**: MXINT8 K-dot은 반드시 **wide-load**
(int4×2)로 측정 — scalar 바이트 로드로 재면 MXINT8가 인위적으로 2.5× 느려져 naive가 0.37로 거짓 승리.

### 3b. Weight decode E2E (S1 W-only, S2 W+A) — 32L Llama-3.1-8B, KV=bf16, GPU1, reps=10

total(prefill+decode) µs와 비율. `nv/ms`<1=naive가 빠름.

| scope | B | bf16 | mxint8 | ms | naive | **nv/ms** | nv/mx | ms/mx | nv/bf |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| S1 W-only | 1 | 3856 | 2523 | 2113 | 2112 | **1.000** | 0.837 | 0.837 | 0.548 |
| | 8 | 8135 | 7152 | 7131 | 7125 | **0.999** | 0.996 | 0.997 | 0.876 |
| | 32 | 21602 | 25613 | 25233 | 25277 | **1.002** | 0.987 | 0.985 | 1.170 |
| S2 W+A | 1 | 3863 | 2565 | 2280 | 2257 | **0.990** | 0.880 | 0.889 | 0.584 |
| | 8 | 8154 | 7191 | 7281 | 7242 | **0.995** | 1.007 | 1.013 | 0.888 |
| | 32 | 21613 | 25632 | 25346 | 25434 | **1.003** | 0.992 | 0.989 | 1.177 |

→ **naive-ms ≈ ms (nv/ms 0.990~1.003), 전 구간 사실상 동일.** vs MXINT8도 ms와 같다(nv/mx≈ms/mx):
B=1 승(0.84/0.88), B=8 타이(≈1.00), B=32 타이(0.99). 즉 weight 경로에서도 naive-ms와 ms는 **구분 불가**.

## 4. 결론 — 병목 분리

> **naive-ms ≈ ms — KV K-dot 0.96~0.99, weight E2E 0.99~1.00 (사실상 구분 불가).** 정교한 float
> decompose(ms)와 단순 정수평균(naive-ms)이 거의 같은 정확도(normal)에 **동일한 커널 시간**을 낸다.

- **decompose 알고리즘은 런타임에서 free** — 인코딩은 offline이고, 커널은 동일 포맷(upper+shared
  평면)을 읽어 복원할 뿐. ms의 병목은 decompose 정교함이 **아니다**.
- **병목 = 저장 포맷 자체** — 서브바이트 (8-u)-bit upper 추출(언팩 ALU) + 바이트 수. 이건 ms·naive-ms
  공통이며 MXINT8(int8 load, 언팩 0)와의 격차의 근원. memory-bound일 때만 바이트 우위가 시간으로
  전환되고(B=1·대형배치/긴문맥 KV), unpack-bound 구간(K-dot, weight batched)에선 MXINT8에 진다.
- 즉 naive-ms는 ms 대비 **속도 동률·정확도 약간 손해**(heavy-tail) — ms의 decompose는 **공짜로
  정확도만 더 사는** 것임을 보여준다.

vs MXINT8 비율은 §3 및 공정 occupancy E2E(`fair_occupancy_260625.md`) 참조. 배경: 메모리
`msaq-vs-mxint8-{w,kv}-decode-state`.
