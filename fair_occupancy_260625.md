# Fair-occupancy E2E — deployed mult=3 vs Blackwell-fair mult=4 (260625)

배포된 `kv_split_count` 기본 multiplier `mult=3`은 **RTX 3090(82 SM) 튜닝**이다. 70-SM
Blackwell에서 이 값은 **중간 배치(B=8)의 MXINT8 KV-decode를 under-occupy**시켜 MSAQ의 KV-scope
우위를 부풀린다(MSAQ는 mult에 둔감 = 이미 잘 occupy). 본 문서는 양 포맷을 **모두 well-occupied**
시키는 공정 설정(`mult=4`)으로 다시 측정해, 부풀림을 제거한 정직한 MSAQ-vs-MXINT8 수치를 남긴다.

- **GPU**: NVIDIA RTX PRO 4000 Blackwell (sm_120, 70 SM), CUDA 13.2, torch 2.12. **idle GPU에서 측정**
  (공유 박스 — GPU 0 경합이 한 차례 측정을 오염시킨 사례 있음. `nvidia-smi`로 빈 GPU 확인 후 측정).
- **Workload**: Llama-3.1-8B 32L, L_in=1024, L_out=128, per-scope robust (u,gs). reps=10.
- **Harness**: `MS_FAST=1 MS_KV_SPLIT_MULT=<3|4> python tests/e2e_perscope2.py --reps 10 --Bs 1,8,32`.
- 비율 `mq/mx`<1 = MSAQ가 MXINT8보다 빠름. `mq/bf`<1 = bf16보다 빠름.

> 대칭성(공정성): `kv_split_count`는 MSAQ·MXINT8가 **공유**하므로 mult 변경은 양쪽에 동일 적용된다
> (format 축만 차이). mult=4는 둘 다 saturate시키는 Blackwell 값으로, 6/8과 동급(추가 이득 없음).

---

## 1. 공정 비교 (mult=4, 둘 다 well-occupied) — total mq/mx

| scope | B=1 | B=8 | B=32 |
|---|--:|--:|--:|
| S1 W-only | **0.84** | 0.98 | 0.98 |
| S2 W+A | **0.89** | 1.01 | 0.99 |
| S3 KV-only | 0.98 | 0.95 | **0.91** |
| S4 W-only+KV | **0.87** | 1.02 | 0.98 |
| S5 W+A+KV | **0.87** | 1.02 | 0.97 |
| S6 W+A+KV+AA | **0.87** | 1.02 | 0.97 |

(`mq/bf` total, mult=4: S1 0.55/0.86/1.16 · S3 0.95/0.76/0.61 · S4 0.55/0.68/0.83 — bf16 대비는 mult 무관.)

## 2. 부풀림이 어디에 있었나 (배포 mult=3 → 공정 mult=4, mq/mx Δ)

| scope · B | 배포 mult=3 | 공정 mult=4 | Δ | 해석 |
|---|--:|--:|--:|---|
| **S3 KV-only B8** | 0.864 | **0.951** | **+0.087** | MXINT8가 mult=3에서 under-occupy → MSAQ 우위 부풀려짐 |
| **S4 W+KV B8** | 0.916 | **1.017** | **+0.100** | 공정하게 보면 **승→패** (타이~근소 열세) |
| 그 외 전부 (B=1, B=32, weight-scope) | — | — | ±0.00~0.01 | mult과 무관, 변화 없음 |

→ 부풀림은 **정확히 KV-scope 중간 배치(B=8)** 에 국한. 마이크로벤치(u2/gs8 Lk2048 B8: mult3 ms/mx
0.854 → mult4 1.325; MXINT8가 408µs→258µs로 빨라지고 MSAQ는 340µs로 둔감)와 E2E가 일치.

## 3. 정직한 결론 (vs MXINT8, 공정 occupancy)

- **B=1 (memory-bound): MSAQ 전 scope 승** (0.84~0.98) — 가장 견고한 우위 (양자화 바이트가 그대로 시간).
- **B=8 (중간 배치): 타이~근소 열세** (0.95~1.02) — 서브바이트 언팩 세금 + occupancy-bound(바이트 미전환).
- **B=32 (대형): KV-only 0.91로 승**, 나머지 타이(0.97~0.99).

즉 MSAQ의 MXINT8 대비 우위는 **B=1과 large-batch KV에 실재**하고, **중간 배치(B=8)에선 타이~근소
열세**다 — 배포(mult=3) 설정이 KV-scope B=8에서 가렸던 진실. (vs bf16은 mult 무관하게 동일:
B=1·B=8 광범위 승, weight-scope B=32 패.)

출처: `tests/e2e_perscope2.py` (reps=10, GPU1), 원자료 scratchpad `jsonl_mult{3,4}.jsonl`.
배경/구조적 분석: 메모리 `msaq-vs-mxint8-{w,kv}-decode-state`.
