# 커널 latency caveat 분석 — BF16/MXINT8 대비 (ver.260625, E2E 기준)

현재 production 커널(ver.260625, 오늘 커밋 `6fbba37`+`b83ba6b` 포함)이 **실제 호출되는 경로**로 측정한
per-scope **E2E**(32-layer Llama-3.1-8B, TTFT + 통합 decode) 분석이다. 각 커널이 **(1) BF16을 왜 못
이기는지/타이인지**, **(2) MXINT8과 동률이면 왜인지**, **(3) batch가 커질 때 어떻게 변하는지**를 본다.

- **측정**: `tests/e2e_perscope_260625.py`(=`e2e_perscope2`의 출력명만 변경) → `tests/harness_perscope_results_260625.md`.
  (L_in,L_out)=(1024,128), per-scope robust u/gs, B∈{1,8,32}. 시간 µs, 비율 `<1`=MSAQ 빠름.
- **배포 dispatch**(`kernel_ver260625.md` §1): prefill = `ms_dequant_bf16`+cuBLAS · decode B=1 = wide-GEMV ·
  **B=2..15 = shared-activation/W+A-float batched GEMV(오늘 수정)** · **B≥16 = dequant+cuBLAS** ·
  KV = `kv_decode_wide`(u,gs 특수화).
- **하드웨어 근거**: `tests/caveat_ncu_driver.py`+ncu(sudo), B=1 deployed 커널의 DRAM/L2/stall(§4).

---

## 1. Prefill (TTFT, 1024 tok) — bf16과 "타이"가 천장

| scope | B | bf16 | MXINT8 | MSAQ | mq/bf | mq/mx |
|---|--:|--:|--:|--:|--:|--:|
| S1 W-only | 1 | 254 | 280 | 280 | 1.10 | 1.00 |
| | 8 | 2020 | 2036 | 2028 | **1.00** | 1.00 |
| | 32 | 7795 | 7803 | 7772 | **1.00** | 1.00 |
| S2 W+A | 8 | 2019 | 2034 | 2026 | 1.00 | 1.00 |

**왜 BF16을 못 이기나(타이가 한계):** prefill은 intensity≈1024 ≫ ridge 76 = **compute-bound**다. 가중치
바이트 절감(0.56×)은 시간으로 **전혀 전환 안 됨**(가중치 read가 전체의 ~7%). 그래서 production은
"양자화 weight를 **한 번** bf16로 풀어(`ms_dequant_bf16`) **cuBLAS 텐서코어**에 태우는" 전략 →
연산은 cuBLAS와 동일, 추가는 dequant 1회뿐 → **bf16과 타이**. (옛 fused-WMMA는 MMA를 굶겨 4× 느렸음;
이번 측정의 옛 microbench 4~8× 패배는 그 비-production 커널이었던 게 원인 — 정정됨.)
**왜 MXINT8과 타이:** 둘 다 dequant+cuBLAS로 동일 구조 → mq/mx≈1.00. (B=1의 1.10은 TTFT가 작아
dequant 1회 오버헤드가 보일 뿐, B↑이면 1.00로 흡수.)

---

## 2. Decode (128 step 통합) — scope별

절대값 `bf/mx/mq`(µs)와 비율.

| scope (u/gs) | B | bf16 | MXINT8 | MSAQ | mq/bf | mq/mx |
|---|--:|--:|--:|--:|--:|--:|
| **S1 W-only** (u3/gs16) | 1 | 2881 | 1876 | 1562 | **0.54** | **0.83** |
| | 8 | 5733 | 4784 | 4725 | **0.82** | 0.99 |
| | 32 | 12512 | 15635 | 15441 | 1.23 | 0.99 |
| **S2 W+A** (u2/gs8) | 1 | 2894 | 1904 | 1753 | **0.61** | 0.92 |
| | 8 | 5720 | 4799 | 4812 | **0.84** | 1.00 |
| | 32 | 12511 | 15660 | 15575 | 1.24 | 0.99 |
| **S3 KV-only** (u4/gs2) | 1 | 2895 | 2720 | 2743 | 0.95 | 1.01 |
| | 8 | 5721 | 3934 | 3865 | **0.68** | 0.98 |
| | 32 | 12519 | 5197 | 5253 | **0.42** | 1.01 |
| **S4 W-only+KV** (u2/gs8) | 1 | 2895 | 1704 | 1578 | **0.55** | 0.93 |
| | 8 | 5721 | 2867 | 3175 | **0.55** | 1.11 |
| | 32 | 12503 | 8355 | 9167 | **0.73** | 1.10 |
| **S5 W+A+KV** (u2/gs8) | 1 | 2880 | 1741 | 1641 | **0.57** | 0.94 |
| | 8 | 5716 | 2904 | 3233 | **0.57** | 1.11 |
| | 32 | 12489 | 8355 | 9172 | **0.73** | 1.10 |

(S6=W+A+KV+AA는 decode에서 Q를 bf16으로 읽어 **S5와 동일 커널** → 수치 S5와 같음. AA는 정확도 비용이지 latency 비용 아님.)

**분석 — 왜 BF16을 못 이기나 / 왜 MXINT8과 동률인가**

- **B=1: 모든 scope가 BF16·MXINT8 둘 다 이긴다(또는 KV는 BF16만).** decode 1토큰은 intensity≈1 ≪ ridge
  = **memory-bound** → 가중치/ KV 바이트가 그대로 시간. MSAQ가 가장 빛나는 영역(S1 0.54, S4/S5 0.55~0.57).
- **가중치 scope(S1/S2)의 B=32에서 BF16에 짐(1.23~1.24):** B≥16은 dequant+cuBLAS 경로인데, decode는
  스텝마다 weight를 bf16로 다시 풀어야 해서(**dequant 비용이 토큰 1개에 amortize 안 됨**) bf16 순정
  cuBLAS보다 느리다. MXINT8도 같은 경로라 둘 다 bf16에 짐 → **mq/mx≈0.99(동률).** "B≥16은 bf16과
  타이"는 prefill 얘기였고, **decode B≥16은 dequant 오버헤드 때문에 오히려 bf16에 손해**임이 이번
  E2E로 드러남.
- **KV scope(S3)는 batch로 갈수록 BF16 압승(0.95→0.42):** KV read는 batch가 커져도 memory-bound가
  유지되고 KV 바이트 절감이 누적된다. **u4/gs2 nibble**이라 unpack이 단순 → **MXINT8과 동률(1.01)**
  하면서 bf16엔 0.42까지. (KV가 정확도·속도를 모두 잡는 유일한 축.)
- **결합 scope(S4/S5)의 decode가 MXINT8에 근소 패배(B=8/32 mq/mx 1.10~1.11):** S4/S5는 W+A 정확도
  때문에 **KV도 u2/gs8(non-nibble)** 을 써야 한다. u2 KV unpack은 straddle이라 MXINT8 직접 int8 read보다
  무겁다 → KV read mq/mx>1. 그래도 bf16엔 0.55~0.73으로 이긴다. **즉 "MXINT8을 이기려면 KV가 u4
  nibble이어야 하는데 W+A 동반 정확도는 u2를 강제"** → S3(KV-only, u4)는 MXINT8 타이지만 S4/S5는 근소 손해.

---

## 3. Total (prefill+decode) + batch 추세

| scope | B=1 | B=8 | B=32 |
|---|--:|--:|--:|
| S1 W-only | **0.59** | **0.87** | 1.14 |
| S2 W+A | **0.64** | **0.88** | 1.15 |
| S3 KV-only | 0.95 | **0.76** | **0.65** |
| S4 W-only+KV | **0.59** | **0.67** | **0.84** |
| S5 W+A+KV | **0.61** | **0.68** | **0.84** |
| S6 W+A+KV+AA | **0.61** | **0.68** | **0.84** |

(값 = total mq/bf, `<1`=MSAQ 빠름)

**batch 추세 요약**
- **가중치-only(S1/S2):** B=1 win(0.59) → B=8 still win(0.87) → **B=32 lose(1.14~1.15)**. decode 가중치
  matmul이 B≥16에서 dequant+cuBLAS 오버헤드로 역전.
- **KV 포함(S3/S4/S5/S6):** **모든 B에서 bf16 승**(0.59~0.95). KV의 memory-bound 이득이 batch로
  커져, 가중치 쪽 손해를 상쇄·역전. 특히 **KV-only는 batch로 갈수록 더 벌어짐(0.95→0.65)**.
- prefill은 전 구간 bf16 타이(≈1.00)라 total의 방향은 **decode가 결정**한다(L_out=128에서도 decode가
  total의 대부분).

---

## 4. ncu 하드웨어 근거 — B=1 deployed decode 커널 내부 ("왜")

| 커널 (deployed, B=1) | DRAM read(MB) | 달성 BW(%peak) | time | L2 hit% | long-SB stall%¹ | math-pipe%² |
|---|--:|--:|--:|--:|--:|--:|
| W-only `wonly_gemv_wide_uspec` (MSAQ) | **11.55** | **68.3** | 21.6µs | 8.0 | 31.4 | 12.5 |
| W-only MXINT8 baseline | 17.32 | 38.3 | 55.5µs | 2.9 | **75.9** | 0.2 |
| W+A `wa_gemv` (MSAQ) | 13.64 | 71.2 | 24.1µs | 6.2 | 59.8 | 5.2 |
| KV `kv_decode_wide` (MSAQ, Lk16k) | 26.32 | 7.9 | 391µs | 83.9 | 11.3 | 3.6 |
| KV MXINT8 baseline (Lk16k) | 34.62 | 8.9 | 453µs | 69.8 | 45.6 | 0.2 |

¹ long_scoreboard = 글로벌 메모리 응답 대기(=memory **latency** 바운드). ² math_pipe = ALU 포화(unpack 연산 바운드).

- **왜 B=1에서 MSAQ가 이기나:** MSAQ는 **바이트가 적고**(0.67×/0.76×) wide `uint4` 코얼레스 로드로
  **달성 BW가 높고 latency stall이 낮다**(W-only 68% vs 38%, long-SB 31% vs 76%). MXINT8 scalar
  int8 로드는 메모리 latency에 갇힘. KV도 long-SB 11% vs 46%로 MSAQ가 덜 막힘.
- **왜 KV는 동률에 가깝나:** **B=1 KV는 달성 BW가 8%뿐**(KV 한 세트로 SM 점유 부족=occupancy-bound).
  이 구간에선 바이트 절감(26 vs 35MB)이 BW로 전환되지 못해 시간차가 작다 → MXINT8과 동률, latency
  stall 차이만큼만 근소 우위. **B↑/Lk↑로 BW-bound가 되어야** 바이트 절감이 시간으로 전환(§2의 S3 0.42).

---

## 5. 종합

| 커널/scope | BF16 못 이기는(또는 타이) 이유 | MXINT8 동률/우위 이유 | batch 추세 |
|---|---|---|---|
| **Prefill GEMM** | compute-bound → 바이트 절감 무의미, dequant+cuBLAS는 cuBLAS와 동일 연산 = **타이가 천장** | 둘 다 dequant+cuBLAS 동일 경로 → ≈1.00 | B 무관 ≈1.00 |
| **W-only/W+A decode** | B=1은 이김; **B≥16은 dequant+cuBLAS 오버헤드가 토큰당 amortize 안 됨 → bf16에 짐** | B≥16 둘 다 dequant+cuBLAS → ≈0.99 | B1 win(0.54) → B32 lose(1.23) |
| **KV decode** | B=1·작은 Lk는 occupancy-bound(BW 8%)라 바이트 이득 미전환 → bf16 0.95 | u4/gs2 nibble이라 unpack 단순 → 동률(1.01) | B↑·Lk↑로 BW-bound → 0.42까지 압승 |
| **W+A+KV(S4/S5)** | bf16엔 전 구간 승(0.55~0.84) | **KV가 u2(non-nibble) 강제 → MXINT8에 근소 패배(1.10)** | total: B1 0.59 → B32 0.84, 전 구간 win |

**한 줄 결론**: 모든 caveat은 **"바이트 절감은 memory-bound(BW-bound)일 때만 시간으로 전환된다"**.
- **BF16 못 이김** = ⓐ prefill(compute-bound, dequant+cuBLAS로 타이가 한계) · ⓑ 가중치 decode B≥16
  (dequant 오버헤드 미-amortize). 그 외 B=1과 KV-batch는 전부 이김.
- **MXINT8 동률/패배** = ⓐ prefill/B≥16(같은 dequant+cuBLAS) · ⓑ KV가 u2 non-nibble로 강제되는
  S4/S5(unpack이 int8 read보다 무거움). u4 nibble을 쓸 수 있는 S3 KV-only는 동률.
- **확실한 win** = **memory-bound decode** — 모든 scope의 B=1, 그리고 batch/long-context **KV**(bf16 0.42).

출처: `tests/harness_perscope_results_260625.md`(E2E), `tests/kernel_caveat_ncu.txt`(ncu), `kernel_ver260625.md`(dispatch 지도).
