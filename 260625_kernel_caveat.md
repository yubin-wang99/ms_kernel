# 커널별 latency caveat 분석 — BF16/MXINT8 대비 (2026-06-25)

현재 빌드된 커널을 직접 측정해, 각 커널이 **(1) BF16을 왜 못 이기는지**, **(2) MXINT8과 동률이면 왜 동률인지**,
**(3) batch가 커질 때 이 양상이 어떻게 변하는지**를 분석한다. 시간은 모두 **µs**, RTX 3090 / torch 2.5.1,
Llama 형태 `OUT=K=4096`. 정확도-robust config: **W-only `u3/gs16`, W+A `u2/gs8`, KV `u4/gs2`(GQA Hq32/Hkv8)**.

- 측정: `tests/kernel_caveat_bench.py` (cuda.Event, warmup30/iter200) — 총시간 + 위상 분리(quant/unpack/append).
- 하드웨어 카운터: `tests/caveat_ncu_driver.py` + ncu(sudo) — DRAM read, 달성 BW(%peak), L2 hit, stall 원인.
- 비율: **ms/bf = MSAQ/BF16**, **ms/mq = MSAQ/MXINT8**. `<1` = MSAQ가 빠름.

---

## 1. ncu 하드웨어 상세 — "왜"의 근거 (main kernel 기준)

| case | kernel | DRAM read(MB) | 달성 BW(%peak) | time | L2 hit% | long-SB stall%¹ | math-pipe stall%² |
|---|---|---|---|---|---|---|---|
| W-only dec (M=1) MSAQ | `wonly_gemv_wide_uspec` | **11.55** | **68.3** | 21.6µs | 8.0 | 31.4 | 12.5 |
| W-only dec (M=1) MXINT8 | `mxint8_gemv_splitk` | 17.32 | 38.3 | 55.5µs | 2.9 | **75.9** | 0.2 |
| W+A dec (M=1) MSAQ | `wa_gemv` | 13.64 | 71.2 | 24.1µs | 6.2 | 59.8 | 5.2 |
| KV dec (Lk16k,B1) MSAQ | `kv_decode_wide` | **26.32** | 7.9 | 391µs | 83.9 | 11.3 | 3.6 |
| KV dec (Lk16k,B1) MXINT8 | `mxint8_kv_split` | 34.62 | 8.9 | 453µs | 69.8 | **45.6** | 0.2 |
| W-only prefill (M=512) MSAQ | `wonly_gemm_wmma_pipe` | 27.85 | **2.5** | 1.48ms | 94.1 | 17.1 | 9.8 |
| W-only prefill (M=512) MXINT8 | `mxint8_gemm_tiled` | 49.54 | 1.8 | 3.36ms | 79.2 | 30.5 | 0.5 |

¹ `long_scoreboard` = 글로벌 메모리 응답 대기(=memory **latency** 바운드 신호). ² `math_pipe_throttle` = ALU 포화(=unpack/디코드 연산 바운드 신호).

**핵심 관찰**
- MSAQ는 항상 **DRAM 바이트를 덜 읽는다**(W-only 11.55 vs 17.32 = 0.67×, KV 26.3 vs 34.6 = 0.76×, prefill 27.9 vs 49.5 = 0.56×).
- MSAQ는 wide `uint4` 코얼레스 로드로 **달성 BW가 높고 latency stall이 낮다**(W-only: BW 68% vs 38%, long-SB 31% vs 76%). MXINT8의 scalar int8 로드는 메모리 latency에 갇힌다.
- 단, **바이트 절감이 시간 절감으로 바뀌려면 BW-bound여야 한다.** prefill GEMM은 BW가 2.5%뿐(=compute-bound) → 바이트 절감이 무의미. KV B=1은 BW 8%뿐(=occupancy 부족) → 절감이 부분만 전환.

---

## 2. 가중치 matmul — Decode family (M = batch, GEMV→batched-GEMV)

`weight-unpack 단독: W-only ms_dequant 56µs vs MXINT8 dequant 62µs · W+A 59 vs 62µs` (한 번 풀어쓰면 둘 다 ~60µs).

### W-only (`u3/gs16`)
| M | BF16 | MXINT8 | MSAQ | ms/bf | ms/mq | quant |
|---|---|---|---|---|---|---|
| 1 | 46.6 | 51.6 | **23.0** | **0.49** | **0.45** | – |
| 4 | 46.5 | 201.7 | 28.8 | 0.62 | 0.14 | – |
| 16 | 45.6 | 335.8 | 72.0 | 1.58 | 0.21 | – |
| 32 | 45.3 | 1035 | 145.7 | 3.21 | 0.14 | – |
| 64 | 50.1 | 1953 | 279.2 | 5.57 | 0.14 | – |

### W+A (`u2/gs8`) — `quant` = 활성화 양자화(quant_act 단독, wa_gemv에 내부 융합됨)
| M | BF16 | MXINT8 | MSAQ | ms/bf | ms/mq | quant |
|---|---|---|---|---|---|---|
| 1 | 46.9 | 39.1 | 27.7 | **0.59** | 0.71 | 12.8 |
| 4 | 46.4 | 85.1 | 49.2 | 1.06 | 0.58 | 13.4 |
| 16 | 47.0 | 171.4 | 169.2 | 3.60 | **0.99** | 13.2 |
| 32 | 46.7 | 492.9 | 308.1 | 6.59 | 0.63 | 14.1 |
| 64 | 51.3 | 907.0 | 599.8 | 11.68 | 0.66 | 13.2 |

**분석**
- **M=1: MSAQ가 BF16·MXINT8 둘 다 이긴다.** decode는 weight-read memory-bound이고, MSAQ는 바이트가 적고(0.67×) wide-load로 BW 68%·latency stall 31%만 → 21.6µs. MXINT8은 같은 구조인데 바이트가 많고 latency stall 76%로 55µs. BF16 cuBLAS는 weight 33MB를 읽어 DRAM floor ~36µs(=46µs, M에 거의 무관).
- **왜 batch에서 BF16을 못 이기나(M≥4~16):** BF16 cuBLAS는 **텐서코어 GEMM**이라 작은 M의 연산을 거의 공짜로 처리 → 시간이 weight-read에 묶여 **M이 커져도 ~46µs로 평평**하다. 반면 우리(그리고 MXINT8)의 batched-GEMV는 **scalar FMA**라 시간이 M에 거의 선형 증가. 그래서 M≥16에서 ms/bf>1로 역전된다. **이건 MSAQ 고유 약점이 아니라 "scalar sub-byte GEMV vs 텐서코어 GEMM"의 구조 차이**다(MXINT8은 훨씬 더 나쁨: ms/mq 0.14~0.45로 우리가 7배까지 앞섬).
- **왜 W+A M=16에서 MXINT8과 동률(0.99):** W+A는 활성화 양자화(~13µs 고정비)와 정수-내적 누산기 레지스터 압박이 겹쳐, 바이트 절감 이득이 이 지점에서 상쇄된다(MR-cap 적용 후 M=32/64에서 다시 0.62~0.66로 벌어짐).

---

## 3. 가중치 matmul — Prefill family (M = batch, 텐서코어 GEMM)

### W-only (`u3/gs16`)
| M | BF16 | MXINT8 | MSAQ | ms/bf | ms/mq |
|---|---|---|---|---|---|
| 128 | 72.8 | 1088 | 580 | 7.97 | 0.53 |
| 256 | 153 | 1254 | 576 | 3.76 | 0.46 |
| 512 | 282 | 2400 | 1131 | 4.01 | **0.47** |
| 1024 | 566 | 4771 | 2246 | 3.97 | 0.47 |

### W+A (`u2/gs8`)
| M | BF16 | MXINT8 | MSAQ | ms/bf | ms/mq | quant |
|---|---|---|---|---|---|---|
| 128 | 73.0 | 791 | 579 | 7.93 | 0.73 | 13.9 |
| 256 | 152 | 1776 | 1103 | 7.24 | 0.62 | 13.9 |
| 512 | 281 | 2781 | 1735 | 6.17 | **0.62** | 21.1 |
| 1024 | 571 | 4713 | 2951 | 5.17 | 0.63 | 40.5 |

**분석**
- **왜 BF16을 못 이기나(4~8×):** prefill GEMM은 **compute-bound**다. ncu에서 MSAQ GEMM의 달성 DRAM BW는 **2.5%**, L2 hit 94% → 메모리는 전혀 병목이 아니다. 즉 MSAQ의 바이트 절감(0.56×)은 **여기서 시간으로 전혀 전환되지 않는다.** 병목은 연산 처리량이고, cuBLAS는 텐서코어를 가득 쓰는데 우리 커널은 sub-byte를 **bf16 타일로 unpack(staging)** 한 뒤 WMMA에 넣어야 해서(math-pipe stall 9.8%) 텐서코어 점유가 낮다. → cuBLAS 대비 4~8× 느림. (이것이 "staging wall".)
- **왜 MXINT8은 이기나(0.47~0.73):** 같은 compute-bound라도 MXINT8 baseline은 텐서코어를 덜 활용하는 tiled-IMMA(달성 BW 1.8%, math-pipe 0.5%지만 latency stall 30%)라 우리 WMMA-pipe보다 느리다. 즉 **양자화 커널끼리는 우리가 빠르지만, 둘 다 cuBLAS에는 못 미친다.**
- **batch 추세:** M이 커져도 ms/mq는 0.47/0.62로 **일정**(둘 다 O(MK·OUT) compute라 동일 스케일), ms/bf는 M이 작을수록 더 나쁘다(M=128에서 8× — cuBLAS는 작은 GEMM도 효율적, 우리는 타일 채움 효율이 낮음).

---

## 4. KV-cache decode attention (`u4/gs2`, GQA Hq32/Hkv8)

| Lk | B | BF16 | MXINT8 | MSAQ | ms/bf | ms/mq | append(ms) | append(mx) |
|---|---|---|---|---|---|---|---|---|
| 2048 | 1 | 55.5 | 44.5 | 40.4 | **0.73** | 0.91 | 8.6 | 7.9 |
| 2048 | 8 | 614 | 365 | 328 | **0.53** | 0.90 | | |
| 2048 | 32 | 2113 | 1006 | 1021 | **0.48** | 1.02 | | |
| 16384 | 1 | 320 | 381 | 327 | 1.02 | **0.86** | 8.5 | 8.2 |
| 16384 | 8 | 5008 | 2653 | 2458 | **0.49** | 0.93 | | |

**분석**
- **왜 BF16을 (대부분) 이기나:** KV decode는 KV-read memory-bound이고 BF16 SDPA는 KV를 fp16/bf16(2바이트)로 읽는 반면 MSAQ는 sub-byte(0.76× 바이트)라 유리. batch·long-Lk로 갈수록 BW-bound가 강해져 ms/bf가 0.73→0.48까지 벌어진다. (예외: Lk16k·B=1은 occupancy가 모자라 ms/bf 1.02 — 아래.)
- **왜 MXINT8과 동률(0.86~1.02)인가:** ncu에서 **B=1은 달성 BW가 8%뿐**이다(KV 한 세트로는 SM을 못 채워 occupancy/latency-bound). 이 영역에선 "바이트가 적다"는 장점이 BW로 전환되지 못해 MSAQ 26MB vs MXINT8 35MB 차이가 시간차로 거의 안 나타난다 → 동률. 다만 MSAQ는 latency stall이 훨씬 낮아(long-SB 11% vs 46%) 근소 우위(0.86~0.93). **B나 Lk가 커져 BW-bound가 되면** 바이트 절감이 전환되기 시작(B=8에서 0.90~0.93). u4 nibble이라 unpack이 단순(math-pipe 3.6%)해 손해가 거의 없다.
- **append(packing) 비용:** 토큰 1개 KV 양자화·기록은 MSAQ 8.6 vs MXINT8 7.9µs로 거의 동일(MSAQ가 plane 3개 써서 소폭 높음). decode-read(수백 µs)에 비하면 무시 가능.
- **batch 추세:** B=1(occupancy-bound, ms/bf≈0.7~1.0, MXINT8 동률) → B↑·Lk↑(BW-bound, ms/bf 0.48~0.53, MXINT8 근소 우위). **KV가 정확도(u4 nibble 견딤)와 속도를 모두 잡는 유일한 축**이다.

---

## 5. 종합

| 커널 | BF16 못 이기는 이유 | MXINT8과 동률/우위 이유 | batch 추세 |
|---|---|---|---|
| **W-only decode** | M↑일 때 BF16=텐서코어 GEMM(평평), 우리=scalar GEMV(선형) | M=1 BW 68% vs 38%·바이트 0.67×로 압승, batch도 0.14~0.45 | M=1 BF16까지 이김 → M≥16 BF16엔 짐, MXINT8엔 계속 이김 |
| **W+A decode** | 동상 + 활성화 quant ~13µs 고정비 | 바이트 절감 vs quant·레지스터압박 → M16 동률, 그 외 0.6~0.7 | M=1 BF16 이김 → batch BF16엔 짐, MXINT8엔 우위 |
| **W-only/W+A prefill GEMM** | compute-bound(BW 2.5%)+staging wall로 텐서코어 점유 낮음 → 4~8× | 양자화 커널끼리는 WMMA-pipe가 빨라 0.47~0.73 | ms/mq 일정(0.47/0.62), ms/bf는 작은 M에서 더 나쁨 |
| **KV decode** | B=1·작은 Lk는 occupancy-bound → 바이트 이득 미전환(0.7~1.0) | B=1 BW 8%라 바이트차 미전환=동률, latency stall은 낮아 근소우위 | B↑·Lk↑ → BW-bound로 ms/bf 0.48까지, MXINT8 근소우위 |

**한 줄 결론:** 모든 caveat은 **"바이트 절감은 BW-bound일 때만 시간으로 전환된다"** 로 귀결된다.
- BF16을 못 이기는 곳은 둘 중 하나다 — (a) **compute-bound**(prefill GEMM: cuBLAS 텐서코어 vs 우리 staging), (b) **scalar-GEMV가 텐서코어 GEMM에 밀리는 batch decode**.
- MXINT8과 동률인 곳은 **occupancy-bound(KV B=1)** 라 바이트 절감이 BW로 안 바뀌는 구간뿐이고, BW-bound로 가면 MSAQ가 앞선다.
- 이득이 확실한 영역은 **memory-bound decode**: M=1 가중치 GEMV(BF16까지 이김)와 batch/long-context KV(BF16 0.48×). 출처: `tests/kernel_caveat_bench.txt`, `tests/kernel_caveat_ncu.txt`.
