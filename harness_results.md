# End-to-End Harness 결과 (Llama-3.1-8B, RTX 3090) — CUDA-graph, 4 시나리오

`tests/harness.py` 실측. 32-layer full forward, **prefill=800 / decode=3880**, GQA 32Q:8KV,
vocab 128256. glue(RMSNorm·RoPE·SwiGLU·SDPA)는 bf16 공통. 시간 ms(별도 표기 외).
설계는 [harness_design.md](harness_design.md), 커널별 분석은 [kernel_ver1.md](kernel_ver1.md).

**측정 방식:** decode TPOT는 **CUDA graph**로 측정(스텝 1개를 컨텍스트 체크포인트마다 capture→replay
→trajectory 적분)해 Python per-op dispatch 오버헤드를 제거. weight/KV 양자화를 **독립 knob**으로
분리해 4가지 시나리오로 적용 대상별 기여를 격리. (반복 graph capture가 같은 프로세스의 다음 eager
prefill을 wedge해서 시나리오마다 **별도 subprocess**로 격리.)

## 4 시나리오 × {MXINT8, MSAQ-u2/u3/u4} (bf16 baseline / 각 S의 MXINT8 대비)

각 시나리오의 **MXINT8 커널이 그 시나리오의 baseline**(`/mxint8`은 같은 S의 MXINT8 대비).
아래는 **커널 최적화 라운드 후**(KV write occupancy, KV read streaming unpack, W+A GEMV
qx-staging — §"커널 최적화 라운드" 참조). 괄호는 최적화 전 `/mxint8`.

| 시나리오 | 포맷 | TTFT | TPOT | total | /bf16 | /mxint8 |
|---------|------|------|------|-------|-------|---------|
| baseline | bf16 | 275 | 37.5 | 145.6s | 1.00 | — |
| **S1 W-only** | MXINT8 | 1597 | 37.1 | 145.5s | 1.00 | — |
| | MSAQ-u2 | 1720 | 34.6 | 136.1s | 0.93 | 0.94 |
| | MSAQ-u3 | 1697 | 34.2 | 134.6s | 0.92 | 0.92 |
| | **MSAQ-u4** | 1496 | 29.4 | 115.7s | **0.79** | **0.80** |
| **S2 W+A** | MXINT8 | 1585 | 32.8 | 128.9s | 0.89 | — |
| | MSAQ-u2 | 1480 | 33.6 | 131.9s | 0.91 | 1.02 (1.06) |
| | MSAQ-u3 | 1447 | 33.1 | 129.9s | 0.89 | 1.01 (1.05) |
| | **MSAQ-u4** | 1212 | 29.7 | 116.3s | **0.80** | 0.90 |
| **S3 KV-only** | MXINT8 | 276 | 25.9 | 100.6s | 0.69 | — |
| | MSAQ-u2 | 277 | 25.1 | 97.8s | 0.67 | **0.97** (1.07) |
| | MSAQ-u3 | 277 | 25.1 | 97.6s | 0.67 | **0.97** (1.07) |
| | **MSAQ-u4** | 277 | 23.9 | 93.0s | **0.64** | 0.92 |
| **S4 W-only+KV** | MXINT8 | 1600 | 25.5 | 100.7s | 0.69 | — |
| | MSAQ-u2 | 1720 | 22.5 | 88.9s | 0.61 | **0.88** (0.98) |
| | MSAQ-u3 | 1699 | 22.0 | 87.2s | 0.60 | **0.87** (0.96) |
| | **MSAQ-u4** | 1502 | **16.1** | **64.0s** | **0.44** | **0.64** |

## TPOT 성장곡선 (graph, ms) — 컨텍스트별 순간 per-step

최적화 후. (KV read streaming unpack 덕에 S3·S4의 u2/u3 곡선이 후반 ctx에서 MXINT8 아래로 내려옴.)

| 경로 | ctx 801 | 1056 | 1824 | 2848 | 4680 |
|------|------|------|------|------|------|
| bf16 baseline | 26.95 | 28.34 | 32.47 | 38.08 | **47.94** |
| S1 W-only mxint8 | 26.60 | 27.97 | 32.12 | 37.69 | 47.53 |
| S1 W-only msaq-u4 | 19.04 | 20.40 | 24.56 | 30.03 | 39.73 |
| S2 W+A mxint8 | 22.34 | 23.75 | 27.89 | 33.43 | 43.23 |
| S2 W+A msaq-u2 | 23.13 | 24.54 | 28.69 | 34.20 | 44.06 |
| S2 W+A msaq-u4 | 19.18 | 20.58 | 24.76 | 30.28 | 40.06 |
| S3 KV-only mxint8 | 22.92 | 23.33 | 24.46 | 26.02 | 28.78 |
| S3 KV-only msaq-u2 | 22.83 | 23.34 | 24.05 | 25.20 | **27.36** |
| S3 KV-only msaq-u4 | 22.43 | 22.66 | 23.20 | 23.96 | 25.33 |
| S4 W-only+KV mxint8 | 22.57 | 22.98 | 24.12 | 25.69 | 28.52 |
| S4 W-only+KV msaq-u2 | 20.21 | 20.69 | 21.41 | 22.55 | 24.68 |
| S4 W-only+KV msaq-u3 | 19.79 | 20.28 | 20.99 | 22.12 | 24.27 |
| **S4 W-only+KV msaq-u4** | **14.70** | 14.90 | 15.42 | 16.20 | **17.55** |

## 해석 — 적용 대상별 기여가 분리됨

1. **weight 양자화는 baseline을 낮추고, KV 양자화는 성장곡선을 평탄화한다.** 둘은 직교:
   - **bf16**: ctx 길어지며 26.9→47.9ms (KV read 폭증).
   - **S3 KV-only(msaq)**: 22.4→25.3ms **평탄** — KV만 줄여도 성장 제거(긴 컨텍스트 decode의 진짜 병목).
   - **S1 W-only(msaq)**: 19.0→39.8ms — baseline은 낮지만 KV가 bf16이라 **여전히 성장**.
   - **S4 둘 다(msaq)**: 14.7→17.6ms — **가장 낮고 가장 평탄**. 두 이득이 곱해짐.

2. **최고 = S4 W-only+KV MSAQ-u4: bf16의 0.44×, MXINT8의 0.64×** (145.5s→**64.1s**).
   weight·KV 양자화가 **compound**(TPOT 37.4→16.1ms). end-to-end 핵심 결과.

3. **MXINT8 W-only는 end-to-end 이득이 0 (S1 mxint8 = 1.00×bf16).** MXINT8 GEMV는 cuBLAS bf16
   GEMV와 거의 동속(커널벤치 47 vs 46µs)이라 decode를 못 줄인다. 반면 **MSAQ W-only는 0.80** —
   wide-load u4 GEMV(커널 0.66×cuBLAS)가 cuBLAS를 이겨서 실제로 baseline을 내린다. → **W-only
   scope에서 MSAQ가 MXINT8보다 명백히 가치 있다(u4 0.80).**

4. **KV-only(S3)만으로도 0.64~0.69** — weight를 bf16로 둬도 KV 양자화가 긴 컨텍스트 decode를 크게
   가속(TTFT는 bf16 그대로 279ms 유지 → prefill 손해 없이 decode 이득만). MSAQ KV가 MXINT8보다
   약간 우위(0.92, packed KV가 더 적은 read).

5. **MSAQ vs MXINT8 (시나리오별):** S1 W-only **0.80**(완승) · S2 W+A 0.91 · S3 KV 0.92 ·
   S4 W+KV **0.64**(두 scope의 이득이 곱해져 격차 최대). 커널 단위 결과(W-only GEMV·KV dequant에서
   MSAQ 우위)가 end-to-end로 그대로 누적.

6. **TTFT 트레이드오프:** weight를 양자화하는 S1·S2·S4는 prefill GEMM이 cuBLAS 대신 커스텀
   IMMA/타일이라 TTFT 1.2~1.6s(bf16 0.28s 대비 ~5×). **KV-only(S3)는 bf16 weight라 TTFT 손해 0.**
   decode 3880이 total을 지배해 weight-quant도 total에선 이득이지만, **짧은 생성**이면 KV-only가
   TTFT까지 안전한 선택.

7. **u-스윕: u4가 전 시나리오에서 압도적, u2/u3는 W-only에서만 MXINT8을 이긴다.** u4는 upper 코드가
   nibble-정렬이라 wide int4 load + 값싼 bfe로 언팩이 가볍지만, u2/u3는 streaming bit-buffer
   언팩이 무거워 더 적은 bit를 읽는 대역폭 이득을 까먹는다(커널 단위 결과 그대로):
   - **S1 W-only**: u2/u3도 MXINT8을 이김(/mxint8 0.93/0.92) — MXINT8 GEMV가 cuBLAS와 동속이라
     u2/u3의 약한 이득으로도 역전. u4는 0.80.
   - **S2 W+A·S3 KV-only**: u2/u3가 **MXINT8보다 느림**(/mxint8 1.05~1.07) — 언팩 비용이 IMMA/KV-read
     이득을 초과. **u4만** crossover(0.91, 0.92).
   - **S4**: u2/u3는 ~동률(0.96~0.98), u4만 0.64.
   → **실전 권장 = u4.** u2/u3은 정확도가 더 필요할 때의 옵션이되, end-to-end 속도는 u4가 정답.

## 커널 최적화 라운드 (Phase 31) — KV write / KV read / W+A GEMV

마지막에 짠 세 커널(KV write·KV append·W+A GEMV)은 "matched + mantissa-sharing 정확" 수준까지만
와 있었음. mantissa-sharing 쪽 최적화를 적용(숨길 수 있는 실행시간은 숨기고, BW/occupancy를 높이는
방향). 결과(괄호=최적화 전 `/mxint8`):

| 커널 | lever | 결과 |
|------|-------|------|
| **KV write** | nb를 grid.z로 올려 occupancy 확보(GQA H=8에서 32→224 블록). **generic이라 MXINT8에도 동일 적용**(matched). | L=800에서 u4 **1.09→0.88** |
| **KV read** | u2/u3에 **streaming bit-buffer unpack** 이식(general-straddle→롤링버퍼). MXINT8은 int8 직읽이라 무변경 → mantissa-sharing-only. **KV-only decode의 진짜 병목**(write/append이 아님). | Lk=4680 u2 **1.29→0.79**, u3 1.27→0.77 |
| **W+A GEMV** | 활성화 qx를 블록당 shared에 1회 stage(컬럼마다 재로드 제거). MSAQ만 적용. | u2 **1.17→1.11**, u3 1.15→1.07, u4 0.83 |
| **KV append** | 미적용(아래 "이길 수 없는 이유"). | ~1.04 |

**end-to-end 효과 (u2/u3가 지던 곳이 역전):**
- **S3 KV-only u2/u3: 1.07 → 0.97** (이제 win). KV read streaming이 직접 효과.
- **S4 W-only+KV u2/u3: 0.98/0.96 → 0.88/0.87**. KV read win + W-only GEMV win 누적.
- **S2 W+A u2/u3: 1.06/1.05 → 1.02/1.01** (개선되나 ~tie, 미세하게 짐).
- u4는 전부 그대로 win, S1은 decode가 W-only GEMV(이미 최적)라 무변.

### MXINT8보다 빨라지지 못한 케이스 — 이유

1. **W+A GEMV u2/u3 (1.01~1.11), KV read u2/u3 @ short ctx (~1.18): extraction-bound (근본).**
   u2/u3은 byte를 0.69~0.78×만 줄이는데(u4는 0.56×), sub-byte를 꺼내는 streaming-unpack 추출
   비용이 그 절약을 까먹는다. 게다가 W+A의 MXINT8 baseline은 이미 int-dot로 BW 효율적(~400GB/s).
   → **upper 코드가 nibble(4bit) 정렬되는 u4만** wide int4 load + 값싼 bfe로 추출이 가벼워 이긴다.
   u2/u3(비-nibble)은 이 아키텍처에서 구조적으로 MXINT8 직읽기를 못 넘는다.
2. **KV append: MSAQ가 항상 MXINT8보다 일을 더 한다(decompose **+ bit-pack** vs decompose+store).**
   게다가 단일 토큰이라 H·nb=32 스레드 1-블록 → launch/tiny-work 지배. 더 적은 일로 줄일 수가
   없다(bit-pack은 포맷 그 자체). 단 CUDA graph 하에선 launch가 숨겨져 GPU 시간이 미미 → end-to-end
   영향 거의 0.
3. **방법론 노트:** qx-staging은 MSAQ(unpack-stall로 latency-bound)에는 도움되지만 MXINT8(이미 BW
   효율)에는 **해가 됨**(39.6→54µs). generic lever라도 한쪽에만 맞으면 양쪽 강제는 불공정 → 각 커널을
   **각자의 최적 구성(best-vs-best)**으로 비교. occupancy 같이 양쪽에 똑같이 이로운 lever만 mirror.

**요약:** 세 커널 최적화 후 **u2/u3가 S1·S3·S4에서 MXINT8을 이기게 됨**(전엔 S3/S4 일부에서 짐).
유일하게 못 이기는 곳은 **W+A GEMV u2/u3(~tie)**와 **KV append**(둘 다 위의 근본 이유)이며, u4는
처음부터 전 구간 win. → **권장 u4에서 4개 시나리오 전부 MXINT8보다 빠름이 증명됨**(S1 0.80·S2 0.90·
S3 0.92·S4 0.64).

## 이전(Python-loop) 대비
이전 coupled-run의 W-only+KV(=S4) MSAQ-u4는 0.60×bf16이었으나, **CUDA graph로 dispatch
오버헤드를 제거하니 0.44×**로 더 벌어짐 — Python 루프의 per-op dispatch가 TPOT를 부풀려 비율을
1.0쪽으로 희석했음을 확인. graph 측정이 커널의 실제 이득을 드러낸다.

## 주의
타이밍 하니스: 가중치 랜덤·레이어 재사용(타이밍은 값 무관), glue·lm_head는 bf16 공통. graph는 스텝당
back-to-back 커널 시간(dispatch-free)을 재고, total은 trajectory 적분.
