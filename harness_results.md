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

| 시나리오 | 포맷 | TTFT | TPOT | total | /bf16 | /mxint8 |
|---------|------|------|------|-------|-------|---------|
| baseline | bf16 | 274 | 37.5 | 145.6s | 1.00 | — |
| **S1 W-only** | MXINT8 | 1596 | 37.1 | 145.6s | 1.00 | — |
| | MSAQ-u2 | 1714 | 34.6 | 136.0s | 0.93 | 0.93 |
| | MSAQ-u3 | 1695 | 34.2 | 134.5s | 0.92 | 0.92 |
| | **MSAQ-u4** | 1501 | 29.5 | 116.0s | **0.80** | **0.80** |
| **S2 W+A** | MXINT8 | 1586 | 32.8 | 129.0s | 0.89 | — |
| | MSAQ-u2 | 1483 | 34.9 | 136.8s | 0.94 | 1.06 |
| | MSAQ-u3 | 1447 | 34.4 | 135.0s | 0.93 | 1.05 |
| | **MSAQ-u4** | 1210 | 29.9 | 117.1s | **0.80** | 0.91 |
| **S3 KV-only** | MXINT8 | 278 | 25.9 | 100.7s | 0.69 | — |
| | MSAQ-u2 | 279 | 27.8 | 108.2s | 0.74 | 1.07 |
| | MSAQ-u3 | 278 | 27.7 | 107.9s | 0.74 | 1.07 |
| | **MSAQ-u4** | 278 | 23.9 | 93.0s | **0.64** | 0.92 |
| **S4 W-only+KV** | MXINT8 | 1603 | 25.6 | 100.7s | 0.69 | — |
| | MSAQ-u2 | 1724 | 25.0 | 98.8s | 0.68 | 0.98 |
| | MSAQ-u3 | 1701 | 24.6 | 97.0s | 0.67 | 0.96 |
| | **MSAQ-u4** | 1504 | **16.1** | **64.1s** | **0.44** | **0.64** |

## TPOT 성장곡선 (graph, ms) — 컨텍스트별 순간 per-step

| 경로 | ctx 801 | 1056 | 1824 | 2848 | 4680 |
|------|------|------|------|------|------|
| bf16 baseline | 26.96 | 28.34 | 32.48 | 38.08 | **47.95** |
| S1 W-only mxint8 | 26.61 | 27.98 | 32.15 | 37.72 | 47.58 |
| S1 W-only msaq-u2 | 24.16 | 25.55 | 29.69 | 35.15 | 45.09 |
| S1 W-only msaq-u3 | 23.78 | 25.17 | 29.33 | 34.83 | 44.62 |
| S1 W-only msaq-u4 | 19.06 | 20.47 | 24.61 | 30.12 | 39.81 |
| S2 W+A mxint8 | 22.34 | 23.73 | 27.89 | 33.48 | 43.27 |
| S2 W+A msaq-u2 | 24.37 | 25.77 | 29.93 | 35.45 | 45.35 |
| S2 W+A msaq-u3 | 23.94 | 25.33 | 29.46 | 35.00 | 44.85 |
| S2 W+A msaq-u4 | 19.40 | 20.80 | 24.96 | 30.48 | 40.23 |
| S3 KV-only mxint8 | 22.93 | 23.35 | 24.49 | 26.05 | 28.81 |
| S3 KV-only msaq-u2 | 23.73 | 24.12 | 25.86 | 28.02 | 32.02 |
| S3 KV-only msaq-u3 | 23.71 | 24.10 | 25.85 | 27.97 | 31.82 |
| S3 KV-only msaq-u4 | 22.44 | 22.69 | 23.19 | 23.97 | 25.34 |
| S4 W-only+KV mxint8 | 22.58 | 23.00 | 24.13 | 25.70 | 28.54 |
| S4 W-only+KV msaq-u2 | 21.00 | 21.41 | 23.14 | 25.20 | 29.20 |
| S4 W-only+KV msaq-u3 | 20.61 | 21.00 | 22.69 | 24.77 | 28.63 |
| **S4 W-only+KV msaq-u4** | **14.69** | 14.94 | 15.42 | 16.21 | **17.56** |

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

## 이전(Python-loop) 대비
이전 coupled-run의 W-only+KV(=S4) MSAQ-u4는 0.60×bf16이었으나, **CUDA graph로 dispatch
오버헤드를 제거하니 0.44×**로 더 벌어짐 — Python 루프의 per-op dispatch가 TPOT를 부풀려 비율을
1.0쪽으로 희석했음을 확인. graph 측정이 커널의 실제 이득을 드러낸다.

## 주의
타이밍 하니스: 가중치 랜덤·레이어 재사용(타이밍은 값 무관), glue·lm_head는 bf16 공통. graph는 스텝당
back-to-back 커널 시간(dispatch-free)을 재고, total은 trajectory 적분.
