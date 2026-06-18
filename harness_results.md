# End-to-End Harness 결과 (Llama-3.1-8B, RTX 3090)

`tests/harness.py` 실측. 32-layer full forward, **prefill=800 / decode=3880**, GQA 32Q:8KV,
vocab 128256. glue(RMSNorm·RoPE·SwiGLU·SDPA)는 bf16 공통. 시간 단위 ms(별도 표기 외).
설계·가정은 [harness_design.md](harness_design.md), 커널별 분석은 [kernel_ver1.md](kernel_ver1.md).

## 전체 표

| 경로 | TTFT | TPOT(mean) | total | /bf16 | /mxint8 |
|------|------|-----------|-------|-------|---------|
| bf16 | **272** | 41.4 | 161.1s | 1.00 | — |
| mxint8_wonly | 1615 | 31.6 | 124.2s | 0.77 | — |
| mxint8_wa | 1594 | 26.0 | 102.4s | 0.64 | — |
| msaq_wonly-u2 | 1726 | 28.4 | 112.1s | 0.70 | 0.90 |
| msaq_wonly-u3 | 1706 | 28.1 | 110.7s | 0.69 | 0.89 |
| **msaq_wonly-u4** | 1504 | **24.4** | **96.1s** | **0.60** | **0.77** |
| msaq_wa-u2 | 1485 | 29.9 | 117.3s | 0.73 | 1.15 |
| msaq_wa-u3 | 1454 | 28.6 | 112.6s | 0.70 | 1.10 |
| msaq_wa-u4 | 1215 | 26.1 | 102.7s | 0.64 | 1.00 |

> total = TTFT + Σ decode. decode 3880 ≫ prefill 800이라 **TPOT가 total을 지배**.
> `/mxint8`은 같은 변형(wonly↔wonly, wa↔wa)의 MXINT8 대비.

## TPOT 성장곡선 (스텝별 순간 TPOT, ms) — 핵심 그림

| 경로 | t=1 | t=256 | t=1024 | t=2048 | t=3880 |
|------|-----|-------|--------|--------|--------|
| bf16 | 34.8 | 34.9 | 35.1 | 41.0 | **53.2** |
| mxint8_wonly | 33.1 | 32.1 | 32.3 | 28.6 | 31.8 |
| mxint8_wa | 27.7 | 25.5 | 25.8 | 25.2 | 27.5 |
| **msaq_wonly-u4** | 25.0 | 24.3 | 24.1 | 24.2 | **25.0** |
| msaq_wa-u4 | 26.7 | 26.3 | 26.2 | 25.8 | 26.5 |

- **bf16는 컨텍스트가 길어질수록 TPOT가 급증**(34.8→53.2ms): KV 캐시가 커지며 매 스텝 KV read
  대역폭이 폭증. **msaq u4는 거의 평탄**(25.0→25.0): packed KV(~0.6B/elem)라 KV read가 길어져도
  싸게 유지. → **긴 컨텍스트일수록 MSAQ 이득이 벌어진다**(설계 가설 확인). u2/u3은 streaming
  unpack이 무거워 후반에 다소 상승(24→33), u4만 평탄.

## 해석

1. **최고 end-to-end: `msaq_wonly-u4` = bf16의 0.60×, MXINT8의 0.77×** (총 161→96s).
   decode가 memory-bound라 W-only GEMV(u4) + 작은 KV read가 총시간을 끌어내림.

2. **TTFT는 bf16 압승(0.27s vs 양자화 1.2~1.7s).** prefill은 compute-bound이고 bf16은 cuBLAS
   텐서코어를, 양자화는 커스텀 IMMA/타일 커널(~5–6× 느림)을 쓴다. 하지만 decode 3880스텝이
   total을 지배해 **양자화가 total에서 역전**.

3. **W-only vs W+A (방향이 갈림):**
   - **MSAQ:** wonly-u4(96s) > wa-u4(102.7s). decode GEMV가 지배하는데 W-only GEMV(커널 0.63)가
     W+A GEMV(0.82)보다 빠르고, prefill 비중은 작아 wonly가 total 우위.
   - **MXINT8:** wa(102.4s) > wonly(124.2s). MXINT8는 W+A에서 INT8 IMMA(prefill)+int-dot(decode)로
     양쪽 다 가속 → wa가 우위.

4. **MSAQ vs MXINT8 (동일 변형):**
   - **W-only: MSAQ 완승** (u4 0.77, u2/u3 0.89~0.90). 적은 KV·weight read가 그대로 이득.
   - **W+A: 박빙** (u4 1.00, u2/u3 1.10~1.15). W+A decode GEMV는 u4만 crossover하고 u2/u3은
     streaming unpack 비용이 남아 MXINT8과 비슷하거나 약간 뒤짐 — 커널 단위 결과와 일치.

5. **결론:** 디코드 지배 워크로드(긴 생성)에서 **mantissa-sharing(u4)이 end-to-end로 이긴다.**
   가장 강한 조합은 **W-only u4**(KV·weight read 절감이 평탄한 TPOT로 직결). 짧은 프롬프트·긴
   생성일수록(이 시나리오: 800/3880) 이득이 크고, 컨텍스트가 길수록 더 벌어진다.

## 주의(측정의 한계)

- autoregressive decode가 Python 루프 구동이라 절대 TPOT에 **per-op dispatch 오버헤드**가 포함된다
  (모든 경로 공통 → 비율은 유효하나 절대 비율은 커널 단위(0.54~0.63)보다 1.0쪽으로 희석).
  실제 fused 엔진(CUDA graph)이면 격차가 더 벌어질 것.
- 타이밍 하니스: 가중치 랜덤·레이어 재사용(타이밍은 값과 무관), glue·lm_head는 bf16 공통.
