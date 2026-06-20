# Weight matmul scope 결과 총정리 (W-only / W+A · GEMM / GEMV)

가중치 양자화 4개 커널(prefill GEMM 2종 + decode GEMV 2종)에서 **MSAQ가 MXINT8을 공정·정확하게
이기는가**의 기록. 결론부터: **memory-bound인 decode GEMV와 INT8-IMMA로 compute를 싸게 만든
W+A GEMM에서 u4가 명확히 win**(KV-read와 대조 — 거긴 tie). KV-read 시도는 [kv_read_attempts.md],
7개 커널 전체는 [kernel_ver2.md], 단계별 로그는 [change.md].

---

## 0. 문제 구조

가중치 matmul `Y = X · W^T` (reduction 축 = K, 입력 채널). 가중치는 column-major packed plane.
**reduction이 한 출력 컬럼의 연속 packed 바이트를 따라가므로(=element-내부)** thread-per-column wide
load로 **full-sector coalesced** → sub-byte도 깨끗(KV-read P·V의 키-가로 half-sector 문제 없음).

승부를 가르는 건 **연산이 memory-bound냐 compute-bound냐**:

- **GEMV(decode, M=1):** 가중치를 재사용 없이 1회 read → **순수 memory-bound** → 0.58× 바이트가 곧장
  시간으로 → **MSAQ가 가장 크게 win**.
- **GEMM(prefill, M큼):** 가중치를 M행이 재사용 → AI 높아 **compute-bound** → 대역폭 절약이 가려지고
  unpack만 critical path에 더해짐 → bf16 누적이면 본전치기, **INT8 IMMA로 compute를 싸게 만들어야 win**.

MSAQ u4 = MXINT8의 **0.58× 바이트**, 원소당 ~4.7bit.

---

## 1. 4개 커널 — 설계·lever·결과

### 1.1 W-only GEMV (decode Linear, batch=1) — **memory-bound, 최대 win**
- 가중치를 **column-major packed**로 깔고 스레드가 자기 출력 컬럼의 packed 바이트를 wide-load해 언팩·누적.
- lever: u4 **wide int4 load + nibble `bfe`**(straddle 없음); u2/u3 **streaming bit-buffer unpack**(추출
  병목이라 여기선 이득); OUT 작을 때 **split-K**(occupancy).
- **결과(4096²):** u2 0.92 · u3 0.92 · **u4 0.63** (MSAQ/MXINT8). u4는 **cuBLAS bf16마저 0.59×로 이김**.

### 1.2 W+A GEMV (decode Linear, 활성화도 양자화) — memory-bound
- W-only GEMV 골격 + **활성화 MSAQ-s pre-pass(M=1)** + 블록 내적을 정수 dot으로, 두 스케일 한 번에 fold.
- weight read가 지배 → W-only와 동일 crossover 프로파일.
- **결과(4096²):** u2 1.17 · u3 1.14 · **u4 0.82**. u4만 win(wide load+bfe), u2/u3은 streaming unpack 비용이 int-dot 위에 남아 not-win.

### 1.3 W-only GEMM (prefill, M=512) — **compute-bound, 본전치기**
- 가중치를 packed로 저장, 타일 프롤로그에서 **1회 언팩**해 공유타일에 올리고 M행이 재사용.
- lever: 공유메모리 타일링(언팩 1회/타일), u4 **divide→shift**, **double-buffered WMMA 파이프라인**
  (다음 타일 streaming-unpack을 현재 타일 MMA와 overlap, opt-in `MS_TILE_CFG=11`).
- **결과(M=512, 4096²):** u2 1.09 · u3 1.10 · **u4 0.95**. compute-bound라 본전치기 근처; u4만 언팩이 싸 살짝 이김.

### 1.4 W+A GEMM (prefill, INT8 IMMA) — **compute, crossover**
- 가중치 packed + **활성화 MSAQ-s runtime decompose**(별도 pre-pass) → **INT8 IMMA**로 곱.
- lever: **2-stage 분리**(활성화 양자화를 memory-bound pre-pass로 빼면 IMMA 메인루프는 weight 언팩만 →
  W-only로 환원), **double-buffered 파이프라인 IMMA**(weight 언팩이 MMA 그림자에), 블록 스케일 epilogue fold.
- **결과(M=512):** **u2 0.90 · u3 0.92 · u4 0.79** — IMMA로 compute가 싸진 만큼 weight 대역폭 절약이 드러나 **전 u crossover**.

---

## 2. 성공한 lever 표 (왜 이기는가)

| lever | 적용 커널 | 효과 | 원리 |
|-------|-----------|------|------|
| column-major wide load (thread-per-col) | GEMV | full-sector coalesced read | reduction이 element-내부라 연속 바이트 → 0.58× 바이트 그대로 |
| u4 nibble `bfe` (divide→shift) | 전부 | 언팩 거의 공짜 | upper가 정확히 nibble, straddle 없음 |
| streaming bit-buffer unpack (u2/u3) | GEMV | 추출 병목 완화 | 롤링 64-bit 버퍼(코드당 shift+mask 1회) |
| split-K (occupancy) | GEMV | SM 채움 | OUT 작아 base block 부족 → K-reduction 분할 |
| shared-mem 타일링(언팩 1회/타일) | GEMM | 언팩을 FMA에 묻음 | M행이 타일 재사용(compute-bound) |
| 2-stage(활성화 pre-pass + IMMA) | W+A GEMM | W-only로 환원 | 활성화 양자화 O(MK)≪O(MK·OUT)를 분리 |
| double-buffered 파이프라인(WMMA/IMMA) | GEMM | 언팩이 텐서코어 그림자에 | 다음 타일 unpack ∥ 현재 타일 MMA |

> 주의: streaming unpack은 **추출-병목 GEMV엔 이득, FMA-병목 GEMM엔 손해** — 같은 기술이 연산마다 부호가 다름.

---

## 3. 근본 원인 (KV-read와 대조)

가중치 matmul이 이기는 두 조건이 충족된다:
1. **reduction 축이 element-내부** → column-major wide-load가 full-sector(staging 불필요) → **sub-byte도
   0.58× 바이트를 그대로 실현**. (KV-read P·V는 키-가로 reduction이라 sub-byte가 half-sector → staging
   필요 → 그 staging이 BW를 throttle해서 tie.)
2. **memory-bound(GEMV) 또는 IMMA로 compute를 싸게(W+A GEMM)** → 0.58× 바이트가 시간으로 환원.
   (W-only GEMM은 compute-bound라 본전치기; W+A는 IMMA로 compute를 4× 싸게 만들어 crossover.)

즉 **GEMV가 KV-read P·V와 갈리는 지점 = "staging 없이 wide-load→직접 누적이 되느냐"**. GEMV는 되고
(win), P·V는 sub-byte half-sector라 안 됨(tie). [kv_read_attempts.md] §3 참조.

---

## 4. 최종 결론

- **W-only GEMV u4 0.63(vs cuBLAS 0.59)** — MSAQ의 대표 win, decode TPOT를 직접 줄임. memory-bound +
  element-내부 reduction의 정석.
- **W+A GEMM u4 0.79** — INT8 IMMA + 2-stage로 compute-bound GEMM에서도 crossover.
- **W-only GEMM u4 0.95 / W+A GEMV u4 0.82** — u4만 미세/명확 win, u2/u3은 추출 비용으로 not-win.
- **end-to-end(Llama-3.1-8B):** S1 W-only **0.79**, S4 W+KV **0.44× bf16**(KV-read가 tie여도 weight GEMV
  win이 지배). [kernel_ver2.md] §4.

## 관련 산출물
- 커널: `csrc/w_gemv.cu`(W-only/W+A GEMV), `csrc/wa_gemm.cu`(W-only/W+A GEMM, WMMA/IMMA), `csrc/mxint8.cu`(짝).
- 벤치/테스트: `tests/benchmark.py`, `tests/test_w.py`, `tests/test_wa.py`.
- 설계 규칙·1차 수치: [kernel_ver1.md]; 공정성 감사: [for_fair_comparison.md].
