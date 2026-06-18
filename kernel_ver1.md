# Kernel ver.1 — 7종 커널 설계·최적화·성능 정리

MSAQ-signed(mantissa-sharing) 양자화를 LLM 추론의 7개 핵심 연산에 어떻게 녹였는지,
각 연산의 특성과 GPU(RTX 3090) 특성에 맞춰 어떤 설계 규칙·최적화를 적용했는지,
그리고 BF16 / MXINT8 / MSAQ 세 경로의 실측 비교를 한 곳에 모았다. 구현 변수·함수
세부가 아니라 "왜 이렇게 설계했는가"를 high-level로 기술한다.

---

## 0. 공통 배경

### 0.1 Mantissa-sharing 포맷 (MSAQ-s)
32개 원소를 한 블록으로 묶고:
- 블록당 **E8M0 base scale** 1개 (지수 8-bit),
- 원소당 **upper 코드** (8−u) bit — mantissa의 상위 비트,
- gs개 원소 그룹당 **shared 코드** u bit — 그룹이 **공유**하는 하위 mantissa.

복원값은 `(upper·2^u + shared) · base_scale`. 핵심은 **하위 u비트를 그룹이 공유**하여
원소당 저장 비트를 줄이는 것이다. 예: u4·gs8이면 32원소가 upper 16B + shared 2B +
scale 1B ≈ **19B**(원소당 ~4.7bit)로, MXINT8의 32B(원소당 8bit) 대비 **~40% 적다**.

### 0.2 모든 커널을 관통하는 설계 규칙 (성공한 lever만)
이 프로젝트에서 반복적으로 확인된 규칙들:

1. **이득의 본질은 "대역폭"이지 "연산량"이 아니다.** mantissa-sharing은 저장 비트를
   줄여 HBM 트래픽을 줄인다. 따라서 **memory-bound 연산(GEMV·KV read·KV write)**에서는
   바로 이기고, **compute-bound 연산(GEMM)**에서는 절약된 대역폭이 가려지고 unpack
   연산만 critical path에 더해져 본전치기에 가까워진다. → **연산이 memory-bound인지
   compute-bound인지가 sharing이 이기는지를 결정한다.**

2. **u4는 특별하다.** upper 코드가 정확히 nibble(4-bit)이라 한 컬럼의 바이트들이
   깔끔한 int4 벡터를 이룬다 → **한 번의 wide aligned load + 값싼 nibble 비트추출**,
   straddle(바이트 경계 걸침) 없음. u2/u3은 비트가 바이트 경계를 넘나들어 **streaming
   bit-buffer unpack**(롤링 64-bit 버퍼)이 필요하고 더 무겁다. 그래서 **u4가 항상 가장
   크게 이긴다.**

3. **coalescing 규칙: 연속 스레드가 연속 바이트를 만지게 배치한다.** KV write는
   thread-per-token, GEMV·KV read는 thread-per-column/key, 식으로 매핑하면 로드가
   coalesce되고 broadcast를 피한다. 레이아웃(plane을 어느 축으로 깔지)이 곧 성능이다.

4. **양자화를 matmul에서 분리한다(2-stage).** runtime에 활성화를 양자화해야 하는
   W+A에서는, 활성화 양자화를 **별도의 memory-bound pre-pass**로 빼면(O(MK), 전체의
   ~1%) matmul 프롤로그가 "weight unpack만" 하면 되는 형태가 되어 **이미 푼 W-only
   문제로 환원**된다.

5. **bottleneck에 맞춰 unpack 기법을 고른다.** streaming bit-buffer unpack은 추출이
   병목인 GEMV에선 이득이지만, FMA가 병목인 GEMM에선 오히려 손해다. 같은 기술이
   연산마다 부호가 다르다.

6. **launch-bound 연산은 최적화하지 말고 fuse한다.** 단일 토큰 KV append처럼 일거리가
   사소한 커널은 시간의 대부분이 커널 런치다 → 단독 측정/튜닝이 무의미, 인접 epilogue에
   합치는 게 정답.

### 0.3 왜 MXINT8 비교가 "동등"한가
모든 MSAQ 커널의 MXINT8 짝은 **같은 커널 골격**으로 만들었다 — 같은 레이아웃, 같은
split-K, 같은 타일링, 같은 IMMA 파이프라인, 같은 coalescing. **딱 하나, 원소별 언팩만**
다르다(MSAQ는 비트 언팩 + mantissa 재조합, MXINT8은 int8 직접 읽기). 그래서 측정 차이가
"mantissa-sharing 자체의 비용/이득"만 격리하고, 커널 구현 품질 차이로 오염되지 않는다.

> **단 하나의 의도적 예외:** runtime 활성화 양자화 **포맷**(MSAQ는 MSAQ-s decompose,
> MXINT8은 plain MXINT8)은 미러하지 않는다. 이건 최적화가 아니라 포맷 정의 차이이며,
> "MSAQ 경로는 MSAQ로, 비교 대상은 MXINT8로 runtime 양자화한다"는 규약을 따른 것이다.

---

## 1. Prefill 단계 커널

### 1.1 W-only GEMM  (가중치만 양자화, 활성화는 bf16)
- **mantissa-sharing 적용:** 가중치 plane을 packed로 저장하고, 타일 프롤로그에서 한 번
  언팩해 공유 메모리 타일에 올린 뒤 그 타일을 M개 행이 재사용.
- **연산·하드웨어 특성 → 설계 규칙:** GEMM은 가중치를 M행에 걸쳐 재사용하므로
  **arithmetic intensity가 높아 compute-bound**다. → sharing의 대역폭 절약이 가려진다.
  타일당 가중치를 **딱 한 번만** 언팩해 FMA 비용에 묻는 것이 최선.
- **성공한 lever:** 공유메모리 타일링(언팩 1회/타일 재사용), u4의 divide→shift(나눗셈을
  시프트로). *streaming unpack은 여기선 손해라 적용 안 함*(규칙 5).
- **결과(M=512, 4096×4096):** u2 1.09 · u3 1.10 · **u4 0.95** (MSAQ/MXINT8).
  compute-bound라 본전치기 근처; u4만 언팩이 싸서 살짝 이김. cuBLAS bf16 대비 ~8–10×
  (저비트 GEMM의 구조적 한계, INT8 IMMA를 못 쓰는 W-only의 본질적 비용).

### 1.2 W+A GEMM  (가중치·활성화 모두 양자화)
- **mantissa-sharing 적용:** 가중치는 W-only처럼 packed. 활성화는 **MSAQ-s로 runtime
  decompose**(별도 pre-pass)하여 int8 word로 만든 뒤 INT8 IMMA로 곱한다.
- **연산·하드웨어 특성 → 설계 규칙:** 둘 다 int8이면 텐서코어 **INT8 IMMA**를 쓸 수
  있어 compute가 4× 빨라진다. 단 활성화를 runtime 양자화해야 한다. → **규칙 4(2-stage)**:
  활성화 양자화를 memory-bound pre-pass로 분리(전체의 ~1%, 각 원소가 OUT번 재사용되어
  O(MK)≪O(MK·OUT))하면, IMMA 메인루프 프롤로그는 weight 언팩만 남아 W-only로 환원.
- **성공한 lever:** Stage 분리(pre-pass + IMMA), **double-buffered 파이프라인 IMMA**(다음
  블록 weight 언팩을 현재 MMA와 overlap), 블록별 스케일 epilogue fold.
- **결과(M=512):** **u2 0.90 · u3 0.92 · u4 0.79** (MSAQ/MXINT8) — **전 u에서 crossover.**
  IMMA로 compute가 싸진 만큼 weight 대역폭 절약이 드러난다. (cuBLAS bf16 대비 ~8×는
  여전하지만, 이 scope의 baseline은 동일 IMMA를 쓴 MXINT8이다.)

### 1.3 KV write  (prefill에서 K/V 캐시를 packed로 기록)
- **mantissa-sharing 적용:** bf16 K/V[H,L,D]를 블록 단위로 decompose하여 **token-major
  packed plane**으로 기록. 이후 decode read가 보는 포맷과 동일.
- **연산·하드웨어 특성 → 설계 규칙:** 이건 양자화+저장이라 **memory-bound**. → **규칙 3**:
  thread-per-token으로 매핑해 연속 토큰이 연속 바이트를 store → coalesced. 블록 수가
  H·⌈L/TPB⌉라 occupancy가 공짜(split 불필요).
- **성공한 lever:** token-major coalesced store, 블록별 1-스레드 decompose.
- **결과(H32 D128, L=2048):** MSAQ u4 193µs · MXINT8 226µs · BF16 43µs.
  **MSAQ/MXINT8 0.85**(packed plane이 ~40% 적은 바이트를 store → 대역폭으로 이김),
  MSAQ/BF16 ~4.5×(BF16은 순수 memcpy라 당연히 쌈; 이건 prefill 1회 비용이고 decode
  read에서 회수). L=1024~4096에서 0.78~0.91.

---

## 2. Decode 단계 커널

### 2.1 W-only GEMV  (decode의 Linear, batch=1)
- **mantissa-sharing 적용:** 가중치를 **column-major packed plane**으로 깔고, 스레드가
  자기 출력 컬럼의 packed 바이트를 wide-load해 언팩하며 누적.
- **연산·하드웨어 특성 → 설계 규칙:** GEMV는 가중치를 **재사용 없이 한 번씩** 읽으므로
  **순수 memory-bound** → sharing이 가장 크게 이기는 연산. → **규칙 2·3**: u4는 wide
  int4 load + nibble bfe, u2/u3은 streaming bit-buffer unpack(추출 병목이라 여기선 이득),
  그리고 OUT이 작아 SM이 빌 때를 대비한 **split-K(occupancy)**.
- **성공한 lever:** wide column-major load, split-K, u4 nibble bfe / u2u3 streaming unpack.
- **결과(4096×4096):** u2 0.92 · u3 0.92 · **u4 0.63** (MSAQ/MXINT8).
  u4는 cuBLAS bf16마저 이긴다(**MSAQ/cuBLAS 0.66**) — memory-bound 영역에서 더 적은
  바이트를 읽으니 당연.

### 2.2 W+A GEMV  (decode Linear, 활성화도 양자화)
- **mantissa-sharing 적용:** W-only GEMV 골격에 **활성화 MSAQ-s pre-pass(M=1)**를 더해
  활성화도 int8 word로 만들고, 블록 내적을 **정수 dot**으로 돌린 뒤 두 블록 스케일을
  한 번에 fold.
- **연산·하드웨어 특성 → 설계 규칙:** GEMV는 weight read가 지배하므로 활성화 pre-pass
  (M=1, 사소)는 비교에 중립. 내적을 int로 바꿔도 병목은 여전히 weight 언팩/로드. →
  **W-only와 동일한 crossover 프로파일**을 가진다.
- **성공한 lever:** W-only GEMV의 lever 전부 + 활성화 pre-pass 재사용 + per-block int-dot fold.
- **결과(4096×4096):** u2 1.17 · u3 1.14 · **u4 0.82** (MSAQ/MXINT8). u4만 win(wide
  load+bfe), u2/u3은 streaming unpack 비용이 int-dot 위에 남아 not-win.

### 2.3 KV quantize  (decode마다 새 토큰의 K/V를 캐시에 append)
- **mantissa-sharing 적용:** KV write의 **L=1·위치 지정 in-place** 특수화. 새 토큰을
  decompose하여 캐시의 해당 slot에 기록. write·read와 같은 포맷.
- **연산·하드웨어 특성 → 설계 규칙:** 일거리가 H·nb개(사소) → **launch-latency 지배**
  (~8–17µs ≈ 커널 런치). → **규칙 6**: 단독 최적화 무의미, 배포 시 projection/RoPE
  epilogue나 attention prologue에 **fuse**.
- **성공한 lever:** (알고리즘 lever 없음 — 의도적. write 포맷·primitive 재사용만.)
- **결과(단일 토큰, H32 D128):** MSAQ u4 8.7µs · MXINT8 8.7µs · BF16 copy 16µs.
  **MSAQ ≈ MXINT8**(런치 지배라 동률), BF16 `copy_`가 오히려 느린 것도 torch generic
  copy 런치 오버헤드 차이지 양자화가 공짜라서가 아니다.

### 2.4 KV cache dequantize  (decode attention에서 packed KV를 읽어 복원·attend)
- **mantissa-sharing 적용:** flash-decode attention이 매 key/value마다 packed plane을
  읽어 **on-the-fly 복원**(W-only와 동일한 언팩 경로 재사용)해 score·output을 누적.
- **연산·하드웨어 특성 → 설계 규칙:** decode attention은 **KV read 대역폭이 지배**
  (Q는 한 토큰, KV는 Lk×D 전체). → memory-bound → sharing이 크게 이긴다. thread-per-key로
  coalesce.
- **성공한 lever:** packed KV의 적은 read, fused online-softmax(복원값을 HBM에 다시 쓰지
  않음), W-only 언팩 재사용.
- **결과(H8 Lk4096 D128):** u2 1.04 · u3 1.02 · **u4 0.54** (MSAQ/MXINT8).
  SDPA bf16 대비 **MSAQ/SDPA 0.19**(u4) — 긴 컨텍스트일수록 KV read가 지배하고 packed
  KV가 훨씬 작아 압도적.

---

## 3. 종합 결과 (RTX 3090, warm; 표기는 시간[µs]과 MSAQ/MXINT8 비)

| 단계 | 커널 | 특성 | BF16 | MXINT8 | MSAQ u4 | MSAQ/MX (u2·u3·u4) |
|------|------|------|------|--------|---------|----------------------|
| Prefill | W-only GEMM | compute-bound | 279(cuBLAS) | 2514 | 2400 | 1.09 · 1.10 · **0.95** |
| Prefill | W+A GEMM | compute(IMMA) | 279(cuBLAS) | 2765 | 2182 | **0.90 · 0.92 · 0.79** |
| Prefill | KV write | memory-bound | 43(copy) | 226 | 193 | 0.89 · — · **0.85** |
| Decode | W-only GEMV | memory-bound | 45.6(cuBLAS) | 47.4 | 30.1 | 0.92 · 0.92 · **0.63** |
| Decode | W+A GEMV | memory-bound | 45.6(cuBLAS) | 40.5 | 33.2 | 1.17 · 1.14 · **0.82** |
| Decode | KV quantize(append) | launch-bound | 16(copy) | 8.7 | 8.7 | ~1.0 (런치 지배) |
| Decode | KV dequant(attention) | memory-bound | 190(SDPA) | 68 | 36.5 | 1.04 · 1.02 · **0.54** |

> 사이즈: GEMM/GEMV는 4096×4096(M=512 prefill / batch=1 decode), KV write H32·D128·L2048,
> KV append 단일 토큰 H32·D128, KV dequant H8·Lk4096·D128.

**한 줄 요약:** **memory-bound 연산(GEMV·KV read·KV write)에서 mantissa-sharing이 명확히
이기고(특히 u4), compute-bound GEMM은 INT8 IMMA로 compute를 싸게 만든 W+A에서만 crossover**한다.
launch-bound append는 fuse 대상.

---

## 4. End-to-end harness로의 융합

이 7개 커널은 하나의 추론 루프에 다음처럼 맞물린다. **KV 캐시 포맷(packed plane)이
write·append·dequant 세 커널에서 동일**하다는 점이 결합의 핵심 — prefill이 쓴 캐시를
decode가 그대로 append/read한다.

**Prefill (TTFT 측정):**
1. 입력 토큰 임베딩 → **W-only GEMM**(Q/K/V·MLP projection; 활성화 bf16) 또는
   **W+A GEMM**(활성화도 양자화하는 레이어)으로 선형층 계산.
2. 생성된 K/V를 **KV write**로 packed plane 캐시에 기록.
3. → 첫 토큰까지 시간 = **TTFT**.

**Decode (TPOT 측정, 토큰마다 반복):**
1. 직전 토큰 hidden → **W-only GEMV** / **W+A GEMV**로 Q/K/V·MLP projection.
2. 새 토큰의 K/V를 **KV quantize(append)**로 캐시 slot에 기록 — launch-bound라
   projection/RoPE epilogue에 **fuse**하는 것이 권장.
3. **KV cache dequantize**가 packed 캐시 전체를 읽어 flash-decode attention 수행.
4. → 토큰당 시간 = **TPOT**.

**총 추론시간 ≈ TTFT + (생성 토큰 수 − 1) × TPOT.**

설계 함의:
- decode 경로(GEMV·KV read)는 전부 memory-bound라 mantissa-sharing(u4)이 직접 TPOT를
  줄인다 — 본 프로젝트의 주 타겟.
- prefill 경로는 compute-bound라, W+A를 INT8 IMMA 2-stage로 돌려야 TTFT에서 이득이 난다.
- append는 단독으로 의미 있는 시간이 아니므로 fuse하여 TPOT 오버헤드를 0에 가깝게 만든다.
- 세 KV 커널이 단일 포맷을 공유하므로, prefill→decode 전환에서 캐시 재포맷 비용이 없다.
