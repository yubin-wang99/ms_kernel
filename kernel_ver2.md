# Kernel ver.2 — 7종 커널 설계·최적화·성능 정리 (공정성 수정 반영)

MSAQ-signed(mantissa-sharing) 양자화를 LLM 추론의 7개 핵심 연산에 어떻게 녹였는지,
각 연산의 특성과 GPU(RTX 3090) 특성에 맞춰 어떤 설계 규칙·최적화를 적용했는지,
그리고 BF16 / MXINT8 / MSAQ 세 경로의 실측 비교를 한 곳에 모았다.

> **ver.1 → ver.2 변경점(요약).** 골격·설계 규칙·GEMM/GEMV/KV-write 결과는 그대로다.
> 바뀐 것은 **KV cache dequant(§2.4) 하나**다. ver.1은 MXINT8 KV read가 구버전
> warp-per-key였던 비대칭 때문에 MSAQ가 **u4 0.54로 "압승"**한다고 적었으나, MXINT8을
> **동일 thread-per-key**로 올려 공정화하니 그 압승의 ~2×가 MXINT8 under-optimization
> 산물이었음이 드러났다 → **공정 결과는 u4 ~tie**. 이어 design B(warp-transpose)·점유율
> lever·sector 진단까지 시도했고 결론은 **현 스칼라 decode 구조에선 KV read tie가 공정
> 최선**이다(상세 §2.4, [for_fair_comparison.md]). end-to-end(§4)의 S3·S4도 이에 맞춰 정정.

---

## 0. 공통 배경

### 0.1 Mantissa-sharing 포맷 (MSAQ-s)
32개 원소를 한 블록으로 묶고:
- 블록당 **E8M0 base scale** 1개 (지수 8-bit),
- 원소당 **upper 코드** (8−u) bit — mantissa의 상위 비트,
- gs개 원소 그룹당 **shared 코드** u bit — 그룹이 **공유**하는 하위 mantissa.

복원값은 `(upper·2^u + shared) · base_scale`. 핵심은 **하위 u비트를 그룹이 공유**하여
원소당 저장 비트를 줄이는 것이다. 예: u4·gs8이면 32원소가 upper 16B + shared 2B +
scale 1B ≈ **19B**(원소당 ~4.7bit)로, MXINT8의 33B(원소당 8.25bit) 대비 **~0.58×**.

### 0.2 모든 커널을 관통하는 설계 규칙 (성공한 lever만)
1. **이득의 본질은 "대역폭"이지 "연산량"이 아니다.** mantissa-sharing은 저장 비트를
   줄여 HBM 트래픽을 줄인다. → **memory-bound 연산(GEMV·KV read·KV write)**에서 이기고,
   **compute-bound(GEMM)**에서는 절약된 대역폭이 가려지고 unpack만 critical path에 더해진다.
2. **u4는 특별하다.** upper 코드가 정확히 nibble(4-bit)이라 **wide aligned load + 값싼
   nibble bfe**, straddle 없음. u2/u3은 비트가 바이트 경계를 넘나들어 **streaming
   bit-buffer unpack**이 필요해 더 무겁다 → **u4가 항상 가장 크게 이긴다.**
3. **coalescing 규칙: 연속 스레드가 연속 바이트를 만지게 배치한다.** 레이아웃이 곧 성능.
4. **양자화를 matmul에서 분리한다(2-stage).** W+A는 활성화 양자화를 memory-bound
   pre-pass(전체의 ~1%)로 빼면 IMMA 메인루프는 weight 언팩만 남아 W-only로 환원된다.
5. **bottleneck에 맞춰 unpack 기법을 고른다.** streaming unpack은 추출 병목 GEMV엔 이득,
   FMA 병목 GEMM엔 손해.
6. **launch-bound 연산은 최적화하지 말고 fuse한다.** (단일 토큰 KV append.)
7. **(ver.2 신규) sub-byte는 "reduction 축이 무엇이냐"에 따라 sector 효율이 갈린다.**
   P·V처럼 **키에 대한 reduction**이라 출력이 thread-per-d로 잡히는 연산에서는 **int8 V가
   자연히 full-sector**(워프 32 d-lane = 32 연속 바이트 = 정확히 1 섹터)인 반면, **sub-byte
   V는 half-sector**(32 nibble=16B)다. 이를 피하려면 staging(→L1TEX) 또는 warp-transpose
   (→shfl) 오버헤드를 **반드시** 지불해야 하고, 그 비용이 0.58× 바이트 절감을 거의 상쇄한다.
   → **sub-byte의 win은 reduction 축이 element-내부일 때(=GEMV/Q·K dot: thread-per-key로
   연속 적재) 깨끗하고, reduction이 키들을 가로지를 때(=P·V)엔 구조적으로 막힌다.**

### 0.3 왜 MXINT8 비교가 "동등"한가
모든 MSAQ 커널의 MXINT8 짝은 **같은 커널 골격**(같은 레이아웃·split-K·타일링·IMMA·
coalescing·**thread mapping**)이고 **딱 하나, 원소별 언팩만** 다르다. ver.2에서 KV read의
MXINT8을 thread-per-key로 올려 이 원칙을 KV read까지 관철했다(ver.1의 마지막 비대칭 제거).

> **단 하나의 의도적 예외:** runtime 활성화 양자화 **포맷**(MSAQ-s vs plain MXINT8)은
> 미러하지 않는다 — 최적화가 아니라 포맷 정의 차이.

---

## 1. Prefill 단계 커널 (ver.1과 동일)

### 1.1 W-only GEMM  (compute-bound)
- 가중치 plane을 packed로 저장, 타일 프롤로그에서 1회 언팩해 공유타일에 올리고 M행이 재사용.
- GEMM은 AI가 높아 **compute-bound** → sharing의 대역폭 절약이 가려진다. 타일당 1회 언팩이 최선.
- **결과(M=512, 4096²):** u2 1.09 · u3 1.10 · **u4 0.95** (MSAQ/MXINT8). compute-bound라 본전치기.

### 1.2 W+A GEMM  (compute, INT8 IMMA)
- 가중치 packed + 활성화 **MSAQ-s runtime decompose**(별도 pre-pass) → INT8 IMMA.
- 규칙 4(2-stage) + double-buffered 파이프라인 IMMA + 블록 스케일 epilogue fold.
- **결과(M=512):** **u2 0.90 · u3 0.92 · u4 0.79** — IMMA로 compute가 싸져 weight 대역폭 절약이 드러남.

### 1.3 KV write  (memory-bound)
- bf16 K/V[H,L,D]를 블록 decompose하여 **token-major packed plane**으로 기록(decode read와 동일 포맷).
- 규칙 3: thread-per-token coalesced store, grid=H·⌈L/TPB⌉·nb로 occupancy 공짜.
- **결과(H32 D128 L2048):** MSAQ u4 193µs · MXINT8 226µs · BF16 43µs → **u4 0.85**(L1024~4096서 0.78~0.91).

---

## 2. Decode 단계 커널

### 2.1 W-only GEMV  (memory-bound — sharing이 가장 크게 이김)
- 가중치 **column-major packed**, 스레드가 자기 출력 컬럼 바이트를 wide-load해 언팩·누적.
- GEMV는 가중치를 **재사용 없이 한 번** 읽음 → 순수 memory-bound. u4 wide int4+bfe,
  u2/u3 streaming unpack, OUT 작을 때 **split-K(occupancy)**.
- **결과(4096²):** u2 0.92 · u3 0.92 · **u4 0.63**. u4는 cuBLAS bf16마저 이김(0.66).

### 2.2 W+A GEMV  (memory-bound)
- W-only GEMV 골격 + 활성화 MSAQ-s pre-pass(M=1) + per-block int-dot fold.
- weight read가 지배 → W-only와 동일 crossover 프로파일.
- **결과(4096²):** u2 1.17 · u3 1.14 · **u4 0.82**. u4만 win.

### 2.3 KV quantize / append  (launch-bound)
- KV write의 L=1·in-place 특수화. 일거리 H·nb개 → **launch-latency 지배**(~8–17µs).
- 규칙 6: 단독 최적화 무의미, projection/RoPE epilogue에 **fuse**.
- **결과(단일 토큰 H32 D128):** MSAQ u4 8.7µs ≈ MXINT8 8.7µs(런치 지배 동률), BF16 copy 16µs.

### 2.4 KV cache dequantize / attention  (memory-bound이론 / **실측 latency-bound → 공정 tie**)
> **KV-read win 시도 총정리는 [kv_read_attempts.md] 참조** — 이 절의 tie 이후 8개 lever(split-K/
> warp-transpose/batch/channel-major/텐서코어-P·V/2-pass/scalar-Q·K)를 끝까지 시도한 기록과
> 근본 원인(dequant throughput ~0.5× MXINT8 BW), 최종 결론(공정·정확 win 불가)을 담았다.

**설계.** flash-decode attention이 매 key/value마다 packed plane을 읽어 on-the-fly 복원해
score·output을 누적. split-K(키축)+online-softmax+combine. K(Q·K dot)는 thread-per-key로
연속 적재(규칙 7의 "이기는 쪽"). V(P·V)는 키에 대한 reduction이라 thread-per-d가 자연스러운데
sub-byte V가 half-sector라(규칙 7) **chunk V를 shared에 coalesced staging** 후 thread-per-d로
누적한다.

**공정성 수정(ver.2 핵심).** MXINT8 KV read를 MSAQ와 동일 **thread-per-key**(in-thread dot,
워프 reduction 제거)로 올렸다. 그러자 MXINT8 KV read가 **~2× 빨라짐**(Lk4680 254→120µs) —
ver.1의 MSAQ "압승"(u4 0.54)은 거의 전부 이 비대칭의 산물이었다.

**공정 실측(둘 다 thread-per-key, best-vs-best):**

| Lk | u4 /MX | u3 /MX | u2 /MX |
|----|--------|--------|--------|
| 1056 | 1.04 | ~1.5 | 1.83 |
| 2848 | ~1.0 | ~1.5 | 1.46 |
| 4680 | 1.0–1.03 | ~1.5 | 1.64 |

→ **u4는 tie(best-vs-best에서 MX로 미세하게 기움), u2/u3은 명확히 손해.**

**왜 못 이기나(진단 확정).** MXINT8는 이미 **memory-bound로 ~300–480 GB/s**까지 스케일,
MSAQ wide는 **~140–220 GB/s**에서 saturate(둘 다 점유율 ~23%). ncu sector 진단: **DRAM
read sector 비 = 0.59 ≈ 바이트비 0.58**(DRAM 레벨엔 sector inflation **없음** — 바이트 이점은
실재). 그런데 커널이 DRAM 14%로 **latency-bound**(BW 미포화)라 그 이점이 시간으로 환원 안 됨.
지배 stall = memory latency(long_scoreboard)+staging barrier. 게다가 적게 읽는 MSAQ는
**in-flight 메모리 병렬도(MLP)가 작아** 오히려 latency를 덜 숨긴다.

**시도하고 버린 lever(documented negatives):**
- **design B (warp-transpose P·V, staging 제거; opt-in `MS_KV_WARPT`, bit-exact).** shared
  11→2 KB로 줄었으나 점유율 불변(grid가 0.5–0.76 wave로 머신을 한 wave도 못 채움 → per-SM
  block 캡이 한계가 아님), shfl issue가 L1TEX 57%로 상승 → **wide보다 ~5–10% 느림**.
- **점유율 lever.** split mult↑는 **MXINT8만** 이득(per-block 일이 적어) → 격차 확대.
  `__launch_bounds__`(regs 121→80 강제)는 spill로 더 느림. cp.async 커널은 2× 느림.
- **GQA-batched 기존 커널**(KV 1회 read·G query 재사용)은 점유율 붕괴(40 GB/s)로 미달.

**남은 정공법.** KV read 공정 win은 **per-output 오버헤드를 G개 query에 분산하는 제대로 된
GQA-amortized 재작성**(memory-bound 진입 → 0.59× DRAM이 시간으로 환원)이 전제이며, 공정하려면
MXINT8에도 동일 GQA 커널이 필요하고 그 또한 memory win을 받으므로 win이 보장되진 않는다. →
**현 라운드 공정 최선 = u4 tie.** (상세 [for_fair_comparison.md] Phase 34.)

**최종 설계 확정:** 기본 경로 = `kv_decode_wide_kernel`(staging). `MS_KV_WARPT`(design B)·
`MS_KV_CPASYNC`·split kernel은 gated 대안/documented negative로 보존.

---

## 3. 종합 결과 (RTX 3090, warm; 시간[µs]과 MSAQ/MXINT8 비)

| 단계 | 커널 | 특성 | BF16 | MXINT8 | MSAQ u4 | MSAQ/MX (u2·u3·u4) |
|------|------|------|------|--------|---------|----------------------|
| Prefill | W-only GEMM | compute-bound | 279(cuBLAS) | 2514 | 2400 | 1.09 · 1.10 · **0.95** |
| Prefill | W+A GEMM | compute(IMMA) | 279(cuBLAS) | 2765 | 2182 | **0.90 · 0.92 · 0.79** |
| Prefill | KV write | memory-bound | 43(copy) | 226 | 193 | 0.89 · — · **0.85** |
| Decode | W-only GEMV | memory-bound | 45.6(cuBLAS) | 47.4 | 30.1 | 0.92 · 0.92 · **0.63** |
| Decode | W+A GEMV | memory-bound | 45.6(cuBLAS) | 40.5 | 33.2 | 1.17 · 1.14 · **0.82** |
| Decode | KV append | launch-bound | 16(copy) | 8.7 | 8.7 | ~1.0 (런치 지배) |
| Decode | KV dequant(attention) | **latency-bound** | 190(SDPA) | ~33 | ~34 | ~1.6 · ~1.5 · **~1.0 (tie)** |

> KV dequant는 ver.1의 (불공정) "u4 0.54"에서 **공정 tie**로 정정. MXINT8 thread-per-key 수정 후
> 사이즈 H8·Lk4680·D128 best-vs-best 기준. sub-byte의 P·V half-sector 구조 한계(§2.4).

**한 줄 요약:** **memory-bound이고 reduction 축이 element-내부인 연산(GEMV·KV write)에서
mantissa-sharing이 명확히 이기고(특히 u4), compute-bound GEMM은 INT8 IMMA W+A에서만 crossover,
KV read는 P·V의 키-가로지르는 reduction이 sub-byte를 half-sector로 만들어 공정 tie**다.

---

## 4. End-to-end harness (Llama-3.1-8B, prefill=800/decode=3880, CUDA-graph)

weight/KV 양자화를 독립 knob으로 4 시나리오. KV read MXINT8을 thread-per-key로 공정화한 **최종** 수치:

| 시나리오 | 포맷 | total | /bf16 | /mxint8 |
|---------|------|-------|-------|---------|
| baseline | bf16 | 145.7s | 1.00 | — |
| S1 W-only | MXINT8 | 145.6s | 1.00 | — |
| | **MSAQ-u4** | 115.7s | **0.79** | **0.79** |
| S2 W+A | MXINT8 | 128.9s | 0.88 | — |
| | **MSAQ-u4** | 116.3s | 0.80 | 0.90 |
| S3 KV-only | MXINT8 | 91.9s | 0.63 | — |
| | **MSAQ-u4** | 92.9s | 0.64 | **1.01 (tie)** |
| S4 W-only+KV | MXINT8 | 91.9s | 0.63 | — |
| | **MSAQ-u4** | **63.8s** | **0.44** | **0.69** |

설계 함의:
- **S1 W-only가 MSAQ의 대표 win(0.79).** MXINT8 GEMV는 cuBLAS와 동속이라 end-to-end 이득 0이지만,
  MSAQ wide-load u4 GEMV는 cuBLAS를 이긴다.
- **S3 KV-only는 tie(u4 1.01).** ver.1의 0.92 win은 MXINT8 under-optimization 산물이었다(정정).
- **S4가 최고(0.69).** KV read가 tie여도 **W-only GEMV(진짜 BW-bound)의 win이 지배**하고,
  weight·KV 양자화가 TPOT에서 compound(37.5→16.0ms). bf16 대비는 전부 win.
- weight 양자화는 baseline을 낮추고, KV 양자화는 TPOT 성장곡선을 평탄화(직교).
