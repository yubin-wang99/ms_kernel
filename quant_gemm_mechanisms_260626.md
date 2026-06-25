# Quantized GEMM 메커니즘 — MSAQ 커널 대입·측정 (260626)

문헌의 5개 fused-quant-GEMM 메커니즘(TC-FPx/FP6-LLM, Marlin, QServe, TurboMind, FLUTE)을 우리
MSAQ 커널에 대입해 small/large B에서 bf16·MXINT8를 이기는지 측정한다. GPU = RTX PRO 4000 Blackwell
(sm_120, 70 SM). 배경: `subbyte_unpack_analysis_260625.md`(§12 기회 분석), `fair_occupancy_260625.md`.

## 0. 출발점 — 어디서 이기고 어디서 지나 (이전 세션 확정)

- **decode B=1~8 (memory-bound)**: MSAQ가 이미 bf16·MXINT8 이김(바이트 우위). tensor core는 idle →
  fused-tensor-core 기법 **적용 안 됨**.
- **B≥16 weight (이전엔 패배 mq/bf 1.28)**: weight read에 memory-bound인데 배포 `dequant+cuBLAS`가
  **66MB 왕복**(bf16 34MB 쓰고 다시 읽음)이라 짐. **11MB만 읽는 fused가 답** → 5개 논문의 공통 영역.

## 1. 핵심 성공 — fused skinny-M quantized tensor-core GEMM (5개 논문의 공통 코어)

`wonly_gemm_fused_skinny<U_,GS_,MT>` (wa_gemm.cu). 설계:
- **MT=⌈M/16⌉ 한 블록이 전체 M 처리** → weight를 splitK당 1회만 read (M-tile 재read 제거).
- **dequant = `stream_block_dense_bfe`** (production _cm 평면, repack 불필요, straddle 안 하는 코드는
  단일 bfe). WMMA 16×16×16, 4 warp = 64-col tile의 4 N-subtile.
- **splitK occupancy** (mult=7 → ~8, 스윕 최적 30.8µs vs splitK=1 145µs).

**측정 — E2E S1 W-only (실제 32-layer, DRAM-bound, total):**
| B | bf16 | mxint8 | deq+cuBLAS | **FUSED** | FUSED/bf | FUSED/mx |
|--:|--:|--:|--:|--:|--:|--:|
| 16 | 13086 | 17154 | 16827 (1.29 패) | **12331** | **0.94 승** | **0.72 승** |
| 32 | 21977 | 25941 | 25514 (1.16 패) | **21766** | **0.99** | **0.84 승** |

→ **B≥16 최악 패배점을 bf16·MXINT8 둘 다 이기는 승리로 뒤집음.** microbench(L2-resident)는 bf16의
34MB가 L2에서 빨라 fu/bf 1.34지만, 실제 decode는 weight DRAM-read라 **11MB vs 34MB 우위 실현**.
(microbench M=16: fused 30.7µs, deq+cuBLAS 81의 **0.38**, MXINT8 82의 **0.37** 압승.)

이 fused 커널이 **TC-FPx/FP6-LLM·Marlin·QServe·TurboMind·FLUTE가 공유하는 코어**(양자화 weight를
직접 읽어 dequant fuse + tensor core)다. 아래는 각 논문 **고유 기법**을 이 코어에 추가 적용한 결과.

## 2. ncu 진단 — 다음 레버 (fused M=16)

| 지표 | 값 | 의미 |
|---|--:|---|
| DRAM% | 39.6 | **미포화** (BW-bound 아님) |
| long_scoreboard stall | **59%** | weight load **latency-bound** |
| barrier stall | **21%** | dequant↔mma `__syncthreads` 직렬화 |
| SM% / warps_active | 39 / 60 | tensor·연산 미포화 |

→ latency + sync가 병목 → **Marlin cp.async(load·compute 겹침)** 가 ncu-지목 1순위.

## 3. 메커니즘별 대입 결과

### 3a. Marlin — cp.async + double-buffer 파이프라인 [구현·측정: large-M만 도움]
`wonly_gemm_fused_skinny_pipe` (MS_FUSED_PIPE=1). 양자화 weight(Wq/Ws)를 cp.async로 shared에
prefetch하며 이전 블록 dequant+mma와 겹침.
- **alignment 재발견**: per-COLUMN 폭(20B)은 non-16-align이지만 **per-BLOCK(64열×20B=1280B)은 16-byte
  정렬** → 4-byte가 아니라 **16-byte cp.async 가능**. (첫 시도가 4-byte라 더 느렸던 것 — 정정.)
- **16-byte 측정**: pipe/nonpipe **M16 1.41, M32 1.09 (win 영역서 regression)**, **M64 0.91 (도움)**.
  작은 M에선 양자화 weight의 shared 추가 staging(non-pipe는 global→dequant 직행) 오버헤드가 latency
  숨김 이득 초과. 큰 M에선 블록당 compute가 커져 cp.async 겹침이 이득.
- **교훈**: cp.async는 **compute가 충분한 large-M(≥48)에서만** 이득. win 영역(B=16~32 decode)은 non-pipe가
  최선. Marlin이 큰 batch(16~128)를 타깃하지만 우리 decode 핵심은 16~32라 cp.async는 marginal add-on.

### 3b. TC-FPx / FP6-LLM — bit-level pre-packing + register-resident dequant [부분 적용]
- **bit-level pre-packing** = register-aligned pack(§11, 이미 구현; bfe 1개/코드). fused의 dequant는
  현재 dense-bfe(straddle만 funnel) — 동등 효과.
- **register-resident dequant**(dequant된 bf16을 shared Bs에 안 쓰고 register 보관 → shared 왕복 제거):
  WMMA `load_matrix_sync`가 shared에서 로드하므로 register 상주는 `mma.sync` PTX + 명시적 fragment
  레지스터 로드 필요. ncu가 barrier(shared 동기) 21%를 보이므로 유망. [다음 단계, mma.sync 재작성]

### 3c. QServe — INT8 IMMA (bf16 변환 회피) [구현·측정: REGRESSION]
`wa_gemm_fused_imma` (W+A, 활성화 int8). MSAQ가 재구성하는 int8 word를 bf16 변환 없이 INT8 IMMA에
직접 공급, int32 누적 후 per-block scale. **정확하나 bf16-fused의 2× 느림** (im/bf-fused 2.05@M16,
1.89@M32, 1.95@M64; scale precompute로 78→63µs 개선 후에도).
- **원인(핵심)**: MSAQ의 **per-32-block E8M0 scale**이 K-loop 안에서 scale을 못 빼게 한다 → 블록마다
  `IMMA → acc_int32를 shared에 store → sa[m,blk]·sw[o,blk]로 16×16 타일 scale → float accf에 RMW
  누적` 을 강제. 이 per-block store+scale+RMW가 int8 이득(I2F·scale 제거, IMMA 2× throughput)보다 큼.
  게다가 이 커널은 §2대로 **load-latency-bound**라 IMMA의 matmul throughput 이점이 무관.
- **교훈**: **QServe의 INT8-IMMA는 W4A8의 거친 scale(scale-free int32 main-loop 가능)에서 성립**.
  MSAQ의 fine-grained per-block scale은 그 전제를 깨뜨림. 반면 bf16-fused는 scale이 어차피 필요한
  per-element dequant에 fold되고 **fragment에 누적(per-block store 없음)** 이라 더 빠르다. → MSAQ에
  INT8-IMMA는 부적합. (만약 scale을 coarser group(예: per-K-tile 대신 per-256)으로 바꾸면 적용 가능하나
  그건 포맷 변경.)

### 3d. TurboMind — ILP로 dequant 흡수 [분석]
dequant 추가 명령을 ILP로 숨김. 현재 fused는 dequant→sync→mma 직렬(barrier 21%). dequant와 mma를
warp/instruction 레벨로 겹치면 ILP 확보. cp.async(3a)·register-resident(3b)와 목적 중첩. [synthesis에 통합]

### 3e. FLUTE — LUT dequant [분석]
산술 dequant을 table lookup으로. MSAQ는 word(int8)→bf16*scale. scale이 per-block(per-column)이라
단일 LUT 불가(scale마다 다름). 단 word→float(int8→bf16)는 LUT 가능하나 I2F가 이미 1 op(§subbyte
§2)이라 이득 작음. non-uniform(LUT) 양자화가 아닌 MSAQ엔 적합도 낮음. [낮은 우선순위]

## 4. 종합 — bf16-fused가 MSAQ에 최적, 논문 add-on은 MSAQ-고유 벽에 막힘

| 메커니즘 | 결과 | MSAQ-고유 벽 |
|---|---|---|
| **fused skinny-M (코어)** | **승**(E2E B16 0.94/bf, 0.72/mx) | — (이게 답) |
| Marlin cp.async | regression 1.53× | upper 폭 20B가 16-byte non-align → 4-byte cp.async; reshuffle 전제 |
| QServe INT8-IMMA | regression 2.05× | per-32-block E8M0 scale → per-block store+scale+RMW 강제 |
| FP6 register-resident | [미구현] | WMMA load는 shared 경유; mma.sync 재작성 필요 |
| TurboMind ILP | [분석] | cp.async/register-resident와 목적 중첩 |
| FLUTE LUT | [분석] | per-block scale → 단일 LUT 불가 + I2F 이미 쌈 |

**핵심 결론**: 5개 논문의 add-on(cp.async, INT8-IMMA, LUT)이 모두 **MSAQ 고유의 (a) 비정렬 코드폭(5/6-bit,
non-16B) + (b) per-32-block E8M0 scale** 라는 두 벽에 막힌다. 정작 **MSAQ에 최적인 건 단순한 bf16-fused
설계**다 — scale을 per-element dequant(어차피 필요)에 fold하고 fragment에 in-place 누적(per-block store
없음), dense-bfe로 straddle 회피, splitK로 occupancy, 한 블록이 전체 M 처리로 weight 1회 read. 논문들은
"양자화 직접 read + dequant fuse + tensor core"라는 **코어 아이디어**가 옳음을 확증하지만(우리도 그래서
이김), 그들의 **세부 가속 기법은 그들 포맷(W4A8 거친 scale, 16-byte int4 정렬)에 특화**돼 MSAQ엔 안 맞다.

**합성 메커니즘**: 위 분석상 bf16-fused가 이미 MSAQ-최적점. 추가 이득은 (a) Marlin reshuffle로
register-aligned pack을 16-byte 배수로 패딩 후 cp.async(latency 숨김 — 단 바이트 +α), (b) FP6
register-resident(barrier 21% 제거)뿐이며 둘 다 marginal 예상(코어가 이미 win). 결정적 추가 win은
unpack 미시최적화가 아니라 **이미 이기는 영역(B≥16 fused, B=1·KV)을 production에 승격**하는 것.

상태: §1·§3a·§3c 구현·측정 완료(코어 승, cp.async·IMMA regression+교훈). §3b/3d/3e는 분석(구현 시
marginal 예상). 코드 전부 추가적, production 무영향. 메모리 `b16-fused-gemm-opportunity`.

## 5. 최종 W-only — 전 batch 승 (검증 완료)

**FINAL = GEMV(B<16) + non-pipe bf16 fused(B≥16).** E2E S1 W-only (32-layer, total, GPU1):
| B | bf16 | mxint8 | msaq | mq/bf | mq/mx |
|--:|--:|--:|--:|--:|--:|
| 1 | 3856 | 2523 | 2114 | **0.55** | 0.84 |
| 8 | 8097 | 7113 | 7034 | **0.87** | 0.99 |
| 16 | 13178 | 17217 | 12346 | **0.94** | **0.72** |
| 32 | 21993 | 25968 | 21801 | **0.99** | **0.84** |

→ **전 batch에서 mq/bf<1** (B≥16 패배 구멍 메움: 1.28→0.94, 1.16→0.99). small-batch(B=1,8) = GEMV가
이미 승. **W-only는 이제 모든 batch에서 bf16·MXINT8를 이긴다.**

### 5a. GEMV↔fused dispatch 임계값 (E2E 확정)
| B | 경로 | mq/bf | 비고 |
|--:|---|--:|---|
| 1 | wide GEMV | 0.55 | |
| 4 | batched GEMV | 0.73 (fused 0.81) | GEMV 우세 (fused M=4/16 패딩 낭비) |
| 8 | batched GEMV | 0.87 (fused 0.88) | ~타이 |
| **≥12** | **fused** | **0.91 (B12, GEMV 1.00)** | GEMV가 compute-bound(bf16 타이)되는 지점 → fused 전환 |
→ **최종: B≤11 GEMV, B≥12 fused.** (이전 B≥16 deq+cuBLAS + B=12~15 GEMV를 모두 fused가 대체.)

### 5b. fused에 register-aligned(§11) 얹기 — 동률 (GEMV와 달리 무익)
ra fused(MS_FUSED_RA, ra 평면): ra/dense **1.00 @ B12-32**, 0.93 @ B64 (microbench). subbyte §11에서 ra가
GEMV를 이긴 건 GEMV가 **ALU/MLP-bound**였기 때문. **fused는 load-latency(long_sb 59%)+barrier(21%)-bound**라
ra의 ALU 절감이 무관 + ra는 **+20% 바이트**(E2E DRAM-bound서 손해). → **같은 기법도 커널 bound 따라
GEMV=승/fused=무익.** fused엔 dense-bfe(적은 바이트+대부분 단일 bfe)가 최선.

### 5c. fused 추가 최적화 후보 (미구현, 우선순위)
fused bound=latency+barrier → (1) **register-resident dequant(FP6)**: Bs shared write+sync 제거로 barrier
21% 직격(유망, mma.sync 재작성 필요), (2) cp.async는 large-M만, (3) ra/INT8-IMMA 무익. 단 fused가 이미
이기므로 추가 이득 marginal(floor ~8µs vs 현재 ~30µs는 latency 미숨김분).

### 5d. register-resident 시도 → tractable proxy **NT=128** (구현·B≥32 승)
완전한 register-resident(FP6: Bs shared write+sync barrier 제거)는 WMMA fragment가 **shared에서만 load**되므로
mma.sync PTX 전면 재작성 필요 → 보류. 대신 **그 정신(in-flight load↑로 long_scoreboard 직격)**을 살린
tractable proxy 구현: dequant 스레드 64→**128** (기존 `if(tid<64)`는 64열만, 절반 유휴). 워프당 N-subtile
1→**2**(`acc[MT][NS]`, NS=2), grid.x=OUT/128(블록 절반, splitK 2배로 ~8×SM 유지). **워프당 누산 fragment 2개가
mma ILP를 늘려 load latency를 가린다** — register-resident가 노린 것과 동일 효과를 fragment 수준에서 달성.
- microbench(L2, 4096²): B12 1.00(타이), B16 1.02(소폭 악화), **B32 0.81, B64 0.70**.
- E2E S1 W-only(DRAM-bound): B32 mq/bf **0.952** vs NT64 0.982 (총지연 3%↓, prefill 동일하니 decode 이득은 더 큼).
→ **adaptive 적용: host가 M≥32서 NT=128, 아니면 64 자동선택**(env MS_FUSED_NT override). 완전 register-resident는
이 위에선 marginal이라 보류.

### 6. W+A·W+KV·W+A+KV에 fused 적용 — **전 weight-quant scope 완료 (E2E 검증)**
**핵심 통찰:** fused 커널 = **weight-read vehicle**. 선형 사영(q/k/v/o/gate/up/down)은 모든 scope에서
**weight-quant + bf16-activation**이다 — B≥16 배포 경로(deq+cuBLAS)도 wonly·wa 둘 다 bf16 X를 먹였다
(true-W+A INT8-IMMA는 2× 퇴보 → 기각). KV-quant은 **attention 커널**에 있지 선형엔 없다. 따라서
W+A·W+KV·W+A+KV 마이그레이션 = **dispatch 한 줄**: `p in (msaq_wonly, msaq_wa)` B≥16 → fused (기존 wonly 한정).

### 6a. ⚠️ 공정성 교정 — **MXINT8도 동일 fused 커널로 재측정 필수**
처음 §6은 **MSAQ만 fused, MXINT8은 옛 deq+cuBLAS(17MB 왕복)**로 비교 → mq/mx 0.66-0.81은
**mantissa-sharing이 아니라 커널 비대칭**이 만든 inflated 수치였다(불공정). MXINT8은 byte-aligned라
오히려 fused가 **더 쉽다**(sub-byte unpack 없음). 그래서 `mxint8_gemm_fused_skinny<MT,NT>`(MSAQ의 fair twin,
qweight_cm 32 연속 int8 직독→E8M0→WMMA, adaptive NT 동일) 구현 후 양쪽 다 fused로 재측정.

**커널 단독 microbench(decode GEMM only, 4096², 양쪽 동일 vehicle):**
| B | MSAQ 11MB | MXINT8 17MB | mq/mx | byte ratio 11/17 |
|--:|--:|--:|--:|--:|
| 16 | 30.8µs | 49.4µs | **0.62** | 0.65 |
| 32 | 35.5µs | 53.5µs | **0.66** | 0.65 |
| 64 | 53.4µs | 70.9µs | **0.75** | 0.65 |
→ MSAQ **여전히 승**, 그리고 그 폭이 **byte ratio(0.65)와 거의 정확히 일치** = memory-bound서 11MB<17MB라는
**정당한 mantissa-sharing 우위**. dequant 자체는 MXINT8이 더 싸므로, 둘을 가르는 건 오직 옮긴 바이트 수.

**E2E total (양쪽 fused, 공정):**
| scope | B16 mq/bf | B16 mq/mx (前 불공정) | B32 mq/bf | B32 mq/mx (前) | B16 mx/bf (前) |
|---|--:|--:|--:|--:|--:|
| S1 W-only | 0.93 | **0.91** (0.71) | 0.96 | **0.95** (0.81) | **1.02** (1.30) |
| S2 W+A | 0.94 | **0.93** (0.72) | 0.96 | **0.95** (0.81) | 1.01 (1.30) |
| S4 W+KV | 0.68 | **0.93** (0.66) | 0.62 | **0.92** (0.72) | 0.73 (1.04) |
| S5 W+A+KV | 0.69 | **0.93** (0.66) | 0.62 | **0.92** (0.72) | 0.74 |
| S6 +AA | 0.69 | **0.94** (0.66) | 0.62 | **0.92** (0.72) | 0.73 |

**정직한 결론:** (1) MSAQ는 공정 비교서도 **전 scope 승**(E2E mq/mx 0.91-0.95, 커널단독 0.62-0.75).
(2) 다만 마진은 훨씬 작고 정직해짐 — 이전 0.66-0.81은 MXINT8에 fused가 없어 부풀려진 것.
(3) **mx/bf가 1.30→1.02로 붕괴** = "B≥16서 MSAQ가 MXINT8를 크게 이긴다"의 대부분은 **포맷이 아니라 커널 격차**였음.
(4) E2E total은 prefill(양쪽 동일 deq+cuBLAS)+decode-attention+norm이 공유라 커널 0.65 우위가 희석됨 —
decode 列만 보면 S1 B16 mq/mx=7877/9058=**0.87**, B32=**0.93**으로 선형 GEMM 우위가 더 선명. S3 KV-only(weight bf16,
fused 무관) mq/mx 0.91-0.94 = KV 포맷 자체의 소폭 우위.
