# 커널 성능 분석 — MS가 BF16/MXINT8를 언제·왜 이기나 (260625)

GPU = RTX PRO 4000 Blackwell (sm_120, 70 SM). 비율 표기: **ms/mx, ms/bf < 1 = MS 승.**
근거: `subbyte_unpack_analysis_260625.md`, `perf_analysis_principle.md`,
`quant_gemm_mechanisms_260626.md`, `kernel_ver260625.md`, 메모리
`msaq-vs-mxint8-{w,kv}-decode-state`, `b16-fused-gemm-opportunity`.

---

# 1. MS가 이기려면 대상이 어떤 bound여야 하나 — 그리고 왜

## 핵심 명제

> **MS의 무기는 단 하나, "바이트 절감"(MXINT8의 0.66× / bf16의 0.32×)이고, 이건 오직 대상이
> *DRAM-bandwidth-bound*일 때만 시간으로 전환된다.** MS는 그 무기에 **서브바이트 언팩 ALU 세금**
> (MXINT8 대비 정수 ALU 3.7×, 발행 명령 1.41×)을 항상 같이 낸다. 승패 = "바이트 이득이 언팩
> 세금을 넘느냐"이고, 이건 bound가 결정한다.

| 대상의 bound | MS의 무기(바이트↓)가 통하나 | 결과 |
|---|---|---|
| **compute-bound** (intensity ≫ ridge; prefill GEMM) | ✗ — 바이트는 7%, FLOP이 한계 | **bf16과 타이** |
| **memory-latency-bound** (B=1 decode, idle 슬롯 풍부) | △ — 바이트 일부 전환 + 언팩이 idle에 숨음 | **둘 다 승** (vs bf16 0.5~0.6, vs mx ~0.84) |
| **부분 bandwidth-bound** (B=2~15 decode, u2/u3 KV 중배치) | ✗ — 언팩이 MLP를 빼앗아 BW 미포화 | **bf16은 이기나 MXINT8엔 짐** (mx 1.07~1.21) |
| **깊은 bandwidth-bound** (대배치·긴 문맥 KV, B≥16 fused weight) | ✓ — 바이트가 그대로 시간 | **둘 다 압승** (0.4~0.7) |

## 1-A. compute-bound에서 못 이기는 이유 (가설 ① 확인 + 정정)

"어차피 INT8로 돌려야 하고, MXINT8보다 복잡한 서브바이트 언팩이 추가돼서"는 방향이 맞지만,
정확히는 **"지는" 게 아니라 "타이"** 다.

- prefill GEMM은 intensity≈M(1024) ≫ ridge(76) → **13.5× compute-bound**. lever는 바이트가 아니라
  **FLOP / 연산 포맷(INT8 텐서코어 2× throughput)**. 가중치 read는 전체의 **7%** → 절반 줄여도 ~4% → bf16 동등.
- 이기려면 INT8 IMMA 2× FLOP를 살려야 하는데 **MS는 못 쓴다**:
  1. 서브바이트 코드를 먼저 int8 word로 언팩해야 함(추가 ALU).
  2. **MSAQ의 per-32-block E8M0 scale**이 K-loop 안에서 scale을 못 빼게 함 → 블록마다
     `IMMA → int32 shared store → 16×16 타일 scale → float RMW 누적` 강제. 이 per-block 부기가 IMMA
     이득을 잡아먹음 → 실측 **bf16-fused 대비 2.05× 느림(regression)**. (QServe INT8-IMMA가 성립하는
     건 W4A8의 거친 scale = scale-free main-loop일 때뿐.)
- 그래서 MS는 prefill에서 **"양자화 weight를 bf16으로 한 번 풀어 cuBLAS에 태우는"** 길로 후퇴 →
  weight read 7%라 **bf16 타이**. (MXINT8도 동일 — `torch._int_mm` eff 38%라 INT8 2× FLOP 못 살림.)

**정정:** compute-bound에서 MS는 "지는" 게 아니라 **"바이트 무기가 무의미해져서 타이"**. 진짜 패배는 1-B.

## 1-B. 부분 bandwidth-bound에서 MXINT8에 지는 이유 (가설 ② = 정답)

"ALU를 점유해서 in-flight memory 요청 수가 적어져 BW 활용도가 떨어지고 bits 절감 효과가 줄어든다"가
**ncu로 확정된 정확한 메커니즘**. 이 프로젝트에서 가장 중요한 발견.

decode B=4 W-only, ncu (ms vs mx):

| 지표 | MSAQ | MXINT8 | 의미 |
|---|--:|--:|---|
| **메모리 대기 워프 / issue (= MLP)** | **5.6** | **18.0** | Little's law: in-flight 로드 수 |
| DRAM throughput % | 74 | **82** | MS가 BW 미포화 |
| long-scoreboard stall % (메모리 대기) | 46 | **72** | mx는 순수 메모리 대기 |
| SM/issue % | **60** | 30 | MS는 ALU로 바쁨 |
| 정수 thread-op | **93.7M** | 26.1M (3.6×) | 언팩 세금 |
| FP thread-op | 84.7M | 84.7M (**동일**) | 실제 MAC은 같음 |

메커니즘 (트럭 비유):
- **MXINT8 워프**: `LDG → 즉시 메모리 대기`(언팩 없음). 항상 ~18워프 분량 로드 in-flight → 높은 MLP
  → **DRAM 82%(BW 포화)** → 바이트가 곧 시간.
- **MSAQ 워프**: `LDG → 언팩 연산(바쁨) → FFMA → 다음 LDG`. 언팩 중엔 그 워프 로드가 이미 끝나
  **대기 아님** → in-flight 5.6뿐 → 낮은 MLP → **DRAM 74%(BW 미포화)**.

즉 언팩은 **"메모리 대기 사이클"을 "연산 사이클"로 바꿔치기**하며 동시에 **메모리 동시성(MLP)을
빼앗아** BW를 못 채운다 → 0.66× 바이트 우위 미실현. **MS는 자기 memory floor의 ~50%에서 돈다**
(floor 닿으면 vs mx 0.54~0.66일 텐데).

정정 두 개:
- **병목은 발행 슬롯 경합도, 레지스터 압박도, ALU 처리량도 아님** (issue 67%·reg 46≈48·math throttle
  3%). 매개는 순수 **MLP**.
- **명령어 스케줄 변경은 no-op**: funnel-shift 독립추출(`MSAQ_INDEP_UNPACK`)은 bit-exact지만
  SASS·wall-clock 동일(nvcc가 이미 동등 최적화). 진짜 레버는 **언팩 명령 *총량* 감소**뿐.
- wall-clock 패널티(1.0~1.21×) ≪ 명령어 패널티(1.4~3.7×): 언팩 ALU가 메모리 latency 뒤에 **부분
  은닉**. 완전 issue-bound면 ~1.4×, 부분 은닉으로 ~1.1×.

## 1-C. memory-bound 안의 "골짜기(U자)" — 빠지기 쉬운 핵심

memory-bound 안에서도 **latency-bound와 bandwidth-bound가 다르고**, MS 승패는 batch에 따라 U자.

```
B=1 (latency-bound)        B=2..15 (부분 BW-bound)        대배치/긴문맥 (깊은 BW-bound)
idle 슬롯 多 → 언팩 숨음      idle 줄고 언팩이 MLP 빼앗음        바이트가 압도적
→ MS 승 (vs bf 0.55)        → MS 패 (vs mx 1.07~1.21)        → MS 승 (vs mx ~0.9, bf 0.4~0.6)
        \________________________ 골짜기(valley) _______________________/
```

- **B=1**: memory-**latency**-bound. idle issue 슬롯 多 → 언팩 숨음 + 여전히 memory-bound라 바이트 전환
  → vs bf16 **0.5~0.6**, vs mx **~0.84**.
- **B=2~15 (W) / u2·u3 KV 중배치**: batch가 메모리 amortize → idle 감소 → 언팩 노출 + MLP 절도 →
  **골짜기**. bf16은 이기지만 MXINT8엔 짐.
- **깊은 BW-bound (대배치·긴 문맥 KV, B≥16 fused weight)**: 바이트 절감 누적 → 언팩 세금 압도 → 둘 다 승.

그리고 **언팩이 싼 경우(u4 nibble)는 골짜기 소멸**: u4는 nibble 정렬 → 단일 `bfe.s32`(HW sign-extend,
1 op). straddle(u2/u3) 대비 정수 ALU 1.7×↓, 시간 1.9×↓ → **u4 KV는 전 배치에서 MXINT8 승**.

---

# 2. Scope × Phase × Batch — 무슨 연산이고, BF16/MXINT8 특성은

## 2-A. 어떤 연산이 도는가 (production 커널)

| Phase | 연산 | production 커널 | 무슨 일 |
|---|---|---|---|
| **Prefill** | W GEMM | `ms_dequant_bf16`+cuBLAS | X[M,K]@W[K,OUT], M=L_in |
| | A-quant (W+A) | `quant_act_msaq` | X→int8 word + per-block E8M0, 1 op/elem |
| | KV write | `kv_write` | 전체 [B,Hkv,Lp,hd] K/V 1회 양자화 저장 |
| | Attention | bf16 fused flash(SDPA) | (AA low×low는 prefill 2.1~2.7× 짐 → 비production) |
| **Decode** | W GEMV (B=1) | `wonly_gemv_wide_uspec` | x[1,K]@W |
| | batched GEMV (B=2~15) | `wonly_gemv_batched_uspec` ⭐ | x[B,K]@W, 활성화 shared staging |
| | fused GEMM (B≥12~16) | `wonly_gemm_fused_skinny` ⭐260626 | 양자화 weight 직접 read→bf16 fragment→WMMA |
| | A-quant (W+A) | `quant_act_msaq` | 매 step, ~13µs |
| | KV append | `kv_append`/`_rot` | 새 K/V 1칸 양자화 |
| | KV read=attention | `kv_decode_wide_kernel` | 양자화 K/V 읽어 Q·Kᵀ, P·V (**Q는 bf16**) |

## 2-B. BF16 기준 각 연산의 bound (그리고 왜)

| 연산 | intensity | BF16 bound | 왜 |
|---|---|---|---|
| **Prefill W GEMM** | ≈M(1024)≫76 | **compute-bound** | FLOP 한계, weight read 7% |
| **Prefill attention** | L에 비례 | compute-bound (flash) | O(L²) MAC |
| **KV write** | — | **bandwidth-bound (store)** | 전체 K/V 스트리밍 1회 |
| **Decode W GEMV B=1** | ≈1≪76 | **memory-latency-bound** | weight read 지배, idle 多 (MLP 낮음) |
| **Decode W batched B=2~15** | ≈B≪76 | **memory-bound (→ BW로 이동)** | batch가 weight read amortize |
| **Decode W B≥16** | ≈B | memory-bound(weight read) | 텐서코어 일하나 weight DRAM-read가 한계 |
| **A-quant (decode)** | — | **bandwidth-bound** | X 읽고 qX 쓰기, traffic 고정 |
| **KV append** | — | **launch-bound** | 일감 1칸, 그리드 극소 |
| **KV read (attention)** | ≈B≪76 | **bandwidth-bound** | KV 바이트가 batch·문맥에 비례 누적 |

## 2-C. MXINT8로 가면

- **Prefill GEMM**: compute-bound이라 바이트(17MB) 무의미. INT8 2× FLOP 살려야 이기는데 eff 38% →
  **bf16 타이**. (block-scaled IMMA는 cuBLAS 비호환 + IMMA 5.6%로 실패.)
- **Decode (W·KV)**: int8 = `load → I2F → ×scale(FFMA fold)`, **언팩 0** → LDG 직후 곧장 메모리 대기 →
  **MLP 18, DRAM 82%** = 순수 memory-(latency→bandwidth)-bound → 바이트(0.5×bf16)가 그대로 시간.
  **MXINT8가 강한 이유 = 0-언팩 → 높은 MLP → BW 포화**. MS가 골짜기에서 못 넘는 벽.
- **A-quant / KV write/append**: 메모리-(또는 launch-)bound라 포맷 차이(MSAQ-s vs MXINT8 활성화)는
  BW 뒤 은닉 → **타이** (`quant_act_unsigned` un/sg 0.96~1.02).

---

# 3. 현재 커널 scope별 승패 — 왜 + batch 변화

scope: S1 W-only, S2 W+A, S3 KV-only, S4 W+KV, S5/S6 W+A(+AA)+KV. (ms/bf, ms/mx < 1 = MS 승)

## 3-A. Prefill (모든 scope·batch)

- **vs bf16: 타이** (dequant+cuBLAS = bf16 GEMM + dequant, weight read 7%). compute-bound이라 바이트 무의미.
  fused-WMMA는 MMA를 굶겨 4× 느려 폐기.
- **vs MXINT8: 타이** (둘 다 dequant→cuBLAS, INT8 2× FLOP 못 살림).
- **batch 무관** (prefill M은 batch가 아니라 L_in).

## 3-B. Decode W-only (S1) / W+A (S2)

| B | mq/bf | mq/mx | 경로 | 사유 |
|--:|--:|--:|---|---|
| 1 | **0.55** | 0.84 | wide GEMV | latency-bound + idle에 언팩 숨음 → **둘 다 승** |
| 8 | **0.87** | ~1.0 | batched GEMV (shared-act) | bf16 추월(바이트), **mx 타이** (골짜기 진입) |
| 2~15 | <1 | **1.07~1.21** | batched GEMV | 골짜기: 언팩이 MLP 절도 → **mx에 짐** |
| ≥16 | **0.93** | **0.71** | fused skinny GEMM ⭐ | 11MB 직접 read(bf 34/mx 17 round-trip) → **둘 다 압승** |
| 32 | **0.96** | **0.81** | fused (NT=128) | 동일 메커니즘, decode 비중↑ |

- **B=1 승**: latency-bound (1-C).
- **B=2~15**: shared-activation 수정(260625)으로 활성화 재로드 병목(L1 87%) 제거 → **bf16 추월(0.87)**,
  **MXINT8엔 골짜기라 짐**. W+A는 `(qa·sa)·(qw·sw)`로 int8 dot 대신 float MAC → S2≈S1.
- **B≥16 fused (260626, 이번 핵심 반전)**: 기존 `dequant+cuBLAS`는 bf16 34MB 쓰고 cuBLAS가 다시 읽는
  **66MB 왕복**이라 bf16보다 *더* 읽어 짐(mq/bf 1.28). fused는 양자화 weight **11MB만** 직접 read →
  bf16 0.32×/mx 0.5× → **둘 다 압승**. weight read가 memory-bound인 텐서코어 regime이라 바이트가
  그대로 시간. (MXINT8는 fused 커널 없어 17MB 왕복 → mx/bf 1.30, MS 11MB 직접 read가 차별점.)
- **register-aligned(§11)는 GEMV는 빠르게(ra/ms 0.91, ALU−18%→MLP+52%), fused엔 무익(tie)**: GEMV는
  ALU/MLP-bound, fused는 load-latency(long_sb 59%)+barrier(21%)-bound라 ALU 절감 무관 + ra는 +20%
  바이트(DRAM-bound 손해). **같은 기법도 커널 bound 따라 승/무익**.

## 3-C. Decode KV (S3 KV-only / S4 W+KV / S5·S6 +AA)

| 포맷 | B=8 | B=32 | vs bf16 | vs MXINT8 | 사유 |
|---|--:|--:|---|---|---|
| **u4/gs2 (S3 nibble)** | 0.80 | **0.55** | **승** | **승**(0.55~0.80) | nibble 단일 bfe(언팩 쌈) + 바이트↓ → 골짜기 없음 |
| **u2/gs8 (straddle)** | 1.33 | **0.92** | 승 | **중배치 패(1.33)/대배치 승(0.92)** | straddle 언팩 세금 = W와 동형 |

- **KV는 batch·문맥이 클수록 잘 이긴다**: KV 바이트 누적 → 깊은 BW-bound (1-C 오른쪽). B=32 최고.
- **u2/u3 straddle은 중배치에서 occupancy-bound라 짐**: nibble 재배치는 3번째 필드(low_un) 불가피해
  **30~37% 더 느림** → "streaming straddle이 u2/u3엔 이미 최적, 어떤 재배치도 못 넘음".
- **공정성 주의 2개**: ① KV occupancy `mult=3`(82-SM 3090 튜닝)은 70-SM Blackwell서 MXINT8 under-occupy
  → MS 승 부풀림. 공정값 `mult≈4`로 재면 MS 우위 축소. ② isolated K-dot MXINT8 baseline은 반드시
  **wide-load(int4×2)** — scalar 바이트 로드면 mx 2.5× 느려져 거짓 승.
- **AA(S6)는 latency 비용 0**: decode는 Q를 bf16으로 읽고 KV 바이트에 memory-bound, Q는 극소 →
  **S6=S5 같은 커널**. AA는 정확도 비용(~+0.9pp PPL)일 뿐.
- **S4/S5 W+KV는 fused linear(B≥16) + KV-quant decode 합산으로 최고 승**: B16 **0.68/0.66**,
  B32 **0.62/0.72** (이전 0.84/1.05 = mx에 지던 것 반전).

## 3-D. 종합 — 모든 scope × batch (mq/bf, 최신 260626)

| scope | B=1 | B=8 | B=16 | B=32 |
|---|--:|--:|--:|--:|
| S1 W-only | 0.55 | 0.88 | **0.93** | 0.96 |
| S2 W+A | ~0.55 | 0.89 | **0.94** | 0.96 |
| S3 KV-only | 승 | 0.76 | 승 | 0.65 |
| S4 W+KV | 승 | 0.67 | **0.68** | 0.62 |
| S5/S6 +AA | 승 | 0.68 | 0.68 | 0.62 |

**결론**: 260626 fused로 **B≥16 weight scope 패배 구멍(1.28→0.93)을 메워, 이제 전 batch·전 scope에서
bf16·MXINT8 모두 승**. 단 그 승리는 두 길로만 옴 — **(a) 깊은 BW-bound regime**(B=1, 대배치/긴문맥
KV, B≥16 fused weight-read)에서 바이트가 시간 전환, **(b) u4 nibble**처럼 언팩이 애초에 쌀 때.
**골짜기(B=2~15 W-GEMV, u2/u3 KV 중배치)에서 MXINT8를 결정적으로 못 넘는 건 구조적**(언팩 ALU가
MLP를 빼앗아 BW 미포화)이며, 언팩 미시최적화(funnel/nibble/unsigned/register-aligned 전부
tie-or-worse)로는 해소 불가 — 소진됨.

---

# 부록 — 정량 요약 (한눈)

```
원소당 정수 ALU   : MXINT8 ~0   |  MSAQ ~5-7 (straddle u2/u3) | ~3 (nibble u4)
발행 명령(warp)    : MSAQ/MXINT8 = 1.41×   |  정수 ALU = 3.7×   |  LSU(로드) = 1.08×(동일)
정수 thread-op     : MXINT8 26.1M | MSAQ 93.7M (3.6×)  |  FP thread-op 84.7M (동일)
bound 이동         : MXINT8 memory-latency(long-SB 72%) → MSAQ issue/ALU(SM 60%)
MLP(메모리대기/iss): MXINT8 18.0  | MSAQ 5.6  → DRAM 82% vs 74%
시간 패널티        : 명령 1.4~3.7× → wall-clock 1.0~1.21× (latency 뒤 부분 은닉)
straddle 세금       : u2(1436M ALU/502µs) vs u4(853M/266µs) = 1.7× ALU / 1.9× 시간
weight 바이트       : bf16 34MB | MSAQ 11MB | MXINT8 17MB  (B≥16 fused가 11MB 직접 read = 승)
```
