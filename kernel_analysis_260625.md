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

---

# 4. Insight (요약 문장)

## 4-A. Workload 특성 (prefill vs decode)

- Prefill stage는 입력 시퀀스 전체(L_in개 토큰)의 activation matrix와 weight 간 GEMM을 수행하므로
  arithmetic intensity가 ridge point를 크게 상회하는 **compute-bound** 연산이며, weight load는 전체
  실행시간의 7% 수준에 불과해 양자화의 memory traffic 절감이 시간으로 전환되지 않습니다.
- 반면 decode step은 토큰 1개의 activation vector와 전체 weight의 GEMV를 수행하므로, 매 token 생성
  시 모델 전체 weight와 누적된 KV cache 전체를 메모리에서 load하는 **memory-bound** 연산이고, 따라서
  byte를 줄이는 양자화가 latency로 직결됩니다.
- 같은 memory-bound라도 batch=1 decode는 in-flight load가 적어 **latency-bound**(유휴 issue slot이
  많음)인 반면, batch가 커지면 weight load가 amortize되며 점차 **bandwidth-bound**로 이동하므로,
  동일한 양자화 기법이라도 batch 구간에 따라 효과가 달라집니다.
- KV write(prefill 일괄)와 KV append(decode 매 step)는 각각 streaming store와 launch-bound 연산이라
  양자화 포맷 차이가 memory latency 뒤에 은닉되어 latency에 거의 영향을 주지 않으며, KV 양자화의
  실질 이득은 오직 KV read(attention) 단계의 traffic 감소에서 나옵니다.

## 4-B. Batch size·sequence length 증가에 따른 영향

- Decode에서 weight load는 batch 전체가 공유(amortize)하지만 KV cache는 sequence·batch에 비례해
  누적되므로, sequence length와 batch size가 커지면 KV cache 용량이 weight 용량을 초과하게 되어
  **KV cache quantization의 이점이 weight-only quantization의 이점을 능가**합니다.
- Weight-only 양자화의 byte 이득은 batch가 커져도 일정(weight 크기 고정)하지만 GEMV의 arithmetic
  intensity는 batch에 비례해 증가하므로, 일정 batch를 넘으면 weight scope가 compute-bound 쪽으로
  이동해 cuda-core GEMV가 폭발하고, 이를 피하려면 **양자화 weight를 직접 읽어 tensor-core에 공급하는
  fused GEMM**으로 전환해야 합니다.
- 반대로 KV scope는 batch·문맥이 커질수록 점점 더 깊은 bandwidth-bound가 되어 byte 절감이 그대로
  시간이 되므로, **batch가 커질수록 KV 양자화가 가장 잘 이기는 구간**이 됩니다.

## 4-C. MS 자체의 이점과 한계

- MS의 유일한 무기는 mantissa-sharing을 통한 **byte 절감(MXINT8 대비 약 0.66×, BF16 대비 0.32×)**
  이며, 이 무기는 대상 연산이 bandwidth-bound일 때만 시간으로 전환됩니다.
- MS는 그 byte 이득에 대해 항상 **sub-byte unpack ALU 세금**(MXINT8 대비 정수 명령 약 3.7×)을 함께
  지불하는데, 이 unpack 연산이 단순히 명령 수를 늘리는 것을 넘어 **warp가 메모리를 기다리는 대신
  연산하게 만들어 in-flight load 수(MLP)를 떨어뜨리고, 그 결과 DRAM bandwidth를 포화시키지 못해 byte
  절감 이득이 미실현**되는 것이 한계의 본질입니다.
- 따라서 MS는 unpack 세금이 유휴 사이클에 숨는 **latency-bound 구간(batch=1)** 과 byte가 unpack을
  압도하는 **깊은 bandwidth-bound 구간(대배치·긴 문맥)** 에서는 이기지만, 그 사이의 **중간 batch
  구간에서는 unpack이 MLP를 빼앗아 MXINT8의 0-unpack memory-bound 우위를 넘지 못하는 "골짜기"** 를
  보입니다.
- 이 골짜기는 명령 스케줄링(funnel-shift 재배치)이나 bit re-layout(nibble 재배치, register-aligned)로는
  해소되지 않는 **구조적 한계**이며, 유일한 예외는 코드 폭이 nibble(4-bit)로 정렬되어 단일 hardware
  bfe로 unpack이 끝나는 경우(unpack이 애초에 싸짐)뿐입니다.

## 4-D. Quantization scope별 특징과 batch에 따른 변화

- **Weight-only / KV cache quantization은 memory traffic 감소를 통해 decode latency 절감에 직접 기여**
  하며, MS는 latency-bound인 batch=1과 bandwidth-bound인 대배치에서 BF16·MXINT8를 모두 이기지만,
  중간 batch 구간(W-only)·중배치 straddle 포맷(KV)에서는 MXINT8와 타이거나 패배합니다.
- 반면 **weight and activation quantization은 low-bitwidth GEMM을 통해 compute throughput을 높이지만,
  decode는 memory-bound 특성을 띠므로 그 throughput 이득은 prefill stage에 국한**되며, decode에서는
  `(qa·sa)·(qw·sw)` 항등식으로 int8 dot 대신 float MAC을 돌려 weight-only와 동일한 memory-bound
  특성을 따릅니다.
- 게다가 **activation quantization은 runtime quantization overhead(decode 매 step의 quant_act)를
  수반하므로 그 자체로 decode latency를 증가**시키며, 이 overhead는 memory-bound라 양자화 포맷을
  바꿔도 줄지 않습니다.
- MS 고유의 활성화 양자화(per-block E8M0 scale + sub-byte)는 prefill에서 INT8 tensor-core의 2×
  throughput을 살리려 할 때 **per-block scale이 scale-free main-loop를 막아 per-block store·scale·RMW를
  강제**하므로, MS는 compute-bound prefill에서 INT8 IMMA로 이기지 못하고 dequant→BF16→cuBLAS 경로로
  후퇴해 **BF16과 타이**에 머뭅니다.
- B≥16 weight scope에서 기존 dequant+cuBLAS 경로는 BF16 weight를 쓰고 다시 읽는 round-trip 때문에
  오히려 BF16보다 많이 읽어 패배했으나, **양자화 weight를 직접 읽어 tensor-core fragment로 dequant하는
  fused skinny-GEMM**은 weight read가 memory-bound인 이 구간에서 byte 우위(11MB vs BF16 34MB·MXINT8
  17MB)를 그대로 실현해 BF16·MXINT8를 모두 압승하며, 이 한 번의 전환으로 weight 계열 모든 scope가
  전 batch에서 두 baseline을 이기게 됩니다.
- KV scope에서 attention의 Q는 항상 BF16으로 읽고 decode latency는 KV byte에 의해 결정되므로,
  **attention activation까지 양자화(AA)해도 latency는 변하지 않고 정확도 비용만 추가**되어, W+A+KV
  scope의 decode 특성은 W+KV scope와 동일합니다.

## 4-E. Nibble(u4) vs non-nibble(u2/u3) — 코드 정렬에 따른 이득 변화

**무엇이 nibble 정렬을 가르나**

- MSAQ는 원소당 **upper code (8−u) bit** + 그룹당 **shared (u) bit**를 저장하므로, 원소마다 추출하는
  코드 폭은 **(8−u) 비트**입니다.
- **u=4 → upper 4비트 = nibble**: 32비트 워드에 정확히 8개가 떨어져 어떤 코드도 워드 경계를 넘지
  않습니다(straddle 없음). shared도 4비트라 **딱 2개의 깨끗한 nibble 필드**가 됩니다.
- **u=2 → upper 6비트 / u=3 → upper 5비트**: 4의 배수가 아니라 byte/word 경계를 **가로지릅니다
  (straddle)** → 두 위치의 비트를 결합(rolling buffer/funnel) + 가변 shift 필요.

**unpack 비용 차이 (메커니즘 + 측정)**

- **u=4(nibble)**: 추출이 **단일 `bfe.s32`**(HW bit-field-extract + sign-extend, 1 op)로 끝남.
- **u=2/3(straddle)**: 원소당 rolling-buffer refill(조건부 OR) + mask(LOP) + sign-extend(XOR+IADD) +
  가변 shift + combine = **정수 ALU 약 5~7개**(nibble은 ~3개).
- 측정 (KV decode B=8, Lk=2048, isolated): **u4 = 853M 정수 ALU / 266µs**, **u2 = 1436M / 502µs** →
  straddle만으로 **정수 ALU 1.7×, 시간 1.9×**. DRAM%도 u4 15.4 vs u2 8.5로 하락(unpack이 MLP를 더
  빼앗아 BW 미포화).

**승패 구조 변화 (가장 중요)**

- **u=4(nibble)는 "골짜기"가 사라짐**: unpack이 애초에 싸 MLP를 거의 안 빼앗으므로 byte 절감이 거의
  그대로 시간 전환. KV u4/gs2 실측: B8 **0.80**, B32 **0.55** → **전 배치에서 BF16·MXINT8 모두 승**.
- **u=2/3(straddle)는 "골짜기"가 그대로**: unpack 세금이 무거워 중간 배치에서 MXINT8의 0-unpack 우위를
  못 넘음. KV u2/gs8 실측: B8 **1.33(mx 패)**, B32 **0.92(대배치만 승)**. W-only가 u3(5비트 straddle)라
  batched B=2~15가 골짜기에 빠지는 것도 동일 원인.
- 즉 **nibble 여부가 "이기는 배치 구간의 폭"을 결정**: u4는 batch=1~대배치까지 연속 승, u2/u3는
  batch=1(latency-bound)과 대배치(깊은 BW-bound) 양 끝에서만 승, 가운데가 패배 골짜기.

**nibble 재배치로 u2/u3를 구제 못하는 이유**

- u=4가 빠른 근본 이유는 **정확히 2개의 깨끗한 nibble 필드**를 갖기 때문. u=2/3는 8−u=6/5비트라 nibble로
  나누면 **3번째 필드(low_un)가 불가피**.
- nibble 재배치는 3개 평면(hi4 + low_un + shared)이 되어 stride 로드 증가 + recombine
  (`hi*16 + low_un*2^u + shared` ≈ 5~6 op) 추가 → 기존 streaming straddle보다 **30~37% 더 느림**
  (bit-exact). → **u2/u3엔 streaming straddle unpack이 이미 최적, 어떤 re-layout도 못 넘음.**

**byte·정확도 trade-off (왜 그냥 u4를 안 쓰나)**

- u가 클수록 per-element upper code가 작아져(u4=4비트) **바이트도 적고 unpack도 싸지만** 원소당 정밀도가
  낮음. u가 작을수록(u2=6비트) 정밀도는 높지만 바이트 증가(W에서 u3 0.64× → u2 0.76×) + straddle 비용.
- 따라서 **u 선택은 속도가 아니라 정확도 요구가 결정**하며, 정확도가 u2/u3를 요구하는 scope(weight,
  attention activation frontier가 u2 고정)에서는 cheap-to-unpack한 u4를 못 쓰고 **강제로 straddle
  regime(골짜기)** 에 진입. 이것이 "nibble이면 쉽게 이기는데 정작 정확도가 필요한 곳에선 nibble을 못
  쓴다"는 MS의 구조적 긴장.

## 4-F. 260626 fused 측정 기반 batch별 현상 정밀 해설 (메커니즘 보강)

`blackwell_results_260625_2.md`(B≥16 fused 적용 + MXINT8 공정 twin)를 근거로, 4-A~E의 명제를 **유휴 slot·
unpack 구성·fused 오버헤드·traffic 지표·DRAM 포화 여부**까지 내려 정밀화한다.

### (0) "unpack"이란 — mantissa-shared 포맷의 저장 구조와 복원 비용

- **MXINT8**은 원소를 **8-bit 정수 하나로 그대로** 저장하므로 dequantization이 `int8 load → I2F → ×scale`로
  끝난다(비트 추출 없음).
- 반면 **mantissa-shared(MS) 포맷은 한 원소를 두 조각으로 쪼개** 저장한다 — 원소마다 다른
  **unshared upper code((8−u) bit)** + gs개 원소가 공유하는 **shared code(u bit)** + 32-블록당 E8M0 scale.
  바이트 절감은 바로 이 *shared code를 gs개가 나눠 쓰는 것*에서 온다.
- 이 upper/shared 비트들은 byte 경계에 맞지 않게(u3=upper 5 bit / u2=upper 6 bit) **연속 패킹**돼 있어,
  dequantization 시 **각 원소의 upper bits와 그 그룹의 shared bits를 비트 단위로 추출해 정수 코드로 결합·복원**
  (`code = upper·2^u + shared`, 이어서 ×scale)해야 한다. **이 비트 추출·결합 과정이 "unpack"** 이며, int8을
  그대로 읽는 MXINT8에는 없는 추가 단계다(연산 구성은 아래 §(1)의 ①~⑥). upper code 폭이 4의 배수가 아니면
  (u2/u3) 코드가 32-bit 워드 경계를 가로질러(straddle) 결합이 더 비싸진다(§4-E).

### (0′) Batch에 따른 weight↔KV 이득 교환 — 종합 논리

위 unpack 세금이 batch에 따라 weight·KV 양자화의 이득을 갈라놓는다.
1. Batch가 커지면 같은 weight 한 줄이 **여러 token의 연산에 재사용**되어 byte당 연산(intensity)이 늘고, warp가
   메모리 반환을 순수 대기하는 사이클이 줄어 — 저배치에서 unpack을 공짜로 흡수하던 **유휴 issue slot이 사라진다**.
2. 그러면 **unpack 명령이 메모리 로드 명령 발행(LDG issue)과 issue 대역폭을 두고 경쟁**해, warp가 연속으로 load를
   쏘는 대신 unpack 연산에 머물게 되어 **in-flight memory request 수(MLP)가 떨어진다**.
3. 메모리 시스템은 Little's law(in-flight = 대역폭 × latency)상 충분한 동시 요청이 있어야 대역폭을 채우므로,
   unpack이 없어 load 직후 곧장 다음 load를 쏘는 **MXINT8(높은 MLP → DRAM 포화)** 대비 MS는 **더 낮은 대역폭
   활용**에 머물러 byte 절감의 시간 전환이 줄어든다 → **weight 양자화의 vs-MXINT8 이득이 batch와 함께 tie로 수렴**
   (B=8 골짜기; vs-BF16은 margin이 축소될 뿐 여전히 승).
4. 반면 **KV cache는 batch에 비례해 누적**되므로(§(2) traffic 지표) KV read traffic 절감 효과는 batch와 함께 커져,
   일정 batch를 넘으면 **KV 양자화 이득이 weight 양자화 이득을 능가**한다.

> **완결 주의(단조가 아니라 U자)**: 1~3의 "weight 이득 감소"는 **GEMV regime(B≤~15)의 골짜기까지**다. B≥16에서는
> weight 경로가 fused 텐서코어 GEMM으로 바뀌며 bound가 MLP가 아니라 **load-latency/barrier**로 옮겨가므로(§(1) 끝),
> unpack의 MLP 절도 논리가 더는 지배적이지 않고 **weight 양자화가 다시 이긴다**(vs-BF16 0.93~0.96, vs-MXINT8
> 0.93~0.95). 즉 weight 이득은 단조 감소가 아니라 **B=8 골짜기를 지나 B≥16에서 회복하는 U자**다.

### (1) Weight-only(S1): B=1 또렷 → B=8 동률 → B=16/32 소폭

- **"유휴 issue slot"이란**: SM 스케줄러는 매 사이클 eligible warp 하나의 명령을 발행하는데, **모든 warp가
  stall(메모리 대기 등)이면 그 사이클은 발행할 게 없어 빈다 — 이것이 유휴 issue slot**. B=1 decode는 weight
  재사용이 없어 memory-latency-bound라 warp가 load 반환을 기다리며 자주 stall → 유휴 slot 多 → MSAQ의
  unpack ALU가 *그 빈 사이클에 공짜로* 실행되어 숨는다 → byte 절감만 latency가 되어 BF16 0.55·MXINT8 0.84.
- **유휴 slot이 batch와 함께 주는 이유**: batch가 커지면 같은 weight 한 줄이 **B개 토큰의 MAC에 재사용**되어
  arithmetic intensity가 오르고(load 1회당 연산 B배), warp가 순수 대기하는 사이클이 줄어 유휴 slot이 감소
  → unpack 명령이 숨을 빈자리를 잃고 **메모리 로드 명령 발행(LDG issue)** 과 경쟁하며 MLP를 빼앗는다
  → B=8 골짜기 진입(mq/mx 1.00). (메커니즘 종합은 §(0′).)
- **sub-byte unpack ALU의 구성**(u2/u3 straddle, 원소당): ① rolling-buffer 보충(부족 시 다음 32-bit 워드를
  조건부 OR) → ② mask(`&(2^wbits−1)`, LOP) → ③ sign-extend(XOR+IADD) → ④ 가변 shift로 버퍼 전진(SHF)
  → ⑤ 그룹 경계마다 shared code 보충+mask+sign-extend → ⑥ combine(`up×2^u+sh`, SHL+IADD) = **정수 ~5~7 op**
  (MXINT8은 0; u4 nibble은 단일 `bfe.s32` 1 op).
- **B=16/32에서 "dequant+WMMA+barrier 오버헤드"란**(fused 커널): DRAM에서 양자화 weight를 읽고 K-블록마다
  **① dequant**(shared의 각 코드를 bfe 추출 → E8M0 scale 곱 → bf16 변환 → Bs shared write), **② WMMA**(Bs·As
  16×16 fragment를 shared에서 load → `mma.sync`), **③ barrier**(dequant→Bs write 완료 보장 위해 `__syncthreads`
  + mma 후 1회 = K-블록당 2회)를 반복. 시간 = "11/17MB DRAM 읽기" + dequant ALU + shared write/read + barrier.
- **load-latency-bound / barrier-bound의 의미**: 전자는 warp가 global load의 *반환 지연*(수백 cycle)을 기다리며
  stall — DRAM 버스 대역폭이 아니라 *latency*를 못 가려 SM이 노는 상태(ncu long_scoreboard 59%). 후자는 warp가
  `__syncthreads`에서 *블록 내 가장 느린 warp* 도착을 기다리며 노는 상태(ncu barrier 21%). 즉 한계가 DRAM
  처리량이 아니라 *기다림*이라 byte 절감이 일부만 시간이 됨 → MXINT8(17MB+dequant)은 BF16(34MB·순수
  텐서코어)과 동률(mx/bf ~1.00), MSAQ(11MB)만 11/17=0.65의 일부만큼 소폭(0.93~0.95) 앞섬.

### (2) Weight+KV(S4): KV 효과의 batch 의존 — traffic 지표 + DRAM 포화 검증

- **weight vs KV traffic 지표**(Llama-3.1-8B 32L, Hkv=8, hd=128, transformer GEMM weight=6.98B): decode는 매 step
  weight 전체(batch 공유, **bf16 13.96 GB 고정**) + KV 캐시 전체(sequence당 1회, `B·Lk·32·2·8·128`)를 읽는다.

  | B | KV @Lk=1024(prefill 끝) | KV @Lk=1152(decode 끝) | KV 비중(weight+KV) |
  |--:|--:|--:|--:|
  | 1 | 0.13 GB | 0.15 GB | **1.0 → 1.1 %** |
  | 8 | 1.07 GB | 1.21 GB | 7.1 → 8.0 % |
  | 16 | 2.15 GB | 2.42 GB | 13.3 → 14.8 % |
  | 32 | 4.29 GB | 4.83 GB | **23.5 → 25.7 %** |

  → **B=1엔 KV가 traffic의 ~1%뿐**이라 KV 양자화가 안 보이고(S4 0.56 ≈ S1 0.55), B=32엔 ~26%로 커져 좌우.
- **MSAQ가 MXINT 대비 소폭만 이기는 비용 관점**: S4/S5/S6 KV는 정확도 요구상 **u2/gs8 straddle**(코드 폭 6비트가
  4의 배수가 아니라 byte/word 경계를 가로지름). MXINT8은 int8을 그대로 읽어 `load→I2F→×scale(FFMA)`로 unpack
  이 0인데, **MSAQ-u2는 KV 원소마다 위 ①~⑥의 5~7 정수 op를 추가 지불**(측정 KV B=8 Lk2048: u4 853M ALU/266µs
  vs u2 1436M/502µs = 1.7× ALU·1.9× 시간). 이 세금이 byte 우위(0.81 vs 1.03 = 0.79×)를 갉고, 근본적으로 MSAQ의
  무기가 MXINT8 대비 byte 절감(~0.66×)뿐이라 **우위 천장 자체가 byte-ratio 수준** → mq/mx 0.92~0.94.
- **B=16까진 잠잠 → B=32 두드러짐, 그리고 DRAM은 포화되지 않는다**: W+KV total은 *깊어지는 KV 우위* + *줄어드는
  fused weight 우위(0.88→0.93→0.95)*의 합이라 B=8→16 평평(0.68→0.69), B=32에서 KV(traffic 26%)가 압도해
  두드러짐(0.62). **그러나 "깊은 bandwidth-bound"를 DRAM 포화로 읽으면 틀림**: 측정 peak DRAM BW = **~553(copy)/
  616(read) GB/s**인데 B=32 MSAQ 풀양자화 decode(S5)는 step당 47ms·traffic 7.63 GB → **유효 162 GB/s = peak의
  ~27%**로 **DRAM 미포화**. 즉 B=32 이득은 raw DRAM 포화가 아니라 **KV-read 커널이 shared/L2 처리량(ncu L1 81%·
  DRAM 20%)에 묶여 있고 그 한계가 옮긴 byte에 비례**하기 때문(traffic-proportional)이며, batch가 크면 블록이
  많아져 SM 점유율이 차는 것도 더해진다. **→ 4-B/4-C의 "bandwidth-bound"는 'DRAM 포화'가 아니라 'byte-비례'로
  읽어야 정확하다**(측정 정정).

### (3) AA·A가 prefill·decode 어느 쪽도 못 빠르게 하는 이유

- **prefill에서 MS×MS GEMM인데도 안 빨라짐**: prefill GEMM은 intensity≈M(1024)≫ridge(76)라 compute-bound →
  빨라지려면 INT8 텐서코어 2× throughput이 필요. 그러나 **MS 코드는 GEMM 전 반드시 INT8 워드로 복원(unpack)**
  되어야 하고(추가 ALU), 결정적으로 **per-32-block E8M0 scale이 scale-free INT8 main-loop를 막아** 블록마다
  `IMMA→int32 store→타일 scale→float RMW`를 강제해 IMMA 이득을 상쇄(실측 bf16-fused 대비 2.05× regression).
  → **활성화·가중치를 둘 다 양자화해도 GEMM 처리시간 이득 0**, MS는 `dequant→bf16→cuBLAS`로 후퇴해 BF16 타이;
  A 양자화는 **MXINT8보다 복잡한(sub-byte+per-block-share) quantize 오버헤드만 추가**.
- **decode 어텐션을 AA로 저비트끼리 돌려도 안 빨라짐**: decode 어텐션은 토큰 1개의 Q(극소)와 *누적 KV 캐시 전체*
  를 곱하므로 KV byte에 memory-bound. ① Q·K/P·V FLOP 자체가 (쿼리 1토큰이라) 미미해 연산 가속이 latency를 못
  줄이고, ② 병목인 KV read+unpack을 AA가 안 바꾸며, ③ production 커널은 **Q를 어차피 bf16으로 읽어**(traffic
  기여 무시) 양자화해도 byte가 안 줄음 → **AA = 정확도 비용(~+0.9~1.0pp PPL)일 뿐 latency ≈0 → S6≈S5**.
- **(260626 측정 정밀화) "latency 0"은 AA를 accuracy-only로 둘 때만**: 배포 기본 decode 커널은 Q를 bf16으로 읽어
  AA를 *계산하지 않으므로*(harness에서 S5≡S6 동일 튜플) 그 가정 하에서만 S6=S5다. 실제로 attention을 low-bit
  활성화로 돌리려면 Q·K, P·V의 Q와 P를 양자화해야 하는데, 이를 `kv_decode_wide_kernel`에 **`MS_KV_AA=1` 경로로
  구현**했다(Q는 prologue, P는 매 chunk; 둘 다 (u,gs) MSAQ로 fake-quant = quantize→dequantize, 워프 병렬로
  `__shfl` amax·group-sum, transcendental 없는 IEEE-지수 비트추출). **측정(u2/gs8, B=32, Lk=1024): AA-off와
  byte-identical, AA 출력 rel 0.011(=Q/P u2 양자화 오차), KV-read +4.5% → decode 전체 ~+1.7%.** 초기 직렬 버전은
  +16%였으나 P-quant를 4-thread 직렬→워프 병렬로 바꿔 4.5%로 절감. **즉 faithful AA decode는 "latency 0"이 아니라
  ~1.7%이며, 그 비용은 Q/P를 실제 저비트로 돌리는 값이지 KV 양자화와는 무관**(prefill AA가 bf16 SDPA에 2.1~2.7× 지는
  것과는 차원이 다름). 배포는 여전히 accuracy-only(기본 off)가 최적.
- **W+KV→W+A+KV+AA가 동률~미세 악화인 이유**: AA는 (accuracy-only 기본값에서) latency 0, +A만 매 step `quant_act`
  (~13µs/step·memory-bound, BF16 baseline엔 없는 단계, 포맷 무관)를 더해 S6가 W+KV 대비 동률이거나 그 분
  (B=8 decode +54µs)만큼 BF16 비율이 미세하게 올라(악화) 보임.
