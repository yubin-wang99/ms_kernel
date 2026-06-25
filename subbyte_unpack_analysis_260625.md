# 서브바이트 unpack 비용 분석 — 왜, 얼마나 느린가 (260625)

MSAQ가 MXINT8 대비 느려지는 근원인 **서브바이트 언팩(sub-byte unpack)** 의 비용을 op·latency·
메모리 접근 패턴 관점에서 측정·분석한다. GPU = NVIDIA RTX PRO 4000 Blackwell (sm_120, 70 SM),
CUDA 13.2. 측정: ncu (idle GPU), per-scope robust (u,gs).

## 0. 한 줄 결론

> 서브바이트 unpack의 비용은 **메모리가 아니라 정수 ALU 파이프**(shift/mask/sign-extend; MXINT8의
> **3.7×**, +2M 명령어/커널)다. **로드는 동일하게 coalesce**되지만 로드 *후* 비트 추출이 ALU를 채워
> **bound를 memory-latency에서 issue/ALU로 이동**시킨다. 그래서 **명령어는 1.4~3.7× 늘지만 wall-clock
> 은 1.0~1.21× 느림**(메모리 latency 뒤 부분 은닉). 5/6비트 straddle은 nibble(u4)보다 ALU를 1.7× 더
> 써 이 비용을 키운다. 양자화 바이트 우위(0.66×)는 BW-bound 구간에서만 이 ALU 세금을 이긴다.

---

## 1. 근본 비대칭 — 원소 1개 복원에 드는 일

**MXINT8** (int8 직접 저장):
```
load int8  →  I2F  →  (×scale, FFMA에 fold)            # ALU ~0, FMA 1~2
```
바이트가 곧 값. 언팩이 **없다**.

**MSAQ** (per-element upper (8−u)b + per-group shared u b 공유 저장):
```
extract upper((8−u)b) : rolling-buffer refill(조건부 OR) + mask(LOP) + sign-extend(XOR+IADD)
extract shared(u b)   : 그룹당 1회(amortize) + mask + sign-extend
combine               : up<<u + sh                       (SHF + IADD)
to float              : I2F → ×scale
```
원소당 **정수 ALU ~5–7개** (MXINT8는 ~0). 이게 비대칭의 전부다. (`csrc/core/ms_utils.cuh`
`stream_block_uspec`.)

## 2. Op-level 비용 (ncu, W-only decode B=4, 파이프별 warp-instruction)

| pipe | MSAQ | MXINT8 | 배수 |
|---|--:|--:|--:|
| **ALU** (shift/logic/iadd) | 2,792,576 | 756,480 | **3.7×** |
| FMA (FFMA + I2F) | 3,509,120 | 2,870,144 | 1.22× |
| LSU (글로벌 로드) | 694,144 | 644,992 | 1.08× |
| CBU (제어) | 144,512 | 144,512 | 1.00× |
| **합계** | **7,625,088** | **5,408,768** | **1.41×** |

- 추가 명령어의 **92%가 ALU 파이프** (+2.04M) = shift/mask/sign-extend/rolling-buffer 부기.
- FMA가 약간 높음(+0.64M) = 원소당 I2F + scale (MXINT8는 I2F만).
- **LSU(로드)는 사실상 동일** → 메모리 접근 패턴은 병목이 **아니다**.
- thread-op 레벨: 정수 연산 **93.7M vs 26.1M (3.6×, +67.6M)**, FP는 **84.7M로 완전 동일**(실제 MAC은 같음).
  즉 MSAQ는 matmul(FP)에 더해 **그만큼의 정수 ALU를 추가 발행**한다(언팩 > matmul).

## 3. straddle — 5/6비트가 4비트보다 더 느린 이유

코드 폭 (8−u)이 8·32의 약수가 아니면 **byte/word 경계를 가로지른다(straddle)** → 두 위치의 비트를
결합(rolling buffer/funnel) + 가변 shift가 필요. 4비트(u4)는 nibble 정렬이라 straddle이 없다.

ncu (KV decode, B=8, Lk=2048, isolated):

| | DRAM% | SM% | 정수 ALU(thread-op) | 시간 |
|---|--:|--:|--:|--:|
| **u4** (4b nibble, straddle 없음, 단일 `bfe.s32`) | 15.4 | 62.5 | **853M** | **266µs** |
| **u2** (6b, straddle) | 8.5 | 47.3 | **1436M** | **502µs (1.9×)** |

→ straddle만으로 **정수 ALU 1.7×, 시간 1.9×**. nibble(u4)은 단일 `bfe.s32`(HW sign-extend, 1 op)
로 절반. **이게 u4 KV는 MXINT8를 이기고 u2/u3는 지는 핵심**이다. (u2/u3를 nibble 재배치해도 3번째
필드(low_un)가 불가피해 현재 streaming보다 30~37% 느림 — `naive_ms_260625.md`/메모리 참조.)

## 4. 메모리 접근 패턴 — 병목이 아님

- 평면은 column-major dense. 스레드가 자기 열의 UB바이트를 **4정렬 uint32 wide-load**로 읽음 →
  **완전 coalesce** (LSU가 MXINT8와 동일한 근거).
- 비용은 로드 **후** 레지스터에서의 **비트 추출**. straddle은 rolling-buffer가 `ubuf`/`unb` 상태를
  원소 간 **직렬 전파**(각 추출이 이전 shift에 의존) → 스레드 내 ILP 제한.
- **단, 직렬성 자체는 SASS에서 묶이는 제약이 아님**: funnel-shift 독립추출로 바꿔도 no-op(컴파일러가
  완전-unroll된 rolling-buffer를 이미 동등 코드로 최적화). 진짜 제약은 **명령어 총량(ALU 발행)**이다.

## 5. Latency — 왜 부분적으로만 느려지나 (bound 이동)

decode는 이상적으로 memory-latency-bound라 ALU가 **메모리 대기 중 빈 issue 슬롯에 숨어야** 한다.

| ncu (W-only B=4) | MSAQ | MXINT8 |
|---|--:|--:|
| DRAM throughput % | 73.7 | **82.3** |
| SM/issue % | **60.7** | 30.0 |
| long-scoreboard stall (글로벌 메모리 대기) % | 46.4 | **72.2** |
| not_selected (발행 경합: 준비된 워프 多) % | **12.9** | 2.3 |
| wait (고정-지연 의존) % | 11.3 | 5.0 |

- **MXINT8 = 순수 memory-latency-bound** (long-SB 72%, SM 30%, 연산 파이프 한가). 바이트가 곧 시간.
- **MSAQ = 언팩 ALU가 빈 슬롯을 채워** long-SB 46%로↓, SM 60%·not_selected 13%↑ → **bound가
  메모리에서 issue/ALU 파이프로 이동**. (B=1도 같은 방향: DRAM 76 vs 81, long-SB 62 vs 54.)

즉 **언팩의 추가 명령어는 "공짜 유휴 사이클"을 잡아먹을 만큼 많아져야 비로소 병목**이 된다. B=1(유휴
많음)은 거의 숨고, B≥2(메모리 amortize → 유휴 감소)부터 ALU가 노출된다.

## 6. 얼마나 느려지나 (regime별, 측정)

| 척도 | 배수 |
|---|---|
| 발행 명령어 (warp-level) | **1.41×** |
| 정수 ALU 명령어 | **3.7×** |
| wall-clock — W decode (vs 배포 MXINT8 `_wide`) | B=1 **1.02**(타이) · B=2~12 **1.07~1.21** |
| wall-clock — KV K-dot (isolated, wide-load MXINT8) | **1.06~1.14** |

핵심: **wall-clock 패널티(1.0~1.21×) ≪ 명령어 패널티(1.4~3.7×)** — 추가 ALU가 메모리 latency 뒤에
**부분 은닉**되기 때문. 완전 issue-bound면 ~1.4×까지 가지만 부분 은닉으로 ~1.1×에 머문다.

> 주의(공정성): isolated K-dot의 MXINT8 baseline은 반드시 **wide-load(int4×2)** 로 측정해야 한다.
> scalar 바이트 로드로 재면 MXINT8가 인위적으로 2.5× 느려져 MSAQ가 거짓 승리(0.37)한다.

## 7. 그럼에도 이기는 구간 — 바이트 우위가 전환될 때

MSAQ는 바이트를 **0.656×(W u3) / 0.781×(W+A u2)** 적게 읽는다. 이 우위는 **DRAM-BW-bound일 때만**
시간이 된다:
- **B=1 decode**: memory-bound → vs MXINT8 0.84까지 승(단 언팩이 BW%를 76%로 눌러 0.66× 완전전환은 막음).
- **대형 배치·긴 문맥 KV**: 바이트 절감 누적 + BW-bound → vs bf16 0.42~0.61, vs MXINT8 ~0.91.
- 반대로 **unpack-bound(B≥2 batched, isolated K-dot)** 에선 바이트 우위 미전환 → MXINT8의 0-언팩 승.

---

## 부록 — 정량 요약 (한눈)

```
원소당 정수 ALU   : MXINT8 ~0   |  MSAQ ~5-7  (straddle u2/u3)  |  MSAQ ~3 (nibble u4)
커널 ALU 명령어    : MXINT8 0.76M | MSAQ 2.79M  (3.7×)            [W-only B=4]
커널 LSU(로드)     : MXINT8 0.64M | MSAQ 0.69M  (1.08× = 동일)
정수 thread-op     : MXINT8 26.1M | MSAQ 93.7M  (3.6×, +67.6M)
FP thread-op       : 84.7M (양쪽 동일 — 실제 MAC은 같음)
bound              : MXINT8 memory-latency(long-SB 72%) → MSAQ issue/ALU(SM 60%)
시간 패널티        : 명령어 1.4~3.7×  →  wall-clock 1.0~1.21× (latency 뒤 부분 은닉)
straddle 세금       : u2(1436M ALU/502µs) vs u4(853M/266µs) = 1.7× ALU / 1.9× 시간
```

출처: ncu(`scratchpad/ncu_driver.py`, `kv_ncu.py`), fair bench, 메모리
`msaq-vs-mxint8-{w,kv}-decode-state`. 관련 문서: `fair_occupancy_260625.md`, `naive_ms_260625.md`.

---

## 8. memory-floor 관점 — MS의 미실현 잠재력

"MS가 진짜 BW-bound가 되면 바이트로 이긴다"를 정량화한다. **memory floor** = (읽는 바이트)/(peak BW)
= 완전 BW-bound일 때의 최소 시간. MS는 MXINT8의 **0.656×** 바이트.

W-only decode **B=8** (mult=16):
| | 측정 시간 | DRAM% | memory floor(추정) | floor 대비 |
|---|--:|--:|--:|--:|
| MXINT8 | 28.4µs | **82%** | ~23µs | **floor의 81% (거의 BW-bound)** |
| MS | 30.7µs | 74% | ~15µs (0.656×) | **floor의 ~50% (잠재력 미실현)** |

- MXINT8은 이미 자기 floor의 81%에서 돈다 → 더 빨라질 여지 작음.
- **MS는 자기 floor(~15µs)의 절반에서 돈다** → 바이트 우위가 통째로 미실현. MS가 floor에 닿으면
  vs MXINT8 = **0.54~0.66× (압승)**. 지금 1.08(패)인 이유는 BW-bound에 못 갔기 때문일 뿐.
- 이미 BW-bound인 구간(B=1, 대형 배치/긴 문맥 KV)에선 이 잠재력이 실현돼 MS가 이긴다 — 가설이
  옳다는 직접 증거.

## 9. occupancy(splitK) 레버는 소진됨 — "더 읽기/더 깔기"로는 BW-bound 못 감

MS_GEMV_SPLITK_MULT 스윕 (W-only B=8, 블록 수 = occupancy 레버):
| mult | ms µs | mxint8 µs | ms/mx |
|--:|--:|--:|--:|
| 8 | 31.1 | 28.7 | 1.084 |
| **16** | **30.7** | 28.4 | 1.081 |
| 24 | 32.8 | 30.7 | 1.067 |
| 32 | 33.2 | 30.7 | 1.081 |
| 48 | 36.9 | 34.8 | 1.059 |

- 블록을 더 깔아도(16→48) **MS는 안 빨라지고**(reduction 오버헤드로 오히려 ↑) 비율도 1.08 고정.
- ncu: occupancy(warps_active) MS 67% ≈ MXINT8 68% (동일), **math-pipe throttle 3%**(ALU 처리량 포화
  아님). 즉 MS가 floor에 못 가는 건 **occupancy 부족도 ALU 처리량 한계도 아님**.
- 진짜 제약 = **per-thread 언팩이 LDG→unpack→FFMA 의존사슬의 critical path에 얹혀** latency가 안
  숨겨짐. 블록을 더 깔아도 각 스레드가 똑같이 언팩하니 안 풀린다. → **"많이 읽어서 BW-bound로"는
  불가; 유일한 길은 언팩 자체를 싸게 만드는 것.**

## 10. 왜 "언팩 줄이기"가 DRAM 이용률을 끌어올리나 (메커니즘 — MLP)

> **정정**: 처음엔 "발행 슬롯 경합"으로 설명했으나, ncu로 보니 issue_active가 MSAQ 67%(100% 아님)·
> 레지스터/occupancy도 MXINT8과 동일 → **발행 슬롯 포화도 레지스터 압박도 아니다.** 진짜 매개는 **MLP
> (memory-level parallelism, 동시 in-flight 로드 수)** 다.

ncu (W-only B=4) 결정 지표 — **메모리(long-SB) 대기 워프 수 / issue**:
| | MSAQ | MXINT8 |
|---|--:|--:|
| 메모리 대기 워프/issue | **5.6** | **18.0** |
| issue_active | 67% | 33% |
| registers/thread | 46 | 48 |
| DRAM% | 74 | 82 |

peak BW를 내려면 메모리에 동시 in-flight 로드가 충분해야 한다(Little's law: in-flight = peak_BW×latency).
- **MXINT8 워프**: `LDG → 즉시 메모리 대기`(언팩 없음, 할 일 없음) → 항상 ~18워프 분량 로드가 떠 있음
  (높은 MLP → DRAM 82%).
- **MSAQ 워프**: `LDG → unpack 연산(바쁨) → FFMA → 다음 LDG`. 언팩하는 동안 그 워프의 로드는 이미
  끝나 **대기 상태가 아님** → 어느 순간이든 메모리 대기 워프가 5.6뿐 (낮은 MLP → DRAM 74%).

즉 언팩은 **"메모리 대기 사이클"을 "연산 사이클"로 바꿔치기**하며(그래서 long-SB가 70→46%로 내려감)
**동시에 메모리 동시성을 빼앗아** BW를 떨어뜨린다. → **언팩을 줄이면** 워프가 연산을 빨리 끝내고 다시
메모리 대기로 돌아가 **로드를 더 많이 띄움** → MLP↑(5.6→18 방향) → DRAM%↑ → memory floor 도달.

> 직관: 메모리 트럭(BW)이 74%만 차는 건, 인부(워프)가 서류작업(언팩)하느라 트럭 옆에서 짐(로드)을 덜
> 쌓아둬서다. MXINT8은 서류가 없어 항상 짐을 잔뜩 쌓아둠(82%). 서류를 줄이면 짐이 더 쌓이고(MLP↑),
> MS는 트럭에 실을 짐(바이트)이 적으니 더 빨리 출발한다. **occupancy(splitK)로는 안 되고 언팩을 줄여야
> 되는 이유가 같은 동전의 양면** — 제한된 워프 수 하에서 per-warp 언팩이 MLP를 깎는다. 매개가 ALU
> 처리량(math-pipe 3%)이 아니라 **MLP**라서, 언팩 명령 *총량*을 줄이는 게 본질이다(funnel 재정렬 같은
> 스케줄 변경은 no-op — §4).

## 11. register-aligned + bfe 실측 — 메커니즘 검증 (구현·측정)

언팩 명령 총량을 실제로 줄이는 유일한 미시도 경로: **각 (8-u)비트 코드를 32비트 워드 안에 통째로
패딩**(CPW=⌊32/(8-u)⌋ 코드/워드)하여 코드당 **단일 `bfe.s32`**(HW sign-extend, no rolling-buffer/mask/
sign-extend)로 추출. dense·straddle을 제거. 구현: `pack_weight_ra`/`dequant_weight_ra`(pack.py),
`stream_block_ra`+`wonly_gemv_batched_ra`(csrc), op `wonly_gemv_batched_ra`. 비트정확.

**측정 (W-only u3/gs16, ra/ms<1 = ra 빠름):**
| B | mxint8 | ms | ra | ra/ms | ra/mx | (ms/mx) |
|--:|--:|--:|--:|--:|--:|--:|
| 2 | 16.4 | 18.5 | 17.4 | **0.94** | 1.06 | 1.12 |
| 4 | 18.6 | 22.5 | 20.5 | **0.91** | 1.10 | 1.21 |
| 8 | 28.6 | 30.7 | 29.7 | 0.97 | 1.04 | 1.07 |

**ncu가 §10 메커니즘을 정확히 확정 (u3 B=4, ms→ra):**
| | ms | ra | 변화 |
|---|--:|--:|--:|
| ALU 파이프 | 2.79M | 2.28M | **−18%** |
| LSU(로드) | 0.69M | 0.73M | +5% (바이트↑) |
| **DRAM%** | 73.7 | 77.2 | **+3.5pp** |
| **MLP (메모리대기/issue)** | 5.6 | 8.53 | **+52%** |

→ **메커니즘 검증 완료**: 언팩 ALU −18% → MLP +52% → DRAM +3.5pp → 시간 −9%(ra/ms 0.91).
**register-aligned bfe는 funnel/nibble/unsigned(전부 동률)와 달리 MS를 실제로 빠르게 만든 첫 경로.**

**그러나 MXINT8은 못 이김 (ra/mx 1.04~1.15, ms의 1.07~1.21보다는 개선):** 두 한계 때문.
1. **ALU 감소가 18%뿐** — 추출은 bfe로 줄었지만 combine(`up<<u + sh`)·scale·주소계산 잔여 ALU가 남고,
   rolling-buffer는 컴파일러가 이미 최적화하고 있어 기대한 −75%가 안 나옴.
2. **정렬 패딩 +20% 바이트** (u3 0.64→0.76, u2 0.76→0.88) — ALU 이득을 memory floor 상승으로 상쇄.

즉 §8의 "memory floor 도달하면 0.66× 승"은 옳지만, register-aligned는 floor 자체를 0.76×로 올려버려
(패딩) 순이득이 줄어든다. **MS를 빠르게 한 첫 확증된 레버이자, 그 천장도 함께 드러낸 결과**다.

### 11b. ALU 더 줄이기 — 두 후속 시도 (combine 제거 / 바이트 안 늘림)

1. **sepsc로 per-element combine(`up<<u + sh`) 제거 → REGRESSION** (`wonly_gemv_batched_ra_sepsc`).
   scale·2^u를 upper FFMA에 fold하고 shared를 그룹합 qg로 분리해 combine을 없앴으나 **ra보다 2~12%
   느림** (sep/ra 1.02~1.12). 이유: combine은 컴파일러가 이미 ~1 fused op으로 처리해 절감분이 작은데,
   qg 사전계산(블록당 MR×NG 합) + **추가 __syncthreads**가 더 비쌈. → combine은 뗄 가치 없음.

2. **dense-selective-bfe로 패딩 바이트 제거 → ≈ register-aligned** (`wonly_gemv_batched_densebfe`,
   production dense 평면 그대로, repack 불필요). dense 0.64× 바이트를 유지하되 32비트 워드 경계를
   안 넘는 코드(대부분)는 단일 bfe, straddle(블록당 ~4개)만 funnel+bfe. 결과 **db/ms 0.92~0.96,
   db/ra 0.98~1.05** = ra와 동급. 바이트 이득(0.76→0.64)과 funnel 비용(4코드×2op)이 상쇄.
   **실용적으로는 dense-bfe가 ra보다 나음**: 같은 속도인데 바이트 0.64× 유지(정확도-당-바이트↑) +
   repack·재인증 불필요(production 평면 사용). 비트정확(production GEMV와 diff 0.0).

> **종합 천장**: register-aligned / dense-bfe가 unpack-ALU 감소의 바닥(ms 대비 0.91~0.97). combine은
> 컴파일러가 이미 fuse, scale은 MXINT8와 동일(격차 아님), 추출은 bfe 1개가 irreducible. 두 후속 시도
> (combine 제거 regression, 바이트 제거 tie)로 **ALU 추가 감소 경로는 소진**. MS는 MXINT8 대비
> 1.04~1.17(ms의 1.07~1.21보다 개선)에서 멈춤 — 서브바이트 추출 bfe 1개 + 잔여가 MXINT8의 0-언팩을
> 못 넘는다. 더 큰 이득은 unpack 미시최적화가 아니라 **memory-bound regime 자체(B=1, 대형/긴문맥 KV)**
> 에서만 (§7, §8) 나온다.

## 12. B≥16 weight — fused quantized tensor-core GEMM의 기회 (Marlin / FP6-LLM)

배포는 B≥16에서 `dequant→bf16 + cuBLAS`를 쓴다. 이게 **bf16에 2~3.5× 지는(mq/bf E2E 1.28) 진짜 이유**를
측정으로 분해하면 — **구현 artifact**다:

| weight 평면 | bf16 | MSAQ-quant | MXINT8 |
|---|--:|--:|--:|
| 바이트 (4096²) | **34MB** | **11MB** | 17MB |

| M (decode) | bf16-cuBLAS | dequant 단독 | deq+cuBLAS | GEMV(cuda) |
|--:|--:|--:|--:|--:|
| 16 | 22.9µs | **37µs** | 81µs | 49µs (deq+cB 이김) |
| 32 | 24.4 | 37 | 64 | 98 (compute-bound 폭발) |
| 64 | 30.8 | 37 | 81 | 238 |

- **dequant 단독(37µs)이 bf16 GEMM 전체(23µs)보다 비싸다** — bf16 [K,OUT] 34MB를 쓰고, cuBLAS가
  그 34MB를 다시 읽는 **66MB 왕복**이 핵심. bf16(34MB 1회 읽기)보다 *더* 읽어서 진다.
- M=16~32 decode GEMM은 intensity≈M ≪ ridge → **weight-read에 memory-bound**. 그러니 **11MB만 읽는
  fused 커널이면 11/34 = 0.32× → ~8~10µs로 bf16 압승** 가능(MXINT8 17MB=0.5×도 이김).
- 그런데 **GEMV(cuda-core)는 M>16에서 compute-bound로 폭발**(98→238µs), **deq+cuBLAS는 왕복**,
  **기존 fused WMMA(`wonly_gemm_tc`=`wmma_pipe`)는 324µs** — 그리드가 (OUT/64, M/64)라 M=16을 64로
  패딩(WMMA 75% 낭비) + 64블록 under-occupancy. 셋 다 11MB-read 잠재력에 못 간다.

**→ 유일한 경로 = skinny-M fused quantized tensor-core GEMM**: 양자화 weight를 cp.async로 shared에
싣고, register-aligned bfe(§11)로 bf16 fragment에 직접 dequant, mma.sync로 matmul, double-buffer로
다음 타일 load+dequant를 현재 mma와 겹침. **이게 정확히 Marlin·FP6-LLM이 푸는 문제**다:
- **Marlin (arXiv:2408.11743)**: dequant(SIMT)·MMA(tensor) 명령을 세밀히 인터리브해 두 파이프 상호
  stall 방지, weight를 offline reshuffle해 tensor-core fragment 레이아웃으로 곧장 dequant, column-wise
  accumulation으로 다음 operand dequant를 현재 MMA와 pipeline. **타깃 M이 바로 16~64(batched decode)**.
- **FP6-LLM / TC-FPx (arXiv:2401.14112)**: SIMT(dequant)+tensor(matmul)+cp.async 동시 실행 SW pipeline.
  **비-2^n bitwidth(6/5/3-bit)를 다루는 유일 통합 커널** → MSAQ의 5/6-bit straddle과 가장 가까움.

**적용 범위 주의**: 이 기법들은 **tensor-core가 일하는 regime(B≥16)** 전용. MSAQ가 이미 이기는
**decode B=1~8은 memory-bound라 tensor core가 idle**(§7) → 적용 안 됨(코드베이스도 fused-WMMA decode는
documented-negative). 즉 Marlin/FP6는 **decode를 더 빠르게 하는 게 아니라 B≥16의 진 싸움을 뒤집는** 도구.

**기대 상한**: M=16~32에서 11MB-read fused가 memory-bound로 ~0.32×bf16(cuBLAS 23µs → ~8µs)에 근접하면,
B≥16 weight scope가 **mq/bf 1.28(패) → ~0.4(압승)** 으로 뒤집힌다. §8 byte-floor 논리가 여기서 가장 크게
실현 — B≥16은 weight read가 전부라 0.32× 바이트가 그대로 시간이 된다.

**구현 난이도(정직)**: naive fused가 14× 느린 데서 보듯, 경쟁력(cuBLAS급) 도달엔 Marlin의 풀 머신러리
(cp.async 다중 버퍼, bank-conflict-free swizzle, mma.sync fragment-direct dequant, offline weight
reshuffle)가 필요한 **별도 대형 커널 프로젝트**다. register-aligned bfe(§11)가 그 dequant 단계의 재료가
된다. 본 절은 **그 기회·설계·기대치를 측정으로 확정**한 것이며, 구현은 다음 집중 과제로 권장한다.
