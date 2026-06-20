# Kernel ver.3 — 7종 커널 + KV-read 텐서코어 심층 조사 (Phase 34–39)

ver.2 대비 변경은 **KV cache dequant(§2.4) 하나에 집중**된 심층 조사다. 나머지 6개 커널
(GEMM/GEMV/KV-write/append)의 설계·결과는 ver.2와 동일하므로 여기선 요약만 싣고, **KV-read의
"공정·정확 win이 가능한가"를 7개 lever로 끝까지 추적한 결론**을 기록한다. 전체 1차 결과는
[kernel_ver2.md], 공정성 감사는 [for_fair_comparison.md], 단계별 로그는 [change.md] Phase 32–39.

---

## 0. 배경 (ver.2와 동일, 요약)
MSAQ-s: 32원소 블록당 E8M0 scale + 원소당 upper(8−u)bit + gs그룹당 shared u-bit. u4·gs8 ≈ 19B/블록
= MXINT8 33B의 **0.58×**. 이득의 본질은 **대역폭**이라 memory-bound 연산에서 이기고 compute-bound에선
unpack만 critical path에 더해진다. MXINT8 짝은 element 언팩만 다르고 골격(레이아웃·split·타일·매핑)은 동일.

---

## 1. 6개 커널 결과 (ver.2와 동일)

| 단계 | 커널 | 특성 | MSAQ/MXINT8 (u4) |
|------|------|------|------------------|
| Prefill | W-only GEMM | compute-bound | 0.95 |
| Prefill | W+A GEMM (INT8 IMMA) | compute | **0.79** |
| Prefill | KV write | memory-bound | **0.85** |
| Decode | W-only GEMV | memory-bound | **0.63** |
| Decode | W+A GEMV | memory-bound | **0.82** |
| Decode | KV append | launch-bound | ~1.0 (fuse 대상) |

**MSAQ의 실제 end-to-end 가치는 weight 경로(W-only GEMV/GEMM)에서 온다.** S1 0.79, S4 0.43(vs bf16).

---

## 2.4 KV cache dequant (decode attention) — 7 lever 조사 후 **공정 win 불가** 확정

**연산 구조.** decode attention = Q·K^T(키별 score, **D에 대한 reduction**) → softmax →
P·V(**키들에 대한 reduction**). 핵심 비대칭: reduction 축이 element 내부냐 키를 가로지르냐에 따라
sub-byte의 sector 효율이 갈린다.

**ver.2까지의 결론(Phase 32):** MXINT8을 thread-per-key로 공정화하니 스칼라 커널에서 **u4 KV-read는
tie**(이전 0.54 "압승"은 MXINT8 under-optimization 산물). 스칼라 커널은 latency-bound(~66→실측 후
재측정시 MXINT8 480 GB/s까지, MSAQ ~220)라 0.58× 바이트가 시간으로 안 환원.

**Phase 34–39: win을 만들려는 7개 lever와 결과**

| lever | 결과 | 이유 |
|-------|------|------|
| split-K mult↑ (occupancy) | **악화** | per-block 일 적은 MXINT8만 이득, combine 오버헤드가 MSAQ에 불리 |
| warp-transpose P·V (staging 제거) | **악화(~5–10%)** | shfl issue↑, occupancy 불변(one-wave) |
| batch (점유율 공짜 확보) | **악화(1.07–1.22)** | 머신 채워 BW-bound 도달하나 **MSAQ 실효 BW가 MXINT8의 ~0.5×**(dequant throttle) |
| channel-major V (KIVI식) | 속도는 win이나 **기각** | dense block이 token축 grouping 강제 → 정확도 ×1.3–1.8 악화(KIVI 역행) |
| 텐서코어 P·V (split-K WMMA) | **악화** | MMA가 가속하는 reduction은 병목이 아님; bf16-staging이 병목이고 두 포맷 동일 타일 → 0.58× 무관 |
| **+coalesced load +파이프라인** | **P·V 단독 WIN(0.84–0.94, M≥32)** | scatter→full-sector로 MSAQ 실효 BW 0.5→0.6×, 0.58× 바이트와 합쳐 win |
| 완전 2-pass attention (shared-prefix) | **tie~loss(best ~1.0)** | Q·K(D-contraction 짧음)가 unpack-bound loss → P·V win을 희석 |

**근본벽(확정).** binding constraint는 점유율·reduction·layout이 아니라 **"MSAQ를 텐서코어/누적기가
소비 가능한 형태로 dequant하는 throughput"**(MXINT8 대비 ~0.5× 실효 BW). 텐서코어는 bf16 타일을
강제 → 두 포맷이 같은 타일을 만들어 0.58× DRAM이 무관해진다. GEMV가 이기는 건 staging 없이
wide-load→직접 누적이라 DRAM-bound가 되기 때문(텐서코어는 그 메커니즘을 잃음).

**유일한 부분 win = shared-prefix(prefix caching / beam)에서 P·V만.** 독립 배치 decode는 V가 배치마다
달라 M=G(2~4)로 작아 win 영역(M≥32) 미달 → 일반 decode는 여전히 tie/loss. shared-prefix에서 P·V는
0.84–0.94지만, **완전 attention은 Q·K가 희석해 ~tie**(M=128 기준 Llama 1.00·Gemma 0.99·Mistral 1.00).

**정확도·공정성은 전 구간 보존:** V를 token-major(per-token group, KIVI 정렬)로 두고 d-major bf16
타일로 on-chip transpose → rel_fro 2.4e-3(= u4 양자화 오차), 양쪽 동일 WMMA·unpack만 차이.

---

## 3. 종합 결론

- **memory-bound + element-내부 reduction(W-only GEMV/GEMM, KV write)에서 MSAQ가 명확히 win**
  (u4 0.63/0.85). end-to-end 가치의 본체.
- **KV-read(P·V)는 키-가로 reduction + sub-byte라 구조적으로 막힘.** 스칼라/staging/transpose/batch/
  channel-major/텐서코어/2-pass 전부 시도 → **공정·정확 win은 완전 attention 레벨에선 shared-prefix
  대형 M에서도 ~tie가 한계**. P·V만 떼면 win이나 Q·K가 상쇄, 일반 decode는 tie/loss.
- **남은 가능성(미시도, modest·niche):** Q·K를 bf16-staging 없는 scalar/wide(=tie)로 바꿔 shared-
  prefix 완전 attention을 ~0.93까지. 또는 native sub-byte MMA(하드웨어 미지원).

## 4. end-to-end (ver.2와 동일, batch=1 단일 스트림)
| 시나리오 | MSAQ-u4 /mxint8 | /bf16 |
|---------|------|------|
| S1 W-only | **0.79** | 0.79 |
| S2 W+A | 0.90 | 0.80 |
| S3 KV-only | 1.01 (tie) | 0.64 |
| S4 W-only+KV | **0.68** | **0.44** |

KV-read가 tie여도 **S4가 0.68인 건 W-only GEMV win이 지배**하기 때문. KV-read 텐서코어 win-track은
shared-prefix·대형-M·P·V-단독에 국한되어 이 batch=1 표엔 나타나지 않는다(§2.4).
