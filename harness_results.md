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
**공정성 수정 최종**: KV read MXINT8 baseline을 MSAQ와 동일 **thread-per-key**로 올려 매핑 비대칭
제거([for_fair_comparison.md]). 그 결과 KV를 쓰는 S3·S4의 MXINT8이 ~빨라져 **이전(비공정) S3 KV
win은 tie로 정정**. 괄호 = 공정성 수정 전 값.

| 시나리오 | 포맷 | TTFT | TPOT | total | /bf16 | /mxint8 |
|---------|------|------|------|-------|-------|---------|
| baseline | bf16 | 275 | 37.5 | 145.7s | 1.00 | — |
| **S1 W-only** | MXINT8 | 1600 | 37.1 | 145.6s | 1.00 | — |
| | MSAQ-u2 | 1715 | 34.6 | 136.0s | 0.93 | 0.93 |
| | MSAQ-u3 | 1691 | 34.2 | 134.5s | 0.92 | 0.92 |
| | **MSAQ-u4** | 1496 | 29.4 | 115.7s | **0.79** | **0.79** |
| **S2 W+A** | MXINT8 | 1586 | 32.8 | 128.9s | 0.88 | — |
| | MSAQ-u2 | 1480 | 33.6 | 132.0s | 0.91 | 1.02 |
| | MSAQ-u3 | 1447 | 33.1 | 129.9s | 0.89 | 1.01 |
| | **MSAQ-u4** | 1210 | 29.7 | 116.3s | **0.80** | 0.90 |
| **S3 KV-only** | MXINT8 | 276 | 23.6 | 91.9s | **0.63** | — |
| | MSAQ-u2 | 276 | 25.1 | 97.8s | 0.67 | 1.06 (0.97) |
| | MSAQ-u3 | 277 | 25.1 | 97.6s | 0.67 | 1.06 (0.97) |
| | **MSAQ-u4** | 277 | 23.9 | 92.9s | 0.64 | **1.01** (0.92) |
| **S4 W-only+KV** | MXINT8 | 1600 | 23.3 | 91.9s | **0.63** | — |
| | MSAQ-u2 | 1717 | 22.5 | 88.9s | 0.61 | 0.97 (0.88) |
| | MSAQ-u3 | 1696 | 22.0 | 87.0s | 0.60 | 0.95 (0.87) |
| | **MSAQ-u4** | 1497 | **16.0** | **63.8s** | **0.44** | **0.69** (0.64) |

> **공정성 수정 효과:** KV를 양자화하는 S3·S4의 MXINT8이 thread-per-key로 ~빨라짐(total 100.7→
> 91.9s). **S3 KV-only는 MSAQ가 tie(u4 1.01)로 정정**(이전 0.92 win은 MXINT8 under-optimization
> 산물 — [for_fair_comparison.md]). **S4는 u4 0.69로 여전히 win**: KV read가 tie여도 **W-only
> GEMV(진짜 BW-bound)의 win이 지배**하기 때문. bf16 대비는 전부 여전히 win.

## TPOT 성장곡선 (graph, ms) — 컨텍스트별 순간 per-step

최적화 후. (KV read streaming unpack 덕에 S3·S4의 u2/u3 곡선이 후반 ctx에서 MXINT8 아래로 내려옴.)

| 경로 | ctx 801 | 1056 | 1824 | 2848 | 4680 |
|------|------|------|------|------|------|
| bf16 baseline | 26.95 | 28.34 | 32.47 | 38.08 | **47.94** |
| S1 W-only mxint8 | 26.60 | 27.97 | 32.12 | 37.69 | 47.53 |
| S1 W-only msaq-u4 | 19.04 | 20.40 | 24.56 | 30.03 | 39.73 |
| S2 W+A mxint8 | 22.34 | 23.75 | 27.89 | 33.43 | 43.23 |
| S2 W+A msaq-u2 | 23.13 | 24.54 | 28.69 | 34.20 | 44.06 |
| S2 W+A msaq-u4 | 19.18 | 20.58 | 24.76 | 30.28 | 40.06 |
| S3 KV-only mxint8 (fair) | 22.28 | 22.52 | 23.00 | 23.71 | **24.88** |
| S3 KV-only msaq-u4 | 22.41 | 22.66 | 23.17 | 23.96 | 25.33 |
| S4 W-only+KV mxint8 (fair) | 21.92 | 22.17 | 22.65 | 23.35 | 24.61 |
| S4 W-only+KV msaq-u3 | 19.76 | 20.22 | 20.91 | 22.07 | 24.24 |
| **S4 W-only+KV msaq-u4** | **14.66** | 14.89 | 15.38 | 16.11 | **17.46** |

> 공정성 수정 후 **MXINT8 KV(thread-per-key)도 평탄**해졌다(S3 4680: 28.8→**24.9ms**). 즉 곡선 평탄화는
> "KV를 양자화하는 것" 자체의 효과(MSAQ·MXINT8 공통)이고, **MSAQ vs MXINT8 KV는 tie**(S3 u4
> msaq 25.3 ≈ mxint8 24.9). S4 msaq-u4가 최저·최평탄인 건 weight GEMV 이득이 더해졌기 때문.

## 해석 — 적용 대상별 기여가 분리됨

0. **⚠️ 공정성 정정(가장 중요):** KV read MXINT8 baseline을 fair(thread-per-key)로 올린 뒤
   **MSAQ의 KV-read 우위는 사라졌다(tie)**. 따라서 MSAQ의 *공정한* end-to-end 이득은 거의 전부
   **weight GEMV(W-only scope)**에서 온다. KV 양자화는 bf16 대비 둘 다 이득이나 **MSAQ vs MXINT8
   KV는 tie**(자세히는 아래 §"KV read 공정성"). 이전 라운드 표(S3 0.92 win 등)는 MXINT8
   under-optimization 산물이었다.

1. **weight 양자화는 baseline을 낮추고, KV 양자화는 성장곡선을 평탄화한다(직교).** 단 곡선 평탄화는
   **MSAQ·MXINT8 공통**(KV를 양자화하면 둘 다 평탄):
   - **bf16**: ctx 길어지며 26.9→47.9ms (KV read 폭증).
   - **S3 KV-only**: msaq 22.4→25.3 ≈ mxint8(fair) 22.3→24.9 — **둘 다 평탄, 서로 tie**.
   - **S1 W-only(msaq)**: 19.0→39.7 — baseline 낮지만 KV가 bf16이라 여전히 성장.
   - **S4 둘 다(msaq-u4)**: 14.7→17.5ms — 최저·최평탄. weight GEMV 이득 + KV 평탄.

2. **최고 = S4 W-only+KV MSAQ-u4: bf16의 0.44×, MXINT8의 0.69×** (145.7s→**63.8s**).
   bf16 대비는 weight+KV 양자화가 compound(TPOT 37.5→16.0ms). **MXINT8 대비 0.69의 출처는
   weight GEMV**(KV는 tie라 중립) — 즉 공정 비교에서도 S4가 이기는 건 W-only scope 덕분이다.

3. **MXINT8 W-only는 end-to-end 이득이 0 (S1 mxint8 = 1.00×bf16).** MXINT8 GEMV는 cuBLAS bf16
   GEMV와 거의 동속(커널벤치 47 vs 46µs)이라 decode를 못 줄인다. 반면 **MSAQ W-only는 0.79** —
   wide-load u4 GEMV(커널 0.66×cuBLAS)가 cuBLAS를 이겨서 실제로 baseline을 내린다. → **W-only
   scope가 MSAQ의 진짜 BW-bound win이고 MXINT8 대비 0.79로 가장 명확.**

4. **KV-only(S3): bf16 대비 0.63~0.64(둘 다 win)이나 MSAQ vs MXINT8는 tie(u4 1.01).** weight를
   bf16로 둬도 KV 양자화가 긴 컨텍스트 decode를 크게 가속(TTFT는 bf16 그대로 ~276ms 유지). 단
   **MSAQ KV가 MXINT8 KV를 이기지는 못한다**(fair thread-per-key 기준 tie; §"KV read 공정성").

5. **MSAQ vs MXINT8 (시나리오별, fair):** S1 W-only **0.79**(완승, weight GEMV) · S2 W+A 0.90 ·
   **S3 KV 1.01(tie)** · S4 W+KV **0.69**(weight GEMV win이 KV tie 위에 얹힘). 즉 공정 비교에서
   MSAQ의 end-to-end 우위는 **weight GEMV에서 오고 KV read는 중립**이다.

6. **TTFT 트레이드오프:** weight를 양자화하는 S1·S2·S4는 prefill GEMM이 cuBLAS 대신 커스텀
   IMMA/타일이라 TTFT 1.2~1.6s(bf16 0.28s 대비 ~5×). **KV-only(S3)는 bf16 weight라 TTFT 손해 0.**
   decode 3880이 total을 지배해 weight-quant도 total에선 이득이지만, **짧은 생성**이면 KV-only가
   TTFT까지 안전한 선택.

7. **u-스윕: u4가 전 시나리오에서 압도적, u2/u3는 W-only에서만 MXINT8을 이긴다.** u4는 upper 코드가
   nibble-정렬이라 wide int4 load + 값싼 bfe로 언팩이 가볍지만, u2/u3는 streaming bit-buffer
   언팩이 무거워 더 적은 bit를 읽는 대역폭 이득을 까먹는다(커널 단위 결과 그대로):
   - **S1 W-only**: u2/u3도 MXINT8을 이김(/mxint8 0.93/0.92) — MXINT8 GEMV가 cuBLAS와 동속이라
     u2/u3의 약한 이득으로도 역전. u4는 0.80.
   - **S2 W+A**: u2/u3가 MXINT8보다 ~tie/약간 느림(1.01~1.02), **u4만** crossover(0.90).
   - **S3 KV-only(fair)**: u4 **tie(1.01)**, u2/u3 손해(1.06) — KV read는 어느 u도 MXINT8을 못 이김.
   - **S4**: u2/u3 ~동률(0.95~0.97), **u4 0.69**(weight GEMV win 지배).
   → **실전 권장 = u4.** u2/u3은 정확도용 옵션. (단 KV read 자체는 u4도 fair tie — win은 weight scope.)

## KV read 공정성 (요약 — 상세는 [for_fair_comparison.md])
이전 라운드의 KV read "win"(S3 0.92 등)은 **MXINT8 baseline이 구버전 warp-per-key라 ~2× 느렸던
산물**이었다. MXINT8을 MSAQ와 동일 **thread-per-key**로 올리니(공정) KV read는 **u4 tie·u2/u3 손해**.
근본 이유: flash-decode가 BW 시간의 ~20× 느리게 도는 **latency/overhead-bound**라(텐서코어 미사용)
"바이트를 덜 읽는" 이점이 시간에 안 나타난다. BW-bound로 끌어올리려 했으나 **sub-byte V의 Pass-2
(per-d reduction)**가 half-sector(직접) 또는 staging-overhead(occupancy 캡)라는 장애물에 막혀
(direct-V 실험은 tie→1.40 악화), 깨끗한 KV win은 **완전한 FlashDecoding 재설계**(미해결 과제)를
요구한다. → 본 표의 S3/S4는 **공정 최선(KV tie)** 상태다.

## 커널 최적화 라운드 (Phase 31) — KV write / KV read / W+A GEMV

마지막에 짠 세 커널(KV write·KV append·W+A GEMV)은 "matched + mantissa-sharing 정확" 수준까지만
와 있었음. mantissa-sharing 쪽 최적화를 적용(숨길 수 있는 실행시간은 숨기고, BW/occupancy를 높이는
방향). 결과(괄호=최적화 전 `/mxint8`):

| 커널 | lever | 결과 |
|------|-------|------|
| **KV write** | nb를 grid.z로 올려 occupancy 확보(GQA H=8에서 32→224 블록). **generic이라 MXINT8에도 동일 적용**(matched). | L=800에서 u4 **1.09→0.88** |
| **KV read** | u2/u3에 **streaming bit-buffer unpack** 이식(general-straddle→롤링버퍼). MXINT8은 int8 직읽이라 무변경 → mantissa-sharing-only. **KV-only decode의 진짜 병목**(write/append이 아님). | Lk=4680 u2 **1.29→0.79**, u3 1.27→0.77 |
| **W+A GEMV** | 활성화 qx를 블록당 shared에 1회 stage(컬럼마다 재로드 제거). MSAQ만 적용. | u2 **1.17→1.11**, u3 1.15→1.07, u4 0.83 |
| **KV append** | 미적용(아래 "이길 수 없는 이유"). | ~1.04 |

**end-to-end 효과 (u2/u3가 지던 곳이 역전):**
- **S3 KV-only u2/u3: 1.07 → 0.97** (이제 win). KV read streaming이 직접 효과.
- **S4 W-only+KV u2/u3: 0.98/0.96 → 0.88/0.87**. KV read win + W-only GEMV win 누적.
- **S2 W+A u2/u3: 1.06/1.05 → 1.02/1.01** (개선되나 ~tie, 미세하게 짐).
- u4는 전부 그대로 win, S1은 decode가 W-only GEMV(이미 최적)라 무변.

### MXINT8보다 빨라지지 못한 케이스 — 이유

1. **W+A GEMV u2/u3 (1.01~1.11), KV read u2/u3 @ short ctx (~1.18): extraction-bound (근본).**
   u2/u3은 byte를 0.69~0.78×만 줄이는데(u4는 0.56×), sub-byte를 꺼내는 streaming-unpack 추출
   비용이 그 절약을 까먹는다. 게다가 W+A의 MXINT8 baseline은 이미 int-dot로 BW 효율적(~400GB/s).
   → **upper 코드가 nibble(4bit) 정렬되는 u4만** wide int4 load + 값싼 bfe로 추출이 가벼워 이긴다.
   u2/u3(비-nibble)은 이 아키텍처에서 구조적으로 MXINT8 직읽기를 못 넘는다.
2. **KV append: MSAQ가 항상 MXINT8보다 일을 더 한다(decompose **+ bit-pack** vs decompose+store).**
   게다가 단일 토큰이라 H·nb=32 스레드 1-블록 → launch/tiny-work 지배. 더 적은 일로 줄일 수가
   없다(bit-pack은 포맷 그 자체). 단 CUDA graph 하에선 launch가 숨겨져 GPU 시간이 미미 → end-to-end
   영향 거의 0.
3. **방법론 노트:** qx-staging은 MSAQ(unpack-stall로 latency-bound)에는 도움되지만 MXINT8(이미 BW
   효율)에는 **해가 됨**(39.6→54µs). generic lever라도 한쪽에만 맞으면 양쪽 강제는 불공정 → 각 커널을
   **각자의 최적 구성(best-vs-best)**으로 비교. occupancy 같이 양쪽에 똑같이 이로운 lever만 mirror.

**요약:** 세 커널 최적화 후 **u2/u3가 S1·S3·S4에서 MXINT8을 이기게 됨**(전엔 S3/S4 일부에서 짐).
유일하게 못 이기는 곳은 **W+A GEMV u2/u3(~tie)**와 **KV append**(둘 다 위의 근본 이유)이며, u4는
처음부터 전 구간 win. → **권장 u4에서 4개 시나리오 전부 MXINT8보다 빠름이 증명됨**(S1 0.80·S2 0.90·
S3 0.92·S4 0.64).

## 이전(Python-loop) 대비
이전 coupled-run의 W-only+KV(=S4) MSAQ-u4는 0.60×bf16이었으나, **CUDA graph로 dispatch
오버헤드를 제거하니 0.44×**로 더 벌어짐 — Python 루프의 per-op dispatch가 TPOT를 부풀려 비율을
1.0쪽으로 희석했음을 확인. graph 측정이 커널의 실제 이득을 드러낸다.

## 주의
타이밍 하니스: 가중치 랜덤·레이어 재사용(타이밍은 값 무관), glue·lm_head는 bf16 공통. graph는 스텝당
back-to-back 커널 시간(dispatch-free)을 재고, total은 trajectory 적분.
