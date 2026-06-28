# MSAQ on MXFP8 — mantissa-sharing applied to FP8 elements (E4M3 / E5M2 / E3M4)

> ⚠️ **2026-06-28 PIVOT — 6비트 타깃이면 커스텀 MSAQ 대신 하드웨어-native MXFP6-E2M3 권장.**
> bits-vs-QSNR Pareto에서 E2M3(6.25b)가 ~30dB를 가장 싸게 주고 **유일하게 텐서코어 native**(언팩 불필요).
> 정정: 보정 MSAQ(efb)는 정당한 frontier 경쟁자로 **MXINT6를 이긴다**(처음의 "sharing이 손해"는 6.0b vs 6.25b
> 비트 불공정에서 온 착시였음). E2M3가 이기는 건 "더 싼 frontier 점 + HW-native" 조합 때문. → **`precision/mxfp6_results.md`**.

`precision/msaq_mxfp8_ppl.py`. MSAQ까지 INT8(MXINT8)에 적용해온 mantissa-sharing을 **FP8 원소**에 적용한다.
block=32 + per-block E8M0 scale(=MX)는 그대로 두고, 각 원소를 FP8(sign + eb지수 + mb만티사)로 두되
**만티사 하위 u비트를 mg개 원소가 공유**(원소별 지수 단위로 정규화해 적용 → 서로 다른 지수에도 올바르게 더해짐).

- 원소 포맷: E4M3(eb4/mb3), E5M2(eb5/mb2), E3M4(eb3/mb4).
- 유효 bits/elem = `1(sign) + eb + (mb−u) + u/mg + 8/32(E8M0)`.
- cross-check (selftest, 정확히 성립): **u=0 → 순수 MXFP8** (Δ=0.0), **mg=1 → full FP8**.

## 결과 — power-weighted QSNR (dB), block=32

3개 합성 분포: `W`=Gaussian 가중치, `Wt`=약한 heavy-tail, `Ws`=블록 내부 dynamic range 큼(log-uniform, 2^10 범위).

| 포맷 | cfg | bits | QSNR_W | QSNR_Wt | QSNR_Ws |
|---|---|--:|--:|--:|--:|
| E3M4 | MXFP8(u0) | 8.25 | 37.50 | 37.49 | 37.80 |
| E3M4 | u2/mg8 | 6.50 | 25.86 | 25.87 | 26.68 |
| E3M4 | u3/mg4 | 6.00 | 20.84 | 20.86 | 21.56 |
| E4M3 | MXFP8(u0) | 8.25 | 31.45 | 31.44 | 31.39 |
| E4M3 | u2/mg8 | 6.50 | 20.02 | 20.04 | 20.94 |
| E5M2 | MXFP8(u0) | 8.25 | 25.38 | 25.35 | 24.67 |
| **MXINT8** | **MXINT8(u0)** | 8.25 | **42.37** | **41.98** | **41.15** |
| **MXINT8** | **u2/mg8** | 6.50 | **30.67** | **30.29** | **30.57** |
| **MXINT8** | **u3/mg4** | 6.00 | **25.51** | **25.13** | **25.89** |

(위 표의 u>0 행은 **plain 인코더**(wshare·efb 없음) 기준 — 베이스 순위 비교용. 개선된 인코더 수치는 아래
"최적화" 섹션과 `msaq_mxfp8_selftest.txt`. u0(=MXFP8 baseline)은 인코더와 무관하게 동일.)

## 최적화 — 인코더 2단계 coordinate descent (wshare + error-feedback)

shared/upper를 번갈아 최적화하는 **coordinate descent**로 인코더를 개선했다. **저장 포맷·decode는 불변**
(per-elem upper + group shared) — 전부 **인코더 전용, 추론에선 공짜**. `msaq_mxfp8(..., wshare=True, efb_iters=2)`(기본값).

**(a) wshare — FP-특화 가중 shared** (shared | upper 고정). plain `mean(frac)`은 모든 원소가 같은 선형 scale을
쓰는 **INT8에서 물려받은 형태**다. FP8은 원소마다 quantum `step_up_i = 2^(e_i−(mb−u))`가 달라 복원오차가
`(frac_i − shared)·step_up_i`이므로, 그룹 L2 `Σ(frac_i − shared)²·step_up_i²`를 최소화하는 **최적 shared는
`step_up_i²`(∝ 4^e_i) 가중평균**(큰 지수 원소가 오차 지배 → shared가 그쪽에 맞춰짐).

**(b) efb — error feedback** (upper | shared 고정). 초기 upper는 shared를 모른 채 라운딩됐으므로, shared 확정 후
`upper2 = round((y − shared·step_up)/step_up)`로 **재라운딩**해 그룹 공유오차 `(frac_i − shared)`를 per-element
upper가 흡수한다(INT8의 `upper_bits_correction`의 FP 아날로그). (a)→(b)를 번갈아 반복(`efb_iters`)하면
**L2 단조 감소, ~2회에 수렴**. cross-check(u=0, mg=1) 보존, **L2상 절대 손해 없음**.

**(c) bit-storable (디코드 커널 도입 시 발견·수정)**: per-elem upper는 (mb−u) mantissa 비트만 가지므로 top 지수에서
full-mb maxval을 못 담는다. 블록 스케일에 **지수 1칸 headroom**을 줘 top 라운딩이 saturation 대신 promote되게 하면
실제 비트포맷이 모든 값을 정확히 표현(손실 weight ~0.1pp, outlier scope ~0.3–0.5pp). shared 양자는 저장된 지수 ee
기준이라 reconstruction = `2^(ee−mb)·(m_up·2^u + sh)`가 디코드 커널과 bit-exact. u>0에만 적용(u0은 full-mb).

- QSNR(합성, storable): wshare가 plain 대비 큰 폭, efb가 그 위에 추가 +0.4~0.6 dB(공격적 u3/u4에서 최대).
- **결정적으로, 이 둘은 base format의 우열을 실제로 뒤집는다(아래 PPL).** 합성 Gaussian W의 QSNR에선 여전히
  INT8이 +3 dB 우세지만, **실제 모델 weight/activation은 블록 내 dynamic range가 커서 `Ws` regime에 가깝고**,
  거기선 FP8-MSAQ가 INT8과 대등~우세(E3M4 6.0b Ws 26.85 > INT8 25.89). 전체 표: `msaq_mxfp8_selftest.txt`.

## 결론 — **개정(2026-06-27): 저비트에서 MXFP8-MSAQ가 MXINT8-MSAQ를 이긴다**

⚠️ 이전 결론("INT8이 전 scope 승")은 **plain-mean 인코더(wshare·efb 둘 다 없음)** 기준이었고, 개선된 인코더의
PPL은 측정된 적이 없었다. 실제로 측정하니 **결론이 뒤집힌다**. **proxy(SmolLM2-1.7B)와 정식 Llama-3.1-8B 양쪽에서 확인**.

1. **MSAQ는 MXFP8에 깨끗하게 적용된다** — 구현·검증 완료(u0=MXFP8, mg1=full FP8 cross-check 정확).
2. **비트가 낮을수록 E3M4-MSAQ가 MXINT8-MSAQ를 더 확실히 이긴다 — bit-dependent crossover** (Llama-3.1-8B 정식,
   bit-storable 인코더):
   - **7.38b: INT8 전 scope 승** (고비트에선 INT8 baseline 우위 지배).
   - **6.75b: 박빙** — activation만 E3M4 뚜렷(1.13 vs 1.61), weight 동률, KV/wKV는 INT8(미세). (storable 보정 전엔 3/4승)
   - **6.00b: E3M4가 KV 포함 4/4 승** (weight+act 4.27% vs INT8 6.02%; weight 2.57 vs 2.98; KV 0.96 vs 0.99).
   - **activation(weight+act)이 가장 견고한 E3M4 승부처** — 6.0~6.75b 전부 우세. outlier가 많은 scope에서
     **per-element 지수가 실제로 값을 한다**는 원래 가설이 정식 8B + bit-storable 포맷에서도 입증.
3. **무엇이 레버였나**: 주역은 **wshare**(plain→wshare가 대부분: proxy 6.0b weight+act 46→7.8%). **efb는 그 위에 일관되게
   −1~2pp 추가**(7.8→6.0%)하는 보조 레버 — 공짜라 가치 있으나 주역은 아니다.
4. **여전한 한계**: E5M2(mb2)는 +15~60%로 사용 불가, E4M3(mb3)도 E3M4에 밀린다. **mantissa가 많은 E3M4만**
   INT8과 경쟁 가능. 8.25b baseline QSNR은 INT8(42)≫E3M4(37.5)로 INT8이 우세하나, **저비트 MSAQ regime에선
   FP8의 per-element 지수 이점이 INT8의 baseline 우위를 역전**한다.

**함의**: "블록 스케일이 range를 주니 uniform INT가 항상 낫다"는 통설은 **고비트에선 맞지만 저비트 MSAQ에선 깨진다** —
mantissa 비트를 깎는 MSAQ가 INT8을 더 크게 손상시키고(uniform grid가 작은 원소를 굶김), FP8의 지수가 그 손상을
완화하기 때문. **MSAQ는 base format 선택을 INT8→FP8(E3M4)로 뒤집을 수 있다 — 6.0b에서 깨끗이, activation scope에선
6.75b까지.** 그리고 이 정확도 우위는 **디코드 지연 없이** 온다(아래 커널 섹션: E3M4 6.0b dequant = INT8-MSAQ 6.0b).

## PPL 측정 — 정식 Llama-3.1-8B (wikitext-2, BF16 PPL=5.6877) ★

모델 = `NousResearch/Meta-Llama-3.1-8B`(gated 원본과 동일 가중치의 ungated 미러; config 동일 확인). 인코더 =
**bit-storable** wshare + efb_iters=2 (아래 ⚠️ 저장가능성 보정 반영). BF16 대비 PPL 증가율(%). (`msaq_mxfp8_ppl_llama31_8b.txt`)

⚠️ **저장가능성(storable) 보정**: 디코드 커널 구현 중, 기존 `msaq_mxfp8` 레퍼런스가 블록 최대 원소를 **full-mb
mantissa maxval로 clamp**해 실제 비트포맷(per-elem upper는 mb−u mantissa 비트만 가짐)이 **저장할 수 없는 값**을
만들고 있었음을 발견. 올바른 bit-storable 인코더(블록 스케일에 지수 1칸 headroom → top 라운딩이 saturation 대신
promote)로 교체. 손실은 일반 weight ~0.1pp, outlier-heavy scope ~0.3–0.5pp로 작아 **결론은 유지**되나, 6.75b가
약간 좁혀짐. 아래는 **storable** 수치.

**bit-matched 직접 비교 — E3M4(최선 FP8) vs MXINT8** (승자 굵게):

| bits | scope | E3M4-MSAQ | MXINT8-MSAQ |
|--:|---|--:|--:|
| 7.38 | weight | +0.43 | **+0.19** |
| 7.38 | weight+act | +0.83 | **+0.40** |
| 7.38 | KV | +0.23 | **+0.12** |
| 7.38 | weight+KV | +0.66 | **+0.28** |
| 6.75 | weight | **+0.73** | +0.75 |
| 6.75 | weight+act | **+1.13** | +1.61 |
| 6.75 | KV | +0.32 | **+0.23** |
| 6.75 | weight+KV | +1.03 | **+1.00** |
| 6.00 | weight | **+2.57** | +2.98 |
| 6.00 | weight+act | **+4.27** | +6.02 |
| 6.00 | KV | **+0.96** | +0.99 |
| 6.00 | weight+KV | **+3.48** | +4.29 |

- **bit-dependent crossover (storable)**: 7.38b INT8 전승 → 6.75b 박빙(activation만 E3M4 뚜렷, weight 동률, KV/wKV INT8)
  → **6.0b E3M4 4/4 승**(weight 2.57 vs 2.98, act 4.27 vs 6.02, KV 0.96 vs 0.99, wKV 3.48 vs 4.29).
- **activation(weight+act)이 E3M4의 가장 견고한 승부처** — 6.0~6.75b 전부 우세. outlier가 많은 scope에서 per-element
  지수가 값을 한다는 가설이 정식 8B + bit-storable 포맷에서도 유지.
- storable 보정 후 6.75b가 박빙으로 좁혀졌으나(이전 non-storable은 6.75b 3/4승), **6.0b의 깨끗한 우세와 activation
  우위는 견고**. 8B는 proxy보다 절대 열화가 작다(6.0b weight+act 4.27% vs proxy 6.59%).
- E5M2 여전히 사용 불가(+4~20%), E4M3는 8B에선 꽤 쓸 만(weight 6.75b +2.74 OK)하나 E3M4에 밀린다.

## PPL 측정 — proxy SmolLM2-1.7B (wikitext-2, BF16 PPL=6.9955)

⚠️ **proxy**: 작은 모델 빠른 검증용(초기 측정). 절대값은 8B와 다르나 **포맷 간 상대 순위·scope 거동**은 위 8B와 일치.
표는 BF16 대비 PPL 증가율(%), within 3% 기준. (`msaq_mxfp8_ppl_smollm2.txt`, 인코더 = **storable** wshare + efb_iters=2)

**bit-matched 직접 비교 — E3M4(최선 FP8) vs MXINT8** (승자 굵게):

| bits | scope | E3M4-MSAQ | MXINT8-MSAQ |
|--:|---|--:|--:|
| 7.38 | weight | +0.31 | **+0.30** |
| 7.38 | weight+act | **+0.88** | +0.90 |
| 7.38 | KV | +0.50 | **+0.14** |
| 7.38 | weight+KV | +0.90 | **+0.46** |
| 6.75 | weight | +1.20 | **+1.18** |
| 6.75 | weight+act | **+2.02** | +2.76 |
| 6.75 | KV | +0.74 | **+0.57** |
| 6.75 | weight+KV | +1.83 | **+1.71** |
| 6.00 | weight | **+4.14** | +4.22 |
| 6.00 | weight+act | **+6.59** | +10.68 |
| 6.00 | KV | **+1.52** | +1.82 |
| 6.00 | weight+KV | **+6.10** | +6.36 |

8B와 동일 패턴: **6.0b E3M4 4/4 승, activation은 6.75b까지 우세, 7.38b INT8**.

**레버 분리 (wshare가 주역, efb 보조)** — 아래는 *non-storable* 인코더의 초기 측정(레버 크기 예시; `MSAQ_EFB=0`으로
efb-off 재현). storable 절대값은 위 표와 다르나 plain→wshare의 큰 점프, efb의 −1~2pp 추가라는 **구조는 동일**:

| bits | scope | plain | wshare | wshare+efb | INT8 |
|--:|---|--:|--:|--:|--:|
| 6.00 | weight+act | +46.06 | +7.79 | **+5.96** | +10.68 |
| 6.00 | KV | +10.52 | +1.75 | **+1.22** | +1.82 |
| 6.00 | weight+KV | +28.77 | +7.34 | **+5.75** | +6.36 |

- **wshare가 주역, efb가 보조**(−1~2pp 추가). 둘 다 인코더-only → **추론 공짜**.
- **E5M2 사용 불가, E4M3 < E3M4**; **mantissa가 많은 E3M4만 INT8과 경쟁**.

## 디코드 커널 — E3M4 6.0b vs INT8 (속도) ★

E3M4 6.0b dequant-to-bf16 커널 구현(`csrc/w_gemv.cu` `msfp8_e3m4_dequant_bf16_kernel`,
`ms_lib.pack.pack_weight_msfp8`, 벤치 `tests/msfp8_decode_bench.py`). **레퍼런스와 bit-exact**(QSNR 457 dB,
max err 0). 핵심 통찰: E3M4 u3의 per-element 필드 폭 = 1+eb+(mb−u) = 1+3+1 = **5비트 = INT8의 8−u = 5비트와 동일**
(1+eb+mb=8). 즉 **비트스트림 언팩·메모리 트래픽이 INT8-MSAQ와 완전히 동일**하고, FP 복원 ALU(exp/mantissa split +
`ldexpf`)만 추가된다.

격리된 dequant latency (OUT=K=4096, 16.8M elem → bf16, Blackwell sm_120):

| 포맷 | bits | latency | vs |
|---|--:|--:|--|
| MXFP8-MSAQ E3M4 | 6.0 | 64.2 us | — |
| MXINT8-MSAQ | 6.0 | 64.2 us | **E3M4 = 1.001×** (동일) |
| plain MXINT8 | 8.25 | 86.5 us | E3M4 = 0.743× (더 빠름) |

- **E3M4 6.0b 디코드는 INT8-MSAQ 6.0b와 사실상 동일 속도**(+0.1%). dequant는 bf16 write-bound(출력 2B/elem ≫
  입력 0.75B/elem)라 추가 FP 복원 ALU가 write 뒤에 완전히 숨는다. **정확도 우위가 디코드 지연 없이 공짜.**
- plain MXINT8(8.25b)보다는 빠르다(바이트 적음). 즉 6.0b E3M4는 **INT8-MSAQ 대비 정확도↑·속도=, plain MXINT8 대비
  bits↓·속도↑**.
- ⚠️ 위는 **격리 dequant**(write-bound) 기준. 융합 커널에서의 overlap 거동은 아래 별도 섹션.

## 융합 GEMV/GEMM — overlap 거동 ★ (E3M4 vs INT8-MSAQ)

융합 커널 구현(`csrc`: `wonly_gemv_wide_fp8`/`wonly_gemv_batched_fp8` via `stream_block_uspec_fp8_e3m4`;
GEMM은 `wonly_gemm_wmma_pipe<CM,FP8>` + `dequant_col_stream_fp8`. 런처 `msfp8_gemv_wide/batched/gemm`,
벤치 `tests/msfp8_gemv_gemm_bench.py`). correctness rel ~1.7e-3 (bf16 수준). **핵심: dequant은 write-bound라
ALU가 숨지만, 융합 GEMV(M=1)는 latency/extraction-bound라 FP 복원 ALU(`ldexpf`)가 노출된다.**

| 연산 | M | E3M4 / INT8-MSAQ | 해석 |
|---|--:|--:|---|
| 격리 dequant | — | **1.00×** | write-bound, ALU 완전 은닉 |
| 디코드 GEMV (wide) | 1 | **1.74×** (느림) | overlap 없음 → `ldexpf` 완전 노출 |
| batched GEMV | 4 | 1.46× | compute 늘며 점점 은닉 |
| batched GEMV | 8 | 1.35× | |
| batched GEMV | 16 | 1.18× | |
| batched GEMV | 32 | 1.18× | |
| prefill GEMM (wmma_pipe) | 128 | **0.99×** (동률) | column dequant이 WMMA와 overlap → 은닉 |
| prefill GEMM | 256–1024 | 1.07× | |

(GEMV는 둘 다 fast uspec 경로가 있는 u3/gs16에서 공정 비교 — gs4에선 INT8 batched가 generic fallback이라 불공정.)

- **E3M4의 FP-복원 오버헤드는 compute가 많을수록 숨는다**: M=1(1.74×) → batched(1.2–1.5×) → GEMM(≈1.0×).
- **decode(M=1)에서 E3M4는 INT8-MSAQ보다 느리다** — 기존 "저비트 W-decode는 MXINT8에 구조적으로 불리" 결과의 연장
  (FP 복원이 그 위에 ldexpf를 더함). [[msaq-vs-mxint8-w-decode-state]]와 같이 봐야 함.
- **GEMM(prefill)에선 정확도 우위가 사실상 공짜** (≈INT8-MSAQ). 즉 **E3M4는 compute-bound(prefill/배치) 경로에 적합**,
  latency-bound 단일 decode엔 불리.
- 참고: 이 wmma_pipe GEMM 경로는 bf16 cuBLAS보다 5–9× 느리다(MSAQ·INT8 공통; 빠른 `fused_skinny`는 u3/gs16·u2/gs8
  전용이라 gs4 미지원). E3M4의 결론(overlap 시 ALU 은닉)은 경로와 무관.

## Blackwell MXFP8 하드웨어 활용 — 조사 결과 (2026-06-28)

**결론: E3M4-MSAQ는 Blackwell 텐서코어 경로를 전혀 탈 수 없다 (2중 차단).**

1. **E3M4는 하드웨어 포맷이 아니다** — OCP MX v1.0 / PTX ISA / FP8 표준(arXiv 2209.05433) 모두 FP8 = **E4M3·E5M2뿐**
   (FP6=E3M2/E2M3, FP4=E2M1). E3M4는 학술 포맷, 텐서코어 datapath 없음(sm_100·sm_120 공통).
2. **공유-mantissa / 임의 sub-byte 메커니즘이 없다** — block-scaled MMA(`tcgen05.mma`/`mma.sync ... block_scale`)는
   고정된 `{e4m3,e5m2,e3m2,e2m3,e2m1}` + UE8M0/UE4M3 scale(block 32, NVFP4만 16)만 받는다. 게다가 `mxf8f6f4`에선
   6-bit·4-bit도 **바이트 패딩**(1B/elem)이라 sub-byte 저장 이득도 MMA 경계에선 없음(진짜 패킹은 FP4 전용 `mxf4`).
   → 6.0b 공유-mantissa는 MMA 전에 표준 MX 원소로 **언팩**해야 하고, 그 언팩이 바로 우리가 아끼려던 ALU.

**그래도 남는 활용 방안(실행 가능 옵션):**
- **(권장) MSAQ를 표준 MX 원소를 직접 내보내도록 재정식화** — 예: E3M2/E2M3(MXFP6) 또는 E4M3(MXFP8)을 UE8M0 블록
  스케일로 바로 출력하면 per-element 언팩 없이 native block-scaled 텐서코어에 올린다. 이때만 하드웨어 가속이 가능.
  단 MXFP6/8은 6.0b MSAQ보다 비트가 크거나(8.25b) 정밀도가 다르므로, accuracy↔throughput 재평가 필요.
- **소비자 5090(sm_120) 주의**: FP8을 FP32 누산하면 텐서 처리량이 **절반**(BF16 수준)이라 FP8의 2× 이점이 상쇄 —
  즉 bf16-WMMA fallback의 손해가 생각보다 작다. FP4/block-scaled 경로는 스로틀 없음.
- 요구 버전: block-scaled MXFP8 GEMM은 CUDA≥12.8(권장 12.9)·CUTLASS≥3.8(현 4.x). 예제 `72/79/84_blackwell_*`,
  cuBLASLt `LtMxfp8Matmul`. sm_120은 `sm_120a` 타깃 필요(ptxas/lowering 버그 잔존, version-fragile).

## 한계 / 다음

- ✅ **Llama-3.1-8B 정식 재현 완료**(bit-storable 인코더) — 6.0b E3M4 4/4승, activation 6.75b까지 우세. 위 ★ 섹션.
- ✅ **디코드/융합 커널 구현·측정 완료** — 격리 dequant=동률, GEMM=동률(overlap 은닉), 단 M=1 decode는 1.74× 느림
  (ALU 노출). 위 두 ★ 섹션.
- ✅ **Blackwell MXFP8 조사 완료** — E3M4·공유mantissa 모두 HW 경로 없음; 재정식화(표준 MX 원소 출력)만이 가속 옵션.
- 합성 Gaussian W QSNR은 여전히 INT8 우세지만, **모델 PPL은 저비트에서 FP8(E3M4) 우세** — 실제 weight/act가 합성보다
  intra-block dynamic range가 커서 `Ws` regime에 가깝기 때문. 즉 QSNR(Gaussian)은 더 이상 최종 판정 기준이 아니다.
- **다음**: (1) E3M4 batched/GEMV의 M=1 ALU 노출을 줄일 수 있는지(`ldexpf` 제거·정수 지수 삽입 등) — 단 INT8의
  sepsc 우위는 구조적. (2) **Blackwell 가속을 노린다면 MSAQ→표준 MXFP6(E3M2/E2M3) 재정식화** + accuracy↔throughput
  재평가. (3) crossover(~6.75b) sub-scope별 정밀화. (4) attention(KV) 융합 경로. (5) 다른 모델군 일반화.
- PPL 스윕은 동일 파일에 구현돼 있어 precision 환경에서 재실행 가능(`MSAQ_MODEL`로 모델 지정):
  ```bash
  # 정식 8B(미러). MSAQ_EFB=0 으로 efb 끄고 wshare-only 기여도 분리 가능.
  MSAQ_MODEL=NousResearch/Meta-Llama-3.1-8B CUDA_VISIBLE_DEVICES=0 \
    python precision/msaq_mxfp8_ppl.py > precision/msaq_mxfp8_ppl_llama31_8b.txt 2>&1
  ```
  scope(weight / weight+act / KV / weight+KV) × 포맷(E4M3/E5M2/E3M4) × (u,mg)로 BF16 대비 PPL %를 출력(within 3% 기준).
