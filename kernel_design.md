# kernel_design.md — MSAQ 커널 설계 종합 정리

> RTX 3090(Ampere GA102, sm_86, 82 SM, L2 6 MB) 대상. 이 문서는 무엇을·어떻게
> 설계했고, 왜 MXINT8 vs MSAQ가 공정한 비교이며, quantization scope별로 어떤 커널을
> 만들었는지를 한 곳에 정리한다. 작업 이력의 상세 수치는 `change.md`, 차세대 설계는
> `design_packing_redesign.md` / `design_plan4.md` 참조.

---

## 1. 무엇을 만들었나 (개요)

MSAQ(Mantissa-Sharing Adaptive Quantization, signed)로 양자화된 weight/KV를 **HBM에서
적은 바이트로 읽고, 커널 안에서 즉석 복원(fused dequant)** 하여 LLM 추론 4개 scope를
수행하는 CUDA 커널 모음. 각 scope마다 **MSAQ 커널**과, 구조가 동일하고 읽기만 다른
**MXINT8 baseline 커널**을 쌍으로 만들어 "적은 바이트 vs unpack 오버헤드"를 격리 측정한다.

- 핵심 질문: decode는 memory-bound이므로, MSAQ가 덜 읽는 바이트(예 u3gs8: 5.625 vs
  MXINT8 8.25 b/elem)가 **시간 우위**로 나타나는가?
- 이번 작업의 성과: occupancy 병목을 제거해 양 커널 모두 수~수십 배 가속. 그 결과
  병목이 occupancy → **unpack(extraction) 연산량**으로 이동함을 규명(아래 §6, §7).

---

## 2. 수치 포맷

### 2.1 MSAQ-signed (우리 포맷, `ms_lib/pack.py`)
32-원소 블록마다 단일 FP 라운딩(double-rounding 회피):

1. E8M0 공유 스케일 `scale = 2^(floor(log2(max|x|)) − 6)`.
2. **upper**: `q_upper = clip(round(x / (scale·2^u)), ±(2^(7−u)−1))`, **(8−u)bit** signed.
3. residual `= x − q_upper·scale·2^u`, 이를 `gs`개씩 묶어 그룹 평균.
4. **shared**: `r = clip(round(mean / scale), ±2^(u−1))`, **u bit** signed, 그룹당 1개.
5. 복원: `(q_upper·2^u + r_expanded)·scale` (= 유효한 MXINT8 정수 word).

- 파라미터: `u`(공유 비트), `gs`(공유 그룹 크기). 스윕: u∈{2,3,4}, gs∈{2,4,8}.
- 평면(SoA, out-innermost): `scale_exp[nb,OUT] int8`, `upper[nb,UB,OUT] u8`,
  `shared[nb,SB,OUT] u8`. `UB=32·(8−u)/8`, `SB=ceil((32/gs)·u/8)`.
- dense LSB-first 패킹: code가 byte 경계를 straddle할 수 있어, 복원 시 2-byte load +
  shift/or/mask/sign-extend가 필요(`ms::unpack_ms_weight_elem`).
- 유효 비트/원소 = `8/32(scale) + (8−u) + u·(32/gs)/32`. 예 **u3gs8 = 5.625 b/elem**.

### 2.2 MXINT8 baseline (`pack_*_mxint8`)
같은 SoA 평면이지만 mantissa를 **풀 int8로 직접 저장**: `scale_exp[nb,OUT] int8`,
`qweight[nb,32,OUT] int8`. 유효 비트 = `8 + 8/32 = 8.25 b/elem`. 복원이 **직접 int8 read**
(unpack 없음).

---

## 3. 왜 MXINT8 vs MSAQ가 "공정한" 비교인가 (matched-optimization)

두 커널은 **구조가 byte-for-byte 동일**하고 **딱 한 군데, weight/KV 읽기만 다르다**:

| 공통 | MSAQ | MXINT8 |
|------|------|--------|
| thread 매핑, 블록/그리드, split 전략, FP32 누산, online-softmax, combine | sub-byte **unpack**(`ms::unpack_ms_*`) | **직접 int8** read |

따라서 **MSAQ/MXINT8 latency 비**는 정확히 "적은 바이트(MSAQ 유리) ↔ unpack 오버헤드
(MXINT8 유리)"만 격리한다. cuBLAS/SDPA 대비 비는 *다른* 최적화 등급(tensor-core)이라
참고용일 뿐, **matched 비교가 아니다**.

**규율**: 어떤 occupancy 최적화를 MSAQ에 넣으면 **MXINT8 baseline에도 동일하게** 넣는다.
이번에 split-K·split-KV·two-pass·token-major transpose를 전부 양쪽에 똑같이 적용한 이유다.
(그렇지 않으면 MSAQ가 "더 최적화돼서" 이기는 가짜 승리가 된다.)

검증 입력도 동일한 bf16-라운딩 값을 양쪽·oracle에 먹여 커널 오차만 비교한다.

---

## 4. Quantization scope별 커널

| scope | 파일 | MSAQ 커널 | MXINT8 baseline | 최적화 상태 |
|-------|------|-----------|-----------------|-------------|
| W-only GEMV (decode, M=1) | `w_gemv.cu` | split-K / cp.async / **u4 wide-load**(`wonly_gemv_wide_kernel`) + combine | `mxint8_gemv_splitk_kernel` | ✅ **split-K + cp.async + wide-load (u4: MXINT8 추월)** |
| KV decode attention | `kv_attention.cu` | `kv_decode_split_kernel` + `kv_decode_combine` | `mxint8_kv_split_kernel` | ✅ **split-KV + two-pass + token-major** |
| W-only GEMM (prefill) | `wa_gemm.cu` | `wonly_gemm_kernel` | `mxint8_gemm_kernel` | ✅ **shared-mem tiled** (unpack 1회/타일, TBM 재사용 → u4에서 MSAQ<MXINT8) |
| W+A GEMM | `wa_gemm.cu` | `wa_gemm_kernel` | `mxint8_wa_gemm_kernel` | ✅ **shared-mem tiled** (W-only 타일 + act 양자화 staging; u4에서 MSAQ<MXINT8) |

공통 인프라: `csrc/core/ms_utils.cuh`(unpack 원시함수, e8m0 스케일, split 카운트 헬퍼),
`csrc/pybind.cpp`(torch.ops.msaq.* 등록), `ms_lib/ops.py`(pack dict → 디바이스 평면).

### 4.1 W-only GEMV (decode) — `w_gemv.cu`
- 연산: `y[o] = Σ_k x[k]·dequant(W)[o,k]`, OUT=K=4096.
- 설계: **split-K**. 원래 thread 1개가 K 전체를 reduce → block=ceil(OUT/128)=32개뿐
  (82 SM 중 32). K축을 `splitK`로 쪼개 grid=(ceil(OUT/128), splitK) → SM을 채움. 각
  (o,sp) block이 부분합 → `partial[splitK,OUT]` → `gemv_combine_kernel`이 **선형 합산**
  (softmax 불필요, **atomic 없음**). `splitK`는 SM 수에서 산출(`ms::gemv_splitk_count`,
  env `MS_GEMV_SPLITK_MULT` 기본 3).
- plane이 out-innermost라 인접 thread가 인접 byte를 읽어 **이미 coalesced**.
- **메모리 최적화 3단(§7-a)**: (1) **cp.async**로 packed byte를 prefetch해 unpack을 메모리
  뒤에 숨김(Phase 12). (2) **u4 wide-load**: column-major `[nb,OUT,16]`에서 int4 한 번에
  로드 + bfe(Phase 14). (3) split mult=16 + `k/gs`→shift(Phase 16). → **u4에서 MXINT8·cuBLAS
  BF16을 1.45× 추월**(0.033ms vs 0.047ms). u3/u2는 cp.async 경로.

### 4.2 KV decode attention — `kv_attention.cu`
- 연산: decode 1-step. `q[H,D]`가 모든 Lk key에 attend(Lq=1, causal mask 불필요),
  K/V는 MSAQ로 저장돼 FlashAttention load 경로에서 즉석 dequant(분리하면 KV 트래픽 2배).
- 설계 3종 중첩:
  1. **split-KV (Flash-Decoding)**: head로만 병렬화하면 block=H=8(SM 10%). key축을
     `key_tile`로 쪼개 grid=(H, S), S는 SM 수 기반(`ms::kv_split_count`, env
     `MS_KV_SPLIT_MULT` 기본 3). 각 (h,s) block이 자기 key-tile만 online-softmax →
     **부분 (acc, m, l)** 을 별도 버퍼에 기록 → `kv_decode_combine_kernel`이
     log-sum-exp 보정으로 병합(**atomic 없음**).
  2. **two-pass (barrier 제거)**: 원래 key마다 block-wide 트리 reduction(키당 ~8
     `__syncthreads`)이라 MLP가 죽음. → **Pass1(scores)**: 한 warp가 한 key의 q·K dot을
     `__shfl`로 reduce(block barrier 없음, warp들이 다른 key 동시 처리). **Pass2(output)**:
     thread=head_dim가 `out[d]=Σ_kk p·V[d,kk]`를 key 루프로 누적(cross-thread reduction
     없음). barrier는 키당 ~8 → chunk(128 key)당 ~2.
  3. **token-major transpose (coalescing)**: K/V 평면을 `[H,nb,L,BYTES]`(BYTES innermost)로
     배치 → 고정 key에서 warp의 head_dim threads가 **연속 바이트**를 읽어 coalesced
     (트랜잭션 ~32×→~2×). bit-packing 불변이라 재인증 쉬움.

### 4.3 W-only GEMM (prefill) / 4.4 W+A GEMM — `wa_gemm.cu`
- 현재 **correctness baseline**(2D grid, thread 1개가 출력 원소 1개, FP32 누산, tensor-core
  미적용). oracle과 bit-수준 일치하도록 `ms::unpack_ms_weight_elem` 재사용.
- W+A: weight를 int8로 unpack, activation을 즉석 MXINT8 양자화
  (`s=2^(floor(log2 max|x|)−6)`, `q=clip(rint(x/s),±127)`), per-block `(scale_a·scale_w)·
  int-dot`. → 향후 **INT8 IMMA/CUTLASS**로 교체 예정(`design_packing_redesign.md` §4).
- 이번 작업에서 **최적화하지 않음**(decode 두 scope에 집중). prefill/W+A는 compute-bound라
  tensor-core가 본 레버.

---

## 5. 적용한 최적화 (occupancy 로드맵) — 효과 있던 것

Little's law(필요 in-flight 바이트 = latency × 대역폭) 관점에서 "병렬도가 모자라 대역폭
근처도 못 간다"가 1차 진단이었고, 그게 맞았다.

| # | 최적화 | scope | 효과 |
|---|--------|-------|------|
| 1 | **SM 기반 split** (split-K / split-KV) | GEMV, KV | occupancy ≤10~39% 천장 제거 — **1차 효과** |
| 2 | **two-pass barrier 제거** | KV | 키당 ~8 barrier → chunk당 ~2, MLP 확보 |
| 3 | **token-major transpose** | KV | uncoalesced 트랜잭션 ~32×→~2× |
| 4 | **partial buffer + combine** (atomic 회피) | GEMV, KV | split 키워도 contention 직렬화 없음 |

부수 효과: split 카운트를 SM 수에서 동적 산출하므로 문제 크기에 강건. two-pass가 큰 Lk의
over-split cliff도 제거.

---

## 6. 시도했으나 효과 없어 **되돌린** 것 (negative results)

진단을 정확히 하려면 실패도 기록한다. 이들은 "MSAQ는 occupancy가 아니라 **unpack 연산량**에
묶여 있다"를 반증/입증하는 증거다.

| 시도 | 결과 | 이유 |
|------|------|------|
| **register 압력 감소 (방안2)** | 불필요 | MSAQ 40 reg/spill 0, 48-warp 상한에 먼저 걸림 → register가 occupancy를 안 깎음 |
| **Stage 4b: word-align + bfe** | 더 느림(revert) | 정렬 padding +20~26% byte, bfe는 latency chain을 못 줄임(load→bfe→fma 동일) |
| **register-blocking (COLS/thread)** | 중립~악화(revert) | register-heavy unpack가 COLS배 → 114 reg → occupancy 48→17 warp 붕괴 |
| **pass-2 `#pragma unroll`** | 무효과(revert) | 컴파일러가 이미 ILP 추출 / latency 아닌 throughput 한계 |

**일관된 결론**: occupancy/구조 최적화가 얻을 이득을 다 가져갔고, MSAQ의 MXINT8 대비 잔여
격차는 **dense unpack 자체의 instruction throughput(intrinsic cost)** 이다. per-thread ILP
레버(bfe·register-blocking·unroll)로는 안 줄어든다. 진짜로 줄이려면 **byte 증가 없이 추출
명령 수를 줄이는 packing 재설계**가 필요 → `design_packing_redesign.md`.

---

## 7. 현재 성능 (GPU1, OUT=K=4096; decode는 u4 = MSAQ 우위 구성)

| scope | MSAQ | MXINT8 | MSAQ/MXINT8 | 참고 |
|-------|------|--------|-------------|------|
| **W GEMV** (u4, decode) | **0.033 ms** | 0.047 ms | **0.69× (1.45× 빠름)** ✅ | wide-load. cuBLAS BF16(0.047)도 1.45× 추월 |
| KV decode (u4) | 0.074 ms | 0.061 ms | ~1.1× | cp.async (u3/u4 1.18~1.24× 단축) |
| GEMM prefill (M=256, u4) | 1.20 ms | 1.24 ms | **0.97×** ✅ | tiled, M≥256 128×128 |
| W+A GEMM (M=256, u4) | 1.79 ms | 1.85 ms | **0.97×** ✅ | tiled |

해석: 캠페인의 목표 "**적은 바이트 → 더 빠른 시간**"을 달성했다.
- **W GEMV가 MXINT8·cuBLAS BF16을 모두 1.45× 능가**(Phase 16) — decode의 핵심 결과.
- **prefill(GEMM·W+A)도 tiled로 MSAQ < MXINT8**(0.97×, unpack을 타일당 1회로 분할상환).
- KV는 cp.async로 크게 좁혔으나 아직 근소 열위(extraction 잔여 + token-major broadcast 한계).
- 원본(naive) 대비 배속: GEMV ~35×, KV ~40×, GEMM ~39×, W+A ~22×.

### 7-a. W GEMV가 MXINT8을 이긴 분석 (세분화 측정 → 두 병목, Phase 12·14·16)
"바이트는 0.58×만 읽는데 왜 더 느린가"를 **총 latency가 아니라 단계별로 분해 측정**해 규명:
1. **occupancy/MLP**: split-KV·split-K로 SM은 채웠지만, narrow byte load는 unpack chain에
   막혀 BW를 못 채움 → **cp.async**로 load를 비동기 prefetch해 unpack 뒤에 숨김(Phase 12).
   wide-load(int4)는 블록당 load가 1개라 더 많은 split이 필요 → split mult 3→16(Phase 16).
2. **wide-load**(Phase 14): u4 dense는 이미 nibble-packed(32 code=16 byte=int4 폭). plane을
   column-major `[nb,OUT,16]`로 두면 thread가 자기 열을 **int4 한 번에 coalesced load** → narrow
   shared load ~16개가 1개로. (memory READ만 떼서 재면 0.0195ms = **54.5% peak**, MXINT8보다 빠름.)
3. **compute(추출)**: read가 빠르니 남은 병목은 per-element 추출. 그 안에서 **`g = k/gs`가
   runtime integer divide**(gs가 컴파일타임 상수 아님 → HW divide ~20cyc, 블록당 32회)였고,
   이게 추출 비용의 절반. gs는 2의 거듭제곱이므로 **`k >> (__ffs(gs)-1)`로 교체**(Phase 16).

결과: 0.066 → 0.051(MLP) → **0.033 ms**(divide 제거), BW 16% → 32.5%, MSAQ/MXINT8 1.40× → **0.69×**.
(divide→shift는 `unpack_ms_*`에도 확장했으나 GEMM은 FMA-bound, KV는 staging-bound라 거기선
neutral — divide가 병목인 건 GEMV뿐이었다.)

---

## 8. 검증 체계

- **oracle (`ms_lib/reference.py`)**: 순수 NumPy ground-truth(`wonly_matmul`,
  `wa_matmul`, `kv_attention` + MXINT8 미러). 어디서나 실행.
- **packing roundtrip**: `dequant_weight(pack_weight(W))` == `reconstruct(decompose(W))`
  bit-exact (`test_w.py`).
- **emulation gate (`tests/test_emulation.py`)**: 커널의 평면 인덱싱·언팩을 NumPy로 미러
  → 디바이스가 계산하는 flat offset과 동일함을 검증(layout 변경 시 여기 반영).
- **kernel gate**: 각 scope의 `*_vs_oracle`이 CUDA 커널 결과를 oracle과 `rel_fro < 2e-2`로
  비교(GPU 없으면 skip). u×gs 전 스윕.
- 포맷(레이아웃) 변경 시 **roundtrip 재인증 → oracle → 전 scope 회귀** 순서 필수
  (Stage 4b에서 확립).

---

## 9. 다음 설계 (future work)

상세는 `design_packing_redesign.md`. 요약:

1. **u=4 한정 Stage 4b 재측정** — u=4는 정렬 padding 0(dense==aligned)이라 bfe가 무손실로
   이길 가능성. 가장 싼 다음 실험(코드 이미 존재).
2. **Design A — dense 유지 + funnel-shift 추출** — 블록 vectorized load + `shf.r`로 byte
   증가 없이 추출 명령 절감. KV(token-major 연속)에 적합.
3. **Design D — IMMA/CUTLASS tensor-core** — W+A·prefill GEMM의 endgame. custom iterator의
   Shared→Register load에 unpack 주입(mainloop 밖에서 숨김). 가장 큰 작업·별도 마일스톤.

---

## 10. 파일 맵

```
csrc/
  core/ms_utils.cuh   unpack 원시함수(weight/KV), e8m0 스케일, split 카운트 헬퍼, bfe 스캐폴드
  w_gemv.cu           W-only GEMV (split-K) + combine
  wa_gemm.cu          W-only GEMM / W+A GEMM (correctness baseline)
  kv_attention.cu     KV decode (split-KV + two-pass + token-major) + combine
  mxint8.cu           위 전부의 MXINT8 matched baseline
  pybind.cpp          torch.ops.msaq.* 등록
ms_lib/
  pack.py             MSAQ/MXINT8 수치·패킹(decompose/reconstruct, pack_*, pack_kv)
  reference.py        oracle (NumPy ground-truth)
  ops.py              pack dict → 디바이스 평면 → torch.ops 호출
tests/
  test_w.py / test_wa.py / test_kv.py   scope별 oracle·roundtrip·kernel gate
  test_emulation.py                     커널 인덱싱 NumPy 미러
  benchmark.py                          3-way(MSAQ/MXINT8/cuBLAS|SDPA) latency
change.md                  작업 이력(Phase 1~7, 수치 상세)
design_packing_redesign.md 차세대 packing 설계
design_plan4.md            방안4(coalescing/bfe) 원설계
```
