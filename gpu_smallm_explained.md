# small-M GEMM, 타일, split-K — GPU를 모르는 사람을 위한 설명

> 이 문서는 두 가지 역할을 한다.
> 1. small-M prefill GEMM에서 `wonly_gemm_tc`(기본) vs skinny split-K의 차이와, 거기서 나온 8~11× 측정 결과 기록.
> 2. **설명 난이도의 "gold standard" 예시** — `~/.claude/CLAUDE.md`가 이 문서를 가리킨다. 비전문가에게
>    기술 개념을 설명할 때 이 톤(비유 → 왜 → 어떻게 → 한 줄 요약, 용어는 첫 등장에서 표로 정의)을 따른다.

---

## 0. 측정 결과 요약 (weight를 GPU에 상주시킨 실전 조건)

W-only GEMM, OUT=K=4096, u3/gs16, RTX PRO 4000 Blackwell. weight를 GPU에 한 번만 올려둔 상태(= 실제 추론).

| M | OLD (wmma 기본) | NEW (skinny+split-K) | 속도차 |
|---|---|---|---|
| 8 | 308 µs | 28 µs | **10.9×** |
| 16 | 325 | 30 | 10.7× |
| 32 | 357 | 36 | 9.8× |
| 64 | 422 | 55 | 7.7× |
| 128 | 443 | (skinny 미지원) | — |
| 192 | 459 | (skinny 미지원) | — |

핵심: **작은 M(≤64)에서 기본 경로는 GPU를 거의 놀려서 8~11× 손해.** skinny split-K가 이를 회복. M≥128은
기본 wmma가 최적이라 그대로 둠.

### 커널에 반영됨 (kernel-level dispatch, 재빌드 완료)

라우팅을 **커널 host(`csrc/wa_gemm.cu::wonly_gemm_tc_cuda`)** 안에 넣었다 — 기본 op
`torch.ops.msaq.wonly_gemm_tc`가 스스로 small-M을 skinny로 분기. 따라서 Python wrapper든 C++ 직접 호출이든
모든 경로가 자동으로 혜택을 본다. (`u3/gs16`·`u2/gs8`만 특수화, M≤64만; 그 외엔 wmma로 fall-through.
`MS_NO_SKINNY=1`로 비활성화 가능. Python 쪽 중복 분기는 제거 — 단일 소스.)

이전(강제 wmma) vs 적용(자동) — 같은 op, weight 상주, 정확도 동시 검증:

| M | BEFORE (wmma) | AFTER (auto) | speedup | rel-err |
|---|---|---|---|---|
| 8 | 308 µs | 28 µs | **10.87×** | 3.8e-5 |
| 16 | 324 | 30 | 10.66× | 8.5e-5 |
| 32 | 356 | 36 | 9.80× | 6.8e-5 |
| 64 | 422 | 54 | 7.85× | 6.5e-5 |
| 128 | 443 | 443 | 1.00× | 0 (동일 경로) |
| 192 | 458 | 458 | 1.00× | 0 |

rel-err은 bf16 수준(≤8.5e-5)으로 결과 동일, M≥128은 경로 불변(회귀 0).

### W+A 경로에도 동일 적용 (`wa_gemm_cm_cuda` → `wa_gemm_fused_imma_cuda`)

W+A(weight+activation)도 같은 small-M 문제가 있어 동일 라우팅을 넣었다. 기본 op `wa_gemm_cm`이 M≤64를
fused split-K IMMA 커널로 분기. (W-only와 쌍둥이 구조: Stage-0 activation quant은 공통, Stage-1 GEMM만
split-K로 가속.)

| M | BEFORE (imma) | AFTER (auto) | speedup | rel-err |
|---|---|---|---|---|
| 8 | 328 µs | 56 µs | **5.81×** | 0 |
| 16 | 329 | 63 | 5.27× | 0 |
| 32 | 348 | 87 | 4.00× | 0 |
| 64 | 387 | 153 | 2.53× | 0 |
| 128 | 450 | 448 | 1.00× | 0 |
| 192 | 514 | 513 | 1.00× | 0 |

W+A 속도차(5.8×)가 W-only(10.9×)보다 작은 이유: W+A는 **Stage-0 활성 양자화 패스가 before/after 공통 고정비**라
split-K가 GEMM 단계만 줄이고 이 패스는 그대로 → 작은 M일수록 그 고정비가 바닥을 만든다. rel-err은 정수 IMMA라
**정확히 0**(bit-identical).

### MXINT8도 matched 라우팅 (`mxint8_gemm_cm_cuda` → `mxint8_gemm_fused_skinny_cuda`)

공정 비교를 위해 MXINT8 **W-only**에도 같은 라우팅 추가(rel-err ≤9.6e-5). MXINT8 **W+A는 skinny twin이 없어
매칭 불가** — 정직하게 W-only만. 결과가 **순위를 뒤집는다** (W-only, weight 상주):

| M | MSAQ before | MSAQ after | MX before | MX after | 결론 before → after |
|---|---|---|---|---|---|
| 8 | 308 | **28** | 139 | 49 | MX 2.2× 빠름 → **MSAQ 1.7× 빠름** |
| 16 | 325 | **30** | 140 | 51 | MX 2.3× → **MSAQ 1.7×** |
| 32 | 357 | **36** | 157 | 55 | MX 2.3× → **MSAQ 1.5×** |
| 64 | 422 | **54** | 192 | 71 | MX 2.2× → **MSAQ 1.3×** |
| 128 | 442 | 442 | 212 | 212 | MX 2.1× → MX 2.1× (둘 다 미라우팅) |

weight 크기: MSAQ(u3/gs16) **11.5MB** vs MXINT8 **17.3MB** (1.5×).

**핵심 — 라우팅이 결론을 뒤집는다:**
- **before(GPU 미충전)**: MXINT8이 ~2.2× 빨라 *보인다*. small-M에선 대역폭 병목이 아직 아니고, **MSAQ의 weight 언팩
  (upper/shared → int8 복원)이 노출 비용**으로 드러나는데 MXINT8은 생 int8이라 언팩이 없어서다 → footprint가 작아도
  MSAQ가 느려 보임.
- **after(split-K로 둘 다 충전)**: 둘 다 weight 읽기에 대역폭 병목이 되어 **MSAQ의 1.5× 작은 footprint가 그대로 발현**
  → MSAQ가 1.3~1.7× 빠름(byte 비율과 정합). **이게 진짜 그림.**
- 즉 **matched 라우팅 없이 small-M을 비교하면 잘못된 결론(MXINT8 > MSAQ)**이 난다. 라우팅을 양쪽에 넣어야 apples-to-apples.

**남은 갭(M=65~256)** → 아래에서 해결됨.

### 후속 1 — M=65~256 갭 메움 (skinny MT 상한 확장)

skinny가 M≤64(MT≤4)만 커버하던 것을, 커널이 `m<M`을 가드하는 점을 이용해 MT를 지원 집합
{1,2,3,4,6,8,12,16}으로 **올림(round-up)** 하여 확장. NT=128은 accumulator fragment를 2배로 늘려
레지스터 압박이 크므로 MT>4에선 NT=64로 강제. 경로별 상한은 **측정된 crossover**로 설정:

| 경로 | 라우팅 상한 | 이유 |
|---|---|---|
| MSAQ W-only | **M≤256** | 전 구간 승리(M=256서 1.81×) |
| MSAQ W+A (IMMA) | **M≤128** | shared-mem 한계(MT≤8) |
| MXINT8 W-only | **M≤128** | 언팩 없어 일찍 교차(M=256서 0.71× → 손해) |
| MXINT8 W+A | **M≤96** | 더 일찍 교차(M=128서 0.94×) |

(MXINT8이 더 일찍 교차하는 건 plain int8이라 base 커널이 이미 효율적 → split-K 오버헤드가 일찍 역전.)
각 포맷이 **자기 crossover까지만** 라우팅하므로 어디서도 회귀 없음(경계 밖은 기존 경로로 fallback).

### 후속 2 — MXINT8 W+A 비대칭 해결 (신규 twin 커널)

MXINT8 W+A엔 skinny twin이 아예 없었음. `wa_gemm_fused_imma`(MSAQ)에서 **weight 로드만** plain int8
직접 읽기로 바꾼 신규 커널 `mxint8_wa_gemm_fused_skinny`(+host, +op, +라우팅)를 작성. 정확도 **정확히 0**
(정수 IMMA). 이로써 W-only·W+A 두 scope 모두 MSAQ·MXINT8이 matched.

**최종 matched 비교 (both auto-route to own best, weights resident):**

| M | W-only MSAQ/MX | W+A MSAQ/MX |
|---|---|---|
| 8 | MSAQ 1.72× | MSAQ 1.36× |
| 32 | 1.38× | 1.40× |
| 64 | 1.31× | 1.34× |
| 128 | 1.11× | 1.17× |
| 256 | 0.97× (동률) | — |

모든 M에서 두 포맷이 각자 최적 경로 → 공정 비교. MSAQ가 footprint(0.67×) 우위로 small~중간 M에서
일관되게 앞서고, 큰 M에서 동률로 수렴.

> 주의: `wonly_gemm(p, X)`를 매번 호출하면 weight(11.5MB)를 CPU→GPU로 재업로드(692µs)하므로 end-to-end로는
> 1.4×만 보인다. 실전은 weight 상주라 위 표의 8~11×가 맞는 그림이다.

---

## 1. 우리가 하는 계산이 뭔가

`Y = X @ Wᵀ` — 입력 `X`(M×4096)에 weight `W`(4096×4096)를 곱해 `Y`(M×4096)를 만드는 **행렬 곱**. LLM의
거의 모든 레이어가 이것이다.
- **W는 고정**(학습된 가중치, 11.5MB). 한 번 GPU에 올려두고 계속 재사용.
- **X는 매번 바뀜**(지금 처리 중인 토큰들).

## 2. GPU가 일하는 방식 (공장 비유)

GPU = **워크스테이션 70대짜리 공장**(이 카드는 SM이 70개). 행렬 곱을 하려면 출력 `Y`를 **타일(작은 직사각형
조각)**로 잘라, 타일 하나를 워크스테이션 하나에 맡긴다. 70대가 동시에 자기 타일을 계산 → 병렬.

규칙 둘:
- 워크스테이션이 자기 타일을 계산하려면 **W의 해당 부분을 메모리에서 읽어야** 한다 (이 메모리 읽기가 계산보다
  훨씬 느림 = 진짜 병목).
- 타일이 70개보다 많아야 70대가 다 바쁘다. **타일이 적으면 일부가 논다 = 공장이 놀고 = 느림.**

## 3. M이 뭐고 왜 중요한가

**M = 한 번에 처리하는 입력 행 개수 = 토큰(단어) 개수.**
- **Prefill(프롬프트 읽기)**: 프롬프트 전체를 한꺼번에 → M = 수백~수천 (**큰 M**).
- **Decode(답변 생성)**: 토큰 한 개씩 → **M = 1, 배치/스펙데코드로 몇 개 (아주 작은 M)**.

작은 M이 까다로운 이유: `Y`가 M×4096인데 M=8이면 아주 납작(skinny). **타일이 몇 개 안 나와 공장이 텅텅 빈다.**
게다가 W(11.5MB)는 곱셈에 통째로 필요해 다 읽어야 하는데 계산량은 쥐꼬리(M=8) → **읽기 많고 계산 적은** 최악의
메모리 병목.

## 4. wmma vs sc32/sc64/sc128 — "어떻게 계산하느냐"

전부 같은 행렬 곱이지만 **계산 하드웨어 + 타일 크기**가 다르다:

| 이름 | 쓰는 하드웨어 | 출력 타일 | 한 줄 |
|---|---|---|---|
| **wmma** | **텐서코어**(행렬곱 전용 유닛) | 64×64 | 기본값. 큰 M에서 압도적 |
| **sc32** | 일반 코어(CUDA core) | 32×32 | 타일 작음 |
| **sc64** | 일반 코어 | 64×64 | 타일 중간 |
| **sc128** | 일반 코어 | 128×128 | 타일 큼 |

- **텐서코어(wmma)** = 16×16 블록을 한 방에 곱하는 가속기. 일반 코어가 곱셈을 하나씩 한다면 텐서코어는 통째로 →
  보통 가장 빠르고 **기본값**.
- **타일 크기** = 워크스테이션 하나가 맡는 출력 조각. 크면 W 재사용이 좋아 큰 M에서 유리하지만, **작은 M에선 큰
  타일이 빈 행을 잔뜩 들고 낭비**.
- (별도 측정) occupancy(워크스테이션을 얼마나 빽빽이 채우나)는 sc32가 100%로 최고인데도 제일 느렸다 → **빽빽함이
  아니라 텐서코어 처리량 + 메모리 재사용이 속도를 정한다.**

## 5. split-K와 sk1~sk16 — "큰 합을 쪼개 더 많은 일꾼에게"

`Y = X @ Wᵀ` 속은 **길이 4096짜리 덧셈**(곱한 걸 다 더하기)이다. 이 4096이 **K 차원**(곱해서 합치는 축).
- **보통**: 워크스테이션 하나가 K=4096 전체를 혼자 더한다.
- **split-K**: K를 **S조각으로 쪼개** 각 조각을 다른 워크스테이션에 주고, 마지막에 부분합을 더하는 싸구려 단계 추가.

작은 M에서 약발 받는 이유: 작은 M은 타일(=일거리)이 적어 공장이 비는데, **split-K로 일거리를 S배 늘리면** 놀던
워크스테이션이 다 일하고 → W 읽기를 70대가 나눠 해 → 메모리 대역을 꽉 채운다.

**sk1, sk2, …, sk16 = K를 몇 조각으로 쪼갰나.** 많이 쪼갤수록 일꾼이 늘어 빨라지지만 **sk8에서 최적**, sk16은
부분합 뒷정리 비용이 이득을 넘어 살짝 도로 느려진다. (커널 자동 기본값이 마침 ≈8조각.)

## 6. skinny vs wmma(tc) — 핵심 한 줄씩

- **wonly_gemm_tc(wmma 기본)**: **큰 M(prefill)용.** 64×64 타일 + 텐서코어. 행 많을 땐 최고. M 작으면 64행 타일에
  빈 행 가득 + 타일 수 적어 공장이 비어 **8~11× 손해**.
- **skinny split-K**: **작은 M(decode)용.** 타일을 16행으로 줄여 낭비 감소 + split-K로 일거리 8배 → 공장 꽉 채움.
  M≤64 압승.

라우팅 = **"M≤64면 skinny, 아니면 wmma"** 자동 분기. (skinny는 u3/gs16·u2/gs8만 특수화, M≤64만 지원.)

## 7. 한 줄 요약

**M** = 동시에 처리하는 토큰 수. **타일(wmma/sc*)** = 출력을 쪼갠 조각 + 계산 하드웨어. **split-K(sk*)** = 합산
축을 몇 조각으로 나눠 일꾼을 늘리나. 작은 M은 **공장이 비는 게** 문제라 split-K가 약이고, 큰 M은 텐서코어
wmma가 최고. occupancy(빽빽함) ≠ 속도.

## 재현
- 커널-only 스윕: `CUDA_VISIBLE_DEVICES=1 python precision/smallm_tile_splitk_bench.py`
- occupancy: `MS_GEMM_SCALAR=1 MS_TILE_CFG=<0|1|5> PROBE_M=2048 ncu -k regex:wonly_gemm_tiled_cm -c 1 --section Occupancy --section LaunchStats python precision/ncu_tile_probe.py`
