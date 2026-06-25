# Kernel ver.260625 — 현재 production 커널 지도 (어느 파일·커밋·줄을 보면 되는가)

이 문서는 **"지금 E2E가 실제로 호출하는 커널이 어느 파일의 어느 커널인가"**를 한 곳에 모은
지도다. `csrc/`에는 실험·폐기된 커널이 ~50개 섞여 있어서, 그중 **production 경로만** 골라
링크·줄번호·최신 커밋·한국어 설명을 단다.

- 권위 있는 라우팅 = [`tests/harness_batchsweep.py`](tests/harness_batchsweep.py)의 `QLinear.gemm`(prefill) /
  `QLinear.gemv`·`QLinear.fwd`(decode) / `KVCache.attend`(어텐션). 이 문서의 "어느 커널?"은 전부 거기서 역추적했다.
- 이전 버전: [kernel_ver1.md](kernel_ver1.md) 7종 골격 → [kernel_ver2.md](kernel_ver2.md) 포맷/와이드로드
  → [kernel_ver3.md](kernel_ver3.md) `(u,gs)` 컴파일타임 특수화. **ver.260625 = ver.3 위에 올린 두 개의 decode 메모리패턴 수정**
  (오늘, §4). 골격·포맷·특수화는 그대로다.

빌드: `python setup.py build_ext --inplace` → `cp build/lib*/ms_cuda*.so .`
측정: `MS_FAST=1 python tests/e2e_perscope2.py` → [`tests/harness_perscope_results2.md`](tests/harness_perscope_results2.md)

---

## 1. Production 경로 지도

| 단계 | 조건 | op (torch.ops.msaq) | host (C++) | **production 커널** | 파일:줄 |
|---|---|---|---|---|---|
| **Prefill GEMM** | 모든 M | `ms_dequant_bf16` + cuBLAS | `ms_dequant_bf16_cuda` | `ms_dequant_bf16_kernel` → bf16 [K,OUT], 그다음 `X@W` (cuBLAS) | [w_gemv.cu:759](csrc/w_gemv.cu#L759) / host [:797](csrc/w_gemv.cu#L797) |
| **Decode W-only** | B=1 | `wonly_gemv_wide` | `wonly_gemv_wide_cuda` | `wonly_gemv_wide_uspec` | [w_gemv.cu:295](csrc/w_gemv.cu#L295) / host [:858](csrc/w_gemv.cu#L858) |
| | B=2..15 | `wonly_gemv_batched` | `wonly_gemv_batched_cuda` | **`wonly_gemv_batched_uspec`** ⭐ | [w_gemv.cu:434](csrc/w_gemv.cu#L434) / host [:915](csrc/w_gemv.cu#L915) |
| | B≥16 | `ms_dequant_bf16` + cuBLAS | (= prefill 경로) | `ms_dequant_bf16_kernel` + cuBLAS | 위와 동일 |
| **Decode W+A** | B=1 | `wa_gemv` | `wa_gemv_cuda` | `wa_gemv_wide_uspec` (+ `quant_act`) | [w_gemv.cu:672](csrc/w_gemv.cu#L672) / host [:975](csrc/w_gemv.cu#L975) |
| | B=2..15 | `wa_gemv_batched` | `wa_gemv_batched_cuda` | **`wa_gemv_batched_uspec`** ⭐ (+ `quant_act`) | [w_gemv.cu:703](csrc/w_gemv.cu#L703) / host [:1032](csrc/w_gemv.cu#L1032) |
| | B≥16 | `ms_dequant_bf16` + cuBLAS | (= prefill 경로) | 위와 동일 | |
| **활성화 양자화** | W+A 전용 | `quant_act` | `ms_launch_quant_act_msaq` | `quant_act_msaq_kernel` | [wa_gemm.cu:81](csrc/wa_gemm.cu#L81) / host [:792](csrc/wa_gemm.cu#L792) |
| **KV write (prefill 일괄)** | prefill 1회 | `kv_write` | `kv_write_cuda` | `kv_write_kernel` (전체 [B,Hkv,Lp,hd] K/V를 MSAQ 평면으로 양자화 저장) | [kv_attention.cu:1599](csrc/kv_attention.cu#L1599) / host [:2061](csrc/kv_attention.cu#L2061) |
| **KV append (decode 매 step)** | 매 step | `kv_append` / `kv_append_rot` | — | `kv_append_kernel` / `kv_append_rot_kernel` (새 K/V 1칸을 양자화해 캐시에 덧씀; `_rot`은 K 회전 융합) | [kv_attention.cu:1636](csrc/kv_attention.cu#L1636) / [:1668](csrc/kv_attention.cu#L1668) |
| **KV read = 어텐션 연산** | 모든 B | `kv_decode_attention_batched` | `kv_decode_attention_batched_cuda` | `kv_decode_wide_kernel` `<U4,VPACK,U_,GS_>` (양자화 K/V 읽어 Q·Kᵀ, P·V; **Q는 bf16**) | [kv_attention.cu:336](csrc/kv_attention.cu#L336) / host [:1887](csrc/kv_attention.cu#L1887) |
| **온라인 회전** | (옵션) | `hadamard_rotate` | — | `hadamard_rotate_kernel` | [rotate.cu:40](csrc/rotate.cu#L40) |

⭐ = **오늘(260625) 수정된 커널** (§4).

**MXINT8 baseline (공정 비교용, 대칭 구조)** — [`csrc/mxint8.cu`](csrc/mxint8.cu):

| 단계 | production 커널 | 줄 |
|---|---|---|
| Prefill / B≥16 decode | `mxint8_dequant_bf16_kernel` + cuBLAS | [:183](csrc/mxint8.cu#L183) |
| Decode W-only B=1 | `mxint8_gemv_wide_kernel` | [:196](csrc/mxint8.cu#L196) |
| Decode W-only B=2..15 | **`mxint8_gemv_batched_wide_kernel`** ⭐ | [:216](csrc/mxint8.cu#L216) |
| Decode W+A B=1 | `mxint8_wa_gemv_wide_kernel` | [:255](csrc/mxint8.cu#L255) |
| Decode W+A B=2..15 | **`mxint8_wa_gemv_batched_wide_kernel`** ⭐ | [:277](csrc/mxint8.cu#L277) |
| KV write(prefill) / append(decode) | `mxint8_kv_write_kernel` / `mxint8_kv_append_kernel` | [:865](csrc/mxint8.cu#L865) / [:891](csrc/mxint8.cu#L891) |
| KV read = 어텐션 연산 | `mxint8_kv_split_kernel` | [:750](csrc/mxint8.cu#L750) |

공유 헬퍼 ([`csrc/core/ms_utils.cuh`](csrc/core/ms_utils.cuh)): `e8m0_to_scale` [:87](csrc/core/ms_utils.cuh#L87),
`stream_block_uspec<U_,GS_,F>` (레지스터 상주 스트리밍 언팩) [:153](csrc/core/ms_utils.cuh#L153),
`e8m0_exp_from_amax` / `decompose_ms_block_int` (활성화·가중치 분해) [:249](csrc/core/ms_utils.cuh#L249).

---

## 2. 각 production 커널 — 한국어 설명

### Prefill: `ms_dequant_bf16_kernel` + cuBLAS
가중치를 **한 번** bf16 [K,OUT]로 언팩(메모리-bound, 코얼레스 store, grid.y=NB로 풀 점유)한 뒤
**cuBLAS bf16 GEMM**에 넘긴다. prefill은 intensity≈1024 ≫ ridge 76 = **compute-bound**라 바이트
절감이 무의미하고, 가중치 read는 전체의 7%뿐. 그래서 "양자화 weight를 bf16으로 풀어 cuBLAS에
태우는" 게 최선 — fused WMMA(언팩을 MMA에 융합)는 MMA를 굶겨 4× 느렸다. **bf16과 타이**가
이 영역의 천장이다. (자세히: [perf_analysis_principle.md](perf_analysis_principle.md) worked example.)

### Decode B=1: `wonly_gemv_wide_uspec` / `wa_gemv_wide_uspec`
한 토큰(M=1) decode는 intensity≈1 ≪ ridge = **memory-bound**. 가중치 바이트가 그대로 시간 →
양자화가 빛나는 곳. 스레드 1개가 출력 열 1개를 맡아, column-major 평면을 **와이드 로드**(인접 열이
coalesce)하고 `(u,gs)` 컴파일타임 특수화된 `stream_block_uspec`로 32 코드를 **레지스터 상주**
언팩(가변 shift→상수, local-mem spill 제거). K축은 splitK로 쪼개 SM을 채운다. bf16 대비 **0.5–0.6×**
(B=1에서 항상 승).

### ⭐ Decode B=2..15 W-only: `wonly_gemv_batched_uspec` (오늘 수정)
B=8도 여전히 memory-bound(intensity=B≪76)인데, **이전엔 bf16에 졌다.** ncu로 보니 출력 열마다
스레드가 `x[m,kk]`를 global에서 재로드 → **L1 87%, DRAM 17%** = 활성화가 병목이었다.
**수정: `[MR][BLOCK]` 활성화 타일을 K-블록당 shared에 한 번만 적재**(블록의 128 스레드가
broadcast read). L1 붕괴 → **M=8에서 81µs→40µs, bf16(46) 추월.** (§4)

### ⭐ Decode B=2..15 W+A: `wa_gemv_batched_uspec` (오늘 수정) + `quant_act`
W+A = 활성화도 int8 양자화. 이전엔 int8 dot(`idot[MR]`)+블록 스케일 fold라 **70µs**(W-only float
40µs보다 1.8× 느림: Ampere에서 IMAD<FFMA, int8 shared read, `idot[MR]`+`acc[MR]` 2× 누산기로
점유율 제한). **핵심: decode는 memory-bound라 int8 dot의 이점이 0이고,
`(qa·qw)·sa·sw == (qa·sa)·(qw·sw)`로 수학적으로 동일.** 그래서 활성화 스케일 `sa`를 staged
활성화에 접어넣어(`As=qx·sa`, float) **W-only의 float MAC를 그대로** 돌린다 → GEMV 70→31µs,
전체(quant_act 13µs 포함) 84→44µs로 bf16 추월. (§4)

### `quant_act_msaq_kernel`
W+A의 활성화 전처리. bf16 `x[M,K]`를 블록(32)당 E8M0 스케일 `sa_exp` + int8 워드 `qx`로 분해.
메모리-bound, ~13µs. (decode에선 위 GEMV가 이 값을 dequant해 float로 쓴다.)

### Decode B≥16: `ms_dequant_bf16` + cuBLAS
B≥16이면 **bf16 cuBLAS가 텐서코어로 memory-bound를 거의 천장까지** 친다(M=32 ≈ 46µs). 양자화
cuda-core GEMV는 MAC 연산이 M과 함께 커져(M=32 점유율 23%, `acc[MR]` 레지스터 압박) 진다.
그래서 B≥16은 prefill과 같은 dequant+cuBLAS(≈101µs, bf16과 타이/약간 손해)를 쓴다.
**교차점: shared-activation GEMV는 ~M=10까지 bf16 승, ~M=20까지 dequant+cuBLAS 승.**

### 어텐션 3종: `kv_write` (prefill) / `kv_append` (decode) / `kv_decode_wide_kernel` (read)
KV는 세 커널이 나눠 담당한다: **write**(`kv_write_kernel`, prefill에서 전체 K/V 일괄 양자화),
**append**(`kv_append_kernel`, decode 매 step 새 K/V 1칸 양자화 저장; `_rot`은 K 회전 융합),
**read**(`kv_decode_wide_kernel`, 아래). read 커널이 양자화 K/V를 와이드 로드 + `(u,gs)` 특수화 언팩으로 Q·Kᵀ / P·V를 푼다.
Q는 bf16으로 읽는다 → **AA(어텐션 활성화 양자화)를 켜도 decode latency = KV-decode**(AA는 정확도
비용이지 latency 비용이 아님; S6=S5). 배치가 커질수록 KV 바이트 절감이 커져 가장 잘 이기는 경로
(B=32에서 0.65×). (KV 특수화: [kernel_ver3.md](kernel_ver3.md), 커밋 `57d6212`.)

---

## 3. 현재 E2E 결과 (한 줄)
[`tests/harness_perscope_results2.md`](tests/harness_perscope_results2.md) — (1024,128), B=8 total mq/bf:
**6개 scope 전부 bf16 승** (S1 0.88, S2 0.89, S3 0.76, S4 0.67, S5/S6 0.68). B=1 전부 승,
B≥16은 weight scope 타이/손해 + KV scope 승.

---

## 4. 오늘(260625) 변경 요약 — 두 커밋
1. **`6fbba37` shared-activation batched GEMV** — 활성화를 shared에 staging(재로드 제거). 4개 msaq
   batched 경로 + MXINT8 대칭. W-only B=8: 1.12→0.87.
2. **`b83ba6b` W+A dequant-in-staging float GEMV** — int8 dot→float MAC(`As=qx·sa`). msaq+MXINT8
   wa batched. W+A B=8: 1.12→0.89, S5/S6: 0.93→0.68.

진단은 둘 다 **ncu**로 했다: L1 87–89% vs DRAM <20% → "활성화 재로드가 병목"을 확정 후 수정.
원칙: [perf_analysis_principle.md](perf_analysis_principle.md).

---

## 5. Production이 아닌 커널 (참고만 — 호출 안 됨)
- `wonly_gemv_tc_kernel` / `wonly_gemm_wmma_pipe` (텐서코어 fused) — **decode에선 짐**(M=8에서
  텐서코어 4% idle, fragment가 레지스터 잡아 점유율 32%). [wa_gemm.cu:580/498]. documented-negative.
- `wa_imma` / `mxint8_wa_imma` (block-scaled INT8 IMMA prefill) — IMMA 5.6% (블록 스케일 flush),
  cuBLAS와 비호환. [wa_gemm.cu:651].
- **AA low-bit×low-bit 어텐션 matmul** (= Q·Kᵀ, P·V를 *양자화끼리* 연산) — `qk_wmma_kernel`
  [kv_attention.cu:1218](csrc/kv_attention.cu#L1218), `pv_wmma_kernel` [:1096](csrc/kv_attention.cu#L1096)
  (+ `_mx`/`qk_scalar` 변형). **production 경로엔 없음.** 호출처는 `tests/aa_kernel_bench.py`,
  `tests/shared_prefix_attn_bench.py`뿐. 이유: ① **decode** — production `kv_decode_wide_kernel`은 Q를
  bf16으로 읽는다(decode는 KV 바이트에 memory-bound, Q는 [B,Hq,hd]로 극소). Q를 양자화(low×low)해도
  latency 이득 0 → **S6(AA)가 S5와 같은 커널**을 쓴다(AA = 정확도 비용, latency 비용 아님). ② **prefill** —
  qk_wmma/pv_wmma(low×low)는 bf16 SDPA에 2–2.7× 짐 → prefill 어텐션도 bf16 SDPA. 즉 low×low 어텐션은
  **만들어 측정했으나 documented-negative**라 E2E에서 빠졌다. (측정: [precision/aa_attn_results.md](precision/aa_attn_results.md),
  [change.md](change.md) Phase 49.)
- 초기 버전들: `wonly_gemv_splitk_kernel`(ver.1), `wonly_gemv_cpasync_kernel`, `*_tiled`,
  `*_wmma`(비-pipe), `kv_decode_cpasync/warpT/gqa` — ver.2/3 와이드로드·특수화로 대체됨.
- generic fallback(`*_kernel` 비-uspec): 특수화 안 된 `(u,gs)`용. E2E(u2/u3, gs8/16)는 전부 uspec 사용.
- **`wa_gemv_batched_fused_uspec`** (`MS_WA_FUSED=1`, [w_gemv.cu](csrc/w_gemv.cu)) — quant_act를 GEMV
  staging에 완전 융합(활성화를 in-kernel fake-quant). **documented-negative, 기본 OFF**: split 대비 4–9%
  *느림*(M=8 46 vs 44µs). quant_act는 [M,K]당 1회 연산인데, output-column 블록(≈OUT/128≈32개)마다
  fake-quant를 재실행해 그 중복 연산이 제거한 13µs보다 크다. → split(quant_act 1회 + dequant-in-staging
  float MAC = `wa_gemv_batched_uspec`)이 기본·최적.

전체 연혁: [change.md](change.md). 설계 원리: [packing_explained.md](packing_explained.md),
[compile_time_optimization.md](compile_time_optimization.md).
