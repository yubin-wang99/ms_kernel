# Change log — KV decode occupancy fix (split-KV / Flash-Decoding)

## Why
Decode KV attention was launching **one block per head** (grid = H = 8). On the
RTX 3090 (82 SMs) that is a hard launch-config ceiling: ≤10% of SMs ever do
work, so the kernel ran at ~0.2% of peak HBM bandwidth regardless of per-SM
tuning. The MSAQ byte savings could not show up in wall-clock because the
machine was idle, not because unpack is fundamentally limiting.

## What changed
Split the **key axis** into tiles of `KV_TILE = 256` keys (Flash-Decoding):

- grid `H` → `(H, S)` with `S = ceil(Lk/KV_TILE)`. Block count `8 → 8·S`
  (e.g. Lk=4096 → S=16 → **128 blocks** > 82 SMs, machine fills).
- **Phase 1 `*_split_kernel`**: each `(h, s)` block runs online softmax over its
  key tile only and writes an *unnormalized* partial `(acc, m, l)` to scratch
  (`part_o [H,S,D]`, `part_m [H,S]`, `part_l [H,S]`, fp32).
- **Phase 2 `*_combine_kernel`**: one block per head merges the S partials with
  the standard online-softmax rescale (`m_g = max_s m_s`, weight each tile by
  `exp(m_s − m_g)`), normalizes, writes bf16 `out [H,D]`.

Applied **identically to both** `csrc/kv_attention.cu` (MSAQ) and the
`mxint8_kv_*` path in `csrc/mxint8.cu`, so the benchmark stays a matched
optimization comparison (only the K/V read differs: sub-byte unpack vs direct
int8). No layout/packing change; dequant path untouched.

Files: `csrc/kv_attention.cu`, `csrc/mxint8.cu` (pybind/ops/schemas unchanged —
launcher-internal only).

## Correctness
`pytest tests/test_kv.py tests/test_emulation.py` → all pass (oracle gate
`test_kv_decode_attention_vs_oracle` + emulation gate, full u×gs sweep).

## Result (benchmark, H=8, Lk=4096, D=128, u3gs8)
| kernel | before | after | speedup |
|--------|--------|-------|---------|
| MSAQ   | 2.962 ms | **0.239 ms** | 12.4× |
| MXINT8 | 3.002 ms | 0.221 ms | 13.6× |
| SDPA (ref) | 0.182 ms | 0.182 ms | — |

- Achieved HBM BW (MSAQ): ~0.21% → **~2.6%** of peak (≈12× higher).
- MSAQ/SDPA: 16.3× → **1.3×** (now within 1.3× of tuned SDPA).
- **MSAQ/MXINT8: 0.99× → 1.08×.** Occupancy no longer hides anything; the
  residual is now genuinely unpack-overhead vs byte-savings. The ratio is
  **above 1.0**, so by the agreed gate the next lever is the **unpack
  optimization** (vectorized load + `bfe`), not more parallelism.

---

# Phase 2 — occupancy roadmap (in-flight bytes / Little's law)

## 방안 1 — split factor sized from the live SM count ✅ (implemented)
`csrc/core/ms_utils.cuh::kv_split_count()` queries `cudaDevAttrMultiProcessorCount`
and picks S so grid blocks `H*S ≈ mult·#SM`. `mult` is `MS_KV_SPLIT_MULT` (env,
default **3**); `MIN_TILE=32` floors keys/tile. Both KV kernels now take a runtime
`key_tile` (was the fixed `KV_TILE=256`). Default mult=3 → 248 blocks (3×82).

Result (Lk=4096, default mult=3): MSAQ **0.239 → 0.173 ms**, MXINT8 0.221 → 0.149 ms.
MSAQ/SDPA **1.3× → 0.9×** (MSAQ now *beats* tuned SDPA). MSAQ/MXINT8 = 1.16×.

## 방안 2 — register pressure: NOT the limiter (measured, no change made)
`nvcc -Xptxas -v`: MSAQ split kernel **40 reg**, MXINT8 **28 reg**, both **0 spill**.
On sm_86 a 128-thread block at 40 reg/thread → ~51 warps available, capped at the
48-warp hardware max → MSAQ can already reach full occupancy. Register is not the
ceiling here, so `__launch_bounds__` was deliberately **not** added.

## 방안 5 — atomic contention: already avoided (by Phase-1 design)
The split path writes per-tile partials to separate buffers + a 2nd reduction
(combine) kernel — no `atomicAdd` on shared outputs. Nothing to do for KV.

## What the sweep actually exposed (the honest caveat)
Sweeping `MS_KV_SPLIT_MULT` 1→12: MSAQ scales with split (best ~0.124 ms / 48.7
GB/s ≈ 5% peak at mult≈8); MXINT8 *cliffs* ~2× at mult≥4. Cause = **L2 residency**,
not an HBM crossover: RTX 3090 L2 = 6 MB; the benchmark re-reads the SAME KV each
iter; MSAQ footprint **6.03 MB ≈ L2 (fits)**, MXINT8 **8.65 MB > L2 (thrashes once
concurrency rises)**. Confirmed by Lk=16384 (both ≫ L2): the small-Lk behaviour
inverts (MSAQ mult=8 → 2.3 ms / 10 GB/s). So the sub-1.0 ratios at high mult are a
benchmark/L2 artifact, NOT proof of bandwidth-saving wins.

Underneath occupancy, the real wall is now the **uncoalesced strided load**: token-
innermost `[..,L]` layout + one-thread-per-head-dim means a warp reads L-strided
addresses → ~32× transaction amplification (why achieved BW stays ≤5% peak and
collapses at large Lk). That is exactly **방안 4's** target.

## Default chosen
`mult=3` (not the L2-optimal 8): robust across Lk — high mult only wins when KV
fits L2 and *hurts* large-Lk (DRAM row thrashing from too many concurrent
uncoalesced streams).

## Next (re-prioritized by the data)
- **방안 4 — vectorized 128-bit load + register-aligned repack** is now the critical
  path (occupancy is solved; access pattern is the bottleneck). Needs the
  pack-layout fork + roundtrip re-cert before flipping `MSAQ_USE_BFE`.
- 방안 3 (ILP/register-blocking) is secondary — limited by the per-key
  `__syncthreads()` reduction barrier (the QK reduction the header flags for a
  tensor-core rewrite).
- GEMV split-K still pending (separate kernel).

---

# Phase 3 — 방안 4 Stage 4a: token-major transpose (coalescing) ✅ 구현 완료

## 변경
KV byte plane을 **BYTES innermost (token-major)**로 transpose: `[H,nb,BYTES,L]`
→ `[H,nb,L,BYTES]`. 고정 key에서 한 warp의 32 head_dim thread가 **연속 바이트
span**을 읽어 coalesce. bit-packing·code는 불변(주소 순서만).
- `ms_lib/pack.py`: `pack_kv` upper/shared, `pack_kv_mxint8` qweight를 transpose
  (MXINT8 baseline도 동일 적용 → matched 유지). `_per`/oracle은 불변.
- `ms_utils.cuh::unpack_ms_kv_elem`: 주소를 `byteIdx*L + key` → `key*UB + byteIdx`
  로 (straddle 바이트도 인접). base_u/base_h는 값 동일(UB*L == L*UB)이라 불변.
- `mxint8.cu`: `kq[..+k*Lk+j]` → `kq[..+j*BLOCK+k]`.
- `tests/test_emulation.py`: KV mirror에 token-major `_unpack_kv` 추가(새 주소 검증).

## 재인증
`pytest tests/test_kv.py tests/test_emulation.py` → 전부 통과 (oracle + emulation,
full u×gs). code가 같아 oracle 불변, emulation은 새 주소를 mirror.

## 결과
| config | 지표 | 4a 이전 | **4a 이후** |
|--------|------|---------|------------|
| Lk=4096 (L2 잔류) | MSAQ | 0.173 ms | **0.125 ms** |
| | MXINT8 | 0.149 ms | 0.127 ms |
| | MSAQ/MXINT8 | 1.16× | **0.98× ✅** |
| | MSAQ/SDPA | 0.9× | **0.7×** |
| Lk=16384 (≫L2, 순수 HBM) | MSAQ/MXINT8 | 1.09× | **1.004×** |
| | MSAQ achieved BW | 35.6 GB/s | **49.3 GB/s** |

- coalescing으로 두 커널 모두 ~30~40% 단축. **L2 잔류 크기에서 MSAQ가 matched
  MXINT8를 진짜로 추월(0.98×)** — MXINT8도 같이 빨라졌는데 MSAQ가 이김 = L2 artifact 아님.
- 순수 HBM(Lk=16384)에선 **parity(1.004×)**: MSAQ가 0.70× 바이트를 0.69× 대역폭으로
  읽어 바이트 절약 ↔ extraction 오버헤드가 정확히 상쇄. → **Stage 4b(bfe)**가 이걸 < 1로 넘김.

## 남은 ceiling (다음)
- achieved BW가 아직 49~71 GB/s(peak의 ~5~8%)로 낮음. coalescing은 풀렸으나 inner
  loop의 **key별 `__syncthreads()` reduction barrier**가 MLP를 막아 HBM을 못 채움(=방안3).
- **Stage 4b** (upper word-align + `bfe` + `int4` load): 순수 HBM에서 MSAQ<MXINT8 확정용.
- 둘(방안3 reduction 재작성 + Stage 4b)이 이제 공동 다음 레버. 설계: `design_plan4.md`.

---

# Phase 4 — 방안 3: barrier-light two-pass reduction ✅ 구현 완료

## 변경 (`kv_attention.cu`, `mxint8.cu` split kernel 재작성)
기존: thread=head_dim, **key마다** block-wide 트리 reduction(키당 ~8 `__syncthreads`)
→ inner loop이 barrier-bound, MLP 없음. 새 구조(two-pass):
- **Pass 1 (scores)**: 한 **warp가 한 key**의 q·K dot을 `__shfl`로 reduce → block
  barrier 없음, warp들이 다른 key 동시 처리(MLP). 4a coalescing 유지(같은 key).
- **Pass 2 (output)**: thread d가 `out[d]=Σ_kk p_kk·V[d,kk]`를 key 루프로 누적 →
  cross-thread reduction 없음, key 루프 내 load 겹침(MLP).
- barrier는 **키당 ~8 → chunk당 ~2**. scores를 `KV_CHUNK=128` chunk로 shared에
  버퍼링(메모리 bound), chunk간 online-softmax 결합. unpack 총량 불변(중복 없음).
- MXINT8도 동일 구조(matched). 재인증: `test_kv` + `test_emulation` 전부 통과.

## 결과 — 두 커널 다 빨라졌지만 ratio는 벌어짐 (중요한 진단)
| config | 지표 | 4a 이후 | **방안3 이후** |
|--------|------|---------|---------------|
| Lk=4096 (L2) | MSAQ | 0.125 ms | **0.0946 ms** |
| | MXINT8 | 0.127 ms | **0.0632 ms** |
| | MSAQ/MXINT8 | 0.98× | **1.50×** |
| | MSAQ/SDPA | 0.7× | **0.5×** |
| Lk=16384 (HBM) mult=3 | MSAQ BW | 49 GB/s | **65 GB/s** |
| | MXINT8 BW | 71 GB/s | **156 GB/s** |
| | MSAQ/MXINT8 | 1.004× | **1.68×** |
| Lk=16384 mult=8 | MXINT8 BW | (cliff 2.3ms) | **290 GB/s (31% peak)** |

## 진단 — 이게 핵심 소득
- barrier 제거로 **두 커널 다 절대 시간 단축**(MSAQ도 0.125→0.0946). MLP/대역폭 상승.
- **부가 효과**: 큰 Lk에서 mult=8이 무너지던 over-split cliff가 **사라짐**(barrier가
  증폭하던 것). 이제 큰 Lk에서 높은 mult가 안전·유리(MXINT8 290 GB/s).
- 그러나 **ratio는 악화**(0.98→1.50). 이유: barrier가 사라지자 MXINT8(가벼운 int8
  read)은 **memory-bound**가 되어 대역폭만큼 질주(→290 GB/s)하는데, MSAQ는 dense bit
  unpack(shift/or/straddle/sign-extend)의 **ALU/instruction에 묶여(extraction-bound)**
  대역폭을 못 올림(65→97 GB/s에서 포화). 둘 다 빨라졌지만 MXINT8 headroom이 더 컸음.
- **결론**: MSAQ의 남은 병목은 occupancy도 coalescing도 아닌 **extraction 연산량**으로
  확정됨. 이는 정확히 **Stage 4b(bfe + vectorized load)**의 타깃. 방안3이 barrier를
  걷어내 준 덕에, 이제 4b의 extraction 감소가 **시간에 직접** 드러난다(전엔 barrier에 가려짐).
  → 방안3은 유지(절대 속도·MLP·진단 가치), **다음은 Stage 4b가 크로스오버의 결정 레버**.

## 기본값
`mult=3` 유지(안전). 단 방안3로 cliff가 사라져 큰 Lk에선 높은 mult가 유리 —
재튜닝 여지 있음(`MS_KV_SPLIT_MULT`).

---

# Phase 5 — 방안 4 Stage 4b: register-aligned + bfe ❌ 시도 후 REVERT (negative result)

## 시도
KV upper/shared code를 32-bit word 정렬(floor(32/width) codes/word, no straddle)로
재배치 → device에서 aligned `uint32` load + `bfe.s32` 단일 추출. `ms_lib/pack.py`
(`_pack_codes_word_aligned`), `ms_utils.cuh`(`unpack_ms_kv_elem_aligned`),
`kv_attention.cu`, emulation mirror까지 구현하고 **전 테스트 통과**(roundtrip 정상).

## 결과 — 더 느려짐
| config | 방안3 (dense 4a) | Stage 4b (aligned+bfe) |
|--------|------------------|------------------------|
| Lk=4096 MSAQ | **0.0946 ms** | 0.1519 ms (느림) |
| Lk=4096 ratio | **1.50×** | 2.46× |
| Lk=16384 MSAQ | **0.372 ms (65 GB/s)** | 0.599 ms (51 GB/s) |
| MSAQ bytes | **24.1 MB** | 30.4 MB (+26% padding) |

## 왜 실패했나 — 가설(extraction-bound)이 반증됨
- 정렬 padding으로 바이트 +26%(u3 upper 20→24B, shared 2→4B).
- 그러나 시간은 +60%로 **바이트 증가(+26%)보다 더** 늘어남 → `bfe` 자체가 도움 안 됨.
  register도 40으로 동일.
- 결론: KV decode의 binding constraint는 **extraction instruction 수가 아니라
  load→use latency / MLP 부족**. `bfe`는 load→bfe→fma로 dependency chain 길이가
  그대로라 latency를 못 줄이고, 바이트만 늘렸다. → 방안3 분석에서 "extraction-bound"로
  본 것은 부정확. 실제는 **latency-bound**.

## 조치
**Stage 4b 전체 revert** → best-known(방안3 + dense token-major 4a)로 복귀.
복귀 후 재확인: MSAQ **0.0923 ms**, ratio 1.49, 전 테스트 통과. pack.py/ms_utils.cuh/
kv_attention.cu/test_emulation 모두 dense로 되돌림(주석에 negative result 기록).

## 다음 방향 (재설정)
latency-bound이므로 진짜 레버는 **MLP/ILP 증가** — thread/warp가 **독립적인 load를
더 많이 동시에 띄우기**(thread당 여러 key/element prefetch, K-loop unroll로 load를
issue 후 나중에 소비). bfe(연산량)도 byte(정렬)도 아닌 **outstanding load 수**가 관건.
(방안 3의 ILP 사촌. pass1/pass2의 짧은 루프를 register-blocking으로 깊게.)

---

# Phase 6 — GEMV split-K ✅ 구현 완료

## 변경 (`w_gemv.cu`, `mxint8.cu` GEMV)
W-only GEMV는 "thread 1개가 출력 열 o의 K=4096 전체를 혼자 reduce" → block =
ceil(OUT/128)=32개뿐(82 SM 중 32만). **K축(reduction)을 splitK로 쪼개** grid를
(base_blocks, splitK)로:
- 각 (o, sp) block이 자기 K-slice만 부분합 → `partial[splitK, OUT]`에 기록.
- `gemv_combine_kernel`이 partial을 **선형 합산**(KV와 달리 softmax 보정 불필요).
  atomic 없음(KV 교훈). splitK는 `ms::gemv_splitk_count`가 SM 수로 결정
  (`MS_GEMV_SPLITK_MULT` env, 기본 3), NB로 cap.
- MXINT8도 동일 적용(matched). GEMV plane은 out-innermost라 **이미 coalesced**(4a 불필요).
- 재인증: `test_w` + `test_emulation` 전부 통과(split은 math 불변, 선형합).

## 결과 (OUT=4096, K=4096, u3gs8)
| 지표 | 원본 | split-K (기본) | best (mult sweep) |
|------|------|---------------|-------------------|
| MSAQ | 1.162 ms | **0.305 ms (3.8×)** | ~0.184 ms (mult=16) |
| MXINT8 | 0.640 ms | **0.052 ms (12.4×)** | ~0.038 ms (mult=4) |
| cuBLAS | 0.046 ms | 0.047 ms | — |
| MSAQ/MXINT8 | 1.82× | 5.89× | ~3.5–4.5× |

## 진단 (KV와 동일)
- occupancy가 GEMV에서도 1차 천장이었음 → 두 커널 다 큰 절대 단축.
- **MXINT8 GEMV가 cuBLAS에 도달**(0.038~0.052 vs 0.047) = 사실상 memory-bound 최적.
- MSAQ는 unpack(extraction/latency)에 묶여 5.89×로 벌어짐 — KV와 같은 구조적 원인.
  → 남은 레버도 동일: **MLP/ILP**(thread당 여러 열 register-blocking, K-loop 깊은 unroll).
- 기본 mult=3 유지(MSAQ는 mult 8~16에서 더 좋지만 MXINT8은 mult 4가 최적 — 타협값).

---

# Phase 7 — MLP/ILP register-blocking ❌ 시도 후 REVERT (negative result)

latency-bound 가설에 따라 "thread당 독립 load를 더 많이 띄우는" register-blocking을
KV·GEMV에 시도. **둘 다 효과 없음 → revert.**

## GEMV (COLS columns/thread)
thread가 COLS개 출력 열을 맡아 per-k에 COLS개 독립 weight-unpack 발행 + x[k] 재사용.
- 같은 세션 공정 비교(머신이 그날 ~2× 혼잡 — cuBLAS 0.047→0.096): MSAQ
  COLS=1 **0.46~0.49 ms**, COLS=2 0.50, COLS=4 0.55 → **neutral~악화**.
- 원인: register-heavy unpack가 COLS배 → **114 reg(COLS=4)**, occupancy 48→**17 warp**
  붕괴. latency-bound 커널에서 warp 수가 ILP보다 중요 → 손해. → COLS=1로 revert.

## KV (pass-2 key 루프 `#pragma unroll 4`)
독립 V-unpack을 in-flight로 — MSAQ 0.189~0.194 ms vs baseline 0.1915 → **noise(무효과)**.
컴파일러가 이미 ILP를 뽑았거나 latency가 아닌 throughput 한계. → revert.

## 결론 (4b·register-blocking·unroll = 3연속 negative)
occupancy/구조 최적화(split, two-pass, coalescing)가 **얻을 수 있는 이득을 다 가져갔고**,
MSAQ의 MXINT8 대비 잔여 격차는 **unpack 자체의 instruction throughput(=intrinsic cost)**.
per-thread ILP(bfe·register-blocking·unroll)로는 줄지 않음 — register만 더 먹어 occupancy를
깎거나 컴파일러가 이미 한 일. 진짜로 더 줄이려면 **unpack instruction 수 자체를 (byte 증가
없이) 줄이는 packing 재설계**가 필요(연구성 과제). 현 단계 커널 최적화는 여기서 수렴.

---

# 한글 요약 — 방안 1~5 정리 (효과 / 구현 / 남은 과제)

(아래 표는 Phase 1~5를 모두 반영한 최종 상태. 상세는 위 각 Phase 참조.)

| 방안 | 효과 | 구현 반영 | 상태 |
|------|------|-----------|------|
| 1. SM 2~4배 split | **매우 큼 (1차 효과)** | ✅ 반영 | 완료 |
| 2. register 압력 감소 | **효과 없음 (이미 한계 아님)** | ❌ (측정만) | 불필요로 판명 |
| 3. barrier 제거 (two-pass) | **큼 (절대 속도↑·MLP↑)** | ✅ 반영 | 완료 |
| 4a. token-major transpose (coalescing) | **큼** | ✅ 반영 | 완료 |
| 4b. register-align + bfe | **음수(더 느림)** | ❌ revert | negative result |
| 5. atomic contention 제거 | 해당 없음 | ✅ 이미 그렇게 설계됨 | 완료 |

## 누적 결과 (KV decode, Lk=4096, u3gs8)
원본 **2.962 ms** → 최종 **0.092 ms (~32배)**. MSAQ/SDPA 16.3× → **0.5×(SDPA의 2배 빠름)**.
MSAQ/MXINT8 = 1.49× (아직 매칭 베이스라인보단 느림 — 아래 참조).

## 효과가 있어 구현에 넣은 것
- **방안 1 (동적 split):** 실제 SM 수로 `H·S ≈ mult·#SM`. occupancy ≤10% 해소(1차 효과).
- **방안 4a (token-major transpose):** plane을 `[H,nb,L,BYTES]`로 → 고정 key에서 warp가
  연속 바이트 읽어 coalesced. 트랜잭션 ~32×→~2×. bit-packing 불변이라 재인증 쉬움.
- **방안 3 (two-pass barrier 제거):** key당 `__syncthreads` 트리 reduction →
  Pass1(warp-per-key, `__shfl`) + Pass2(thread=head_dim, key 루프). barrier 키당 ~8 →
  chunk당 ~2. 두 커널 다 빨라지고 MLP↑(MXINT8 290 GB/s까지). 큰 Lk over-split cliff도 제거.
- **방안 5 (이미 설계됨):** 별도 partial buffer + 2차 combine → atomic 없음.

## 측정 결과 안 넣은 것 / 되돌린 것
- **방안 2 (불필요):** MSAQ 40 reg / MXINT8 28 reg, spill 0 → 48-warp 상한에 먼저 걸려
  register가 occupancy를 안 깎음. `__launch_bounds__` 불필요.
- **방안 4b (negative result, revert):** word-align + `bfe` 구현·인증했으나 **더 느림**
  (Lk=4096 0.095→0.152 ms). padding +26% bytes + bfe 무효과. binding constraint가
  extraction이 아니라 **load latency/MLP**임이 반증됨. 전부 dense로 revert.

## MSAQ가 아직 MXINT8보다 느린 이유 + 다음 방향
방안 3로 barrier가 사라지자 MXINT8은 memory-bound로 질주(290 GB/s)하는데 MSAQ는
**load→unpack→use latency chain**에 묶여 65~97 GB/s에서 포화. 4b(bfe)로 chain 길이가
안 줄어 실패. → 진짜 레버는 **MLP/ILP 증가**: thread/warp가 독립적인 load를 더 많이
동시에 띄우도록 register-blocking(thread당 여러 key/element prefetch, K-loop 깊은 unroll).
bfe도 byte도 아닌 **outstanding load 수**가 관건.

## 게이트 해석 정정 (참고)
split을 키워 나온 MSAQ/MXINT8 < 1.0은 **대역폭 절약이 아니라 L2 잔류 효과**였음
(RTX 3090 L2 = 6 MB; MSAQ 6 MB는 들어가고 MXINT8 8.65 MB는 thrash). Lk=16384(둘 다
L2 초과)에서 역전 확인. 기본값은 안정적인 **mult=3**(단 방안3로 cliff가 사라져 큰 Lk엔
높은 mult도 유리).
