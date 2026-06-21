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

# Phase 8 — u=4 nibble-bfe (weight) + batch sweep ❌ 효과 없음 (negative)

## 동기
Stage 4b(KV)는 정렬 padding(+26%) 때문에 졌다. u=4(wbits=4)는 **dense가 이미 nibble
정렬**(byte당 코드 2개, straddle 없음, padding 0)이라, **포맷·레이아웃 변경 없이** 추출만
`bfe.s32`로 바꿔 "padding 0에서 bfe가 이기는가"를 깨끗이 테스트.

## 구현 + A/B (GPU1, u=4 gs=8, OUT=K=4096)
`unpack_ms_weight_elem`에 u==4 nibble-bfe 분기 추가(macro `MSAQ_W_U4_BFE`로 on/off).
결과 동일(재인증 통과). bfe **ON vs OFF**:
- `wonly_gemm`(M=1..256): **완전히 동일**(M=1 1.826 vs 1.826, M=256 46.6 vs 46.6).
- `wonly_gemv`(split-K, M=1): 0.2500 → 0.2459 ms (**~1.6%, noise 수준**). MSAQ/MX 5.2×.

## 진단 (중요 — 병목 재규명)
bfe는 u=4에서 **같은 바이트를 로드**하고 ALU(shift/mask/sign-extend)만 1 op로 줄인다.
그게 무효 → **병목은 extraction ALU가 아니라 byte LOAD 그 자체**(MSAQ는 upper+shared
2회 로드 vs MXINT8 int8 1회). bfe로는 못 줄인다. → 진짜 레버는 **로드 명령 수 감소**
(연속 레이아웃 + vectorized load), bfe가 아님. (Phase 5·7과 일관: ALU/ILP 미세최적은 무효.)

## Batch sweep (u=4, wonly_gemm vs mxint8_gemm vs cuBLAS, GPU1)
| M | cuBLAS | MXINT8 | MSAQ | MSAQ/MX | MX/cuBLAS |
|---|--------|--------|------|---------|-----------|
| 1 | 0.047 | 0.631 | 1.83 | 2.9× | 13× |
| 16 | 0.045 | 0.639 | 3.64 | 5.7× | 14× |
| 64 | 0.051 | 1.87 | 11.98 | 6.4× | 37× |
| 256 | 0.150 | 6.18 | 46.6 | 7.6× | 41× |

- M이 커질수록 MSAQ/MX가 2.9×→7.6×로 **벌어짐**: `wonly_gemm`이 **배치 행마다 weight를
  재-unpack**(재사용 없음)하기 때문. MXINT8도 cuBLAS 대비 13×→41×로 벌어짐(naive GEMM).
- prefill(큰 M)의 진짜 레버는 **tiled GEMM**(weight 타일을 한 번 unpack→shared/레지스터에
  올려 M행·출력타일에 재사용) = tensor-core/CUTLASS(Design D). bfe도 split-K도 아님.

## 조치
u4-bfe 분기·macro·setup hook **전부 revert**(clean), 재빌드·재인증 통과.

## 인프라 메모
이 세션 내내 **GPU0가 외부 작업에 점유**(24/24 GB, OOM·~2× 저하)돼 측정이 불안정했음.
**GPU1은 비어 있어** `CUDA_VISIBLE_DEVICES=1`로 돌리면 깨끗·정상속도. 이후 측정은 GPU1 권장.

---

# Phase 9 — tiled GEMM (prefill) ✅ 구현 완료 — **첫 MSAQ < MXINT8 달성**

## 변경 (`wa_gemm.cu` wonly_gemm, `mxint8.cu` mxint8_gemm — matched)
naive GEMM은 thread 1개가 (m,o) 출력 1개 → **배치 행마다 weight를 재-unpack**(재사용 0)
→ unpack 비용이 M에 비례. → **shared-memory tiled GEMM**으로 교체:
- block이 `TBM×TBN = 64×64` 출력 타일 담당, 256 threads, thread당 `RTM×RTN=4×4` 레지스터 타일.
- K-loop이 `TBK=32`(=MSAQ 블록) 단위로 shared에 stage: `As[k][m]`(X), `Bs[k][o]`(=dequant W).
  **unpack은 Bs stage에서 단 1회** → 그 타일의 **TBM(64)개 행이 재사용**(unpack 64× 분할상환).
- MXINT8도 동일 타일링(B-tile만 직접 int8 read). 재인증: 전 117 테스트 통과.

## 결과 (GPU1, u=4 gs=8, OUT=K=4096)
| M | MSAQ 이전 | **MSAQ 이후** | MXINT8 이후 | **MSAQ/MX** |
|---|----------|--------------|-------------|-------------|
| 64 | 11.98 ms | **0.56 ms** | 0.58 ms | **0.96×** |
| 128 | 23.19 ms | **0.98 ms** | 1.08 ms | **0.91×** |
| 256 | 46.64 ms | **1.91 ms** | 2.12 ms | **0.90×** |

- **MSAQ가 모든 배치에서 MXINT8을 추월(0.84~0.97×)** — 캠페인 전체의 목표였던
  "적은 바이트 → 시간 우위" 첫 달성. M=256에서 ~24× 절대 가속.
- 원리: unpack이 타일당 1회로 분할상환되자 커널이 memory-bound로 넘어가고, MSAQ가
  weight를 적게 읽어(u4: 4.75 vs 8.25 b/elem = 0.58×) 그만큼 빨라짐.

## W+A GEMM도 동일 타일링 적용 (`wa_gemm.cu` wa_gemm, `mxint8.cu` mxint8_wa)
W+A는 activation도 (row,block)별 MXINT8 양자화 후 정수 dot. **sa·sw가 둘 다 2의
거듭제곱**이라 `As=qx·sa`, `Bs=qw·sw`로 fold하면 정수 dot과 bit-exact → W-only 타일에
activation 양자화 staging만 추가(amax→sa→quant)하면 끝. 재인증 통과.
- 결과(u=4): MSAQ가 모든 배치에서 MXINT8 추월(**0.94~0.98×**), M=256 3.07 vs 3.26 ms.
- canonical M=512 u2gs8: 39.6 → **6.11 ms (~6.5×)**, ratio 1.08×(u=2라 근소 열위).

## u 의존성 (중요)
- canonical M=512: **u3gs8 = 1.08×**(MSAQ 근소 열위), **u4 = 0.90×**(MSAQ 우위).
- 이유: u=4는 바이트가 더 적고(0.58×) unpack도 nibble(straddle 없음)이라 둘 다 유리.
  u=3은 바이트 0.68× + straddle unpack이라 거의 parity. → **고-u(작은 upper폭=적은 바이트
  +nibble)일수록 MSAQ 우위**(W-only·W+A 공통). 단 절대 속도는 둘 다 큰 폭 향상.

## 의의 / 다음
- Phase 5·7·8의 micro-opt(bfe·ILP)는 다 무효였는데, **구조(tiled로 unpack 분할상환)** 가
  결정적이었다. decode(M=1)는 split-K, prefill(M큼)은 tiled — scope별 구조가 답.
- 다음 절대 성능 레버는 **tensor-core(IMMA/CUTLASS)** — 현재 FP32 CUDA core라 cuBLAS의
  ~12~14× 뒤(matched MSAQ<MXINT8은 이미 달성). M=1 decode용 tiled 분기는 split-K GEMV가
  이미 더 빠름(0.25 vs 0.55 ms).

---

# Phase 10 — tile-size 조사 + 최적 타일 도출 (tensor-core 전 단계)

## 방법
`wonly_gemm`/`wa_gemm`(+MXINT8 matched)을 `<TBM,TBN,RTM,RTN>`로 **템플릿화**하고
env `MS_TILE_CFG`로 8개 config를 한 빌드로 런타임 sweep (TBK=32 고정). 전 config 정확
(rel_fro 0.0017). GPU1, u=4, OUT=K=4096.

## W-only M-sweep (ms) — 타일별
| M | 64×64 r4×4 | 128×128 r8×8 | 128×128 r8×4 |
|---|-----------|--------------|--------------|
| 16 | **0.65** | 1.38 | 1.20 |
| 64 | **0.57** | 1.20 | 1.20 |
| 128 | **0.98** | 1.22 | 1.21 |
| 256 | 1.93 | **1.22** | 1.20 |
| 512 | 3.38 | **2.20** | 2.25 |
| 1024 | 6.30 | **4.36** | 4.47 |

## 결론 — 최적 타일은 **M-adaptive**
- **M < 256: 64×64** (작은 타일이 block을 더 많이 만들어 SM을 채움). 128×128은
  grid.x=OUT/128=32(<82 SM)라 작은 M에선 block 부족으로 느림.
- **M ≥ 256: 128×128 r8×8** (큰 타일이 reuse↑로 compute-bound 진입, SM도 충분히 참).
- crossover ≈ M=256. → host dispatch에 `(M>=256)?128×128:64×64` 반영(MXINT8도 동일 matched).

## 효과 (adaptive vs 고정 64×64, M=256)
- W-only MSAQ: 1.93 → **1.22 ms (1.6×)**, MSAQ/cuBLAS 12.9 → **8.2×**.
- W+A MSAQ: 3.07 → **1.79 ms (1.7×)**. MSAQ/MXINT8은 모든 M에서 0.91~0.98× 유지.

## 다음
tensor-core(IMMA/CUTLASS)로 cuBLAS의 ~8~12× 격차를 좁히는 단계. W-only=BF16 WMMA,
W+A=INT8 IMMA가 자연스러운 매핑.

---

# Phase 11 — tensor-core (BF16 WMMA) W-only prefill ✅ (opt-in)

## 구현 (`wa_gemm.cu` wonly_gemm_wmma)
FP32 타일과 동일 staging(Bs=dequant(W) bf16, As=X bf16)이되 inner product를
`nvcuda::wmma` 16×16×16 fragment(tensor core)로. 블록 64×64, 4 warp(2×2)가 각
32×32(2×2 frag) 담당. 레이아웃: C[m,o]=Σ_k X[m,k]·Wdq[o,k] → A=As[m][k] row_major,
B=Bs[o][k]를 col_major K×O로 읽어 Wdq와 일치. `MS_TILE_CFG=10`으로 선택(opt-in).

## 결과 (GPU1, u=4) — 정확(rel_fro 0.0017)
| M | cuBLAS | FP32 tile | WMMA | WMMA/cuBLAS | WMMA/FP32 |
|---|--------|-----------|------|-------------|-----------|
| 128 | 0.073 | 0.984 | **0.657** | 9.0× | 0.67× |
| 256 | 0.149 | 1.21 | **0.91** | 6.1× | 0.75× |
| 512 | 0.276 | 2.18 | **1.62** | 5.9× | 0.75× |
| 1024 | 0.550 | 4.37 | **2.80** | 5.1× | 0.64× |

- WMMA가 M≥128에서 FP32 타일보다 **25~36% 빠름**, cuBLAS 대비 8~12× → **5~6×**로 단축.
- 단 이득이 "modest"인 이유: 커널이 **staging/unpack-bound**(64×64 소형 타일, 4 warp,
  async 파이프라인 없음)라 tensor core가 부분적으로 굶음. M=64에선 오히려 약간 느림
  (block/occupancy 부족).

## 현재 상태 / 남은 일
- WMMA는 **opt-in**(default 미변경) — matched 비교(MSAQ vs MXINT8) 보존을 위해 default는
  FP32 타일 유지. WMMA를 default로 올리려면 **MXINT8도 WMMA**(matched)가 선행돼야 함.
- cuBLAS의 ~5×까지 더 좁히려면: **(1) 큰 타일(128×128)+8 warp, (2) `cp.async` double-buffer
  로 staging↔compute overlap, (3) W+A는 INT8 IMMA(`s8` frag, int32 accum, sa·sw epilogue)**.
  이게 CUTLASS-급 작업의 핵심 요소들.

---

# Phase 12 — cp.async로 unpack을 메모리 뒤에 숨김 (W GEMV) ✅ — **드디어 통한 레버**

## 동기 (KVQuant 개념 + "unpack을 메모리 뒤에 숨기기")
KVQuant: KV/weight 로딩이 memory-bound면 dequant가 load 그늘에 숨어 공짜. 단 **전제는
진짜 memory-bound일 때**. 측정상 MXINT8 GEMV=39% peak(그 regime), MSAQ=4.7%(아님 — unpack이
critical path). → MSAQ를 memory-bound로 밀어넣어야 함. bfe/register-blocking/unroll은 다 실패
(각각 ALU-bound 아님 / occupancy 붕괴). **cp.async**는 load를 register 없이 백그라운드로
빼서 둘의 실패 원인을 모두 피함.

## 구현 (`w_gemv.cu` wonly_gemv_cpasync_kernel)
split-K 구조 동일, 단 블록의 packed 바이트(upper/shared/scale, 128열분)를 `cp.async`로
**다음 블록 prefetch ↔ 현재 블록 unpack+accumulate**가 겹치게 double-buffer. 타일을
`[BYTES][128]`로 그대로 staging → unpack은 OUT=128,blk=0으로 `unpack_ms_weight_elem` 재사용.
OUT%128==0에서 default(아니면 scalar fallback). `MS_GEMV_CPASYNC=0`으로 A/B.

## 결과 (GPU1, OUT=K=4096) — 재인증 통과
| u | scalar | cp.async | speedup | BW |
|---|--------|----------|---------|-----|
| u4 | 0.228 ms | **0.128 ms** | **1.78×** | 4.7% → 8.3% |
| u3 | 0.274 ms | **0.129 ms** | **2.12×** | 4.7% → 10.0% |
| u2 | 0.292 ms | **0.130 ms** | **2.26×** | 5.0% → 11.2% |

- **GEMV에서 처음으로 통한 micro-arch 레버.** unpack을 async load 그늘에 숨겨 BW를
  ~2배로(memory-bound 쪽으로). 저-u일수록 숨길 unpack이 많아 이득 큼(2.26× vs 1.78×).
- MSAQ/MXINT8 GEMV 격차 5.2× → **~2.7×**. (MXINT8은 이미 39% memory-bound라 cp.async 불필요.)
- 아직 MXINT8(39%)엔 못 미침 — 남은 격차는 staging overhead + load 폭(1~2 byte)이라,
  다음은 **wide-load packing 재설계**가 BW를 더 끌어올릴 후보.

## 다음
- **KV decode에도 동일 cp.async** 적용(같은 메커니즘, two-pass 커널).
- 그 뒤 wide-load packing(int4 vectorized)로 8~11% → 더 높이기.

---

# Phase 13 — cp.async KV decode ✅ (u3/u4 이득, u2 break-even)

## 구현 (`kv_attention.cu` kv_decode_cpasync_kernel)
two-pass 구조 유지, 단 `CP_CHUNK=64` key씩 K/V의 packed upper 평면(전 NB block)을
`cp.async`로 **다음 chunk prefetch ↔ 현재 chunk unpack(pass1 K + pass2 V)** double-buffer.
token-major라 block당 key-major 연속 → `UB` mult-4로 4-byte cp.async 정렬됨. 작은 shared-code
평면(SB)은 odd-key offset에서 2-mod-4라 cp.async 정렬 불가 → 동기 복사(작아서 무방). scale은
global에서 직접. unpack은 staged shared를 base_u=blk*CP_CHUNK*UB로 재사용.

## 결과 (GPU1, H=8 Lk=4096 D=128) — 재인증 통과
| u | two-pass | cp.async | speedup | vs MXINT8 |
|---|----------|----------|---------|-----------|
| u4 | 0.0873 ms | **0.0738 ms** | 1.18× | 1.29× → **1.09×** |
| u3 | 0.0954 ms | **0.0772 ms** | 1.24× | → 1.14× |
| u2 | 0.1002 ms | 0.1000 ms | 1.00× | 1.47× |

- u3/u4에서 18~24% 단축, MSAQ KV가 MXINT8의 **1.09×(u4)** 까지 근접(near-parity).
- u2는 break-even: UB=24로 staging shared가 커져(~26KB) occupancy가 깎이면서 cp.async 이득 상쇄.
  (regression은 없어 default-on 안전.)
- GEMV(1.8~2.3×)보다 이득이 작은 이유: KV는 애초에 덜 unpack-bound(6~7% vs 4.7%) + sync-copy/
  복잡한 two-pass 오버헤드. 그래도 default로 채택(`MS_KV_CPASYNC=0`로 A/B).

## 위치
- 이제 decode 양쪽(GEMV·KV)에 cp.async 적용 완료. MSAQ가 memory-bound 쪽으로 더 이동.
- 남은 격차: load 폭(1~2 byte). 다음은 **wide-load packing 재설계(int4 vectorized)**.

---

# Phase 14 — wide-load packing 재설계 (u=4 GEMV) ✅ — **큰 도약**

## 진단 (cp.async 이후 병목 재확인)
cp.async가 global load를 숨긴 뒤에도 bfe(ALU 절감)는 **여전히 무효**(0.1284 vs 0.1285ms)
→ 병목은 extraction ALU가 아니라 **shared에서의 narrow per-byte load throughput**(원소당
~16 byte-load). 해법은 load **수**를 줄이는 것 = **wide(int4) load**.

## 구현 (`w_gemv.cu` wonly_gemv_wide_kernel + 새 op + pack/ops dispatch)
u=4는 dense가 이미 nibble-packed(32 code = 16 byte)이고 **16 byte == int4 폭**. weight를
**column-major `[nb, OUT, 16]`** 로 두면(= 기존 u4 plane의 단순 transpose, 비트 재패킹 없음)
thread o가 자기 열의 16 byte를 **한 번의 coalesced int4 load**로 읽고 32 code를 레지스터에서
`bfe`로 추출 → narrow load ~16개가 1개로 붕괴. (연속 열이 16 B 간격 = warp-contiguous → coalesced.)
- `pack_weight`가 u4일 때 `upper_cm`/`shared_cm`(transpose) 추가, `ms_lib.ops.wonly_gemv`가
  u4 → `wonly_gemv_wide`로 라우팅. split-K + 기존 combine 재사용. **u4 전용**(UB=16).
- 재인증: wide vs 기존 max|diff|=**0.0**(bit-identical), test_w + 전 117 테스트 통과.

## 결과 (GPU1, OUT=K=4096, u4)
| | time | BW | 비고 |
|---|------|-----|-----|
| scalar | 0.227 ms | 4.7% | |
| cp.async | 0.141 ms | 7.5% | |
| **WIDE (int4)** | **0.0665 ms** | **16.0%** | cp.async 대비 **2.13×**, scalar 대비 **3.4×** |
| MXINT8 | 0.048 ms | 38.9% | **WIDE/MXINT8 = 1.40×** |

- MSAQ/MXINT8 GEMV 격차: 5.2× → 2.7×(cp.async) → **1.40×(wide)**. achieved BW 4.7% → 16%.
- 가설대로: 병목은 narrow shared-load 수였고 int4가 ~16→1로 붕괴시켜 해결. (wide는 cp.async
  없이도 충분 — coalesced int4가 이미 효율적.)

## 한계 / 다음
- **u=4 전용**(UB=16=int4). u3(20)/u2(24)는 int4 미정렬이라 미적용 → 그쪽은 cp.async 유지.
- MXINT8(39%)엔 아직 1.4× 못 미침 — 추가로 wide+cp.async 결합, 또는 KV wide-load가 후보.

---

# Phase 15 — KV decode wide-load 시도 ❌ REVERT (구조적으로 부적합)

## 시도
KV 평면은 이미 token-major `[H,nb,L,16]`라 u4면 key당 16 byte 연속 → GEMV처럼 int4
load. warp-per-key/thread-per-d 구조라 **한 warp의 32 lane이 같은 key-block int4를
broadcast** load + bfe.

## 결과 (GPU1, u4) — 정확하지만 **더 느림** (revert)
| | time | BW |
|---|------|-----|
| cp.async | 0.0737 ms | 68 GB/s |
| WIDE(int4) | 0.0980 ms | 51 GB/s (**0.75×, 느림**) |
| MXINT8 | 0.0614 ms | — |

## 왜 GEMV와 달리 실패했나 (구조 차이)
- **GEMV wide**: thread당 **서로 다른 열**을 int4 load → warp가 32×16=512 byte의 **distinct
  데이터**를 한 트랜잭션에 가져옴(고효율).
- **KV wide**: warp가 **하나의 key**를 처리 → 32 lane이 **같은 16 byte**를 broadcast → 트랜잭션당
  16 byte의 distinct 데이터뿐(GEMV의 1/32). 메모리 효율 낮아 오히려 느림.
- 결론: KV의 "warp-per-key(모든 lane이 같은 key)" 매핑에선 per-thread wide load가 broadcast가
  돼 무효. KV엔 **cp.async(여러 key를 bulk-coalesced staging)** 가 맞는 구조. → cp.async 유지.

조치: kv_decode_wide_kernel + dispatch 전부 revert. 전 117 테스트 통과. (GEMV wide는 유지.)

---

# Phase 16 — 세분화 측정으로 병목 규명 → **MSAQ GEMV가 MXINT8/BF16 능가** ✅✅

총 latency만 보지 말고 쪼개서 측정하자는 지적에 따라, wide u4 GEMV(Phase 14)를
단계별로 분해 측정해 병목 두 개를 잡았다.

## 진단 (MS_WIDE_MEMCEIL diagnostic 경로 + splitK sweep)
1. **splitK sweep**: 기본 mult=3에선 0.10ms, mult를 키우니 0.045ms (2.2×) → wide는
   블록당 int4 load가 **1개뿐**이라 MLP가 부족했음(narrow 커널은 32 load/block라 mult~3로
   충분). → **MLP-bound (낮은 split에서).**
2. **memory-ceiling 분해**(read만, no bfe/fma/x): weight READ 자체는 **0.0195ms, 54.5%
   peak** — MXINT8(43%)보다도 빠름! 즉 memory/access는 전혀 병목이 아니고 **compute-bound**.
3. **compute 분해**(memceil=2: bfe는 하되 x/fma 생략): bfe+추출이 +29us로 dominant.
   그 안에서 **`g = k/gs`가 runtime divide**(gs가 컴파일타임 상수 아님 → HW 정수 divide
   ~20cyc)였고, 원소당 1회 = 블록당 32 divide. **이게 추출 비용의 절반(~15us).**

## 두 줄 수정
1. wide 커널 splitK default **mult 3 → 16** (`gemv_splitk_count(..., 16)`): MLP 확보.
2. `g = k / gs` → **`g = k >> (__ffs(gs)-1)`** (gs는 2의 거듭제곱): divide 제거.

## 결과 (GPU1, OUT=K=4096, u4) — 전 117 테스트 통과
| 단계 | time | BW | vs MXINT8 |
|------|------|-----|-----------|
| Phase 14 wide (mult=3) | 0.066 ms | 16% | 1.40× |
| + mult=16 (MLP) | 0.051 ms | 21% | 1.10× |
| + divide→shift (compute) | **0.0328 ms** | **32.5%** | **0.69×** |

- **MSAQ GEMV가 MXINT8을 1.45× 추월**(0.69×). cuBLAS BF16(0.0475ms)도 1.45× 추월.
- 캠페인 전체의 목표였던 **"적은 바이트 → 더 빠른 시간"** 을 GEMV에서 결정적으로 달성.
  (memceil 0.0195ms = 0.41× MXINT8 → 이론 상한; 현재 추출 잔여 비용으로 0.69×.)
- "17× vs BF16"은 정정: 그건 우리 naive 원본 대비였음. 이제 BF16 대비 **실측 0.69×(빠름)**.

## 추출 잔여 — vectorized int4 dequant 시도 ❌ (0.41×는 도달 불가, revert)
0.41×(memceil)는 "추출 0"의 **이상적 하한**이라 실제로는 도달 불가. bf16x2 Marlin식
bit-trick(`(nibble & 0x000F000F)|0x43004300` → hsub2)을 시도했으나:
1. **부정확**(rel 3.48): offset-binary 가정이라 우리 **two's-complement** code에 안 맞음
   (`u-8` ≠ signed; u≥8엔 `u-16` 필요). 고치려면 offset-binary **repack**(또 포맷 변경) 필요.
2. 깨진 채로도 **3%만 빠름**(32.2 vs 33.4us): gathered-x + hsub2/reinterpret 오버헤드가
   bfe 절감을 상쇄.
→ divide 제거(Phase 16)가 추출의 *진짜* 비용을 이미 걷어냈고, 남은 bfe는 scalar로 거의
최적. dense two's-complement에서 vectorize는 이득 없음. revert. (실제 한계 ~0.5×는
Marlin식 offset-binary repack 필요 = 별도 큰 작업, 이미 0.70×라 ROI 낮음.)

## divide→shift를 일반 unpack에도 확장 (`ms_utils.cuh`)
`unpack_ms_weight_elem`/`unpack_ms_kv_elem`의 `g = k/gs`도 `k >> (__ffs(gs)-1)`로 교체
(scalar/cp.async GEMV·GEMM·W+A·KV가 전부 사용). 결과: **GEMM·KV는 neutral**(GEMM은 tiled로
unpack이 분할상환돼 FMA-bound, KV cp.async는 staging/memory-bound라 divide가 병목이 아님).
즉 divide가 병목이었던 건 **wide GEMV뿐**. 그래도 strict simplification(divide→shift, 절대
안 느림)이라 정확성·neutral 확인 후 유지. 전 117 테스트 통과.

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

---

# Phase 17 — KV decode 병목 정밀 분해 (GEMV식 memceil) + 정직한 결론

> 목적: GEMV에서 했던 것처럼 KV decode를 **전체 latency가 아니라 memory-access /
> dequant(unpack) / softmax(exp)** 로 쪼개, MXINT8을 못 넘는 *진짜* 병목을 규명.
> 계측: `MS_KV_DIAG` env (커널에 진단 분기 추가) — diag1=메모리 ceiling(스테이징+dot,
> unpack·exp 제외), diag2=+unpack(exp 제외), diag0=full. GPU1, warm-clock 고정 후
> **call마다 MSAQ/MXINT8 교차** 측정(클럭 부스트 착시 제거).

## 계측 방법론 정정 (중요)
이전 phase들의 "MSAQ/MXINT8 < 1.0" 승리는 **측정 착시**였음:
- **클럭 부스트:** idle 후 첫 커널은 클럭 미부스트(cold)라 느리게 찍힘. MXINT8을 cold로,
  MSAQ를 warm으로 재면 MXINT8이 41→68 us로 출렁여 가짜 승리가 나옴. **3s warmup +
  call별 교차 측정**으로 제거하면 MXINT8 = ~36 us(warm) 안정.
- 결론: warm·공정 측정에서 **MSAQ KV decode는 MXINT8보다 ~1.5× 느림** (B=1, u4gs8).

## 분해 결과 (u4gs8, H=8 Lk=4096, warm 교차)
| 구간 | 시간 | 비고 |
|---|---|---|
| diag1 memory-access ceiling | 42–43 us | unpack·exp 0인데도 이미 **MXINT8 full(37)보다 1.17×** |
| diag2 +unpack | +13–15 us (~26%) | |
| diag0 full +exp | +1 us (~2%) | 58 us = **1.57× MXINT8** |

## footprint 스윕 — latency-bound가 아니라 **구조적 메모리 비효율**
H·Lk를 키워 footprint를 5 MB→151 MB(L2 6 MB 한참 초과, 확실한 BW-bound)로 30× 키워도
**비율이 1.49×로 고정**. 즉 작은 footprint·latency 탓이 아님.
- 151 MB에서 재분해: **memory-access = 시간의 100%**, **unpack·exp = 0**
  (BW-bound라 완벽히 숨겨짐).
- 실효 대역폭: **MXINT8 276 GB/s vs MSAQ 104 GB/s = MXINT8의 38%**. 둘 다 peak(936)
  미달 → **순수 메모리 접근 패턴 비효율**.

## 진짜 병목 (확정)
**MSAQ는 바이트를 덜 읽는데(18 vs 32 B/blk = 0.56×) 실효 대역폭이 MXINT8의 38%라
시간이 1.49× 더 걸린다.** unpack도 exp도 아니다 — `unpack_ms_kv_elem_u4`(bfe 특수화)로
unpack을 줄여도 BW-bound에선 애초에 공짜라 무효. 병목은 **메모리 접근 패턴**:
1. **2-plane 분리**(upper 16B + shared 2B): 주소 스트림 2개, 작은 shared(2B) 트랜잭션 비효율.
2. **per-element scatter**: warp-per-key 매핑에서 한 warp가 block당 upper 16B만 distinct로
   읽음(MXINT8는 32B). 트랜잭션당 유효 바이트 절반. (Phase 15 broadcast 문제의 본질.)
3. **cp.async가 오히려 손해**: 이 영역은 BW/latency-bound라 더블버퍼 배리어 오버헤드가
   직접읽기보다 느림 — `MS_KV_CPASYNC=0`(split 직접읽기)가 cp.async보다 ~15–20% 빠름.

## 다음 방안 (fix plan) — GEMV 승리 패턴 이식
핵심: KV 읽기를 **wide·coalesced**로 바꿔 MXINT8 수준의 실효 BW를 확보.
- **(A, 최우선) key-per-thread wide-read 재설계:** Pass1을 warp-per-key가 아니라
  **thread-per-key**로. thread t가 key_t의 D=128원소(=NB×16B)를 `uint4` wide-load로 읽고
  GEMV식 bfe로 unpack. 연속 thread→연속 key→연속 16B → **512B/warp coalesced**
  (GEMV가 0.70×로 이긴 바로 그 패턴). q·K dot은 thread 내 128 madd(=warp-reduce 제거).
- **(B) 2-plane 융합:** upper|shared를 키당 1개 연속 레코드로 패킹(재인증 필요) → 주소
  스트림 1개, shared의 2B 트랜잭션 제거. pack.py 재배치 + roundtrip 재인증.
- **(C) 기본 경로를 직접읽기로:** BW-bound 영역에선 cp.async OFF가 유리. 기본값 재고.
- 보조 계측: `MS_KV_DIAG`(1/2/3) 분기는 커널에 잔존(diag=0 기본, 오버헤드 무시 가능),
  `tests/kv_diag.py`로 재측정 가능.

---

# Phase 18 — 방안 A: key-per-thread wide-read 재설계 ✅✅ — **MSAQ KV가 MXINT8을 ~2배 추월**

## 구현 (`kv_attention.cu` kv_decode_wide_kernel, u=4 전용)
Phase 17 진단(병목 = unpack/exp 아님, **메모리 접근 패턴** — warp-per-key 매핑이 sector당
유효 바이트 절반만 읽어 실효 BW가 MXINT8의 38%)을 그대로 정조준. GEMV-wide(Phase 14/16)
승리 패턴을 KV에 이식:
- **Pass1 = thread-per-key**: thread t가 key cs+t를 통째로 소유. 블록당 K upper 16B를
  **`uint4` 1회 wide-load**(token-major `[H,NB,Lk,UB]`라 key가 16B contiguous → **재패킹 불필요,
  same bytes**), 레지스터에서 32 code를 `bfe`로 추출, q·K dot을 **thread 내부에서 완결**
  (warp-reduce 제거). 연속 thread→연속 key→**512B/warp 완전 coalesced(100% sector util)**.
- **Pass2 = staged V + thread-per-d**: 출력은 key축으로 reduce라 thread-per-key 불가
  (scatter). 대신 청크 V를 **coalesced로 shared에 staging**(연속 복사라 이미 100% util) 후
  thread-per-d로 unpack-accumulate → global V 읽기는 wide, narrow는 on-chip. cp.async와
  달리 **동기 단일 복사**(BW-bound 영역에서 더블버퍼 배리어 오버헤드 없음).
- `MS_KV_WIDE`로 A/B(기본 on, u4 한정). `diag!=0`이면 wide 비활성(kv_diag 도구 보존).
- 재인증: `test_kv`+`test_emulation` 54개 전부 통과. wide vs cp.async **max|diff|=0.0(bit-exact)**.

## 결과 (GPU1, u4 gs8, D=128, warm 교차 측정)
| config | MXINT8 | MSAQ cp.async | **MSAQ WIDE** | wide/cpasync | **wide/MX** |
|--------|--------|---------------|---------------|--------------|-------------|
| Lk=4096 (L2) | 61.2 us / 141 GB/s | 66.7 / 75 (1.09×) | **31.2 / 160** | **0.47×** | **0.51×** |
| Lk=16384 (HBM) | 221 / 156 | 266 / 75 (1.20×) | **105 / 190** | **0.39×** | **0.47×** |
| Lk=16384 H16 (151MB, 순수 HBM) | 425 / 163 | 515 / 77 (1.21×) | **193 / 206** | **0.38×** | **0.45×** |

- **MSAQ KV decode가 MXINT8을 ~2배(0.45~0.51×) 추월** — 캠페인 목표 "적은 바이트 → 빠른 시간"을
  decode KV에서 결정적으로 달성(GEMV에 이어 두 번째 scope). 실효 BW **75 → 160~206 GB/s**.
- 0.45×는 바이트 비율(18/32=0.56×)보다도 **낮음** = MSAQ wide가 MXINT8보다 **실효 BW까지 높음**
  (160~206 vs 141~163): 적은 바이트 + thread-per-key 100% coalescing + 동기 staging의 합.
- 진단이 정확했음을 입증: 병목은 access pattern이었고, K를 wide-direct·V를 coalesced-staged로
  바꾸자 즉시 풀림. (Phase 15 wide 시도가 실패한 건 warp-per-key broadcast였기 때문 — 이번엔
  thread-per-key라 각 lane이 distinct key를 읽어 broadcast가 아님.)

## 측정 오염(hot/cold) 검증 — 아티팩트 아님 (`tests/kv_clock_verify.py`)
Phase 17의 클럭-부스트 착시 우려를 직접 차단:
- **SM 클럭이 시종 부스트·안정**(1905~1980 MHz, idle 0 아님). cold 진입 없음.
- **측정 순서를 뒤집어도 비율 불변**(Lk=16384 wide/MX = 0.473 ↔ 0.473). 클럭 착시면 순서로
  승자가 뒤집혀야 하는데 안 뒤집힘.
- **within-MSAQ(cp.async vs wide, 동일 커널군이라 cross-kernel 클럭 편향 면역)** = 0.39× →
  wide가 진짜 ~2.5배 빠름.
- 단 **Lk=4096은 L2 잔류 캐비엇**: standalone 0.51×지만 mx와 인터리브하면 0.84×(L2 thrash).
  클럭 아님. **클럭·L2 모두 무관한 깨끗한 수치는 Lk=16384(순수 HBM)의 0.47×**.

## u2/u3 확장 — 방안 B(재패킹) 불필요, dense thread-per-key로 충분 ✅
플랜은 "u2/u3는 int4 미정렬이라 방안 B(2-plane 융합)로 정렬해야 wide 가능"이라 봤으나,
**wide 승리의 핵심은 int4 정렬이 아니라 thread-per-key coalescing**임을 확인. dense
token-major도 key가 UB-contiguous라 thread-per-key로 충분 → **재패킹·재인증 없이** 확장.
- 커널을 `template<bool U4>`로: u4는 `uint4`+bfe, u2/u3은 key의 UB(20/24)바이트를 4정렬
  `uint32` word로 레지스터에 적재 후 general straddle 추출. **`if constexpr`로 분기**해
  각 인스턴스가 상대 경로의 레지스터를 안 먹게 함(중요: 런타임 분기로 합쳤더니 죽은 uint4
  경로가 u4 레지스터를 부풀려 occupancy 붕괴 → u4가 105→193us 퇴행. constexpr로 해결).
- 결과(Lk=16384, warm): **세 u 모두 MXINT8 추월** — u2 **0.89×**, u3 **0.86×**, u4 **0.47×**.
  bit-exact(max|diff| u2 3e-8, u3 0.0, u4 2e-6). u2/u3가 u4보다 이득이 작은 건 stride(20/24)
  ≠ load폭이라 per-instruction coalescing이 u4의 uint4만큼 완벽하지 않고 straddle unpack이
  무거워서지만, 바이트가 적어(22/26 vs 32) 여전히 승리.

## 위치 / 다음
- decode 양쪽(GEMV Phase 14·16, KV Phase 18) 모두 wide-read로 **전 u에서 MSAQ<MXINT8 달성**.
- 보조: `tests/kv_wide_bench.py`(warm 교차 A/B), `tests/kv_clock_verify.py`(클럭/순서 검증),
  `tests/kv_diag.py`(memceil 분해).

---

# Phase 19 — W-only GEMV wide-load을 u2/u3로 확장 (laggard 최적화)

## 동기 — 4커널 중 GEMV u2/u3가 유일한 큰 laggard
4커널 × u{2,3,4}를 warm 교차 측정(MSAQ/MXINT8)한 결과:

| 커널 | u2 | u3 | u4 |
|------|----|----|----|
| **W-only GEMV** | **2.12** | **2.11** | 0.56 (wide) |
| W-only GEMM | 1.07 | 1.05 | 0.92 |
| W+A GEMM | 1.04 | 1.02 | 0.93 |
| KV decode (Phase 18) | 0.89 | 0.86 | 0.47 |

→ **GEMV u2/u3가 ~2.1×로 압도적 laggard**. 원인: wide-load(Phase 14)가 **u4 전용**이고
u2/u3는 dense cp.async로 폴백. (측정 함정 주의: `ops.wonly_gemv` 래퍼는 timed-loop 안에서
numpy→GPU 복사를 해 ~48× H2D 아티팩트를 만든다. 평면을 GPU에 1회 올리고 raw op으로 측정해야 함.)

## 구현 — wide kernel을 전 u로 일반화 (재패킹 불필요, KV Phase 18과 동형)
- `pack.py`: `upper_cm`/`shared_cm`(column-major transpose)를 **전 u**에 생성(기존 u4만). 단순
  transpose라 코드/바이트 불변 → oracle·roundtrip 무영향.
- `w_gemv.cu`: `wonly_gemv_wide_kernel`을 `template<bool U4>`로. u4=uint4+bfe(무straddle),
  u2/u3=열의 UB(20/24)B를 4정렬 uint32 word로 레지스터 적재 후 contiguous straddle 추출
  (`unpack_ms_kv_elem` 재사용). `if constexpr`로 분기해 상호 레지스터 점유 차단(Phase 18 교훈).
- `pybind.cpp`/`ops.py`: `wonly_gemv_wide`에 `u` 추가, 래퍼가 **전 u**를 wide로 라우팅.
- 재인증: `test_w`+`test_emulation` 81개 통과. wide vs dense max|diff| u3 0.0, u4 0.0, u2 3.9e-3
  (bf16 누적순서 차이, tol 내).

## 결과 (GPU1, OUT=K=4096, warm)
| u | 이전(dense) | **wide** | 커널 가속 | MSAQ/MX |
|---|------------|---------|----------|---------|
| u2 | 2.12× | **1.54×** | ~1.4× | 1.54 |
| u3 | 2.11× | **1.43×** | ~1.5× | 1.43 |
| u4 | 0.56× | 0.56× | — | **0.56 (이미 승)** |

- **laggard를 2.1× → 1.4~1.5×로 단축**(u2/u3 커널 1.4~1.5× 가속). u4는 이미 crossover(0.56×).

## staging 시도 ❌ (negative, revert) — KV와 달리 GEMV엔 안 맞음
u2/u3가 아직 >1.0이라, KV Phase 18에서 통한 coalesced staging(타일의 128열 바이트를 shared로
연속 적재→narrow 추출)을 시도. **더 느림**(u3 1.43→1.53, u2 1.54→1.52 동률~악화) → revert.
- 이유: KV pass2는 출력이 key축 reduce라 thread-per-d(narrow)가 강제돼 staging이 글로벌
  coalescing을 살렸지만, **GEMV는 thread-per-column으로 reduce가 thread 내부**이고 cm 레이아웃이
  이미 연속 열이라 word-load의 warp가 연속 영역을 덮어 DRAM이 어느 정도 coalesce됨. staging의
  shared 왕복 + blk당 2×`__syncthreads`가 이득을 초과. → word-load 직접 적재가 최적.

## 남은 격차 (정직)
u4는 byteRatio(0.56)에 도달해 memory-bound 최적이지만, u2/u3는 byteRatio(0.69/0.78) 대비 실효
BW가 ~0.5× — **열당 stride(20/24)가 어떤 벡터폭과도 안 맞아 DRAM sector를 덜 채우는 구조적 한계**.
완전 crossover(<1.0)는 **register-aligned repack**(코드를 32-bit word에 정렬, no straddle)이
필요하지만, 이는 Phase 5(KV 4b)에서 **padding +26%로 더 느려진** 전례가 있어 별도 연구·측정이 선행돼야 함.
현 단계는 dense 레이아웃에서의 최선(2.1→1.4×)으로 수렴. (KV가 u2/u3에서도 이긴 건 token-major라
바이트 정렬이 더 유리했기 때문 — 레이아웃 차이.)

## 종합 스코어보드 (RTX 3090, OUT=K=4096, warm, µs) — 4커널 × u, MSAQ/MXINT8/BF16
절대 시간(µs)과 적용된 SOTA 기법, crossover 판정. (KV는 Lk=16384 순수 HBM; Lk=4096은 L2 잔류로 눌림.)

**1. W-only GEMV (decode M=1)** — SOTA: split-K(P6)+wide-load(P14)+mult16·divide→shift(P16)+u4 nibble-bfe / u2·u3 streaming bit-buffer unpack(P20). **전 u에서 MXINT8·BF16 추월 ✅**
| u | MSAQ | MXINT8 | cuBLAS | MSAQ/MX |
|---|------|--------|--------|---------|
| u2 | 40.3 | 48.2 | 46.0 | **0.84 ✅** (1.50→ P20) |
| u3 | 39.5 | 48.2 | 46.0 | **0.82 ✅** (1.43→ P20) |
| u4 | 27.4 | 48.5 | 46.3 | **0.56 ✅** |

**2. W-only GEMM (prefill M=512)** — SOTA: tiled GEMM(P9/10) → **software-pipelined BF16 WMMA + streaming unpack(P23, cfg=11)**. **전 u 추월 ✅**
| u | MSAQ | MXINT8 | cuBLAS | MSAQ/MX |
|---|------|--------|--------|---------|
| u2 | 1589 | 1624 | 283 | **0.98 ✅** (FP32 1.07→ P23) |
| u3 | 1599 | 1633 | 285 | **0.98 ✅** (FP32 1.06→ P23) |
| u4 | 1530 | 1648 | 287 | **0.93 ✅** |

**3. W+A GEMM (prefill M=512)** — SOTA: **2-stage = MSAQ-s 활성화 선-양자화(P27) + pipelined INT8 IMMA(P26)**(weight unpack을 MMA 뒤에 숨김). **전 u 추월 ✅**
| u | MSAQ | MXINT8 | cuBLAS | MSAQ/MX |
|---|------|--------|--------|---------|
| u2 | 2360 | 2763 | 286 | **0.85 ✅** (FP32 1.03→ P26) |
| u3 | 2316 | 2765 | 286 | **0.84 ✅** (FP32 1.02→ P26) |
| u4 | 1968 | 2776 | 287 | **0.71 ✅** |

**4. KV decode (H=8 Lk=16384 D=128, HBM)** — SOTA: split-K flash-decode(P1)+token-major coalescing(4a)+two-pass barrier-light(방안3)+key-per-thread wide-read(P18)
| u | MSAQ | MXINT8 | SDPA | MSAQ/MX |
|---|------|--------|------|---------|
| u2 | 194 | 221 | 608 | **0.88 ✅** |
| u3 | 193 | 223 | 609 | **0.87 ✅** |
| u4 | 109 | 223 | 609 | **0.49 ✅** |

## 원인 분석 (갱신: Phase 20이 GEMV 진단을 정정)
- **GEMV u2/u3 (1.43→0.82–0.84, P20에서 crossover)**: 처음엔 "memory-access-pattern bound"로
  봤으나 **틀렸음**. plane-split(완전 coalesced load)이 0 speedup이고 u2(25B)·u3(22B)가 같은
  40µs(바이트 무관)였던 게 증거 → 실제는 **extraction(straddle unpack)-bound**. streaming
  bit-buffer unpack으로 추출 ALU를 줄여 해결(P20). **register-aligned repack/padding 불필요**.
- **GEMM/W+A u2/u3 (당시 1.02–1.07로 열위) → 이후 전부 crossover**: 그땐 FP32-tiled가 FMA-bound라
  바이트 절약이 안 보였음. **해결책은 tensor-core로 매트멀을 빨라지게 해 memory-bound로 밀고, weight
  unpack을 MMA 뒤에 숨기는 것**: W-only는 pipelined WMMA(P23, 0.93~0.98), **W+A는 활성화를 별도
  선-패스로 빼 GEMM prologue를 weight-unpack 전용으로 만든 2-stage IMMA(P26/27, 0.71~0.85)**.
  분해 결과 B-stage가 바이트 비례(unpack 100% 숨음) → MSAQ가 바이트를 덜 읽어 이김.
- **KV가 u2/u3도 넘는 이유(대조)**: token-major라 key가 contiguous → key-per-thread가 u2/u3도
  coalesced. GEMV out-innermost와 달리 레이아웃이 정렬에 유리.
- **현재 상태**: **4 scope(GEMV·KV·W-only GEMM·W+A) 전부 전 u에서 MXINT8 추월 완료.** 남은 레버는
  cuBLAS 격차(tensor-core 본선 튜닝) + W+A epilogue(~12%)를 CUTLASS fused-fragment로 줄이기.

---

# Phase 20 — W-only GEMV u2/u3 CROSSOVER via streaming bit-buffer unpack ✅✅

목표: Phase 19에서 1.43~1.50×로 남은 GEMV u2/u3를 MXINT8 아래로 (register-aligned repack 시도).

## 진단 — load가 아니라 EXTRACTION이 병목
"register-aligned repack(=정렬된 wide load)"을 먼저 의심했으나 측정으로 반증:
- **plane-split**(upper를 16B int4-plane + tail-plane으로 쪼개 stride==width 완전 coalesced,
  padding 0) 구현 → **0 speedup**(1.58→1.10는 funnel-shift 덕, plane-split 자체는 무효).
- 결정적 증거: **u2(25B)와 u3(22B)가 똑같이 ~40µs** → 시간이 바이트와 무관 = **compute-bound**.
  u4(nibble bfe, 18B)=27µs인데 memory-bound면 u3는 ~34µs여야 하나 실제 75µs → ~55%가 추출 ALU.
- 원인: u2/u3 코드(5/6-bit)가 byte 경계를 안 맞아 `unpack_ms_kv_elem`이 **원소당 shift/or +
  조건분기(2nd-byte straddle)**를 함. u4의 nibble bfe(1 op)보다 훨씬 무거움.

## 해법 — streaming bit-buffer unpack (extraction ALU 격감)
dense 바이트를 레지스터(uint32 word)로 적재한 뒤, **rolling 64-bit 비트버퍼**로 스트리밍 추출:
- 원소당 **shift+mask 1회**, 워드가 부족할 때만 32-bit refill(블록당 ~6회). 조건분기 없음.
- shared 코드는 **그룹당 1회**만 advance(gs 원소마다, 매 원소 X). → 원소당 funnel-shift 2회
  (= random 64회/블록)가 **OR ~6회 + shift 32회**로 붕괴.
- 단일 `upper_cm` plane 유지(재패킹·op 시그니처·padding 전부 불필요). `csrc/w_gemv.cu` 한 파일만 변경.

## 결과 (GPU1, OUT=K=4096, warm) — 전 117 test 통과, bit-exact
| u | Phase19 | funnel-shift | **streaming(최종)** | MSAQ/MX | MSAQ/BF16 |
|---|---------|--------------|---------------------|---------|-----------|
| u2 | 1.50 (73µs) | 1.09 (53µs) | **0.84 (40µs)** | **0.84 ✅** | 0.88 |
| u3 | 1.43 (68µs) | 1.10 (53µs) | **0.82 (40µs)** | **0.82 ✅** | 0.86 |
| u4 | 0.56 | — | 0.56 (불변) | 0.56 | 0.59 |

- **GEMV u2/u3가 드디어 MXINT8·cuBLAS BF16를 모두 추월**(0.82~0.84×). u3 40µs는 memory floor
  (~34µs)에 근접 = 거의 최적. bit-exact(max|diff| u3 0.0, u2 4e-3=bf16, u4 0.0).
- 추출 단계별 기여: byte-straddle→funnel-shift가 75→53µs(extraction이 병목임을 확정),
  funnel-shift→streaming이 53→40µs(funnel-shift도 원소당 2회라 여전히 무거웠음).

## 부정 결과 (기록)
- **plane-split load**: 완전 coalesced인데 0 speedup → load는 병목 아님(compute-bound). 채택 안 함.
- 즉 "register-aligned repack(정렬 wide load)"이라는 처음 가설은 **틀렸고**, 진짜 레버는
  **extraction 알고리즘**(streaming)이었다. (Phase 5/16의 "ALU/bfe 미세최적 무효" 교훈과 일관:
  단, 거긴 memory/latency-bound라 무효였고, 여기 GEMV u2/u3는 진짜 compute-bound라 유효.)

## 스코어보드 갱신 (GEMV 행)
| u | MSAQ | MXINT8 | cuBLAS | MSAQ/MX |
|---|------|--------|--------|---------|
| u2 | 40.3µs | 48.2 | 46.0 | **0.84 ✅** |
| u3 | 39.5µs | 48.2 | 46.0 | **0.82 ✅** |
| u4 | 27.4µs | 48.5 | 46.3 | **0.56 ✅** |

→ **이제 W-only GEMV·KV decode는 전 u에서 MXINT8 추월.** 남은 근소 열위는 GEMM/W+A의 u2/u3
(1.02~1.07, tiled라 compute-bound → tensor-core 필요)뿐.

---

# Phase 21 — GEMM/W+A에 streaming unpack 이식 ❌ 시도 후 REVERT (negative result)

GEMV에서 통한 streaming bit-buffer unpack(P20)을 GEMM/W+A의 B-tile staging에 이식 시도.
B-staging을 per-(o,k)에서 **column-streaming**(스레드가 열 하나를 통째로: UB바이트를 열 간
coalesced로 모아 32코드를 rolling buffer로 풀기)으로 바꿈. 헬퍼 `dequant_col_stream` 추가,
wonly_gemm_tiled·wa_gemm_tiled 둘 다 적용. 재인증 99개 통과(bit-exact).

## 결과 — 전부 악화 (revert)
| | u2 | u3 | u4 |
|---|----|----|----|
| GEMM (P19→P21) | 1.05→**1.21** | 1.05→**1.12** | 1.00→**1.04** |
| W+A (P19→P21) | 1.05→**1.12** | 1.05→**1.07** | 1.00→**1.07** |

## 왜 실패했나 (GEMV와 정반대)
- **GEMM/W+A는 FMA-bound** (tiled가 unpack을 타일당 1회로 분할상환 → unpack은 전체의 ~1.5%).
  즉 staging은 애초에 병목이 아님 → "싼 추출"로 바꿔도 시간이 안 줄어듦.
- column-streaming은 staging 스레드를 TBN×TBK개 → **TBN개로 줄여 병렬성 손실**, streaming
  레지스터(ureg/sreg/buf)가 **occupancy를 깎음**. 병목 아닌 곳을 만지며 **구조적 손해만** 봄.
- **u4도 악화**(1.00→1.04)된 게 결정적 증거: u4는 원래 unpack(nibble bfe)이 싼데도 느려짐
  = 이득이 extraction이 아니라 staging 구조(병렬성/occupancy)에서 나왔던 것.
- 대조: GEMV decode는 unpack이 **분할상환 안 되는**(M=1) extraction-bound라 streaming이 결정적.
  같은 기법이 bound 종류에 따라 정반대 결과 → **병목 규명이 전부**.

## 조치 / 결론
`csrc/wa_gemm.cu` 전부 revert(per-element staging 복귀). GEMM/W+A u2/u3의 잔여 열위(1.02~1.07)는
unpack이 아니라 **FP32 CUDA-core 연산 자체**가 한계 → 진짜 레버는 **tensor-core(INT8 IMMA /
BF16 WMMA)** 뿐(P11 WMMA는 opt-in). unpack 미세최적으로는 안 줄어듦이 측정으로 확정됨.

---

# Phase 22 — tensor-core(BF16 WMMA)로 GEMM u2/u3 crossover 시도 → u4만 성공, u2/u3 ❌

목표: GEMM/W+A u2/u3의 잔여 열위(FP32 tiled 1.02~1.07)를 tensor-core로 넘기기.

## 구현 — matched WMMA 비교 완성
MSAQ는 이미 `wonly_gemm_wmma`(BF16 WMMA, P11, opt-in `MS_TILE_CFG=10`)가 있었으나 **MXINT8 쪽
WMMA가 없어 matched 비교 불가**였음. `mxint8_gemm_wmma`를 추가(동일 구조, staging만 int8→bf16
직접). 둘 다 `MS_TILE_CFG=10`에서 WMMA. 재인증 `test_w`(cfg=10) 통과, rel 1.7e-3.

## 결과 (M=512, OUT=K=4096, matched) — WMMA가 비율을 **악화**
| u | FP32 tiled MSAQ/MX | **WMMA MSAQ/MX** | WMMA 절대가속(MSAQ/MX) |
|---|--------------------|------------------|------------------------|
| u2 | 1.07 | **1.17 ❌** | 1.40× / 1.53× |
| u3 | 1.06 | **1.14 ❌** | 1.42× / 1.53× |
| u4 | 0.92 | **0.89 ✅** | 1.59× / 1.54× |

## 진단 — faster matmul이 unpack을 노출
- WMMA는 matmul을 ~1.5× 가속(둘 다 절대 빨라짐) → 커널이 **staging-bound로 이동**(P11이 지적한
  "small-tile WMMA, no cp.async → tensor core 부분 기아"). 그러자 staging의 MSAQ **per-element
  unpack이 더 큰 비중**이 됨.
- u4(nibble bfe, 싼 unpack)는 노출돼도 싸서 crossover 유지(0.89, 절대 1.6× 빠름).
- u2/u3(straddle unpack, 무거움)는 노출이 손해 → 비율 **악화**(MXINT8의 싼 int8 read만 matmul
  가속을 온전히 누림). FP32 tiled에선 FMA-bound라 unpack이 가려져 1.0 근처였는데, WMMA가 그걸 벗김.

## streaming unpack을 WMMA staging에 시도 ❌ (negative, revert)
P20의 streaming bit-buffer unpack을 WMMA Bs staging에 이식(column-streaming + bf16 출력 헬퍼).
**훨씬 악화**(u2 1.17→1.82). 원인: WMMA 커널은 128 thread + fragment 레지스터가 큰데, column-
streaming이 staging thread를 64개로 줄이고 streaming 레지스터(ureg/sreg/buf)까지 더해 **occupancy
붕괴**(P21과 동일, fragment 때문에 더 심함). → revert. byte-straddle staging 유지.

## 결론 / 남은 레버
- **BF16 WMMA로는 u2/u3 crossover 불가** — small-tile WMMA가 unpack을 노출시키고, 그 unpack을
  싸게 만들 방법(streaming)은 이 커널 구조(적은 thread + fragment 레지스터)에서 occupancy를 깎음.
- u4는 tensor-core로 더 빨라짐(0.89, 절대 1.6×) — WMMA는 u4에 유효.
- u2/u3 진짜 crossover는 **cp.async로 unpack을 load 그늘에 숨기는 CUTLASS-급 파이프라인**
  (custom iterator의 load()에 unpack 주입 + 큰 타일 + double-buffer)이 필요 — 별도 대형 작업.
  (matched WMMA infra는 opt-in으로 보존; default는 FP32 tiled 유지해 headline 비교 불변.)

---

# Phase 23 — cp.async-style 파이프라인으로 GEMM u2/u3 crossover ✅ (W-only 성공 / W+A 실패)

P22가 가리킨 레버: unpack을 matmul 그늘에 숨기기. (literal cp.async 대신 **double-buffer +
stage↔MMA overlap**으로 같은 효과.)

## W-only GEMM — 성공 (cfg=11 opt-in)
`wonly_gemm_wmma_pipe`: As/Bs를 **double-buffer**하고, 루프에서 **현재 타일 MMA를 먼저 발행 →
다음 타일을 stage**(unpack). stage(ALU/load)가 MMA(tensor core, 다른 pipe)와 겹쳐 unpack이 숨음.
+ stage의 Bs는 **column-streaming unpack**(P20, 싼 ALU). matched `mxint8_gemm_wmma_pipe`도 동형.

| u | FP32 (default) | plain WMMA(P22) | **WMMA-pipe(P23)** |
|---|----------------|-----------------|--------------------|
| u2 | 1.07 | 1.17 | **0.98 ✅** |
| u3 | 1.06 | 1.13 | **0.98 ✅** |
| u4 | 0.91 | 0.89 | **0.93 ✅** |

- **세 u 모두 MXINT8 추월**(0.93~0.98), cuBLAS 격차 ~8× → **5.3×**, FP32 대비 절대 ~1.6× 빠름.
- **두 수정이 모두 필요**: pipeline만(1.09), streaming만(비-pipe 1.82=occupancy 붕괴),
  **합치면 0.98** — streaming의 싼-but-좁은(64 thread) stage가 MMA 그늘에 숨어 병렬성 손실이
  무의미해짐. (P21에서 streaming이 FP32 tiled에 손해였던 건 거긴 FMA-bound라 stage가 안 숨었기 때문.)
- 재인증 `test_w`(cfg=11) 통과, rel 1.7e-3.

## W+A GEMM — 실패 (시도 후 제거)
`wa_gemm_wmma_pipe`(activation quant를 thread-per-row로 stage에 추가) + matched MXINT8도 구현.
**MSAQ 비율 악화**: streaming 1.27~1.30 / per-element 1.34~1.40 (FP32 1.02~1.04보다 나쁨).
- 원인: W+A stage는 **activation quant(amax→sa→requant, X 2회 읽기)가 이미 stage 예산을 다 씀**.
  거기에 weight unpack까지 얹히면 MSAQ stage > MMA 예산 → 안 숨음. MXINT8은 quant만(Bs는 싼 int8)
  이라 stage < MMA → 숨어서 **MXINT8만 크게 빨라짐**(3548→2942) → 비율이 벌어짐.
- → W+A pipe 커널·dispatch 전부 **제거**. W+A는 FP32 tiled(1.02~1.04, near-parity) 유지.

## 위치 / default
- W-only GEMM WMMA-pipe는 **opt-in(cfg=11)** 유지(64×64 고정이라 M-adaptive 검증 후 default 승격
  가능). default는 FP32 tiled 불변(headline 비교 안정).
- **이제 GEMV·KV·W-only GEMM이 전 u에서 MXINT8 추월.** 남은 건 W+A u2/u3(1.02~1.04, near-parity)
  뿐 — activation quant가 stage를 점유해 unpack을 못 숨김. INT8 IMMA(quant를 epilogue scale로
  빼서 stage를 비움)가 다음 후보.

---

# Phase 24 — INT8 IMMA로 W+A u2/u3 crossover 시도 ❌ (per-block epilogue가 발목, revert)

P23이 가리킨 W+A의 마지막 레버: scale을 stage에서 빼 epilogue로 보내고 INT8 tensor core 사용.
`wa_gemm_imma`(+matched `mxint8_wa_gemm_imma`, cfg=12): stage가 **raw int8**(qx, qw) 생산
(scale fold 없음) → INT8 IMMA가 블록당 int32 dot → **per-block epilogue**에서 sa*sw 적용·fp32 누적.

## 결과 (M=512, clean 측정) — FP32보다 느림
| u | FP32 | **IMMA** | IMMA 절대(MSAQ) |
|---|------|----------|------------------|
| u2 | 1.05 | **1.10 ❌** | 4146us (FP32 ~3700보다 느림) |
| u3 | 1.02 | **1.07 ❌** | 4064us |
| u4 | 0.93 | **0.95 ❌** | 3604us |

(재인증 `test_wa`(cfg=12) 통과 — int dot이라 오히려 정확. 단 느림.)

## 왜 실패했나 — per-block scaling epilogue 오버헤드
- MXINT8 scale은 **블록(32-key)당**이라 int32 dot을 **블록마다** sa·sw로 스케일해야 함 →
  블록간 int32 누적 불가 → 블록마다 `store_matrix_sync`(Cs 16KB) + `__syncthreads` ×2 +
  thread-per-output 스케일-누적. 이게 **NB=128번** 반복 → int8 matmul 이득을 다 잡아먹음.
- 그래서 IMMA가 matmul을 빠르게 해도 **kernel 전체는 epilogue-bound**라 FP32보다 느림. unpack
  노출(P22/23)에 더해 epilogue 오버헤드까지 → MSAQ 비율도 악화.
- sw를 블록당 precompute(`sw_s`)해 epilogue의 exp2f 제거도 시도 → 무효(병목은 exp2f가 아니라
  store+sync 구조).

## 결론 — W+A crossover는 CUTLASS fused epilogue가 필요
hand-written으로는 per-block scale을 `store_matrix_sync`로밖에 못 빼는데 그 오버헤드가 치명적.
진짜 해법은 **fragment를 레지스터에서 직접 sa·sw로 스케일하는 fused epilogue**(CUTLASS의
EpilogueWithBroadcast류) — 정의되지 않은 fragment 레이아웃을 다뤄야 해 hand-write 영역 밖.
→ IMMA 커널 전부 revert. W+A는 **FP32 tiled(1.02~1.05, near-parity)** 유지.

## 최종 GEMM/W+A 정리
- **W-only GEMM: WMMA-pipe로 전 u crossover**(0.93~0.98, cfg=11 opt-in) ✅ (P23)
- **W+A GEMM: u4만 crossover(0.93), u2/u3 near-parity(1.02~1.05)** — BF16 pipe(P23)·INT8 IMMA(P24)
  둘 다 hand-write로는 못 넘김. activation quant(stage 점유) + per-block scale(epilogue 오버헤드)이
  근본 원인. CUTLASS 급 fused 파이프라인이 유일한 잔여 레버.

---

# Phase 25 — mantissa-sharing을 대수적으로 이용한 shared-factored dot ❌ (느림, revert)

질문: W+A는 weight가 MSAQ-shared인데, naive는 full-word `qw=wu·2^u+ws`를 **원소마다** 복원해
dot한다. shared를 **group당 1회**만 쓰도록 factor하면 arithmetic을 줄일 수 있지 않나?

## 구현 — `wa_gemm_factored` (cfg=20, MSAQ 전용; MXINT8은 shared 없음)
정수 dot을 분해(bit-exact):
`Σ_k qx·qw = 2^u·Σ_k qx·wu  +  Σ_g ws[g]·(Σ_{k∈g} qx)`
- stage가 **per-element combine(`·2^u+ws`) 제거** → Bsu = wu·sw·2^u만, ws·sw는 group당 staging.
- `Sxs[m][g]=Σ_{k∈g}(qx·sa)` (group합) × `wss[o][g]=ws·sw` 의 **(K/gs)-wide correction matmul** 추가.
- 재인증 `test_wa`(cfg=20) 통과 — oracle 정수 dot과 **bit-exact**(수식은 정확).

## 결과 (M=512) — ~1.7× 느림
| u | naive FP32 | **factored** |
|---|-----------|--------------|
| u2 | 3684us (1.05) | **6415us (1.13)** ❌ |
| u3 | 3638us (1.02) | **6353us (1.11)** ❌ |
| u4 | 3310us (0.93) | **5807us (1.01)** ❌ |

## 왜 느린가 — naive full-word dot이 이미 madd-최적
- **factorization은 연산을 줄이지 않고 늘린다**: naive는 shared를 full-word에 fold해 단일 K-wide
  정수 dot **한 번**으로 끝낸다(= madd 하한). factored는 main(K-wide) + correction(K/gs-wide)으로
  **쪼개서 합이 K(1+1/gs) madd** > naive K. shared 곱의 redundancy 제거분(group당 1회)보다
  correction이 더 비쌈.
- W+A는 **FMA-bound**(P21)라 연산↑ = 즉시 느려짐. + factored는 staging 패스·`__syncthreads`도 더 많음.
- factored가 줄이는 건 **shared-추출/combine**인데, tiled에서 unpack은 이미 TBM(64)행에 분할상환돼
  미미. correction FMA는 분할상환 안 됨 → 순손해.

## 언제 이길 수 있나 (현 아키텍처에선 불가)
factorization이 이기려면 **main(upper×activation)을 sub-byte tensor core로** 돌려야 함:
- u4는 upper=4비트 → **s4 IMMA**(int8의 2× throughput) 가능. 그러면 main이 절반 속도라 +correction을
  상쇄하고 이득 가능. 단 u2/u3는 upper=5/6비트라 s4 미적합(분해 더 필요), 게다가 P24의 per-block
  scale epilogue 문제도 재발.
→ 현 FP32/int8 tiled에선 negative. revert. (수식·correctness는 검증됨 — 향후 s4-IMMA 경로의 기반.)

## 결론 (대수적 축도 탐색 완료)
W+A u2/u3는 **메모리(P22-23)·tensor-core(P24)·대수(P25)** 세 축 모두 hand-write로는 못 넘김.
naive int8 dot이 madd-최적이고 W+A가 FMA/epilogue-bound인 게 근본. 유일한 잔여 경로는 **CUTLASS
s4/s8 IMMA + fused epilogue**(upper를 sub-byte로, shared correction과 scale을 epilogue에 융합).

---

# Phase 26 — W+A GEMM을 2-stage(활성화 선-양자화 + 순수 int8 IMMA)로 → 전 u CROSSOVER ✅✅

P23/24가 실패한 이유는 **fused W+A가 GEMM prologue 예산에 activation-quant과 weight-unpack을
둘 다 얹어** MMA 그림자에 못 숨겼기 때문. 활성화 양자화를 **별도 선-패스로 분리**하면 GEMM에는
weight unpack만 남아 **이미 P23에서 푼 W-only GEMM + int8 epilogue**로 환원된다.

## 구조
- **Stage 0 (`quant_act_kernel`, memory-bound)**: X bf16[M,K] → qX int8[M,K] + sa_exp int8[M,nb].
  1 warp/(m,blk), amax warp-reduce, E8M0, clip. 원소당 **정확히 1회** 양자화(fused는 OUT/TBN=64회 중복).
  reference.quant_act과 **bit-exact**(qX·sa_exp diff 0).
- **Stage 1 (`wa_imma`, tensor-core)**: A-stage=qX int8 직접 로드(가벼움), B-stage=weight unpack
  (유일한 무거운 일) → **double-buffer로 다음 블록 unpack ↔ 현재 블록 INT8 MMA 오버랩**. MMA int8×int8
  →int32, **블록별 epilogue** `accf += c_int32·2^sa_exp[m,blk]·sw[o,blk]`(fp32). matched MXINT8은
  B-stage만 직접 int8. → MSAQ/MXINT8 차이가 정확히 **weight unpack↔바이트절감**으로 격리.
- `wa_gemm_cuda`가 Stage0→Stage1을 한 호출로(타이밍에 선-패스 포함). `MS_WA_FOLD=1`로 옛 FP32-fold A/B.

## 결과 (M=512, OUT=K=4096, warm 한 세션)
| u | **IMMA 2-stage** MSAQ/MX | FP32-fold | IMMA가 FP32보다 (MSAQ/MX) |
|---|--------------------------|-----------|---------------------------|
| u2 | 2326/2759 = **0.84 ✅** | 1.04 | 1.56× / 1.27× |
| u3 | 2288/2766 = **0.83 ✅** | 1.02 | 1.57× / 1.27× |
| u4 | 1951/2774 = **0.70 ✅** | 0.93 | 1.70× / 1.28× |

- **전 u에서 MXINT8 추월**(0.70~0.84). vs FP32-fold **bit-exact**(rel 0). Stage 0 = **16µs = 전체의 1%**.
- IMMA-pipe는 **MXINT8도 FP32 대비 1.27× 가속**(공정 — 구조가 양쪽 다 도움). MSAQ가 더 크게 이득(1.56×)인 건
  pipe가 **unpack을 MMA 뒤에 숨기고** + MSAQ가 **weight 바이트를 덜 읽기**(u2 25 vs 32, u4 18 vs 32) 때문 →
  B-stage가 memory-bound가 되어 바이트 절약이 시간으로 드러남.
- 핵심: **unpack은 더 이상 병목이 아님**. MSAQ(unpack 有)가 MXINT8(unpack 無)보다 빠르다는 게 증거 — unpack이
  MMA 그림자에 완전히 숨었고, 남은 차이는 순수 바이트 수.

## 의의
- P21-25(memory·tensor-core·algebra)가 모두 fused 구조의 prologue 충돌로 실패했는데, **연산 분리(2-stage)**가
  그 충돌을 없애 W+A를 W-only로 환원 → crossover 상속. **이제 4 scope(GEMV·KV·W-only GEMM·W+A) 전부 전 u
  추월**(W+A는 IMMA 2-stage 기본).
- 배포 시 선-패스는 상류(layernorm/직전 GEMM) epilogue로 fuse하면 비용 0. 벤치는 forward마다 발생하므로 타이밍 포함.

## 세부 병목 분해 (MSAQ wa_imma, M=512, us; MS_WA_DIAG로 구간 격리)
| u | full | St0 quant | epilogue(store+scale) | MMA+qX-load | **B-stage(weight read)** | bytes/blk |
|---|------|-----------|------------------------|-------------|--------------------------|-----------|
| u2 | 2386 | 17 (0.7%) | 247 (10%) | 638 (27%) | **1501 (63%)** | 25 |
| u3 | 2328 | 17 | 258 | 626 | **1444 (62%)** | 22 |
| u4 | 1975 | 17 | 305 (15%) | 653 | **1082 (55%)** | 18 |

**핵심 (무엇이 무엇에 숨는가):**
- **B-stage 비용이 weight 바이트 수에 정확히 비례**: 60us/byte (u2 1501/25, u4 1082/18, MXINT8 1887/32=59).
  → **weight unpack(ALU)은 INT8 MMA 그림자에 100% 숨었고**, B-stage는 순수 **weight-memory-bound**. MSAQ가
  바이트를 덜 읽는 만큼(u4 18 vs 32) 그대로 시간이 줄어 crossover. (unpack이 병목이면 u2가 u4보다 훨씬 느려야
  하는데, 시간차는 정확히 바이트차 = unpack 무관.)
- **Stage 0(활성화 양자화)은 17us=0.7%** — 선-패스로 분리해 GEMM 파이프라인과 자원 경쟁 안 함(P23 충돌 해소의 직접 증거).
- **per-block epilogue(store_matrix_sync + sa·sw scale) = 10~15%** — 남은 주 오버헤드(P24를 죽였던 것이 이제
  분할상환돼 ~12%로 축소). fragment를 레지스터에서 직접 스케일하는 fused epilogue(CUTLASS)로 더 줄일 여지.
- MMA+qX-load = ~27% (tensor-core 본선 + 가벼운 int8 활성화 로드).

**요약**: 커널은 이제 **weight-memory-bound**(B-stage 55~63%)라 MSAQ의 바이트 절약이 시간으로 직결. 숨김 관계는
(1) weight unpack → INT8 MMA 뒤, (2) 활성화 양자화 → 별도 선-패스(GEMM 밖). 남은 레버는 epilogue(~12%)뿐.

---

# Phase 27 — W+A 활성화 양자화를 plain MXINT8 → MSAQ-s(mantissa-sharing)로 (포맷 정의 수정)

MSAQ 포맷에서는 **활성화도 weight/KV와 동일한 mantissa-sharing**으로 양자화해야 한다(plain MXINT8 아님).
지금까지 W+A의 활성화는 `round(x/sa)`(full int8)였는데, 이를 `pack.decompose`/`reference.quant_act(share=True)`
경로(upper + 그룹-shared)로 in-kernel 구현. **scale 처리는 동일(base sa), int8 word 계산만 변경.**

## 변경 (`wa_gemm.cu`만; MXINT8 baseline은 불변)
- 새 `quant_act_msaq_kernel`(Stage-0): 1 warp/(m,blk). ① E8M0 base sa ② q_upper=clip(round(x/(sa·2^u)),
  ±(2^(7−u)−1)) ③ residual ④ **gs-그룹 평균(warp shfl_xor)** ⑤ r_shared=clip(round(mean/sa), ±2^(u−1))
  ⑥ qx=q_upper·2^u+r_shared. `wa_gemm_cuda`가 이 커널로 라우팅(plain은 MXINT8 path가 계속 사용).
- `wa_gemm_tiled`(FP32-fold A/B)도 **thread-per-row decompose**로 교체(As=qx_msaq·sa).
- `quant_act` op에 (u,gs) 추가 → MSAQ-s.
- **matched-baseline 규율 예외**: 이건 최적화가 아니라 **포맷 차이**라 MXINT8 baseline(`mxint8.cu`)의
  활성화는 plain MXINT8 그대로 유지. 비교의 의미 = "MSAQ-s 활성화 vs MXINT8 활성화".

## 검증
- Stage-0 출력 vs `quant_act(share=True)`: **qX·sa_exp diff 0**(전 u/gs, bit-exact to decompose).
- W+A 커널 vs `wa_matmul(share_act=True)`: **rel 1.7e-3**(bf16 라운딩). (vs share_act=False는 3e-2~1e-1로 불일치 = 의도대로.)
- MXINT8 baseline vs `wa_matmul_mxint8`(=MXINT8 활성화): rel 1.7e-3(불변 확인).
- 테스트 배선: `test_wa::test_wa_gemm_vs_oracle` → share_act=True; `test_emulation::test_wa_gemm_logic` →
  MSAQ-s 미러 + share_act=True. **117 테스트 전부 통과.**

## 성능 (M=512, MSAQ-s 활성화)
| u | MSAQ | MXINT8 | MSAQ/MX | MSAQ-s quant |
|---|------|--------|---------|--------------|
| u2 | 2360 | 2763 | **0.85** | 20µs (1%) |
| u3 | 2316 | 2765 | **0.84** | 20µs |
| u4 | 1968 | 2776 | **0.71** | 20µs |
- decompose 양자화는 plain보다 약간 무겁지만(17→20µs) 여전히 **1%** — crossover 그대로 유지(P26의 0.70~0.84 ≈ 0.71~0.85).

# Phase 28 — End-to-end 하니스용 quantize 커널 3종 (KV write / KV append / W+A GEMV)

지금까지 모든 scope는 "이미 packed된 입력"을 받아 GEMV/GEMM/attention을 측정했다. TTFT·TPOT·총 추론시간을 재려면 **런타임 양자화 자체**를 커널로 가져와야 한다(prefill에서 KV write, decode마다 KV append, decode마다 W+A GEMV의 활성화). 세 커널 모두 같은 §0 프리미티브를 공유.

## §0 공유 프리미티브 (`ms_utils.cuh`) — `ms_lib.pack.decompose`의 device 대응
- `e8m0_exp_from_amax(amax)` — 블록 amax → E8M0 지수(`floor(log2 amax) - E_MAX`, [-127,127] 클램프).
- `decompose_ms_block(x[32], u, gs, q_upper[32], r_shared[ng])` — 한 스레드가 32-블록을 통째로 들고 base scale `sa`·(8−u)-bit upper·u-bit shared로 분해. `reference.quant_act(share=True)`와 비트 동일.
- `pack_codes_lsb(codes, n, width, buf, nbytes)` — dense LSB 비트팩(extract_code의 역). KV plane용.

## KV write (prefill) — `kv_attention.cu kv_write_kernel`
- bf16 `X[H,L,D]` → 인증된 token-major MSAQ plane(`scale_exp[H,nb,L]`, `upper[H,nb,L,UB]`, `shared[H,nb,L,SB]`). **thread-per-token**(P18 read의 write 거울): 고정 (h,blk)에서 연속 토큰이 UB-연속 바이트를 store → coalesced. `H·ceil(L/256)` 블록이라 split 불필요.
- matched `mxint8_kv_write` (int8 직접 `qweight[H,nb,L,32]`).

## KV append (decode) — `kv_attention.cu kv_append_kernel`
- write의 **L=1·pos** 특수화: decode 한 스텝의 새 토큰 `X[H,D]`를 미리 잡아둔 cache의 slot `pos`(stride Lcap)에 in-place 기록. thread=(h,blk), work=H·nb(작음) → launch-latency 지배 → 배포 시 projection/RoPE epilogue 또는 attention prologue에 fuse 대상. 같은 decompose+pack·token-major slot이라 read 경로가 하나의 포맷만 본다.
- matched `mxint8_kv_append`.

## 검증
- write·append plane 모두 `pack_kv`에 **byte-exact**(scale_exp/upper/shared diff 0, 모든 u/gs). append는 토큰별로 채운 cache(Lcap=L)가 통째 write와 동일함을 확인.
- end-to-end write/append → kv_decode → oracle **rel_fro 1.6e-3** (< 2e-2). 게이트: `test_kv_write_vs_pack`, `test_kv_write_then_decode_vs_oracle`, `test_kv_append_vs_pack`, `test_kv_append_then_decode_vs_oracle`.

## 성능 (KV write/append, H=32 D=128, warm) — vs MXINT8·BF16
KV write (prefill, 전체 [H,L,D] → 캐시, MSAQ u4):
| L | MSAQ | MXINT8 | BF16 | MSAQ/MX | MSAQ/BF16 |
|---|------|--------|------|---------|-----------|
| 1024 | 101µs | 130µs | 23µs | **0.78** | 4.3 |
| 2048 | 193µs | 226µs | 43µs | **0.85** | 4.5 |
| 4096 | 326µs | 370µs | 83µs | **0.88** | 3.9 |
- **MSAQ < MXINT8** (0.78~0.91): bf16 로드+amax+양자화는 동일하지만 MSAQ는 packed plane(u4 ≈ 32원소/4.5B)을 store, MXINT8은 full int8(32B) → **store 대역폭 ~7× 적음** → decompose 비용 상쇄하고 이김.
- vs BF16 ~4×: bf16 캐시는 순수 memcpy(양자화 없음). prefill 1회 비용이고 decode read에서 회수(read가 매 스텝).

KV append (decode, 단일 토큰 [H,D] → slot, Lcap=4096):
| u | MSAQ | MXINT8 | BF16 copy | MSAQ/MX |
|---|------|--------|-----------|---------|
| u2 | 10.5µs | 8.8µs | 17.5µs | 1.20 |
| u4 | 8.7µs | 8.7µs | 16.1µs | 1.00 |
- work = H·nb = 128 스레드 → **launch-latency 지배**(~8~17µs ≈ 커널 런치). 알고리즘 차이가 아니라 런치 비용(MSAQ/MXINT8가 torch generic `copy_`보다 빠른 것도 런치 오버헤드 차이). 배포 시 projection/RoPE epilogue나 attention prologue에 **fuse** 대상.

## W+A GEMV (decode) — `w_gemv.cu wa_gemv_wide_kernel` + `wa_gemv_cuda`
- W-only wide GEMV(column-major plane + bfe(u4)/streaming(u2u3) unpack)에 **활성화도 MSAQ-s로 양자화**한 버전. Stage 0 pre-pass(`ms_launch_quant_act_msaq`, M=1)가 x[K]를 int8 word `qx=q_upper·2^u+r_shared` + 블록 base exp `sa_exp[NB]`로 분해(W+A GEMM의 활성화 prepass 재사용). qw unpack은 W-only와 **바이트 동일**; 유일한 차이는 누적: 블록마다 **정수 dot** `idot=Σ qw·qx`(int8·int8→int32) 후 두 블록 스케일을 **한 번** fold(`acc += idot·sw·sa`) — 원소당 float madd가 아님. sw는 (blk,o)별(weight), sa는 blk별(activation, 컬럼 공유).
- matched `mxint8_wa_gemv`: plain-MXINT8 활성화 prepass(`ms_launch_quant_act`) + 동일 int-dot. 두 operand 모두 full int8(decompose 없음) = W+A decode scope의 matched baseline. (활성화 MSAQ-s vs MXINT8 포맷 차이는 Phase 27의 규약대로 baseline에 미러하지 않음.)

## 검증·성능 (OUT=K=4096, warm)
- 게이트 `test_wa_gemv_vs_oracle`(MSAQ-s, `wa_matmul(share_act=True)`) + `test_mxint8_wa_gemv_vs_oracle`(`wa_matmul_mxint8`) 모두 통과(rel_fro < 2e-2, 모든 u/gs).

| u | MSAQ | MXINT8 | MSAQ/MX |
|---|------|--------|---------|
| u2 | 47.6µs | 40.5µs | 1.17 |
| u3 | 46.6µs | 40.5µs | 1.15 |
| u4 | 33.2µs | 40.5µs | **0.82** |
- W-only GEMV와 **동일한 crossover 프로파일**: u4만 win(wide int4 load + nibble bfe), u2/u3는 streaming unpack 비용이 int-dot 위에 그대로 남아 not-win. 활성화 prepass(M=1)는 공유되어 비교에 중립.

# Phase 29 — End-to-end harness (Llama-3.1-8B full forward) 결과

7종 커널을 실제 Llama-3.1-8B 32-layer 디코더에 끼워 TTFT·TPOT·총시간 측정. GQA(32Q:8KV)
지원을 위해 decode attention에 `num_kv_heads` 추가(q head→kv head h/group), 고정용량 KV 캐시를
위해 plane stride `Lcap`을 attended `Lk`와 분리(둘 다 게이트 통과). 설계 [harness_design.md],
전체 결과·곡선 [harness_results.md].

## 핵심 결과 (prefill=800, decode=3880, RTX 3090)
| 경로 | TTFT | TPOT | total | /bf16 | /mxint8 |
|------|------|------|-------|-------|---------|
| bf16 | 272ms | 41.4ms | 161.1s | 1.00 | — |
| mxint8_wonly | 1615ms | 31.6ms | 124.2s | 0.77 | — |
| mxint8_wa | 1594ms | 26.0ms | 102.4s | 0.64 | — |
| **msaq_wonly-u4** | 1504ms | **24.4ms** | **96.1s** | **0.60** | **0.77** |
| msaq_wa-u4 | 1215ms | 26.1ms | 102.7s | 0.64 | 1.00 |

- **최고 = msaq_wonly-u4: bf16의 0.60×, MXINT8의 0.77×.** decode가 지배(3880≫800).
- **TPOT 성장곡선**: bf16 34.8→53.2ms(KV 커지며 폭증) vs msaq u4 25.0→25.0ms(packed KV로 평탄)
  → **긴 컨텍스트일수록 MSAQ 이득↑**(설계 가설 확인).
- **TTFT는 bf16 압승**(cuBLAS prefill vs 커스텀 IMMA ~5–6× 느림)이나 decode가 total 지배해 역전.
- W-only/W+A: MSAQ는 **wonly-u4**가 최강(decode GEMV 0.63), MXINT8은 wa가 최강(IMMA+int-dot).
  MSAQ vs MXINT8 — W-only 완승(0.77~0.90), W+A 박빙(u4 1.00).
- 한계: Python 루프 decode라 절대 TPOT에 dispatch 오버헤드 포함(공통)→비율은 커널단위(0.54~0.63)보다 희석.

# Phase 30 — CUDA-graph 측정 + weight/KV 양자화 분리(4 시나리오)

dispatch 오버헤드 제거 위해 decode TPOT를 **CUDA graph**로 측정. 발견·수정: 모든 커널이 **default
stream**에 런치돼 graph capture가 **빈 그래프**가 됐음(capture는 current stream 감시) → 31개 런치
전부 `at::cuda::getCurrentCUDAStream()`로 수정(stream 정합성 버그이기도). 반복 capture가 같은
프로세스의 다음 eager prefill을 wedge → 시나리오마다 **subprocess 격리**. weight/KV 양자화를 독립
knob으로 분리해 4 시나리오(S1 W-only/S2 W+A/S3 KV-only/S4 W-only+KV) 측정. 전체 [harness_results.md].

## 핵심 결과 (graph, prefill=800/decode=3880)
| 시나리오 | 포맷 | TPOT | total | /bf16 | /mxint8 |
|---------|------|------|-------|-------|---------|
| baseline | bf16 | 37.4 | 145.5s | 1.00 | — |
| S1 W-only | MXINT8 | 37.2 | 145.8s | **1.00** | — |
| S1 W-only | MSAQ-u4 | 29.5 | 115.9s | 0.80 | **0.79** |
| S3 KV-only | MSAQ-u4 | 23.9 | 93.0s | 0.64 | 0.92 |
| **S4 W-only+KV** | **MSAQ-u4** | **16.1** | **64.1s** | **0.44** | **0.64** |

- **weight 양자화=baseline↓, KV 양자화=성장곡선 평탄화(직교)**. bf16 ctx별 27→48ms 폭증,
  KV-only(msaq) 22→25 평탄, 둘 다 15→18 최저·최평탄.
- **최고 S4 MSAQ-u4 0.44×bf16** (weight·KV 이득 compound).
- **MXINT8 W-only=이득 0**(GEMV가 cuBLAS와 동속)인데 **MSAQ W-only=0.80**(wide u4 GEMV가 cuBLAS
  를 이김) → W-only scope에서 MSAQ가 MXINT8 대비 0.79로 명확히 가치.
- graph로 dispatch 제거하니 이전 coupled-run의 0.60(S4 상당)이 **0.44로 더 벌어짐** → Python 루프가
  비율을 희석했음 확인.
- **u-스윕(u2/u3/u4, 각 S의 MXINT8 baseline 대비):** u4가 전 시나리오 압도(S1 0.80·S2 0.91·S3 0.92·
  S4 0.64). u2/u3는 streaming 언팩이 무거워 S2·S3에선 MXINT8보다 느리고(/mxint8 1.05~1.07) W-only
  에서만 이김(0.92~0.93) → **실전 권장 u4.** 전체 17-run 표는 [harness_results.md].

# Phase 31 — 마지막 세 커널 최적화(KV write/read/W+A GEMV) + end-to-end 재측정

세 Phase-28 커널이 "matched + 정확"까지만 와 있어 KV-only·W+A의 u2/u3가 MXINT8에 짐. mantissa-
sharing 쪽 최적화 적용. 전체 표·곡선·근거 [harness_results.md].

- **KV write**: nb를 grid.z로 → occupancy(GQA H=8서 32→224 블록). generic이라 MXINT8에도 mirror.
  L=800 u4 1.09→0.88.
- **KV read**(KV-only decode의 진짜 병목): u2/u3에 streaming bit-buffer unpack 이식(general-straddle
  →롤링버퍼, mantissa-sharing-only). Lk=4680 u2 1.29→0.79, u3 1.27→0.77.
- **W+A GEMV**: 활성화 qx를 shared에 1회 stage(MSAQ만; MXINT8엔 해가 됨→best-vs-best). u2 1.17→1.11.
- **KV append**: 미적용(MSAQ가 decompose+pack로 항상 일이 더 많고 1-블록 launch지배 → 못 이김. graph서 영향~0).

end-to-end(graph) u2/u3 역전: S3 KV-only 1.07→**0.97**, S4 W-only+KV 0.98/0.96→**0.88/0.87**,
S2 W+A 1.06→1.02(~tie). u4는 전 구간 win 유지(S1 0.80·S2 0.90·S3 0.92·S4 **0.64**). 최고 S4 u4 =
bf16 0.44× / MXINT8 0.64×(64.0s). **권장 u4에서 4 시나리오 전부 MXINT8보다 빠름 증명.** 못 이기는
W+A GEMV u2/u3·KV append은 extraction-bound/launch-bound라는 근본 이유 제시(u2/u3 byte절약 0.69~0.78×
가 sub-byte 추출비용 못 넘음; nibble정렬 u4만 win).

# Phase 32 — KV read 공정성 정정 + BW-bound 시도 + 공정 재측정

`for_fair_comparison.md` 감사에서 적출한 🟥(KV read 매핑 비대칭) 해결. MXINT8 KV read를 MSAQ와
동일 **thread-per-key**(in-thread dot, 워프 reduction 제거)로 올림(`mxint8_kv_split_kernel`,
정확도 GQA 1.7e-3). 정량화: MXINT8 KV read ~2× 빨라짐(Lk=4680 254→120µs) → **이전 KV "압승"
(u4 0.41~0.77)은 거의 전부 그 under-optimization 산물**, mantissa-sharing 효과 아님.

공정 비교(둘 다 thread-per-key): MSAQ KV read **u4 tie(1.01~1.04)·u2/u3 손해**. BW-bound 재작성
시도했으나 근본 장애물 확인 — flash-decode가 BW의 ~20× 느리게 도는 overhead-bound라 바이트 절약이
시간에 안 나타나고, sub-byte V의 Pass-2(per-d reduction)는 half-sector(직접) 또는 staging
occupancy-cap이라 BW-bound로 못 감(direct-V 실험: tie→1.40 악화). 깨끗한 KV win은 완전한
FlashDecoding 재설계(미해결 과제). → **공정 최선(KV tie) 상태로 end-to-end 재측정.**

## 공정 end-to-end (graph, prefill=800/decode=3880) — S3 KV win 정정
| 시나리오 | u4 /mxint8 (이전 → 공정) |
|---------|--------------------------|
| S1 W-only | 0.79 (무변, weight GEMV) |
| S2 W+A | 0.90 |
| **S3 KV-only** | 0.92 → **1.01 (tie)** |
| **S4 W-only+KV** | 0.64 → **0.69** (여전히 win) |
- KV를 쓰는 S3·S4의 MXINT8이 thread-per-key로 빨라짐(total 100.7→91.9s). **S3는 tie로 정정**,
  S4는 **weight GEMV win이 KV tie 위에 얹혀 여전히 0.69 win**.
- **공정 결론: MSAQ의 end-to-end 우위는 weight GEMV(W-only scope, 진짜 BW-bound)에서 오고,
  KV read는 현 스칼라 flash-decode에선 tie.** bf16 대비는 전부 win(S4 u4 0.44).

# Phase 33 — GQA-batched FlashDecoding KV read 시도 (design A 핵심 lever)

KV read 공정 win을 위해 design A(텐서코어 FlashDecoding + GQA 쿼리배칭 + cp.async) 착수. 먼저
핵심 lever인 **GQA 쿼리배칭**(`kv_decode_gqa_kernel`)을 구현: 블록당 KV head 1개가 G=Hq/Hkv개
쿼리를 함께 처리 → K/V를 한 번 읽고/언팩해 G개 쿼리 행에 재사용(V 트래픽 G배 amortize). Pass-2의
exp는 (g,kk)별 1회만 계산해 shared에 저장(스레드-d별 중복 제거). GQA 게이트(Hq8 Hkv2) 통과(정확).

**roofline 정정:** P·V는 AI = G·D/(V bytes/key)로 ridge(75) 한참 아래 → memory-bound. 정량(Lk=4680,
Llama): MXINT8 9.6MB→~11µs, MSAQ 5.4MB→~6µs memory + ~5µs unpack → ~7µs → **roofline win ~0.64×
(사용자 0.56×는 unpack 무시한 낙관, 0.64×가 실현치).**

**그러나 현 스칼라+full-chunk-staging 구현은 ~25× off roofline**(172µs vs ~7µs): staging shared
(~13KB)가 occupancy를 캡하고 chunk 단위 동기 staging이 load latency를 노출해 15 GB/s에 머문다(split
늘려도 안 풀림 — wide와 같은 벽). → **roofline 실현엔 cp.async 더블버퍼 + 작은 MMA 타일(+텐서코어)
파이프라인이 필요**(genuinely hard remaining work). GQA 커널은 구조적으로 옳고 **MS_KV_GQA=1 opt-in**
으로 남겨 토대로 보존(default는 wide). 미해결: cp.async/텐서코어 pipeline.

# Phase 34 — design B(warp-transpose) + 점유율 lever + sector 진단 → 공정 tie 확정, D=256 fix, 3-model sweep

Phase 32/33의 "KV read는 BW-bound 재작성이 전제" 가설을 끝까지 추적. 먼저 ncu로 병목 재확정
(MS_KV_WIDE u4 Lk4680): **MXINT8는 이미 memory-bound로 300~480 GB/s 스케일**(Phase 32의 "66 GB/s"는
stale), MSAQ wide는 140~220 GB/s saturate, 둘 다 점유율 ~23%. 지배 stall = long_scoreboard
(메모리 latency)+barrier 15%, **math_pipe 3.9%(=ALU bound 아님)**. regs 121→4 blk/SM, waves 0.76.

**design B (warp-transpose P·V, staging 완전 제거; `kv_decode_warpT_kernel`, opt-in `MS_KV_WARPT`,
u4·D128, bit-exact rel_fro 0~6e-5).** V를 thread-per-key coalesced 적재→레지스터 unpack→32-lane
broadcast all-reduce로 key→d 전치(스칼라 누적기). shared 11→2 KB로 줄였으나 **점유율 그대로 ~23%**
(grid가 0.5 wave로 더 작아짐 — per-SM block 캡이 한계가 아니었음), shfl이 L1TEX 57%로 상승 →
**wide보다 ~5~10% 느림**. 즉 병목은 staging tax 단독이 아니다.

**점유율 lever 전부 실패.** split mult 3→24는 **MXINT8만** 이득(per-block 일 적음)→격차 확대;
`__launch_bounds__`(regs 121→80 강제)는 spill로 더 느림; cp.async 2× 느림. per-kernel 분해(Lk4680
mult3): decode MSAQ 48.5µs vs MXINT8 45.6µs(**6% tie**), combine 8.2µs 동일.

**sector 진단(핵심).** DRAM read-sector 비 = **0.59 ≈ 바이트비 0.58**(DRAM 레벨 inflation 없음 —
바이트 이점은 실재). L2 비는 0.61~0.69(작은 scale/shared plane over-fetch, Lk로 amortize). →
**latency-wall이지 sector inflation 아님.** 근본: P·V는 키 reduction이라 thread-per-d, **int8 V는
자연히 full-sector·sub-byte V는 half-sector** → staging/transpose 오버헤드가 0.58× 바이트 절감을
상쇄. 적게 읽는 MSAQ는 MLP가 작아 latency도 덜 숨김.

**부수 버그 fix:** head_dim=256(Gemma)에서 staged-V shared가 48KB 초과 → `cudaFuncSetAttribute`로
>48KB dynamic smem opt-in(wide/gqa/cpasync launcher). D=128 무영향.

**문서/하니스:** `kernel_ver2.md`(7-커널 정리, KV dequant 0.54→tie 정정 + sub-byte sector 규칙),
하니스를 3-model(Llama-3.1-8B/Gemma-2-9B/Mistral-7B) × u{2,3,4} × gs{2,8,32} × 4 scope로 확장
(`tests/harness.py`, `gen_results_md.py`→`results.md`). 3개 모델 ~1~2% 내 일치: **u4 gs32 최적,
S1 ~0.79·S2 ~0.90·S3 ~1.01(tie)·S4 ~0.68 (vs MXINT8)**. Gemma D=256 KV는 staged-V 점유율 때문에
gs 민감(u4 gs2 0.70 vs gs32 0.63). **결론: design B/C/D로 KV tie 못 뒤집음.**

# Phase 35 — batch sweep: latency-wall 가설의 반증

"KV tie의 근본은 one-wave 저점유율 → batch로 점유율을 (combine 오버헤드 없이) 채우면 BW-bound가 되어
0.58× 바이트가 0.58× 시간으로 환원"이라는 가설을 직접 검증. 배치 flash-decode 커널 추가
(`blockIdx.z=batch`, MSAQ wide + MXINT8 짝, single-token 경로는 b==0으로 byte-identical, ops
`kv_decode_attention_batched`/`mxint8_kv_decode_batched`, `tests/kv_batch_bench.py`, 배치 slice ==
single-token 검증). B∈{1,4,8,16,32} × GQA 32:8 × Lk4096 × u4:

| B | MXINT8 (useful GB/s) | MSAQ (useful GB/s) | MSAQ/MX |
|---|---|---|---|
| 1 | 95µs (91) | 90µs (55) | **0.94 win** |
| 8 | 689µs (100) | 753µs (53) | 1.09 |
| 32 | 2082µs (133) | 2329µs (68) | 1.12 (gs32: 1.07) |

**반증: batch는 MSAQ를 더 지게 만든다.** per-q-head 커널은 GQA 그룹(4) 안에서 KV를 4× 재독 →
실제 DRAM은 useful×4. B=32에서 **MXINT8 실효 ~530 GB/s(피크 57%, 진짜 BW-bound 도달)** vs
**MSAQ ~270 GB/s(29%)**. batch가 머신을 채워 BW-bound로 올린 건 맞으나 **MSAQ의 달성가능 BW 천장이
MXINT8의 ~절반**(dequant unpack+staging이 throughput throttle) → 0.58× 바이트를 0.51× BW로 읽어
0.58/0.51≈1.14× 시간 = 관측 ~1.1× 손해. B=1 win(0.94)은 둘 다 BW-bound 아닐 때 적은 바이트의 미세 이점.

**확정(3 lever 종합: split-K / warp-transpose / batch).** KV read의 binding constraint는 점유율이
아니라 **MSAQ의 dequant-throughput BW 천장(MXINT8 대비 ~0.5×)**. 병렬도를 더 줘도 대역폭을 실제로
소비하는 MXINT8만 이득. 공정 win은 *dequant 자체를 int8 read와 throughput 동급*(근본적으로 싼
unpack 또는 텐서코어 dequant 파이프라인)으로 만들어야 가능 — 단순 병렬화로는 불가능함이 입증됨.

# Phase 36 — channel-major V (KIVI식 layout) 탐색: 속도 win 확인, 그러나 정확도 trade로 기각

P·V half-sector의 *원인*을 고치는 접근: V를 **channel-major `[d, token]`**(토큰을 sub-byte 연속
패킹)로 깔면 P·V가 `out[d]=Σ_t p[t]·V[d,t]` = **coalesced GEMV**가 되어 transpose/staging이 사라짐.

**(1) 속도 proxy(`tests/pv_gemv_proxy.py`)** — V를 channel-major로 패킹해 기존 MSAQ wide GEMV vs
MXINT8 GEMV로 P·V 모양(OUT=D, K=Lk) 측정. 결과: token-major에서 MSAQ를 0.5× BW로 묶던 천장이
**사라짐** — BW-bound 영역(여러 head fuse OUT=1024 + Lk≥8192)에서 MSAQ가 MXINT8과 **동일 BW**
(284 vs 288, 340 vs 320 GB/s) 도달 → **ratio 0.54~0.58 WIN**(weight GEMV 0.56과 동일 메커니즘).
단 per-head 단독(OUT=128)은 launch-bound라 loss → 다중 head fuse + 긴 context 필요(=프로덕션 서빙).

**(2) 정확도 probe(`tests/v_grouping_accuracy.py`)** — 그러나 channel-major는 dense block이 연속이라
**quant grouping을 32-token 블록 = reduction(토큰)축으로 강제**한다(layout과 grouping이 분리 불가).
이는 KIVI가 V에 권장하는 *per-token grouping*(현 token-major = 블록이 한 토큰의 head_dim)의 **반대**.
현실적 V(토큰별 norm 변동 token_var=3)에서 dequant rel_fro:

| | MSAQ u4 | MSAQ u2 | MXINT8 |
|---|---|---|---|
| token-major (per-token, 현재) | 0.127 | 0.033 | 0.0084 |
| channel-major (token-block) | 0.166 | 0.053 | 0.0148 |
| 배율 | ×1.31 | ×1.64 | ×1.76 |

→ channel-major는 현실적 V에서 **정확도 1.3~1.8× 악화**(양쪽 포맷 모두 — fair는 유지되나 둘 다 나빠짐).

**근본 tension.** dense sub-byte packing에선 layout↔grouping이 묶여 있어:
- per-token grouping(정확도 ✓, KIVI) ⟹ token-major ⟹ P·V half-sector(MSAQ tie/loss)
- channel-major(속도 ✓) ⟹ token-block grouping ⟹ 정확도 1.3~1.8× ↓

**둘을 동시에 못 가진다 → channel-major win은 "공짜"가 아니라 속도↔정확도 trade이며 KIVI 관례에 역행**
하므로 기각. KIVI가 이 tension을 피하는 길은 *V per-token 유지 + tile staging + 텐서코어 MMA*(staging을
받아들이되 텐서코어로 P·V compute를 흡수) — 즉 Phase 32~35에서 "남은 정공법"으로 지목한 BW-bound
FlashDecoding 재작성과 동일. **공정·정확도 둘 다 지키는 KV-read win은 텐서코어 P·V 파이프라인이 전제.**

# Phase 37 — 텐서코어 P·V (bf16 WMMA + split-K): "남은 정공법"마저 기각, 근본벽 확정

Phase 32~36이 미해결로 남긴 *유일한* 경로(텐서코어 P·V)를 실제 구현. P·V = `O[M,D]=P[M,Lk]@V[D,Lk]`를
bf16 WMMA로(`pv_wmma_kernel` + MXINT8 짝 `pv_wmma_mx_kernel`, `wonly_gemm_wmma` 골격 재사용).
**정확도/공정성 보존 핵심:** V를 **token-major(per-token group, KIVI 정렬)로 DRAM에서 coalesced
0.58× 읽고**, unpack하면서 **d-major bf16 shared 타일로 on-chip transpose** → channel-major의 정확도
trade 없이 텐서코어에 먹임. 정확도 검증 통과(rel_fro 2.3e-3 = u4 양자화 오차). 양쪽 동일 WMMA 경로,
unpack만 다름(공정).

- **1차(split-K 전):** 64×64 타일이 decode 모양(M=16~128, D=128)에 너무 커 grid가 16블록 → 처참한
  under-occupancy(5~11 GB/s, 357~1027µs, 스칼라 대비 10~25× 느림). 이 영역에선 MSAQ가 ratio
  0.76~0.90으로 "이김" — 그러나 이는 **저점유율 latency 영역에서 적은 바이트가 latency를 덜 노출**한
  artifact(batch B=1 win과 동일 성질).
- **split-K(Lk 분할 + fp32 partial + combine, `pv_split_count`):** occupancy 회복 → 절대속도 **10×**
  개선(1027→186µs, 스칼라와 경쟁력). **그러자 ratio가 뒤집힘: MSAQ가 짐(1.07~1.22, M=64서만 0.98).**
  MXINT8 49~115 GB/s vs MSAQ 25~56 GB/s = **다시 ~0.5× BW 천장**.

**근본 메커니즘(확정).** 텐서코어는 **reduction을 가속**하는데, reduction은 애초에 병목이 아니었다.
병목은 **dequant → bf16 shared 타일 생성 throughput**이고, **MMA를 쓰려면 V를 bf16 타일로 풀어야**
하므로 **두 포맷이 똑같은 bf16 타일을 만든다** → bf16 staging은 format-무관이고 MSAQ는 그 위에 unpack
ALU만 더 얹는다 → MSAQ ≥ MXINT8 시간. **DRAM 0.58× footprint는 무관**(DRAM이 bound가 아니므로).
GEMV가 이기는 이유와 대조적: GEMV는 staging 없이 wide-load→직접 누적이라 DRAM-bound가 되지만,
**텐서코어는 bf16 staging을 강제**해 그 win 메커니즘을 없앤다.

**최종 결론(6 lever: split-K / warp-transpose / batch / channel-major / 텐서코어 / split-K WMMA).**
KV read의 binding constraint는 일관되게 **"MSAQ를 텐서코어/누적기가 소비 가능한 형태로 dequant하는
throughput"**(MXINT8 대비 ~0.5× 실효 BW)이며, 점유율도 reduction도 layout도 아니다. P·V의 키-가로
reduction + sub-byte는 (a) half-sector(직접) (b) 정확도 trade(channel-major) (c) bf16 staging
천장(텐서코어) 중 하나를 강제하고, 셋 다 0.58× 바이트 이점을 상쇄한다. → **공정·정확한 MSAQ KV-read
win은 본 스칼라/WMMA dequant 패러다임에선 불가능**. 가능하려면 V를 dequant 없이 텐서코어가 직접
먹는 native sub-byte MMA(하드웨어 미지원) 또는 GEMV처럼 staging을 회피하는 sub-byte-coalesced
reduction(half-sector라 불가)이 필요. 커널/ops/bench는 보존(`pv_wmma*`, `tests/pv_wmma_bench.py`).

# Phase 38 — coalesced B-load + 소프트웨어 파이프라인: 텐서코어 P·V가 드디어 WIN (batched)

Phase 37 결론을 뒤집은 두 가지(사용자 제안: overlap/fused로 dequant throughput↑):
1. **버그 수정 — coalesced thread-per-key B-load.** Phase 37 `pv_wmma`는 스레드를 (d,token)로
   매핑해 토큰당 nibble 1개를 16B stride로 **scatter read**(half-sector) → MSAQ가 0.58× full-sector
   이점조차 못 받았다. 수정: 스레드가 한 키의 16B 레코드를 통째로 읽고(32스레드=512B 연속=full
   sector, 0.58×) 32개 d를 d-major bf16 타일에 on-chip transpose-write.
2. **double-buffer 소프트웨어 파이프라인.** `stage(next)`를 MMA(current) 뒤로 보내 unpack(+그 DRAM
   load)이 텐서코어 MMA와 overlap(W+A GEMM `wonly_gemm_wmma_pipe` 기법).

**결과(token-major V 유지 = 정확, rel_fro 2.3e-3; 양쪽 동일 WMMA, unpack만 다름 = 공정):**

| M | Lk4096 ratio | Lk8192 ratio |
|---|---|---|
| 16 | 1.01 (tie) | 1.01 |
| 32 | **0.94 WIN** | 0.93 |
| 64 | **0.89 WIN** | 0.88 |
| 128 | **0.87 WIN** | 0.84 |

scatter→coalesce로 MSAQ 실효 BW가 0.5×→~0.6× MXINT8로 회복, 0.58× 바이트와 합쳐져 **M≥32에서
win**. M=16은 combine/소타일 오버헤드가 지배해 tie.

**의의:** Phase 32~37의 모든 negative를 뒤집는 **첫 공정·정확 KV-read win track**. half-sector도
정확도 trade도 없이, 텐서코어 + coalesced read + 파이프라인으로 0.58× 바이트가 시간으로 환원됨.

**범위/한계(정직):**
- **batched 영역(M=batch×G ≥ 32)에서만 win.** single-stream decode(batch=1, G=4 → M=4)는 M<16이라
  tie/loss — 즉 **배치 서빙 win**이지 단일 토큰 디코드 win이 아니다.
- **P·V 단독 커널**(Pass-2). 완전한 attention은 Pass-1(Q·K+softmax) 필요 → 2-pass 또는 fused.
- 아직 end-to-end 하니스(batch=1)에 통합 안 됨 → 기존 S1~S4(batch=1)엔 이 win이 안 나타남.
  end-to-end 입증엔 **batched 2-pass decode 경로 + batched 하니스**가 필요(별도 작업).

# Phase 39 — shared-prefix 2-pass attention(Q·K WMMA + softmax + P·V WMMA): P·V win이 Q·K에 희석

Phase 38 P·V win이 **완전한 attention**에서도 성립하는지 검증. win 조건 정정: M=batch×G는 *독립 배치*
에선 V가 배치마다 달라 안 통하고, **M개 query가 한 KV를 공유**(shared-prefix caching / beam)할 때만
M=N·G로 커진다. 그래서 **shared-prefix** 시나리오로 2-pass attention 구현: `qk_wmma`(scores=Q@K^T,
contract D, K token-major coalesced) + torch softmax + `pv_wmma`(Phase 38). 정확도 통과(rel 2.4e-3).

**완전 attention 실측(MSAQ/MXINT8, M=N·G):**

| 모델 | M 범위 | ratio (small M → large M) |
|------|--------|----------------------------|
| Llama (G4,D128) | 32→128 | 1.17 → **1.00** |
| Gemma (G2,D256) | 16→64 | 1.25 → **0.99 (간신히 win)** |
| Mistral (G4,D128) | 32→128 | 1.18 → 1.00 |

**P·V 단독은 0.87 win인데 완전 attention은 tie~loss(best ~1.0 at M=128)** — Q·K가 희석한다.
원인: **Q·K는 contraction이 D=128(작음, MMA 4청크)이라 MMA가 K-unpack을 못 숨겨 unpack-bound →
MSAQ loss**. 즉 P·V(키 reduction, 긴 contraction→MMA가 unpack 숨김)는 win이지만 Q·K(D reduction,
짧은 contraction)는 loss라 합치면 ~tie.

**확정(7 lever).** scalar/staging/warp-transpose/batch/channel-major/tensorcore-P·V/tensorcore-2pass
전부 시도: KV-read의 공정·정확 win은 **완전 attention 레벨에선 shared-prefix 대형 M에서도 ~tie가
한계**. P·V만 떼면 win이나 Q·K가 상쇄. (가능한 개선: Q·K를 bf16-staging 없는 scalar/wide(=tie)로
바꾸면 full attention이 ~0.93까지 갈 여지 — P·V win 절반만 남음. 미시도, 효과 modest·niche.)
커널/ops/bench 보존(`qk_wmma*`, `pv_wmma*`, `tests/shared_prefix_attn_bench.py`).

# Phase 40 — scalar Q·K 시도: ratio는 뒤집히나 "느린 영역" artifact, best-vs-best는 여전히 tie

Phase 39의 "scalar Q·K(=tie)면 full attention win 회복" 가설을 구현·검증. `qk_scalar_kernel`(+MX):
thread-per-key(연속 키 = full-sector coalesced K read, MSAQ 0.58×), K를 d-block마다 레지스터로 dequant,
M-tile(MQK=32 누적기) dot — **bf16 staging 없음**(WMMA Q·K를 unpack-bound로 만든 tax 제거). env
`MS_QK_SCALAR=1`. 정확도 rel 2.4e-3.

**결과:** ratio가 **전 구간 win(0.78~0.92)** 으로 뒤집힘. **그러나 absolute가 2~3× 느림** —
Llama Lk4096 M128: scalar MSAQ 541µs / MXINT8 619µs vs **WMMA MSAQ 243µs / MXINT8 241µs(tie)**.
scalar Q·K는 텐서코어를 안 써(M×Lk×D scalar FMA) 둘 다 ~2.3× 느린 영역으로 떨어뜨리고, 그 영역에서만
K-byte 절감이 드러나 ratio가 win.

**best-vs-best(각 포맷 자기 최선):** MSAQ 최선 = min(WMMA 243, scalar 541) = 243(WMMA); MXINT8 최선 =
min(WMMA 241, scalar 619) = 241(WMMA) → **243/241 = tie(1.008)**. scalar은 MSAQ에게도 항상 더 느려
best가 아니므로 ratio-win이 best-vs-best에 안 들어온다(pre-split-K P·V·batch-B=1과 동일한 "느린 영역
ratio-win" artifact).

**최종 확정(8 lever).** fast = 텐서코어 = bf16 staging = tie / ratio-win = 느린 scalar 영역(아무도 안
고름) — 동시 불가. 공정(best-vs-best)·정확 MSAQ KV-read decode win은 **존재하지 않음**(스칼라/staging/
warp-transpose/batch/channel-major/텐서코어-P·V/텐서코어-2pass/scalar-Q·K 전부 시도 완료). 진짜 잔여
가능성은 native sub-byte tensor-core(하드웨어 미지원)뿐. MSAQ의 실제 가치는 weight 경로(GEMV), KV-read는
tie로 최종 확정. 커널/ops/bench 보존(`qk_scalar*`, env `MS_QK_SCALAR`).

# Phase 41 — single-token MHA decode WIN: Phase 32의 wide-scalar "tie"를 정정 (0.97×)

Phase 32–40은 "best-vs-best tie / 불가능"으로 닫았으나, 그 결론은 (a) fast=텐서코어=bf16-staging 천장,
(b) ratio-win=스칼라는 2–3× 느린 영역 — 두 전제 위였다. 그러나 **autoregressive single-token decode의
best-vs-best는 텐서코어가 아니라 wide 스칼라**(M=1 → MMA 타일 무의미; 텐서코어 P·V win은 M≥32 shared-
prefix 한정)다. 그 wide 스칼라를 ncu로 재측정하니 병목은 dequant-throughput(0.5× 실효 BW 가설)이 아니라
**Pass-2 staged-V의 L1/TEX shared-transaction**이었다: DRAM 20.7%, **L1/TEX 63.5%**, occupancy 24%
(reg-limited). 즉 BW-bound도 occupancy-bound도 아닌 **shared-traffic-bound**.

4개 lever를 쌓아 정정(u4/gs2, bit-exact test_kv 72/72):
1. **패킹-친화 config**: nibble u4/gs2 — robust(+2.72% PPL)인 최저-aggressive nibble(2 codes/byte). 출발 1.45×.
2. **v8** (`MS_KV_V8`, u4/gs≤2 디폴트): V를 Pass-1서 int8 코드(up·16+sh∈[-120,119])로 복원 staging →
   Pass-2가 MXINT8과 동일(1 int8 read, no bfe, gs shared-plane 제거). →1.25×.
3. **sepsc** (`MS_KV_SEPSC`, u4 디폴트): K dot 인수분해 `Σq·(up·2^u+sh)·s = s·(2^u·Σq·up + Σ_g sh·qg)`,
   qg=query group-sum 1회 precompute → shared항 per-group + scale를 block-level로. →1.02–1.13×.
4. **vt** (`MS_KV_VT`, v8과 함께 디폴트): staged int8 V를 **transposed+padded** `pV8[blk·32·CH+kd·CH+kk]`,
   CH=chunk+4(CH/4 홀수 → bank-conflict-free)로 배치 → 각 스레드의 kk-코드가 연속 → Pass-2가 **int32당
   4코드** read(conflict-free, 4× 적은 shared txn). →**0.97× WIN**.

**결과:** MHA(Hq=Hkv) H8 Lk4096/16384, H16 Lk16384 **전부 0.97× best-vs-best WIN**(MXINT8 wide 대비).
ncu 확인: L1/TEX 63.5→60.2%, Duration 148→135µs. **MSAQ 최초의 공정·정확 KV-read decode win.**

**미해결(정직):** GQA(Hq>Hkv)는 wide 경로가 KV head를 query-head당 1번(=4×) 중복 read해 1.01–1.07×
parity. KV-reuse(design-A `kv_decode_gqa_kernel`)에 v8+sepsc 이식했으나 scalar+full-staging 구조가
occupancy-bound(~25× off roofline)라 wide에 못 미침(1.45×, documented negative). 배치/텐서코어(Phase
35–40) regime은 bf16-staging 천장 가설 그대로 미재검토. 따라서 **Phase 32의 wide single-token tie만
win으로 정정**되고, 다른 regime의 "불가능"은 유지. 부속물: `tests/kv_pack_results.md`(전체 측정),
`tests/kv_pack_bench.py`, 같은 sepsc가 W-only GEMV u3에도 +3–5%(`tests/gemv_sepsc_results.md`).

# Phase 42 — vpack(packed-transposed nibble V staging): GQA도 win, vpack가 디폴트

Phase 41(v8+sepsc+vt)은 MHA만 0.97× win이고 GQA(32/8)는 1.05× loss였다. 사용자 진단: v8이 V를 int8로
*복원 후* staging하므로 0.58× 바이트 이점이 비병목 DRAM(20%)에만 실리고, 병목인 shared(L1/TEX 60%)엔
1.0×라 시간이 0.58×로 안 내려간다. FP6-LLM 사상(한가한 ALU로 dequant 흡수 → memory/shared wall 회피)대로
**v8 결정을 뒤집어 packed sub-byte를 shared에 staging하고 unpack을 Pass-2 레지스터로 미룸**.

구현 = **transposed + nibble-packed** staging: vp[blk][kd][kk]에 2 codes/byte로 packing(write-race
회피 위해 2 keys/thread), CHP=chunk/2 round-up & CHP/4 홀수 → Pass-2가 int32 1개로 8코드 conflict-free
read. up·sh를 각각 nibble plane으로.

**no-fake-win 검증(ncu 트랜잭션을 시간보다 먼저):** shared-load wavefronts **불변**(1,721,826 vs
1,722,816) — up-nibble + sh-nibble 2-plane은 int32 2개/8코드 = v8의 int8 1개/4코드와 동일 로드 수.
**예측한 shared-transaction 감소는 실현되지 않음.** 그런데 Duration은 135→122µs로 감소(L1/TEX
51.4→49.9%). 진짜 이득은 **occupancy** — vpack smem 13 KB vs v8+vt 16.5 KB(packed nibble, int8 패딩
없음) → 블록/SM↑. bit-exact(rel_fro 0.00 vs v8+vt; test_kv 72/72).

**결과(u4/gs2, vpack vs MXINT8):** MHA H8 Lk4096 0.97× · H8 Lk16384 0.83× · H16 Lk16384 0.90×;
**GQA 32/8 Lk4096 0.90× · Lk16384 0.93×**(전부 WIN). 같은 wide 커널 하나가 Hq/Hkv로 MHA·GQA 둘 다
처리 → GQA가 long-context서도 win(occupancy↑로 wide의 L2-cached 4× 중복 read + latency가 더 잘 hide).
design-A KV-reuse 전용 커널은 불필요(occupancy-bound documented negative). vpack를 u4/gs≤2 디폴트로,
v8/vt는 fallback(`MS_KV_VPACK=0`). 정직한 단서: win은 occupancy/smem 효과이지 FP6-LLM 프레이밍이 예측한
shared-transaction 감소가 아님(ncu상 트랜잭션 평탄) — 다만 packed-staging 방향 자체는 옳았다.

**KV-read 최종(정정):** single-token decode에서 MSAQ가 MXINT8을 MHA·GQA 모두 공정·정확하게 이긴다
(스택: nibble u4/gs2 + sepsc + vpack). Phase 32 tie와 Phase 41 GQA parity를 모두 정정. 배치/텐서코어
regime은 3090 native sub-byte MMA 부재로 미해결(Blackwell 과제).

# Phase 43 — 배치 decode를 vpack로 재측정: B<=16 win (Phase 35 정정)

Phase 35는 배치를 "universal loss(1.07-1.26x)"로 닫았으나 그건 vpack 이전 커널. 배치 경로
(kv_decode_attention_batched)는 같은 wide 커널을 쓰므로 vpack를 그대로 상속한다. 재측정(GQA Hq32/Hkv8
= Llama-3.1-8B 구성, bit-exact rel_fro 2e-8):

  Lk4096 : B1 0.86 / B4 0.90 / B8 0.92 / B16 0.99 (WIN) | B32 1.03 (loss)
  Lk16384: B1 0.90 / B4 0.93 / B8 0.94 / B16 1.00 (WIN) | B32 1.01 (loss)

저배치는 둘 다 occupancy-limited(~80-100 GB/s)라 vpack의 occupancy 이득이 win을 만들고, B32에서
MXINT8이 BW-bound(133 GB/s) 도달하는 반면 MSAQ는 dequant-throttle로 97(0.73x)에 saturate → 0.76x
바이트가 시간으로 다 환원 안 돼 slight loss. 즉 crossover가 "항상 loss"에서 "B>=32에서만 loss"로 이동.

텐서코어 regime은 여전히 하드웨어 벽: 3090에 native sub-byte MMA가 없어 두 포맷이 같은 bf16/int8 MMA
입력 타일을 만들면 바이트 이점이 MMA 직전 소멸(vpack 무관). 또한 decode 경로도 아님 — 텐서코어 P.V는
large-M(prefill/shared-prefix)용이고, autoregressive decode는 M=1~small-batch라 scalar+vpack가 이미
win. Blackwell(native MXFP8/FP4)만 여는 과제.

# Phase 44 — B>=32 loss 시도: cooperative-exp(공정), 그러나 MXINT8이 더 이득 (documented negative + 정정)

B>=32 slight-loss를 줄이려 ncu로 병목 재확인: **DRAM 9.5%(BW-bound 아님!), L1/TEX 74.9%, Compute
70.9%** — Phase 43의 "dequant-throttle/BW-bound 0.73x" 프레이밍은 틀렸다. B=32는 compute+shared 포화이고
MSAQ의 loss는 bfe-decode(순수 ALU tax, MXINT8엔 없음)다.

레버: 두 커널 모두 score sc[kk]가 per-key라 m_new가 d-lane 간 동일한데 exp(sc[kk]-m_new)를 32 d-lane이
**32× 중복** 계산. GQA 커널처럼 **cooperative하게 kk당 1회** 계산으로 wide(MSAQ)와 mxint8 decode 둘 다
수정(공정 best-vs-best). bit-exact(test_kv 72/72).

**결과 = documented negative:** 두 커널 다 빨라졌으나 **단순한 MXINT8이 더 이득**(Pass-2에 bfe가 없어
exp가 더 큰 비중) → MSAQ 고배치 비율 오히려 약간 악화(Lk16384 B16 0.998→1.010, B32 1.012→1.019). 즉
cooperative-exp는 B>=32 loss를 못 줄인다. 동시에 Phase 43의 "B<=16 win"이 MXINT8 미최적화에 기댄
fragile 결과였음이 드러남 — 공정(둘 다 최적)하면 **robust win은 B<=8**, B16 ~tie, B32 slight loss.

cooperative-exp는 유지(공정 최적화, no-fake-win; single-token은 occupancy-bound라 무영향, 여전히 win).
**정정된 배치 결론(공정):** robust win B<=8(0.84-0.93x), B>=16 near-tie/slight-loss(1.00-1.04x).
고배치 loss는 near-fundamental — full-occupancy compute+shared 포화에 MSAQ bfe-decode가 DRAM(9.5%)
여유 없이 그대로 노출. single-token MHA/GQA win은 불변.
