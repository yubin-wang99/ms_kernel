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

## 추출 잔여 (follow-up 후보)
- 추출 잔여 14us를 vectorized int4 dequant(Marlin식 lop3/prmt)로 더 줄이면 0.41×까지 여지.

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
