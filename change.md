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

**2. W-only GEMM (prefill M=512)** — SOTA: shared-mem tiled GEMM(P9)+M-adaptive tile 64²/128²(P10)
| u | MSAQ | MXINT8 | cuBLAS | MSAQ/MX |
|---|------|--------|--------|---------|
| u2 | 2560 | 2401 | 284 | 1.07 ❌ |
| u3 | 2542 | 2408 | 287 | 1.06 ❌ |
| u4 | 2227 | 2427 | 287 | **0.92 ✅** |

**3. W+A GEMM (prefill M=512)** — SOTA: tiled GEMM + on-the-fly MXINT8 activation quant fold(P9)
| u | MSAQ | MXINT8 | cuBLAS | MSAQ/MX |
|---|------|--------|--------|---------|
| u2 | 3685 | 3574 | 286 | 1.03 ❌ |
| u3 | 3632 | 3574 | 286 | 1.02 ❌ |
| u4 | 3335 | 3574 | 287 | **0.93 ✅** |

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
- **GEMM/W+A u2/u3 (1.02–1.07) = compute(FMA) bound**: tiled가 unpack을 타일당 1회로 분할상환해
  memory-bound를 벗어남 → 바이트 절약이 안 보이고, u2/u3의 무거운 straddle unpack만 근소 열위로 남음.
  (참고: GEMV처럼 streaming unpack을 B-tile staging에 이식하면 줄어들 여지 — 미시도.)
- **KV가 u2/u3도 넘는 이유(대조)**: token-major라 key가 contiguous → key-per-thread가 u2/u3도
  coalesced. GEMV out-innermost와 달리 레이아웃이 정렬에 유리.
- **남은 레버**: GEMM/W+A u2/u3 → tensor-core(IMMA/WMMA, P11 WMMA는 현재 opt-in) 또는 streaming
  unpack 이식. (GEMV·KV는 전 u crossover 완료.)

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
