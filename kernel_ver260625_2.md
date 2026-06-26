# Kernel ver.260625_2 — 현재 production 커널 지도 (fused quantized TC-GEMM 추가)

[kernel_ver260625.md](kernel_ver260625.md)의 후속. **ver.260625 = ver.3 위에 decode 메모리패턴 2종 수정**
이었고, **ver.260625_2 = 그 위에 B≥16 decode를 fused 양자화 텐서코어 GEMM으로 교체 + KV/W 전 scope 적용
+ 공정 비교용 MXINT8 fused twin**을 올린 버전이다. 골격·포맷·특수화·B≤15 decode·prefill·어텐션은 그대로다.

- 권위 있는 라우팅 = [`tests/harness_batchsweep.py`](tests/harness_batchsweep.py)의 `QLinear.gemv`·`QLinear.fwd`(decode) /
  `QLinear.gemm`(prefill) / `KVCache.attend`(어텐션). 아래 "어느 커널?"은 전부 거기서 역추적.
- 커밋 `99bd0ae` (perf: fused quantized TC-GEMM + fair MXINT8 twin + all-scope migration).
- 빌드: `pip install -e . --no-build-isolation`. 측정: `MS_FAST=1 MS_FUSED_B16=1 MS_FUSED_MINB=16 python tests/e2e_perscope_260625.py`.

---

## 1. ver.260625 → ver.260625_2에서 바뀐 것 (단 한 곳: B≥16 decode 경로)

| 단계 | ver.260625 (이전) | **ver.260625_2 (현재)** |
|---|---|---|
| Decode W-only B≥16 | `ms_dequant_bf16` + cuBLAS (66MB 왕복 → mq/bf 1.14 **패배**) | **`wonly_gemm_fused_skinny`** (11MB 직독 fused, mq/bf 0.93–0.96 **승**) |
| Decode W+A B≥16 | `ms_dequant_bf16` + cuBLAS (동일, mq/bf 1.15 패배) | **`wonly_gemm_fused_skinny`** (bf16-활성화 vehicle 공유 → 공짜 상속) |
| MXINT8 B≥16 (baseline) | `mxint8_dequant_bf16` + cuBLAS (17MB 왕복) | **`mxint8_gemm_fused_skinny`** (공정 비교용 동일 vehicle) |

그 외(prefill, B=1, B=2..15, KV write/append/read)는 ver.260625와 동일.

---

## 2. 신규/변경 production 커널 — 한국어 설명

### ⭐ Decode B≥16 W-only & W+A: `wonly_gemm_fused_skinny<U_,GS_,MT,RA,NT>`
[wa_gemm.cu:821](csrc/wa_gemm.cu#L821) / host `wonly_gemm_fused_skinny_cuda` [:1243](csrc/wa_gemm.cu#L1243) /
op `wonly_gemm_fused_skinny` [pybind.cpp](csrc/pybind.cpp), 디스패치 [harness_batchsweep.py:94](tests/harness_batchsweep.py#L94).

**문제(ver.260625):** B≥16 decode는 `dequant→bf16[K,OUT] + cuBLAS`였는데, 이게 **34MB bf16을 쓰고
cuBLAS가 다시 읽는 66MB 왕복**이라 bf16 단일 read(34MB)보다도 느렸다(dequant 단독 37µs > bf16 GEMM 23µs).
M=16–32 decode GEMM은 intensity≈M ≪ ridge 76 = **가중치 read에 memory-bound** → 11MB만 읽으면 이긴다.

**수정(Marlin/FP6-LLM 코어):** 양자화 가중치(11MB) 평면을 **직접 읽어** K-블록당 shared에 dequant→bf16,
**WMMA 16×16×16**으로 곱한다. `MT=ceil(M/16)` 한 블록이 **모든 M행**을 담당(가중치 1회 read/splitK),
4 warp = 4 N-subtile, splitK로 SM 채움(`gemv_splitk_count` mult≈7→~8). dequant는
`stream_block_dense_bfe`(production column-major 평면, repack 없음, 코드 대부분 단일 bfe). bit-exact(WMMA tol 2e-3).

**adaptive NT (register-resident 정신의 tractable proxy):** `M≥32`이면 **NT=128** — dequant 스레드 64→128(절반
유휴 제거), 워프당 N-subtile 1→2(`acc[MT][2]`), grid.x=OUT/128(블록 절반, splitK 2배). **워프당 누산 fragment
2개가 mma ILP를 늘려 long_scoreboard를 가린다**(완전 register-resident는 mma.sync 재작성 필요 → 보류).
측정: B32 0.81×, B64 0.70× vs NT=64. `M<32`이면 NT=64(B12/16은 NT=128이 타이~소폭 악화). env `MS_FUSED_NT` override.

**W+A도 동일 커널:** 선형 사영은 모든 scope에서 weight-quant + **bf16 활성화**(배포 deq+cuBLAS도 wonly·wa 둘 다
bf16 X를 먹였음; 진짜 W+A INT8-IMMA는 2× 퇴보 → 기각). KV-quant은 어텐션 커널에 있지 선형엔 없음. 따라서
S2/S4/S5/S6도 같은 fused 선형 사용 → 디스패치 한 줄(`p in (msaq_wonly, msaq_wa)`)로 전 scope 적용.

**결과:** **모든 weight-quant scope가 B≥16에서 bf16·MXINT8 동시 승.** ALU(언팩)를 줄여 in-flight load를 늘린다는
원래 목적과 부합 — 다만 fused는 latency/barrier-bound라 §11 register-aligned는 무익(GEMV에서만 승, fused는 타이).

### ⭐ 공정 비교용 MXINT8 twin: `mxint8_gemm_fused_skinny<MT,NT>`
[wa_gemm.cu:895](csrc/wa_gemm.cu#L895) / host [:1286](csrc/wa_gemm.cu#L1286) / op [pybind.cpp:238](csrc/pybind.cpp#L238) /
디스패치 [harness_batchsweep.py:98](tests/harness_batchsweep.py#L98).

**왜 추가했나:** ver.260625에선 MSAQ만 fused고 MXINT8은 옛 deq+cuBLAS라 mq/mx 0.66–0.81이 **불공정**(포맷이 아니라
커널 격차). MXINT8은 byte-aligned라 fused가 **더 쉽다**(sub-byte unpack 없음). 그래서 동일 vehicle로 짠 twin:
`qweight_cm`의 **연속 int8 32개를 직독 → E8M0 scale 곱 → WMMA**. adaptive NT·splitK·combine 모두 MSAQ와 동일 →
**오직 weight 포맷/바이트만 다름**(11MB vs 17MB).

**공정 재측정:** 커널 단독 mq/mx **0.62–0.75**(byte ratio 11/17=0.65와 거의 일치 = 정당한 mantissa-sharing 우위);
E2E total **0.91–0.95**; **mx/bf 1.30→1.02 붕괴**(B≥16 "MSAQ ≫ MXINT8"의 대부분은 커널 격차였음). MSAQ는 공정
비교서도 전 scope 승, 단 honest한 byte-ratio 마진. (상세: [quant_gemm_mechanisms_260626.md §6a](quant_gemm_mechanisms_260626.md))

### `gemm_skinny_combine` [wa_gemm.cu:1101](csrc/wa_gemm.cu#L1101)
splitK partial(`[splitK,M,OUT]` fp32)을 합산 → bf16 Y. MSAQ·MXINT8 fused 공용.

---

## 3. 변경 없는 production 경로 (ver.260625와 동일 — 요약)
- **Prefill (모든 M):** `ms_dequant_bf16` + cuBLAS (compute-bound, bf16 타이). [w_gemv.cu:797]
- **Decode B=1:** `wonly_gemv_wide_uspec` / `wa_gemv_wide_uspec` (memory-bound, 0.5–0.6× bf16). [w_gemv.cu:295/672]
- **Decode B=2..15:** ⭐ver.260625의 `wonly_gemv_batched_uspec` / `wa_gemv_batched_uspec` (shared-activation, bf16 추월). [w_gemv.cu:434/703]
- **KV write/append/read:** `kv_write_kernel` / `kv_append_kernel` / `kv_decode_wide_kernel`. [kv_attention.cu]
  - KV read는 decode_step의 **38%**(실측 17.2/45.2ms @32L,B32). shared-load-BW bound(L1 81%). chunk·GQA·warpT·
    barrier 제거 모두 no-win(documented). 추가 이득은 cp.async+MMA-GQA 재작성 필요 → 보류. ([[msaq-vs-mxint8-kv-decode-state]])
  - **AA(어텐션 활성화 양자화)**: 기본 decode 커널은 Q를 bf16으로 읽어 AA 미적용(S6≡S5). `MS_KV_AA=1`이면
    `kv_decode_wide_kernel`이 Q·P를 (u,gs) MSAQ로 워프병렬 fake-quant → 실제 low-bit Q×P×KV. 측정: AA-off
    byte-identical, AA 오차 rel 0.011(u2/gs8), KV-read +4.5%(decode ~+1.7%). 기본 off(배포는 accuracy-only 최적).

---

## 4. 디스패치 임계값 (E2E 확정)
- **W decode:** B=1 wide GEMV / B=2..11 batched GEMV / **B≥12 fused**(GEMV가 compute-bound 되는 지점; 배포 임계는
  `MS_FUSED_MINB`=16, B12–15도 fused가 약간 빠르나 보수적으로 16). [harness_batchsweep.py:89]
- **MXINT8:** 대칭으로 B≥16 fused twin.
- env: `MS_FUSED_B16=1`(켜기), `MS_FUSED_MINB`(임계), `MS_FUSED_NT`(64/128), `MS_FUSED_SPLITK`, `MS_FUSED_PIPE`(cp.async, large-M 옵션), `MS_FUSED_RA`(register-aligned, 타이).

---

## 5. Documented-negative (이번 버전에서 시도/기각)
- **cp.async double-buffer** (`MS_FUSED_PIPE=1`, `wonly_gemm_fused_skinny_pipe`): small-M 1.41× 느림, large-M만 이득.
- **INT8-IMMA W+A** (`wa_gemm_fused_imma`): 2.05× 느림(MSAQ per-block E8M0가 per-block int32-store+scale+RMW 강제).
- **register-aligned on fused** (`MS_FUSED_RA`): 타이(GEMV-specific 승, fused는 barrier-bound + 20% 바이트 손해).
- **완전 register-resident**: mma.sync 재작성 필요, fused가 이미 이겨 marginal → 보류.
- (KV) chunk<128, GQA, warpT, 별도 prob 버퍼: 전부 no-win(§3, 메모리 기록).

전체 연혁: [change.md](change.md). 메커니즘 분석: [quant_gemm_mechanisms_260626.md](quant_gemm_mechanisms_260626.md),
[subbyte_unpack_analysis_260625.md](subbyte_unpack_analysis_260625.md).
