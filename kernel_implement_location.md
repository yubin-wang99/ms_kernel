# MSAQ 커널 구현 위치 — 어디를 보면 설계를 알 수 있나

각 커널이 "어떻게 설계됐는지"를 이해하려면 보통 **3겹**을 본다:
1. **레이아웃 정의(CPU pack)** — packed plane이 메모리에 어떻게 깔리는지. `ms_lib/pack.py`
2. **device 언팩/분해 primitive** — bit 단위로 element를 꺼내는 인증된 함수. `csrc/core/ms_utils.cuh`
3. **커널 본체** — 그 primitive를 써서 GEMV/GEMM/attention을 수행. 각 `.cu`
그리고 **정답 기준(numpy reference)** `ms_lib/reference.py` + **게이트 테스트** `tests/`.

---

## 0. 공통 토대 (모든 커널이 공유)

| 무엇 | 파일 | 핵심 심볼 |
|------|------|-----------|
| 레이아웃 정의·CPU 패킹 | `ms_lib/pack.py` | `pack_weight`(weight planes), `pack_kv`(KV planes), `pack_weight_mxint8`/`pack_kv_mxint8`(baseline), `decompose`/`reconstruct` |
| **device primitive (element 취급의 핵심)** | `csrc/core/ms_utils.cuh` | `unpack_ms_weight_elem`, `unpack_ms_kv_elem`(+`_u4` fast), `bfe_s32`, `decompose_ms_block`, `pack_codes_lsb`, `e8m0_to_scale`/`e8m0_exp_from_amax`, `gemv_splitk_count`/`kv_split_count` |
| numpy 정답 기준 | `ms_lib/reference.py` | `_msaq_signed_ref`, `wonly_matmul`, `wa_matmul`, `quant_act`, `kv_attention` |
| Python wrapper | `ms_lib/ops.py` | `wonly_gemv`/`wa_gemv`/`wonly_gemm`/`wa_gemm`/`kv_decode_attention` + `mxint8_*` |
| op 등록 | `csrc/pybind.cpp` | `TORCH_LIBRARY(msaq, …)` 스키마 |
| 설계 변천사(왜 이렇게 됐나) | `change.md` | Phase별 시도·실패·결정 로그 |
| 커널별 설계 규칙 요약 | `kernel_ver1.md` | 7종 커널 high-level 설계·최적화·근거 |

> **포맷 한 줄 요약(MSAQ-s):** 32-원소 블록당 E8M0 base scale `sa` 1개 + 원소당 (8−u)-bit
> upper 코드 + gs-원소 그룹당 u-bit shared 코드. 복원 = `(upper·2^u + shared)·sa`.
> 이 "element 꺼내기"가 `unpack_*` primitive에 캡슐화돼 있고, 커널들은 이 함수만 호출한다.

---

## 1. Decode 커널

### W-only GEMV — `csrc/w_gemv.cu`
- 커널: `wonly_gemv_wide_kernel`(template `<bool U4>`), 런처 `wonly_gemv_wide_cuda`.
- 설계: column-major plane(`upper_cm/shared_cm [NB,OUT,UB]`), **thread-per-output-column**,
  u4=wide `uint4` load + nibble `bfe`, u2/u3=word-load + **streaming bit-buffer unpack**, fp32
  누적, **split-K**(`gemv_splitk_count(...,16)`) + `gemv_combine_kernel`.
- (구버전 `wonly_gemv_splitk_kernel`/`wonly_gemv_cpasync_kernel`도 같은 파일에 있음 — A/B용.)
- 게이트: `tests/test_w.py::test_wonly_gemv_vs_oracle`.

### W+A GEMV — `csrc/w_gemv.cu`
- 커널: `wa_gemv_wide_kernel`(template), 런처 `wa_gemv_cuda`.
- 설계: 위 wide GEMV에 **활성화도 양자화**. (a) Stage-0 prepass `ms_launch_quant_act_msaq`
  (정의는 `wa_gemm.cu`)가 x→int8 word `qx`+`sa_exp`, (b) 커널이 블록당 `qx` slice를
  **shared에 1회 stage** 후 **정수 dot** `idot=Σ qw·qx` → 블록 스케일 `·sw·sa` fold.
- 게이트: `tests/test_wa.py::test_wa_gemv_vs_oracle`.

### KV cache dequantize (decode attention) — `csrc/kv_attention.cu`
- 커널: `kv_decode_wide_kernel`(template, **기본 경로**), 보조 `kv_decode_split_kernel`(구버전),
  `kv_decode_cpasync_kernel`(cp.async), `kv_decode_combine_kernel`. 런처 `kv_decode_attention_cuda`.
- 설계: **thread-per-key**(Pass-1 점수: 키별 in-thread q·K dot, wide load), V는 shared에
  staging 후 thread-per-d 누적(Pass-2), online-softmax, **split-K over keys**(`kv_split_count`)
  + combine. GQA(`Hkv`), 고정용량 캐시(`Lcap` stride). u2/u3 Pass-1은 streaming unpack.
- 게이트: `tests/test_kv.py::test_kv_decode_attention_vs_oracle`(+GQA/`Lcap` 게이트).

### KV quantize (decode append) — `csrc/kv_attention.cu`
- 커널: `kv_append_kernel`, 런처 `kv_append_cuda`. thread=(h,blk), 새 토큰을 slot `pos`(stride
  `Lcap`)에 in-place. `decompose_ms_block`+`pack_codes_lsb` 재사용.
- 게이트: `tests/test_kv.py::test_kv_append_vs_pack` / `…_then_decode_vs_oracle`.

---

## 2. Prefill 커널

### W-only GEMM — `csrc/wa_gemm.cu`
- 커널: `wonly_gemm_tiled<TBM,TBN,RTM,RTN>`(shared-mem 타일), `wonly_gemm_wmma`/`_wmma_pipe`
  (BF16 텐서코어). 런처 `wonly_gemm_cuda`(`MS_TILE_CFG` 디스패치).
- 설계: 타일 프롤로그에서 weight를 **한 번 언팩**해 공유메모리 타일에 올리고 M행이 재사용.
- 게이트: `tests/test_w.py::test_wonly_gemm_vs_oracle`.

### W+A GEMM — `csrc/wa_gemm.cu`
- 커널: `wa_imma`(pipelined INT8 IMMA), Stage-0 `quant_act_msaq_kernel`. 런처 `wa_gemm_cuda`
  (2-stage: prepass + IMMA; `MS_WA_FOLD=1`은 legacy fp32-fold `wa_gemm_tiled`).
- 설계: 활성화 MSAQ-s decompose→int8(Stage 0), weight 언팩을 IMMA 프롤로그에 double-buffer
  로 overlap(Stage 1), per-block `·2^sa·sw` epilogue.
- 게이트: `tests/test_wa.py::test_wa_gemm_vs_oracle`.

### KV write (prefill) — `csrc/kv_attention.cu`
- 커널: `kv_write_kernel`, 런처 `kv_write_cuda`. thread-per-token, **nb를 grid.z로**(occupancy).
  `decompose_ms_block`+`pack_codes_lsb`로 token-major plane 생성(decode가 그대로 읽음).
- 게이트: `tests/test_kv.py::test_kv_write_vs_pack` / `…_then_decode_vs_oracle`.

---

## 3. MXINT8 baseline (비교 상대) — `csrc/mxint8.cu`
모든 MSAQ 커널의 짝. 같은 골격, **element 취급(int8 직읽 vs sub-byte 언팩)만 다름**(이 원칙의
검증은 [for_fair_comparison.md]):
`mxint8_gemv`/`mxint8_wa_gemv`/`mxint8_gemm`/`mxint8_wa_gemm`/`mxint8_kv_decode`/
`mxint8_kv_write`/`mxint8_kv_append` + 활성화 prepass `quant_act_kernel`(plain MXINT8).

---

## 4. 측정·실험 코드 — `tests/`
- `benchmark.py` — 커널 단위 warm 측정(`measure_latency`, `_cuda`).
- `harness.py` — Llama-3.1-8B full forward(CUDA-graph, 4 시나리오). `Model`(weight quant)+
  `KVCache`(KV quant)+`run_scenario`.
- `test_w.py`/`test_wa.py`/`test_kv.py` — 정확도 게이트(oracle/byte-exact). `conftest.py`.
- 결과: `harness_results.md`(end-to-end), `change.md`(Phase 로그).

---

## 빠른 길잡이 (한 커널을 이해하려면)
1. `ms_lib/pack.py`에서 그 plane의 **레이아웃**을 본다.
2. `csrc/core/ms_utils.cuh`에서 그 plane을 푸는 **`unpack_*` primitive**를 본다(여기에 element
   취급이 전부 있음).
3. 해당 `.cu`의 **커널 본체**에서 thread mapping·split·누적·scale fold를 본다.
4. `tests/`의 **게이트**로 정답 대비 무엇이 보장되는지 확인한다.
5. `change.md`에서 **왜 그 설계가 됐는지**(이전 시도·실패) 맥락을 얻는다.
