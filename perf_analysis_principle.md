# 성능 분석 원칙 — "쪼개고, roofline으로 해석한다"

이 repo에서 커널/연산 성능을 분석할 때의 **고정 절차**. total 시간만 보면 어디서 이기고 지는지
숨는다. 항상 **(a) 구성 커널별로 시간을 쪼개고**, **(b) roofline으로 compute-bound인지
memory-bound인지 판정한 뒤**, **(c) 어느 lever가 통하는지** 해석한다.

양자화 연산은 보통 여러 커널의 합이다:
`dequant`(또는 weight unpack) + `activation-quant`(W+A) + `GEMM/GEMV` + `combine/epilogue`.
각 조각의 bound가 다르므로 **반드시 따로 측정**한다.

---

## 절차 (매 분석마다)

### 1. 연산을 구성 커널로 쪼개고 각각 따로 측정
- 예: `dequant+cuBLAS prefill` = `ms_dequant_bf16`(메모리-bound) + `cuBLAS GEMM`(compute-bound).
- 예: `W+A GEMM` = `quant_act`(activation int8화, 메모리-bound) + `IMMA mainloop`(compute) + `epilogue scale`.
- 각 커널을 **CUDA event로 단독 벤치**(warmup → P0 ramp → min-of-N). total만 재지 말 것.

### 2. roofline 수치를 직접 계산
한 연산에 대해:
```
arithmetic_intensity = FLOPs / bytes_moved          # FLOP per byte
ridge_point          = peak_compute / peak_BW        # 하드웨어 상수
  RTX 3090: bf16 71 TFLOPS / 936 GB/s = 76 FLOP/byte;  INT8 142 TFLOPS / 936 = 152 FLOP/byte
compute_floor_us = FLOPs / peak_compute * 1e6
mem_floor_us     = bytes_moved / peak_BW   * 1e6
```
- `intensity > ridge` → **compute-bound** (연산이 한계; FLOP을 줄이거나 더 빠른 연산 포맷이 lever).
- `intensity < ridge` → **memory-bound** (바이트가 한계; 바이트를 줄이는 게 lever — 양자화가 빛나는 곳).

### 3. 측정값을 floor와 비교 (efficiency)
- `efficiency = floor / measured`. bf16 cuBLAS는 보통 ~85–90% (튜닝 기준선). 커스텀 커널이 30–40%면
  커널 비효율(타일링/파이프라인/점유율) 문제.
- ncu로 한 단계 더: `sm__pipe_tensor_op_hmma/imma_cycles_active`(텐서코어 활용 %),
  `dram__throughput.%`, `sm__throughput.%`, `l1tex__throughput.%`, `sector-util%`.
  - 텐서코어 % 낮음 → MMA가 굶고 있음(dequant/staging이 직렬화). DRAM% 낮고 SM/L1 높음 → compute/LSU-bound.

### 4. 해석 — "어느 lever가 통하는가"
| 병목 | lever | 양자화가 통하나 |
|---|---|---|
| memory-bound (intensity < ridge) | 바이트↓ | **예** — 8-bit가 시간으로 직결 (decode) |
| compute-bound (intensity > ridge) | FLOP↓ 또는 *연산 포맷* 2× (INT8 IMMA) | **바이트는 무의미**; INT8 텐서코어 2× FLOP만 lever (prefill) |
| launch-bound (일감 작음) | 융합/배치 | 둘 다 무의미 (kv_append) |
| kernel-inefficiency (floor 대비 30%) | 타일링/파이프/점유율 또는 cuBLAS/CUTLASS | 포맷 무관 |

**핵심**: 양자화의 무기는 두 가지로 분리해서 봐야 한다 —
- **바이트 절감**(memory advantage): memory-bound에서만 시간으로 전환.
- **연산 포맷**(INT8 텐서코어 2× FLOP): compute-bound에서만 lever.
total만 보면 이 둘이 섞여 오해한다.

---

## Worked example: prefill GEMM (M=1024, K=OUT=4096, RTX 3090)

쪼갠 측정:
| 조각 | 시간 | 해석 |
|---|---|---|
| 연산 floor (2·M·K·OUT / 71 TFLOPS) | 484 µs | — |
| 가중치 read floor (34 MB / 936 GB/s) | 36 µs | 전체의 **7%** |
| bf16 cuBLAS GEMM (실측) | 558 µs | eff 87% |
| dequant 커널 | 56 µs | 메모리-bound |
| **dequant + cuBLAS** | **578 µs** | = bf16 GEMM + dequant |
| int8 GEMM (`torch._int_mm`) | 638 µs | eff 38% (튜닝 안 됨) |

판정: **intensity 1024 ≫ ridge 76 → 13.5× compute-bound.**
- 8-bit가 줄이는 가중치 바이트(36µs)는 전체의 7% → 절반 줄여도 ~4% → **bf16과 동등**. 바이트 advantage가
  compute-bound라 무의미.
- 이기려면 **INT8 텐서코어 2× FLOP**(연산 floor 484→242µs) — 단 튜닝된 int8 커널 + (MX는) block-scaled
  IMMA 필요. `torch._int_mm`은 eff 38%라 못 살림.
- 대조: **decode**는 intensity ≈ M = 1 < ridge → memory-bound → 8-bit가 시간 절반(B=1 win 0.54).

이 사례가 원칙의 표준 적용이다: **쪼개니** dequant+GEMM이 분리되고, **roofline 계산하니** 7%·13.5×가
보이고, **해석하니** "바이트 무의미, INT8 FLOP만 lever"가 나온다.

---

## 체크리스트 (복붙용)
```
[ ] 연산을 구성 커널로 나눴는가? 각각 단독 벤치(warmup+min-of-N)?
[ ] intensity = FLOPs/bytes, ridge = peak/BW 계산?  compute vs memory bound 판정?
[ ] compute_floor, mem_floor 계산하고 measured와 efficiency 비교?
[ ] ncu: 텐서코어%, DRAM%, SM%, L1%, sector-util%  — 병목 단계 확인?
[ ] lever 매핑: memory→바이트, compute→FLOP/포맷, launch→융합, ineff→타일링/cuBLAS?
[ ] 양자화의 "바이트 절감"과 "연산 포맷(INT8 2×)"을 분리해서 결론냈는가?
```

관련: [packing_explained.md] §7–§12(진단 사례), [kernel_ver3.md](최적화 종합),
[compile_time_optimization.md](특수화 메커니즘).
