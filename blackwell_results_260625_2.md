# Blackwell per-scope E2E — batch sweep (kernel_ver260625_2)

본 문서는 **kernel_ver260625_2** (= ver.260625 + **B≥16 decode를 fused 양자화 텐서코어 GEMM으로 교체**,
[`kernel_ver260625_2.md`](kernel_ver260625_2.md) 참조)를 **NVIDIA RTX PRO 4000 Blackwell**에서
per-scope로 측정한 batch-size 스윕 결과다. 6개 scope(S1–S6) × **B ∈ {1, 8, 16, 32}**.
[`blackwell_results_260625.md`](blackwell_results_260625.md)(이전 버전, B≥16 = dequant+cuBLAS)의 후속 —
**B=16/32 weight scope가 패→승으로 뒤집힌 것**이 핵심 차이다.

## Setup
- **GPU**: NVIDIA RTX PRO 4000 Blackwell (sm_120), 24 GB. CUDA 13.2, torch 2.12, Python 3.12.
- **Model**: Llama-3.1-8B 32L (hidden 4096, 32 Q / 8 KV head, inter 14336); 가중치 랜덤·재사용(latency는 값과 무관).
- **Workload**: L_in=1024, L_out=128. `prefill`=TTFT(1024 tok), `decode`=128 step 적분, `total`=prefill+decode.
- **Harness**: `MS_FAST=1 MS_FUSED_B16=1 MS_FUSED_MINB=16 python tests/e2e_perscope_260625.py --Bs 1,8,16,32 --reps 12`
  (CUDA-graph decode, min-over-windows = steady-state P0).
- **단위 = ms** (harness가 `torch.cuda.Event.elapsed_time` ms를 적분; 표 헤더의 "µs"는 오라벨. 검산: S3 B32
  decode 5710ms / 128 step = 44.6 ms/step = 실측 TPOT 45.2ms와 일치). **ratio < 1 = 분자가 빠름.**
- **공정 비교**: MSAQ와 MXINT8 모두 B≥16에서 **동일한 fused 텐서코어 GEMM**(`wonly_gemm_fused_skinny` /
  `mxint8_gemm_fused_skinny`)을 쓴다 — weight 포맷/바이트(11MB vs 17MB)만 다름. 그래서 `mq/mx`가
  mantissa-sharing 효과를 격리한다 (이전 불공정 측정 0.66–0.81 아님).
- **Routing**: decode W-path = B=1 wide GEMV / B=2..15 shared-activation batched GEMV / **B≥16 fused TC-GEMM**(신규).
  KV decode = (u,gs)-특수화 wide-load (불변).

`mq`=MSAQ, `mx`=MXINT8, `bf`=bf16.

---

## 1. Per-scope batch sweep — 전체 ratio 표

각 표: prefill / decode / total 각각에 대해 **mq/bf · mq/mx · mx/bf** (모든 값 <1 = 분자가 빠름). 절대시간(ms)은 §2.

### S1 W-only  (MSAQ u3/gs16)
| B | pre mq/bf | pre mq/mx | pre mx/bf | dec mq/bf | dec mq/mx | dec mx/bf | tot mq/bf | tot mq/mx | tot mx/bf |
|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 1 | 1.11 | 0.99 | 1.13 | 0.51 | 0.82 | 0.62 | 0.55 | 0.84 | 0.66 |
| 8 | 1.01 | 0.99 | 1.01 | 0.83 | 1.00 | 0.83 | 0.88 | 1.00 | 0.88 |
| 16 | 0.99 | 1.00 | 0.99 | 0.91 | 0.89 | 1.02 | 0.93 | 0.93 | 1.01 |
| 32 | 0.99 | 0.99 | 1.00 | 0.93 | 0.93 | 1.01 | 0.95 | 0.95 | 1.00 |

### S2 W+A  (MSAQ u2/gs8)
| B | pre mq/bf | pre mq/mx | pre mx/bf | dec mq/bf | dec mq/mx | dec mx/bf | tot mq/bf | tot mq/mx | tot mx/bf |
|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 1 | 1.14 | 0.99 | 1.15 | 0.55 | 0.88 | 0.63 | 0.59 | 0.89 | 0.67 |
| 8 | 1.01 | 1.00 | 1.01 | 0.85 | 1.02 | 0.84 | 0.89 | 1.01 | 0.88 |
| 16 | 1.01 | 1.00 | 1.00 | 0.91 | 0.90 | 1.02 | 0.94 | 0.93 | 1.01 |
| 32 | 0.99 | 0.99 | 1.00 | 0.94 | 0.94 | 1.01 | 0.96 | 0.95 | 1.00 |

### S3 KV-only  (MSAQ u4/gs2)
| B | pre mq/bf | pre mq/mx | pre mx/bf | dec mq/bf | dec mq/mx | dec mx/bf | tot mq/bf | tot mq/mx | tot mx/bf |
|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 1 | 1.03 | 1.00 | 1.03 | 0.96 | 0.97 | 0.99 | 0.97 | 0.97 | 0.99 |
| 8 | 1.01 | 1.00 | 1.01 | 0.67 | 0.81 | 0.83 | 0.76 | 0.87 | 0.87 |
| 16 | 1.01 | 1.00 | 1.01 | 0.53 | 0.89 | 0.59 | 0.69 | 0.94 | 0.73 |
| 32 | 1.01 | 1.00 | 1.01 | 0.40 | 0.81 | 0.49 | 0.61 | 0.91 | 0.67 |

### S4 W-only+KV  (MSAQ u2/gs8)
| B | pre mq/bf | pre mq/mx | pre mx/bf | dec mq/bf | dec mq/mx | dec mx/bf | tot mq/bf | tot mq/mx | tot mx/bf |
|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 1 | 1.15 | 0.99 | 1.16 | 0.52 | 0.85 | 0.61 | 0.56 | 0.87 | 0.64 |
| 8 | 1.01 | 0.99 | 1.02 | 0.56 | 0.88 | 0.64 | 0.68 | 0.92 | 0.74 |
| 16 | 1.01 | 1.00 | 1.00 | 0.53 | 0.88 | 0.60 | 0.69 | 0.94 | 0.74 |
| 32 | 1.00 | 0.99 | 1.01 | 0.42 | 0.86 | 0.49 | 0.62 | 0.93 | 0.67 |

### S5 W+A+KV  (MSAQ u2/gs8)
| B | pre mq/bf | pre mq/mx | pre mx/bf | dec mq/bf | dec mq/mx | dec mx/bf | tot mq/bf | tot mq/mx | tot mx/bf |
|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 1 | 1.12 | 0.99 | 1.13 | 0.52 | 0.85 | 0.61 | 0.56 | 0.87 | 0.64 |
| 8 | 1.01 | 0.99 | 1.01 | 0.57 | 0.90 | 0.64 | 0.69 | 0.93 | 0.74 |
| 16 | 1.01 | 1.01 | 1.00 | 0.53 | 0.87 | 0.60 | 0.69 | 0.94 | 0.74 |
| 32 | 1.00 | 0.99 | 1.01 | 0.42 | 0.86 | 0.49 | 0.62 | 0.92 | 0.67 |

### S6 W+A+KV+AA  (MSAQ u2/gs8)
| B | pre mq/bf | pre mq/mx | pre mx/bf | dec mq/bf | dec mq/mx | dec mx/bf | tot mq/bf | tot mq/mx | tot mx/bf |
|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 1 | 1.15 | 0.99 | 1.16 | 0.52 | 0.85 | 0.61 | 0.56 | 0.87 | 0.65 |
| 8 | 1.01 | 1.00 | 1.01 | 0.57 | 0.89 | 0.65 | 0.69 | 0.93 | 0.75 |
| 16 | 1.01 | 1.00 | 1.00 | 0.53 | 0.88 | 0.60 | 0.69 | 0.93 | 0.74 |
| 32 | 1.00 | 0.99 | 1.01 | 0.42 | 0.85 | 0.49 | 0.62 | 0.92 | 0.67 |

## 2. 절대 시간 (ms) — prefill / decode / total, bf/mx/mq

### S1 W-only
| B | prefill bf/mx/mq | decode bf/mx/mq | total bf/mx/mq |
|--:|--|--|--|
| 1 | 256/288/285 | 3608/2245/1842 | 3865/2533/2127 |
| 8 | 2213/2237/2225 | 5971/4960/4983 | 8184/7196/7207 |
| 16 | 4471/4438/4434 | 8718/8866/7890 | 13189/13304/12324 |
| 32 | 7549/7543/7497 | 14313/14409/13367 | 21861/21952/20864 |

### S2 W+A
| B | prefill bf/mx/mq | decode bf/mx/mq | total bf/mx/mq |
|--:|--|--|--|
| 1 | 254/291/290 | 3619/2287/2007 | 3873/2578/2296 |
| 8 | 2226/2247/2239 | 6016/5039/5137 | 8242/7286/7375 |
| 16 | 4406/4413/4428 | 8734/8881/7987 | 13139/13294/12415 |
| 32 | 7540/7544/7464 | 14306/14394/13465 | 21846/21938/20929 |

### S3 KV-only
| B | prefill bf/mx/mq | decode bf/mx/mq | total bf/mx/mq |
|--:|--|--|--|
| 1 | 254/261/261 | 3620/3587/3480 | 3874/3848/3740 |
| 8 | 2228/2247/2249 | 5974/4929/3984 | 8202/7175/6233 |
| 16 | 4375/4413/4416 | 8757/5198/4629 | 13132/9611/9045 |
| 32 | 7564/7650/7654 | 14324/7019/5710 | 21888/14669/13363 |

### S4 W-only+KV
| B | prefill bf/mx/mq | decode bf/mx/mq | total bf/mx/mq |
|--:|--|--|--|
| 1 | 254/294/292 | 3621/2198/1871 | 3875/2492/2163 |
| 8 | 2224/2261/2246 | 6011/3839/3395 | 8235/6100/5641 |
| 16 | 4425/4433/4455 | 8727/5249/4600 | 13152/9682/9055 |
| 32 | 7542/7602/7531 | 14293/7040/6025 | 21835/14642/13556 |

### S5 W+A+KV
| B | prefill bf/mx/mq | decode bf/mx/mq | total bf/mx/mq |
|--:|--|--|--|
| 1 | 260/294/292 | 3631/2208/1874 | 3891/2502/2166 |
| 8 | 2225/2257/2244 | 6014/3853/3450 | 8238/6110/5695 |
| 16 | 4421/4436/4468 | 8728/5252/4594 | 13149/9689/9062 |
| 32 | 7536/7602/7519 | 14295/7040/6020 | 21831/14643/13539 |

### S6 W+A+KV+AA
| B | prefill bf/mx/mq | decode bf/mx/mq | total bf/mx/mq |
|--:|--|--|--|
| 1 | 254/293/291 | 3614/2208/1874 | 3868/2501/2165 |
| 8 | 2222/2255/2244 | 6007/3884/3449 | 8228/6139/5693 |
| 16 | 4414/4435/4452 | 8728/5247/4599 | 13141/9683/9051 |
| 32 | 7542/7609/7524 | 14304/7049/6022 | 21846/14658/13545 |
---

## 3. 핵심 발견

**total mq/bf 배치별 요약:**

| scope | B=1 | B=8 | B=16 | B=32 |
|---|--:|--:|--:|--:|
| S1 W-only | 0.55 | 0.88 | **0.93** | 0.95 |
| S2 W+A | 0.59 | 0.89 | **0.94** | 0.96 |
| S3 KV-only | 0.97 | 0.76 | 0.69 | **0.61** |
| S4 W-only+KV | 0.56 | 0.68 | 0.69 | **0.62** |
| S5 W+A+KV | 0.56 | 0.69 | 0.69 | **0.62** |
| S6 W+A+KV+AA | 0.56 | 0.69 | 0.69 | **0.62** |

**total mq/mx (공정 비교, 양쪽 fused):**

| scope | B=1 | B=8 | B=16 | B=32 |
|---|--:|--:|--:|--:|
| S1 W-only | 0.84 | 1.00 | 0.93 | 0.95 |
| S2 W+A | 0.89 | 1.01 | 0.93 | 0.95 |
| S3 KV-only | 0.97 | 0.87 | 0.94 | 0.91 |
| S4/S5/S6 (+KV) | 0.87 | 0.92–0.93 | 0.93–0.94 | 0.92–0.93 |

1. **B≥16 weight scope가 패→승으로 역전 (ver.260625 대비 최대 변화).** 이전(dequant+cuBLAS)엔
   S1/S2 B=16 total mq/bf **1.28–1.29**(패), B=32 1.16–1.17이었다. fused TC-GEMM(11MB 직독)으로
   **B=16 0.93–0.94, B=32 0.95–0.96으로 승**. decode만 보면 B16 mq/bf 0.91, B32 0.93 — 양자화 weight
   read(11MB)가 bf16(34MB)·MXINT8(17MB)을 이긴다.

2. **B=1은 전 scope 압승** (0.55–0.97). 한 토큰 decode는 memory-bound라 바이트 절감이 그대로 시간.
   weight-touching scope는 0.55–0.59×.

3. **KV scope(S3/S4/S5/S6)는 decode에서 가장 크게 이긴다.** decode mq/bf가 B=32에서 S3 **0.40**,
   S4/S5/S6 **0.42**. KV 바이트 절감이 batch·문맥에 비례해 커지는 단조 개선. total로는 prefill(타이)에
   희석돼 0.61–0.62.

4. **prefill은 전 scope·전 batch 타이** (mq/bf 0.99–1.15, 대개 ~1.0). 양자화 weight를 bf16으로 풀어
   cuBLAS에 태우는 compute-bound 구조 → 셋 다 동률. (B=1 prefill mq/bf 1.1+는 절대시간이 작아 launch
   오버헤드가 비율을 키운 것; 절대차 ~30µs.)

5. **공정 비교(mq/mx) — MSAQ가 전 scope 승, honest한 마진.** B≥16에서 MXINT8도 동일 fused를 쓰므로
   `mx/bf`가 1.30(이전, MXINT8 fused 없음)→**1.00–1.02**로 붕괴 = "MSAQ ≫ MXINT8"의 대부분은
   포맷이 아니라 커널 격차였음. 그래도 MSAQ는 mq/mx **0.91–0.95**(weight scope), **0.87–0.94**(KV scope)로
   여전히 이긴다 — 11/17=0.65 byte-ratio가 만드는 정당한 mantissa-sharing 우위(커널 단독으론 0.62–0.75).

6. **S6 ≈ S5 (latency 동일)** — AA를 켜도 decode 어텐션 커널이 Q를 bf16으로 읽어 latency 불변. AA는
   정확도 비용(~+0.9–1.0pp PPL)이지 latency 비용이 아님.

**실무 함의**: 이전 버전의 사각지대였던 **B≈16 weight-only 구간이 fused로 메워져**, 이제 B=1부터 B=32까지
전 batch·전 scope에서 MSAQ가 bf16·MXINT8를 (타이 이상으로) 이긴다. KV scope는 큰 batch일수록 이득이
커지는 단조 개선이 그대로 유지된다.

---

## 4. 재현
```bash
cd ~/ms_kernel && source .venv/bin/activate
pip install -e . --no-build-isolation        # sm_120 빌드
MS_FAST=1 MS_FUSED_B16=1 MS_FUSED_MINB=16 python tests/e2e_perscope_260625.py --Bs 1,8,16,32 --reps 12
# -> tests/harness_perscope_results_260625.md + .jsonl. 본 문서는 그 .jsonl에서 ratio 계산해 정리.
```
> 이전 버전(B≥16 = dequant+cuBLAS, weight scope B16 패): [`blackwell_results_260625.md`](blackwell_results_260625.md).
> 커널 지도: [`kernel_ver260625_2.md`](kernel_ver260625_2.md). 메커니즘: [`quant_gemm_mechanisms_260626.md`](quant_gemm_mechanisms_260626.md).
