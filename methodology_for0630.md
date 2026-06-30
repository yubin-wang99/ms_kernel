# Methodology + Results (0630) — 운용 가능 Max Batch 증가 및 RPS 전환 측정

> 목표: 기존 RPS_results.md(고정 batch에서 RPS)를 확장하여 **(i) 메모리 절감이 운용 가능 max batch를
> 실제로 올리는가**, **(ii) 그 batch 증가분이 RPS 상승으로 전환되는가**를 동일 세팅·동일 커널에서
> 관찰·정량. 분석적 근거는 `final_methodology.md` §2(B_max)·§3(ridge)·§4(iso-batch vs peak-to-peak).
> **구현·측정 완료 (2026-06-30, `rps_iso_memory_bench.py`).** 결과는 §결과 참조.

## Hardware / Software / Model (기존과 동일)

- **Hardware**: NVIDIA RTX PRO 4000 (Blackwell, 70 SM, VRAM 24 GB, peak HBM BW 672 GB/s). GPU 0 단일.
- **Software**: CUDA 13.2.
- **Model**: Llama-3.1-8B (`n_layer=32, d_model=4096, n_kv=8 GQA, d_head=128, d_ff=14336, V≈128k, P_w≈8.03B`).

## Workload configuration (기존과 동일)

- conversation anchor **input 1,024 / output 128 tokens** (Azure median, KV block 경계 정렬 위해 32 배수) [1]. `L_seq = 1,152`, `O = L_out = 128`.

## Kernel comparison (기존과 동일 — RPS_results.md를 뽑은 그 커널)

- **BF16 reference**: Linear=cuBLAS, Attention=SDPA.
- **MXINT8 (baseline)**: element 처리만 제외하고 kernel 구조를 mantissa-sharing kernel과 동일. (`OPS.mxint8_kv_decode_batched`)
- **Mantissa-shared kernel**: quant 시 unshared/shared bits 분리·plane packing; dequant 시 두 plane load 후 INT8 code 합치는 unpacking. (`OPS.kv_decode_attention_batched`, `pack_kv`)

## Mantissa-sharing parameter (BF16 PPL ≤3% 오차 하 최소 bpe)

- **KV**: (u, gs) = **(3, 16)**, 5.44 bpe.
- **Weight+KV**: (u, gs) = **(2, 8)**, 6.5 bpe.
- (본 0630 실험 scope는 위 둘로 한정. Weight-only는 보조.)

## Cases

각 case에 MXINT8 / mantissa-shared 적용 (BF16은 무양자화 reference):
- **Case KV**: KV만 양자화 (weights BF16 16 GB).
- **Case Weight+KV**: weight·KV 동시 양자화.

---

## 측정 설계 — iso-memory B_max → RPS 분해 (구현: `rps_iso_memory_bench.py`)

기존 고정 batch {1,8,16,32} 대신, **각 format을 *자기 운용 max batch* `B_max`에서** 측정 (iso-batch가
아니라 **iso-memory** operating point — 공유 메모리 예산 `M_avail` 하에서 각 format이 담을 수 있는 최대
동시 request 수). RPS를 두 채널로 **분해**해, 이득의 출처를 분리:

```
RPS  =  B_max / (O · t_step)
        └ capacity 채널 ┘  └ latency 채널 ┘
```

- **B_max (capacity 채널)** — `final_methodology.md` §2 닫힌형 `B_max = (M_avail − W) / (L_seq·κ)` (분석)
  **+ 실제 GPU alloc 검증**(weights + B_max개 request의 KV가 실제로 할당되는지 OOM 없이 확인). MSAQ의
  바이트 절감이 **무조건** 여기로 전환됨(seq당 바이트↓ → 더 많은 seq, 순수 산술 — bandwidth/latency 역설 없음).
  `M_avail = 0.86 × 24 GB = 20.6 GB` (gpu_mem_util 0.9 × paged-KV usable ~0.96).
- **t_step (latency 채널)** — 실측 커널. decode 1 layer = linear GEMM(M=B, bf16 cuBLAS) + format별 KV-read
  attention. 1 layer 측정 × `n_layer`(한 시점에 한 layer footprint만 상주하므로 B_max batch가 timing에 fit).
  reps=3 median(140 W clock drift 완화).
- **정규화**: BF16 reference(W16/KV16, 전 case 공통)를 1회 측정·재사용 → case 간 clock drift 오염 없음.

### 두 질문 ↔ 두 채널
- **Q1 "max batch 증가?"** → **B_max 채널** (`B_max(MSAQ)/B_max(MXINT8)`, `/BF16`).
- **Q2 "그만큼 RPS 증가?"** → **RPS = B_max채널 × t_step채널** 분해. RPS 이득이 B_max 채널로 추적되면 전환 성립.

---

## 결과 (2026-06-30, `rps_iso_memory_0630.txt` / `rps_iso_memory_results.jsonl`)

`# BF16 reference (공통): B_max=30, t_step=51.46 ms, RPS=4.55`

### Case KV — (3,16) = 5.44 bpe

| format | b_kv | **B_max** | t_step (ms) | tok/s | RPS | RPS/BF16 | **B_max/BF16** |
|---|---:|---:|---:|---:|---:|---:|---:|
| BF16 | 16.0 | 30 | 51.5 | 583 | 4.55 | 1.00× | 1.00× |
| MXINT8 | 8.25 | 58 | 62.0 | 936 | 7.31 | 1.61× | 1.93× |
| **MSAQ** | **5.44** | **89** | 58.4 | 1525 | **11.91** | **2.62×** | **2.97×** |

**vs MXINT8: RPS 1.63× = B_max 채널 1.53× × t_step 채널 1.06×**

### Case Weight+KV — (2,8) = 6.5 bpe

| format | b_w/b_kv | **B_max** | t_step (ms) | tok/s | RPS | RPS/BF16 | **B_max/BF16** |
|---|---:|---:|---:|---:|---:|---:|---:|
| BF16 | 16.0 | 30 | 51.5 | 583 | 4.55 | 1.00× | 1.00× |
| MXINT8 | 8.25 | 158 | 136.0 | 1162 | 9.08 | 1.99× | 5.27× |
| **MSAQ** | **6.5** | **230** | 126.7 | 1815 | **14.18** | **3.11×** | **7.67×** |

**vs MXINT8: RPS 1.56× = B_max 채널 1.46× × t_step 채널 1.07×**

### 해석 — 두 질문 모두 YES, 전환은 거의 1:1

1. **운용 max batch 증가** (Q1): MSAQ가 MXINT8 대비 **1.53×(KV)·1.46×(W+KV)**, BF16 대비 **2.97×·7.67×**.
   b_kv(·b_w) 절감의 순수 capacity 효과.
2. **그 증가분의 RPS 전환** (Q2): RPS 이득(1.63×·1.56×)을 분해하면 **B_max 채널(1.53×·1.46×)이 지배**,
   t_step 채널은 1.06~1.07×뿐 → **max batch 증가가 RPS로 ~1:1 전환됨.**
3. **왜 전환되는가** (`final_methodology.md` §3.4의 "1024/128 plateau로 전환 0" 우려와 달리): iso-memory
   에선 MSAQ가 더 많은 seq를 담으면서도 *총 KV-read 바이트가 비슷*(KV `89×5.44 ≈ 58×8.25`)해 t_step이
   거의 안 늘어 → capacity 헤드룸이 깨끗이 RPS로 전환. (latency 역설은 t_step 채널에 갇히고, capacity
   채널은 무조건 전환.)

---

## 실측 OOM 검증 (2026-06-30, `oom_sweep_0630.py` / `oom_sweep_0630.txt`)

분석 B_max를 **경험적 하드웨어 cliff로 검증**: 각 (case, format)에서 weights + B개 request의 **전레이어
KV**를 실제 GPU에 할당하며 B를 OOM까지 step-up(coarse 16→fine 4→1), last-stable B = **실측 B_max**.
실측 B_max에서 **진짜 decode attention step 실행 확인**(activation transient 포함, `decode@emp`).

| case | format | 분석 B_max (operational) | **실측 B_max (raw OOM)** | decode@실측 |
|---|---|---:|---:|:--:|
| KV | BF16 | 30 | 58 | n/a |
| KV | MXINT8 | 58 | 113 | ok |
| KV | **MSAQ (3,16)** | 89 | **172** | ok |
| W+KV | MXINT8 | 158 | 213 | ok |
| W+KV | **MSAQ (2,8)** | 230 | **299** | ok |

**실측 max-batch 비율 (Q1을 실측으로 확정):**
- **KV: MSAQ 172 / MXINT8 113 = 1.52×** (vs BF16 = 2.97×) — 분석 1.53×와 일치.
- **W+KV: MSAQ 299 / MXINT8 213 = 1.40×** (vs BF16 = 5.16×) — 분석 1.46×와 근접.

세 가지 확인: (1) **실측 비율 ≈ 분석 비율**(BF16 대비 2.97×는 정확 일치) → allocator 아티팩트 아님,
모델 검증됨. (2) **실측 절대값 > 분석값**(172 vs 89): 분석은 operational cap(`M_avail=0.86×24=20.6GB`,
vLLM gpu_mem_util 모사), 실측은 raw 하드웨어(~24GB) → **분석값은 보수적 운용값**, 실측은 하드 천장(둘 다
제시하면 "0.9 cap이 임의적" 반론도 닫힘). (3) **decode@실측=ok** → static alloc만이 아니라 운용 가능.

## 정직성 / 리뷰어 방어 (caveats)

- **B_max 二重 보고**: (i) **operational** = 분석식 `(M_avail−W)/(L_seq·κ)` + static-alloc 검증(§결과
  표의 89/230, RPS 계산에 사용 — vLLM gpu_mem_util 모사), (ii) **empirical** = raw-device OOM sweep
  cliff(위 표의 172/299, decode 실행 확인). 비율은 둘이 일치(1.52×/1.40×), 절대값은 (ii)>(i).
- **t_step = 레이어당 실측 커널 × n_layer**(한 layer만 상주시켜 큰 batch timing). RPS는 operational
  B_max에서 계산(보수적). linears는 전 format bf16 cuBLAS(weight-quant fused 경로엔 보수적) → weight
  quant는 t_step에 capacity로만 기여, 과대평가 없음.
- **regime 진단**: t_step 채널이 작다(1.06×)는 것은 step-latency가 ridge/plateau-bound(`B_max>ridge
  B*≈81`)임을 뜻하나, **RPS 전환은 capacity(B_max) 채널에서 발생**하므로 plateau여도 전환 성립 —
  `final_methodology.md` §3.4의 "plateau면 이득 없음"을 정밀화(plateau는 t_step 채널만 무력화).
- **confound 통제**(§7): 동일 GPU·`M_avail`·동일 kernel 구조(element 처리만 차이), BF16 1회 측정 공유 정규화.
- **regime 진단**: t_step 채널이 작다(1.06×)는 것은 step-latency가 ridge/plateau-bound임을 뜻함(`B_max
  89/230 > ridge B*≈81`). 그러나 **RPS 전환은 t_step이 아니라 capacity(B_max) 채널에서 발생**하므로
  plateau여도 전환은 성립 — 이 분해가 §3.4의 "plateau면 이득 없음"을 정밀화함(plateau는 t_step 채널만
  무력화; capacity 채널은 별개).
- **confound 통제**(§7): 동일 GPU·`M_avail`·동일 kernel 구조(element 처리만 차이). 정규화는 BF16 1회
  측정 공유. linears는 전 format bf16 cuBLAS(weight-quant fused 경로엔 보수적) → weight quant는 t_step에
  **B_max(capacity)로만** 기여, 과대평가 없음.

## 한 줄 요약

동일 커널·iso-memory에서 **메모리 절감 → 운용 max batch 1.46~1.53×↑(MXINT8 대비) → RPS 1.56~1.63×↑**,
그리고 그 RPS 이득이 **capacity(B_max) 채널로 ~1:1 추적**됨을 두 scope((3,16)·(2,8)) 모두에서 정량 입증.
(BF16 대비 max batch 2.97~7.67×, RPS 2.62~3.11×.)
