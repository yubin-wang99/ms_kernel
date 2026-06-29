# vLLM 통합 계획 — KV/weight 양자화의 capacity 이점을 실측으로 증명

## 0. 왜 이게 필요한가 (이전 결론)
- 우리 microbench 하니스는 **고정 배치**이고 **peak가 prefill-activation에 지배**되며 **weight를 bf16 resident**로 들어서, capacity 이점(저비트 → 더 많은 동시 시퀀스 → RPS↑)이 **구조적으로 안 나온다**. 교수님이 본 "MXINT8 대비 개선 없음"은 이 하니스의 한계.
- capacity 이점은 **실제 서빙 스택(vLLM)** 에서만 드러난다: (a) weight가 quantized-resident(bf16 master 없음), (b) PagedAttention + chunked-prefill로 activation peak가 억제, (c) KV가 진짜 지배적 메모리.

## 1. capacity 이점이 vLLM에서 나오는 메커니즘 (자동)
vLLM은 시작 시 `determine_num_available_blocks`로 메모리를 프로파일하고:
```
num_gpu_blocks = (gpu_mem*util − weights − activation_peak) / (block_size × kv_bytes_per_token)
max_num_seqs(L) ≈ num_gpu_blocks × block_size / L
```
**KV bits↓ → kv_bytes_per_token↓ → num_gpu_blocks↑ → 동시 admit 가능한 요청↑ → 부하 하에서 throughput↑.**
weight bits↓ → weights↓ → KV에 쓸 메모리↑ → 같은 효과. **즉 우리는 "더 작은 KV/weight"만 제공하면 vLLM scheduler가 자동으로 capacity 이점을 throughput으로 전환**한다. 우리가 측정할 것은 그 결과.

## 2. vLLM 확장 지점 (어디를 건드리나)
| 대상 | vLLM 위치 | 우리가 할 것 |
|---|---|---|
| **Weight 양자화** | `vllm/model_executor/layers/quantization/` (`QuantizationConfig`+`LinearMethodBase`), `--quantization <name>` | 우리 `wonly_gemm`/`mxint8_gemm` 등을 LinearMethod로 등록 → weight가 quantized-resident |
| **KV cache dtype/shape** | `vllm/worker/cache_engine.py`(`_get_cache_block_size`), backend의 `get_kv_cache_shape` | 우리 packed 포맷의 bytes/elem 반영 → block이 작아져 block 수↑ |
| **Attention backend** | `vllm/attention/backends/` (flash_attn/flashinfer 류) | 우리 `kv_decode_attention`(decode) + prefill 경로를 backend로 구현 |
| **KV append/write** | backend의 `reshape_and_cache` 류 | 우리 `kv_append`(per-token 양자화 저장) 연결 |

## 3. 핵심 난관: 포맷 레이아웃 불일치
vLLM KV cache는 **레이어당 단일 텐서** `[num_blocks, block_size, n_kv_heads, head_dim]`(dtype=cache dtype, fp8까지 지원). 우리 MSAQ는 **3-plane**(upper / shared / scale_exp). → 두 옵션:
- **(권장) 단일 byte 텐서로 packed**: KV "원소"를 우리 sub-byte 스트림으로 정의, block_size를 바이트 정렬되게 잡고, backend가 unpack. `get_kv_cache_shape`가 packed byte shape를 반환. fp8(1 byte/elem)의 sub-byte 버전.
- (대안) cache engine을 확장해 레이어당 3 텐서 할당 — 더 침습적, 비권장.
- ⚠️ **포맷 선택이 통합 난이도를 좌우**: 우리 발견상 **MXFP6-E2M3(표준 MX 원소, 6-bit, 단일 원소)** 가 커스텀 3-plane MSAQ보다 훨씬 꽂기 쉽고 정확도도 best였다. **서빙 데모는 E2M3(또는 단일-원소 sub-byte)로 가는 걸 강력 권장** — 3-plane MSAQ는 마지막에.

## 4. 단계별 계획 (de-risk: 싼 것부터)

**Phase 0 — 방법론 검증 (커널 불필요, 며칠).**
vLLM이 **이미 `--kv-cache-dtype fp8`** 을 지원한다. fp16-KV vs fp8-KV로 `benchmark_serving.py`(ShareGPT)를 돌려:
- 시작 로그의 `# GPU blocks`(fp8가 ~2× 많음), `max_num_seqs`,
- throughput-vs-request-rate, **긴 컨텍스트에서 fp16이 OOM나는 지점**을 확인.
→ **"저비트 KV → block↑ → throughput↑"라는 우리 주장의 메커니즘을 vLLM에서 재현**(우리 기여 없이). 이게 baseline이자 sanity check. **여기서 이미 1차 그림이 나온다.**

**✅ Phase 0 측정 결과 (`vllm_phase0_capacity.py`, 실행됨).**
vLLM 0.23.0(V1) + Llama-3.1-8B를 RTX PRO 4000 Blackwell(sm_120, 24.5GB)에서 실측. kv_cache_dtype `auto`(bf16) vs `fp8`로 엔진을 init하고 vLLM 자체 프로파일러의 `num_gpu_blocks`를 직접 읽음:

| kv_dtype | num_gpu_blocks | KV pool(tok) | seqs@1k | seqs@16k | KV pool ratio |
|---|--:|--:|--:|--:|--:|
| auto (bf16 KV) | 2615 | 41,840 | 40 | 2 | 1.00× |
| **fp8 KV** | **5230** | **83,680** | **81** | **5** | **2.00×** |

- **메커니즘 확정**: vLLM이 fp8에 **정확히 2.00× 블록**을 할당 → 동시 seq·컨텍스트 2×. 우리 `capacity_maxbatch.py`의 해석적 결과가 **gold-standard 서빙 스택에서 네이티브로 재현**됨(커널 기여 0).
- **throughput 전환**: offline `generate()` 512 prompts @ctx1024/L_out128에서 **output tok/s 2884 → 3594 (1.25×)**. capacity 2×가 이 설정에선 1.25×로 부분 전환(짧은 ctx는 compute도 섞임) — capacity가 binding되는 **긴 컨텍스트일수록 2×에 근접**(capacity_model.py와 일치).
- **운영 노트**: torch 다운그레이드(2.12→2.11)가 MSAQ 커널을 깨므로 **격리 venv `.venv-vllm`** 에 설치. sm_120에서 vLLM 0.23 정상 동작 확인.
- 드라이버는 **kv_dtype당 엔진 1회 init**(2회)으로 풀을 읽고 컨텍스트별 seq를 도출 — block-pool은 길이 무관 단일 자원이므로 2×N init 불필요.

**✅ Phase 0 online — RPS-vs-P99 Pareto (`vllm_phase0_serving.py`, 실행됨).**
2× 풀이 **서빙 throughput으로 전환**되는지를 vLLM OpenAI 서버 + `vllm bench serve`(random 고정길이 in2048/out128, ctx 2176)로 요청률 sweep해 측정. 단일 24GB Blackwell, util0.9:

| offered req/s | auto RPS / P99 TTFT | fp8 RPS / P99 TTFT |
|--:|--|--|
| 1.0 | 0.98 / 1489ms | 0.98 / 1492ms (무부하 tie) |
| 1.5 | 1.45 / 4925ms | 1.45 / **2192ms** (2.2× ↓) |
| 2.0 | 1.66 / 15391ms | 1.90 / **4682ms** (3.3× ↓) |
| 2.5 | 1.70 / 26723ms | 2.11 / 10136ms |
| inf(포화) | **1.75** (224 tok/s) | **2.14** (274 tok/s) |

- **포화 throughput**: fp8 **1.22×** (1.75→2.14 RPS). **SLO goodput @ P99 TTFT≤5s**: fp8 **1.32×** (1.45→1.90 RPS). iso-load(1.5–2.0 req/s)에서 fp8가 **P99 TTFT 2.2–3.3× 낮음** — 큰 admission 풀의 latency 여유.
- **⚠️ 정직한 nuance**: (1) 풀은 2×인데 goodput 이득은 1.2–1.3×뿐 — **단일 24GB에선 compute가 풀을 다 쓰기 전에 포화**(decode/prefill 연산 한계). 메모리가 더 세게 binding되는 **긴 컨텍스트/큰 모델일수록 2×에 근접**(§2b·capacity_model caveat와 일치). (2) fp8는 **P99 TPOT가 더 높다**(150 vs 95ms): 더 많은 동시 seq를 decode step에 담아 per-token latency를 throughput과 맞바꾼 결과 — throughput-latency 트레이드오프의 정상 동작.
- **✅ 컨텍스트 스케일링 (핵심 검증, 실행됨)** — 같은 Pareto를 긴 컨텍스트(in8192/out256, ctx 8448)에서 반복. capacity가 더 세게 binding되면 gain이 **2× 풀 비율로 수렴**하는지 확인:

  | ctx | auto 포화 RPS | fp8 포화 RPS | gain |
  |--:|--:|--:|--:|
  | 2176 | 1.75 | 2.14 | 1.22× |
  | **8448** | **0.271** (69 tok/s) | **0.391** (100 tok/s) | **1.44×** |

  → **컨텍스트가 길수록 gain 1.22→1.44× 증가** (예측대로 2×로 수렴 추세). 긴 컨텍스트에선 auto 풀이 ~5 seq만 담아 rate 0.5에도 포화(P99 TTFT 94s) vs fp8 ~10 seq(31s, 3× 낮음). 2× 미달은 여전히 부분 compute-bound(decode GEMV); 더 길수록 decode가 memory-bound화되어 풀 비율에 수렴.
- **70B/멀티GPU**: 이 박스는 bf16 70B(141GB) > 총합 98GB(4×24.5)라 **실측 불가**, NVLink도 없어 PCIe-TP는 비대표적. → **`max_batch.md` §2b 분석으로 갈음**: 70B@1×H100 binary 승리(MXINT8 OOM), @1×H200 KV4.50 2.46×. (capacity가 binary로 binding되는 곳이라 분석이 오히려 더 강력.)
- **운영**: `vllm serve` 0.23은 요청 로깅 기본 OFF(`--enable-log-requests` opt-in), `--disable-log-requests` 플래그 제거됨. 서버는 process-group으로 띄워 SIGINT clean-kill.
- **MSAQ ≠ fp8 (과소평가 주의)**: Phase 0/이 Pareto의 fp8(8b, ≈무손실)은 **MSAQ의 보수적 하한**이다. MSAQ KV는 sub-byte(u4 4.5b / u3·gs16 5.44b)라 풀이 **fp8보다 1.5–1.8× 더 큼** → Pareto가 fp8보다 더 오른쪽. 대가는 정확도(+~1% PPL) → **iso-accuracy + accuracy-vs-bits Pareto로 페어링** 필수. 정확한 vLLM 수치는 Phase 2(MSAQ KV 백엔드)에서.
- **✅ ShareGPT 현실 분포 (실행됨)** — 고정길이 대신 ShareGPT_V3(94k 대화, 가변 길이)로 재현. `--dataset sharegpt`:

  | offered | auto RPS / P99 TTFT / TPOT | fp8 RPS / P99 TTFT / TPOT |
  |--:|--|--|
  | 16 | 5.82 / 542ms / **131ms** | 6.27 / 118ms / **32ms** |
  | 포화(inf) | **6.66** (825 tok/s) / TPOT **347ms** | **8.39** (1084) / TPOT **71ms** |

  → **포화 throughput 1.26×**(6.66→8.39), **SLA goodput @ P99 TTFT≤2s 1.44×**(5.82→8.39). **발견**: 부하가 커지면 auto는 KV 풀이 차서 vLLM **preemption/recompute**에 빠져 TPOT가 폭발(131→347ms); fp8는 2× 풀로 이를 피해 TPOT ~5× 낮게 유지 → **현실 trace에선 fp8가 throughput·latency 둘 다 우위**(고정길이 땐 fp8 TPOT가 더 높았던 것과 대조 — 거긴 preemption이 아닌 단순 큰 배치).

- **3개 워크로드 종합** (fp8 vs auto, 단일 24GB 8B):

  | 워크로드 | 포화 RPS auto→fp8 | 포화 gain | goodput@SLO |
  |---|--|--:|--:|
  | random ctx2176 | 1.75→2.14 | 1.22× | 1.32×@P99TTFT5s |
  | random ctx8448(긴) | 0.271→0.391 | 1.44× | (전부 포화) |
  | ShareGPT(현실) | 6.66→8.39 | 1.26× | 1.44×@P99TTFT2s |

  capacity가 셀수록(긴 ctx) gain↑; 현실 분포는 long-tail이 풀을 압박해 1.26–1.44×. 모두 **2× 풀의 부분 전환**(나머지는 compute).
- **후속**: Phase 1(weight-quant LinearMethod) / Phase 2(sub-byte MSAQ KV 백엔드 → fp8 초과 정량화) + accuracy-vs-bits Pareto 페어링.

**Phase 1 — Weight 양자화 통합 (중간 난이도).**
우리 weight GEMM을 vLLM `QuantizationConfig`로 등록 → weight가 quantized-resident(bf16 master 제거). 측정: 같은 GPU에서 **weights GB↓ → num_gpu_blocks↑ → max_num_seqs↑ → throughput↑**. (vLLM의 AWQ/GPTQ 통합 코드가 템플릿.)

**Phase 2 — Sub-byte KV attention backend (full win).**
KV를 fp8 미만(E2M3 6.25b / u4 4.5b)으로. §3의 packed 단일 텐서 + 우리 decode 커널 backend. 측정: fp8 → sub-byte로 **block 추가 증가 → throughput 추가 증가**. **이게 "fp8보다도 낫다"는 핵심 기여.**

## 5. 벤치마크 프로토콜 & 논문 figure
- **스택**: vLLM, 고정 모델(Llama-3.1-8B 또는 70B), 고정 GPU(H100-80GB 권장; 70B는 capacity가 binary로 binding).
- **워크로드**: `benchmark_serving.py` + ShareGPT trace(현실적 길이 분포), 그리고 **long-context 합성 trace**(prompt 8k–128k).
- **figure 1 (헤드라인)**: **throughput(req/s 또는 out-tok/s) vs P99 TPOT (Pareto)** — 우리 곡선이 오른쪽으로 더 뻗음(같은 SLA에서 더 높은 RPS).
- **figure 2 (capacity)**: **num_gpu_blocks / max_num_seqs vs 포맷**(bf16 / fp8 / 우리) — vLLM 로그 직접 인용.
- **figure 3 (long-context, 가장 강력)**: **max sustained throughput vs context length** — fp16/fp8가 OOM나는 컨텍스트 너머에서 우리는 계속 servable (binary 승리).
- **figure 4 (필수 페어링)**: **accuracy-vs-bits Pareto** — 우리 포맷이 같은 정확도를 더 적은 비트로(MXINT8/INT6/fp8 대비). 이게 없으면 "그냥 fp8/INT4 쓰지" 반박당함.

## 6. baseline (MXINT8보다 강해야 함)
- bf16 KV (상한), **vLLM 네이티브 fp8 KV(8-bit)** ← 진짜 경쟁자, INT8 등가. 우리는 **sub-byte로 fp8를 capacity에서 능가**해야 함.
- weight: fp16 / fp8 / AWQ-INT4 등 vLLM 기존 quant과 비교.

## 7. 리스크 / 노력 / 권고
- **노력**: Phase 0 며칠, Phase 1 1–2주, Phase 2 2–4주(backend가 핵심 리스크). vLLM은 빠르게 바뀌므로 버전 고정.
- **리스크 완화**: ① **포맷을 E2M3 등 단일-원소 sub-byte로** → §3 난관 최소화. ② Phase 0/1만으로도 "capacity→throughput" 1차 결과는 나옴(Phase 2 없이도 논문 절반). ③ decode 커널이 vLLM에서 정확/안정해야 generation이 끝까지 돎 → correctness gate 먼저.
- **권고 순서**: **Phase 0(즉시) → Phase 1 → (포맷=E2M3) Phase 2.** Phase 0 결과만으로도 교수님께 "capacity 축에서 효과가 실재한다"를 즉시 보일 수 있다.

## 8. 다음 액션 (이 repo에서 바로 할 수 있는 것)
- Phase 0용 스크립트: vLLM 설치 후 `--kv-cache-dtype {auto,fp8}` × context sweep으로 `# GPU blocks` + serving throughput을 뽑는 드라이버(우리 커널 불필요). → **첫 1차 결과 + 방법론 검증.**
- Phase 1용: 우리 `wonly_gemm` 기반 vLLM `LinearMethod` skeleton.
- Phase 2용: packed 단일-텐서 KV layout 정의 + `get_kv_cache_shape` + decode backend skeleton.
