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
