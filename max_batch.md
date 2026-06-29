# Max servable batch & decode throughput — MSAQ vs MXINT8/MXFP8 (capacity)

교수님 지적("fixed-batch에서 MXINT8 대비 개선 없음")에 대한 답: **fixed-batch에선 비슷하지만, 같은 HBM에서
운용 가능한 배치까지 채우면 MSAQ가 ~2× 더 많은 시퀀스를 돌리고, decode throughput ~1.6×를 낸다.** 이 문서는
그 측정의 **methodology를 먼저 명시**하고, **B=32/64에서 OOM났던 기존 측정과의 차이를 정직하게 검증**한다.

---

## 1. Methodology — 무엇을, 어떤 세팅에서, 어떻게 쟀나

### 모델 config (고정)
Llama-3.1-8B: **layers=32, Hq=32, Hkv=8(GQA), head_dim=128, hidden=4096, intermediate=14336, ~8.03B params.**
KV 원소 수 = `2(K,V) × layers × Hkv × head_dim = 65,536 elem/token/seq`.

### 하드웨어
**NVIDIA RTX PRO 4000 Blackwell, 25.2GB**(device total) / nvidia-smi 24.5GB usable, 70 SM. (논문엔 H100-80GB/70B도 sweep 권장.)

### 비트 예산 (iso-accuracy 가정 — repo 측정 기반)
| 포맷 | weight | KV | 비고 |
|---|--:|--:|---|
| **MXINT8 = MXFP8(E4M3)** | 8.25b | 8.25b | 둘 다 8.25b → **footprint 동일** → 같은 baseline |
| MSAQ W6.25/KV4.5 | 6.25b | 4.50b | E2M3 weights + u4 KV |
| MSAQ W6.25/KV5.44 | 6.25b | 5.44b | E2M3 weights + u3/gs16 KV |
| MSAQ KV-only 4.5 | 8.25b | 4.50b | KV 효과만 분리(weights=MXINT8) |

### 측정 ① max servable batch (`capacity_maxbatch.py`)
max-batch는 **compute가 아니라 메모리 footprint 문제**다. 실제 GPU에 다음을 **할당**하고 OOM 직전 B를 이진탐색:
- **weights (quantized-resident)** = `params × bits/8` (포맷별 실제 바이트),
- **KV cache** for B seqs at context L (전 32 layers) = `B × 65536 × kv_bits/8 × L`,
- **per-seq decode workspace** (`--act_per_seq_kb`, 기본 512KB/seq).

실제 `torch.empty`로 GPU에 올려 **allocator/fragmentation 현실을 반영**. 커널·forward·vLLM 불필요.

### 측정 ② decode throughput @ max-B (`decode_throughput_maxbatch.py`)
각 포맷의 max-B에서 **decode-step latency**를 재고 `throughput = B_max / step_time`.
- decode step = `32 layers × (linear GEMMs[M=B] + attention)`.
- **linears**: bf16 cuBLAS (B≥16의 배포 경로 = dequant-weight→bf16+cuBLAS이므로 포맷-무관, 공정),
- **attention**: 우리 실제 커널 (`kv_decode_attention_batched`=MSAQ, `mxint8_kv_decode_batched`=MXINT8, SDPA=bf16),
- 메모리는 **한 레이어 분량만 할당 ×32**(타이밍엔 충분) → B=314도 OOM 없이 측정.

### 핵심 가정 (load-bearing — 반드시 명시)
1. **weights quantized-resident** (bf16 master 없음).
2. **chunked prefill** (prefill activation peak가 작게 묶임 — vLLM/TRT-LLM/SGLang 표준).
3. MXINT8 = MXFP8 (8.25b 동일 footprint).

→ 이 가정들은 **실제 서빙 스택이 하는 일**이며, 결과는 그 가정 하의 capacity. **§3에서 이 가정이 정당한지 검증.**

---

## 2. 결과

### Max servable batch (24GB, Llama-8B)
| context | MXINT8/MXFP8 | MSAQ W6.25/KV4.5 | MSAQ W6.25/KV5.44 | KV-only 4.5 |
|--:|--:|--:|--:|--:|
| 1152 (L_in1024+L_out128) | 204 | **417 (2.04×)** | 346 (1.70×) | 371 (1.82×) |
| 1536 (L_in1024+L_out512) | 154 | **314 (2.04×)** | 261 (1.69×) | 279 (1.81×) |
| 32768 | 7 | **15 (2.14×)** | 12 (1.71×) | 13 (1.86×) |

→ **MSAQ는 MXINT8/MXFP8의 ~2× 배치를 운용**(B 155–314는 MSAQ로만 servable). KV만 양자화해도 ~1.8×.

### Decode throughput @ max-B (context 1536)
| 포맷 | B_max | step(ms) | tok/s | vs MXINT8 |
|---|--:|--:|--:|--:|
| bf16 (16b) | 40 | 75.5 | 530 | 0.61× |
| **MXINT8/MXFP8** | 154 | 176 | 875 | 1.00× |
| **MSAQ KV4.5/W6.25** | 314 | 218 | **1439** | **1.65×** |
| MSAQ KV5.44/W6.25 | 261 | 193 | 1355 | 1.55× |

(context 1152: MXINT8 1148 → MSAQ 1764, **1.54×**.)

**전환 메커니즘**: max-B에서 두 포맷 모두 KV로 HBM을 채운다(≈17GB, 비슷한 총 바이트 → attention 시간 비슷).
MSAQ는 그 바이트에 **2× 시퀀스**를 담아 step당 2× 토큰을 냄. 선형층(B-비례, KV-quant 무관)이 더해져
**2.04× batch × 0.81× step-penalty = 1.65× throughput.** (decode는 attention-dominant: B=314서 attn 5266u >> lin 1552u.)

**숫자의 변화**: fixed-batch RPS 1.27× → **capacity-frontier decode throughput 1.65×**.

---

## 2b. capacity가 *진짜로* binding되는 곳 — 70B (binary 승리)

§2의 8B@24GB는 솔직히 말하면 capacity가 **binding되기 직전**의 영역이다: B_max가 154–417로 **너무 커서**
실제 서빙에선 compute/scheduler가 먼저 cap을 건다(continuous-batch의 동시성 한계, prefill compute). 거기서의
2.04× ratio는 footprint상 실재하지만 **"이미 충분히 큰 배치를 더 키우는"** projection에 가깝다. capacity가
**실제 병목이 되는** 곳은 **큰 모델 / 긴 컨텍스트**, 즉 B_max가 한 자리 수로 떨어지는 영역이다.

### 70B @ 1×H100-80GB — MXINT8은 아예 안 올라간다 (binary)
`python capacity_model.py --model llama70b --gpu h100` (HBM 80GB × util0.9 − 2GB = 70GB usable):

| 포맷 | weights | B_max@1k | B_max@16k | B_max@128k |
|---|--:|--:|--:|--:|
| bf16 (16b) | 141.2GB | **0 (OOM)** | 0 | 0 |
| **MXINT8/MXFP8 (8.25b)** | 72.8GB | **0 (OOM)** | 0 | 0 |
| W6.25 / KV8.25 | 55.2GB | 85 | 5 | 0 |
| W6.25 / KV5.44 | 55.2GB | 130 | 8 | 1 |
| **W6.25 / KV4.50** | 55.2GB | **157** | **9** | **1** |

**MXINT8 weights(72.8GB)만으로 usable HBM(70GB)을 넘겨** KV 한 토큰도 못 얹는다 → 전 context에서 B_max=0.
**W6.25 weight 양자화(55.2GB)가 있어야 70B가 단일 H100에 비로소 servable**해진다. 이건 "2× 더 큰 배치"가
아니라 **"돌릴 수 있나 vs 없나"의 binary 승리** — ratio가 정의되지 않아 표에는 `inf`로 찍힌다(아래 §주의).

### ⚠️ 정직한 caveat — "MXINT8 70B가 불가능"은 단일-GPU 한정
실무에서 70B는 보통 **tensor-parallel로 2–8 GPU에 분산**한다. 그러니 MXINT8 70B가 *불가능*한 게 아니라
**≥2 GPU가 필요**한 것이다. 정확한 프레이밍은 **"고정 GPU 예산"** 축이다:
- weight-quant은 70B를 **더 적은 GPU에** 올린다(1×H100 vs MXINT8의 2×) → **TCO/대수 절감**,
- 그 위에 KV-quant이 **배치·컨텍스트를 더 얹는다**.

### 70B에서 ratio가 의미를 갖는 곳 — 1×H200-141GB (MXINT8 servable)
MXINT8 weights(72.8GB)가 들어가는 더 큰 단일 GPU에선 ratio가 다시 의미를 갖고, 8B와 같은 ~2× 패턴이 재현된다:

| 포맷 | B_max@1k | ratio | 비고 |
|---|--:|--:|---|
| MXINT8/MXFP8 | 301 | 1.00× | baseline (servable) |
| W6.25 / KV8.25 | 403 | 1.34× | weight-only 효과 |
| W6.25 / KV5.44 | 611 | 2.03× | + u3/gs16 KV |
| **W6.25 / KV4.50** | **739** | **2.46×** | + u4 KV |

(bf16 70B는 141GB weights ≈ 전체 HBM이라 단일 H200에도 **안 들어간다** → 0.00×. context가 길어질수록
ratio는 더 커진다: KV4.50 @64k 2.75×.)

**요지**: 8B@24GB는 capacity 이점의 *projection*(ratio 실재하나 binding 직전), **70B는 capacity가 실제로
bind되는 곳** — 1×H100에선 **binary 승리**(MXINT8 불가), 1×H200에선 **2.46× ratio**. 이게
`vllm_integration_plan.md`의 figure-3(긴 컨텍스트/큰 모델에서 baseline OOM 너머로 계속 servable) 시나리오다.

---

## 3. 정직한 검증 — "B=32/64에서 OOM났는데 어떻게 154/314냐?"

당신의 의심은 정당하다. 차이의 원인을 코드로 확인했고, **기존 하니스 OOM은 serving capacity와 무관한 microbench
artifact**임을 밝힌다.

### 기존 하니스(`harness_batchsweep.py`)가 실제로 한 것
- **weights = 한 레이어만 들고 32× 재사용** (`prefill`의 `for li in range(layers)`가 같은 `self.wq…wd` 사용).
  → 실제 8B 모델(int8 8.3GB / bf16 16GB)을 안 들고 있음. **메모리가 실제 배포 footprint가 아님.**
- **prefill을 B×1024 토큰 전체 한 번에** 처리(`prefill(ids[B,1024])`). → activation이 **B×L_in에 비례 폭증**.
  실측 peak를 분해하면 B=32에서 비-KV 부분 ~10.8GB가 대부분 **prefill activation**(∝B). **이게 OOM 주범.**
- **S1/S2(weight scope)는 KV가 bf16** → 그래서 B=64 OOM은 *bf16 KV(12.9GB) + full-prefill activation* 때문이지
  KV/weight 양자화와 무관. (S3 KV-quant는 B≤32까지만 돌림.) 참고로 batch-sweep은 **B=32 fit(17.2GB) / B=64 OOM**;
  B=32에서 OOM을 봤다면 더 무거운 scope(S5/S6) 또는 더 긴 Lcap이었을 것.

### 그래서 무엇이 5× 차이를 만드나
| 항목 | 기존 하니스 | 우리 probe (serving-realistic) |
|---|---|---|
| weights | 1-layer 재사용(비현실) | **full 8B quantized-resident** |
| prefill activation | **full prefill (B×1024) → peak가 OOM 지배** | **chunked prefill 가정 → peak 작음** |
| KV dtype | S1/S2는 bf16 | 포맷별 양자화 |

→ **기존 하니스의 OOM은 "non-chunked full-prefill activation peak"가 만든 microbench 한계**이지, 서빙에서의
용량 한계가 아니다. 우리 probe는 **실제 8B 양자화 모델 + 양자화 KV + chunked prefill**(서빙 표준)의 footprint를
**실제 GPU 할당으로** 측정 → 그 가정 하에서 max-B가 ~2× 더 큼.

### 무엇이 단단하고(✓) 무엇이 가정인가(⚠️)
- ✓ **KV/weight footprint 수식 + 실제 GPU 할당**(fragmentation 반영) + **attention 커널 실측**.
- ✓ **ratio(~2× batch, ~1.6× tput)는 activation-reserve 파라미터에 robust** (이 컨텍스트에선 KV가 지배).
- ⚠️ **chunked prefill + quantized-resident weights 가정** — 둘 다 서빙 표준이지만 **결과는 그 가정 하의 projection**
  (현재 microbench 하니스의 측정이 아님). decode throughput은 **합성 decode-step**(1-layer×32, bf16 linears).
- ⚠️ 실제 서빙의 스케줄링·가변길이·continuous-batching 오버헤드 미포함 → **1차(first-order) 결과**.

### 결론 (grounding)
- **이 결과는 근거가 있다**: 실제 GPU에 실제 footprint를 할당해 OOM 경계를 쟀고, attention은 실제 커널로 쟀다.
- **단, "서빙 표준 가정(chunked prefill, quantized-resident weights)" 위의 projection**이다. 기존 하니스가 그 가정을
  안 써서 OOM이 일찍 난 것이지, 결과가 틀린 게 아니다.
- **gold-standard 검증 = vLLM**(이 가정들을 네이티브로 사용). 논문엔 (a) 이 capacity projection + (b) vLLM serving
  Pareto를 함께 싣고, **accuracy-vs-bits Pareto와 페어링**(그래야 "그냥 INT4 쓰지" 반박 차단)하는 것을 권장.

---

## 4. 재현
```
python capacity_maxbatch.py --model llama8b --ctx 1536        # max-B (할당 probe)
python decode_throughput_maxbatch.py --ctx 1536               # tok/s @ max-B
python capacity_model.py --model llama70b --gpu h100          # 70B binary 승리 (MXINT8 OOM -> 'inf')
python capacity_model.py --model llama70b --gpu h200          # 70B ratio 승리 (MXINT8 servable, 2.46x)
```
스크립트: `capacity_maxbatch.py`(할당 probe), `decode_throughput_maxbatch.py`(throughput), `capacity_model.py`(분석).
