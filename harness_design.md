# End-to-End Harness 설계 (Llama-3.1-8B, full forward)

목적: 7종 커널을 실제 Llama-3.1-8B 디코더 스택에 끼워, **BF16 / MXINT8 / MSAQ** 세 경로의
**TTFT · TPOT · 총 생성시간**을 prefill=800 / decode=3880 시나리오로 측정한다. RTX 3090.

> 이건 **타이밍 하니스**다(정답 logits이 목표가 아님). 따라서 가중치는 랜덤이어도 되고,
> glue(norm·RoPE·softmax)는 세 경로 공통이라 비교에 중립이다. 측정값만이 산출물이다.

---

## 1. 모델 config (Llama-3.1-8B)

| 항목 | 값 |
|------|-----|
| hidden_size | 4096 |
| num_attention_heads (Q) | 32 |
| num_key_value_heads (KV) | **8 (GQA, group=4)** |
| head_dim | 128 |
| intermediate_size | 14336 |
| num_hidden_layers | 32 |
| vocab_size (lm_head) | 128256 |
| norm | RMSNorm (eps 1e-5) |
| pos enc | RoPE (theta 500000) |

레이어당 선형층(7개): q 4096→4096, k 4096→**1024**, v 4096→**1024**, o 4096→4096,
gate 4096→14336, up 4096→14336, down 14336→4096. (GQA로 K/V projection이 1024로 작음 →
KV 캐시도 8 head라 4× 작음.)

---

## 2. 세 정밀도 경로

| 경로 | 선형층(W) | 활성화(A) | KV 캐시 | attention 계산 |
|------|-----------|-----------|---------|----------------|
| **BF16** | bf16 (torch `@`) | bf16 | bf16 | bf16 SDPA |
| **MXINT8** | MXINT8 kernel | (W-only: bf16 / W+A: MXINT8) | MXINT8 plane | prefill: bf16 SDPA · decode: MXINT8 dequant-attn |
| **MSAQ** | MSAQ kernel | (W-only: bf16 / W+A: MSAQ-s) | MSAQ-s plane | prefill: bf16 SDPA · decode: MSAQ dequant-attn |

- **W-only / W+A 분리 보고 (확정):** 두 변형을 **별도 경로로** 측정·보고한다. 따라서
  리포트 경로 = `{BF16}` + `{MXINT8, MSAQ}` × `{W-only, W+A}`, MSAQ는 `{u2,u3,u4}`.
  W-only = 활성화 bf16(가중치만 양자화), W+A = 활성화도 양자화.
- glue(RMSNorm·RoPE·softmax·residual·embedding·lm_head)는 **항상 bf16/fp32**, 모든 경로 공통.

---

## 3. Phase별 커널 매핑

### Prefill (입력 800 토큰, TTFT)
레이어마다:
1. RMSNorm (bf16 glue)
2. Q/K/V projection → **W-only/W+A GEMM** (BF16은 torch `@`)
3. RoPE 적용 (bf16 glue)
4. **두 갈래 동시:**
   - prompt 출력용 **causal attention = bf16 SDPA** (GQA: KV 8→32 broadcast). *prefill
     attention은 7종에 없음 → bf16.*
   - decode 대비 **KV write**로 K/V를 packed plane 캐시에 기록 (MXINT8/MSAQ만; BF16은
     bf16 캐시에 그냥 저장).
5. O projection → GEMM, residual
6. MLP (gate·up·down) → GEMM, SwiGLU(glue), residual

→ 마지막 토큰 hidden에 lm_head(128256) 적용까지 = **TTFT**.

### Decode (3880 토큰 autoregressive, TPOT)
토큰마다, 레이어마다:
1. RMSNorm (glue)
2. Q/K/V projection → **W-only/W+A GEMV** (BF16은 torch `@`)
3. RoPE (glue)
4. 새 토큰 K/V를 **KV quantize(append)**로 캐시 slot=pos에 기록 (→ RoPE epilogue에 fuse 권장)
5. **KV cache dequantize attention**: 단일 query가 길어진 캐시(pos+1개 key) 전체에 attend
   (GQA: q head h → kv head h/4). BF16은 SDPA.
6. O projection → GEMV, residual
7. MLP → GEMV, SwiGLU, residual
8. (생성 토큰 샘플링은 lm_head 한 토큰만 — 매 스텝 vs 마지막만? → 매 스텝 lm_head 포함이
   현실적, TPOT에 포함)

→ 토큰당 평균 = **TPOT**; KV가 자라서 후반 토큰이 느려지므로 **TPOT(t) 성장곡선**도 기록.
**총시간 = TTFT + Σ_t TPOT(t).**

---

## 4. 필요한 커널 확장 (최소)

1. **GQA decode attention:** `kv_decode_attention`에 `num_kv_heads` 인자 추가, q head `h`가
   kv plane을 `h / (Hq/Hkv)`로 인덱싱. block-per-(q)head 유지, KV 인덱스만 group으로 매핑.
   (MXINT8 짝도 동일 확장 — 동등 비교 유지.) ~수십 줄.
2. **(선택) prefill 경로 glue:** RMSNorm·RoPE·SwiGLU·SDPA(GQA)는 torch로 구현 — 새 CUDA
   불필요. KV write/append는 이미 존재.

그 외 6종 커널은 그대로 사용. KV write/append/dequant가 **동일 packed 포맷**을 공유하므로
prefill→decode 캐시 재포맷 비용 없음.

---

## 5. 메모리 전략 (24GB)

- **가중치 재사용:** 타이밍은 가중치 값과 무관 → 레이어 1개분 packed 가중치를 GPU에 한 번
  올리고 **32 레이어가 재사용**. (실가중치 32레이어 = BF16 14GB·MX 7GB·MSAQ 4GB로 3090에
  올라가긴 하나 불필요.) 헤드라인은 재사용, 필요시 실가중치 모드 옵션.
- **KV 캐시:** Lcap = 800+3880 = 4680, 8 head, head_dim 128, 32 layer.
  - BF16: 8·4680·128·32·2B·2(K,V) ≈ 6.1GB
  - MSAQ u4: ≈ 1.8GB / MXINT8: ≈ 2.4GB
  전부 24GB 내. 캐시는 미리 Lcap으로 할당, append가 in-place로 slot 채움.

---

## 6. 측정 프로토콜

- warmup: prefill 1회 + decode 32스텝 버린 뒤 측정(커널 캐시·클럭 워밍).
- TTFT: prefill forward 전체 + 마지막 토큰 lm_head, `cuda.Event`로 1회(또는 수회 평균).
- decode: 전체 3880 스텝 실제 실행, 누적시간 = decode total. TPOT_mean = total/3880.
  추가로 t ∈ {1,256,1024,2048,3880}에서 순간 TPOT 기록 → 성장곡선.
- 출력 표: 경로(BF16/MXINT8/MSAQ-u2/u3/u4) × {TTFT, TPOT_mean, total, MSAQ/MX, MSAQ/BF16}.
- 산출: stdout 표 + (선택) change.md/kernel_ver1.md에 결과 추가.

---

## 7. 예상 결과 가설 (커널 단위 측정 기반)

- **TTFT(prefill, compute-bound):** W+A GEMM이 INT8 IMMA로 u4 0.79 → TTFT에서 MSAQ가
  MXINT8 대비 ~0.8. 단 prefill attention이 bf16 SDPA라 그 비중만큼 희석.
- **TPOT(decode, memory-bound):** GEMV(u4 0.63) + KV dequant(u4 0.54)가 지배 → MSAQ가
  TPOT에서 가장 크게 이김(예상 ~0.6). 컨텍스트가 길어질수록(후반 토큰) KV read 비중↑ →
  MSAQ 이득↑ (성장곡선에서 MSAQ/MX 비율이 더 내려감).
- **총시간:** decode 3880 >> prefill 800이라 **TPOT 이득이 총시간을 지배** → MSAQ end-to-end
  win 예상.

---

## 8. 확정된 결정 (locked)

1. **W-only / W+A 분리 보고** — 둘을 별도 경로로 측정(§2).
2. **GQA decode attention 커널 확장** 추가 — `num_kv_heads` 인자, q head `h`→kv head
   `h/(Hq/Hkv)` 매핑. KV replicate(메모리·대역폭 왜곡)는 채택 안 함.
3. **매 decode 스텝 lm_head(128256) 포함** — TPOT에 vocab projection 1회 포함(현실적).
4. **가중치 재사용** — 레이어 1개분 packed 가중치를 32레이어가 재사용(타이밍 무관). 실가중치
   모드는 옵션으로만.
