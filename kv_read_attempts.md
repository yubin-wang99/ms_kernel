# KV-read MSAQ win 시도 총정리 (Phase 32–40, 8 lever) + Phase 41 (해결)

decode attention의 KV cache dequant(KV-read)에서 **MSAQ가 MXINT8을 공정·정확하게 이길 수 있는가**를
끝까지 추적한 기록. 단계별 로그는 [change.md] Phase 32–41, 공정성 감사는 [for_fair_comparison.md],
7개 커널 전체는 [kernel_ver2.md], 패킹-친화/v8/sepsc/vt 스택은 [tests/kv_pack_results.md].

> **갱신 (Phase 41):** Phase 32–40은 "tie/불가능"으로 닫혔으나, **single-token MHA decode**에서
> `u4/gs2` + **int8-staged V(v8)** + **separated-scale K dot(sepsc)** + **transposed-padded V staging(vt)**
> 스택으로 **wide 스칼라 커널이 MXINT8을 0.97× best-vs-best로 이김**(아래 §5). 즉 Phase 32의 "wide
> single-token = tie"는 **MHA에서 win으로 정정**된다. GQA(Hq>Hkv)는 wide 경로가 KV를 query-head당
> 1번(=4×) 중복 read해 parity(1.01–1.07×)에 머물고, 배치/텐서코어 regime(Phase 35–40)은 미재검토.

---

## 0. 문제 구조

decode attention = **Q·K^T**(키별 score, *D에 대한 reduction*) → softmax → **P·V**(*키들에 대한 reduction*).
KV는 token-major `[H, nb, L, *]`로 packed. 핵심 비대칭:

- **Q·K (K read):** reduction이 한 키의 head_dim 내부(연속 packed 바이트) → thread-per-key로 **full-sector**
  coalesced. int8/sub-byte 둘 다 깨끗 → tie.
- **P·V (V read):** reduction이 **키들을 가로지름** → 출력이 thread-per-d. **int8 V는 자연히 full-sector**
  (워프 32 d-lane = 32 연속 바이트 = 1 섹터), **sub-byte V는 half-sector**(32 nibble=16B) → 여기가 MSAQ의
  구조적 약점.

MSAQ u4 = MXINT8의 **0.58× 바이트**. 이게 시간으로 환원되려면 KV-read가 **BW-bound**여야 하는데,
스칼라 decode는 latency-bound라 안 된다(아래).

---

## 1. ver2 기준선 (Phase 32)

`kv_decode_wide_kernel`(스칼라 융합): Pass-1 thread-per-key Q·K + online-softmax + Pass-2 **V를 shared에
staging**(sub-byte half-sector 회피) 후 thread-per-d P·V. MXINT8도 thread-per-key로 공정화.

**결과: u4 KV-read = tie**(이전 0.54 "압승"은 MXINT8 under-optimization 산물). 커널이 저점유율
latency-bound(MXINT8 실측 300~480 GB/s까지 스케일, MSAQ ~220에서 saturate)라 0.58× 바이트가 시간에 안 나타남.

---

## 2. 8개 lever와 결과

| # | lever | 결과 | 왜 실패/한계인가 |
|---|-------|------|------------------|
| 1 | split-K mult↑ (occupancy) | **악화** | per-block 일 적은 MXINT8만 이득, combine 오버헤드가 MSAQ에 불리 |
| 2 | warp-transpose P·V (staging 제거) | **악화 ~5–10%** | broadcast shuffle issue↑, occupancy 불변(grid 0.5–0.76 wave) |
| 3 | batch (점유율 공짜 확보) | **악화 1.07–1.22** | 머신 채워 BW-bound 도달하나 MSAQ 실효 BW가 MXINT8의 **~0.5×**(dequant throttle) |
| 4 | channel-major V (KIVI식 layout) | 속도 win이나 **기각** | dense block이 token축 grouping 강제 → 정확도 ×1.3–1.8 악화(KIVI는 V per-token 권장) |
| 5 | 텐서코어 P·V (split-K WMMA, scattered load) | **악화** | MMA가 가속하는 reduction은 병목 아님; bf16-staging이 병목, 두 포맷 동일 타일 |
| 6 | **+coalesced load +파이프라인** | **P·V 단독 WIN 0.84–0.94 (M≥32)** | scatter→full-sector로 MSAQ 실효 BW 0.5→0.6×, 0.58× 바이트와 합쳐 win |
| 7 | 완전 2-pass attention (WMMA Q·K) | **tie~loss (best ~1.0)** | Q·K(D-contract 짧음)가 unpack-bound loss → P·V win을 희석 |
| 8 | scalar Q·K (bf16-staging 제거) | ratio win 0.78–0.92 **이나 best-vs-best tie** | 텐서코어 미사용 → 2–3× 느린 영역; MSAQ 최선은 여전히 WMMA(tie) |

### 핵심 수치
- **sector 진단(Phase 34):** DRAM read-sector 비 = **0.59 ≈ 바이트비 0.58** → DRAM 레벨 inflation 없음.
  바이트 이점은 실재하나 커널이 DRAM 14%(latency-bound)라 시간에 안 나타남.
- **batch(Phase 35):** B=32에서 MXINT8 실효 ~530 GB/s(BW-bound 도달) vs MSAQ ~270(0.5×) → 0.58/0.5≈1.1× loss.
- **텐서코어 P·V win(Phase 38):** coalesced thread-per-key load(키의 16B 레코드 통째 = full-sector) +
  double-buffer 파이프라인 → P·V만 M≥32서 0.84–0.94 win. token-major 유지(정확 rel 2.4e-3).
- **full attention(Phase 39):** Llama 1.17→1.00, Gemma 1.25→0.99, Mistral 1.18→1.00 (M=32→128). P·V win을
  Q·K가 상쇄 → ~tie.
- **scalar Q·K(Phase 40):** ratio win이나 Llama Lk4096 M128 scalar MSAQ 541µs / MXINT8 619µs vs
  **WMMA MSAQ 243 / MXINT8 241(tie)** → best-vs-best는 WMMA(tie). 느린 영역 ratio-win artifact.

---

## 3. 근본 원인 (확정)

KV-read의 binding constraint는 점유율·reduction·layout이 아니라 **"MSAQ를 텐서코어/누적기가 소비 가능한
형태(bf16/누적)로 dequant하는 throughput"**(MXINT8 대비 **~0.5× 실효 BW**)이다.

- **fast = 텐서코어 = bf16 타일 강제** → 두 포맷이 같은 bf16 타일을 만들어 0.58× DRAM이 무효 → **tie**.
- **ratio-win = staging 없는 스칼라/wide** → DRAM-bound지만 텐서코어 미사용이라 **2–3× 느린 영역** → 아무도
  안 고름 → best-vs-best에 안 들어옴.
- **둘을 동시에 못 가진다.** P·V의 키-가로 reduction + sub-byte는 (a) half-sector(직접) (b) 정확도 trade
  (channel-major) (c) bf16 staging 천장(텐서코어) 중 하나를 강제하고 셋 다 바이트 이점을 상쇄.

대조: **W-only GEMV는 win(0.63)** — staging 없이 wide-load→직접 누적이라 DRAM-bound가 되고 element 내부
reduction이라 sub-byte도 full-sector. KV-read의 P·V는 이 조건을 못 갖춘다.

---

## 5. Phase 41 — single-token MHA decode WIN (wide 스칼라, 0.97×)

Phase 32는 wide 스칼라 single-token decode를 "tie(latency-bound)"로 닫았다. 그 tie를 ncu로 다시
열어 **병목을 정확히 측정**하고(DRAM 20.7%, **L1/TEX shared 63.5%**, occupancy 24% reg-limited →
BW-bound도 occupancy-bound도 아니고 **shared-traffic-bound**), 4개 lever를 쌓아 정정했다:

| lever | 효과 (u4/gs2, vs MXINT8 wide) | 메커니즘 |
|---|---|---|
| 패킹-친화 config (nibble u4/gs2) | 1.45→ — | u4=clean nibble(2/byte). robust(+2.72% PPL)인 최저-aggressive nibble |
| **v8** (int8-staged V Pass-2) | →1.25× | V를 Pass-1서 int8 코드로 복원 staging → Pass-2 = MXINT8과 동일(1 int8 read, no bfe) |
| **sepsc** (separated-scale K dot) | →1.02–1.13× | `Σq·(up·2^u+sh)·s = s·(2^u·Σq·up + Σ_g sh·qg)`, qg=query group-sum 1회 precompute |
| **vt** (transposed+padded V staging) | →**0.97× WIN** | `pV8[blk·32·CH+kd·CH+kk]`, CH=chunk+4(CH/4 홀수) → Pass-2가 int32당 4코드 read(conflict-free, 4× 적은 shared txn) |

**결과(bit-exact, test_kv 72/72):** MHA(Hq=Hkv) H8 Lk4096/16384, H16 Lk16384 **전부 0.97× WIN**.
ncu: L1/TEX 63.5→60.2%, Duration 148→135µs. GQA(32/8)는 1.01–1.07×(wide가 KV head를 query-head당
4× 중복 read; 해소엔 design-A KV-reuse 커널 필요 — occupancy-bound라 미달, [tests/kv_pack_results.md] Opt 3).
gated: `MS_KV_V8`/`MS_KV_SEPSC`/`MS_KV_VT`(u4/gs≤2 디폴트 on). 같은 sepsc는 W-only GEMV u3에도 +3–5%
([tests/gemv_sepsc_results.md]).

**왜 Phase 32–40의 "불가능"을 깨는가:** 그 분석은 (a) fast=텐서코어=bf16-staging 천장, (b) ratio-win=
스칼라는 2–3× 느린 영역, 둘을 동시에 못 가진다고 봤다. 그러나 **single-token decode의 best-vs-best는
텐서코어가 아니라 wide 스칼라**(M=1이라 MMA 타일이 무의미)이고, 그 wide 스칼라의 진짜 병목은 dequant-
throughput이 아니라 **Pass-2 staged-V의 shared-transaction(L1/TEX)**이었다. vt가 그걸 4× 줄여 0.58× 바이트가
아니라 **shared-txn 절감**으로 win한다(DRAM은 여전히 20%, BW-bound 아님).

## 4. 최종 결론 (정정)

- **single-token MHA decode: MSAQ KV-read가 MXINT8을 공정·정확하게 이긴다(0.97×).** v8+sepsc+vt 스택.
  Phase 32의 "wide single-token tie"를 정정. (Phase 41)
- **GQA(Hq>Hkv) / 배치 / 텐서코어 regime: 여전히 미해결.** GQA wide는 KV 4× 중복 read로 parity(1.01–1.07×);
  KV-reuse(design-A)는 occupancy-bound라 wide에 못 미침. 배치/텐서코어(Phase 35–40)는 bf16-staging 천장
  가설 그대로(미재검토). native sub-byte tensor-core(하드웨어 미지원)가 그 regime의 잔여 가능성.
- **부분 win(기존):** shared-prefix(M=N·G≥32) P·V-only 0.84–0.94.
- **MSAQ의 가장 큰 가치는 여전히 weight 경로**(W-only GEMV 0.63, S4 end-to-end 0.44× bf16 —
  [weight_scope_results.md]). **KV-read도 이제 single-token MHA에서 win.**

## 관련 산출물 (보존)
- 커널/ops: `csrc/kv_attention.cu` — `kv_decode_wide_kernel`(디폴트, Phase 41 win 스택 탑재:
  `MS_KV_V8`/`MS_KV_SEPSC`/`MS_KV_VT`, u4/gs≤2 디폴트 on), `kv_decode_gqa_kernel`(design-A, `MS_KV_GQA`),
  `kv_decode_warpT_kernel`(MS_KV_WARPT), `pv_wmma*`/`qk_wmma*`/`qk_scalar*`(텐서코어 2-pass, env
  `MS_QK_SCALAR`), 배치 변형(`*_batched`). 벤치: `tests/kv_pack_bench.py`(Phase 41 패킹/win 측정).
- 벤치: `tests/kv_lever_bench.py`, `kv_batch_bench.py`, `pv_wmma_bench.py`, `shared_prefix_attn_bench.py`,
  `v_grouping_accuracy.py`, `pv_gemv_proxy.py`, `kv_xact_driver.py`(ncu).
- 디폴트 decode 경로는 ver2(`kv_decode_wide_kernel`) 그대로 — 위 텐서코어 커널은 gated 실험/documented negative.
