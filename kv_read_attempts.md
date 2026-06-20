# KV-read MSAQ win 시도 총정리 (Phase 32–40, 8 lever)

decode attention의 KV cache dequant(KV-read)에서 **MSAQ가 MXINT8을 공정·정확하게 이길 수 있는가**를
끝까지 추적한 기록. 결론부터: **불가능**(본 dequant 패러다임에서). 단계별 로그는 [change.md] Phase 32–40,
공정성 감사는 [for_fair_comparison.md], 7개 커널 전체는 [kernel_ver2.md].

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

## 4. 최종 결론

- **공정(best-vs-best)·정확 MSAQ KV-read decode win은 본 dequant 패러다임에서 존재하지 않는다.**
  8개 lever(스칼라/staging/warp-transpose/batch/channel-major/텐서코어-P·V/텐서코어-2pass/scalar-Q·K)
  전부 시도 완료.
- 진짜 잔여 가능성: **native sub-byte tensor-core**(V를 dequant 없이 직접 먹는 MMA) — 하드웨어 미지원.
- **부분 win:** shared-prefix(prefix caching/beam, M=N·G≥32)에서 **P·V만** 떼면 0.84–0.94. 단 완전 attention
  은 Q·K가 희석해 tie. 일반 독립 배치 decode는 M=G(2~4)로 win 영역 미달.
- **MSAQ의 실제 가치는 weight 경로**(W-only GEMV 0.63, S4 end-to-end 0.44× bf16). **KV-read는 tie로 확정.**

## 관련 산출물 (보존)
- 커널/ops: `csrc/kv_attention.cu` — `kv_decode_wide_kernel`(디폴트), `kv_decode_warpT_kernel`(MS_KV_WARPT),
  `pv_wmma*`/`qk_wmma*`/`qk_scalar*`(텐서코어 2-pass, env `MS_QK_SCALAR`), 배치 변형(`*_batched`).
- 벤치: `tests/kv_lever_bench.py`, `kv_batch_bench.py`, `pv_wmma_bench.py`, `shared_prefix_attn_bench.py`,
  `v_grouping_accuracy.py`, `pv_gemv_proxy.py`, `kv_xact_driver.py`(ncu).
- 디폴트 decode 경로는 ver2(`kv_decode_wide_kernel`) 그대로 — 위 텐서코어 커널은 gated 실험/documented negative.
