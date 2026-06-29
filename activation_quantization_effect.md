# LLM 서빙에서 Activation Quantization의 이점

양자화 대상은 세 부류이고, "activation quantization"의 이점은 이 분류에 따라 달라진다:

- **Weight (W)** — linear 가중치
- **Activation (A)** — linear에 들어가는 X, 그리고 attention 내부의 Q/K/V/P
- **KV cache** — 시퀀스에 걸쳐 저장되는 K/V (사실상 "저장된 activation")

---

## 1. Prefill GEMM throughput — 맞음, 핵심 메커니즘 존재

✅ 핵심: **저정밀 텐서코어(INT8 IMMA / FP8 MMA)를 쓰려면 W와 A 둘 다 양자화돼야 한다.**
weight-only 양자화는 W를 bf16으로 dequant해서 bf16 matmul을 해야 하므로 텐서코어 가속을 못 받는다.
즉 **activation quant가 matmul을 저정밀 텐서코어 경로로 "열어주는" 열쇠**다 (bf16 대비 2~4×).

- **compute-bound**인 prefill(큰 M)에서 크게 작동.
- decode도 **배치가 커지면**(continuous batching, M=배치) compute-bound로 넘어가 같은 이점을 받음
  (이 repo의 "B≥16 fused quantized TC-GEMM" 결과가 이것).

## 2. Decode attention — 개념 정정 (GEMV 처리량 ❌ / KV 대역폭 ⭕)

❌ **decode attention은 GEMV "처리량(compute)"이 아니라 KV 캐시 "대역폭(bandwidth)" 문제다.**
decode의 attention은 q[1,D]·K[L,D]ᵀ → GEMV이고, 매 스텝 **KV 캐시 전체를 읽는 memory-bound** 연산이다.

- 이득은 **KV-cache quantization**(K/V를 저비트로 저장)이 **읽는 바이트를 줄여** 빨라지는 것 — 이 repo의
  MSAQ/MXFP6 KV 작업.
- **살아있는 Q activation을 양자화하는 건 decode에서 거의 이득 없음** (Q는 벡터 하나, memory-bound라
  compute가 병목이 아님).
- 즉 2번은 "activation GEMV 처리량"이 아니라 **"KV 캐시 대역폭 감소"** 로 써야 정확하다.

## 3. 다른 커다란 이점들 (자주 더 중요)

**(a) KV 캐시 메모리 → 더 큰 배치/컨텍스트 (서빙에서 가장 큰 이점일 수 있음)**
KV 캐시는 현대 서빙에서 **#1 메모리 소비처**다 (PagedAttention이 존재하는 이유). 양자화로 footprint를 줄이면:
- **동시에 더 많은 시퀀스/더 긴 컨텍스트가 HBM에 들어감 → 동시 처리량(aggregate RPS) 상승.**
- per-op 가속과 별개인 **서빙 레벨 throughput 이득** — 커널이 빨라지는 게 아니라 "더 많이 동시에 돌린다".

**(b) 멀티-GPU 통신량 감소 (자주 간과됨)**
Tensor/Pipeline parallel 서빙에서는 레이어 경계마다 **activation을 GPU 간 통신**(all-reduce/all-gather/
send-recv)한다. activation 양자화로 **통신 볼륨이 줄어** TP/PP가 빨라짐 (특히 통신-bound인 대형 모델/노드 간).

**(c) Disaggregated prefill/decode의 KV 전송**
prefill과 decode를 분리한 서빙(DistServe, Mooncake 등)에서는 **KV 캐시를 prefill 워커→decode 워커로 전송**한다.
KV 양자화가 그 전송량을 줄임.

**(d) (부수) offload / prefix-cache 저장**
긴 컨텍스트나 prefix 캐싱에서 KV를 CPU/SSD로 offload할 때, 양자화가 저장·전송 비용을 줄임.

---

## 정신 모델 (요약)

| 단계 | 병목 | 도움되는 양자화 |
|---|---|---|
| **prefill** (compute-bound) | matmul FLOPs | **A+W** → 저정밀 텐서코어 |
| **decode linear** (memory-bound, 작은 배치) | W 읽기 | **W** (W bytes↓); 큰 배치면 A+W |
| **decode attention** (memory-bound) | KV 캐시 읽기 | **KV** (cache bytes↓) |
| **서빙 전체** | HBM 용량, GPU간 통신 | **KV/A** → 배치↑·통신↓ |

**핵심 한 줄**: activation quant의 진짜 가치는 두 축이다 —
1. **compute를 텐서코어로 여는 것** (prefill, 큰 배치 decode), 그리고
2. **메모리를 줄여 더 많이 동시에 돌리는 것** (KV → 배치·컨텍스트·통신).
decode의 이득은 GEMV 가속이 아니라 **KV 대역폭/용량**에서 온다.

이 repo의 E2E 결과와도 일치: KV-read 1.9×가 RPS 1.27×로 희석되고 prefill은 타이였던 건, decode가
KV-대역폭 이득이고 prefill의 compute 이득은 별도(A+W 양자화)이기 때문이다.
