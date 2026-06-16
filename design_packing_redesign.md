# 설계 문서 — packing 재설계 (extraction 비용 절감, 차세대)

> 목표: MSAQ가 MXINT8보다 **적게 읽는 바이트**를 실제 **시간 우위**로 바꾸는 것.
> 지금까지의 결론(Phase 5·7): occupancy/구조 최적화는 다 끝났고, 남은 병목은
> **unpack(extraction)의 instruction throughput** = packing 포맷의 본질 비용이다.
> 이 문서는 그 비용을 줄이기 위한 **packing 재설계 후보들**을 정리한다(연구성 과제).

---

## 1. 문제 정의 — "제약 삼각형"

좋은 packing은 동시에 셋을 만족해야 한다:

1. **적은 extraction 명령** — code 하나 꺼내는 데 ALU 명령이 적어야(decode가 싸야).
2. **byte 증가 없음** — MSAQ의 대역폭 우위(MXINT8의 0.68×)를 지켜야.
3. **coalesced 접근** — warp가 연속 메모리를 읽어야(트랜잭션 최소).

| 포맷 | (1) 적은 명령 | (2) byte 유지 | (3) coalescing |
|------|:---:|:---:|:---:|
| 현재 dense LSB-first | ❌ (straddle 다단계) | ✅ | ✅ |
| Stage 4b word-aligned | ✅ (bfe 1개) | ❌ (+20~26% padding) | ✅ |
| **목표** | ✅ | ✅ | ✅ |

- **현재 dense**: code가 byte·32bit 경계를 straddle → 2-byte load + shift/or/mask/
  sign-extend(~8~10 op). (1) 위반.
- **Stage 4b**: 32bit word 정렬로 bfe 1개 추출. 하지만 padding으로 (2) 위반 → **느려짐
  (revert됨)**. 단 정렬 overhead는 u에 따라 다름(아래 핵심).

---

## 2. 가장 싸고 유망한 실험 — **u=4에서 Stage 4b 재시도** (미검증)

Stage 4b의 padding overhead는 u에 의존한다(코드 폭 wbits=8-u):

| u | wbits | codes/word | aligned UB | dense UB | overhead |
|---|------|-----------|-----------|----------|----------|
| 2 | 6 | 5 | 28 | 24 | +16.7% |
| 3 | 5 | 6 | 24 | 20 | +20% |
| **4** | **4** | **8** | **16** | **16** | **+0%** |

**핵심: u=4는 wbits=4라 8 codes가 정확히 32bit word에 떨어진다 → dense == aligned,
padding 0.** 즉 u=4에서는 Stage 4b의 (2) 위반이 사라진다. bfe의 (1) 이점만 남아 **이론상
무손실로 더 빠를 수 있다.**

- 우리는 Stage 4b를 **u3gs8(기본, +20% padding)에서만** 측정하고 revert했다. **u=4는
  미검증.** shared도 u=4면 byte당 codes 정수 배수로 정렬 손실이 작다.
- **다음 실험(저비용)**: Stage 4b 코드(이미 작성·인증 완료, git history/Phase 5)를 되살려
  **u=4 한정**으로 benchmark. dense(u4) vs aligned+bfe(u4)를 같은 세션에서 비교.
  - 이기면: "u=4에서는 bfe로 MSAQ가 dense보다 빠르다"는 부분 승리. 정확도 설정이 u=4를
    허용하면 채택.
  - 비기거나 지면: extraction이 throughput이 아니라 latency-bound라는 추가 증거 → 설계 3으로.

---

## 3. Design A — dense 유지 + funnel-shift 추출 (byte 증가 없이 명령 절감)

padding 없이 (1)을 개선하는 정공법: **포맷은 dense 그대로, 커널이 더 똑똑하게 추출**.

- **블록 단위 vectorized load → register 상주 → funnel-shift 추출.**
  한 32-block의 UB 바이트(u3=20B)를 **한 번의 vectorized load**(`int4`+잔여)로 레지스터에
  올리고, 32개 code를 레지스터에서 추출. straddle은 `__funnelshift_r`(PTX `shf.r`, 두
  32bit 레지스터에 걸친 비트필드를 1~2 op로 추출)로 처리 → byte당 load 1회 + code당 ~2 op.
- **적용성**:
  - **KV (token-major, 4a)**: 한 key의 UB 바이트가 **연속** → warp가 협동 load(또는 lane이
    int4 load) 후 `__shfl`로 분배, 각 lane이 자기 code를 funnel-shift. dense라 padding 0,
    coalescing ✅. **KV에 적합.** (Stage 4b와 달리 포맷을 안 바꾸므로 (2) 만족.)
  - **GEMV (out-innermost)**: 한 열 o의 바이트가 OUT 간격으로 **흩어져 있어** vectorized
    load 불가. 대신 thread간(인접 o)은 이미 coalesced. funnel-shift로 straddle만 줄이는 건
    현재 2-byte load와 큰 차이 없음 → **GEMV엔 이득 적음.**
- **재인증**: 포맷 불변이므로 oracle/roundtrip 그대로. 커널 추출 로직만 emulation mirror에 반영.
- **리스크**: 협동 load + shfl 분배의 오버헤드가 extraction 절감을 상쇄할 수 있음(KV
  register-blocking이 occupancy로 손해 본 것과 유사한 함정) → 프로토타입 후 측정 필수.

---

## 4. Design D — IMMA / CUTLASS tensor-core 경로 (GEMM·W+A endgame)

(원 설계 문서의 최종 목표. 현재 GEMM/W+A는 correctness baseline일 뿐 tensor-core 미적용.)

- **W+A GEMM**: weight를 custom iterator의 **Shared→Register load()**에서 int8로 unpack →
  INT8 **IMMA** mainloop → `(scale_w*scale_a)` epilogue. unpack 비트연산이 `cp.async`
  prologue 뒤에 **숨는다**(mainloop 밖). → extraction이 그림자에 가려 사실상 공짜.
- **Prefill GEMM**: weight를 BF16로 dequant하는 prologue + BF16 tensor-core mainloop.
- **전제**: register-aligned + XOR-swizzle packing(bank-conflict-free `ldmatrix`)이 필요 →
  roundtrip 재인증 선행. `setup.py`는 이미 `$CUTLASS_DIR` 수용.
- **의의**: decode(GEMV/KV)는 memory-bound라 IMMA 이득이 작지만, **prefill/W+A(M 큼)는
  compute-bound라 tensor-core가 가장 큰 레버**. MSAQ/MXINT8 비교와 별개로 절대 성능의 본선.
- **리스크/규모**: 가장 큰 작업(CUTLASS 통합 + 재인증 + 전 scope 회귀). 별도 마일스톤.

---

## 5. 기각한 후보

- **bit-plane(planar) packing**: code를 비트 위치별 평면으로 → 추출이 wbits번 합산이라
  명령이 **더 많아짐**. 기각.
- **byte당 1 code 패딩**: straddle은 없지만 8bit/code = MXINT8과 동일 → byte 우위 소멸. 기각.
- **shared-mem LUT decode**: code 위치가 가변이라 테이블화가 지저분하고 shared 압박. 기각.

---

## 6. 권장 순서

```
[1] u=4 한정 Stage 4b 재측정      ── 가장 쌈(코드 이미 있음), 무손실 가설 검증
     └ 이기면 부분 채택 / 지면 latency-bound 확정
[2] Design A (KV funnel-shift)    ── dense 유지로 (2) 만족, KV 한정 프로토타입+측정
[3] Design D (IMMA/CUTLASS)       ── GEMM·W+A endgame, 별도 마일스톤(재인증 큼)
```

모든 단계 공통: **pack↔unpack roundtrip 재인증 → oracle gate → 전 scope 회귀**를
포맷 변경 시 반드시 선행(Stage 4b에서 확립한 규율).
