# Compile-time specialization of the MSAQ GEMV unpack — what it is & why it works

This explains the optimization landed in `csrc/w_gemv.cu` (`wonly_gemv_wide_uspec<int U, int GS>`):
specializing the u2/u3 decode-GEMV kernel on `(u, gs)` at **compile time** made it **1.8× faster**
(u2: 45→26 µs, u3: 44→24 µs), bit-identical output, all `tests/test_w.py` passing. The same numbers and
the surrounding investigation live in `packing_explained.md` §7–§12. This file is the conceptual "why".

---

## English

### 1. The core idea: *when* is a value known?

- **Runtime value** — only known when the kernel *runs*. A function argument like `int u, int gs, int UB`.
  At compile time nvcc sees "some integer in a register", nothing more.
- **Compile-time value** — known while nvcc *compiles*. A template argument `<int U_=2>`, a `constexpr`, or a
  literal `6`.

The generic kernel takes `u, gs, wbits = 8-u` as **runtime arguments**. The specialized kernel bakes them in as
**template constants** (`wonly_gemv_wide_uspec<2, 8>`). The output is bit-for-bit identical; what changes is how
much the compiler can do.

### 2. Why compile-time constants make the kernel faster — four mechanisms

**(1) Constant folding.** `wbits = 8 - u` with `u=2` → the compiler computes `6` and bakes it in. Masks like
`(1u << wbits) - 1` become the literal `63` instead of a runtime computation in a register.

**(2) Constant-width shifts vs variable shifts.** On the GPU, `x >> wbits` with a *runtime* `wbits` is a
variable-distance shift — the distance must sit in a register, it's harder to schedule/fuse, and on some pipes
slower. With `wbits` a constant `6`, `x >> 6` is a single cheap instruction with `6` encoded in it, freely
reorderable.

**(3) Static loop unrolling.** `for (i < UB/4)` with a runtime `UB` keeps a loop counter + branch. The original
code is `for (i < 6) if (i < UB>>2)` — six iterations, each paying a runtime branch. With `UB` constant the
compiler emits exactly `UB/4` straight-line loads, no counter, no branch.

**(4) The big one — statically resolving the rolling-buffer schedule → registers instead of local memory.**

First, a defining GPU fact:

> **A GPU cannot index its registers with a runtime variable.** It has a huge register file, but there is no
> "indexable register array". If you write `ureg[uwi]` and `uwi` is a runtime value, that array **cannot** live in
> registers — it spills to **local memory** (the per-thread stack, backed by L1/DRAM = slow).

The unpack loop is a rolling bit-buffer:
```cuda
uint32_t ureg[6];                                   // 6 packed words for one block
uint64_t ubuf = 0; int unb = 0, uwi = 0;
for (int k = 0; k < 32; ++k) {
    if (unb < wbits) { ubuf |= (uint64_t)ureg[uwi++] << unb; unb += 32; }  // refill, advance uwi
    int code = ubuf & umask;  ubuf >>= wbits;  unb -= wbits;               // pull one wbits-bit code
}
```
- **Generic (`wbits` runtime):** *when* a refill happens depends on `wbits`, so the compiler cannot know what
  `uwi` is at each `k`. → `ureg` spills to **local memory**; every code extraction is a slow local load.
- **Specialized (`wbits=6` constant + the k-loop `#pragma unroll`ed):** the compiler *simulates* the loop at
  compile time — k=0: `unb=0<6` → refill, `uwi→1`, `unb=32`; k=1: `unb=26`; k=2: `unb=20`; … For all 32 steps it
  knows `uwi` exactly. → `ureg[uwi]` becomes `ureg[0]`, `ureg[1]`, … **compile-time-constant indices** → `ureg`
  stays in **6 registers**, zero local loads.

That last point is the heart of the win: the same lines of C++ either live in registers or in local memory
depending only on whether the compiler can statically resolve the index — which requires the loop unrolled **and**
`wbits` constant.

### 3. What the profiler showed (RTX 3090, u2/gs8, 4096² GEMV)

| | generic (runtime u,gs) | specialized (compile-time) |
|---|---|---|
| kernel time | 48.8 µs | **25.3 µs** (1.93×) |
| SM throughput % | 61.5 | 52.6 |
| DRAM throughput % | 33.3 | **65.6** |
| instructions | 5.47 M | 5.33 M |
| registers/thread | 48 | 56 |

The kernel **flipped from SM-bound (61% SM / 33% DRAM) to BW-bound (53% SM / 66% DRAM)** — it now reaches the
~68% DRAM plateau a hand-specialized prototype hit. The bottleneck was never the algorithm; it was overhead the
compiler can erase once it knows `(u, gs)`. Cost: +8 registers (full unroll), but occupancy stays ~56%
warps-active.

### 4. The trade-off (why not always specialize?)

Each `<U, GS>` pair is a separate compiled kernel (code size, compile time). So you specialize the **few hot
combos** and keep the generic runtime kernel as a fallback for the rest — which is exactly the dispatch in
`wonly_gemv_wide_cuda`: a `SPEC(U,GS)` table for the common pairs, `MS_GEMV_NOSPEC=1` to force generic, and the
generic kernel for any combo not instantiated. Same idea as C++ `if constexpr` / template specialization
generally: trade a bit of binary size for letting the optimizer see the constants.

### 5. How to apply it elsewhere

Any kernel that (a) takes a shape/format parameter as a runtime int, (b) uses it for shift distances / mask widths
/ loop bounds, and (c) indexes a small array by a counter derived from it, is a candidate. Template the kernel on
those parameters, mark the inner loop `#pragma unroll`, and add a dispatch table for the hot values. The batched
and `wa` (W+A) GEMV paths still pass `u,gs` at runtime — they are the next candidates.

---

## 한국어

### 1. 핵심 아이디어: 값을 *언제* 아는가

- **Runtime 값** — 커널이 *실행될 때* 비로소 정해지는 값. `int u, int gs, int UB` 같은 함수 인자. 컴파일 시점에
  nvcc는 "레지스터에 담긴 어떤 정수"로만 봅니다.
- **Compile-time 값** — nvcc가 *컴파일하는 순간* 이미 아는 값. 템플릿 인자 `<int U_=2>`, `constexpr`, 리터럴 `6`.

generic 커널은 `u, gs, wbits=8-u`를 **runtime 인자**로 받습니다. 특수화 커널은 이걸 **템플릿 상수**로 박습니다
(`wonly_gemv_wide_uspec<2, 8>`). 출력은 비트 단위로 동일한데, 컴파일러가 할 수 있는 일이 완전히 달라집니다.

### 2. 왜 빨라지는가 — 4가지 메커니즘

**(1) 상수 폴딩.** `wbits = 8 - u`에서 u=2면 컴파일러가 `6`을 미리 계산해 박습니다. `(1u<<wbits)-1` 같은 마스크도
런타임 계산이 아니라 리터럴 `63`이 됩니다.

**(2) 상수폭 shift vs 가변 shift.** GPU에서 `x >> wbits`는 wbits가 런타임이면 거리만큼 미는 가변 shift라 거리가
레지스터에 있어야 하고 스케줄/퓨전이 어렵고 일부 파이프에선 느립니다. wbits가 상수 `6`이면 `x >> 6` — 6이 인코딩된
단일 저렴한 명령이라 자유롭게 재배치됩니다.

**(3) 정적 루프 언롤.** `for (i<UB/4)`에서 UB가 런타임이면 루프 카운터+분기가 남습니다. 원래 코드는
`for (i<6) if (i<UB>>2)` — 6번 돌며 매번 런타임 분기를 냅니다. 상수면 정확히 `UB/4`개의 직선 로드로 펼치고 분기를
없앱니다.

**(4) 가장 중요 — rolling-buffer 스케줄을 정적으로 해소 → local memory 대신 레지스터.**

먼저 GPU의 결정적 특성:

> **GPU는 런타임 변수로 레지스터를 인덱싱할 수 없습니다.** 거대한 레지스터 파일이 있지만 "인덱싱 가능한 레지스터
> 배열"은 없습니다. `ureg[uwi]`에서 `uwi`가 런타임 값이면 그 배열은 레지스터에 **못 두고** **local memory**(스레드별
> 스택, L1/DRAM backed = 느림)로 쫓겨납니다.

unpack 루프는 rolling 비트버퍼입니다:
```cuda
uint32_t ureg[6];                                   // 한 블록의 6개 패킹 워드
uint64_t ubuf = 0; int unb = 0, uwi = 0;
for (int k = 0; k < 32; ++k) {
    if (unb < wbits) { ubuf |= (uint64_t)ureg[uwi++] << unb; unb += 32; }  // refill, uwi 전진
    int code = ubuf & umask;  ubuf >>= wbits;  unb -= wbits;               // wbits비트 코드 하나 추출
}
```
- **generic (wbits 런타임):** refill 시점이 wbits에 따라 달라지므로 컴파일러는 각 k에서 `uwi`가 몇인지 **알 수
  없습니다**. → `ureg`가 **local memory**로 spill → 코드 추출마다 느린 local 로드.
- **특수화 (wbits=6 상수 + k-loop `#pragma unroll`):** 컴파일러가 루프를 컴파일 시점에 *시뮬레이션*합니다 — k=0:
  `unb=0<6`→refill, `uwi→1`, `unb=32`; k=1: `unb=26`; k=2: `unb=20`; … 32스텝 모두 `uwi`를 정확히 압니다. →
  `ureg[uwi]`가 `ureg[0]`, `ureg[1]`… **컴파일타임 상수 인덱스**가 되어 **6개 레지스터에 상주**, local 로드 0회.

마지막 항목이 이득의 핵심입니다: 똑같은 C++ 코드가, 컴파일러가 인덱스를 정적으로 해소할 수 있느냐(=루프 언롤 +
wbits 상수)에 따라 레지스터에 살거나 local memory로 쫓겨납니다.

### 3. 프로파일 결과 (RTX 3090, u2/gs8, 4096² GEMV)

| | generic (런타임 u,gs) | specialized (컴파일타임) |
|---|---|---|
| 커널 시간 | 48.8 µs | **25.3 µs** (1.93×) |
| SM throughput % | 61.5 | 52.6 |
| DRAM throughput % | 33.3 | **65.6** |
| 명령수 | 5.47 M | 5.33 M |
| regs/thread | 48 | 56 |

커널이 **SM-bound(61% SM / 33% DRAM)에서 BW-bound(53% SM / 66% DRAM)로 뒤집혔습니다** — 손으로 특수화한
프로토타입이 닿은 ~68% DRAM 정체에 도달. 병목은 알고리즘이 아니라 컴파일러가 `(u,gs)`만 알면 지울 수 있는
오버헤드였습니다. 비용은 +8 레지스터(완전 언롤)지만 occupancy는 ~56% warps-active로 유지됩니다.

### 4. 트레이드오프 (왜 항상 특수화하지 않나)

`<U, GS>` 조합마다 별도 컴파일 커널이 생깁니다(바이너리 크기, 컴파일 시간). 그래서 **자주 쓰는 소수 조합**만
특수화하고 나머지는 generic 런타임 커널로 fallback합니다 — 이게 `wonly_gemv_wide_cuda`의 dispatch입니다:
흔한 쌍의 `SPEC(U,GS)` 테이블, `MS_GEMV_NOSPEC=1`로 generic 강제, 인스턴스화 안 된 조합은 generic. C++의
`if constexpr` / 템플릿 특수화와 같은 발상 — 약간의 바이너리 크기를 내주고 옵티마이저가 상수를 보게 합니다.

### 5. 다른 곳에 적용하는 법

(a) shape/format 파라미터를 런타임 int로 받고, (b) 그걸 shift 거리 / 마스크 폭 / 루프 바운드에 쓰며, (c) 그로부터
파생된 카운터로 작은 배열을 인덱싱하는 커널은 모두 후보입니다. 그 파라미터로 커널을 템플릿화하고, 안쪽 루프에
`#pragma unroll`을 달고, 핫 값들의 dispatch 테이블을 추가하면 됩니다. 배치/`wa`(W+A) GEMV 경로는 아직 `u,gs`를
런타임으로 넘기므로 다음 후보입니다.
