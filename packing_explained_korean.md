# MSAQ-signed 패킹 / 언패킹 — 설명

하나의 MXINT8 값이 어떻게 **세 개의 평면(plane)**으로 쪼개지는지, 그 레이아웃이 **u=4 (니블)** 일 때와
**u<4** 일 때 어떻게 달라지는지, 그리고 커널이 원소 하나를 언패킹할 때 정확히 어떤 읽기 / 시프트 / 마스크 /
부호 확장(sign-extend)이 일어나는지를 설명한다. 근거 코드: `ms_lib/pack.py` (오프라인 팩, NumPy) 와
`csrc/core/ms_utils.cuh` (디바이스 언팩). 블록 크기는 OCP MX 블록인 **`BLOCK = 32`** 이며, 가중치
`[OUT, K]` 는 출력 행마다 `nb = K/32` 개의 블록을 갖는다 (KV 는 이를 `[L, D]` → `head_dim` 축의 블록으로
재사용한다).

## 1. 값 모델 — 원소마다 E8M0 스케일 1개 + 정수 코드 2개

MSAQ-signed 는 int8 "워드" `q = q_upper·2^u + r_shared` 와 블록당 E8M0 스케일 1개를 저장하며, 값은
`x ≈ q · scale` 이다. 두 개의 조절 손잡이(knob):
- **`u`** = 8비트 만티사 중 그룹 전체에서 **공유(shared)**되는 하위 비트의 개수. 따라서 원소별
  **상위(upper)** 코드는 `wbits = 8 − u` 비트가 된다 (↑u ⇒ 원소당 비트 감소 ⇒ 바이트 감소, 더 공격적).
- **`gs`** = 공유 코드의 그룹 크기. `gs` 개 원소마다 `u`비트 **공유** 코드 1개
  (`n_group = 32/gs` 개의 공유 코드가 블록당 존재). ↑gs ⇒ 더 거친 공유 스케일, 바이트 감소.

### 분해 (`pack.decompose`, FP 반올림 1회)
```
s_base      = 2^(floor(log2(max|x_block|)) − 6)          # E8M0 (8비트 지수), 32-블록당 1개
s_unshared  = s_base · 2^u
q_upper[k]  = clip( round(x[k] / s_unshared), ±q_max ),   q_max = 2^(7−u) − 1     # (8−u)비트 SIGNED, 원소별
residual    = x − q_upper · s_unshared
r_shared[g] = clip( round( mean_{k∈group g}(residual) / s_base ), ±2^(u−1) )      # u비트 SIGNED, 그룹별
```
### 복원 (`pack.reconstruct` / `dequant_weight`)
```
q = q_upper[k] · 2^u + r_shared[g(k)]      # 유효한 MXINT8 정수 워드
x ≈ q · s_base                             # W-only: ×scale 로 float ;  W+A: q 를 int8 로 IMMA 에 투입
```
즉 원소마다 정보는 **3군데**에 나뉘어 있다: 블록의 E8M0 지수, 원소의 `(8−u)`비트 상위 코드,
그리고 그 원소가 속한 그룹의 `u`비트 공유 코드.

## 2. 세 개의 평면 (무엇이 어디에 저장되는가)

`pack_weight` 는 세 개의 SoA 평면을 내보낸다 (out 이 가장 안쪽; 블록 내 바이트 인덱스가 중간 축이다):

| 평면 | dtype | shape (행 우선) | 담는 것 | 32-블록당 바이트 |
|---|---|---|---|---|
| **`scale_exp`** | int8 | `[nb, OUT]` | E8M0 지수 (`scale = 2^exp`) | **1** |
| **`upper`** | uint8 | `[nb, UB, OUT]` | 32개 상위 코드 (각 `wbits` 비트), dense LSB-first | **`UB = 32·(8−u)/8`** |
| **`shared`** | uint8 | `[nb, SB, OUT]` | `n_group` 개 공유 코드 (각 `u` 비트), dense LSB-first | **`SB = ceil(n_group·u/8)`** |

코드들은 **빽빽하게(dense), LSB 우선(LSB-first)** 으로 패킹된다 (`_pack_codes_lsb`): 코드 `i` 는 블록
바이트 구간의 비트 `[i·width, i·width+width)` 를 차지하며 낮은 비트가 먼저 온다. 하나의 코드가 두 바이트에
**걸칠(straddle)** 수 있다. 행 우선 평면에서 걸친 코드의 두 바이트는 메모리상 **`OUT` 원소만큼** 떨어져 있다
(OUT 이 가장 안쪽이기 때문).

**열 우선 쌍둥이(Column-major twins)** `upper_cm [nb, OUT, UB]`, `shared_cm [nb, OUT, SB]` (단순한
전치(transpose)일 뿐, 바이트는 동일) 은 한 열의 `UB`/`SB` 바이트를 **연속(contiguous)**되게 배치하여,
넓은 로드(wide-load) GEMV 가 한 블록의 upper 전체를 하나의 `uint4` (u4) / 몇 개의 `uint32` (u2/u3)
coalesced 로드로 읽을 수 있게 한다. **KV** 는 토큰 우선 `[H, nb, L, BYTES]` (BYTES 가 가장 안쪽) 를 사용하여
한 key 의 바이트들이 연속되고, 고정된 key 에서 warp 의 32개 head-dim 읽기가 coalesce 되게 한다.

### 평면 크기
`UB` 는 오직 `u` 에만 의존하고, `SB` 는 `u` 와 `gs` 에 의존한다 (`n_group = 32/gs`, `SB = ceil(n_group·u/8)`):

| u | wbits=8−u | **UB** = 32·wbits/8 |
|--|--|--|
| 2 | 6 | 24 |
| 3 | 5 | 20 |
| 4 | 4 | 16 |

| 대표 구성 | UB | SB | +scale | bits/elem = (UB+SB+1)·8/32 |
|--|--|--|--|--|
| u4/gs2  | 16 | ceil(16·4/8)=8 | 1 | **6.25** |
| u4/gs8  | 16 | ceil(4·4/8)=2  | 1 | **4.75** |
| u3/gs16 | 20 | ceil(2·3/8)=1  | 1 | **5.50** |
| u2/gs8  | 24 | ceil(4·2/8)=1  | 1 | **6.50** |
(MXINT8 기준선 = 32 만티사 + 1 스케일 = 33 B/block = 8.25 bits/elem.)

## 3. u = 4 (니블) vs u < 4 (걸침) — 핵심 레이아웃 차이

**u = 4** → 상위 코드 = 4비트, 공유 코드 = 4비트 → 둘 다 **니블 정렬(nibble-aligned)**: 바이트당 정확히
2개의 코드, **코드가 바이트 경계를 절대 넘지 않는다**.
- 원소 `k` 의 upper 바이트는 `k>>1`; 니블 위치는 `(k&1)*4`. 그룹 `g` 의 shared 바이트는 `g>>1`.
- `UB = 16` (하나의 `uint4` = 16 B 가 블록의 upper 전체를 덮는다) — 가장 적은 바이트 구성이며 KV/GEMV
  의 기본값. 언팩은 평면당 **바이트 로드 1회 + `bfe.s32` 1회** (HW 가 마스크+부호를 처리):
  `unpack_ms_kv_elem_u4` / `w_gemv.cu`, `kv_attention.cu` 의 `uint4`+`bfe` 경로.

**u < 4 (u = 2, 3)** → wbits = 6 또는 5 (2의 거듭제곱이 아닌 서브바이트 폭) → 코드들이 빽빽하고
**바이트 경계를 걸친다**. 예: u=3 (5비트 upper): k=0 → 비트 0–4 (바이트 0); **k=1 → 비트 5–9 → 바이트 0 상위
3비트 + 바이트 1 하위 2비트 (걸침)**; 등등. 따라서 언팩은 마스킹 전에 조건부 2바이트 로드 + 시프트/OR 가
필요하다. `UB = 20` (u3) / `24` (u2). `extract_code` (랜덤 액세스) 또는 32비트 워드가 부족할 때만 다시
채우면서 코드마다 시프트+마스크 1회로 내보내는 롤링 비트 버퍼("스트리밍 언팩")로 처리된다.

## 4. 원소 하나 언패킹하기 — 읽기, 슬라이스, 마스크, 부호

블록 `blk` 의 원소 `k`, 출력 열 `o` 에 대해 (`g = k >> log2(gs)` 가 그 그룹):

**(a) 스케일** — **1바이트** `scale_exp[blk·OUT + o]` 읽기; `scale = exp2f(exp)`.

**(b) 상위 코드** (`upper` 에서 `wbits` 비트):
```
bit0 = k·wbits ;  byte = bit0>>3 ;  off = bit0&7
code = upper[base_u + byte] >> off                       # 슬라이스: 하위 `off` 비트를 버림
if (off + wbits > 8):  code |= upper[base_u + byte+1] << (8−off)   # 걸침: 다음 바이트에서 상위 비트를 끌어옴
code &= (1<<wbits) − 1                                    # 마스크: wbits 비트로
up_code = (code ^ (1<<(wbits−1))) − (1<<(wbits−1))        # 부호 확장 (2의 보수)
```
**1바이트, 또는 걸치면 2바이트** 를 읽는다 (걸침은 u<4 에서만 가능). 행 우선 평면에서 걸친 바이트는
`+OUT` 만큼 떨어져 있고, CM/KV 평면에서는 인접 바이트(`+1`)다.

**(c) 공유 코드** (`shared` 에서 `u` 비트): `SB` 바이트 위에서 `bit0 = g·u` 로 동일한 패턴 → `sh_code`.

**(d) 결합**: `word = up_code·2^u + sh_code`. **W-only**: `value = word · scale` (→ bf16). **W+A**:
`word` 를 INT8 IMMA 에 곧바로 투입하고, 두 블록 스케일은 에필로그에서 한 번에 접힌다(fold).

### u = 4 빠른 경로 (걸침 없음, 수동 마스크/부호 없음)
```
up_code = bfe_s32( upper[base_u + (k>>1)], (k&1)*4, 4 )   # 1바이트 로드 + 1 bfe.s32 (HW 가 마스크+부호)
sh_code = bfe_s32( shared[base_h + (g>>1)], (g&1)*4, 4 )
word    = up_code*16 + sh_code
```
`bfe.s32 d, word, pos, len` 은 `pos` 위치의 `len` 비트를 마스크하고 부호 확장하는 단일 PTX
bit-field-extract 로, 일반 경로의 시프트 / 조건부 2번째 바이트 로드 / 마스크 / xor-부호를 대체한다.

## 5. 블록당 얼마나 읽는가 (언팩 시점)
전체 32-블록에 대해 커널이 건드리는 것: **`scale` 1바이트** + **`upper` UB 바이트** (u4/u3/u2 에 대해
16 / 20 / 24) + **`shared` SB 바이트** (≤8). 32개 원소당 그 `UB+SB+1` 바이트가 MSAQ 의 풋프린트다 —
예: u4/gs2 = 25 B/block = 6.25 bits/elem vs MXINT8 의 33 B/block (32 만티사 + 1 스케일 = 8.25
bits/elem). 더 적은 바이트가 바로 디코드 커널이 시간으로 전환하는 메모리 트래픽 이득이다.

## 6. 디바이스 팩 꼬리 (쓰기 경로, 역방향)
KV 쓰기/추가(append)는 GPU 에서 다시 팩한다: `decompose_ms_block` (스레드가 `x[32]` 를 보유)이
`q_upper[32]` (`(8−u)`비트) + `r_shared[32/gs]` (`u`비트) + E8M0 지수 (`pack.decompose` 와 동일한
수치)를 만들고, `pack_codes_lsb` 가 이들을 `UB`/`SB`바이트 평면에 dense-LSB-first 로 쓴다 (역방향으로도
동일한 걸침 처리: `buf[by] |= code<<off; if(off+width>8) buf[by+1] |= code>>(8−off)`). 읽기 경로
(`unpack_ms_*`)는 이와 비트 단위로 정확히 일치한다.

— 파일: `ms_lib/pack.py` (`decompose`, `pack_weight`, `_pack_codes_lsb`, `dequant_weight`, `weight_int8`),
`csrc/core/ms_utils.cuh` (`unpack_ms_weight_elem`, `unpack_ms_kv_elem`, `unpack_ms_kv_elem_u4`,
`extract_code`, `bfe_s32`, `sign_extend`, `decompose_ms_block`, `pack_codes_lsb`).

---

## 부록: Q&A — 걸침 레이턴시, 메모리 granularity, coalescing

코드 근거: `csrc/w_gemv.cu`, `csrc/kv_attention.cu`, 본 문서 §3–4.

### Q1. Byte 걸침(straddle) 시 latency 손해는? round-trip이 한 번 더 발생하나?

**핵심: 현재 wide-load 경로에서는 추가 메모리 round-trip이 발생하지 않는다.** 손해는 ALU 몇 개뿐이다.
걸침이 **언제 처리되느냐**에 달려 있다:

- **Wide-load 경로 (현재 u2/u3 기본)**: 한 열/한 key의 `UB` 바이트(20~24B)를 **먼저 4-aligned `uint32`
  워드로 통째로 레지스터에 올린 뒤**, 걸침은 레지스터 안에서 shift/OR로 처리한다 (`w_gemv.cu:251-274`
  의 rolling bit-buffer). 두 바이트가 이미 둘 다 레지스터에 있으므로 걸친 코드를 꺼내는 비용 =
  **시프트 1회 + OR 1회 정도의 ALU**. 메모리 접근은 0번 추가된다. 실제로 `w_gemv.cu:231-234` 주석:
  *"the per-byte split that perfectly coalesced the load was tried and gave ZERO speedup"* → 걸침은
  병목이 아니고, 커널은 ALU-bound가 아니라 BW-bound라는 뜻.
- **Naive per-element 경로 (`unpack_ms_weight_elem`, cp.async/shared 경로)**: 걸침 시 **두 번째 바이트를
  한 번 더 읽는다** (§4b의 `if (off+wbits>8)` 분기). 단, 이 두 번째 읽기는 이미 **shared memory / L1**
  에 있는 바이트라 DRAM round-trip이 아니라 **on-chip 레이턴시(~20–30 cycle, shared)** 이다.

| 경로 | 걸침 비용 | round-trip 추가? |
|---|---|---|
| wide-load (uint32 레지스터) | shift+OR ALU 1~2개 | ❌ 없음 |
| per-element (shared) | shared 바이트 read 1회 | ❌ DRAM 아님, on-chip |

u=4(니블)는 애초에 걸침이 없어 `bfe.s32` 한 방(HW가 mask+sign)으로 끝나고, u<4도 wide-load 덕분에
걸침이 **메모리가 아니라 ALU 문제로 격하**된다.

### Q2. GPU memory access granularity 사다리 — 최소 단위

**(A) 명령어가 요청하는 per-thread 폭** vs **(B) 하드웨어가 실제로 옮기는 단위**를 구분해야 한다.

**(A) Per-thread load 명령어 폭 (레지스터로 들어오는 벡터 폭)**

| C++ 타입 | PTX/SASS | 바이트 | 레지스터 |
|---|---|---|---|
| `uint8_t` | `ld.global.u8` | 1 | 1 (부분) |
| `uint16_t`/`__half` | `ld.global.u16` | 2 | 1 (부분) |
| `uint32_t`/`float` | `ld.global.u32` (LDG.E.32) | 4 | 1 |
| `uint2`/`float2` | `ld.global.v2.u32` (LDG.E.64) | 8 | 2 |
| `uint4`/`float4` | `ld.global.v4.u32` (LDG.E.128) | **16** | 4 |

→ **per-thread 최대 폭 = 16바이트(`uint4`, 128-bit)**. 최소 = 1바이트지만, 1바이트를 요청해도 (B) 때문에
실제로는 32바이트가 움직인다.

**(B) 하드웨어가 실제 옮기는 단위**
- **Sector = 32 바이트** — DRAM↔L2 전송의 최소 단위. **이게 진짜 "최소 가져오는 단위"** (`kv_attention.cu:309`
  *"a DRAM/L2 sector is 32 bytes"*).
- **Cache line = 128 바이트** = 4 sector — L1/L2 캐시 라인 단위.

→ 한 스레드가 1바이트만 원해도 L2에서 32B sector가, 캐시 미스면 128B 라인이 통째로 움직인다.
**coalescing 효율 = 유용한 바이트 / 실제 옮긴 바이트**.

### Q3. 현재 weight / KV read의 coalescing — 축과 개수

**Weight GEMV (`wonly_gemv_wide_kernel`, `w_gemv.cu:172-`)**
- 평면: column-major `upper_cm [NB, OUT, UB]` — `UB`가 가장 안쪽(연속).
- 매핑: 스레드 `o` = **출력 열(OUT 축) 하나** 소유. 인접 스레드 → 인접 출력 열.
- 축: warp의 32 스레드가 **OUT 축으로 32개 열**을 동시에. 인접 열은 `UB`바이트 간격
  (`w_gemv.cu:167` *"consecutive columns are UB B apart = warp-contiguous"*).
- 폭: **u4** → 열당 `UB=16B` = **`uint4` 1회** (`w_gemv.cu:198`). **u2/u3** → `UB=20/24B` =
  **4-aligned `uint32` 5~6개** 로드 후 레지스터 추출.

**KV cache read (`kv_decode_wide_kernel`, `kv_attention.cu:335-`)**
- 평면: token-major `[H, NB, Lk, UB]` — `UB`(BYTES)가 가장 안쪽.
- 매핑 (Pass 1, scores): 스레드 `t` = **key 하나(L 축)** 소유 (thread-per-key). 인접 스레드 → 인접 key.
- 축: warp의 32 스레드가 **key(L) 축으로 32개 key**를 동시에. 인접 key는 16B 간격.
- 폭: **u4 → 블록당 `uint4` 1회**, warp 전체로 **32 key × 16B = 512B 완전 연속·완전 유용**
  (`kv_attention.cu:315-316`).
- Pass 2 (P·V): key 축으로 reduce해야 하므로 thread-per-key 불가 → V를 shared로 coalesced 스테이징 후
  thread-per-d(head_dim 축) 누적.

**왜 이렇게 바꿨나 (`kv_attention.cu:307-312`)**

| 매핑 | warp가 원하는 유용 바이트 | sector util | 결과 |
|---|---|---|---|
| 옛날 warp-per-key (head_dim 축 32 lane) | 32 u4코드 = **16B** / 32B sector | ~50% (실효 BW ~38%) | MSAQ가 MXINT8보다 1.5× 느림 |
| 현재 thread-per-key (L 축 32 thread) | **512B** / 512B | **100%** | BW-bound 해소 |

MXINT8은 1바이트/원소라 head_dim 축 32 lane이 32B sector를 꽉 채우지만, MSAQ u4는 0.5바이트/원소라 같은
매핑에서 sector의 절반만 유용 → **축을 head_dim에서 key로 돌려** 해결.

### Q4. 이 단위들이 L2 / register / SM / warp / thread와 어떻게 연결되나

```
DRAM ──32B sector──▶ L2 ──128B line──▶ L1/SMEM(SM당) ──▶ 레지스터(thread당) ──▶ ALU(bfe/shift)
        ▲ coalescing은 여기서 warp 단위로 결정됨
```

- **Thread ↔ 레지스터 폭**: `uint4` load = **4개의 32-bit 레지스터**를 한 번에 채움. u4 커널이 16B를
  `uw[4]`(=4 레지스터)에 올리고 32개 코드를 `bfe`로 추출 → **언팩이 레지스터 내부 ALU로 끝나 메모리
  재방문 0회** (Q1에서 걸침이 싼 이유).
- **Warp(32 thread) ↔ sector/coalescing**: coalescing은 **warp 단위**로 일어난다. HW가 32 스레드의 주소를
  모아 32B sector / 128B line으로 묶음. "32 스레드 × 16B = 512B 연속"이 의미를 갖는 이유 — 32B sector
  16개가 빈틈없이 채워짐(100% util).
- **SM ↔ L1/Shared**: shared staging(V, cp.async 경로의 upper plane)은 **SM 로컬 L1/SMEM**(Hopper 기준
  SM당 최대 ~228KB)에 올라감. per-element 걸침의 두 번째 바이트 read가 "DRAM이 아니라 on-chip"인 이유.
- **L2 ↔ device-wide**: L2(수십 MB, 전 SM 공유)는 sector(32B) 단위 캐시. weight를 여러 split-K 블록이
  재사용할 때 L2 hit가 DRAM round-trip을 막아줌.

**한 문장 요약**: 최소 전송 단위는 **32B sector**(L2/DRAM), per-thread 최대 로드 폭은 **16B `uint4`**,
coalescing 판정은 **warp(32 thread)** 단위. 이 프로젝트는 (a) 평면을 column/token-major로 깔아 한 thread의
`UB` 바이트를 연속시키고 (b) thread를 OUT 열 / key에 매핑해 warp 32개가 연속 sector를 100% 채우게 하며
(c) wide-load로 올린 레지스터 안에서 bfe/shift로 언팩을 끝내 **걸침을 메모리 문제가 아닌 ALU 문제로 격하**
시킨다.

### Q5. 평면을 4개로 쪼개(upper 4bit / lower (4-u)bit / shared u-bit / scale) 걸침을 제거하면 decode가 빨라질까?

제안: `(8−u)`비트 unshared upper 코드를 **상위 4비트 니블 + 하위 `(4−u)`비트**로 쪼개고,
shared `u`비트와 scale을 더해 **4개 평면**으로 나눈다. 상위 4비트는 니블 정렬이라 걸침이 없고, 나머지는
한데 뭉쳐 비트 슬라이스로 꺼낸다.

**먼저, 비트 레이아웃 관점에서는 아이디어가 정확하다.** `8−u` (u=2 → 6비트, u=3 → 5비트)는 2의 거듭제곱이
아니라 걸치지만, `4 + (4−u)` 로 쪼개면 **4비트는 8을 나누고, `(4−u)` ∈ {2, 1} 도 8을 나눈다** → 두 조각 모두
걸침이 사라진다. shared 평면만 u=3일 때 3비트라 걸칠 수 있으나 그 평면은 작다(≤8B). 즉 **per-element 걸침은
실제로 제거된다.** 비트 예산도 그대로다: u=2 기준 16B(상위 니블) + 8B(하위 2비트) = 24B = 기존 `UB`.

**그러나 decode latency는 거의 줄지 않을 가능성이 높다.** 이유는 §부록 Q1에서 확인한 사실 때문이다:

1. **걸침은 이미 메모리 비용이 아니다.** wide-load 경로에서 걸침은 레지스터 안 shift/OR(ALU 1~2개)로
   처리되고, 이건 메모리 레이턴시 뒤에 숨는다. `w_gemv.cu:231-234`가 *"perfectly coalesced the load …
   gave ZERO speedup"* 라고 못박는다 — 커널은 **BW-bound이지 ALU-bound가 아니다**. 걸침을 없애도
   숨어 있던 ALU를 없애는 것이라 wall-clock에 안 나타난다.
2. **메모리 트래픽이 0만큼 줄어든다.** 총 바이트가 동일(24B → 16+8B)하므로 BW-bound 커널에서 시간 이득 = 0.
3. **평면이 3 → 4개로 늘어 오히려 손해 가능.** base 포인터·주소 계산이 늘고, 한 번의 `UB`바이트 연속 로드가
   `uint4`(16B) + `uint2`(8B) 두 벡터 로드로 쪼개지며, 레지스터 압력이 커진다.
4. **"뭉쳐두기"는 인덱싱을 복잡하게 만든다.** 하위 코드는 per-element(32개), shared는 per-group(`n_group`개)
   으로 개수·접근 패턴이 다르다. 한 평면에 합치면 두 영역의 **경계 자체가 다시 걸칠 수** 있어, 깨끗한 별도
   평면으로 두는 편이 낫다(그러면 다시 4평면 → 3번 문제).
5. **부호 처리 주의.** `q_upper`는 `(8−u)`비트 **signed**다. 니블과 하위를 독립적으로 `bfe.s32` 하면 안 되고,
   둘을 unsigned로 꺼내 `(8−u)`비트로 재조립한 뒤 **한 번** 부호 확장해야 한다. "평면당 깨끗한 bfe" 라는
   매력이 약간 줄어든다(여전히 싸긴 하다).

**언제 도움이 될 수 있나**: 오직 **ALU/shared-latency-bound** 인 경우 — 즉 좁은 per-element
`unpack_ms_weight_elem`(cp.async) 경로. 하지만 이 프로젝트는 그 경로가 패배자라 wide-load로 **이미 떠났고**,
"빠르고 + 가장 적은 바이트 + bit-exact" 가 필요하면 그냥 **u=4** 를 쓰면 된다(걸침이 처음부터 없음).
u=2/3는 per-element 정밀도(upper 비트 ↑)를 위해 **일부러 바이트를 더 쓰는** 정확도 우선 구성이다.

**결론(잠정)**: 걸침 제거라는 메커니즘은 맞지만, **이미 숨겨진 ALU 비용**을, 그것도 **바이트를 안 줄이면서**
없애려는 시도라 BW-bound 커널에선 best-case 중립·plausibly 약간 손해다 ... **라고 Q5에서 추정했으나,
아래 Q6의 Nsight 프로파일이 이 "BW-bound" 전제를 뒤집었다. Q6를 반드시 함께 읽을 것.**

### Q6. (실측) Nsight Compute로 u2/u3 decode가 정말 BW-bound인지 확인

Q5의 핵심 전제 — "decode 커널은 BW-bound라 걸침/언팩 ALU를 줄여도 소용없다" — 를 **직접 측정해 검증**했다.
결과는 **전제가 틀렸다**: 이 크기의 decode는 DRAM-대역폭에 막혀 있지 않다.

환경: RTX 3090 (Ampere sm_86), Nsight Compute 2022.1, split-K wide-load 커널, single-request decode
(GEMV: OUT=K=4096 / KV: Lk=4680, H=8). 재현: `tests/ncu_uprobe.py` (GEMV), `tests/kv_ncu_driver.py` (KV).
지표는 각 유닛의 peak 대비 throughput %, `sector-util` = global load의 sector당 유효 바이트 % (낮을수록
sector 낭비).

| 경로 | u | **SM%** | DRAM% | L2% | L1% | sector-util% | 시간(µs) |
|---|---|---|---|---|---|---|---|
| GEMV | u2/gs8 | **57.7** | 30.9 | 14.3 | 28.5 | 15.7 | 50.9 |
| GEMV | u3/gs8 | **54.6** | 27.1 | 12.6 | 28.2 | 18.3 | 50.1 |
| GEMV | u4/gs8 | 68.3 | 31.7 | 20.6 | **68.3** | 39.6 | **37.5** |
| KV   | u3   | **33.0** | 10.2 | 5.0 | 29.5 | 27.2 | 77.0 |
| KV   | u4   | 37.7 | 13.3 | 7.5 | **35.3** | 49.6 | **50.0** |

**판독:**
1. **DRAM-대역폭 bound이 아니다.** DRAM%가 어디서도 ~32%를 못 넘고 KV는 10~13%에 불과하다. "DRAM BW에
   막혀서 바이트만이 레버"라는 단순 그림은 **이 single-request 크기에선 거짓**이다. (모든 SOL 유닛이
   <70% → 사실상 **occupancy/latency-bound** 영역. 문제 크기가 작아 어떤 유닛도 포화시키지 못함.)
2. **u2/u3 GEMV는 SM(연산/발행 파이프)이 최대 유닛(54~58%)** 이고 DRAM/L1은 ~28%다. 즉 streaming-unpack의
   shift/mask/OR(언팩 ALU+LSU)가 임계경로에 있을 개연성이 크다 — **ALU-leaning**.
3. **u4가 모든 경우 더 빠르다** (GEMV 37.5 vs 50.1µs = 1.34×, KV 50 vs 77µs = 1.54×). u4는 (a) 바이트가
   적고(UB16<20/24) (b) 걸침이 없고 (c) 언팩이 단일 `bfe` 다. sector-util도 u4가 크게 높다(GEMV 39.6 vs
   15~18, KV 49.6 vs 27). u4의 병목은 L1TEX(로드 파이프)로 옮겨가는데, 이게 "좋은" 종류의 병목이다.
4. **KV의 "BW-bound"는 DRAM 포화가 아니라 sector-효율(effective-BW) bound** 이라는 doc의 주장과 일치한다:
   DRAM%는 낮고, 대신 L1%와 낮은 sector-util(27%)이 한계다. u4가 sector-util을 49.6%로 올려 빨라진다.

**Q5에 대한 정정:** 측정 결과 "decode는 BW-bound라 언팩 단순화가 무의미"라는 Q5의 결론은 **틀렸다.**
- 이 크기의 decode는 **DRAM-BW가 아니라 SM/발행/latency**가 한계다. 따라서 걸침 제거 = 언팩 명령 수 감소는
  **실제로 도움이 될 수 있는 레버**다(특히 SM-bound인 GEMV u2/u3).
- Q1의 "ZERO speedup" 실험은 **로드를 coalescing**(메모리 측)한 것이라, 메모리가 병목이 아니라는 본 측정과
  **모순되지 않는다.** 그 실험은 언팩 ALU를 줄이지 않았으므로, 사용자의 제안(언팩 단순화)을 반박하지 못한다.

**그래도 남는 단서(상한이 있다):**
- u4의 우위 1.34×에는 **걸침 제거뿐 아니라 적은 바이트 + 높은 sector-util**이 섞여 있다. 사용자의 제안은
  u2의 바이트 수(24B)를 **그대로 두므로**, 잡을 수 있는 건 언팩-ALU 몫뿐 → 기대 이득은 1.34×보다 **훨씬
  작다.** 게다가 평면 4개로 인한 추가 로드/주소계산/레지스터 압력이 이를 더 깎을 수 있다(Q5의 3·4번).
- 부호 재조립(Q5의 5번) 비용도 그대로 남는다.

**최종 권고:** 프리미스가 깨졌으니 **가치 있는 실험이다.** 다만 "u4만큼 빨라진다"가 아니라 "u2의 바이트는
유지하되 언팩만 u4급으로 싸게" 가 정확한 기대치다. 검증법: u2를 (상위 4bit 니블 평면 + 하위 2bit 평면 +
shared + scale)로 재패킹한 프로토타입을 만들어 **같은 ncu 지표**(특히 SM% 하락과 시간)를 u2 baseline과
비교. SM%가 내려가고 시간이 줄면 채택, 평면 증가로 L1/발행이 되레 오르면 기각. 측정 크기(single-request,
저점유)도 명시할 것 — **prefill/배치(B≫1)에서는 DRAM-BW로 균형이 옮겨가** 결론이 다시 바뀔 수 있다(그땐
바이트 축소가 다시 주 레버).

### Q7. (실측) 4-plane u2 프로토타입 — 만들어 재보니 **이 아이디어는 효과가 없다**

Q6의 권고대로 4-plane u2 재패킹을 실제로 구현해 3-plane streaming u2와 **동일한 launch config** 한 하네스
안에서 비교했다(차이는 오직 언팩). 재현: `tests/proto_split4_u2.py`.
- **streaming** (3-plane): `upper_cm` 24B → 6× `uint32`, rolling 6-bit 걸침 추출.
- **split4** (4-plane): 니블 `[16B]` `uint4` + 하위 2bit `[8B]` `uint2` (둘 다 걸침 없는 shift+mask) 후
  6-bit signed `q_upper = sign6((top4<<2)|lo2)` **재조립**. 블록당 총 바이트는 동일.

비트 정확성: `max|streaming−split4| = 0` (수학적으로 동일), 둘 다 fp 레퍼런스와 상대오차 1.6e-7.

| 지표 (RTX 3090, ncu 단일 launch) | streaming (3-plane) | split4 (4-plane) |
|---|---|---|
| **실행 명령 수** | 5.35 M | **6.37 M (+19%)** |
| sector-util% (global ld) | 15.7 | **48.3** |
| DRAM% | 61.8 | 51.4 |
| L1% | 49.1 | 36.7 |
| SM% | 50.0 | 50.9 |
| registers/thread | 39 | 40 |
| 시간 — 300-iter bench | 33.1 µs | 32.5 µs (**1.018×**) |

**판정: 4-plane 분할은 decode를 빠르게 하지 못한다(~±2%, 무승부).** 가설의 전제 — "byte 걸침이 비싸다" —
가 **틀렸다**: rolling-buffer 걸침은 이미 코드당 ~1 shift+mask(refill은 amortize)다. 6-bit를 4+2 정렬 필드로
쪼개면 원소마다 **재조립**(`(top4<<2)|lo2`)을 두 개의 따로 로드된 레지스터에서 해야 하고, 이게 제거한 걸침보다
**ALU를 더 쓴다** → 명령 수가 줄기는커녕 **+19%**. split4의 유일한 실익은 `uint4`+`uint2` 정렬 로드 덕의
**sector-util(15.7→48.3%)** 인데 single-request decode에선 이게 지배적이지 않다.

이는 구현 탓이 아니라 **구조적**이다: 2의 거듭제곱이 아닌 코드를 정렬 서브필드로 쪼개면 반드시 재조립이
필요하고, 그 재조립이 바로 비용이다. 진짜 레버는 여전히 **u=4** — upper 코드가 곧 4비트라 재조립이 없고
(단일 `bfe`), 바이트도 가장 적고 sector-util도 최고다(Q6의 1.34× 승자). **Q5/Q6의 "해볼 가치 있다"는
잠정 결론은 이 실측으로 기각된다.**

### Q8. "BW-bound가 아니다" → 그럼 BW-bound 영역까지 끌고 갈 수 있나? (실측)

single-request decode가 latency/occupancy-bound라면(Q6), 자연스러운 발상: "한 번에 더 많이 긁어와 BW
roofline까지 밀어붙이자." 함정은 — **더 긁어올 게 없다. 바이트 수는 고정**이다(가중치는 정확히 한 번만 읽음;
이 u2 GEMV는 13.6 MB). 진짜 레버는 "더 큰 fetch"가 아니라 **동시에 떠 있는 in-flight 요청을 늘리는 것**
(occupancy / split-K / MLP / 배치) — DRAM latency를 숨겨 같은 바이트를 더 빨리 흘려보내는 것이다. Q7
프로토타입에서 split-K(동시성 손잡이; 커널·바이트 동일)를 스윕하면:

| split-K | 시간 | DRAM% | SM% | warp-occ% | DRAM 바이트 |
|---|---|---|---|---|---|
| 1  | 136 µs | 11.1 | 8.5 | 16.7 | 13.64 MB |
| 8  | 27.2 µs | 59.1 | 46.5 | 25.7 | 13.70 MB |
| 32 | **25.8 µs** | **68.0** | 55.7 | 73.7 | 13.82 MB |
| — BW roofline (13.6 MB / 936 GB/s) | **14.6 µs** | 100 | — | — | — |

**판독:**
1. **방향은 맞다.** 동시성을 올리니 DRAM 11→68%, 시간 136→26 µs(5.3×): latency-bound → BW-bound 쪽으로
   끌려갔다. 기대한 그대로.
2. **하지만 "더 긁어온다"는 멘탈 모델은 틀렸다 — 바이트는 고정이다.** `dram__bytes`는 모든 split-K에서
   ~일정(13.64→13.82 MB). 이득은 occupancy(16.7→73.7%), 즉 **같은** 바이트를 **더 많이 동시에** 긁은 것이지
   더 크게 긁은 게 아니다. (per-thread 폭은 이미 `uint4`=16B로 최대; 더 키우면 한 명령으로 안 되고 occupancy가
   깎인다.)
3. **~68% DRAM / 26 µs에서 정체 — roofline 14.6 µs까지 ~1.8× 모자란다.** 벽은 **sector 활용률**이다:
   u2-streaming은 32B sector당 유효 15.7%뿐이라, DRAM이 100% 차기 전에 L1TEX/L2 파이프가 절반 낭비된 sector를
   나르느라 먼저 포화한다. **바로 여기서** "transaction당 유효 바이트 늘리기"(정렬 — split4의 `uint4`+`uint2`가
   48%로 올린 것)가 천장을 올린다. B=1에선 명령 벽에 다시 막혀 빛을 못 봤을 뿐(Q7).

요약하면, roofline에 **도달**하는 레버 vs roofline 자체를 **내리는** 레버: (a) **sector 활용률**(정렬/낭비
감소)은 정체 천장을 올리고, (b) **배치 B>1**은 가중치를 토큰들에 재사용 → arithmetic intensity↑ → compute-bound
까지 roofline을 탄다(repo의 batched-decode 경로), (c) **bits/elem 축소(u4)**는 roofline 자체를 내린다(14.6 →
~9 µs). 참고로 프로덕션 `wonly_gemv`는 50 µs / DRAM 31%(Q6)인데 이 lean 프로토타입은 split-K 32에서
26 µs / DRAM 68% — 프로덕션 커널(sepsc 경로, 96 regs, 낮은 occupancy)이 BW를 놀리고 있다는 뜻이라, occupancy를
올리고 per-thread ALU를 줄이는 건 **아직 남은 실질 튜닝 여지**다.

### Q9. occupancy 올리기 + sector-util 최대화 — 둘 다 u2 GEMV에 효과 없는 이유 (실측)

세 가지 후속 실험, 모두 u2 GEMV에서 실측.

**(a) 유효 바이트가 왜 그렇게 적나(15.7%)?** column-major `upper_cm[nb,OUT,UB]`: 스레드 `o`가 column `o`
소유, 그 `UB`바이트는 연속. 하지만 `UB=24`는 2의 거듭제곱 벡터 폭이 아니라 블록이 6× `uint32`로 로드된다.
**한 개의** word-load 명령 안에서 lane `o`는 `base + o*24 + i*4`를 접근 → warp의 32 lane이 **24바이트 stride**
(각 4 유효바이트)로 ~24개 32B sector에 흩어짐 → 128 유효 / 768 fetch = **16.7%**. u4는 탈출: `UB=16` = 1
`uint4`, lane stride 16 = 로드 폭 → 32 lane이 512바이트 연속 → 높은 util.

**(b) sector-util 최대화 — `bytesplit` ("블록을 통째로 로드해 레지스터에서 split"을 제대로 적용).** 문자 그대로의
단일 뚱뚱한 plane은 coalesce 안 된다(stride 24/26 ≠ 2의 거듭제곱). 해법은 **코드가 아니라 24바이트를** 16B +
8B plane으로 쪼개 각 plane의 per-thread stride가 로드 폭과 같게(`uint4`=16, `uint2`=8) 만들고, **같은**
rolling 6-bit 언팩을 6개 레지스터에서 도는 것 — 코드가 16/24 경계를 자연스럽게 걸치므로 **재조립 없음**(split4와
다름). split-K 32 실측:

| 커널 | sector-util% | DRAM% | DRAM 바이트 | 명령수 | regs | 시간 |
|---|---|---|---|---|---|---|
| streaming (6× uint32) | 15.7 | 67.2 | 13.76 MB | 5.47 M | 39 | 26.3 µs |
| **bytesplit (uint4+uint2)** | **48.3** | 68.4 | 13.67 MB | **5.41 M** | 38 | 26.0 µs |
| split4 (의미 분할) | 48.3 | 58.4 | 13.80 MB | 6.49 M | 40 | 26.1 µs |

`bytesplit`은 split4의 coalescing을 streaming의 명령 수로 달성(출력 비트 동일) — 기계적으로 양쪽의 장점.
**하지만 시간은 그대로다.** 이유: `dram__bytes`가 셋 다 동일. streaming의 strided 로드에서 "낭비된" sector
바이트는 다음 word-load가 곧바로 재사용한다(블록 32열의 6워드가 한 768B 런에 있어 L2가 한 번에 서빙). 즉 낮은
sector-util은 L1 **요청 수**만 부풀렸을 뿐 DRAM 트래픽은 그대로고, 커널은 L1-요청-bound가 아니다. **column-major
weight GEMV에서 sector-util은 헛다리**다(L2가 중복을 흡수) — KV token-major는 각 스레드가 서로 다른 key를 한 번만
읽어 반쪽 sector가 재사용 안 되므로 거기선 진짜 이득.

**(c) 프로덕션 occupancy 올리고 재측정.** 손잡이는 env 전용(`MS_GEMV_SPLITK_MULT`, `MS_GEMV_SEPSC`; 리빌드
불필요). full-op u2 시간(kernel+combine): default `MULT=16`→44.9 µs, `32`→43.8 µs, `64`→48.0 µs(과분할은
손해). wide 커널 ncu(`MULT=32`): **warps-active 71.7%**(이미 높음, 48 regs로 reg-limited), **SM 61.5% vs
DRAM 33%**, sector-util 15.7%. 즉 프로덕션 u2는 **높은 occupancy에서 SM/issue-bound — occupancy-bound도
memory-bound도 아니다.** 그러니 split-K를 올려도 소용없고, 한계는 per-element ALU(언팩 + separated-scale
누산)다. `sepsc=1`(48.8 µs)이 `sepsc=0`(53.8 µs)보다 빠른 것도 주석 주장대로 확인됨.

**Q7~Q9 종합.** u2/u3 GEMV decode는 single-request 크기에서 근본적으로 **SM/ALU-bound**다. {occupancy↑,
coalescing/sector-util↑, 걸침 제거} 어느 것도 per-element ALU를 줄이지 못해 도움이 안 된다. 이걸 움직이는 유일한
레버는 **바이트와 언팩 명령을 동시에 줄이는 것** = **u=4**(단일 `bfe`, 16B, 재조립 없음: 33–38 µs vs u2의
44–50 µs), 또는 **배치 B>1** 분할상환. (lean u2 커널은 26 µs / DRAM 68%까지 가므로, 프로덕션 u2 wide 커널은
그 2배·SM-bound라 여지가 있지만 그건 occupancy가 아니라 ALU/레지스터에 있다.)

### Q10. sepsc가 bloat인가? + 배치 스케일링 (실측)

**(a) sepsc는 프로덕션 bloat이 아니다.** lean u2 커널에 separated-scale 누산을 추가(`gemv_u2_sepsc`, bytesplit
로드)해 per-element 경로와 같은 config로 비교:

| lean 변형 (split-K 32) | 시간 | vs streaming |
|---|---|---|
| streaming (per-elem) | 32.5 µs | 1.00× |
| bytesplit (per-elem) | 32.8 µs | 0.99× |
| sepsc (separated-scale) | 31.9 µs | 1.02× |

전부 노이즈 범위 → **lean 커널에서 sepsc는 득도 실도 없다.** 즉 프로덕션 48 µs vs lean 26–32 µs 격차는
sepsc가 아니다. 남은 용의자는 **runtime-generality**: 프로덕션은 `u,gs,UB,SB`를 런타임 int로 넘겨(가변폭 shift,
레지스터 증가, 깔끔한 unroll 불가) 처리하는데, lean 커널은 u2/gs8을 컴파일타임에 박았다. 프로덕션 커널을
(u,gs)별로 특수화하는 게 유력한 여지다 — occupancy도 sepsc도 아님.

**(b) 배치 스케일링.** 배치 커널은 블록당 가중치를 한 번 언팩해 B 토큰에 재사용(weight-DRAM + 언팩-ALU 분할상환).
dense bf16 matmul(cuBLAS, 텐서코어)과 비교 스윕:

| B | naive-MSAQ µs | µs/tok | bf16 µs | bf16 µs/tok |
|---|---|---|---|---|
| 1  | 35.5 | 35.5 | 44.0 | 44.0 |
| 2  | 35.5 | 17.8 | 44.1 | 22.1 |
| 4  | 59.8 | 15.0 | 44.2 | 11.0 |
| 8  | 111  | 13.9 | 44.2 | 5.5 |
| 16 | 179  | 11.2 | 44.8 | 2.8 |
| 32 | 369  | 11.5 | 44.7 | 1.4 |

판독: **MSAQ µs/token은 B가 커지면 실제로 떨어진다**(35→11.5: §Q8 예측대로 weight-read 분할상환 작동). **하지만
총 시간이 B=2 이후 ~선형 증가하고, per-token으로 B≈2부터 bf16에 진다.** 이유: naive 커널은 **가중치**는
재사용하지만 B개 내적을 **CUDA 코어에서 scalar fp32 madd**로 돌리고, `x[b*K+…]` 활성 로드가 K stride로
uncoalesced다 → B가 커지면 weight-BW가 아니라 **연산**이 지배한다. bf16 matmul은 **텐서코어** 덕에 ~44 µs로
평평(B≈공짜). 교훈: weight-read 분할상환은 필요조건일 뿐 충분조건이 아니다 — 경쟁력 있는 배치 경로는 **텐서코어
(INT8 IMMA)**, 즉 repo의 `wa_gemm`/batched-decode 경로를 써야지 scalar 재사용 루프로는 안 된다. (`wa_gemm`은
여기서 정확성은 확인됐으나(rel ~3%) 이 고립 microbench에선 B 무관 ~600 µs 고정 오버헤드를 보여 tuned 영역을
대표하지 못함 — 제대로 된 배치 평가는 repo의 배치 하네스에서.) 정리: single-token decode는 byte/ALU-lean MSAQ
커널이 유리하고, B가 ~2 넘어가면 텐서코어 경로가 이기며, 올바른 비교는 scalar MSAQ가 아니라 INT8-IMMA-MSAQ vs
bf16이다.

### Q11. 적용 완료: (u,gs) 특수화 프로덕션 커널 — 1.8× (리빌드, 테스트 통과)

Q10(a)가 가리킨 runtime-generality 병목을 확인하고 고쳤다: generic `wonly_gemv_wide_kernel<false>` 옆에
`wonly_gemv_wide_uspec<int U, int GS>`(컴파일타임 `U/GS/WBITS/UB/SB`)를 추가하고, `wonly_gemv_wide_cuda`가
알려진 조합은 이쪽으로 dispatch하고 나머지는 generic으로 fallback(`MS_GEMV_NOSPEC=1`로 generic 강제, A/B용).
비트 동일(`max|spec−generic| = 0`, u2/u3 × gs8/16), `tests/test_w.py` 45개 전부 통과.

| config | generic (full op) | **specialized** | speedup |
|---|---|---|---|
| u2/gs8 | 45.1 µs | **25.7 µs** | **1.76×** |
| u3/gs8 | 43.9 µs | **23.6 µs** | **1.86×** |

특수화 u2 커널 vs generic ncu (Q9c):

| | generic | specialized |
|---|---|---|
| 시간 (kernel) | 48.8 µs | **25.3 µs** (1.93×) |
| SM% | 61.5 | 52.6 |
| DRAM% | 33.3 | **65.6** |
| 명령수 | 5.47 M | 5.33 M |
| regs/thread | 48 | 56 |

커널이 **SM-bound(61% SM / 33% DRAM)에서 BW-bound(53% SM / 66% DRAM)로 뒤집혔다** — lean 프로토타입이 닿은
~68% DRAM 정체(Q8)에 도달, 격차가 알고리즘이 아니라 컴파일타임에 해소 가능한 오버헤드였음을 확정. 작동 이유:
상수 `WBITS`/`U`/`GS`가 상수 마스크 + **상수폭 shift**(런타임 가변 shift 대신)를 주고, `UB/4` word-load를
`i<SB`/`i<UB>>2` 경계 분기 없이 완전 unroll하며, k-loop을 `#pragma unroll`하면 **rolling-buffer 스케줄이
정적 해소**되어 `ureg[uwi]`가 레지스터 상주(런타임-WBITS 커널은 불가: refill 주기가 데이터 의존). +8 regs(완전
unroll) 비용은 있으나 occupancy는 ~56% warps-active 유지. (W-only decode GEMV만 특수화; 배치/`wa` 경로는
아직 generic이라 같은 처리 여지 있음.)
