# 방안 4 설계 — vectorized load + register-aligned repack (KV decode 중심)

> 목표: occupancy(방안1)를 채운 뒤 드러난 **memory access 병목**을 제거해
> MSAQ의 "적게 읽는 바이트"가 비로소 "시간 우위"로 바뀌게 한다.
> 즉 KV decode에서 **MSAQ/MXINT8 < 1**(L2 잔류가 아닌 진짜 transaction 기준).

---

## 0. 먼저: 병목은 두 개로 분리된다

측정된 "uncoalesced strided load"는 사실 **독립된 두 문제**다. 난이도·재인증 비용이
달라 반드시 분리해서 단계화해야 한다.

| | Concern 1 — **coalescing** | Concern 2 — **extraction / load width** |
|---|---|---|
| 증상 | warp 32 thread가 L-stride 주소 → 트랜잭션 ~32배 | code마다 1~2 byte load + straddle shift/or |
| 비용 원천 | DRAM/L2 트랜잭션 수 (메모리) | LSU instruction 수 + ALU (연산) |
| 측정상 비중 | **지배적** (achieved BW ≤5%의 주범) | 부차적 |
| 해결 | plane 축 순서만 transpose | bit-level repack + bfe + 128-bit load |
| pack 변경 | **축 순서만** (bit-packing 그대로) | **bit 재배치** (padding 발생) |
| 재인증 | 쉬움 (코드 bit 동일, 주소만 변경) | 무거움 (roundtrip 전면 재인증) |

→ **Concern 1을 먼저(저위험·고효과), 측정 후 Concern 2 판단**. 프로젝트의
"측정하고 결정" 원칙 및 헤더가 명시한 단계화(`MSAQ_USE_BFE`는 repack 후에만 유효)와 일치.

### 적용 범위 주의 — 이건 KV 전용 문제
- **GEMV**: weight plane이 `[nb, BYTES, OUT]`로 **OUT(=출력열)이 innermost**.
  thread가 OUT열에 매핑 → 인접 thread가 인접 byte를 읽어 **이미 coalesced**
  (`w_gemv.cu` 헤더도 명시). 따라서 Concern 1은 **GEMV엔 불필요**.
- **KV**: plane이 `[H, nb, BYTES, L]`로 **L(token)이 innermost**인데 thread는
  head_dim(=BYTES축)에 매핑 → 고정 key에서 BYTES축을 가로질러 읽어 **L-stride**.
  Concern 1의 타깃은 **KV뿐**.
- Concern 2(bfe/128-bit load)는 GEMV·KV 공통으로 적용 가능.

---

## 1. 근본 원인 (왜 KV가 strided인가)

현재 KV plane: `[H, nb, BYTES, L]`, L innermost.
커널: thread `e` = head_dim 원소. 한 warp = 한 nb-block의 k=0..31.
inner loop는 key `j`를 순회하고, **고정 j에서** 32 thread가 서로 다른 byteIdx를 읽음:

```
addr = base + byteIdx * L + j      // byteIdx = (k*wbits)>>3, j 고정
```

→ warp 내 주소가 **L(=Lk)만큼 떨어진 열(column) 접근** = 완전 uncoalesced.
(반대로 inner loop의 연속 j는 stride-1 이지만, 한 번에 한 j만 쓰므로 활용 못 함.)

---

## 2. Stage 4a — token-major transpose (Concern 1, 권장 우선)

### 핵심 아이디어
plane을 **BYTES를 innermost로** 뒤집는다: `[H, nb, BYTES, L]` → `[H, nv, L, BYTES]`.
그러면 고정 `(h,nb,j)`에서 그 블록의 upper/shared 바이트가 **연속**:

```
addr = base + j * BYTES + byteIdx   // 같은 j, byteIdx = 0..(UB-1)
```

한 warp의 32 thread가 읽는 주소 = `base + j*BYTES + [0..~19]` → **연속 ~20 byte span**
= 한 캐시라인/한두 트랜잭션으로 coalesced. **bit-packing은 그대로**, extraction(shift/or)도
그대로. 바뀌는 건 plane 축 순서와 커널의 stride 식뿐.

### pack.py 변경 (작음)
- `pack_kv`는 지금 head별 `pack_weight`(out-innermost) 결과 `[nb,UB,L]`를 stack한다.
  KV 전용으로 마지막에 `[nb,UB,L] → [nb,L,UB]` (및 shared `[nb,SB,L]→[nb,L,SB]`)
  **transpose**만 추가. `_per`(oracle용 per-head dict)와 scale_exp는 불변.
- MXINT8 baseline(`pack_kv_mxint8`)의 `qweight [H,nb,32,L]`도 `[H,nb,L,32]`로 동일 transpose
  (matched 비교 유지).

### 커널 변경 (`kv_attention.cu`, `mxint8.cu`)
- `ms::unpack_ms_kv_elem`의 주소식을 `byteIdx*L + key` → `key*BYTES + byteIdx`로.
  (헤더에 KV 전용 변형 추가 또는 인자로 stride 전달.)
- thread→head_dim 매핑, online-softmax, split-KV(방안1) 구조는 **그대로**.

### 재인증 (쉬움)
- code bit가 동일하므로 `tests/test_kv.py`(oracle gate)와 emulation gate가 그대로 통과해야 함.
- pack↔unpack roundtrip은 transpose만 검증하는 1줄 assert 추가로 충분.

### 예상 효과
- 트랜잭션 ~32× → ~2×. KV achieved BW가 ≤5% → 수십 %로. 큰 Lk에서 무너지던 것도 완화.
- **MSAQ/MXINT8**: MSAQ가 더 적은 연속 바이트를 읽으므로, L2 잔류가 아닌 **트랜잭션
  기준에서 1.0 밑으로 내려갈 1차 후보**. (여기서 멈춰 측정 후 4b 판단.)

### 남은 throttle (정직한 한계)
- inner loop에 **key마다 `__syncthreads()` 블록 reduction**이 남아 load overlap을 막음
  (= 방안 3). Stage 4a의 coalescing 이득을 일부 깎을 수 있음. 완전한 포화는 QK reduction을
  key-tile 배치(여러 key score를 모아 계산)로 재작성해야 — 4b/방안3과 함께.

---

## 3. Stage 4b — register-aligned repack + bfe + 128-bit load (Concern 2)

Stage 4a 측정 후 **LSU instruction / ALU가 병목으로 남으면** 진행.

### 3-1. register-aligned packing
모든 code가 **32-bit word 경계를 넘지 않도록** 패킹 → `bfe.s32(word, pos, len)` 단일
명령 추출 + `int4`(=4×32bit, 16B) vectorized load 가능. straddle 2-byte load 제거.

padding tradeoff (upper 기준, 32-block):

| u | wbits | codes/word | words | aligned B | dense UB | overhead |
|---|------|-----------|-------|-----------|----------|----------|
| 2 | 6 | 5 | 7 | 28 | 24 | +16.7% |
| 3 | 5 | 6 | 6 | 24 | 20 | +20% |
| 4 | 4 | 8 | 4 | 16 | 16 | **+0%** |

- u=4는 무손실 정렬(가장 적합). u=2/3은 upper에 ~17~20% bit overhead.
- **결정점**: shared(n_group개 u-bit, 매우 작음)는 정렬하지 말고 dense shift/or 유지 권장
  (정렬 시 비용 2배인데 이득 미미). upper만 정렬.
- u3gs8 정렬 후 유효비트 ≈ upper 6.0 + shared 0.375 + scale 0.25 = **6.625 b/elem**
  vs MXINT8 8.25 → 여전히 **0.80×** (정렬 padding 후에도 MSAQ가 덜 읽음 → 시간 우위 유지 가능).

### 3-2. 커널 (warp-cooperative vectorized load)
Stage 4a로 바이트가 연속이므로, 한 key(또는 key-tile)의 정렬된 words를 warp가
`int4`로 coalesced load → shared/`__shfl`로 분배 → 각 thread가 자기 code를 `bfe_s32`로 추출.
헤더의 `MSAQ_USE_BFE` 분기와 `bfe_s32()`가 이미 scaffold로 존재 → 구현은 기계적.

### 3-3. 재인증 (무거움 — 반드시 선행)
헤더 주석의 절차 그대로:
1. `ms_lib/pack.py`를 register-aligned code로 재배치 (`_pack_codes_lsb` 옆에 word-aligned 변형).
2. pack↔unpack **roundtrip 재인증** (새 layout 전체) — `dequant_weight`/`weight_int8` bit-exact 유지.
3. `MSAQ_USE_BFE=1`로 flip, bfe 변형 구현, oracle+emulation gate 재통과.
- W-only/W+A(GEMV·GEMM)도 같은 plane을 쓰므로 **함께 재인증** (회귀 위험 가장 큼 → 마지막에).

---

## 4. 단계 순서 / 게이트

```
[4a] KV transpose (coalescing)         ── 저위험, 큰 효과
   └ gate: test_kv + emulation 통과, KV BW 측정
      └ MSAQ/MXINT8 (transaction 기준) 1.0 밑이면 → 여기서 일단 목표 달성
      └ 아직 LSU/ALU 병목이면 ↓
[4b] upper word-align + bfe + int4 load ── 고위험, 전면 재인증
   └ gate: pack roundtrip 재인증 → oracle/emulation → 전 커널 회귀
[옵션] QK reduction을 key-tile 배치로 재작성 (방안3 동반) ── 완전 포화
```

## 5. 미해결 결정점 (구현 전 확인 필요)
1. **4a에서 멈출지**: transaction 기준 MSAQ<MXINT8가 나오면 4b의 재인증 리스크를 감수할지.
2. **u별 padding**: u=2/3의 ~17~20% upper overhead 수용 여부 (정확도 설정이 u3 고정이면 감수).
3. **shared 정렬 여부**: dense 유지(권장) vs 정렬.
4. **QK reduction 재작성 범위**: 4a 후 barrier throttle이 크면 방안3을 같이 할지.
5. **MXINT8 baseline**: 4a transpose는 matched로 적용(공정). 4b의 bfe는 MXINT8엔 무의미
   (직접 int8) → MXINT8엔 transpose+int4 load만 적용해 공정성 유지.
