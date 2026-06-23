# Kernel ver.3 — (u,gs) 컴파일타임 특수화 패스 정리

ver.2의 7종 커널 위에 올린 **단일 최적화 레이어**: decode 언팩 커널을 `(u, gs)`로 **컴파일타임
특수화**해 1.3–1.9× 가속. 무엇을 왜 특수화했고, 어디서 효과가 났고(decode), 어디서 안 났는지
(prefill, 이유와 함께)를 한곳에 모았다. 개념의 "왜"는 [compile_time_optimization.md], 측정 전 과정은
[packing_explained.md] §7–§13.

> **ver.2 → ver.3 한 줄.** 커널 골격·포맷·설계 규칙은 그대로다. 추가된 것은 **언팩의 컴파일타임
> 특수화** 하나 — 런타임 `u/gs`를 템플릿 상수로 바꿔 가변 shift→상수 shift, 동적 배열 인덱싱→레지스터
> 상주로 바꾼 것. decode 4경로 전부 적용(landed), prefill은 **load-bound로 진단되어 효과 0 → 미적용
> (documented negative)**.

---

## 0. 한 줄 메커니즘

generic 커널은 `u, gs, wbits=8-u, UB, SB`를 **런타임 인자**로 받는다. 그러면 nvcc는 (a) 마스크/shift
거리를 상수로 못 박고, (b) rolling 비트버퍼의 refill 시점이 데이터 의존이라 언팩 배열 `ureg[uwi]`를
**local memory로 spill**한다(GPU는 런타임 변수로 레지스터를 인덱싱 못 함). `(u,gs)`를 템플릿 상수로
박고 k-loop을 `#pragma unroll`하면 그 스케줄이 정적 해소돼 **`ureg`가 레지스터 상주**, 마스크·shift가
상수가 된다. 비트 출력은 동일(검증 `max|spec−generic|=0`).

핵심: **컴파일타임 상수 → 언팩이 레지스터·상수연산으로 떨어짐.** 단, 이 이득은 커널이 **언팩 연산/
local-mem에 bound일 때만** 시간으로 전환된다(§3 참조).

## 1. Decode 경로 — 전부 적용 (landed)

`csrc/w_gemv.cu`. 핫 조합(u∈{2,3} × gs∈{8,16}, 배치는 × 모든 MR)을 특수화 커널로 dispatch,
나머지는 generic fallback. `MS_GEMV_NOSPEC=1`로 generic 강제(A/B용). 공유 헬퍼
`ms_stream_block_uspec<U_,GS_,F>`(원소별 콜백, `--expt-extended-lambda`)로 4커널이 언팩을 재사용.

| 경로 | 커널 | generic | **spec** | speedup |
|---|---|---|---|---|
| W-only GEMV (M=1) | `wonly_gemv_wide_uspec` | 45.1 µs | **25.7 µs** | **1.76×** (u3 1.86×) |
| W-only batched (M=8) | `wonly_gemv_batched_uspec` | 134.5 µs | **82.0 µs** | **1.64×** |
| W+A GEMV (M=1) | `wa_gemv_wide_uspec` | 43.7 µs | **27.4 µs** | **1.59×** |
| W+A batched (M=8) | `wa_gemv_batched_uspec` | 125.0 µs | **93.5 µs** | **1.34×** |

(RTX 3090, OUT=K=4096, u2/gs8 기준.) 커널 단독 ncu(W-only): 48.8→25.3 µs, **SM 61%→53% / DRAM
33%→66%로 SM-bound→BW-bound 뒤집힘**, 명령수 5.47M→5.33M. W+A도 동일: 46.2→25.3 µs, DRAM 33→69%,
명령수 11.4M→5.4M(절반). 비트 동일, `test_w.py`+`test_wa.py` 통과. 비용 +8 regs(완전 unroll),
occupancy ~56% 유지.

## 2. Prefill 경로 — 적용 시도 → 효과 0 → 미적용 (documented negative)

`csrc/wa_gemm.cu`의 `wonly_gemm_tiled` / `wa_gemm_tiled` / `wa_imma`도 같은 패턴으로 특수화해보았다
(`if constexpr (U_>0)` 단일 바디, generic=`<-1,-1,…>`, spec=`<2,8,…>`; random-access 언팩의 특수화
헬퍼 `unpack_ms_weight_elem_uspec<U_,GS_>`). **비트 동일**이었으나 **속도 변화 0**(1.00×):

| 경로 | M | generic | spec | speedup |
|---|---|---|---|---|
| `wonly_gemm` | 256 | 1345 µs | 1342 µs | 1.00× |
| `wa_gemm` | 256 | 1503 µs | 1502 µs | 1.00× |
| `wonly_gemm` | 64 | 745 µs | 746 µs | 1.00× |

**왜 안 되나 (진단).** ncu로 spec(`wonly_gemm_tiled<2,8,128,128,8,8>`)과 generic(`<-1,-1,…>`)이
실제로 다른 커널임을 확인했는데도 **시간·명령수·L1·sector-util이 전부 동일**:

| | generic | spec |
|---|---|---|
| 시간 | 1.898 ms | 1.898 ms |
| **L1TEX %** | **72.8** | **72.8** |
| sector-util % (global ld) | 11.8 | 11.8 |
| SM % | 28.1 | 28.1 |

prefill tiled GEMM은 **L1TEX(로드 파이프) bound**다 — `extract_code`가 원소마다 `upper[(…)*OUT+o]`를
**uncoalesced 글로벌 바이트 로드**(sector-util 11.8% = sector의 ~1/8만 유효)로 읽고, 이게 한계다.
특수화는 언팩의 **연산**(shift/mask)만 싸게 하는데, 그 연산은 로드 병목 뒤에 완전히 숨어 시간에 안
나타난다. (decode는 컬럼 전체를 wide-load(coalesced)한 뒤 언팩 연산/local-mem이 한계라 효과가 났다 —
정반대 regime.)

**결론:** prefill 특수화는 **+1.6 MB 바이너리·컴파일시간 증가에 0 이득** → CLAUDE.md(단순성) 원칙대로
**되돌렸다**. prefill의 진짜 레버는 연산 특수화가 아니라 **언팩 로드를 coalesce**하는 것 —
column-major wide-load(현재 prefill은 row-major `[nb,UB,OUT]` 평면을 원소별로 읽음)로 바꾸는 별도
재작성이 필요하다(향후 과제).

## 3. 일반 교훈 — 특수화가 먹히는 조건

`(u,gs)` 컴파일타임 특수화는 **언팩의 연산/local-mem이 임계경로일 때만** 시간 이득이 된다.

| regime | 예 | 특수화 효과 |
|---|---|---|
| wide-load + 언팩-ALU/local-mem bound | decode GEMV/배치/W+A | **1.3–1.9× (적용)** |
| per-element uncoalesced-load bound | prefill tiled/imma | **0 (미적용)** |

즉 "런타임 인자를 상수로"는 만능이 아니라, **병목이 그 상수가 줄이는 비용(연산)에 있을 때** 작동한다.
prefill처럼 병목이 메모리 로드 패턴이면, 먼저 로드를 coalesce해야 한다.

## 4. 상태 / 파일

- **Landed**(branch `perf/gemv-ugs-specialization`, 커밋 2개): `csrc/w_gemv.cu`(decode 4커널 +
  `ms_stream_block_uspec`/`wonly_gemv_wide_uspec`), `setup.py`(`--expt-extended-lambda`).
- **Reverted**: `csrc/wa_gemm.cu`, `csrc/core/ms_utils.cuh`(prefill 특수화 — negative).
- **문서**: [compile_time_optimization.md](EN+KO, 메커니즘), [packing_explained.md] §7–§13 /
  [packing_explained_korean.md] Q5–Q12(전 과정), 본 파일(요약).
- **검증**: 전 경로 `max|spec−generic|=0`; `tests/test_w.py`+`tests/test_wa.py` 통과.
- **남은 후과제**: (a) prefill을 wide-load coalesced 언팩으로 재작성(연산이 아니라 로드가 레버),
  (b) WMMA/IMMA override 경로(`MS_TILE_CFG` 10/11)는 tensor-core라 언팩이 이미 숨음 — 특수화 무의미.
