#!/usr/bin/env python3
"""Phase 0 vLLM driver — reproduce "low-bit KV -> more GPU blocks -> more concurrent
seqs -> higher throughput" in the REAL vLLM allocator, with NO custom kernels.

This is the vLLM analogue of capacity_maxbatch.py: same mechanism, but the block count
comes from vLLM's own determine_num_available_blocks (gpu_memory_utilization profiling),
not our hand-rolled footprint formula. It validates the capacity argument on the gold-
standard serving stack using only vLLM's NATIVE fp8 KV cache (our kernels come in Phase 2).

Mechanism (vLLM):  num_gpu_blocks = (gpu_mem*util - weights - act_peak) / (block_size * kv_bytes/tok)
KV bytes/token halves from fp16->fp8  ->  ~2x blocks  ->  ~2x KV-token pool  ->  more concurrent seqs.

Because the block pool is a single resource that sequences draw from by length, num_gpu_blocks
depends on (kv_dtype, gpu_mem_util, model) but NOT on the per-request context. So we init the
engine ONCE per kv_dtype (max_model_len = max ctx) and derive max_concurrent_seqs(L) = pool/L for
every L in the sweep. -> 2 engine inits total (auto, fp8), not 2 x N.

  python vllm_phase0_capacity.py                              # capacity table (auto vs fp8)
  python vllm_phase0_capacity.py --throughput                 # + offline generate() tok/s
  python vllm_phase0_capacity.py --model unsloth/Meta-Llama-3.1-8B --kv-dtypes auto,fp8

Needs vLLM installed (`pip install vllm`). NOTE: RTX PRO 4000 is Blackwell sm_120 — a recent
vLLM/torch build may be required; if init fails on the arch, that's an install issue not a bug here.
"""
import argparse, time, sys

def _try_import_vllm():
    try:
        from vllm import LLM, SamplingParams
        return LLM, SamplingParams
    except Exception as e:
        sys.exit(f"vLLM not importable ({e}).\nInstall it first: pip install vllm  "
                 f"(Blackwell sm_120 may need a recent/nightly build).")

def get_num_gpu_blocks(llm):
    """Read vLLM's profiled GPU block count + block size across engine-internal layouts.
    V1 (>=0.8, incl 0.23): llm_engine.vllm_config.cache_config. Older: llm_engine.cache_config."""
    eng = getattr(llm, "llm_engine", llm)
    for path in ("vllm_config.cache_config", "cache_config",
                 "model_executor.cache_config", "scheduler.cache_config"):
        obj = eng
        try:
            for p in path.split("."): obj = getattr(obj, p)
            nb = getattr(obj, "num_gpu_blocks", None); bs = getattr(obj, "block_size", None)
            if nb: return nb, bs
        except AttributeError:
            continue
    return None, None

def build_engine(LLM, model, kv_dtype, max_model_len, util):
    """One engine init for a given KV dtype. kv_dtype='auto' = no KV quant (bf16/fp16)."""
    kwargs = dict(model=model, tensor_parallel_size=1, gpu_memory_utilization=util,
                  max_model_len=max_model_len, enforce_eager=True, disable_log_stats=True)
    if kv_dtype != "auto":
        kwargs["kv_cache_dtype"] = kv_dtype           # vLLM native fp8 (e4m3) KV cache
    return LLM(**kwargs)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="NousResearch/Meta-Llama-3.1-8B")
    ap.add_argument("--kv-dtypes", default="auto,fp8", help="comma list; auto=fp16/bf16 baseline")
    ap.add_argument("--ctx", default="1024,4096,16384,32768,65536", help="context lengths to derive seqs")
    ap.add_argument("--gpu-mem-util", type=float, default=0.90)
    ap.add_argument("--max-model-len", type=int, default=0, help="engine max_model_len (0 = max of --ctx)")
    ap.add_argument("--throughput", action="store_true", help="also run offline generate() tok/s")
    ap.add_argument("--tput-ctx", type=int, default=1024, help="prompt context for the throughput run")
    ap.add_argument("--tput-prompts", type=int, default=512, help="num prompts (>> capacity to saturate)")
    ap.add_argument("--lout", type=int, default=128, help="output tokens per request (throughput run)")
    a = ap.parse_args()
    LLM, SamplingParams = _try_import_vllm()

    dtypes = [d.strip() for d in a.kv_dtypes.split(",")]
    Ls = [int(x) for x in a.ctx.split(",")]
    mml = a.max_model_len or max(Ls + [a.tput_ctx + a.lout])

    print(f"# vLLM Phase 0 capacity — {a.model}, gpu_mem_util={a.gpu_mem_util}, max_model_len={mml}")
    print(f"# KV pool = num_gpu_blocks x block_size (tokens). max concurrent seqs at ctx L ~= pool / L.\n")

    # ---- capacity probe: one init per kv_dtype, read the block pool ----
    cap = {}   # kv_dtype -> (num_blocks, block_size, kv_tokens) or None on OOM
    for d in dtypes:
        try:
            t0 = time.time()
            llm = build_engine(LLM, a.model, d, mml, a.gpu_mem_util)
            nb, bs = get_num_gpu_blocks(llm)
            del llm
            import torch, gc; gc.collect(); torch.cuda.empty_cache()
            if nb is None:
                print(f"# [{d:>4}] could not read num_gpu_blocks (vLLM layout changed) — see get_num_gpu_blocks()")
                cap[d] = None; continue
            cap[d] = (nb, bs, nb * bs)
            print(f"# [{d:>4}] num_gpu_blocks={nb}  block_size={bs}  KV pool={nb*bs:,} tokens  "
                  f"(init {time.time()-t0:.0f}s)")
        except Exception as e:                          # OOM / arch / load failure
            print(f"# [{d:>4}] engine init FAILED: {type(e).__name__}: {str(e)[:120]}")
            cap[d] = None

    base = dtypes[0]
    print(f"\n{'kv_dtype':>10} {'KV pool(tok)':>13} | " + " ".join(f"seqs@{L//1024}k" if L>=1024 else f"seqs@{L}" for L in Ls))
    for d in dtypes:
        if cap.get(d) is None:
            print(f"{d:>10} {'OOM/none':>13} | " + " ".join(f"{'-':>7}" for _ in Ls)); continue
        nb, bs, pool = cap[d]
        seqs = [pool // L for L in Ls]                  # max concurrent seqs that fit the KV pool at ctx L
        print(f"{d:>10} {pool:>13,} | " + " ".join(f"{s:>7,}" for s in seqs))

    if cap.get(base):
        bpool = cap[base][2]
        print(f"\n## KV pool ratio vs {base} (>1 = more concurrent seqs at same accuracy & HBM)")
        for d in dtypes:
            if cap.get(d) is None:
                print(f"{d:>10} | servable-vs-{base}-OOM or n/a"); continue
            print(f"{d:>10} | {cap[d][2]/bpool:>4.2f}x  (KV pool {cap[d][2]:,} vs {bpool:,} tokens)")
        # the long-context binary edge: where the baseline pool can't hold even ONE seq but fp8 can
        print(f"\n## Longest single-seq context the KV pool holds (fp16 OOMs first as ctx grows)")
        for d in dtypes:
            if cap.get(d) is None: print(f"{d:>10} | n/a"); continue
            print(f"{d:>10} | {cap[d][2]:>9,} tokens  (one seq up to this ctx; batch>1 divides this)")

    # ---- optional: offline throughput (does fp8's bigger pool convert to tok/s?) ----
    if a.throughput:
        print(f"\n## Offline throughput — generate() {a.tput_prompts} prompts @ ctx {a.tput_ctx}, L_out={a.lout}")
        print(f"{'kv_dtype':>10} {'wall(s)':>8} {'out tok/s':>10} | vs {base}")
        prompt = "The quick brown fox " * (a.tput_ctx // 4)   # ~tput_ctx tokens (rough)
        prompts = [prompt] * a.tput_prompts
        sp = SamplingParams(max_tokens=a.lout, temperature=0.0, ignore_eos=True)
        bt = None
        for d in dtypes:
            try:
                llm = build_engine(LLM, a.model, d, mml, a.gpu_mem_util)
                t0 = time.time(); outs = llm.generate(prompts, sp); wall = time.time() - t0
                out_toks = sum(len(o.outputs[0].token_ids) for o in outs)
                tps = out_toks / wall
                if bt is None: bt = tps
                print(f"{d:>10} {wall:>8.1f} {tps:>10.0f} | {tps/bt:>4.2f}x")
                del llm
                import torch, gc; gc.collect(); torch.cuda.empty_cache()
            except Exception as e:
                print(f"{d:>10} | FAILED: {type(e).__name__}: {str(e)[:100]}")

    print(f"\n# READ: fp8 KV halves bytes/token -> ~2x KV pool -> ~2x concurrent seqs at fixed HBM/accuracy.")
    print(f"# This reproduces our capacity argument NATIVELY in vLLM (no custom kernels). Phase 2 pushes")
    print(f"# KV below fp8 (E2M3 6.25b / u4 4.5b) for a pool BEYOND fp8 — the 'better than fp8' contribution.")

if __name__ == "__main__":
    main()
