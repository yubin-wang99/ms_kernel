#!/usr/bin/env python3
"""Phase 0 (online) — RPS-vs-P99 Pareto from vLLM's own serving benchmark, NO custom kernels.

Capacity probe (vllm_phase0_capacity.py) showed fp8 KV -> 2.00x num_gpu_blocks. This driver shows
that capacity converting to SERVING throughput: for each kv_cache_dtype it boots a vLLM OpenAI
server, sweeps the offered request-rate, and runs `vllm bench serve` (random fixed-length dataset)
to record achieved RPS + P99 TTFT/TPOT/E2E. The headline figure is the Pareto: at a fixed P99-latency
SLO, the bigger KV pool (fp8) sustains a higher request rate before the queue blows up.

Why random fixed-length: on a 24GB card the 8B weights (~16GB) leave a SMALL KV pool, so KV capacity
is the binding constraint. Fixed input=2048/output=256 makes the per-seq KV cost constant, so the only
thing that changes between auto and fp8 is how many sequences fit -> a clean capacity->RPS readout.
(ShareGPT length-distribution is the realistic follow-up; this is the controlled version.)

  .venv-vllm/bin/python vllm_phase0_serving.py                 # auto vs fp8, default sweep
  .venv-vllm/bin/python vllm_phase0_serving.py --rates 4,8,16,inf --in-len 4096 --out-len 256
"""
import argparse, json, os, subprocess, sys, time, urllib.request, signal

def wait_health(port, timeout=300):
    url = f"http://127.0.0.1:{port}/health"
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            with urllib.request.urlopen(url, timeout=5) as r:
                if r.status == 200: return True
        except Exception:
            time.sleep(3)
    return False

def start_server(vllm_bin, model, kv_dtype, port, max_model_len, util, log_path):
    cmd = [vllm_bin, "serve", model,
           "--port", str(port), "--max-model-len", str(max_model_len),
           "--gpu-memory-utilization", str(util), "--disable-log-stats"]
    if kv_dtype != "auto":
        cmd += ["--kv-cache-dtype", kv_dtype]
    log = open(log_path, "w")
    p = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT,
                         start_new_session=True)   # own process group -> clean kill
    return p, log

def stop_server(p, log):
    try:
        os.killpg(os.getpgid(p.pid), signal.SIGINT)   # graceful
        try: p.wait(timeout=30)
        except subprocess.TimeoutExpired:
            os.killpg(os.getpgid(p.pid), signal.SIGKILL); p.wait(timeout=15)
    except ProcessLookupError:
        pass
    finally:
        log.close()

def run_bench(vllm_bin, model, port, rate, in_len, out_len, num_prompts, seed, out_json):
    cmd = [vllm_bin, "bench", "serve",
           "--backend", "openai", "--endpoint", "/v1/completions",
           "--host", "127.0.0.1", "--port", str(port), "--model", model,
           "--dataset-name", "random", "--random-input-len", str(in_len),
           "--random-output-len", str(out_len), "--random-range-ratio", "0",
           "--num-prompts", str(num_prompts), "--request-rate", str(rate),
           "--ignore-eos", "--seed", str(seed),
           "--percentile-metrics", "ttft,tpot,e2el", "--metric-percentiles", "99",
           "--save-result", "--result-filename", out_json]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if not os.path.exists(out_json):
        sys.stderr.write(f"[bench rate={rate}] no result json. tail:\n{r.stdout[-800:]}\n{r.stderr[-400:]}\n")
        return None
    with open(out_json) as f:
        return json.load(f)

def g(d, *keys):
    for k in keys:
        if k in d and d[k] is not None: return d[k]
    return float("nan")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="NousResearch/Meta-Llama-3.1-8B")
    ap.add_argument("--kv-dtypes", default="auto,fp8")
    ap.add_argument("--rates", default="1,1.5,2,2.5,inf", help="offered req/s sweep ('inf'=all at once)")
    ap.add_argument("--in-len", type=int, default=2048)
    ap.add_argument("--out-len", type=int, default=256)
    ap.add_argument("--num-prompts", type=int, default=150)
    ap.add_argument("--util", type=float, default=0.90)
    ap.add_argument("--port", type=int, default=8013)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--venv", default=".venv-vllm/bin/python")
    ap.add_argument("--resdir", default="/tmp/vllm_pareto")
    a = ap.parse_args()
    vllm_bin = os.path.join(os.path.dirname(a.venv), "vllm")   # console script next to python
    os.makedirs(a.resdir, exist_ok=True)
    dtypes = [d.strip() for d in a.kv_dtypes.split(",")]
    rates = [r.strip() for r in a.rates.split(",")]
    mml = a.in_len + a.out_len + 64

    print(f"# vLLM online Pareto — {a.model}, random in={a.in_len}/out={a.out_len}, "
          f"{a.num_prompts} prompts/rate, util={a.util}")
    print(f"# server: vllm openai api_server per kv_dtype; client: vllm bench serve. ctx={a.in_len+a.out_len}\n")

    results = {d: {} for d in dtypes}
    for d in dtypes:
        log_path = os.path.join(a.resdir, f"server_{d}.log")
        print(f"## kv_dtype={d}: starting server (max_model_len={mml})...")
        p, log = start_server(vllm_bin, a.model, d, a.port, mml, a.util, log_path)
        try:
            if not wait_health(a.port):
                print(f"#   server [{d}] did not become healthy — see {log_path}"); continue
            print(f"#   healthy. sweeping rates {rates}")
            for rate in rates:
                rj = os.path.join(a.resdir, f"res_{d}_r{rate}.json")
                res = run_bench(vllm_bin, a.model, a.port, rate, a.in_len, a.out_len,
                                a.num_prompts, a.seed, rj)
                if res is None: continue
                results[d][rate] = res
                print(f"#   rate={rate:>4}: RPS={g(res,'request_throughput'):.2f} "
                      f"out_tok/s={g(res,'output_throughput'):.0f} "
                      f"P99_TTFT={g(res,'p99_ttft_ms'):.0f}ms P99_E2E={g(res,'p99_e2el_ms'):.0f}ms")
        finally:
            stop_server(p, log)
            time.sleep(5)   # let GPU memory free before next server

    # ---- Pareto table ----
    print(f"\n## RPS-vs-P99 Pareto (achieved RPS at each offered rate; lower P99 better)")
    hdr = f"{'kv_dtype':>9} {'offered':>7} {'RPS':>6} {'out tok/s':>9} {'P99 TTFT':>9} {'P99 TPOT':>9} {'P99 E2E':>9}"
    print(hdr)
    for d in dtypes:
        for rate in rates:
            r = results[d].get(rate)
            if not r: continue
            print(f"{d:>9} {rate:>7} {g(r,'request_throughput'):>6.2f} {g(r,'output_throughput'):>9.0f} "
                  f"{g(r,'p99_ttft_ms'):>8.0f}m {g(r,'p99_tpot_ms'):>8.1f}m {g(r,'p99_e2el_ms'):>8.0f}m")

    # ---- iso-SLO headline: max sustained RPS under a P99 TTFT budget ----
    #   TTFT (not E2E) is the capacity-sensitive axis: when the KV pool is full, new requests WAIT
    #   for admission and TTFT balloons. fp8's ~2x pool admits more -> holds TTFT at higher load.
    for slo in (2000, 5000, 10000):
        print(f"\n## Max sustained RPS @ P99 TTFT <= {slo}ms (SLA-constrained 'goodput')")
        base = None
        for d in dtypes:
            ok = [g(r,'request_throughput') for r in results[d].values()
                  if g(r,'p99_ttft_ms') <= slo]
            best = max(ok) if ok else 0.0
            if base is None: base = best
            ratio = f"{best/base:.2f}x" if base else "-"
            print(f"{d:>9} | max RPS under SLO = {best:>5.2f}  {ratio}")

    print(f"\n# READ: at a fixed P99-latency SLO, fp8's ~2x KV pool admits ~2x concurrent seqs -> it")
    print(f"# sustains a higher request rate before the queue saturates. That is the capacity->RPS")
    print(f"# conversion the fixed-batch microbench hides. Phase 2 (sub-byte KV) pushes the pool past fp8.")

if __name__ == "__main__":
    main()
