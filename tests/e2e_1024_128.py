"""E2E latency at (L_in=1024, L_out=128), B=1, per-scope robust (u,gs) — current kernels.
Reuses harness_batchsweep's worker/Model machinery. S1-S5; bf16 / mxint8 / msaq per scope.
Usage: CUDA_VISIBLE_DEVICES=0 python tests/e2e_1024_128.py [--reps 20]"""
import sys, os, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import harness_batchsweep as H

ap = argparse.ArgumentParser()
ap.add_argument("--reps", type=int, default=20)
ap.add_argument("--B", type=int, default=1)
a = ap.parse_args()
B, LIN, LOUT = a.B, 1024, 128

import torch
print(f"E2E latency  {torch.cuda.get_device_name(0)}  Llama-3.1-8B {H.Cfg.layers}L  "
      f"B={B}  L_in={LIN}  L_out={LOUT}  | per-scope robust (u=4 where accuracy-robust)")
print(f"  prefill = TTFT(1024 tok);  decode = integrated over {LOUT} steps;  total = prefill+decode (ms)\n")
hdr = (f"{'scope':<14}{'cfg':>9} | {'prefill (bf/mx/mq)':>26} | {'decode (bf/mx/mq)':>26} | "
       f"{'total (bf/mx/mq)':>26} | {'mq/bf':>6}{'mq/mx':>7}")
print(hdr); print("-" * len(hdr))

rows = []
for scn, wstyle, kvq in H.SCENARIOS:
    cu, cg = H.PERSCOPE_CFG[scn]
    m = {}
    for fmt, u, gs in H.variants(cu, cg):
        w, kv = H.fmt_paths(fmt, wstyle, kvq)
        r = H.spawn(a, w, kv, u, gs, B, LIN, LOUT, f"{scn}/{fmt}")
        m[fmt] = H._metrics(r, LOUT)       # (prefill, decode, total) ms or None
    bf, mx, mq = m["bf16"], m["mxint8"], m["msaq"]
    def tri(i):
        f = lambda v: ("OOM/—" if v is None else f"{v[i]:.1f}")
        return f"{f(bf):>8}{f(mx):>9}{f(mq):>9}"
    r_mqbf = f"{mq[2]/bf[2]:.2f}" if (mq and bf) else "—"
    r_mqmx = f"{mq[2]/mx[2]:.2f}" if (mq and mx) else "—"
    print(f"{scn:<14}{('u%dg%d'%(cu,cg)):>9} | {tri(0):>26} | {tri(1):>26} | {tri(2):>26} | "
          f"{r_mqbf:>6}{r_mqmx:>7}", flush=True)
    rows.append((scn, cu, cg, bf, mx, mq))

print("\n(ratios <1 = MSAQ faster.  mq=MSAQ  mx=MXINT8  bf=bf16)")
