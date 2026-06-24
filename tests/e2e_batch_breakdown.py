"""E2E PREFILL vs DECODE breakdown vs batch at (L_in=1024, L_out=128), per-scope robust.
Shows where the time goes (prefill grows + dominates at batch) and where quant wins/loses
(decode=BW-bound win, prefill=compute-bound loss). Reuses harness_batchsweep.
Usage: MS_FAST=1 python tests/e2e_batch_breakdown.py [--Bs 1,4,8,16,32] [--reps 10]"""
import sys, os, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import harness_batchsweep as H
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--reps", type=int, default=10)
ap.add_argument("--Bs", default="1,4,8,16,32")
a = ap.parse_args()
BS = [int(x) for x in a.Bs.split(",")]
LIN, LOUT = 1024, 128

print(f"PREFILL/DECODE breakdown vs batch  {torch.cuda.get_device_name(0)}  Llama-3.1-8B {H.Cfg.layers}L  "
      f"L_in={LIN} L_out={LOUT} | per-scope robust | ms; ratio <1 = MSAQ faster than bf16")

for scn, wstyle, kvq in H.SCENARIOS:
    cu, cg = H.PERSCOPE_CFG[scn]
    print(f"\n=== {scn}  (MSAQ u{cu}/gs{cg}) ===")
    print(f"{'B':>4} | {'PREFILL bf/mx/mq':>24} | {'DECODE bf/mx/mq':>24} | "
          f"{'pre%':>5}{'  pre':>7}{'  dec':>7}{' tot':>6}  (mq/bf)")
    for B in BS:
        m = {}
        for fmt, u, gs in H.variants(cu, cg):
            w, kv = H.fmt_paths(fmt, wstyle, kvq)
            r = H.spawn(a, w, kv, u, gs, B, LIN, LOUT, f"{scn}/{fmt}/B{B}")
            m[fmt] = H._metrics(r, LOUT)        # (prefill, decode, total) or None
        bf, mx, mq = m["bf16"], m["mxint8"], m["msaq"]
        if None in (bf, mx, mq):
            print(f"{B:>4} | {'OOM':>24} | {'OOM':>24} |"); continue
        pre = lambda t: f"{t[0]:.0f}"
        dec = lambda t: f"{t[1]:.0f}"
        pre_share = 100.0 * mq[0] / mq[2]                # prefill % of MSAQ total
        r_pre, r_dec, r_tot = mq[0]/bf[0], mq[1]/bf[1], mq[2]/bf[2]
        print(f"{B:>4} | {pre(bf)+'/'+pre(mx)+'/'+pre(mq):>24} | "
              f"{dec(bf)+'/'+dec(mx)+'/'+dec(mq):>24} | "
              f"{pre_share:>4.0f}%{r_pre:>7.2f}{r_dec:>7.2f}{r_tot:>6.2f}", flush=True)
print("\npre%=prefill share of MSAQ total.  pre/dec/tot = MSAQ/bf16 ratio for prefill, decode, total.")
print("decode mq/bf<1 = quant win (BW-bound);  prefill mq/bf>1 = quant loss (compute-bound vs cuBLAS).")
