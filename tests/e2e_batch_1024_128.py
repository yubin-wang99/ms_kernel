"""E2E latency vs BATCH at (L_in=1024, L_out=128), per-scope robust (u,gs) — current kernels.
Sweeps B in {4,8,16,32,64}; bf16 / mxint8(wide) / msaq per scope. Reuses harness_batchsweep.
Usage: CUDA_VISIBLE_DEVICES=0 python tests/e2e_batch_1024_128.py [--reps 15]"""
import sys, os, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import harness_batchsweep as H
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--reps", type=int, default=15)
ap.add_argument("--Bs", default="4,8,16,32,64")
a = ap.parse_args()
BS = [int(x) for x in a.Bs.split(",")]
LIN, LOUT = 1024, 128

print(f"E2E latency vs batch  {torch.cuda.get_device_name(0)}  Llama-3.1-8B {H.Cfg.layers}L  "
      f"L_in={LIN} L_out={LOUT}  | per-scope robust (u=4 where robust) | total ms, ratios <1 = faster")

for scn, wstyle, kvq in H.SCENARIOS:
    cu, cg = H.PERSCOPE_CFG[scn]
    print(f"\n=== {scn}  (MSAQ u{cu}/gs{cg}) ===")
    print(f"{'B':>4} | {'bf16':>8}{'mxint8':>8}{'msaq':>8} (total ms) | "
          f"{'mq/bf':>6}{'mq/mx':>6}{'mx/bf':>6} | note")
    for B in BS:
        m = {}
        for fmt, u, gs in H.variants(cu, cg):
            w, kv = H.fmt_paths(fmt, wstyle, kvq)
            r = H.spawn(a, w, kv, u, gs, B, LIN, LOUT, f"{scn}/{fmt}/B{B}")
            mm = H._metrics(r, LOUT)
            m[fmt] = (mm[2] if mm else None)        # total ms (None = OOM/err)
        bf, mx, mq = m["bf16"], m["mxint8"], m["msaq"]
        cell = lambda v: ("OOM" if v is None else f"{v:.0f}")
        rat = lambda n, d: ("—" if (n is None or d is None) else f"{n/d:.2f}")
        note = "OOM" if None in (bf, mx, mq) else ""
        print(f"{B:>4} | {cell(bf):>8}{cell(mx):>8}{cell(mq):>8}             | "
              f"{rat(mq,bf):>6}{rat(mq,mx):>6}{rat(mx,bf):>6} | {note}", flush=True)
print("\n(mq=MSAQ  mx=MXINT8  bf=bf16.  mq/bf,mx/bf<1 = beats bf16;  mq/mx<1 = MSAQ beats MXINT8)")
