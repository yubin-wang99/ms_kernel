#!/usr/bin/env python3
"""Batch sweep at (L_in=1024, L_out=128) reusing the harness_batchsweep deployed dispatch
(per-scope robust u/gs; B=1 wide-GEMV, 2-15 batched-GEMV, >=16 dequant+cuBLAS; prefill dequant+cuBLAS).
Each (scope,fmt,B) is a worker subprocess (random Llama-3.1-8B weights). Prints prefill/decode/total
ratios mq/mx, mq/bf, mx/bf. Run: CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python tests/caveat_e2e.py
"""
import os, sys, types
sys.path.insert(0, "."); sys.path.insert(0, "tests")
import torch
import harness_batchsweep as H

a = types.SimpleNamespace(reps=int(os.environ.get("REPS", "30")))
BATCHES = [1, 8, 32, 64, 128, 256]
LIN, LOUT = 1024, 128

print(f"[caveat_e2e] {torch.cuda.get_device_name(0)} | L_in={LIN} L_out={LOUT} | per-scope robust u/gs | "
      f"ratios <1=faster (mq=msaq, mx=mxint8, bf=bf16)", flush=True)
for scn, wstyle, kvq in H.SCENARIOS:
    cu, cg = H.PERSCOPE_CFG[scn]
    print(f"\n--- {scn} (MSAQ u{cu}/gs{cg}) ---", flush=True)
    print(H._hdr("B"), flush=True)
    for B in BATCHES:
        row = {}
        for f, u, gs in H.variants(cu, cg):
            w, kv = H.fmt_paths(f, wstyle, kvq)
            row[f] = H.spawn(a, w, kv, u, gs, B, LIN, LOUT, f"{scn}/{f}")
        H._print_row(str(B), H._ratiorow(row["bf16"], row["mxint8"], row["msaq"], LOUT))
print("\n[caveat_e2e] done", flush=True)
