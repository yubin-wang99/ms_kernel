"""Per-scope E2E latency (S1-S6, incl. AA) at (L_in=1024, L_out=128), per-scope robust (u,gs),
current kernels (dequant+cuBLAS prefill, tensor-core/dequant batched decode). Writes
tests/harness_perscope_results_260625.md + .jsonl. Usage: MS_FAST=1 python tests/e2e_perscope2.py"""
import sys, os, json, argparse, datetime
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import harness_batchsweep as H
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--reps", type=int, default=12)
ap.add_argument("--Bs", default="1,8,32")
a = ap.parse_args()
BS = [int(x) for x in a.Bs.split(",")]
LIN, LOUT = 1024, 128
outdir = os.path.dirname(os.path.abspath(__file__))
jf = open(os.path.join(outdir, "harness_perscope_results_260625.jsonl"), "w")
lines = []
def emit(s): print(s, flush=True); lines.append(s)

emit(f"# Per-scope E2E latency v2 (S1-S6, incl. AA) — {torch.cuda.get_device_name(0)}")
emit(f"\nLlama-3.1-8B {H.Cfg.layers}L, L_in={LIN}, L_out={LOUT}, per-scope robust (u=4 where robust).")
emit("Current kernels: prefill = dequant-weight + cuBLAS (ties bf16); batched decode B in 2..15 = "
     "shared-activation wide GEMV (W-only 40us@M8; W+A dequant-in-staging float MAC 44us@M8 incl "
     "quant_act; both beat bf16 46us); B>=16 = dequant+cuBLAS; KV decode = (u,gs)-specialized "
     "wide-load. ms in µs; ratio <1 = MSAQ faster.")
emit("\n**S6 W+A+KV+AA** = full quant incl. attention activation×activation (Q,K,P,V). The decode "
     "attention kernel reads Q in bf16, so **AA decode latency == KV-decode (== S5)**; AA adds accuracy "
     "cost (~+0.9–1.0pp PPL, `precision/aa_attn_results.md`), not decode latency. Prefill attention is "
     "bf16 SDPA (AA-prefill loses to it). So S6 ≈ S5 in latency — that equality is the finding.\n")

for scn, wstyle, kvq in H.SCENARIOS:
    cu, cg = H.PERSCOPE_CFG[scn]
    emit(f"## {scn}  (MSAQ u{cu}/gs{cg})")
    emit(f"| B | prefill bf/mx/mq | decode bf/mx/mq | total bf/mx/mq | mq/bf | mq/mx | mx/bf |")
    emit(f"|--:|--|--|--|--:|--:|--:|")
    for B in BS:
        m = {}
        for fmt, u, gs in H.variants(cu, cg):
            w, kv = H.fmt_paths(fmt, wstyle, kvq)
            r = H.spawn(a, w, kv, u, gs, B, LIN, LOUT, f"{scn}/{fmt}/B{B}")
            r.update(scope=scn, fmt=fmt, B=B, lin=LIN, lout=LOUT, u=cu, gs=cg)
            jf.write(json.dumps(r) + "\n"); jf.flush()
            m[fmt] = H._metrics(r, LOUT)
        bf, mx, mq = m["bf16"], m["mxint8"], m["msaq"]
        c = lambda t, i: ("OOM" if t is None else f"{t[i]:.0f}")
        tri = lambda i: f"{c(bf,i)}/{c(mx,i)}/{c(mq,i)}"
        rat = lambda n, d: ("—" if (n is None or d is None) else f"{n[2]/d[2]:.2f}")
        emit(f"| {B} | {tri(0)} | {tri(1)} | {tri(2)} | {rat(mq,bf)} | {rat(mq,mx)} | {rat(mx,bf)} |")
    emit("")

emit("(mq=MSAQ mx=MXINT8 bf=bf16. prefill=TTFT(1024 tok), decode=integrated over 128 steps, total=sum.)")
jf.close()
with open(os.path.join(outdir, "harness_perscope_results_260625.md"), "w") as f:
    f.write("\n".join(lines) + "\n")
print(f"\n[wrote] tests/harness_perscope_results_260625.md + .jsonl")
