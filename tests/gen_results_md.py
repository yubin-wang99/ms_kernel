"""Generate ../results.md (BF16-normalized total-time tables) from sweep jsonl files.
Usage: python gen_results_md.py harness_sweep_*.jsonl   (dedups; missing models skipped)
"""
import json, sys, glob, os

rows, seen = [], set()
for fp in sys.argv[1:] or glob.glob("harness_sweep_*.jsonl"):
    if not os.path.exists(fp): continue
    for l in open(fp):
        l = l.strip()
        if not l: continue
        r = json.loads(l)
        k = (r["model"], r["scope"], r["fmt"], r.get("u"), r.get("gs"))
        if k in seen: continue
        seen.add(k); rows.append(r)
by = {(r["model"], r["scope"], r["fmt"], r.get("u"), r.get("gs")): r for r in rows}

MODELS = [("llama31_8b", "Llama-3.1-8B"), ("gemma2_9b", "Gemma-2-9B"), ("mistral_7b", "Mistral-7B")]
SCOPES = [("S1 W-only", "S1 W-only"), ("S2 W+A", "S2 W+A"),
          ("S3 KV-only", "S3 KV-only"), ("S4 W-only+KV", "S4 W+KV")]

out = []
out.append("# End-to-End Harness Results — BF16-normalized total time\n")
out.append("RTX 3090, CUDA-graph decode (prefill=800 / decode=3880). Timing harness: random reused "
           "weights, glue (RMSNorm/RoPE/SwiGLU/SDPA) in bf16 common to all paths. **Each cell = total "
           "inference time normalized to that (model, scope)'s BF16 = 1.000; lower = faster.** Scopes "
           "apply quantization to: S1 weights (W-only GEMM/GEMV), S2 weights+activations (INT8 IMMA / "
           "int-dot), S3 KV cache only, S4 weights+KV. MSAQ swept over u∈{2,3,4} × gs∈{2,8,32}; MXINT8 "
           "and BF16 have no u/gs. See [kernel_ver2.md](kernel_ver2.md), [for_fair_comparison.md].\n")

for mk, mname in MODELS:
    bf = by.get((mk, "baseline", "bf16", None, None))
    if not bf:
        out.append(f"\n## {mname}\n\n_(pending — sweep not yet complete)_\n")
        continue
    b = bf["total"]
    out.append(f"\n## {mname}  (BF16 = 1.000, abs {b/1e3:.1f}s)\n")
    hdr = "| config | " + " | ".join(s[1] for s in SCOPES) + " |"
    sep = "|---|" + "---|" * len(SCOPES)
    out.append(hdr); out.append(sep)
    out.append("| **BF16** | " + " | ".join("1.000" for _ in SCOPES) + " |")
    mxr = []
    for sk, _ in SCOPES:
        r = by.get((mk, sk, "mxint8", None, None))
        mxr.append(f"{r['total']/b:.3f}" if r else "—")
    out.append("| **MXINT8** | " + " | ".join(mxr) + " |")
    for u in (2, 3, 4):
        for gs in (2, 8, 32):
            cells = []
            for sk, _ in SCOPES:
                r = by.get((mk, sk, "msaq", u, gs))
                cells.append(f"{r['total']/b:.3f}" if r else "—")
            bold = "**" if (u == 4 and gs == 32) else ""
            out.append(f"| {bold}MSAQ u{u} gs{gs}{bold} | " + " | ".join(cells) + " |")

# best-MSAQ summary
out.append("\n## Best MSAQ (min over u,gs) — normalized to BF16 and to MXINT8\n")
out.append("| model | scope | best cfg | /bf16 | /mxint8 |")
out.append("|---|---|---|---|---|")
for mk, mname in MODELS:
    bf = by.get((mk, "baseline", "bf16", None, None))
    if not bf: continue
    b = bf["total"]
    for sk, sname in SCOPES:
        mx = by.get((mk, sk, "mxint8", None, None))
        cands = [by[k] for k in by if k[0] == mk and k[1] == sk and k[2] == "msaq"]
        if not mx or not cands: continue
        best = min(cands, key=lambda r: r["total"])
        out.append(f"| {mname} | {sname} | u{best['u']} gs{best['gs']} | "
                   f"{best['total']/b:.3f} | {best['total']/mx['total']:.3f} |")

dst = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results.md")
with open(dst, "w") as f:
    f.write("\n".join(out) + "\n")
print(f"wrote {os.path.abspath(dst)}  ({len([m for m,_ in MODELS if by.get((m,'baseline','bf16',None,None))])}/3 models)")
