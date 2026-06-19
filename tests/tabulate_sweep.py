"""Tabulate harness sweep jsonl -> per-model/scope tables (total s, /bf16, /mxint8)."""
import json, sys, glob

rows = []
seen = set()
for fp in sys.argv[1:] or glob.glob("harness_sweep_*.jsonl"):
    for l in open(fp):
        l = l.strip()
        if not l: continue
        r = json.loads(l)
        key = (r["model"], r["scope"], r["fmt"], r.get("u"), r.get("gs"))
        if key in seen: continue          # dedup (e.g. bf16 across split runs)
        seen.add(key); rows.append(r)

MODELS = ["llama31_8b", "gemma2_9b", "mistral_7b"]
SCOPES = ["S1 W-only", "S2 W+A", "S3 KV-only", "S4 W-only+KV"]
by = {}
for r in rows:
    by[(r["model"], r["scope"], r["fmt"], r.get("u"), r.get("gs"))] = r

for m in MODELS:
    bf = by.get((m, "baseline", "bf16", None, None))
    if not bf:
        continue
    print(f"\n{'='*72}\n{m}   (bf16 baseline: total {bf['total']/1e3:.2f}s, TPOT {bf['tpot']:.2f}ms)\n{'='*72}")
    for sc in SCOPES:
        mx = by.get((m, sc, "mxint8", None, None))
        if not mx: continue
        print(f"\n  {sc}   MXINT8: total {mx['total']/1e3:7.2f}s  TPOT {mx['tpot']:6.2f}  "
              f"/bf16 {mx['total']/bf['total']:.2f}")
        print(f"    {'MSAQ':>10} | {'total(s)':>9} {'TPOT':>6} | {'/bf16':>6} {'/mxint8':>8}")
        for u in (2, 3, 4):
            for gs in (2, 8, 32):
                r = by.get((m, sc, "msaq", u, gs))
                if not r: continue
                win = "  <-- win" if r["total"] < mx["total"] else ""
                print(f"    u{u} gs{gs:<2}    | {r['total']/1e3:9.2f} {r['tpot']:6.2f} | "
                      f"{r['total']/bf['total']:6.2f} {r['total']/mx['total']:8.2f}{win}")

# compact best-MSAQ-per-(model,scope) summary
print(f"\n\n{'='*72}\nBEST MSAQ per (model, scope)  [min total over u,gs; ratio vs that scope's MXINT8]\n{'='*72}")
print(f"  {'model':12} {'scope':14} {'bestcfg':9} {'/mxint8':>8} {'/bf16':>7}")
for m in MODELS:
    bf = by.get((m, "baseline", "bf16", None, None))
    if not bf: continue
    for sc in SCOPES:
        mx = by.get((m, sc, "mxint8", None, None))
        if not mx: continue
        cands = [(by[k]) for k in by if k[0]==m and k[1]==sc and k[2]=="msaq"]
        if not cands: continue
        best = min(cands, key=lambda r: r["total"])
        print(f"  {m:12} {sc:14} u{best['u']}g{best['gs']:<6} "
              f"{best['total']/mx['total']:8.2f} {best['total']/bf['total']:7.2f}")
