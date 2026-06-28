"""Re-label the per-scope batch sweep as OFFLINE THROUGHPUT (requests/s) at batch B.

Reads one or more latency jsonls (default: the L_out=128 and L_out=512 per-scope sweeps) and
derives throughput WITHOUT re-measuring — RPS is a deterministic transform of (B, total_ms):

    total_ms        = ttft + integ(decode_curve, L_out)          # one batch: prefill + L_out decode steps
    RPS  (req/s)     = B * 1000 / total_ms                        # B requests finish per batch
    decode (tok/s)   = B * L_out * 1000 / decode_ms              # B tokens emitted per decode step

Offline/saturated throughput at a FIXED batch (static batching) — the ceiling an online serving
stack (continuous batching + queueing) approaches, not online RPS itself. Multiple inputs let us
contrast prefill-heavy (L_out=128) vs decode-heavy (L_out=512): KV-quant's win grows as decode
dominates total (prefill is a format-tie for KV scopes).

Writes RPS_results.md. Usage: python tests/rps_report.py [--in a.jsonl,b.jsonl] [--out RPS_results.md]
"""
import os, sys, json, argparse
import numpy as np

HERE = os.path.dirname(__file__)
ap = argparse.ArgumentParser()
ap.add_argument("--in", dest="inp",
                default=",".join([os.path.join(HERE, "harness_perscope_results_260625.jsonl"),
                                  os.path.join(HERE, "harness_perscope_results_260625_l512.jsonl")]))
ap.add_argument("--out", default=os.path.join(os.path.dirname(HERE), "RPS_results.md"))
a = ap.parse_args()

FMTS = ["bf16", "mxint8", "msaq"]
SPARK = "▁▂▃▄▅▆▇█"


def integ(curve, n):
    cks = [c for c, _ in curve]; vals = [v for _, v in curve]
    return float(np.interp(np.arange(1, n + 1), cks, vals).sum())


def metrics(r):
    if r.get("oom") or not r.get("curve"):
        return None
    B, lout = r["B"], r["lout"]
    pre = r["ttft"]; dec = integ(r["curve"], lout); tot = pre + dec
    return dict(pre=pre, dec=dec, tot=tot, rps=B * 1000.0 / tot, tps=B * lout * 1000.0 / dec)


def spark(vals, vmax):
    if vmax <= 0:
        return " " * len(vals)
    return "".join(SPARK[min(7, int(v / vmax * 7.999))] for v in vals)


def load(path):
    """-> (data[scope][fmt][B], scopes, batches, lin, lout)"""
    data, scopes, batches, lin, lout = {}, [], [], None, None
    for line in open(path):
        line = line.strip()
        if not line.startswith("{"):
            continue
        r = json.loads(line)
        if "scope" not in r:
            continue
        s, f, B = r["scope"], r["fmt"], r["B"]
        if s not in data:
            data[s] = {}; scopes.append(s)
        data[s].setdefault(f, {})[B] = metrics(r)
        if B not in batches:
            batches.append(B)
        lin, lout = r.get("lin", lin), r.get("lout", lout)
    batches.sort()
    return data, scopes, batches, lin, lout


L = []
def emit(s=""): L.append(s)


def workload_section(data, scopes, batches, lin, lout):
    emit(f"## Workload: L_in={lin}, L_out={lout}  ({'prefill-heavy' if lout <= 128 else 'decode-heavy'})")
    emit()
    for s in scopes:
        emit(f"### {s}")
        emit()
        emit("**Request throughput — RPS (req/s)**")
        emit()
        emit("| B | bf16 | MXINT8 | MSAQ | mq/bf | mq/mx | mx/bf |")
        emit("|--:|--:|--:|--:|--:|--:|--:|")
        for B in batches:
            m = {f: data[s].get(f, {}).get(B) for f in FMTS}
            if any(m[f] is None for f in FMTS):
                emit(f"| {B} | OOM | OOM | OOM | — | — | — |"); continue
            bf, mx, mq = m["bf16"]["rps"], m["mxint8"]["rps"], m["msaq"]["rps"]
            emit(f"| {B} | {bf:.3f} | {mx:.3f} | {mq:.3f} | {mq/bf:.2f}× | {mq/mx:.2f}× | {mx/bf:.2f}× |")
        emit()
        allrps = [data[s][f][B]["rps"] for f in FMTS for B in batches if data[s].get(f, {}).get(B)]
        vmax = max(allrps) if allrps else 0
        emit(f"**RPS-vs-B curve** (sparkline, norm to scope max {vmax:.2f} req/s; B = {', '.join(map(str, batches))})")
        emit()
        emit("```")
        for f in FMTS:
            vals = [(data[s].get(f, {}).get(B) or {}).get("rps", 0) for B in batches]
            nz = [v for v in vals if v]
            emit(f"  {f:7s} {spark(vals, vmax)}   {min(nz):.2f} → {max(vals):.2f} req/s")
        emit("```")
        emit()
        emit("**Decode throughput (tok/s)** — generation-phase only")
        emit()
        emit("| B | bf16 | MXINT8 | MSAQ | mq/bf | mq/mx |")
        emit("|--:|--:|--:|--:|--:|--:|")
        for B in batches:
            m = {f: data[s].get(f, {}).get(B) for f in FMTS}
            if any(m[f] is None for f in FMTS):
                emit(f"| {B} | OOM | OOM | OOM | — | — |"); continue
            bf, mx, mq = m["bf16"]["tps"], m["mxint8"]["tps"], m["msaq"]["tps"]
            emit(f"| {B} | {bf:.1f} | {mx:.1f} | {mq:.1f} | {mq/bf:.2f}× | {mq/mx:.2f}× |")
        emit()


paths = [p for p in a.inp.split(",") if p and os.path.exists(p)]
loaded = [load(p) for p in paths]                 # list of (data, scopes, batches, lin, lout)
louts = [w[4] for w in loaded]

emit("# Offline throughput (RPS) — per-scope batch sweep")
emit()
emit("Re-label of the per-scope latency sweeps as **offline throughput at batch B** (requests/s). "
     "Derived from the per-scope `torch.cuda.Event` latencies (`tests/harness_perscope_results_260625*.jsonl`); "
     "**no re-measurement** (RPS = `B·1000/total_ms` is an exact transform of B and latency). "
     "`mq`=MSAQ, `mx`=MXINT8, `bf`=bf16. **S3 KV uses u4/gs16** (vpack, byte-roofline KV-read).")
emit()
emit("## Definitions & caveats")
emit()
emit("- **RPS (req/s)** = `B·1000 / total_ms`, `total_ms = ttft + Σ decode-step latency`. Each batch retires B requests.")
emit("- **decode (tok/s)** = `B·L_out·1000 / decode_ms` — pure generation throughput; isolates decode from prefill.")
emit("- **Offline/saturated, static batch** — throughput ceiling at fixed B, not SLA-constrained online RPS.")
emit("- **ratio > 1 = MSAQ faster.** For KV scopes (S3–S6) the win GROWS with L_out: prefill is a format-tie, "
     "so the longer the decode, the less the KV win is diluted (compare the two workloads below).")
emit("- **B≤32**: B≥64 OOMs on 24 GB.")
emit()

for w in loaded:
    workload_section(*w)

# ---- cross-workload contrast: mq/mx RPS by L_out (shows the decode-heavy lift) ----
if len(loaded) >= 2:
    emit("## Workload contrast — MSAQ/MXINT8 RPS by L_out")
    emit()
    emit("How much the KV-quant win grows when decode dominates total (prefill = tie). Each cell = RPS mq/mx.")
    emit()
    scopes0, batches0 = loaded[0][1], loaded[0][2]
    for s in scopes0:
        emit(f"**{s}**")
        emit()
        emit("| L_out | " + " | ".join(f"B={B}" for B in batches0) + " |")
        emit("|---" + "|--:" * len(batches0) + "|")
        for (data, _, batches, _, lout) in loaded:
            cells = []
            for B in batches0:
                mq, mx = data.get(s, {}).get("msaq", {}).get(B), data.get(s, {}).get("mxint8", {}).get(B)
                cells.append(f"{mq['rps']/mx['rps']:.2f}×" if mq and mx else "—")
            emit(f"| {lout} | " + " | ".join(cells) + " |")
        emit()

with open(a.out, "w") as fh:
    fh.write("\n".join(L) + "\n")
print(f"[wrote] {os.path.relpath(a.out)}  (workloads L_out={louts})")
