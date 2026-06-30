#!/usr/bin/env python3
"""Empirical OOM sweep (2026-06-30) — push batch until a REAL device OOM, per (case, format).
Validates the analytical B_max (rps_iso_memory_bench.py) against the hardware cliff. For each B we
actually allocate weights + B requests' ALL-LAYER KV on the GPU (rps_iso_memory_bench.verify_alloc)
and step B up until allocation fails. Reports: empirical last-stable B (raw hard-OOM), the analytical
operational B_max (M_avail=0.86*24GB), and a real-decode confirmation at the empirical B_max.
Run: CUDA_VISIBLE_DEVICES=0 python oom_sweep_0630.py
"""
import numpy as np, torch
import rps_iso_memory_bench as H

L_seq = 1024 + 128
CASES = {"KV": H.CASES["KV"], "Weight+KV": H.CASES["Weight+KV"]}


def empirical_bmax(b_w, b_kv):
    """coarse step up to find OOM band, then fine 1-step to the exact last-stable B (raw device)."""
    B, last = 0, 0
    for step in (16, 4, 1):
        B = last
        while True:
            nb = B + step
            if H.verify_alloc(b_w, b_kv, L_seq, nb):
                last = nb; B = nb
            else:
                break
    return last


def real_decode_ok(b_kv, kv_fmt, u, gs, B):
    """confirm one real decode-attention step actually runs at B (incl. activation transients)."""
    try:
        H.attn_ms(B, L_seq, kv_fmt, u, gs); torch.cuda.synchronize(); return True
    except RuntimeError:
        torch.cuda.empty_cache(); return False


def main():
    dev = torch.cuda.get_device_name(0)
    tot = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"# {dev} | total {tot:.2f}GB | M_avail(operational)={H.M_AVAIL/1e9:.2f}GB | L_seq={L_seq}")
    print(f"# empirical = raw-device hard OOM (no 0.9 cap); analytical = operational (M_avail)\n")
    print(f"{'case':>10} {'format':>8} {'b_w':>5} {'b_kv':>5} | {'analytic B_max':>14} {'empirical B_max':>15} "
          f"{'decode@emp':>11}")
    rows = []
    for case, fmts in CASES.items():
        for nm, b_w, b_kv, kv_fmt, u, gs in fmts:
            ana = H.b_max_analytic(b_w, b_kv, L_seq)
            emp = empirical_bmax(b_w, b_kv)
            dec = "n/a" if nm == "BF16" else ("ok" if real_decode_ok(b_kv, kv_fmt, u, gs, min(emp, 256)) else "OOM")
            print(f"{case:>10} {nm:>8} {b_w:>5.2f} {b_kv:>5.2f} | {ana:>14} {emp:>15} {dec:>11}")
            rows.append((case, nm, ana, emp))
            torch.cuda.empty_cache()
    # ratios vs MXINT8 (empirical)
    print("\n# empirical B_max ratios (MSAQ vs MXINT8) — does max batch really rise?")
    for case in CASES:
        cr = [r for r in rows if r[0] == case]
        mx = next(r for r in cr if r[1] == "MXINT8")[3]
        ms = next(r for r in cr if r[1] == "MSAQ")[3]
        bf = next(r for r in cr if r[1] == "BF16")[3]
        print(f"  {case:>10}: MSAQ {ms} / MXINT8 {mx} = {ms/mx:.2f}x   (vs BF16 {bf} = {ms/bf:.2f}x)")


if __name__ == "__main__":
    main()
