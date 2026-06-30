"""End-to-end RPS for FP4 + vector-VQ KV residual — reuses the reviewed capacity_model.py math.

Plugs the MEASURED iso-accuracy KV operating points (vq_kv_results.md / vq_kv_global_results.md PPL gate)
and the MEASURED decode bandwidth (vq_c2_results.md microbench: ~560 GB/s achievable, correction free)
into the capacity model: M_avail >= W + KV(B) -> B_max -> memory-bound decode throughput -> req/s.

KV-isolated comparison: weights fixed at MXINT8 (8.25 b/elem) across all rows, so the only variable is
b_kv -> pure KV-residual contribution to capacity. Target: Llama-3.1-8B on RTX PRO 4000 Blackwell (24 GB),
workload L_in=1024 / L_out=128 (L_seq=1152), context swept to the KV-bound regime.

Run: python precision/vq_rps.py > precision/vq_rps_results.txt
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import capacity_model as cm

# RTX PRO 4000 Blackwell: 24 GB; decode BW from the C2 microbench (copy peak ~555, kdot ~560-569 GB/s).
cm.GPUS["rtxpro4000"] = (24e9, 560e9)

# KV-isolated: weights = MXINT8 8.25 everywhere; vary ONLY b_kv at each format's measured PPL gate point.
cm.FORMATS = [
    ("bf16",            16.0, 16.0, "lossless ref"),
    ("MXINT8 (W+KV)",    8.25, 8.25, "~lossless; reviewer baseline"),
    ("native FP6 KV",    8.25, 6.25, "+0.30% (lossless-class)"),
    ("mantissa-share KV",8.25, 5.44, "+~1.9% (incumbent, MX+ E2M1)"),
    ("FP4+VQ KV 5.25",   8.25, 5.25, "+1.79% (this work)"),
    ("FP4+VQ KV 4.75",   8.25, 4.75, "+2.44% (this work, aggressive)"),
    ("native FP4 KV",    8.25, 4.25, "+3.46% (FAILS 3% gate — ref only)"),
]

# bw_eff=1.0: BW above is already the MEASURED achievable decode bandwidth (not peak spec).
sys.argv = ["vq_rps", "--model", "llama8b", "--gpu", "rtxpro4000", "--util", "0.90",
            "--workspace_gb", "2.0", "--bw_eff", "1.0", "--lout", "128",
            "--ctx", "1152,4096,16384,65536,131072"]
cm.main()
