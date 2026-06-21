"""(i) quant-kernel time: MSAQ vs light-MS (INT-friendly decompose) vs MXINT8.
KV write (prefill decompose, real workload) + KV append (decode, launch-bound).
Confirms light-MS doesn't increase quant time vs MSAQ.
Run: CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python tests/quant_time_bench.py
"""
import os, time, numpy as np, torch
from ms_lib import ops; assert ops.available()
OPS = torch.ops.msaq
def cuda(a): return torch.from_numpy(a).cuda()

def _t(fn, it=200):
    s = torch.cuda.Event(True); e = torch.cuda.Event(True)
    torch.cuda.synchronize(); s.record()
    for _ in range(it): fn()
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) / it

def warm(fns, sec=3.0):
    t0 = time.time()
    while time.time() - t0 < sec:
        for f in fns: f()

def run_write(H=8, L=2048, D=128, u=4, gs=8):
    rng = np.random.default_rng(0)
    X = cuda((rng.standard_normal((H, L, D)) * 0.5).astype(np.float32)).to(torch.bfloat16)
    nb = D // 32
    msaq = lambda: OPS.kv_write(X, H, L, D, nb, u, gs)
    mx = lambda: OPS.mxint8_kv_write(X, H, L, D, nb)
    def set(v): os.environ["MS_LIGHTMS"] = v
    set("0"); warm([msaq, mx])
    set("0"); t_msaq = min(_t(msaq), _t(msaq))
    set("1"); t_light = min(_t(msaq), _t(msaq))
    set("0"); t_mx = min(_t(mx), _t(mx))
    print(f"  KV write  H{H} L{L} D{D} u{u} gs{gs}:")
    print(f"    MXINT8        {t_mx*1e3:6.2f}us")
    print(f"    MSAQ (FP-avg) {t_msaq*1e3:6.2f}us   {t_msaq/t_mx:.2f}x MX")
    print(f"    light-MS(INT) {t_light*1e3:6.2f}us   {t_light/t_mx:.2f}x MX   ({t_light/t_msaq:.3f}x MSAQ)")

def run_append(H=8, D=128, u=4, gs=8, Lcap=4096):
    rng = np.random.default_rng(1)
    nb = D // 32
    # allocate cache planes
    UB = 32 * (8 - u) // 8; SB = ((32 // gs) * u + 7) // 8
    ks = torch.zeros((H, nb, Lcap), dtype=torch.int8, device="cuda")
    ku = torch.zeros((H, nb, Lcap, UB), dtype=torch.uint8, device="cuda")
    kh = torch.zeros((H, nb, Lcap, SB), dtype=torch.uint8, device="cuda")
    xks = torch.zeros((H, nb, Lcap), dtype=torch.int8, device="cuda")
    xkq = torch.zeros((H, nb, Lcap, 32), dtype=torch.int8, device="cuda")
    x = cuda((rng.standard_normal((H, D)) * 0.5).astype(np.float32)).to(torch.bfloat16)
    msaq = lambda: OPS.kv_append(x, ks, ku, kh, H, D, nb, 0, Lcap, u, gs)
    mx = lambda: OPS.mxint8_kv_append(x, xks, xkq, H, D, nb, 0, Lcap)
    def set(v): os.environ["MS_LIGHTMS"] = v
    set("0"); warm([msaq, mx])
    set("0"); t_msaq = min(_t(msaq), _t(msaq))
    set("1"); t_light = min(_t(msaq), _t(msaq))
    set("0"); t_mx = min(_t(mx), _t(mx))
    print(f"  KV append (1 token, launch-bound) H{H} D{D} u{u}:")
    print(f"    MXINT8 {t_mx*1e3:6.2f}us | MSAQ {t_msaq*1e3:6.2f}us | light-MS {t_light*1e3:6.2f}us "
          f"(light/MSAQ {t_light/t_msaq:.3f}x)")

if __name__ == "__main__":
    torch.cuda.init()
    print("=== (i) quant-kernel time: MSAQ vs light-MS(INT) vs MXINT8 ===")
    for u in (4, 3, 2):
        run_write(u=u)
    run_append(u=4)
