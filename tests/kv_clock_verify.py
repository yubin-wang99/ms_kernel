"""Verify Phase 18 wide win is NOT a hot/cold clock-boost artifact.
- block-interleaved (mx,wide alternated) so both see the same average clock
- measured in BOTH orders (mx-first and wide-first) -> bias would flip the winner
- SM clock sampled during the run (must be boosted + stable)
- within-MSAQ cp.async-vs-wide is immune to cross-kernel clock bias (same kernel family)
Run: CUDA_VISIBLE_DEVICES=1 python tests/kv_clock_verify.py
"""
import os, time, subprocess, numpy as np, torch
from ms_lib.pack import pack_kv, pack_kv_mxint8
from ms_lib import ops
assert ops.available()
def _cuda(a): return torch.from_numpy(a).cuda()

def sm_clock(phys_idx=1):
    out = subprocess.check_output(
        ["nvidia-smi", "-i", str(phys_idx),
         "--query-gpu=clocks.current.sm", "--format=csv,noheader,nounits"])
    return int(out.decode().strip())

def blk(fn, it=300):
    s = torch.cuda.Event(True); e = torch.cuda.Event(True)
    torch.cuda.synchronize(); s.record()
    for _ in range(it): fn()
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) / it

def interleave(fa, fb, rounds=6, warm=4.0):
    t0 = time.time()
    while time.time() - t0 < warm: fa(); fb()          # heavy warm, no idle gap
    a, b, clk = [], [], []
    for r in range(rounds):
        a.append(blk(fa)); clk.append(sm_clock())
        b.append(blk(fb)); clk.append(sm_clock())
    return np.median(a), np.median(b), (min(clk), max(clk))

def run(u=4, gs=8, H=8, Lk=16384, D=128):
    rng = np.random.default_rng(3)
    K = (rng.standard_normal((H, Lk, D))*0.5).astype(np.float32)
    V = (rng.standard_normal((H, Lk, D))*0.5).astype(np.float32)
    q = (rng.standard_normal((H, D))*0.5).astype(np.float32)
    pmK, pmV = pack_kv(K, u, gs), pack_kv(V, u, gs)
    pxK, pxV = pack_kv_mxint8(K), pack_kv_mxint8(V)
    ks,ku,kh = _cuda(pmK["scale_exp"]),_cuda(pmK["upper"]),_cuda(pmK["shared"])
    vs,vu,vh = _cuda(pmV["scale_exp"]),_cuda(pmV["upper"]),_cuda(pmV["shared"])
    kxs,kxq = _cuda(pxK["scale_exp"]),_cuda(pxK["qweight"])
    vxs,vxq = _cuda(pxV["scale_exp"]),_cuda(pxV["qweight"])
    qt = torch.from_numpy(q).to(torch.bfloat16).cuda()
    nb, nbx = pmK["nb"], pxK["nb"]
    def msaq(): return torch.ops.msaq.kv_decode_attention(qt,ks,ku,kh,vs,vu,vh,H,Lk,D,nb,u,gs)
    def mx():   return torch.ops.msaq.mxint8_kv_decode(qt,kxs,kxq,vxs,vxq,H,Lk,D,nbx)

    print(f"\n=== u{u} gs{gs} H{H} Lk{Lk} D{D} ===")
    # (1) cross-kernel both orders
    os.environ["MS_KV_WIDE"]="1"
    w1, x1, c1 = interleave(msaq, mx)                     # wide measured FIRST each round
    x2, w2, c2 = interleave(mx, msaq)                     # mx measured FIRST each round
    print(f"  [wide-first] wide {w1*1e3:6.1f} us | mx {x1*1e3:6.1f} us | wide/mx {w1/x1:.3f}  clk {c1}")
    print(f"  [mx-first  ] wide {w2*1e3:6.1f} us | mx {x2*1e3:6.1f} us | wide/mx {w2/x2:.3f}  clk {c2}")
    print(f"  --> ratio stable across order? {w1/x1:.3f} vs {w2/x2:.3f}")
    # (2) within-MSAQ (immune to cross-kernel clock bias): cp.async vs wide
    def msaq_cp():
        os.environ["MS_KV_CPASYNC"]="1"; return msaq()
    os.environ["MS_KV_WIDE"]="0"; tc = blk(msaq); ccp = sm_clock()
    os.environ["MS_KV_WIDE"]="1"; tw = blk(msaq); cwd = sm_clock()
    os.environ["MS_KV_WIDE"]="0"; tc2 = blk(msaq)
    os.environ["MS_KV_WIDE"]="1"; tw2 = blk(msaq)
    tc, tw = min(tc,tc2), min(tw,tw2)
    print(f"  within-MSAQ: cp.async {tc*1e3:6.1f} us (clk {ccp}) | wide {tw*1e3:6.1f} us (clk {cwd})"
          f" | wide/cpasync {tw/tc:.3f}")

if __name__ == "__main__":
    torch.cuda.init()
    print(f"[clock_verify] {torch.cuda.get_device_name(0)}  idle->measuring")
    run(Lk=4096)
    run(Lk=16384)
