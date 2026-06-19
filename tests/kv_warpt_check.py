"""Correctness: design-B warp-transpose (MS_KV_WARPT) vs wide kernel vs numpy oracle, D=128 u4."""
import os, math, numpy as np, torch
from ms_lib.pack import pack_kv, dequant_weight
from ms_lib import ops; assert ops.available()

def oracle(Q, pK, pV, H, Lk, D):
    out = np.zeros((H, D), np.float64)
    for h in range(H):
        Kd = dequant_weight(pK["_per"][h]); Vd = dequant_weight(pV["_per"][h])
        sc = (Q[h].astype(np.float64) @ Kd.T) / math.sqrt(D)
        sc -= sc.max(); p = np.exp(sc); p /= p.sum()
        out[h] = p @ Vd
    return out

def rel_fro(a, b): return np.linalg.norm(a-b)/ (np.linalg.norm(b)+1e-12)

def run(u=4, gs=8, H=8, Lk=4096, D=128, seed=1):
    rng = np.random.default_rng(seed)
    K = (rng.standard_normal((H, Lk, D))*0.5).astype(np.float32)
    V = (rng.standard_normal((H, Lk, D))*0.5).astype(np.float32)
    Q = (rng.standard_normal((H, D))*0.5).astype(np.float32)
    pK, pV = pack_kv(K, u, gs), pack_kv(V, u, gs)
    qt = torch.from_numpy(Q).to(torch.bfloat16).cuda()
    def call():
        ks,ku,kh = (torch.from_numpy(pK[k]).cuda() for k in ("scale_exp","upper","shared"))
        vs,vu,vh = (torch.from_numpy(pV[k]).cuda() for k in ("scale_exp","upper","shared"))
        return torch.ops.msaq.kv_decode_attention(qt,ks,ku,kh,vs,vu,vh,H,H,Lk,D,pK["nb"],u,gs).float().cpu().numpy()
    os.environ.pop("MS_KV_WARPT", None); os.environ["MS_KV_WIDE"]="1"
    wide = call()
    os.environ["MS_KV_WARPT"]="1"
    warpt = call()
    ref = oracle(Q, pK, pV, H, Lk, D)
    print(f"u{u} H{H} Lk{Lk} D{D}:  wide vs oracle {rel_fro(wide,ref):.2e} | "
          f"warpT vs oracle {rel_fro(warpt,ref):.2e} | warpT vs wide {rel_fro(warpt,wide):.2e}")
    return rel_fro(warpt, ref)

if __name__ == "__main__":
    torch.cuda.init()
    bad = 0
    for Lk in (37, 128, 129, 1056, 2848, 4680):   # incl. non-multiples of chunk/32 (tail paths)
        r = run(u=4, H=8, Lk=Lk, D=128)
        if r > 2e-2: bad += 1
    print("ALL OK" if bad == 0 else f"{bad} FAILED")
