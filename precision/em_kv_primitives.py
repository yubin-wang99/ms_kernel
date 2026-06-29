"""EM_sharing primitive re-examination on REAL KV tensors — same P0-P3 QSNR + zero-mean/envelope
diagnostic as the weight study (em_primitives.py), to answer: does the exponent axis pay on KV, where
the residual was hoped to be non-zero-mean / strongly outlier-enveloped?

Prereq: run em_kv_capture.py first to produce precision/kv_cap.pt (real K/V from a Llama forward).
Run:  CUDA_VISIBLE_DEVICES="" .venv/bin/python precision/em_kv_primitives.py   (CPU; tensors are small)
"""
import sys; sys.path.insert(0,"/home/yubin/ms_kernel/precision")
import torch
import mxfp6_verify
try: torch.zeros(1, device="cuda"); DEV="cuda"
except Exception: DEV="cpu"
mxfp6_verify.DEV = DEV
from em_sharing import _base_encode, qsnr, BASES
OUT="/home/yubin/ms_kernel/precision"
BLOCK=32

def reconstruct(W, base, prim, u, gm, ge, be, efb=2):
    xf=W.reshape(-1,BLOCK); spec=BASES[base]; s,snap=_base_encode(xf,spec)
    y=xf/s; b=snap(y); hi=(1<<(u-1))-1; lo=-(1<<(u-1)); R,C=xf.shape
    ee=torch.floor(torch.log2(y.abs().clamp(min=1e-12)))
    def enc(r):
        if prim=="P0":
            ref=torch.floor(torch.log2(r.abs().amax(-1,keepdim=True).clamp(min=1e-30)/max(hi,1)))
            if be:
                am=r.reshape(R,C//ge,ge).abs().amax(-1,keepdim=True).clamp(min=1e-30)
                raw=torch.floor(torch.log2(am/max(hi,1))); d=(ref.unsqueeze(1)-raw).round().clamp(0,(1<<be)-1)
                sc=torch.exp2((ref.unsqueeze(1)-d).expand(-1,-1,ge).reshape(R,C))
            else: sc=torch.exp2(ref)
            m=(r/sc).reshape(R,C//gm,gm).mean(-1,keepdim=True).round().clamp(lo,hi)
            return m.expand(-1,-1,gm).reshape(R,C)*sc
        if prim=="P1":
            sc=torch.exp2(ee); m=(r/sc).reshape(R,C//gm,gm).mean(-1,keepdim=True).round().clamp(lo,hi)
            return m.expand(-1,-1,gm).reshape(R,C)*sc
        if prim in("P2","P3"):
            sgn=torch.sign(r); qm=(1<<u)-1
            if prim=="P2": sc=torch.exp2(ee)
            else:
                ref=torch.floor(torch.log2(r.abs().amax(-1,keepdim=True).clamp(min=1e-30)))
                am=r.reshape(R,C//ge,ge).abs().amax(-1,keepdim=True).clamp(min=1e-30)
                raw=torch.floor(torch.log2(am)); d=(ref.unsqueeze(1)-raw).round().clamp(0,(1<<be)-1)
                sc=torch.exp2((ref.unsqueeze(1)-d).expand(-1,-1,ge).reshape(R,C))
            mag=(r.abs()/sc).reshape(R,C//gm,gm).mean(-1,keepdim=True).round().clamp(0,qm)
            return sgn*mag.expand(-1,-1,gm).reshape(R,C)*sc
    rh=enc(y-b)
    for _ in range(efb): b=snap(y-rh); rh=enc(y-b)
    return ((b+rh)*s).reshape(W.shape)

def bres(prim,u,gm,ge,be):
    base=u/gm + (be/ge if (prim in("P0","P3") and be) else 0)
    return base + (1.0 if prim in("P2","P3") else 0)

def diag(name, W, base):
    xf=W.reshape(-1,BLOCK); spec=BASES[base]; s,snap=_base_encode(xf,spec)
    y=xf/s; b=snap(y); r=(y-b)
    ge=8; rg=r.reshape(r.shape[0],BLOCK//ge,ge); am=rg.abs().amax(-1)
    cov=(am.std(-1)/am.mean(-1).clamp(min=1e-30)).mean()
    ratio=(am.amax(-1)/am.amin(-1).clamp(min=1e-30)).median()
    zm=(r.mean(-1).abs().mean()/r.std(-1).mean())
    # gm32 single-mantissa washout
    hi=1; ref=torch.floor(torch.log2(r.abs().amax(-1,keepdim=True).clamp(min=1e-30)/hi))
    m=(r/torch.exp2(ref)).mean(-1,keepdim=True).round().clamp(-2,1)
    print(f"  [{name:>10} {base}] envCoV={cov:.3f} ratio={ratio:.1f}x  mean/std={zm:.3f}  "
          f"gm32-washout(u2)={(m==0).float().mean():.3f}")

cap=torch.load(f"{OUT}/kv_cap.pt")
print(f"=== EM_sharing KV primitive re-examination (device={DEV}) ===")
TENS=[("L0/K_rot",cap[0]["K_rot"]),("L0/V_tok",cap[0]["V_tok"]),
      ("L16/K_rot",cap[16]["K_rot"]),("L16/V_tok",cap[16]["V_tok"]),
      ("L0/K_raw",cap[0]["K_raw"])]
print("\n-- diagnostics (is KV residual zero-mean? envelope variation?) --")
for nm,W in TENS:
    W=W.to(DEV).float()
    diag(nm,W,"FP4")

cfgs=[("P0","mant gm8",8,32,0,3),("P0","mant gm4",4,32,0,3),("P0","mant gm2",2,32,0,3),
      ("P0","DCexp ge8 be2 gm32",32,8,2,3),
      ("P2","sgn freeenv gm8",8,32,0,3),
      ("P3","sgn ge8 be2 gm32",32,8,2,3),("P3","sgn ge4 be2 gm32",32,4,2,3),
      ("P3","sgn ge8 be2 gm8",8,8,2,3)]
for nm,W in TENS:
    W=W.to(DEV).float()
    print(f"\n-- {nm}  (FP4 base; QSNR vs B_res) --")
    print(f"  {'prim':>4} {'cfg':>20} {'B_res':>6} {'QSNR':>7}")
    for prim,desc,gm,ge,be,u in cfgs:
        q=qsnr(W,reconstruct(W,"FP4",prim,u,gm,ge,be))
        print(f"  {prim:>4} {desc:>20} {bres(prim,u,gm,ge,be):>6.3f} {q:>7.2f}")
