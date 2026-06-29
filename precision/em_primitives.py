"""EM_sharing residual-PRIMITIVE re-examination (calibration-free QSNR on Llama-3.1-8B weights).

The base dual-hypothesis test (em_sharing.py) found the residual EXPONENT earns no bits on weights,
because the EM_sharing primitive is a DC-mean mantissa and the rounding residual is zero-mean -> a
coarse-group mean washes to ~0, leaving the exponent nothing to scale. This file asks: is that a
property of the DC-mean PRIMITIVE, fixable by a richer shared primitive?

  P0  DC-mean (+ optional be-exp per ge)            -- the EM_sharing primitive
  P1  DC-mean / per-element free envelope (be=0)     -- "exponent for free from base"
  P2  per-element SIGN + shared magnitude * free env -- un-share the sign, free envelope
  P3  per-element SIGN + shared magnitude * be-exp   -- un-share the sign, STORED be-exp per ge

Verdict (see em_sharing_results.md): P1/P2 wash out (free envelope doesn't help; magnitude still ~0
or mis-scaled). P3 REVIVES the exponent axis -- finer ge genuinely raises QSNR -- but only TIES plain
mantissa-fine (P0, ge=32) iso-bit, while paying +1 b/elem for the per-element sign (which destroys the
sharing economics EM_sharing exists for). So no primitive gives exp-fine a usable iso-bit win on
weights; the weight gate stays closed across primitive choices.

Run: CUDA_VISIBLE_DEVICES=0 .venv/bin/python precision/em_primitives.py
"""
import torch
from em_sharing import _base_encode, qsnr, BASES, _load_layers
DEV="cuda"; BLOCK=32

def reconstruct(W, base, prim, u, gm, ge, be, efb=2):
    xf=W.reshape(-1,BLOCK); spec=BASES[base]; s,snap=_base_encode(xf,spec)
    y=xf/s; b=snap(y); hi=(1<<(u-1))-1; lo=-(1<<(u-1)); R,C=xf.shape
    # per-element envelope from the ELEMENT magnitude (free; floor-clamped, no zeros)
    ee=torch.floor(torch.log2(y.abs().clamp(min=1e-12)))
    def enc(r):
        if prim=="P0":  # DC-mean + optional be-exp per ge
            ref=torch.floor(torch.log2(r.abs().amax(-1,keepdim=True).clamp(min=1e-30)/max(hi,1)))
            if be:
                am=r.reshape(R,C//ge,ge).abs().amax(-1,keepdim=True).clamp(min=1e-30)
                raw=torch.floor(torch.log2(am/max(hi,1))); d=(ref.unsqueeze(1)-raw).round().clamp(0,(1<<be)-1)
                sc=torch.exp2((ref.unsqueeze(1)-d).expand(-1,-1,ge).reshape(R,C))
            else: sc=torch.exp2(ref)
            m=(r/sc).reshape(R,C//gm,gm).mean(-1,keepdim=True).round().clamp(lo,hi)
            return m.expand(-1,-1,gm).reshape(R,C)*sc
        if prim=="P1":  # DC-mean / per-elem free envelope, be=0
            sc=torch.exp2(ee); m=(r/sc).reshape(R,C//gm,gm).mean(-1,keepdim=True).round().clamp(lo,hi)
            return m.expand(-1,-1,gm).reshape(R,C)*sc
        if prim in("P2","P3"):  # per-elem sign + shared unsigned magnitude/gm * envelope
            sgn=torch.sign(r); qm=(1<<u)-1
            if prim=="P2": sc=torch.exp2(ee)            # free envelope
            else:                                       # be-bit exponent per ge
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

W=_load_layers()["q_proj"].to(DEV).float()
for base in ("FP4","INT4"):
  print(f"\n{base} q_proj — primitive QSNR vs B_res (efb2)")
  print(f"  {'prim':>4} {'cfg':>20} {'B_res':>6} {'QSNR':>7}")
  cfgs=[("P0","mant gm8",8,32,0,3),("P0","mant gm4",4,32,0,3),("P0","mant gm2",2,32,0,3),
        ("P0","mant gm1",1,32,0,3),
        ("P1","DC/freeenv gm4",4,32,0,3),
        ("P2","sgn freeenv gm32",32,32,0,3),("P2","sgn freeenv gm8",8,32,0,3),
        ("P3","sgn ge16 be2 gm32",32,16,2,3),("P3","sgn ge8 be2 gm32",32,8,2,3),
        ("P3","sgn ge4 be2 gm32",32,4,2,3),("P3","sgn ge8 be3 gm32",32,8,3,3)]
  for prim,desc,gm,ge,be,u in cfgs:
    q=qsnr(W,reconstruct(W,base,prim,u,gm,ge,be)); print(f"  {prim:>4} {desc:>20} {bres(prim,u,gm,ge,be):>6.3f} {q:>7.2f}")
