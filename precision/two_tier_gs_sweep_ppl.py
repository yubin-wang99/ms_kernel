"""gs sweep for two-tier MSAQ residual: is gs=32 (one shared value per 32-block) leaving accuracy on
the table? Sweep gs in {2,4,8,16,32} = sub-groups WITHIN the MX block, each carrying its own u-bit
shared residual. Smaller gs -> finer residual (more shared values) -> better accuracy but +u/gs bits.
The u4~u2 saturation in two_tier_*_ppl.py was measured at gs=32 ONLY; finer gs may break it.

Bit-efficient residual: ONE E8M0 per 32-block (shared by all sub-groups), plus a u-bit signed int per
sub-group. bits = (1+eb+mb) + 8/32[base scale] + 8/32[resid scale] + u/gs[shared] (+5/32 if MX+).

Unified quantizer covers DC (Hblk=None), A-weighted (§4, Hblk=[G,32,32]), and MX+ base.
Selftest:  .venv/bin/python precision/two_tier_gs_sweep_ppl.py --selftest
Run:
    HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 MSAQ_MODEL=NousResearch/Meta-Llama-3.1-8B \
        CUDA_VISIBLE_DEVICES=0 .venv/bin/python precision/two_tier_gs_sweep_ppl.py \
        > precision/two_tier_gs_sweep_llama31_8b.txt 2>&1
"""
import os, sys, torch
from mxfp6_verify import _fp6_grid
from msaq_mxfp8_ppl import BLOCK, msaq_mxint8_efb
from two_tier_ppl import two_tier, _snap_grid
from two_tier_aware_ppl import collect_hessian
from two_tier_mxplus_ppl import _mxplus_snap


def quant(x, eb, mb, u, gs, Hblk=None, mxplus=False, efb_iters=2):
    """Two-tier with sub-group residual. x=[N,K]; gs | 32. Hblk=[G,32,32] -> A-weighted per sub-group
    optimum (uses the gs x gs diagonal sub-blocks of the within-block Gram); None -> DC mean."""
    assert x.dim() == 2 and BLOCK % gs == 0
    N, K = x.shape; G = K // BLOCK; nsg = BLOCK // gs
    grid = _fp6_grid(eb, mb); maxval = grid[-1].item()
    maxexp = ((1 << eb) - 1) - ((1 << (eb - 1)) - 1)
    xf = x.to(torch.float32).reshape(N, G, BLOCK)
    absmax = xf.abs().amax(-1, keepdim=True).clamp(min=1e-30)
    s = torch.exp2(torch.floor(torch.log2(absmax)) - float(maxexp))
    y = xf / s

    def snap(t):
        if mxplus: return _mxplus_snap(t, eb, mb, maxexp, grid, maxval)
        return torch.sign(t) * _snap_grid(t.abs().clamp(max=maxval), grid)

    base = snap(y)
    if u == 0:
        return (base * s).reshape(x.shape).to(x.dtype)
    if Hblk is not None:                                               # A-weighted: sub-group Gram
        H = Hblk.to(torch.float32).reshape(G, nsg, gs, nsg, gs)
        i = torch.arange(nsg, device=x.device)
        Hd = H[:, i, :, i, :]                                          # [nsg,G,gs,gs] (adv-idx -> dim0)
        Hd = Hd.permute(1, 0, 2, 3).contiguous()                      # [G,nsg,gs,gs]
        denom = Hd.sum((-1, -2)).clamp(min=1e-30)                     # [G,nsg]
        Hsum = Hd.sum(-2)                                              # [G,nsg,gs]
    qmax = (1 << (u - 1)) - 1
    shared = None
    for it in range(max(1, efb_iters + 1)):
        r = (y - base).reshape(N, G, nsg, gs)                         # [N,G,nsg,gs]
        if Hblk is None:
            r_cont = r.mean(-1)                                        # [N,G,nsg] DC mean
        else:
            r_cont = torch.einsum("gpk,ngpk->ngp", Hsum, r) / denom    # [N,G,nsg]
        amax = r_cont.abs().amax(-1, keepdim=True).clamp(min=1e-30)   # per-block residual range
        d = torch.exp2(torch.floor(torch.log2(amax / max(qmax, 1))))  # [N,G,1] one E8M0 per block
        si = torch.round(r_cont / d).clamp(-qmax - 1, qmax)           # [N,G,nsg] u-bit signed int
        shared = (si.unsqueeze(-1) * d.unsqueeze(-1)).expand(N, G, nsg, gs).reshape(N, G, BLOCK)
        if it == efb_iters:
            break
        base = snap(y - shared)
    return ((base + shared) * s).reshape(x.shape).to(x.dtype)


def bits(eb, mb, u, gs, mxplus=False):
    b = (1 + eb + mb) + 8.0 / BLOCK + (5.0 / BLOCK if mxplus else 0.0)
    return b + (8.0 / BLOCK + float(u) / gs) if u else b


def _qsnr(x, xq):
    e = (x - xq).double().pow(2).mean()
    return 10.0 * torch.log10(x.double().pow(2).mean() / e.clamp(min=1e-45)).item()


def selftest():
    torch.manual_seed(0); dev = "cuda" if torch.cuda.is_available() else "cpu"
    N, K = 512, 4096; G = K // BLOCK
    W = torch.randn(N, K, device=dev) * 0.02
    print(f"=== gs-sweep quant selftest (device={dev}) ===")
    HI = torch.eye(BLOCK, device=dev).expand(G, BLOCK, BLOCK).contiguous()
    a = quant(W, 2, 1, 4, 32, Hblk=HI); b = quant(W, 2, 1, 4, 32, Hblk=None)
    print(f"  H=I == DC (gs32,u4): dmax={(a-b).abs().max().item():.2e}")
    print(f"  gs32,u4 DC vs original two_tier: dmax={(b-two_tier(W,2,1,4)).abs().max().item():.2e}")
    print(f"\n  recon QSNR vs gs  [E2M1+u4, DC]:")
    for gs in (32, 16, 8, 4, 2):
        print(f"    gs={gs:>2} ({bits(2,1,4,gs):.3f}b): {_qsnr(W, quant(W,2,1,4,gs)):.2f} dB")


def run_ppl():
    import torch.nn.functional as F
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from datasets import load_dataset
    DEV = "cuda"; MAXLEN, STRIDE, MAX_WINDOWS = 2048, 1024, 30
    LINEAR_KEYS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
    MODEL = os.environ.get("MSAQ_MODEL", "meta-llama/Llama-3.1-8B-Instruct")

    def is_target(n, m): return isinstance(m, torch.nn.Linear) and any(p in n for p in LINEAR_KEYS)

    @torch.no_grad()
    def ppl(model, ids):
        seq = ids.size(1); nll, ntok, prev, n = 0.0, 0, 0, 0
        for b in range(0, seq, STRIDE):
            e = min(b + MAXLEN, seq); trg = e - prev
            inp = ids[:, b:e].to(DEV); tgt = inp.clone(); tgt[:, :-trg] = -100
            nll += model(inp, labels=tgt).loss.double().item() * trg; ntok += trg; prev = e; n += 1
            if n >= MAX_WINDOWS or e == seq: break
        return torch.exp(torch.tensor(nll / ntok)).item()

    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16,
                                                 attn_implementation="sdpa").to(DEV).eval()
    def wt(sp):
        try: return load_dataset("wikitext", "wikitext-2-raw-v1", split=sp)
        except Exception: return load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split=sp)
    test_ids = tok("\n\n".join(t for t in wt("test")["text"] if t.strip()), return_tensors="pt").input_ids
    cal_ids = tok("\n\n".join(t for t in wt("train")["text"] if t.strip()), return_tensors="pt").input_ids
    targets = [(n, m) for n, m in model.named_modules() if is_target(n, m)]
    master = {m: m.weight.detach().to("cpu", copy=True) for _, m in targets}
    def restore():
        for _, m in targets: m.weight.data.copy_(master[m].to(DEV))

    bf = ppl(model, test_ids)
    print(f"wikitext-2: test {test_ids.size(1):,} tok | BF16 PPL = {bf:.4f} | gs sweep (one resid E8M0/block)\n", flush=True)
    print("collecting within-group Hessian on TRAIN (16x2048)...", flush=True)
    Hblk = collect_hessian(model, targets, cal_ids, DEV, n_windows=16, win=2048)
    print("done.\n", flush=True)

    def run(qfn):
        restore()
        for _, m in targets: m.weight.data.copy_(qfn(master[m].to(DEV), Hblk[m]).to(m.weight.dtype))
        p = ppl(model, test_ids); restore(); return p

    US = [int(x) for x in os.environ.get("US", "4").split(",")]
    print(f"{'cfg':>26} {'bits':>6} | {'PPL':>9} {'dPPL%':>8}", flush=True)
    if os.environ.get("SKIP_REFS", "0") != "1":
        for nm, b, qfn in [
            ("MXFP4(E2M1) native", bits(2,1,0,32), lambda w,H: quant(w,2,1,0,32)),
            ("MXINT8-MSAQ.efb",    6.000,          lambda w,H: msaq_mxint8_efb(w,3,4,2)),
            ("E2M3 native",        bits(2,3,0,32), lambda w,H: quant(w,2,3,0,32))]:
            p = run(qfn); print(f"{nm:>26} {b:>6.3f} | {p:>9.4f} {(p/bf-1)*100:>+7.2f}%", flush=True)
    for u in US:
        for mxp in (False, True):
            tag = "MX+ " if mxp else ""
            print(f"  --- {tag}E2M1+u{u} A-weighted, gs sweep ---", flush=True)
            for gs in (32, 16, 8, 4, 2):
                p = run(lambda w, H, gs=gs, mxp=mxp, u=u: quant(w, 2, 1, u, gs, Hblk=H, mxplus=mxp))
                print(f"{tag+'E2M1+u'+str(u)+' gs'+str(gs):>26} {bits(2,1,u,gs,mxp):>6.3f} | {p:>9.4f} {(p/bf-1)*100:>+7.2f}%", flush=True)


if __name__ == "__main__":
    if "--selftest" in sys.argv: selftest()
    else: run_ppl()
