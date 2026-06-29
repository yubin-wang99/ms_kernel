"""MX+ (block-max outlier rescue) + A-weighted shared residual — the §3 "realistic E2M1-rescue".
MX+ repurposes the block-max element's exponent bits as mantissa (it always sits at the top exponent
maxexp, so the exponent field is redundant for it): E2M1's outlier goes 1->(1+eb)=3 mantissa bits.
The A-weighted shared residual (§4) then corrects the remaining 31 elements' DC. Orthogonal error
dimensions: MX+ = the one outlier, residual = the 31-element DC bias.

Tests whether the last weight lever crosses the 3% gate, after single-shared-scalar saturated at
+7.9% (two_tier_aware_ppl.py). Honest caveat: weight blocks are less outlier-dominated than KV/act,
so MX+ may help weight less than its KV framing suggests.

Run:
    HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 MSAQ_MODEL=NousResearch/Meta-Llama-3.1-8B \
        CUDA_VISIBLE_DEVICES=0 .venv/bin/python precision/two_tier_mxplus_ppl.py \
        > precision/two_tier_mxplus_ppl_llama31_8b.txt 2>&1
Selftest:  .venv/bin/python precision/two_tier_mxplus_ppl.py --selftest
"""
import os, sys, torch
from mxfp6_verify import _fp6_grid
from msaq_mxfp8_ppl import BLOCK
from two_tier_ppl import two_tier, _snap_grid
from two_tier_aware_ppl import two_tier_aware, collect_hessian


def _mxplus_snap(y, eb, mb, maxexp, grid, maxval):
    """Native E{eb}M{mb} snap, then re-quantize the per-block argmax element with (mb+eb) mantissa
    bits at the top exponent (exponent field reused as mantissa). y=[...,32] scaled domain."""
    sgn = torch.sign(y); ay = y.abs()
    base = sgn * _snap_grid(ay.clamp(max=maxval), grid)
    idx = ay.argmax(dim=-1, keepdim=True)
    amax = torch.gather(ay, -1, idx); asgn = torch.gather(sgn, -1, idx)
    qf = 2.0 ** (maxexp - (mb + eb))                                   # finer quantum for the outlier
    maxval_p = (2.0 - 2.0 ** (-(mb + eb))) * (2.0 ** maxexp)
    fine = asgn * (torch.round(amax / qf).clamp(max=maxval_p / qf)) * qf
    return base.scatter(-1, idx, fine)


def two_tier_mxplus(x, eb, mb, u, Hblk=None, efb_iters=2):
    """MX+ base + (optional) A-weighted shared residual. u=0 -> MX+ only. Hblk=[G,32,32] required
    when u>0 (A-weighted residual on the MX+ base, correcting the 31 non-outlier elements)."""
    assert x.dim() == 2
    N, K = x.shape; G = K // BLOCK
    grid = _fp6_grid(eb, mb); maxval = grid[-1].item()
    maxexp = ((1 << eb) - 1) - ((1 << (eb - 1)) - 1)
    xf = x.to(torch.float32).reshape(N, G, BLOCK)
    absmax = xf.abs().amax(-1, keepdim=True).clamp(min=1e-30)
    s = torch.exp2(torch.floor(torch.log2(absmax)) - float(maxexp))
    y = xf / s
    base = _mxplus_snap(y, eb, mb, maxexp, grid, maxval)
    if u == 0:
        return (base * s).reshape(x.shape).to(x.dtype)
    H = Hblk.to(torch.float32)
    denom = H.sum((-1, -2)).clamp(min=1e-30); Hsum = H.sum(-2)
    qmax = (1 << (u - 1)) - 1
    shared = None
    for it in range(max(1, efb_iters + 1)):
        r = (y - base)
        numer = torch.einsum("gk,ngk->ng", Hsum, r)
        r_cont = (numer / denom).unsqueeze(-1)
        d = torch.exp2(torch.floor(torch.log2(r_cont.abs().clamp(min=1e-30) / max(qmax, 1))))
        shared = torch.round(r_cont / d).clamp(-qmax - 1, qmax) * d
        if it == efb_iters:
            break
        base = _mxplus_snap(y - shared, eb, mb, maxexp, grid, maxval)
    return ((base + shared) * s).reshape(x.shape).to(x.dtype)


def bits_mxplus(eb, mb, u):
    b = (1 + eb + mb) + 8.0 / BLOCK + 5.0 / BLOCK                      # +5/32 outlier index (upper bnd)
    return b + (u + 8.0) / BLOCK if u else b


def _qsnr(x, xq):
    e = (x - xq).double().pow(2).mean()
    return 10.0 * torch.log10(x.double().pow(2).mean() / e.clamp(min=1e-45)).item()


def selftest():
    torch.manual_seed(0)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    N, K = 512, 4096; G = K // BLOCK
    # blocks WITH a planted outlier (one big element per 32) — where MX+ should shine
    W = torch.randn(N, K, device=dev) * 0.02
    Wo = W.clone().reshape(N, G, BLOCK)
    Wo[:, :, 0] += torch.sign(torch.randn(N, G, device=dev)) * 0.2     # 10x outlier at pos 0
    Wo = Wo.reshape(N, K)
    print(f"=== two_tier_mxplus selftest (device={dev}) ===")
    print(f"  native E2M1 QSNR : W={_qsnr(W, two_tier(W,2,1,0)):.2f}  Wo(outlier)={_qsnr(Wo, two_tier(Wo,2,1,0)):.2f}")
    print(f"  MX+    E2M1 QSNR : W={_qsnr(W, two_tier_mxplus(W,2,1,0)):.2f}  Wo(outlier)={_qsnr(Wo, two_tier_mxplus(Wo,2,1,0)):.2f}")
    print("  (MX+ should beat native especially on Wo)")
    # MX+ + residual >= MX+ alone (residual only helps)
    A = torch.randn(2048, K, device=dev) * torch.exp2(4.0*(torch.rand(K,device=dev)-0.5)*2)
    Ag = A.reshape(2048, G, BLOCK); H = torch.einsum("mgi,mgj->gij", Ag, Ag)
    q0 = _qsnr(Wo, two_tier_mxplus(Wo, 2, 1, 0))
    q2 = _qsnr(Wo, two_tier_mxplus(Wo, 2, 1, 2, H))
    print(f"  MX+ vs MX++u2(A-weighted) recon QSNR on Wo: {q0:.2f} -> {q2:.2f}")


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
    print(f"wikitext-2: test {test_ids.size(1):,} tok | BF16 PPL = {bf:.4f} | block={BLOCK} | MX+ + residual\n", flush=True)
    print("collecting within-group Hessian on TRAIN (16x2048)...", flush=True)
    Hblk = collect_hessian(model, targets, cal_ids, DEV, n_windows=16, win=2048)
    print("done.\n", flush=True)

    COLS = [
        ("MXFP4(E2M1) native",   4.250, lambda w, H: two_tier(w, 2, 1, 0)),
        ("E2M1+u2 A-weighted",   4.562, lambda w, H: two_tier_aware(w, 2, 1, 2, H)),
        ("MX+ E2M1 (no resid)",  bits_mxplus(2,1,0), lambda w, H: two_tier_mxplus(w, 2, 1, 0)),
        ("MX+ E2M1 +u2 A-wgt",   bits_mxplus(2,1,2), lambda w, H: two_tier_mxplus(w, 2, 1, 2, H)),
        ("MX+ E2M1 +u4 A-wgt",   bits_mxplus(2,1,4), lambda w, H: two_tier_mxplus(w, 2, 1, 4, H)),
        ("E2M3 native (ref)",    6.250, lambda w, H: two_tier(w, 2, 3, 0)),
    ]
    print(f"{'cfg':>22} {'bits':>6} | {'PPL':>9} {'dPPL%':>8}", flush=True)
    for nm, b, qfn in COLS:
        restore()
        for _, m in targets:
            m.weight.data.copy_(qfn(master[m].to(DEV), Hblk[m]).to(m.weight.dtype))
        p = ppl(model, test_ids); restore()
        print(f"{nm:>22} {b:>6.3f} | {p:>9.4f} {(p/bf-1)*100:>+7.2f}%", flush=True)


if __name__ == "__main__":
    if "--selftest" in sys.argv: selftest()
    else: run_ppl()
