"""Two-tier MSAQ, ACTIVATION-AWARE residual (§4 literal): choose the shared u-bit residual to
minimize DOWNSTREAM output error, not weight recon L2. Retry of the E2M1-rescue that the cheap
recon-L2 efb failed (two_tier_ppl.py: E2M1+u4 = +10.9% PPL, ~1.3pp over native MXFP4).

    R_bar[g,n] = argmin_r || sum_{k in g} A[:,k] R[k,n] - (sum_{k in g} A[:,k]) r ||^2
               = ( A_bar[:,g] . g_{g,n} ) / ( A_bar[:,g] . A_bar[:,g] )

Expanding over calibration samples m, this needs ONLY the within-group 32x32 Gram blocks
H_g = A[:,g]^T A[:,g] (cross-group terms cancel because A_bar is the in-group sum):
    denom[g]    = sum_{i,j in g} H_g[i,j]
    Hsum[g,k]   = sum_{i in g} H_g[i,k]           (k in g)
    numer[g,n]  = sum_{k in g} Hsum[g,k] R[k,n]
    r_cont[g,n] = numer[g,n] / denom[g]            -> snap to u-bit grid, efb re-snap native base
Cross-check: H=I (white activations) -> r_cont = mean -> reduces to two_tier's DC residual.

Calibrate H_g on wikitext-2 TRAIN (no test leakage); eval PPL on TEST.
Run:
    HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 MSAQ_MODEL=NousResearch/Meta-Llama-3.1-8B \
        CUDA_VISIBLE_DEVICES=0 .venv/bin/python precision/two_tier_aware_ppl.py \
        > precision/two_tier_aware_ppl_llama31_8b.txt 2>&1
Selftest (no model):  .venv/bin/python precision/two_tier_aware_ppl.py --selftest
"""
import os, sys, torch
from mxfp6_verify import _fp6_grid
from msaq_mxfp8_ppl import BLOCK
from two_tier_ppl import two_tier, _snap_grid


def two_tier_aware(x, eb, mb, u, Hblk, efb_iters=2):
    """A-weighted two-tier quant. x=[N,K] weight; Hblk=[G,32,32] within-group Gram (over the K axis).
    Shared residual per (group, output-channel) chosen by the §4 output-error optimum, base native."""
    assert x.dim() == 2, "weight [N,K]"
    N, K = x.shape; G = K // BLOCK
    grid = _fp6_grid(eb, mb)
    maxval = grid[-1].item()
    maxexp = ((1 << eb) - 1) - ((1 << (eb - 1)) - 1)
    xf = x.to(torch.float32).reshape(N, G, BLOCK)
    absmax = xf.abs().amax(-1, keepdim=True).clamp(min=1e-30)
    s = torch.exp2(torch.floor(torch.log2(absmax)) - float(maxexp))     # [N,G,1] base E8M0
    y = xf / s

    def snap_base(t):
        sgn = torch.sign(t); ay = t.abs().clamp(max=maxval)
        return sgn * _snap_grid(ay, grid)

    base = snap_base(y)
    if u == 0:
        return (base * s).reshape(x.shape).to(x.dtype)
    H = Hblk.to(torch.float32)                                          # [G,32,32]
    denom = H.sum((-1, -2)).clamp(min=1e-30)                            # [G]
    Hsum = H.sum(-2)                                                    # [G,32]  (sum over i in g)
    qmax = (1 << (u - 1)) - 1
    shared = None
    for it in range(max(1, efb_iters + 1)):
        r = (y - base)                                                  # residual in scaled domain [N,G,32]
        # numer[n,g] = sum_k Hsum[g,k] r[n,g,k]; r_cont[n,g] = numer/denom
        numer = torch.einsum("gk,ngk->ng", Hsum, r)                     # [N,G]
        r_cont = (numer / denom).unsqueeze(-1)                          # [N,G,1]
        d = torch.exp2(torch.floor(torch.log2(r_cont.abs().clamp(min=1e-30) / max(qmax, 1))))
        shared = torch.round(r_cont / d).clamp(-qmax - 1, qmax) * d     # [N,G,1] broadcast over block
        if it == efb_iters:
            break
        base = snap_base(y - shared)
    return ((base + shared) * s).reshape(x.shape).to(x.dtype)


# --------------------------------------------------------------------------- within-group Hessian
def collect_hessian(model, targets, ids, dev, n_windows=16, win=2048):
    """One BF16 forward over `ids` windows; hook each target Linear's input and accumulate the
    within-group 32x32 Gram block H_g = A[:,g]^T A[:,g] per layer. Returns {module: [G,32,32]}."""
    H = {m: None for _, m in targets}
    def hook(mod, inp, out):
        a = inp[0].detach().reshape(-1, inp[0].shape[-1]).to(torch.float32)   # [M, K]
        M, K = a.shape; ag = a.reshape(M, K // BLOCK, BLOCK)
        g = torch.einsum("mgi,mgj->gij", ag, ag)                              # [G,32,32]
        H[mod] = g if H[mod] is None else H[mod] + g
    handles = [m.register_forward_hook(hook) for _, m in targets]
    seq = ids.size(1)
    with torch.no_grad():
        for w in range(min(n_windows, (seq + win - 1) // win)):
            b = w * win; e = min(b + win, seq)
            if e - b < 8: break
            model(ids[:, b:e].to(dev))
    for h in handles: h.remove()
    return H


def _qsnr(x, xq):
    e = (x - xq).double().pow(2).mean()
    return 10.0 * torch.log10(x.double().pow(2).mean() / e.clamp(min=1e-45)).item()


def selftest():
    torch.manual_seed(0)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    N, K = 512, 4096; G = K // BLOCK
    W = torch.randn(N, K, device=dev) * 0.02
    # H=I -> aware reduces to DC-mean (two_tier)
    HI = torch.eye(BLOCK, device=dev).expand(G, BLOCK, BLOCK).contiguous()
    a = two_tier_aware(W, 2, 1, 4, HI, efb_iters=2)
    b = two_tier(W, 2, 1, 4, efb_iters=2)
    print(f"=== two_tier_aware selftest (device={dev}) ===")
    print(f"  H=I  == DC-mean two_tier (E2M1+u4): dmax={(a-b).abs().max().item():.2e}")
    # heterogeneous channel importance (per-column scale spans ~100x WITHIN each group): now the
    # output-optimal shared must weight high-importance channels, so aware should beat DC-mean.
    c = torch.exp2(4.0 * (torch.rand(K, device=dev) - 0.5) * 2)        # per-col scale ~ 2^[-4,4]
    A = torch.randn(2048, K, device=dev) * c
    Ag = A.reshape(2048, G, BLOCK); Hreal = torch.einsum("mgi,mgj->gij", Ag, Ag)
    def out_err(Wq): return (A @ (W - Wq).T).pow(2).mean().item()
    e_dc = out_err(two_tier(W, 2, 1, 4))
    e_aw = out_err(two_tier_aware(W, 2, 1, 4, Hreal))
    print(f"  output-error  DC-mean={e_dc:.4e}  A-weighted={e_aw:.4e}  (aware<dc: {e_aw < e_dc})")
    print(f"  recon-QSNR    DC-mean={_qsnr(W, two_tier(W,2,1,4)):.2f}dB  "
          f"A-weighted={_qsnr(W, two_tier_aware(W,2,1,4,Hreal)):.2f}dB  (aware trades recon for output)")


# --------------------------------------------------------------------------- PPL
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
    def wt(split):
        try: return load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
        except Exception: return load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split=split)
    test_ids = tok("\n\n".join(t for t in wt("test")["text"] if t.strip()), return_tensors="pt").input_ids
    cal_ids = tok("\n\n".join(t for t in wt("train")["text"] if t.strip()), return_tensors="pt").input_ids
    targets = [(n, m) for n, m in model.named_modules() if is_target(n, m)]
    master = {m: m.weight.detach().to("cpu", copy=True) for _, m in targets}
    def restore():
        for _, m in targets: m.weight.data.copy_(master[m].to(DEV))

    bf = ppl(model, test_ids)
    print(f"wikitext-2: test {test_ids.size(1):,} tok | BF16 PPL = {bf:.4f} | block={BLOCK} | A-weighted §4\n", flush=True)
    print("collecting within-group Hessian on wikitext-2 TRAIN (16x2048 tok)...", flush=True)
    Hblk = collect_hessian(model, targets, cal_ids, DEV, n_windows=16, win=2048)
    print("done. quantizing weight-only.\n", flush=True)

    # aware columns + recon-L2 reference (from two_tier) at the same bits
    COLS = [
        ("MXFP4(E2M1) native", 4.250, lambda w, H: two_tier(w, 2, 1, 0)),
        ("E2M1+u2 recon-L2",   4.562, lambda w, H: two_tier(w, 2, 1, 2)),
        ("E2M1+u2 A-weighted", 4.562, lambda w, H: two_tier_aware(w, 2, 1, 2, H)),
        ("E2M1+u4 recon-L2",   4.625, lambda w, H: two_tier(w, 2, 1, 4)),
        ("E2M1+u4 A-weighted", 4.625, lambda w, H: two_tier_aware(w, 2, 1, 4, H)),
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
