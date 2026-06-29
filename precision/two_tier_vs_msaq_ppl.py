"""two-tier (additive shared) vs MXFP-MSAQ (exponent-scaled shared) at MATCHED (per-element format,
shared bits u, group size gs). Both = per-element E{eb}M{m_pe} + a u-bit value shared over gs. The
ONLY structural difference:
  MXFP-MSAQ:  rec_i = upper_i + sh_g · 2^(ee_i − m_pe)   — shared scaled by EACH element's exponent ee_i
              (a sub-LSB mantissa extension). Rides per-element exponents -> no separate residual scale,
              but u ≤ mb and the shared is entangled with per-element exponents (no clean base+corr split).
  two-tier:   rec_i = base_i + sh_g · d_blk             — shared on ONE per-block E8M0 d_blk, uniform.
              Additive residual: u independent of mb, base stays a clean native MX element -> the
              Y=AŴ+(ĀR̄) 1/32-correction GEMM factorizes; costs +8/32 b/elem for its own scale.

Matched at per-element E2M1 (m_pe=1): two-tier = quant(2,1,u,gs); MXFP-MSAQ = msaq_mxfp8(u,gs,2,1+u).
Both recon-L2 (efb coordinate descent). bits: two-tier 4.5+u/gs, msaq 4.25+u/gs (no resid scale).
    --qsnr : synthetic-weight QSNR (instant, no model). PPL otherwise (weight scope).
"""
import os, sys, torch
from msaq_mxfp8_ppl import msaq_mxfp8, bits_mxfp8, BLOCK
from two_tier_gs_sweep_ppl import quant, bits as bits_tt

# matched per-element E2M1; sweep u (shared bits) x gs. msaq base = E2M{1+u} sharing u -> per-elem E2M1.
def tt(x, u, gs):   return quant(x, 2, 1, u, gs, Hblk=None, mxplus=False)       # additive
def ms(x, u, gs):   return msaq_mxfp8(x, u, gs, 2, 1 + u, efb_iters=2)          # exponent-scaled
def b_tt(u, gs):    return bits_tt(2, 1, u, gs)
def b_ms(u, gs):    return bits_mxfp8(2, 1 + u, u, gs)


def _qsnr(x, xq):
    e = (x - xq).double().pow(2).mean()
    return 10.0 * torch.log10(x.double().pow(2).mean() / e.clamp(min=1e-45)).item()


def qsnr_main():
    torch.manual_seed(0); dev = "cuda" if torch.cuda.is_available() else "cpu"
    W = torch.randn(4096, 4096, device=dev) * 0.02                              # Gaussian
    Wt = (torch.randn(4096, 4096, device=dev) * 0.02) * (1 + 3 * torch.rand(4096, 1, device=dev) ** 4)  # heavy-tail
    sgn = torch.sign(torch.randn(4096, 4096, device=dev))
    Ws = sgn * torch.exp2(-10.0 * torch.rand(4096, 4096, device=dev))           # hi intra-block dyn-range
    print("MATCHED per-element E2M1 — QSNR(dB). TT=two-tier(additive)  MS=MXFP-MSAQ(exp-scaled)\n")
    for nm, X in [("Gaussian", W), ("heavy-tail", Wt), ("hi-dyn-range", Ws)]:
        print(f"== {nm} ==")
        print(f"{'u':>2} {'gs':>3} | {'TT bits':>7} {'MS bits':>7} | {'QSNR TT':>8} {'QSNR MS':>8} | {'MS-TT':>6}")
        for u in (1, 2, 3):
            for gs in (32, 8, 2):
                qt = _qsnr(X, tt(X, u, gs)); qm = _qsnr(X, ms(X, u, gs))
                print(f"{u:>2} {gs:>3} | {b_tt(u,gs):>7.3f} {b_ms(u,gs):>7.3f} | {qt:>8.2f} {qm:>8.2f} | {qm-qt:>+6.2f}")
        print()


def ppl_main():
    import torch.nn.functional as F
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from datasets import load_dataset
    DEV = "cuda"; MAXLEN, STRIDE, MW = 2048, 1024, 30
    KEYS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
    MODEL = os.environ.get("MSAQ_MODEL", "meta-llama/Llama-3.1-8B-Instruct")
    tgt = lambda n, m: isinstance(m, torch.nn.Linear) and any(p in n for p in KEYS)

    @torch.no_grad()
    def ppl(model, ids):
        seq = ids.size(1); nll, ntok, prev, n = 0.0, 0, 0, 0
        for b in range(0, seq, STRIDE):
            e = min(b + MAXLEN, seq); trg = e - prev
            inp = ids[:, b:e].to(DEV); t = inp.clone(); t[:, :-trg] = -100
            nll += model(inp, labels=t).loss.double().item() * trg; ntok += trg; prev = e; n += 1
            if n >= MW or e == seq: break
        return torch.exp(torch.tensor(nll / ntok)).item()

    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, attn_implementation="sdpa").to(DEV).eval()
    try: wt = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    except Exception: wt = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    ids = tok("\n\n".join(t for t in wt["text"] if t.strip()), return_tensors="pt").input_ids
    targets = [(n, m) for n, m in model.named_modules() if tgt(n, m)]
    master = {m: m.weight.detach().to("cpu", copy=True) for _, m in targets}
    def restore():
        for _, m in targets: m.weight.data.copy_(master[m].to(DEV))
    bf = ppl(model, ids)
    print(f"wikitext-2: {ids.size(1):,} tok | BF16 PPL = {bf:.4f} | MATCHED per-element E2M1, weight scope\n", flush=True)
    print(f"{'u':>2} {'gs':>3} | {'TT bits':>7} {'TT dPPL':>8} | {'MS bits':>7} {'MS dPPL':>8} | {'winner':>8}", flush=True)
    def run(qfn):
        restore()
        for _, m in targets: m.weight.data.copy_(qfn(master[m].to(DEV)).to(m.weight.dtype))
        p = ppl(model, ids); restore(); return (p / bf - 1) * 100
    for u in (1, 2, 3):
        for gs in (32, 8, 2):
            dt = run(lambda w, u=u, gs=gs: tt(w, u, gs)); dm = run(lambda w, u=u, gs=gs: ms(w, u, gs))
            win = "TT" if dt < dm else "MS"
            print(f"{u:>2} {gs:>3} | {b_tt(u,gs):>7.3f} {dt:>+7.2f}% | {b_ms(u,gs):>7.3f} {dm:>+7.2f}% | {win:>8}", flush=True)


if __name__ == "__main__":
    if "--qsnr" in sys.argv: qsnr_main()
    else: ppl_main()
