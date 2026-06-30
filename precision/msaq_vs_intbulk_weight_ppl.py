"""Controlled comparison (base format held INT): does our bulk-scaled residual beat MXINT8-MSAQ when
both use an INT base? The earlier msaq_vs_bulk confounded the residual scheme with the base format
(MXINT8 base vs MXFP4 base). Here both are INT:
  MXINT8-MSAQ.efb : INT8 grid, low u bits SHARED over mg, on the fixed block scale.  (the deployed)
                    bits = (8-u) + u/mg + 8/32
  INT{n}+bulk     : clean MXINT{n} base + an ADDITIONAL u-bit shared residual on an ADAPTIVE per-group
                    E{bw}M0 (bulk) scale.  bits = n + 8/32 + (bw+u)/mg
So the only difference left is the residual scale scheme (MXINT8's shared-low-bits-of-INT8 vs a clean
low-bit INT base + adaptive bulk residual). Weight scope (deterministic), recon-L2 efb for both.
Run:
    HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 MSAQ_MODEL=NousResearch/Meta-Llama-3.1-8B \
        CUDA_VISIBLE_DEVICES=0 .venv/bin/python precision/msaq_vs_intbulk_weight_ppl.py \
        > precision/msaq_vs_intbulk_weight_llama31_8b.txt 2>&1
Selftest: .venv/bin/python precision/msaq_vs_intbulk_weight_ppl.py --selftest
"""
import os, sys, torch
from msaq_mxfp8_ppl import msaq_mxint8_efb, bits_mxint8, BLOCK


def quant_int_bulk(x, nbits, u, mg, bulk_bw, efb_iters=2):
    """MXINT{nbits} base + per-group E{bulk_bw}M0 bulk-scaled u-bit residual (DC, recon-L2 efb)."""
    *lead, K = x.shape
    assert K % BLOCK == 0 and BLOCK % mg == 0
    G = K // BLOCK; nsg = BLOCK // mg
    qmax = (1 << (nbits - 1)) - 1
    xf = x.to(torch.float32).reshape(-1, G, BLOCK)
    absmax = xf.abs().amax(-1, keepdim=True).clamp(min=1e-30)
    s = torch.exp2(torch.ceil(torch.log2(absmax / qmax)))          # E8M0 INT scale
    y = xf / s
    base = y.round().clamp(-qmax, qmax)
    if u > 0:
        uq = (1 << (u - 1)) - 1
        k_top = 0.0; k_lo = k_top - float((1 << bulk_bw) - 1)       # residual in [-.5,.5] -> anchor 0
        for it in range(max(1, efb_iters + 1)):
            r = (y - base).reshape(-1, G, nsg, mg)
            r_cont = r.mean(-1)
            kstar = torch.floor(torch.log2((r_cont.abs() / max(uq, 1)).clamp(min=1e-30)))
            d = torch.exp2(kstar.clamp(k_lo, k_top))
            si = torch.round(r_cont / d).clamp(-uq - 1, uq)
            shared = (si * d).unsqueeze(-1).expand(-1, G, nsg, mg).reshape(-1, G, BLOCK)
            if it == efb_iters: break
            base = (y - shared).round().clamp(-qmax, qmax)
        base = base + shared
    return (base * s).reshape(x.shape).to(x.dtype)


def bits_int_bulk(nbits, u, mg, bulk_bw): return nbits + 8.0 / BLOCK + (bulk_bw + u) / mg


def selftest():
    torch.manual_seed(0); dev = "cuda" if torch.cuda.is_available() else "cpu"
    W = torch.randn(4096, 4096, device=dev) * 0.02
    def qsnr(x, xq):
        e = (x - xq).double().pow(2).mean(); return 10 * torch.log10(x.double().pow(2).mean() / e.clamp(min=1e-45)).item()
    print("INT{n}+bulk vs MXINT8-MSAQ (synthetic QSNR):")
    for n, u, mg, bw in [(4, 2, 32, 2), (5, 2, 32, 2), (6, 2, 32, 2), (4, 3, 16, 4)]:
        print(f"  INT{n}+u{u}/mg{mg}/bw{bw} ({bits_int_bulk(n,u,mg,bw):.3f}b): {qsnr(W, quant_int_bulk(W,n,u,mg,bw)):.2f} dB")
    for u, mg in [(4, 16), (3, 16), (2, 8)]:
        print(f"  MXINT8-MSAQ u{u}/mg{mg} ({bits_mxint8(u,mg):.3f}b): {qsnr(W, msaq_mxint8_efb(W,u,mg)):.2f} dB")


def main():
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
    print(f"wikitext-2: {ids.size(1):,} tok | BF16 PPL = {bf:.4f} | INT base held | MXINT8-MSAQ vs INT+bulk\n", flush=True)
    pts = []
    def run(qfn):
        torch.cuda.empty_cache()
        for _, m in targets:
            w = qfn(master[m].to(DEV)).to(m.weight.dtype); m.weight.data.copy_(w); del w
        p = ppl(model, ids); restore(); return (p / bf - 1) * 100
    print("--- MXINT8-MSAQ.efb ---", flush=True)
    for u in (2, 3, 4):
        for mg in (2, 4, 8, 16, 32):
            d = run(lambda w, u=u, mg=mg: msaq_mxint8_efb(w, u, mg)); b = bits_mxint8(u, mg)
            pts.append((b, d, "MXINT8-MSAQ", f"u{u}/mg{mg}"))
            print(f"  u{u}/mg{mg:<2} {b:>6.3f}b {d:>+6.2f}%", flush=True)
    print("--- INT{n} + bulk residual ---", flush=True)
    for n in (4, 5, 6):
        for u in (2, 3):
            for mg in (8, 16, 32):
                for bw in (2, 4):
                    d = run(lambda w, n=n, u=u, mg=mg, bw=bw: quant_int_bulk(w, n, u, mg, bw))
                    b = bits_int_bulk(n, u, mg, bw)
                    pts.append((b, d, f"INT{n}+bulk", f"u{u}/mg{mg}/bw{bw}"))
                    print(f"  INT{n} u{u}/mg{mg}/bw{bw} {b:>6.3f}b {d:>+6.2f}%", flush=True)

    print("\n=== Pareto frontier (lowest dPPL at each bit budget; scheme tagged) ===", flush=True)
    front = []
    for b, d, sch, lab in sorted(pts):
        if not front or d < front[-1][1] - 1e-9:
            front.append((b, d, sch, lab)); print(f"  {b:>6.3f}b  {d:>+6.2f}%  {sch:>12}  {lab}", flush=True)
    print("\n=== matched-bit verdict (1/8-bit bins where both compete) ===", flush=True)
    bins = {}
    for b, d, sch, lab in pts: bins.setdefault(round(b * 8) / 8, []).append((d, sch))
    for key in sorted(bins):
        schemes = {s for _, s in bins[key]}
        has_int8 = any("MXINT8" in s for s in schemes); has_bulk = any("bulk" in s for s in schemes)
        if has_int8 and has_bulk:
            i8 = min(d for d, s in bins[key] if "MXINT8" in s); bk = min(d for d, s in bins[key] if "bulk" in s)
            print(f"  ~{key:.3f}b | MXINT8-MSAQ:{i8:+.2f}%  INT+bulk:{bk:+.2f}%  -> {'INT+bulk' if bk<i8 else 'MXINT8-MSAQ'}", flush=True)


if __name__ == "__main__":
    if "--selftest" in sys.argv: selftest()
    else: main()
