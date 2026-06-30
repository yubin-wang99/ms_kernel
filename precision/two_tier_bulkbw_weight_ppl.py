"""Apply the residual-scale-bitwidth idea to WEIGHT. Sweep (u, gs, bulk_bw) with the proven weight
setup (A-weighted residual + MX+, per-group E{bulk_bw}M0 scale). Find the lowest weight bits within
3% PPL of BF16. Weight has less headroom than KV (the E2M3 wall), so this tests whether E2M0/E4M0
scales help weight uniform quant too.

bits = (1+eb+mb) + 8/32[base scale] + 5/32[MX+] + (bulk_bw+u)/gs.
Run:
    HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 MSAQ_MODEL=NousResearch/Meta-Llama-3.1-8B \
        CUDA_VISIBLE_DEVICES=0 .venv/bin/python precision/two_tier_bulkbw_weight_ppl.py \
        > precision/two_tier_bulkbw_weight_llama31_8b.txt 2>&1
"""
import os, torch
from two_tier_bulkbw_kv_ppl import quant_bw, bits
from two_tier_aware_ppl import collect_hessian

DEV = "cuda"; MAXLEN, STRIDE, MW = 2048, 1024, 30
KEYS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from datasets import load_dataset
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
    def wt(sp):
        try: return load_dataset("wikitext", "wikitext-2-raw-v1", split=sp)
        except Exception: return load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split=sp)
    test_ids = tok("\n\n".join(t for t in wt("test")["text"] if t.strip()), return_tensors="pt").input_ids
    cal_ids = tok("\n\n".join(t for t in wt("train")["text"] if t.strip()), return_tensors="pt").input_ids
    targets = [(n, m) for n, m in model.named_modules() if tgt(n, m)]
    master = {m: m.weight.detach().to("cpu", copy=True) for _, m in targets}
    def restore():
        for _, m in targets: m.weight.data.copy_(master[m].to(DEV))

    bf = ppl(model, test_ids)
    print(f"wikitext-2: test {test_ids.size(1):,} tok | BF16 PPL = {bf:.4f} | WEIGHT bulk_bw sweep (A-wgt+MX+)\n", flush=True)
    print("calibrating per-Linear Hessian on TRAIN (16x2048)...", flush=True)
    Hblk = collect_hessian(model, targets, cal_ids, DEV, n_windows=16, win=2048)
    print("done.\n", flush=True)

    rows = []
    print(f"{'u':>2} {'gs':>3} {'bw':>3} | {'b/elem':>7} | {'dPPL%':>7} {'<3%':>4}", flush=True)
    for u in (1, 2, 3, 4):
        for gs in (32, 16, 8, 4, 2):
            for bw in (2, 4):
                torch.cuda.empty_cache()
                for _, m in targets:
                    w = quant_bw(master[m].to(DEV), 2, 1, u, gs, bw, mxplus=True, Hblk=Hblk[m]).to(m.weight.dtype)
                    m.weight.data.copy_(w); del w
                p = ppl(model, test_ids); restore()
                d = (p / bf - 1) * 100; b = bits(2, 1, u, gs, bw, True)
                rows.append((b, d, u, gs, bw))
                print(f"{u:>2} {gs:>3} {bw:>3} | {b:>7.3f} | {d:>+6.2f}% {'OK' if d <= 3 else '':>4}", flush=True)
    ok = sorted([r for r in rows if r[1] <= 3.0])
    print("\n=== within 3%, sorted by bits (lowest first) ===", flush=True)
    for b, d, u, gs, bw in ok[:10]:
        print(f"  {b:.3f} b/elem : u{u}/gs{gs}/bw{bw}  {d:+.2f}%", flush=True)
    if ok:
        b, d, u, gs, bw = ok[0]
        print(f"\nLOWEST weight bits within 3%: {b:.3f} b/elem  (u{u}/gs{gs}/bulk_bw{bw}, {d:+.2f}%)", flush=True)
    else:
        print("\n(no config within 3% — weight has less headroom)", flush=True)


if __name__ == "__main__":
    main()
