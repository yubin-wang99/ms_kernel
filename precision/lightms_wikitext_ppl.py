"""Confirm light-MS accuracy on wikitext-2-raw PPL (standard sliding window).
weight-only quantization; BF16 / naive-MS / light-MS / MSAQ-signed. Criterion: within
3% of BF16 PPL. Per-config in-place weight quant from a CPU master (no per-window re-quant).
Run: CUDA_VISIBLE_DEVICES=0 python precision/lightms_wikitext_ppl.py
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from lightms_qsnr import naive_ms, light_ms, msaq_signed

MODEL = "meta-llama/Llama-3.1-8B-Instruct"
DEV = "cuda"
MAXLEN, STRIDE, MAX_WINDOWS = 2048, 1024, 40
LINEAR_KEYS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")

def is_target(n, m):
    return isinstance(m, torch.nn.Linear) and any(p in n for p in LINEAR_KEYS)

@torch.no_grad()
def ppl(model, ids):                              # standard HF sliding-window recipe
    seq = ids.size(1)
    nll_sum, ntok = 0.0, 0
    prev_end, n_done = 0, 0
    for begin in range(0, seq, STRIDE):
        end = min(begin + MAXLEN, seq)
        trg = end - prev_end                      # new tokens scored this window
        inp = ids[:, begin:end].to(DEV)
        tgt = inp.clone(); tgt[:, :-trg] = -100
        out = model(inp, labels=tgt)
        nll_sum += out.loss.double().item() * trg
        ntok += trg
        prev_end = end; n_done += 1
        if n_done >= MAX_WINDOWS or end == seq:
            break
    return torch.exp(torch.tensor(nll_sum / ntok)).item()

if __name__ == "__main__":
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(DEV).eval()
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(t for t in ds["text"] if t.strip())
    ids = tok(text, return_tensors="pt").input_ids
    print(f"wikitext-2-raw test: {ids.size(1):,} tokens; window={MAXLEN} stride={STRIDE} max_win={MAX_WINDOWS}")

    targets = [(n, m) for n, m in model.named_modules() if is_target(n, m)]
    master = {n: m.weight.detach().to("cpu", copy=True) for n, m in targets}
    print(f"target linears: {len(targets)} (CPU master cached)\n")

    def set_weights(fn, u, mg):
        for n, m in targets:
            w0 = master[n].to(DEV)
            m.weight.data.copy_(w0 if fn is None else fn(w0, u, mg).to(m.weight.dtype))

    set_weights(None, 0, 0)
    bf = ppl(model, ids)
    print(f"BF16 PPL = {bf:.4f}\n")
    print(f"{'method':>8} {'u':>2} {'mg':>3} | {'PPL':>8} | {'% over BF16':>11} | within 3%?")
    for u, mg in [(4, 8), (4, 4), (4, 2), (3, 4), (2, 2)]:
        for tag, fn in (("naive", naive_ms), ("light", light_ms), ("msaq", msaq_signed)):
            set_weights(fn, u, mg)
            p = ppl(model, ids)
            pct = (p / bf - 1) * 100
            print(f"{tag:>8} {u:>2} {mg:>3} | {p:>8.4f} | {pct:>+10.2f}% | {'OK' if abs(pct) <= 3 else 'FAIL'}")
        print()
