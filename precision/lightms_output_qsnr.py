"""Part 2: output-QSNR (measure_qsnr-style) + short PPL on real Llama-3.1-8B forward.
weight-only quantization; naive-MS / light-MS / MSAQ-signed. Confirms the weight-QSNR
ordering at the output level (what PPL tracks).
Run: CUDA_VISIBLE_DEVICES=0 python precision/lightms_output_qsnr.py
"""
import torch, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from lightms_qsnr import naive_ms, light_ms, msaq_signed   # reuse the quantizers

MODEL = "meta-llama/Llama-3.1-8B-Instruct"
DEV = "cuda"
LINEAR_KEYS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")

TEXT = (  # fixed eval text (relative comparison vs BF16 on the same tokens)
    "The history of artificial intelligence began in antiquity, with myths, stories "
    "and rumors of artificial beings endowed with intelligence or consciousness by "
    "master craftsmen. Modern machine learning grew out of the quest for artificial "
    "intelligence. As a scientific discipline, machine learning developed from the "
    "study of pattern recognition and computational learning theory. In 2012, deep "
    "learning began to dominate benchmark after benchmark, and large language models "
    "later reshaped natural language processing entirely. Quantization reduces the "
    "numerical precision of weights and activations to lower memory and compute cost, "
    "ideally without harming model accuracy. " * 12
)

def is_target(name):
    return any(p in name for p in LINEAR_KEYS)

@torch.no_grad()
def output_qsnr(model, inputs, fn, u, mg):
    sig = [0.0]; noi = [0.0]
    def hook(mod, inp, out):
        x = inp[0].detach(); w = mod.weight.detach(); b = mod.bias.detach() if mod.bias is not None else None
        y_ref = F.linear(x, w, b).float()
        y_q = F.linear(x, fn(w, u, mg).to(w.dtype), b).float()
        sig[0] += y_ref.pow(2).sum().item(); noi[0] += (y_ref - y_q).pow(2).sum().item()
    hs = [m.register_forward_hook(hook) for n, m in model.named_modules()
          if isinstance(m, torch.nn.Linear) and is_target(n)]
    model(**inputs)
    for h in hs: h.remove()
    return 10.0 * torch.log10(torch.tensor(sig[0] / max(noi[0], 1e-12))).item()

@torch.no_grad()
def ppl_weight_only(model, ids, fn, u, mg):
    """Patch each target Linear.forward to use quantized weight; compute NLL over ids."""
    orig = {}
    for n, m in model.named_modules():
        if isinstance(m, torch.nn.Linear) and is_target(n):
            orig[m] = m.forward
            def mk(mod):
                def f(x):
                    return F.linear(x, fn(mod.weight, u, mg).to(mod.weight.dtype),
                                    mod.bias)
                return f
            m.forward = mk(m)
    try:
        out = model(ids, labels=ids)
        return torch.exp(out.loss).item()
    finally:
        for m, f in orig.items(): m.forward = f

if __name__ == "__main__":
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16).to(DEV).eval()
    enc = tok(TEXT, return_tensors="pt").to(DEV)
    ids = enc["input_ids"][:, :1024]
    print(f"loaded {MODEL}; eval tokens = {ids.shape[1]}\n")

    methods = (("naive", naive_ms), ("light", light_ms), ("msaq", msaq_signed))
    cfgs = [(2, 2), (2, 8), (4, 2), (4, 8)]
    print("=== output-QSNR (dB), weight-only ===")
    print(f"{'u':>2} {'mg':>3} | {'naive':>8} {'light':>8} {'MSAQ':>8} | {'light-naive':>11} {'msaq-light':>10}")
    for u, mg in cfgs:
        q = {t: output_qsnr(model, {"input_ids": ids}, fn, u, mg) for t, fn in methods}
        print(f"{u:>2} {mg:>3} | {q['naive']:>8.3f} {q['light']:>8.3f} {q['msaq']:>8.3f} | "
              f"{q['light']-q['naive']:>+11.3f} {q['msaq']-q['light']:>+10.3f}")

    print("\n=== short PPL (weight-only) vs BF16, criterion = within 3% of BF16 ===")
    ppl_bf = torch.exp(model(ids, labels=ids).loss).item()
    print(f"BF16 PPL = {ppl_bf:.4f}")
    print(f"{'u':>2} {'mg':>3} | {'naive':>8} {'light':>8} {'MSAQ':>8}   (% over BF16)")
    for u, mg in [(4, 2), (4, 8), (2, 2)]:
        row = {}
        for t, fn in methods:
            p = ppl_weight_only(model, ids, fn, u, mg); row[t] = p
        pct = lambda p: f"{(p/ppl_bf-1)*100:+.2f}%"
        print(f"{u:>2} {mg:>3} | {row['naive']:>8.4f} {row['light']:>8.4f} {row['msaq']:>8.4f}   "
              f"naive {pct(row['naive'])} / light {pct(row['light'])} / msaq {pct(row['msaq'])}")
