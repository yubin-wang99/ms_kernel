"""Small-sample wikitext PPL: does H32 rotation improve MSAQ weight-scope accuracy?
Replaces each Linear weight with msaq_signed(W) (plain) vs msaq_rot(W) (rotated-quant-dequant;
rotation is folded into the effective weight, so no online x-rotation needed in this accuracy test).
Run: CUDA_VISIBLE_DEVICES=0 python precision/rot_ppl.py > precision/rot_ppl.txt 2>&1
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from lightms_qsnr import msaq_signed
from rot_qsnr import msaq_rot

MODEL = "meta-llama/Llama-3.1-8B-Instruct"; DEV = "cuda"
MAXLEN, STRIDE, MAX_WINDOWS = 2048, 1024, 30
LINEAR_KEYS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")

def is_target(n, m):
    return isinstance(m, torch.nn.Linear) and any(p in n for p in LINEAR_KEYS)

@torch.no_grad()
def ppl(model, ids):
    seq = ids.size(1); nll, ntok, prev, n = 0.0, 0, 0, 0
    for b in range(0, seq, STRIDE):
        e = min(b + MAXLEN, seq); trg = e - prev
        inp = ids[:, b:e].to(DEV); tgt = inp.clone(); tgt[:, :-trg] = -100
        nll += model(inp, labels=tgt).loss.double().item() * trg; ntok += trg; prev = e; n += 1
        if n >= MAX_WINDOWS or e == seq: break
    return torch.exp(torch.tensor(nll / ntok)).item()

if __name__ == "__main__":
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16,
                                                 attn_implementation="sdpa").to(DEV).eval()
    ids = tok("\n\n".join(t for t in load_dataset("wikitext", "wikitext-2-raw-v1", split="test")["text"]
                          if t.strip()), return_tensors="pt").input_ids
    targets = [(n, m) for n, m in model.named_modules() if is_target(n, m)]
    master = {n: m.weight.detach().to("cpu", copy=True) for n, m in targets}
    bf = ppl(model, ids); print(f"BF16 PPL = {bf:.4f}\n", flush=True)

    def run(fn, u, mg):
        for n, m in targets: m.weight.data.copy_(fn(master[n].to(DEV), u, mg).to(m.weight.dtype))
        p = ppl(model, ids)
        for n, m in targets: m.weight.data.copy_(master[n].to(DEV))
        return p

    print(f"{'u':>2} {'mg':>3} | {'MSAQ %':>8} {'MSAQ+rot %':>11} | gain(pp)", flush=True)
    for u, mg in [(3, 8), (3, 4), (4, 8), (2, 8)]:
        pp = run(msaq_signed, u, mg); pr = run(msaq_rot, u, mg)
        gp, gr = (pp/bf-1)*100, (pr/bf-1)*100
        print(f"{u:>2} {mg:>3} | {gp:>+7.2f}% {gr:>+10.2f}% | {gp-gr:>+.2f}", flush=True)
