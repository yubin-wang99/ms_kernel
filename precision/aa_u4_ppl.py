"""AA-only (attention Q,K,V,P quantized = AA+KV) PPL at the u4 NIBBLE config — does the kernel-optimal
u4/gs2 (the only config that beats MXINT8 in latency) survive accuracy? Weights/linear-activations stay
BF16; only the attention matmuls (Q·Kᵀ, P·V) are MSAQ-quantized (Q,K,V and softmax P), reusing the manual
AA SDPA from aa_attn_ppl. wikitext-2, Llama-3.1-8B, teacher-forced, vs BF16. Criterion ref 3.5%.
Run: CUDA_VISIBLE_DEVICES=0 python precision/aa_u4_ppl.py [maxwin] > precision/aa_u4_ppl.txt 2>&1
"""
import sys, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import aa_attn_ppl as A                                  # _aa_sdpa already patched onto F.sdpa at import

A.MAX_WINDOWS = int(sys.argv[1]) if len(sys.argv) > 1 else 30
CONFIGS = [(4, 2), (4, 4), (4, 8), (3, 8), (2, 8)]       # u4 nibble sweep + u3/u2 references

if __name__ == "__main__":
    tok = AutoTokenizer.from_pretrained(A.MODEL)
    model = AutoModelForCausalLM.from_pretrained(A.MODEL, dtype=torch.bfloat16,
                                                 attn_implementation="sdpa").to(A.DEV).eval()
    ids = tok("\n\n".join(t for t in load_dataset("wikitext", "wikitext-2-raw-v1", split="test")["text"]
                          if t.strip()), return_tensors="pt").input_ids
    bf = A.ppl(model, ids)
    print(f"wikitext-2 | BF16 PPL = {bf:.4f} | max_win={A.MAX_WINDOWS} | AA-only (attention Q,K,V,P quant; "
          f"weights/act BF16)\n", flush=True)
    print(f"{'u/gs':>7} | {'PPL':>8} | {'% over BF16':>11} | {'b/elem':>7} | <=3.5%?", flush=True)
    for u, gs in CONFIGS:
        A._AA.update(on=True, u=u, mg=gs)
        p = A.ppl(model, ids)
        A._AA["on"] = False
        pct = (p / bf - 1) * 100
        print(f"u{u}/gs{gs:<3} | {p:>8.4f} | {pct:>+10.2f}% | {A.bits_per_elem(u,gs):>6.2f} | "
              f"{'OK' if pct <= 3.5 else 'FAIL'}", flush=True)
