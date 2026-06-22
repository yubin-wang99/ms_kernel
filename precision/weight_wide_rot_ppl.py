"""Weight-scope u=4: does a FULL input-dim Hadamard (QuaRot-style) beat the per-32-block
H32 that HURTS u4 (rot_results.md §1)? wikitext-2 PPL, weight-only quant, block=32 msaq.

Linear Y = X·Wᵀ = (X·H)(W·H)ᵀ (orthonormal Hn=H/√h, Hn·Hnᵀ=I). We measure the QuaRot
weight rotation via the effective-dequant fold W_deq = msaq(W·Hn)·Hnᵀ (un-rotation exact,
so the activation rotation X·Hn folds for free — no online act-rotation needed). Hadamard
needs a power-of-2 size: K=4096 -> full H4096; down_proj K=14336=2^11·7 -> H2048 in 7 blocks
(largest pow2 dividing K). Methods: none / blockH32 (per-32-block, the §1 baseline that hurts)
/ wideH (full/wide input-dim). Criterion within 3% of BF16.
Run: CUDA_VISIBLE_DEVICES=0 python precision/weight_wide_rot_ppl.py > precision/weight_wide_rot_ppl.txt 2>&1
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from lightms_qsnr import msaq_signed
from rot_qsnr import hadamard                            # Sylvester ±1, H@H.T = n·I

MODEL = "meta-llama/Llama-3.1-8B-Instruct"; DEV = "cuda"
MAXLEN, STRIDE, MAX_WINDOWS = 2048, 1024, 30
LINEAR_KEYS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
CONFIGS = [(4, 8), (4, 4), (4, 2), (3, 8)]               # u=4 sweep + u3/mg8 sanity (rot helped u3)
_HN = {}                                                 # orthonormal Hadamard cache by size

def largest_pow2_div(K):
    h = 1
    while K % (h * 2) == 0: h *= 2
    return h

def hn(h):
    if h not in _HN: _HN[h] = (hadamard(h).to(DEV).to(torch.float32)) / (h ** 0.5)
    return _HN[h]

def rot_lastdim(W, h):                                   # rotate W's last dim in h-blocks by Hn_h
    OUT, K = W.shape
    return (W.reshape(OUT, K // h, h) @ hn(h)).reshape(OUT, K)

def unrot_lastdim(W, h):                                 # inverse rotate (orthonormal: Hn^T)
    OUT, K = W.shape
    return (W.reshape(OUT, K // h, h) @ hn(h).t()).reshape(OUT, K)

def msaq_none(W, u, mg):  return msaq_signed(W.float(), u, mg)
def msaq_rot(W, u, mg, h):                               # rotate K in h-blocks, msaq, un-rotate (fold)
    return unrot_lastdim(msaq_signed(rot_lastdim(W.float(), h), u, mg), h)
def msaq_block32(W, u, mg):  return msaq_rot(W, u, mg, 32)                        # rot_results §1
def msaq_wide(W, u, mg):     return msaq_rot(W, u, mg, largest_pow2_div(W.shape[1]))  # full/wide

METHODS = (("none", msaq_none), ("blockH32", msaq_block32), ("wideH", msaq_wide))

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
    master = {m: m.weight.detach().to("cpu", copy=True) for _, m in targets}
    def set_w(fn, u, mg):
        for _, m in targets:
            w0 = master[m].to(DEV)
            m.weight.data.copy_((w0 if fn is None else fn(w0, u, mg)).to(m.weight.dtype))

    set_w(None, 0, 0); bf = ppl(model, ids)
    print(f"wikitext-2: {ids.size(1):,} tokens | BF16 PPL = {bf:.4f} | weight-only, block=32 | within 3%", flush=True)
    print(f"K=4096 -> full H4096 ; down_proj K=14336 -> H2048 x7 blocks\n", flush=True)
    print(f"{'u':>2}/{'mg':<2} | {'none':>8} {'blockH32':>9} {'wideH':>8} | within3%(wideH)", flush=True)
    for u, mg in CONFIGS:
        r = {}
        for t, fn in METHODS:
            set_w(fn, u, mg); r[t] = (ppl(model, ids) / bf - 1) * 100
        ok = "OK" if r["wideH"] <= 3 else "FAIL"
        print(f"{u:>2}/{mg:<2} | {r['none']:>+7.2f}% {r['blockH32']:>+8.2f}% {r['wideH']:>+7.2f}% | {ok}", flush=True)
    set_w(None, 0, 0)
