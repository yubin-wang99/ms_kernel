"""Small-sample wikitext PPL: does head-dim H128 rotation improve MSAQ KV accuracy?
Key has persistent CHANNEL outliers -> needs full head_dim rotation (H_D, not 32-block) to mix
channels. Effective dequant folds the rotation+unrotation (k_deq = msaq(k@H)@H^T/D), so QK^T is
preserved without online Q-rotation in this accuracy test (QuaRot V is also un-rotated via W_o fold).
Run: CUDA_VISIBLE_DEVICES=0 python precision/rot_kv_ppl.py > precision/rot_kv_ppl.txt 2>&1
"""
import sys, torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from lightms_qsnr import msaq_signed
from rot_qsnr import hadamard

MODEL = "meta-llama/Llama-3.1-8B-Instruct"; DEV = "cuda"
MAXLEN, STRIDE, MAX_WINDOWS = 2048, 1024, 30
HD = hadamard(128).to(DEV).to(torch.float32)            # head_dim=128 Hadamard

def rot_msaq(x, u, mg):                                   # rotate last dim by H128, msaq, un-rotate
    xf = x.float()
    xr = xf @ HD
    xq = msaq_signed(xr, u, mg)
    return ((xq @ HD.t()) / 128.0).to(x.dtype)

_real = F.scaled_dot_product_attention
_C = {"on": False, "u": 3, "mg": 8, "rot": False, "which": "kv"}
def _patch(q, k, v, *a, **kw):
    if _C["on"]:
        f = (lambda t: rot_msaq(t, _C["u"], _C["mg"])) if _C["rot"] else (lambda t: msaq_signed(t.float(), _C["u"], _C["mg"]).to(t.dtype))
        if "k" in _C["which"]: k = f(k)
        if "v" in _C["which"]: v = f(v)
    return _real(q, k, v, *a, **kw)
F.scaled_dot_product_attention = _patch

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
    bf = ppl(model, ids); print(f"BF16 PPL = {bf:.4f}\n", flush=True)
    def run(which, u, mg, rot):
        _C.update(on=True, u=u, mg=mg, rot=rot, which=which); p = ppl(model, ids); _C["on"] = False
        return (p/bf - 1) * 100
    for which in ("k", "v", "kv"):
        print(f"--- scope={which} ---", flush=True)
        print(f"{'u':>2} {'mg':>3} | {'MSAQ %':>8} {'MSAQ+rot %':>11} | gain(pp)", flush=True)
        for u, mg in [(3, 8), (4, 8), (4, 2)]:
            gp = run(which, u, mg, False); gr = run(which, u, mg, True)
            print(f"{u:>2} {mg:>3} | {gp:>+7.2f}% {gr:>+10.2f}% | {gp-gr:>+.2f}", flush=True)
