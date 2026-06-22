"""End-to-end wikitext PPL for the ONLINE Q/K-rotation path (what the kernel actually does).

The prior KV-rotation PPL (rot_kv_ppl.py) used the EFFECTIVE-DEQUANT FOLD:
    k_deq = msaq(K@H) @ H^T / D          # H unnormalized, D=2^7 folds into E8M0 exactly
so Q was never rotated online and no √D scale entered the quantizer. The deployed kernel instead
does the TRUE online rotation with an ORTHONORMAL Hadamard Hn = H/√D:
    Q_rot = Q@Hn   (kv_decode_wide_kernel prologue, MS_KV_QROT=1)
    K_stored = msaq(K@Hn)   (kv_append_rot)        scores = (Q@Hn)·msaq(K@Hn)^T
Since msaq's E8M0 scale is per-32-block absmax and 1/√128 is NOT a power of two, the online path is
NOT bit-identical to the fold — the √D factor can shift each block's E8M0 exponent. This verifies the
online path reproduces the fold's accuracy win end-to-end (and beats no-rotation).

wikitext-2, Llama-3.1-8B-Instruct, BF16. Run:
  CUDA_VISIBLE_DEVICES=0 python precision/rot_qrot_ppl.py > precision/rot_qrot_ppl.txt 2>&1
"""
import torch, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from lightms_qsnr import msaq_signed
from rot_qsnr import hadamard

MODEL = "meta-llama/Llama-3.1-8B-Instruct"; DEV = "cuda"
MAXLEN, STRIDE, MAX_WINDOWS = 2048, 1024, 30
HD = hadamard(128).to(DEV).to(torch.float32)          # 128x128, ±1, H@H^T = 128 I
HN = HD / (128.0 ** 0.5)                                # orthonormal: Hn@Hn^T = I

# mode: none | fold | online  (scope: which of k/v get quantized; Q rotated only in online)
_C = {"on": False, "u": 4, "mg": 8, "mode": "none", "which": "k"}

def _q_msaq(t):  return msaq_signed(t.float(), _C["u"], _C["mg"]).to(t.dtype)
def _k_fold(t):  return ((msaq_signed(t.float() @ HD, _C["u"], _C["mg"]) @ HD.t()) / 128.0).to(t.dtype)
def _k_online(t):return msaq_signed(t.float() @ HN, _C["u"], _C["mg"]).to(t.dtype)   # K in rotated basis

_real = F.scaled_dot_product_attention
def _patch(q, k, v, *a, **kw):
    if _C["on"]:
        m = _C["mode"]
        if m == "online":
            q = (q.float() @ HN).to(q.dtype)                       # mirror Q (online prologue)
            if "k" in _C["which"]: k = _k_online(k)
        else:
            kf = _k_fold if m == "fold" else _q_msaq               # fold un-rotates; none = plain msaq
            if "k" in _C["which"]: k = kf(k)
        if "v" in _C["which"]: v = _q_msaq(v)                      # V never rotated (accuracy-irrelevant)
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
    def run(which, u, mg, mode):
        _C.update(on=True, u=u, mg=mg, mode=mode, which=which); p = ppl(model, ids); _C["on"] = False
        return p, (p / bf - 1) * 100
    for which in ("k", "kv"):
        print(f"--- scope={which} (Q rotated online; V never rotated) ---", flush=True)
        print(f"{'u':>2}{'mg':>3} | {'none %':>8} {'fold %':>8} {'online %':>9} | online-fold(pp)", flush=True)
        for u, mg in [(4, 8), (3, 8), (4, 2)]:
            _, gn = run(which, u, mg, "none")
            _, gf = run(which, u, mg, "fold")
            _, go = run(which, u, mg, "online")
            print(f"{u:>2}{mg:>3} | {gn:>+7.2f}% {gf:>+7.2f}% {go:>+8.2f}% | {go-gf:>+.2f}", flush=True)
        print(flush=True)
