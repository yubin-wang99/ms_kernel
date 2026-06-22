"""two-level d2=2 + Hadamard rotation: does W+A become u=4 robust? wikitext-2 PPL.
block=32, L1=E8M0/32 + L2=2-bit microexp/2 (the d2=2 winner). Rotation = full input-dim
Hadamard on K (QuaRot): Y = X·Wᵀ = (X·Hn)(W·Hn)ᵀ, both act & weight quantized in the rotated
basis. K=4096 -> full H4096; down_proj K=14336 -> H2048 x7.

Per scope the rotation wiring differs:
  weight : weight = unrotate(Q(rotate(W)))   (fold; act stays BF16)            [act-rot folds away]
  act    : weight = rotate(W) (BF16);  fwd = Q(rotate(x)) · weightᵀ            (act quant in rot basis)
  wa     : weight = Q(rotate(W));      fwd = Q(rotate(x)) · weightᵀ            (both quant in rot basis)
Compares no-rot vs +rot at d2=2. Criterion within 3% of BF16.
Run: CUDA_VISIBLE_DEVICES=0 python precision/wa_rot_2lvl_ppl.py > precision/wa_rot_2lvl_ppl.txt 2>&1
"""
import torch, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from rot_qsnr import hadamard
from mx_2lvl_ppl import msaq_2lvl

MODEL = "meta-llama/Llama-3.1-8B-Instruct"; DEV = "cuda"
MAXLEN, STRIDE, MAX_WINDOWS = 2048, 1024, 30
LINEAR_KEYS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
D2 = 2
CONFIGS = [(4, 8), (4, 4), (4, 2)]
SCOPES = ("weight", "act", "wa")
_HN = {}

def Q(x, u, mg): return msaq_2lvl(x, u, mg, D2)
def largest_pow2_div(K):
    h = 1
    while K % (h * 2) == 0: h *= 2
    return h
def hn(h):
    if h not in _HN: _HN[h] = hadamard(h).to(DEV).to(torch.float32) / (h ** 0.5)
    return _HN[h]
def rot(t):                                              # rotate last dim in h-blocks (orthonormal)
    K = t.shape[-1]; h = largest_pow2_div(K); tf = t.float()
    return (tf.reshape(*t.shape[:-1], K // h, h) @ hn(h)).reshape(t.shape)
def unrot(t):
    K = t.shape[-1]; h = largest_pow2_div(K); tf = t.float()
    return (tf.reshape(*t.shape[:-1], K // h, h) @ hn(h).t()).reshape(t.shape)

def is_target(n, m):
    return isinstance(m, torch.nn.Linear) and any(p in n for p in LINEAR_KEYS)

# forward patch: optionally quantize (and rotate) the input activation
def patch_fwd(targets, u, mg, quant_act, rot_act):
    orig = {}
    for _, m in targets:
        orig[m] = m.forward
        def mk(mod):
            w, b = mod.weight, mod.bias
            def f(x):
                if quant_act: x = Q(rot(x) if rot_act else x, u, mg).to(w.dtype)
                return F.linear(x, w, b)
            return f
        m.forward = mk(m)
    return orig
def unpatch(orig):
    for m, f in orig.items(): m.forward = f

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

    def set_weight(scope, mode, u, mg):                  # set m.weight per scope/mode
        for _, m in targets:
            W = master[m].to(DEV).float()
            if scope == "weight":
                Wn = Q(W, u, mg) if mode == "norot" else unrot(Q(rot(W), u, mg))      # fold
            elif scope == "act":
                Wn = W if mode == "norot" else rot(W)                                  # BF16 (rotated)
            else:  # wa
                Wn = Q(W, u, mg) if mode == "norot" else Q(rot(W), u, mg)              # quant (rotated)
            m.weight.data.copy_(Wn.to(m.weight.dtype))
    def restore():
        for _, m in targets: m.weight.data.copy_(master[m].to(DEV))

    bf = ppl(model, ids)
    print(f"wikitext-2: {ids.size(1):,} tokens | BF16 PPL = {bf:.4f} | d2=2 two-level | within 3%", flush=True)
    print(f"rotation = full input-dim Hadamard (K=4096->H4096 ; down_proj->H2048 x7)\n", flush=True)

    def run(scope, mode, u, mg):
        set_weight(scope, mode, u, mg)
        quant_act = scope in ("act", "wa")
        oa = patch_fwd(targets, u, mg, quant_act, rot_act=(mode == "rot")) if quant_act else None
        p = ppl(model, ids)
        if oa: unpatch(oa)
        restore()
        return (p / bf - 1) * 100

    LABEL = {"weight": "weight", "act": "act", "wa": "weight+act"}
    for scope in SCOPES:
        print(f"==== scope = {LABEL[scope]} (d2=2) ====", flush=True)
        print(f"{'mode':>7} | " + " ".join(f"u{u}/mg{mg:<2}" for u, mg in CONFIGS) + " | within 3%?", flush=True)
        for mode in ("norot", "rot"):
            cells, oks = [], []
            for u, mg in CONFIGS:
                pct = run(scope, mode, u, mg); cells.append(f"{pct:>+6.2f}%"); oks.append("OK" if pct <= 3 else "FAIL")
            print(f"{mode:>7} | " + "  ".join(cells) + " | " + " ".join(oks), flush=True)
        print(flush=True)
