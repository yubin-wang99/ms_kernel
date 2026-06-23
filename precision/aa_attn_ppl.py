"""W+A + attention activation×activation (AA): how does the robust (u,gs) drop when the
prefill-attention internal matmuls (Q·Kᵀ and P·V — both operands are ACTIVATIONS) are also
MSAQ-quantized, on top of W+A (weight + linear-input activation)?

AA = manual SDPA that quantizes Q,K (for QKᵀ) and P=softmax(scores), V (for P·V), each along its
last dim (32-blocks); P's key axis is zero-padded to a multiple of 32 for the block reshape. Compared
to plain W+A (fused SDPA, attention in full precision). Find the most aggressive (u,gs) within 3.5% PPL.
Llama-3.1-8B, wikitext-2, teacher-forced sliding-window PPL.
Run: CUDA_VISIBLE_DEVICES=0 python precision/aa_attn_ppl.py [maxwin] > precision/aa_attn_ppl.txt 2>&1
"""
import sys, math, time, torch, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from lightms_qsnr import msaq_signed

MODEL = "meta-llama/Llama-3.1-8B-Instruct"; DEV = "cuda"
MAXLEN, STRIDE = 2048, 1024
MAX_WINDOWS = int(sys.argv[1]) if len(sys.argv) > 1 else 15
TOL = 3.5
LINEAR_KEYS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
GS = [2, 4, 8, 16, 32]
Q = msaq_signed

def bits_per_elem(u, gs):                  # MSAQ-s footprint: upper + shared + E8M0 scale
    UB = 32 * (8 - u) // 8; SB = ((32 // gs) * u + 7) // 8
    return (UB + SB + 1) * 8.0 / 32.0

def qlast(t, u, mg):                      # MSAQ along last dim, padding to a multiple of 32
    D = t.shape[-1]
    if D % 32 == 0: return Q(t, u, mg).to(t.dtype)
    pad = (32 - D % 32)
    tp = F.pad(t, (0, pad))
    return Q(tp, u, mg)[..., :D].to(t.dtype)

def is_target(n, m):
    return isinstance(m, torch.nn.Linear) and any(p in n for p in LINEAR_KEYS)

# ---- W+A: weight in-place + linear-input activation patch -------------------
def patch_act(targets, u, mg):
    orig = {}
    for _, m in targets:
        orig[m] = m.forward
        def mk(mod):
            w, b = mod.weight, mod.bias
            def f(x): return F.linear(Q(x, u, mg).to(w.dtype), w, b)
            return f
        m.forward = mk(m)
    return orig
def unpatch_act(o):
    for m, f in o.items(): m.forward = f

# ---- AA: manual SDPA quantizing Q,K,V,P (activation×activation in attention) ----
_real_sdpa = F.scaled_dot_product_attention
_AA = {"on": False, "u": 0, "mg": 0}
def _aa_sdpa(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, enable_gqa=False, **kw):
    if not _AA["on"]:
        return _real_sdpa(q, k, v, attn_mask=attn_mask, dropout_p=dropout_p,
                          is_causal=is_causal, scale=scale, enable_gqa=enable_gqa, **kw)
    u, mg = _AA["u"], _AA["mg"]
    B, H, Lq, D = q.shape; Hk = k.shape[1]
    if Hk != H:                                            # GQA -> repeat KV to Q heads
        rep = H // Hk; k = k.repeat_interleave(rep, 1); v = v.repeat_interleave(rep, 1)
    qd = qlast(q, u, mg).float(); kd = qlast(k, u, mg).float(); vd = qlast(v, u, mg).float()
    sc = scale if scale is not None else 1.0 / math.sqrt(D)
    scores = (qd @ kd.transpose(-1, -2)) * sc             # Q·Kᵀ  (quantized activations)
    Lk = k.shape[2]
    if is_causal or (attn_mask is None):
        cm = torch.triu(torch.ones(Lq, Lk, device=q.device, dtype=torch.bool), diagonal=1 + Lk - Lq)
        scores = scores.masked_fill(cm, float("-inf"))
    if attn_mask is not None:
        scores = scores + attn_mask if attn_mask.dtype != torch.bool else scores.masked_fill(~attn_mask, float("-inf"))
    P = torch.softmax(scores, dim=-1)
    Pd = qlast(P, u, mg).float()                          # quantize the softmax probs
    out = (Pd @ vd).to(q.dtype)                           # P·V  (quantized activations)
    return out
F.scaled_dot_product_attention = _aa_sdpa

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
    def restore():
        for _, m in targets: m.weight.data.copy_(master[m].to(DEV))
    def quant_w(u, mg):
        for _, m in targets: m.weight.data.copy_(Q(master[m].to(DEV), u, mg).to(m.weight.dtype))

    t0 = time.time(); bf = ppl(model, ids)
    print(f"wikitext-2 | BF16 PPL = {bf:.4f} | max_win={MAX_WINDOWS} | <=+{TOL}% | baseline {time.time()-t0:.0f}s\n", flush=True)

    def run(aa, u, mg):
        restore(); quant_w(u, mg)
        oa = patch_act(targets, u, mg)
        if aa: _AA.update(on=True, u=u, mg=mg)
        p = ppl(model, ids)
        unpatch_act(oa); _AA["on"] = False; restore()
        return (p / bf - 1) * 100

    SCOPES = [("W+A (attn fp)", False), ("W+A+AA (attn act*act quant)", True)]
    best = {}
    for name, aa in SCOPES:
        print(f"=== {name} ===", flush=True); passes = []
        for u in (2, 3):
            for gs in GS:
                pct = run(aa, u, gs); ok = pct <= TOL
                print(f"   u{u}/gs{gs:<2}: {pct:>+6.2f}%  {'OK' if ok else 'FAIL'}  ({bits_per_elem(u,gs):.2f} b/elem)", flush=True)
                if ok: passes.append((u, gs, pct))
                else: break
        mostagg = min(passes, key=lambda r: bits_per_elem(r[0], r[1])) if passes else None  # fewest bits/elem
        best[name] = mostagg
        print(f"   -> max-aggressive robust: {'u%d/gs%d (%+.2f%%, %.2f b/elem)' % (mostagg+(bits_per_elem(mostagg[0],mostagg[1]),)) if mostagg else 'none'}\n", flush=True)

    print("=" * 56 + "\nSUMMARY — max-aggressive robust (u,gs), <=+3.5% PPL", flush=True)
    for name, _ in SCOPES:
        r = best[name]
        print(f"  {name:30s}: {'u%d/gs%d (%+.2f%%, %.2f b/elem)' % (r+(bits_per_elem(r[0],r[1]),)) if r else 'none'}", flush=True)
    print(f"\ntotal {time.time()-t0:.0f}s", flush=True)
