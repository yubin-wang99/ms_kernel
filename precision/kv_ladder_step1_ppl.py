"""KV ladder — Step 1: rung PPL (no kernel). Does a two-tier fractional rung sit USABLE below the
nearest native KV rung — the win weight could not get (crushed by its 6.25b E2M3 wall, see
two_tier_results.md)? KV has no such wall and is the most quant-tolerant scope (scope_uvgs: S3 u4/gs2
+2.89%), so the 4.6–5.7b two-tier Pareto band should pay off here.

K/V axis split (kv_ladder_design.md §2):
  K — contraction = head_dim D. Key has per-channel(D) outliers -> H128 rotation (folded:
      k_deq = two_tier(k@H)@Hᵀ/128, QKᵀ preserved) THEN native MX D-blocking + residual along D.
  V — contraction = token T. T-blocking (transpose tokens to last) + residual along T. No rotation.

Residual: DC mean + MX+ (data-independent, the simplest realizable online form; the §4 A-weighted
optimum is not per-Q-realizable for write-once KV — escalate to calibration-static only if DC floors).
Carries the weight-proven config: u∈{2,3} (u4 never Pareto), gs swept {2,4,8,16,32}.

Run:
    HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 MSAQ_MODEL=NousResearch/Meta-Llama-3.1-8B \
        CUDA_VISIBLE_DEVICES=0 .venv/bin/python precision/kv_ladder_step1_ppl.py \
        > precision/kv_ladder_step1_llama31_8b.txt 2>&1
"""
import os, torch
import torch.nn.functional as F
from msaq_mxfp8_ppl import BLOCK
from two_tier_gs_sweep_ppl import quant, bits
# hadamard + msaq_signed inlined: their home modules (rot_qsnr/lightms_qsnr) glob a gated meta-llama
# snapshot at import time, which is absent when running the NousResearch mirror.

DEV = "cuda"; MAXLEN, STRIDE, MAX_WINDOWS = 2048, 1024, 30


def hadamard(n):                                           # Sylvester, unnormalized (±1), H Hᵀ = n I
    H = torch.ones(1, 1)
    while H.shape[0] < n:
        H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
    return H


def msaq_signed(x, u, mg):                                 # deployed sub-byte MSAQ (S3 vpack) anchor
    xf = x.reshape(-1, BLOCK).to(torch.float32)
    s_base = torch.exp2(torch.floor(torch.log2(xf.abs().amax(-1, keepdim=True).clamp(min=1e-30))) - 6.0)
    s_un = s_base * float(1 << u); q_max = (1 << (7 - u)) - 1
    x_un = torch.round(xf / s_un).clamp(-q_max, q_max) * s_un
    res = xf - x_un; s_min, s_max = -(1 << (u - 1)), (1 << (u - 1)) - 1
    res_avg = res.reshape(res.shape[0], -1, mg).mean(-1, keepdim=True).expand(-1, -1, mg).reshape(res.shape)
    shared = torch.round(res_avg / s_base).clamp(s_min, s_max)
    return (x_un + shared * s_base).reshape(x.shape)


HD = hadamard(128).to(DEV).to(torch.float32)               # unnormalized H128 (H Hᵀ = 128 I)


def quant_nd(x, eb, mb, u, gs, mxplus=False):
    """two-tier (DC residual) along the LAST axis; pads last dim to a multiple of 32. Reuses the
    weight-gate-validated 2D `quant` (Hblk=None -> DC mean)."""
    *lead, L = x.shape
    pad = (-L) % BLOCK
    xp = F.pad(x, (0, pad)) if pad else x
    q = quant(xp.reshape(-1, L + pad).contiguous().float(), eb, mb, u, gs, Hblk=None, mxplus=mxplus)
    q = q.reshape(*lead, L + pad)
    return q[..., :L] if pad else q


def q_K(k, eb, mb, u, gs, mxplus):                         # [B,H,S,D]: rotate D, two-tier D-block, fold back
    kr = k.float() @ HD
    return ((quant_nd(kr, eb, mb, u, gs, mxplus) @ HD.t()) / 128.0).to(k.dtype)


def q_V(v, eb, mb, u, gs, mxplus):                         # [B,H,S,D]: two-tier along token axis (T-block)
    vq = quant_nd(v.transpose(-1, -2), eb, mb, u, gs, mxplus)
    return vq.transpose(-1, -2).to(v.dtype)


_real = F.scaled_dot_product_attention
_C = {"on": False, "fn": None}
def _patch(q, k, v, *a, **kw):
    if _C["on"]:
        k = _C["fn"](k, "k"); v = _C["fn"](v, "v")
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


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from datasets import load_dataset
    MODEL = os.environ.get("MSAQ_MODEL", "meta-llama/Llama-3.1-8B-Instruct")
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16,
                                                 attn_implementation="sdpa").to(DEV).eval()
    try: _wt = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    except Exception: _wt = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    ids = tok("\n\n".join(t for t in _wt["text"] if t.strip()), return_tensors="pt").input_ids
    bf = ppl(model, ids)
    print(f"wikitext-2: {ids.size(1):,} tok | BF16 PPL = {bf:.4f} | scope=KV (K rot+D-block, V T-block)\n", flush=True)

    def run(fn):
        _C.update(on=True, fn=fn); p = ppl(model, ids); _C["on"] = False
        return p, (p / bf - 1) * 100

    # K and V get the same rung (allocation splits them later); KV apply = both.
    def kvfn(eb, mb, u, gs, mxp):
        return lambda t, w: (q_K if w == "k" else q_V)(t, eb, mb, u, gs, mxp)

    print(f"{'rung':>22} {'b/elem':>7} | {'PPL':>9} {'dPPL%':>8}", flush=True)
    # anchors: native KV rungs + the deployed sub-byte MSAQ (S3 vpack u4/gs16)
    anchors = [
        ("E2M1 native (FP4)", bits(2,1,0,32), kvfn(2,1,0,32,False)),
        ("E2M3 native (FP6)", bits(2,3,0,32), kvfn(2,3,0,32,False)),
        ("MSAQ vpack u4/gs16", None, lambda t, w: msaq_signed(t.float(), 4, 16).to(t.dtype)),
    ]
    for nm, b, fn in anchors:
        p, d = run(fn); bs = f"{b:.3f}" if b else "  ~5.5"
        print(f"{nm:>22} {bs:>7} | {p:>9.4f} {d:>+7.2f}%", flush=True)
    for u in (2, 3):
        print(f"  --- MX+ E2M1+u{u} (DC residual), gs sweep ---", flush=True)
        for gs in (32, 16, 8, 4, 2):
            p, d = run(kvfn(2, 1, u, gs, True))
            print(f"{f'MX+ E2M1+u{u} gs{gs}':>22} {bits(2,1,u,gs,True):>7.3f} | {p:>9.4f} {d:>+7.2f}%", flush=True)


if __name__ == "__main__":
    main()
