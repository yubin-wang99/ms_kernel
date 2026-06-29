"""KV ladder — Step 1 ablations, isolating two factors before the allocation step:
  (A) K-rotation: does the folded H128 rotation actually help native D-blocking of Key (channel
      outliers)?  scope=k only (V stays bf16), rot OFF vs ON.
  (B) MX+ contribution, separately on rotated-K vs V: rotation spreads Key's D-outliers, possibly
      stealing MX+'s target (design open Q). Compare DC-only vs MX+ on K(rot) and on V(no rot).
Representative rung: E2M1+u3 gs32 (the Step-1 sweet spot, 4.75b w/ MX+). DC residual, no calibration.
Run:
    HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 MSAQ_MODEL=NousResearch/Meta-Llama-3.1-8B \
        CUDA_VISIBLE_DEVICES=0 .venv/bin/python precision/kv_ladder_ablation_ppl.py \
        > precision/kv_ladder_ablation_llama31_8b.txt 2>&1
"""
import os, torch
import torch.nn.functional as F
from msaq_mxfp8_ppl import BLOCK
from two_tier_gs_sweep_ppl import quant, bits

DEV = "cuda"; MAXLEN, STRIDE, MAX_WINDOWS = 2048, 1024, 30


def hadamard(n):
    H = torch.ones(1, 1)
    while H.shape[0] < n:
        H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
    return H
HD = hadamard(128).to(DEV).to(torch.float32)


def quant_nd(x, eb, mb, u, gs, mxplus=False):
    *lead, L = x.shape
    pad = (-L) % BLOCK
    xp = F.pad(x, (0, pad)) if pad else x
    q = quant(xp.reshape(-1, L + pad).contiguous().float(), eb, mb, u, gs, Hblk=None, mxplus=mxplus)
    q = q.reshape(*lead, L + pad)
    return q[..., :L] if pad else q


def q_K(k, eb, mb, u, gs, mxp, rot):
    kf = k.float()
    if rot:
        return ((quant_nd(kf @ HD, eb, mb, u, gs, mxp) @ HD.t()) / 128.0).to(k.dtype)
    return quant_nd(kf, eb, mb, u, gs, mxp).to(k.dtype)            # D-block, NO rotation


def q_V(v, eb, mb, u, gs, mxp):
    return quant_nd(v.transpose(-1, -2), eb, mb, u, gs, mxp).transpose(-1, -2).to(v.dtype)


_real = F.scaled_dot_product_attention
_C = {"on": False, "scope": "k", "rot": True, "cfg": (2, 1, 3, 32, True)}
def _patch(q, k, v, *a, **kw):
    if _C["on"]:
        eb, mb, u, gs, mxp = _C["cfg"]
        if "k" in _C["scope"]: k = q_K(k, eb, mb, u, gs, mxp, _C["rot"])
        if "v" in _C["scope"]: v = q_V(v, eb, mb, u, gs, mxp)
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
    print(f"wikitext-2: {ids.size(1):,} tok | BF16 PPL = {bf:.4f}\n", flush=True)

    def run(scope, rot, cfg):
        _C.update(on=True, scope=scope, rot=rot, cfg=cfg); p = ppl(model, ids); _C["on"] = False
        return p, (p / bf - 1) * 100

    RUNG = (2, 1, 3, 32, True)                                     # E2M1+u3 gs32 MX+  (4.75b)
    DC   = (2, 1, 3, 32, False)                                    # same, DC only (no MX+)
    NAT  = (2, 1, 0, 32, False)                                    # native E2M1
    E2M3 = (2, 3, 0, 32, False)

    print("=== (A) K-rotation (scope=k only) ===", flush=True)
    print(f"{'cfg':>20} | {'rot OFF':>9} {'rot ON':>9} | {'gain pp':>8}", flush=True)
    for nm, cfg in [("E2M1 native", NAT), ("E2M3 native", E2M3), ("MX+ E2M1+u3 gs32", RUNG)]:
        _, off = run("k", False, cfg); _, on = run("k", True, cfg)
        print(f"{nm:>20} | {off:>+8.2f}% {on:>+8.2f}% | {off-on:>+7.2f}", flush=True)

    print("\n=== (B) MX+ contribution (DC vs MX+), rotated-K vs V ===", flush=True)
    print(f"{'where':>20} | {'DC only':>9} {'MX+':>9} | {'gain pp':>8}", flush=True)
    # K (scope=k, rot ON)
    _, kdc = run("k", True, DC); _, kmx = run("k", True, RUNG)
    print(f"{'K (rot, 4.75b)':>20} | {kdc:>+8.2f}% {kmx:>+8.2f}% | {kdc-kmx:>+7.2f}", flush=True)
    # V (scope=v)
    _, vdc = run("v", True, DC); _, vmx = run("v", True, RUNG)
    print(f"{'V (no rot)':>20} | {vdc:>+8.2f}% {vmx:>+8.2f}% | {vdc-vmx:>+7.2f}", flush=True)


if __name__ == "__main__":
    main()
