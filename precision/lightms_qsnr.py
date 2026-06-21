"""light-MS vs MSAQ-signed — weight-tensor QSNR on real Llama-3.1-8B weights.

MSAQ_signed (current):  residual = x - x_unshared (FP) -> FP group-mean -> quantize to u-bit signed int.
light-MS:               residual -> quantize each elem to u-bit signed int FIRST -> INTEGER group-mean.
  (stored format identical: (7-u)-bit unshared + u-bit signed shared; only the averaging moves FP->INT.)

QSNR(dB) = 10*log10( sum||W||^2 / sum||W - dequant(W)||^2 ) aggregated over all Linear weights.
Run: CUDA_VISIBLE_DEVICES=0 python precision/lightms_qsnr.py
"""
import glob, json, os, torch
from safetensors import safe_open

BLOCK = 32
SNAP = glob.glob(os.path.expanduser(
    "~/.cache/huggingface/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/*"))[0]
DEV = "cuda" if torch.cuda.is_available() else "cpu"

# ---- shared front-end (block-wise, last dim = BLOCK) -------------------------
def _blocks(x):
    return x.reshape(-1, BLOCK).to(torch.float32)

def _mxint8_scale(xf):                                  # OCP E8M0 for 8-bit signed
    max_abs = xf.abs().amax(-1, keepdim=True).clamp(min=1e-30)
    return torch.exp2(torch.floor(torch.log2(max_abs)) - 6.0)

def _unshared(xf, u):
    s_base = _mxint8_scale(xf)
    s_un = s_base * float(1 << u)
    q_max = (1 << (7 - u)) - 1
    q_un = torch.round(xf / s_un).clamp(-q_max, q_max)
    return q_un * s_un, s_base                           # x_unshared (FP), step

def msaq_signed(x, u, mg):                               # current: FP-mean then quantize
    xf = _blocks(x)
    x_un, step = _unshared(xf, u)
    res = xf - x_un
    s_min, s_max = -(1 << (u - 1)), (1 << (u - 1)) - 1
    res_avg = res.reshape(res.shape[0], -1, mg).mean(-1, keepdim=True).expand(-1, -1, mg).reshape(res.shape)
    shared = torch.round(res_avg / step).clamp(s_min, s_max)
    return (x_un + shared * step).reshape(x.shape)

def light_ms(x, u, mg):                                  # light: quantize then INT-mean
    xf = _blocks(x)
    x_un, step = _unshared(xf, u)
    res = xf - x_un
    s_min, s_max = -(1 << (u - 1)), (1 << (u - 1)) - 1
    res_int = torch.round(res / step).clamp(s_min, s_max)             # per-elem u-bit signed int
    grp = res_int.reshape(res_int.shape[0], -1, mg)
    shared = torch.round(grp.mean(-1, keepdim=True)).clamp(s_min, s_max).expand(-1, -1, mg).reshape(res.shape)
    return (x_un + shared * step).reshape(x.shape)        # INTEGER round-to-nearest mean

def naive_ms(x, u, mg):                                  # MXINT8 quant, share low-u bits (unsigned int mean)
    xf = _blocks(x)                                      # = single_level_mantissa_sharing (ssnf round_mean)
    s = _mxint8_scale(xf)
    q = torch.round(xf / s).clamp(-127, 127).to(torch.int32)
    patt = q & 0xFF                                      # two's-complement 8-bit pattern (unsigned)
    mask = (1 << u) - 1
    low = patt & mask
    grp = low.reshape(low.shape[0], -1, mg)
    shared_low = ((grp.sum(-1, keepdim=True) + mg // 2) // mg).clamp(0, mask).expand(-1, -1, mg).reshape(low.shape)
    patt_shared = (patt - low) + shared_low              # upper bits unchanged, low replaced
    q_shared = torch.where(patt_shared >= 128, patt_shared - 256, patt_shared)
    return (q_shared.float() * s).reshape(x.shape)

def qsnr_acc(W, fn, u, mg):
    Wq = fn(W, u, mg)
    sig = W.float().pow(2).sum().item()
    noi = (W.float() - Wq.float()).pow(2).sum().item()
    return sig, noi

# ---- iterate Llama linear weights -------------------------------------------
LINEAR_KEYS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
def linear_weights():
    for f in sorted(glob.glob(os.path.join(SNAP, "*.safetensors"))):
        with safe_open(f, framework="pt", device=DEV) as g:
            for k in g.keys():
                if k.endswith(".weight") and any(p in k for p in LINEAR_KEYS):
                    yield k, g.get_tensor(k)

if __name__ == "__main__":
    configs = [(u, mg) for u in (2, 3, 4) for mg in (2, 4, 8)]
    methods = (("naive", naive_ms), ("light", light_ms), ("msaq", msaq_signed))
    print(f"Llama-3.1-8B-Instruct weight QSNR (dB). block={BLOCK}, {DEV}")
    print(f"{'u':>2} {'mg':>3} | {'naive-MS':>9} {'light-MS':>9} {'MSAQ':>9} | "
          f"{'light-naive':>11} {'msaq-light':>10}")
    acc = {c: {m: [0.0, 0.0] for m, _ in methods} for c in configs}
    n = 0
    for name, W in linear_weights():
        n += 1
        for c in configs:
            u, mg = c
            for tag, fn in methods:
                s, no = qsnr_acc(W, fn, u, mg)
                acc[c][tag][0] += s; acc[c][tag][1] += no
        del W
    print(f"  ({n} linear weight tensors)\n")
    db = lambda a: 10.0 * torch.log10(torch.tensor(a[0] / max(a[1], 1e-12))).item()
    for c in configs:
        u, mg = c
        qn, ql, qm = db(acc[c]["naive"]), db(acc[c]["light"]), db(acc[c]["msaq"])
        print(f"{u:>2} {mg:>3} | {qn:>9.3f} {ql:>9.3f} {qm:>9.3f} | {ql-qn:>+11.3f} {qm-ql:>+10.3f}")
