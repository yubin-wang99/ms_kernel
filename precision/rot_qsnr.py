"""Hadamard-rotation accuracy check for MSAQ (weight-tensor QSNR on real Llama weights).

Block-Hadamard H32 (unnormalized ±1, H@H.T = 32*I) applied to each MSAQ 32-block BEFORE the
E8M0 scale, then MSAQ-quantize, then un-rotate. Tests the hypothesis: rotation spreads in-block
outliers (Laplace->Gaussian) so the E8M0 absmax scale fits all 32 elems -> better upper + residual
codes. The 32=2^5 factor folds into E8M0 exactly (no extra rounding -- power-of-2 block).

QSNR(dB) = 10 log10( sum||W||^2 / sum||W - Wq||^2 ), aggregated over Linear weights, with vs without rot.
Run: CUDA_VISIBLE_DEVICES=0 python precision/rot_qsnr.py
"""
import glob, os, math, torch
from safetensors import safe_open
from lightms_qsnr import msaq_signed, BLOCK   # certified MSAQ-signed block quant

DEV = "cuda" if torch.cuda.is_available() else "cpu"
SNAP = glob.glob(os.path.expanduser(
    "~/.cache/huggingface/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/*"))[0]
LINEAR_KEYS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


def hadamard(n):
    H = torch.ones(1, 1)
    while H.shape[0] < n:
        H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
    return H                                  # n x n, +-1, H@H.T = n*I


H32 = hadamard(BLOCK).to(DEV)                 # 32x32


def msaq_rot(W, u, mg):
    xb = W.reshape(-1, BLOCK).float()
    xr = xb @ H32                             # rotate each 32-block (unnormalized)
    xr_hat = msaq_signed(xr, u, mg)           # MSAQ quant on rotated block
    xb_hat = (xr_hat @ H32.t()) / float(BLOCK)  # un-rotate (H H^T = 32 I)
    return xb_hat.reshape(W.shape)


def _weights():
    for f in sorted(glob.glob(os.path.join(SNAP, "*.safetensors"))):
        with safe_open(f, framework="pt", device=DEV) as g:
            for k in g.keys():
                if k.endswith(".weight") and any(p in k for p in LINEAR_KEYS):
                    yield k, g.get_tensor(k)


if __name__ == "__main__":
    configs = [(3, 8), (3, 4), (2, 2), (4, 8), (2, 8)]
    acc = {c: {"plain": [0.0, 0.0], "rot": [0.0, 0.0]} for c in configs}
    n = 0
    for name, W in _weights():
        n += 1
        Wf = W.float()
        sig = Wf.pow(2).sum().item()
        for c in configs:
            u, mg = c
            wp = msaq_signed(Wf, u, mg)
            wr = msaq_rot(W, u, mg)
            acc[c]["plain"][0] += sig; acc[c]["plain"][1] += (Wf - wp.float()).pow(2).sum().item()
            acc[c]["rot"][0]   += sig; acc[c]["rot"][1]   += (Wf - wr.float()).pow(2).sum().item()
        del W, Wf
    db = lambda a: 10.0 * math.log10(a[0] / max(a[1], 1e-12))
    print(f"Llama-3.1-8B weight QSNR (dB), {n} Linear tensors, block={BLOCK}")
    print(f"{'u':>2} {'mg':>3} | {'MSAQ':>8} {'MSAQ+rot':>9} | {'gain(dB)':>8}")
    for c in configs:
        u, mg = c
        qp, qr = db(acc[c]["plain"]), db(acc[c]["rot"])
        print(f"{u:>2} {mg:>3} | {qp:>8.3f} {qr:>9.3f} | {qr - qp:>+8.3f}")
