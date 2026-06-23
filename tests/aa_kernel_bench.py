"""AA (activation×activation) quantized-attention kernel — documented-negative bench.

Prefill: quantized self-attention = qk_wmma (Q·Kᵀ) -> causal softmax -> pv_wmma (P·V), with K,V in
MSAQ planes and (AA) Q,P also MSAQ-quantized. Decode (AA+KV): kv_decode_attention (K,V MSAQ cache) with
Q (AA) MSAQ-quantized. Measured vs bf16 SDPA and the MXINT8 baseline.

Expected (no-fake-win, per change.md Phase 37/47/49): prefill attention is O(L²D) COMPUTE-bound, and the
tensor-core path must unpack the sub-byte operand to a bf16 tile (staging) -> the sub-byte DRAM saving is
cancelled -> loses to bf16. AA adds the Q,P quantization tax on top -> strictly worse. The win stays the
memory-bound DECODE KV-read. This bench MEASURES that negative.
Run: CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python tests/aa_kernel_bench.py
"""
import sys, os, time, numpy as np, torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "precision"))
from ms_lib.pack import pack_kv, pack_kv_mxint8, dequant_weight  # noqa
from ms_lib import ops; assert ops.available()
from lightms_qsnr import msaq_signed
OPS = torch.ops.msaq
def cuda(a): return torch.from_numpy(a).cuda()

def _t(fn, it=50):
    s = torch.cuda.Event(True); e = torch.cuda.Event(True)
    torch.cuda.synchronize(); s.record()
    for _ in range(it): fn()
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) / it

def qmsaq(t, u, mg):                                # MSAQ-round an activation to bf16 (the AA quant)
    return msaq_signed(t.float(), u, mg).to(torch.bfloat16)


def prefill(H, L, D, u=4, gs=2, seed=0):
    """quantized causal self-attention, H heads, M=Lk=L."""
    rng = np.random.default_rng(seed); nb = D // 32
    K = (rng.standard_normal((H, L, D)) * 0.5).astype(np.float32)
    V = (rng.standard_normal((H, L, D)) * 0.5).astype(np.float32)
    Q = cuda((rng.standard_normal((H, L, D)) * 0.5).astype(np.float32)).to(torch.bfloat16)
    pk, pv = pack_kv(K, u, gs), pack_kv(V, u, gs)
    ks, ku, kh = (cuda(pk[k]) for k in ("scale_exp", "upper", "shared"))
    vs, vu, vh = (cuda(pv[k]) for k in ("scale_exp", "upper", "shared"))
    cmask = torch.triu(torch.ones(L, L, device="cuda", dtype=torch.bool), 1)
    def softmax_causal(sc):                          # sc [H,L,L] fp32
        return torch.softmax(sc.masked_fill(cmask, float("-inf")), -1).to(torch.bfloat16)
    Qa = qmsaq(Q, u, gs)                             # AA: pre-quantized Q (format conversion, once)
    def msaq_1op():                                  # K,V quantized; Q,P bf16
        sc = OPS.qk_wmma(Q, ks, ku, kh, H, L, D, L, nb, u, gs)
        return OPS.pv_wmma(softmax_causal(sc), vs, vu, vh, H, L, D, L, nb, u, gs)
    def msaq_aa():                                   # AA: Q,K,V,P all quantized
        sc = OPS.qk_wmma(Qa, ks, ku, kh, H, L, D, L, nb, u, gs)
        P = qmsaq(softmax_causal(sc).float() if False else softmax_causal(sc), u, gs)
        return OPS.pv_wmma(P, vs, vu, vh, H, L, D, L, nb, u, gs)
    Kb16, Vb16 = cuda(K).to(torch.bfloat16), cuda(V).to(torch.bfloat16)   # precompute (no H2D in the loop)
    def bf16():                                      # full-precision flash SDPA, causal
        return torch.nn.functional.scaled_dot_product_attention(Q, Kb16, Vb16, is_causal=True)
    # correctness (AA vs fp ref)
    out = msaq_aa().float().cpu().numpy()
    Kd0 = dequant_weight(pk["_per"][0]); Vd0 = dequant_weight(pv["_per"][0])
    sc0 = (Q[0].float().cpu().numpy() @ Kd0.T) / np.sqrt(D)
    sc0 = np.where(np.triu(np.ones((L, L)), 1) > 0, -1e30, sc0)
    p0 = np.exp(sc0 - sc0.max(1, keepdims=True)); p0 /= p0.sum(1, keepdims=True)
    rel = np.linalg.norm(out[0] - p0 @ Vd0) / (np.linalg.norm(p0 @ Vd0) + 1e-9)
    for f in (msaq_1op, msaq_aa, bf16): f()
    t1 = min(_t(msaq_1op), _t(msaq_1op)); taa = min(_t(msaq_aa), _t(msaq_aa)); tb = min(_t(bf16), _t(bf16))
    tax = taa - t1                                   # the AA P-quant tax (Q pre-quantized)
    print(f"  L={L:>4} u{u}/gs{gs} | bf16 {tb*1e3:7.1f}us | qk+pv(1op) {t1*1e3:7.1f}us ({t1/tb:.1f}x) | "
          f"AA {taa*1e3:7.1f}us ({taa/tb:.1f}x) | P-quant tax {tax*1e3:.0f}us | AA relerr {rel:.1e}")


def decode(Hq, Hkv, Lk, D, u=4, gs=2, seed=0):
    """decode AA+KV: K,V MSAQ cache + Q (AA) quantized. vs MXINT8 KV and bf16."""
    rng = np.random.default_rng(seed); nb = D // 32
    K = (rng.standard_normal((Hkv, Lk, D)) * 0.5).astype(np.float32)
    V = (rng.standard_normal((Hkv, Lk, D)) * 0.5).astype(np.float32)
    q = cuda((rng.standard_normal((Hq, D)) * 0.5).astype(np.float32)).to(torch.bfloat16)
    pk, pv = pack_kv(K, u, gs), pack_kv(V, u, gs)
    ks, ku, kh = (cuda(pk[k]) for k in ("scale_exp", "upper", "shared"))
    vs, vu, vh = (cuda(pv[k]) for k in ("scale_exp", "upper", "shared"))
    xk, xv = pack_kv_mxint8(K), pack_kv_mxint8(V)
    kxs, kxq = cuda(xk["scale_exp"]), cuda(xk["qweight"]); vxs, vxq = cuda(xv["scale_exp"]), cuda(xv["qweight"])
    qa = qmsaq(q, u, gs)
    g = Hq // Hkv
    Kb = cuda(K).to(torch.bfloat16).repeat_interleave(g, 0); Vb = cuda(V).to(torch.bfloat16).repeat_interleave(g, 0)
    msaq_aa = lambda: OPS.kv_decode_attention(qa, ks, ku, kh, vs, vu, vh, Hq, Hkv, Lk, D, nb, u, gs, Lk)
    mx = lambda: OPS.mxint8_kv_decode(q, kxs, kxq, vxs, vxq, Hq, Hkv, Lk, D, nb, Lk)
    bf16 = lambda: torch.nn.functional.scaled_dot_product_attention(q[:, None, :], Kb, Vb)
    for f in (msaq_aa, mx, bf16): f()
    ta = min(_t(msaq_aa), _t(msaq_aa)); tx = min(_t(mx), _t(mx)); tb = min(_t(bf16), _t(bf16))
    print(f"  Lk={Lk:>5} u{u}/gs{gs} | bf16 {tb*1e3:6.1f}us | mxint8 {tx*1e3:6.1f}us ({tx/tb:.2f}x) | "
          f"AA+KV(msaq) {ta*1e3:6.1f}us ({ta/tb:.2f}x bf / {ta/tx:.2f}x mx)")


if __name__ == "__main__":
    print("=== PREFILL AA self-attention (H=8, D=128): qk_wmma+pv_wmma vs bf16 SDPA ===")
    for L in (512, 1024, 2048):
        prefill(8, L, 128)
    print("\n=== DECODE AA+KV (Hq=32,Hkv=8,D=128) vs bf16 & MXINT8 — u2/gs8 (robust AA) and u4/gs2 ===")
    for cfg in ((2, 8), (4, 2)):
        for Lk in (1024, 4096, 16384):
            decode(32, 8, Lk, 128, u=cfg[0], gs=cfg[1])
