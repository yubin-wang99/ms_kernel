"""End-to-end Llama-3.1-8B TIMING harness (design: harness_design.md).

Full 32-layer decoder forward wired to the 7 MSAQ kernels, comparing BF16 /
MXINT8 / MSAQ across W-only and W+A. Measures TTFT (prefill) and TPOT (decode).

This is a TIMING harness: weights are random and reused across layers (timing is
value-independent), glue (RMSNorm/RoPE/SwiGLU/softmax) is bf16 and common to all
paths so it cancels in the MSAQ-vs-MXINT8 comparison. The autoregressive decode
loop is Python-driven, so absolute TPOT includes per-op dispatch overhead (same
for every path); the ratio is what isolates the kernels.

Usage:  python harness.py [--prefill 800] [--decode 3880] [--layers 32]
                          [--paths bf16,mxint8_wonly,...] [--us 2,3,4]
"""
import argparse, time, sys, gc
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, "..")
from ms_lib import ops  # noqa: F401  (registers torch.ops.msaq)
from ms_lib.pack import pack_weight, pack_weight_mxint8

OPS = torch.ops.msaq


# ---- Llama-3.1-8B config ----------------------------------------------------
class Cfg:
    hidden = 4096
    n_head = 32
    n_kv = 8                      # GQA group = 4
    head_dim = 128
    inter = 14336
    vocab = 128256
    eps = 1e-5
    theta = 500000.0


def cuda(a):
    return torch.from_numpy(a).cuda()


# ---- one quantized linear (weight reused across all layers) -----------------
class QLinear:
    """W [OUT, K]; exposes gemm (prefill, [M,K]->[M,OUT]) and gemv ([K]->[OUT])."""
    def __init__(self, OUT, K, path, u, gs, seed):
        rng = np.random.default_rng(seed)
        W = (rng.standard_normal((OUT, K)) * 0.02).astype(np.float32)
        self.OUT, self.K, self.nb, self.path, self.u, self.gs = OUT, K, K // 32, path, u, gs
        if path == "bf16":
            self.Wt = torch.from_numpy(W).to(torch.bfloat16).cuda().t().contiguous()
        elif path.startswith("mxint8"):
            p = pack_weight_mxint8(W)
            self.s, self.qw = cuda(p["scale_exp"]), cuda(p["qweight"])
        else:                                            # msaq
            p = pack_weight(W, u, gs)
            self.s = cuda(p["scale_exp"])
            self.up, self.sh = cuda(p["upper"]), cuda(p["shared"])         # GEMM planes
            self.upc, self.shc = cuda(p["upper_cm"]), cuda(p["shared_cm"])  # GEMV planes

    def gemm(self, X):                                   # X [M,K] bf16
        p, M = self.path, X.shape[0]
        if p == "bf16":          return X @ self.Wt
        if p == "mxint8_wonly":  return OPS.mxint8_gemm(X, self.s, self.qw, M, self.OUT, self.K, self.nb)
        if p == "mxint8_wa":     return OPS.mxint8_wa_gemm(X, self.s, self.qw, M, self.OUT, self.K, self.nb)
        if p == "msaq_wonly":    return OPS.wonly_gemm(X, self.s, self.up, self.sh, M, self.OUT, self.K, self.nb, self.u, self.gs)
        if p == "msaq_wa":       return OPS.wa_gemm(X, self.s, self.up, self.sh, M, self.OUT, self.K, self.nb, self.u, self.gs)

    def gemv(self, x):                                   # x [K] bf16
        p = self.path
        if p == "bf16":          return x @ self.Wt
        if p == "mxint8_wonly":  return OPS.mxint8_gemv(x, self.s, self.qw, self.OUT, self.nb)
        if p == "mxint8_wa":     return OPS.mxint8_wa_gemv(x, self.s, self.qw, self.OUT, self.nb)
        if p == "msaq_wonly":    return OPS.wonly_gemv_wide(x, self.s, self.upc, self.shc, self.OUT, self.nb, self.u, self.gs)
        if p == "msaq_wa":       return OPS.wa_gemv(x, self.s, self.upc, self.shc, self.OUT, self.nb, self.u, self.gs)


# ---- glue (bf16/fp32, common to all paths) ----------------------------------
def rmsnorm(x):                                          # x [..., hidden]
    xf = x.float()
    return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + Cfg.eps)).to(x.dtype)


def rope_tables(L, dev):
    hd = Cfg.head_dim
    inv = 1.0 / (Cfg.theta ** (torch.arange(0, hd, 2, device=dev).float() / hd))
    t = torch.arange(L, device=dev).float()
    f = torch.outer(t, inv)                              # [L, hd/2]
    return torch.cos(f), torch.sin(f)                    # [L, hd/2]


def rope(x, cos, sin):                                   # x [L, H, hd], cos/sin [L, hd/2]
    x1, x2 = x[..., 0::2], x[..., 1::2]
    c, s = cos[:, None, :], sin[:, None, :]
    return torch.stack([x1 * c - x2 * s, x1 * s + x2 * c], -1).flatten(-2).to(x.dtype)


def silu_mlp(gate, up):
    return (F.silu(gate.float()) * up.float()).to(gate.dtype)


# ---- KV cache (one per layer) -----------------------------------------------
class KVCache:
    def __init__(self, path, Lcap, u, gs):
        H, hd, nb = Cfg.n_kv, Cfg.head_dim, Cfg.head_dim // 32
        self.path, self.Lcap, self.u, self.gs, self.nb = path, Lcap, u, gs, nb
        if path == "bf16":
            self.K = torch.zeros((H, Lcap, hd), dtype=torch.bfloat16, device="cuda")
            self.V = torch.zeros((H, Lcap, hd), dtype=torch.bfloat16, device="cuda")
        elif path.startswith("mxint8"):
            self.ks = torch.zeros((H, nb, Lcap), dtype=torch.int8, device="cuda")
            self.kq = torch.zeros((H, nb, Lcap, 32), dtype=torch.int8, device="cuda")
            self.vs = torch.zeros((H, nb, Lcap), dtype=torch.int8, device="cuda")
            self.vq = torch.zeros((H, nb, Lcap, 32), dtype=torch.int8, device="cuda")
        else:
            wbits = 8 - u
            UB, SB = 32 * wbits // 8, ((32 // gs) * u + 7) // 8
            mk = lambda: (torch.zeros((H, nb, Lcap), dtype=torch.int8, device="cuda"),
                          torch.zeros((H, nb, Lcap, UB), dtype=torch.uint8, device="cuda"),
                          torch.zeros((H, nb, Lcap, SB), dtype=torch.uint8, device="cuda"))
            self.ks, self.ku, self.kh = mk()
            self.vs, self.vu, self.vh = mk()

    def write_prefill(self, Kr, Vr):                     # Kr,Vr [H, L, hd] bf16 (RoPE'd K)
        H, hd, nb, L = Cfg.n_kv, Cfg.head_dim, self.nb, Kr.shape[1]
        if self.path == "bf16":
            self.K[:, :L].copy_(Kr); self.V[:, :L].copy_(Vr); return
        if self.path.startswith("mxint8"):
            ks, kq = OPS.mxint8_kv_write(Kr.contiguous(), H, L, hd, nb)
            vs, vq = OPS.mxint8_kv_write(Vr.contiguous(), H, L, hd, nb)
            self.ks[:, :, :L].copy_(ks); self.kq[:, :, :L].copy_(kq)
            self.vs[:, :, :L].copy_(vs); self.vq[:, :, :L].copy_(vq); return
        ks, ku, kh = OPS.kv_write(Kr.contiguous(), H, L, hd, nb, self.u, self.gs)
        vs, vu, vh = OPS.kv_write(Vr.contiguous(), H, L, hd, nb, self.u, self.gs)
        self.ks[:, :, :L].copy_(ks); self.ku[:, :, :L].copy_(ku); self.kh[:, :, :L].copy_(kh)
        self.vs[:, :, :L].copy_(vs); self.vu[:, :, :L].copy_(vu); self.vh[:, :, :L].copy_(vh)

    def append(self, k_new, v_new, pos):                 # k_new,v_new [H, hd] bf16 (RoPE'd k)
        H, hd, nb, Lc = Cfg.n_kv, Cfg.head_dim, self.nb, self.Lcap
        if self.path == "bf16":
            self.K[:, pos].copy_(k_new); self.V[:, pos].copy_(v_new); return
        if self.path.startswith("mxint8"):
            OPS.mxint8_kv_append(k_new.contiguous(), self.ks, self.kq, H, hd, nb, pos, Lc)
            OPS.mxint8_kv_append(v_new.contiguous(), self.vs, self.vq, H, hd, nb, pos, Lc); return
        OPS.kv_append(k_new.contiguous(), self.ks, self.ku, self.kh, H, hd, nb, pos, Lc, self.u, self.gs)
        OPS.kv_append(v_new.contiguous(), self.vs, self.vu, self.vh, H, hd, nb, pos, Lc, self.u, self.gs)

    def attend(self, q):                                 # q [Hq, hd] bf16 -> [Hq, hd]
        Hq, Hkv, hd, nb, Lk, Lc = Cfg.n_head, Cfg.n_kv, Cfg.head_dim, self.nb, self.pos + 1, self.Lcap
        if self.path == "bf16":
            g = Hq // Hkv
            k = self.K[:, :Lk].repeat_interleave(g, 0)   # [Hq, Lk, hd]
            v = self.V[:, :Lk].repeat_interleave(g, 0)
            o = F.scaled_dot_product_attention(q[:, None, :], k, v)   # [Hq,1,hd]
            return o[:, 0, :].to(torch.bfloat16)
        if self.path.startswith("mxint8"):
            return OPS.mxint8_kv_decode(q, self.ks, self.kq, self.vs, self.vq, Hq, Hkv, Lk, hd, nb, Lc)
        return OPS.kv_decode_attention(q, self.ks, self.ku, self.kh, self.vs, self.vu, self.vh,
                                       Hq, Hkv, Lk, hd, nb, self.u, self.gs, Lc)


# ---- the model (weights reused across layers) -------------------------------
class Model:
    def __init__(self, path, u, gs):
        C = Cfg
        self.path = path
        L = lambda OUT, K, sd: QLinear(OUT, K, path, u, gs, sd)
        self.wq = L(C.n_head * C.head_dim, C.hidden, 1)
        self.wk = L(C.n_kv * C.head_dim, C.hidden, 2)
        self.wv = L(C.n_kv * C.head_dim, C.hidden, 3)
        self.wo = L(C.hidden, C.n_head * C.head_dim, 4)
        self.wg = L(C.inter, C.hidden, 5)
        self.wu = L(C.inter, C.hidden, 6)
        self.wd = L(C.hidden, C.inter, 7)
        rng = np.random.default_rng(8)
        self.embed = torch.from_numpy((rng.standard_normal((C.vocab, C.hidden)) * 0.02).astype(np.float32)).to(torch.bfloat16).cuda()
        self.lm_head = torch.from_numpy((rng.standard_normal((C.hidden, C.vocab)) * 0.02).astype(np.float32)).to(torch.bfloat16).cuda()

    # ---- prefill: process the whole prompt through `layers` layers ----------
    def prefill(self, ids, layers, caches):
        C = Cfg
        x = self.embed[ids]                              # [L, hidden] bf16
        Lp = x.shape[0]
        cos, sin = rope_tables(Lp, x.device)
        g = C.n_head // C.n_kv
        for li in range(layers):
            h = rmsnorm(x)
            q = self.wq.gemm(h).view(Lp, C.n_head, C.head_dim)
            k = self.wk.gemm(h).view(Lp, C.n_kv, C.head_dim)
            v = self.wv.gemm(h).view(Lp, C.n_kv, C.head_dim)
            q = rope(q, cos, sin); k = rope(k, cos, sin)
            caches[li].write_prefill(k.transpose(0, 1).contiguous(), v.transpose(0, 1).contiguous())
            caches[li].pos = Lp - 1
            # prefill attention = bf16 SDPA causal (GQA)
            qa = q.transpose(0, 1)                        # [Hq, L, hd]
            ka = k.transpose(0, 1).repeat_interleave(g, 0)
            va = v.transpose(0, 1).repeat_interleave(g, 0)
            ao = F.scaled_dot_product_attention(qa, ka, va, is_causal=True)  # [Hq,L,hd]
            ao = ao.transpose(0, 1).reshape(Lp, C.n_head * C.head_dim).to(torch.bfloat16)
            x = x + self.wo.gemm(ao)
            h = rmsnorm(x)
            x = x + self.wd.gemm(silu_mlp(self.wg.gemm(h), self.wu.gemm(h)))
        logits = rmsnorm(x[-1:]) @ self.lm_head          # last token -> [1, vocab]
        return x[-1]                                     # [hidden] (next decode input)

    # ---- one decode step: a single token through `layers` layers ------------
    def decode_step(self, x, pos, layers, caches):
        C = Cfg
        cos = torch.cos(torch.outer(torch.tensor([float(pos)], device=x.device),
                                    1.0 / (C.theta ** (torch.arange(0, C.head_dim, 2, device=x.device).float() / C.head_dim))))
        sin = torch.sin(torch.outer(torch.tensor([float(pos)], device=x.device),
                                    1.0 / (C.theta ** (torch.arange(0, C.head_dim, 2, device=x.device).float() / C.head_dim))))
        for li in range(layers):
            h = rmsnorm(x)
            q = self.wq.gemv(h).view(C.n_head, C.head_dim)
            k = self.wk.gemv(h).view(C.n_kv, C.head_dim)
            v = self.wv.gemv(h).view(C.n_kv, C.head_dim)
            q = rope(q[None], cos, sin)[0]; k = rope(k[None], cos, sin)[0]
            caches[li].append(k, v, pos)
            caches[li].pos = pos
            ao = caches[li].attend(q).reshape(C.n_head * C.head_dim)
            x = x + self.wo.gemv(ao)
            h = rmsnorm(x)
            x = x + self.wd.gemv(silu_mlp(self.wg.gemv(h), self.wu.gemv(h)))
        _ = rmsnorm(x[None]) @ self.lm_head              # lm_head every step (glue, bf16)
        return x


def run_path(path, u, gs, prefill, decode, layers, tag):
    torch.cuda.synchronize(); torch.cuda.empty_cache()
    m = Model(path, u, gs)
    Lcap = prefill + decode
    caches = [KVCache(path, Lcap, u, gs) for _ in range(layers)]
    ids = torch.randint(0, Cfg.vocab, (prefill,), device="cuda")

    # warmup INTO the real caches (prefill+decode overwrite them) — saves the
    # peak memory of a second 32-layer cache set.
    xw = m.prefill(ids, layers, caches)
    for p in range(prefill, prefill + 4):
        xw = m.decode_step(xw, p, layers, caches)
    torch.cuda.synchronize()

    # ---- TTFT ----
    t0 = torch.cuda.Event(True); t1 = torch.cuda.Event(True)
    t0.record(); x = m.prefill(ids, layers, caches); t1.record()
    torch.cuda.synchronize(); ttft = t0.elapsed_time(t1)  # ms

    # ---- decode: full loop, sampled growth curve ----
    marks = {1, 256, 1024, 2048, decode}
    curve = {}
    torch.cuda.synchronize(); ds = time.perf_counter()
    e0 = torch.cuda.Event(True); e1 = torch.cuda.Event(True)
    for i in range(decode):
        pos = prefill + i
        if (i + 1) in marks:
            torch.cuda.synchronize(); e0.record()
            x = m.decode_step(x, pos, layers, caches)
            e1.record(); torch.cuda.synchronize()
            curve[i + 1] = e0.elapsed_time(e1)
        else:
            x = m.decode_step(x, pos, layers, caches)
    torch.cuda.synchronize(); dtot = (time.perf_counter() - ds) * 1e3  # ms
    tpot = dtot / decode
    total = ttft + dtot
    del m, caches; gc.collect(); torch.cuda.empty_cache()
    return dict(tag=tag, ttft=ttft, tpot=tpot, total=total, curve=curve)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefill", type=int, default=800)
    ap.add_argument("--decode", type=int, default=3880)
    ap.add_argument("--layers", type=int, default=32)
    ap.add_argument("--paths", default="bf16,mxint8_wonly,mxint8_wa,msaq_wonly,msaq_wa")
    ap.add_argument("--us", default="2,3,4")
    args = ap.parse_args()
    print(f"[harness] {torch.cuda.get_device_name(0)} | Llama-3.1-8B | "
          f"prefill={args.prefill} decode={args.decode} layers={args.layers}")
    print(f"          (random reused weights; glue bf16; ms; lower better)\n")

    jobs = []
    for p in args.paths.split(","):
        if p.startswith("msaq"):
            for u in [int(x) for x in args.us.split(",")]:
                jobs.append((p, u, 8, f"{p}-u{u}"))
        else:
            jobs.append((p, 4, 8, p))

    rows = []
    for path, u, gs, tag in jobs:
        r = run_path(path, u, gs, args.prefill, args.decode, args.layers, tag)
        rows.append(r)
        cur = "  ".join(f"t{k}:{v:.1f}" for k, v in sorted(r["curve"].items()))
        print(f"  {tag:16s} TTFT {r['ttft']:8.1f}  TPOT {r['tpot']:6.3f}  total {r['total']/1e3:7.2f}s | TPOT@ {cur}")

    # ratios vs bf16 and vs matched mxint8
    base = {r["tag"]: r for r in rows}
    print("\n  ratios (total inference time):")
    bf = base.get("bf16")
    for r in rows:
        ref = ""
        if bf: ref += f" /bf16 {r['total']/bf['total']:.2f}"
        mx = base.get("mxint8_wa" if "wa" in r["tag"] else "mxint8_wonly")
        if mx and r["tag"].startswith("msaq"): ref += f"  /mxint8 {r['total']/mx['total']:.2f}"
        print(f"  {r['tag']:16s}{ref}")


if __name__ == "__main__":
    main()
