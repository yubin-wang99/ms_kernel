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
import argparse, time, sys, gc, os, json, subprocess
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


def rope_pos(pos, dev):                                  # cos/sin [1, hd/2] for one position
    hd = Cfg.head_dim
    inv = 1.0 / (Cfg.theta ** (torch.arange(0, hd, 2, device=dev).float() / hd))
    f = torch.outer(torch.tensor([float(pos)], device=dev), inv)
    return torch.cos(f), torch.sin(f)


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
    def __init__(self, w_path, u, gs):
        C = Cfg
        self.w_path = w_path                         # weight quant: bf16/{mx,msaq}_{wonly,wa}
        L = lambda OUT, K, sd: QLinear(OUT, K, w_path, u, gs, sd)
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

    # ---- one decode step (cos/sin precomputed -> graph-capturable) ----------
    def decode_step(self, x, cos, sin, pos, layers, caches):
        C = Cfg
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


def run_scenario(w_path, kv_path, u, gs, prefill, decode, layers, tag, reps=50):
    """CUDA-graph decode timing: capture one decode step per context checkpoint
    (dispatch-free), replay it, then integrate over the decode trajectory."""
    torch.cuda.synchronize(); torch.cuda.empty_cache()
    m = Model(w_path, u, gs)
    Lcap = prefill + decode
    caches = [KVCache(kv_path, Lcap, u, gs) for _ in range(layers)]
    ids = torch.randint(0, Cfg.vocab, (prefill,), device="cuda")

    # prefill twice (warm + timed). Fills caches[:prefill]; slots >= prefill stay 0
    # (zeros are valid bytes -> attend cost is content-independent, so no fill needed).
    m.prefill(ids, layers, caches); torch.cuda.synchronize()
    t0 = torch.cuda.Event(True); t1 = torch.cuda.Event(True)
    t0.record(); m.prefill(ids, layers, caches); t1.record()
    torch.cuda.synchronize(); ttft = t0.elapsed_time(t1)

    # ---- decode TPOT via CUDA graph at each context checkpoint ----
    ctxs = sorted({prefill + 1, prefill + 256, prefill + 1024,
                   prefill + 2048, prefill + decode})
    x_static = torch.zeros(Cfg.hidden, dtype=torch.bfloat16, device="cuda")
    gtpot = {}
    for ctx in ctxs:
        pos = min(ctx, Lcap - 1)
        cos, sin = rope_pos(pos, "cuda")
        for c in caches:
            c.pos = pos
        s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):                    # prime allocator / autotune on side stream
            for _ in range(3):
                m.decode_step(x_static, cos, sin, pos, layers, caches)
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            m.decode_step(x_static, cos, sin, pos, layers, caches)
        torch.cuda.synchronize()
        e0 = torch.cuda.Event(True); e1 = torch.cuda.Event(True)
        e0.record()
        for _ in range(reps):
            g.replay()
        e1.record(); torch.cuda.synchronize()
        gtpot[ctx] = e0.elapsed_time(e1) / reps       # dispatch-free per-step ms at this context
        del g

    # integrate per-step cost over the decode trajectory (ctx = prefill+1 .. prefill+decode)
    cks = sorted(gtpot); vals = [gtpot[c] for c in cks]
    traj = np.interp(np.arange(prefill + 1, prefill + 1 + decode), cks, vals)
    dtot = float(traj.sum())                          # ms
    tpot = dtot / decode
    total = ttft + dtot
    del m, caches; gc.collect(); torch.cuda.empty_cache()
    curve = [[c, gtpot[c]] for c in sorted(gtpot)]    # json-safe (list, not int-keyed dict)
    return dict(tag=tag, ttft=ttft, tpot=tpot, total=total, curve=curve)


# the four decoupled quantization scenarios (weight knob x KV knob).
#   w_style: None -> bf16 weights;  kv: True -> quantized KV cache.
SCENARIOS = [
    ("S1 W-only",    "wonly", False),
    ("S2 W+A",       "wa",    False),
    ("S3 KV-only",   None,    True),
    ("S4 W-only+KV", "wonly", True),
]


def build_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefill", type=int, default=800)
    ap.add_argument("--decode", type=int, default=3880)
    ap.add_argument("--layers", type=int, default=32)
    ap.add_argument("--us", default="4")
    ap.add_argument("--reps", type=int, default=50)
    # worker mode (one scenario per fresh process — repeated CUDA-graph capture in
    # one process wedges the next eager prefill, so each scenario is isolated).
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--wpath"); ap.add_argument("--kvpath")
    ap.add_argument("--u", type=int, default=4); ap.add_argument("--tag", default="")
    return ap.parse_args()


def worker(args):
    r = run_scenario(args.wpath, args.kvpath, args.u, 8, args.prefill, args.decode,
                     args.layers, args.tag, args.reps)
    print("RESULT " + json.dumps(r), flush=True)


def spawn(args, w, kv, u, tag):
    cmd = [sys.executable, "-u", os.path.abspath(__file__), "--worker",
           "--wpath", w, "--kvpath", kv, "--u", str(u), "--tag", tag,
           "--prefill", str(args.prefill), "--decode", str(args.decode),
           "--layers", str(args.layers), "--reps", str(args.reps)]
    out = subprocess.run(cmd, capture_output=True, text=True)
    for line in out.stdout.splitlines():
        if line.startswith("RESULT "):
            return json.loads(line[len("RESULT "):])
    raise RuntimeError(f"worker failed [{tag}]:\nSTDOUT:{out.stdout[-800:]}\nSTDERR:{out.stderr[-800:]}")


def main():
    args = build_args()
    if args.worker:
        worker(args); return
    us = [int(x) for x in args.us.split(",")]
    print(f"[harness] {torch.cuda.get_device_name(0)} | Llama-3.1-8B | "
          f"prefill={args.prefill} decode={args.decode} layers={args.layers} | CUDA-graph decode")
    print(f"          (random reused weights; glue bf16; per-scenario subprocess; ms; lower better)\n")

    bf = spawn(args, "bf16", "bf16", 4, "bf16 baseline")
    print(f"  {bf['tag']:22s} TTFT {bf['ttft']:7.1f}  TPOT {bf['tpot']:6.3f}  total {bf['total']/1e3:6.2f}s",
          flush=True)

    rows = [bf]
    for sname, wstyle, kvq in SCENARIOS:
        print(f"\n  === {sname} ===", flush=True)
        variants = [("mxint8", 4)] + [("msaq", u) for u in us]   # non-quant side stays bf16
        scen_rows = []
        for fmt, u in variants:
            w = "bf16" if wstyle is None else f"{fmt}_{wstyle}"
            kv = fmt if kvq else "bf16"
            usuf = f"-u{u}" if fmt == "msaq" else ""
            r = spawn(args, w, kv, u, f"{sname} [{fmt}{usuf}]")
            scen_rows.append(r); rows.append(r)
            print(f"  {r['tag']:22s} TTFT {r['ttft']:7.1f}  TPOT {r['tpot']:6.3f}  "
                  f"total {r['total']/1e3:6.2f}s  /bf16 {r['total']/bf['total']:.2f}", flush=True)
        mx = scen_rows[0]
        for r in scen_rows[1:]:
            print(f"       {r['tag']:28s} /mxint8 {r['total']/mx['total']:.2f}", flush=True)

    # growth curves (dispatch-free per-step ms at each context)
    print("\n  TPOT growth (graph, ms) by context length:")
    cks = [c for c, _ in rows[0]["curve"]]
    print("    " + "ctx:".ljust(26) + "  ".join(f"{c:>6d}" for c in cks))
    for r in rows:
        d = dict(r["curve"])
        print("    " + r["tag"].ljust(26) + "  ".join(f"{d[c]:6.2f}" for c in cks))


if __name__ == "__main__":
    main()
