"""Batched E2E timing harness — batch + output sweep (kernel_ver2.md §3 workload spec).

Measures PREFILL (TTFT), DECODE (integrated), and TOTAL time for BF16 / MXINT8 / MSAQ
across 5 scopes, and reports all three ratio pairs per metric:
  msaq/mxint8 (format axis, MXINT8 baseline) · msaq/bf16 · mxint8/bf16 (common BF16 baseline)
Operating points (§3): batch sweep B in {1,8,32,64,128,256} at (L_in,L_out)=(1024,512);
output sweep L_in=1024, L_out in {128,512,1024,2048,3880} at fixed B.
Llama-3.1-8B dims; weights random+reused (timing value-independent); glue bf16.

PREFILL uses each weight-path's optimized tile: W-only -> MS_TILE_CFG=11 (pipelined BF16
WMMA, the Readme prefill win); W+A -> auto M-adaptive 2-stage IMMA (cfg=11 would force 64x64
-> wrong). The env is set only around prefill and popped for decode (M-adaptive default;
no optimized small-batch decode GEMM exists, so B>1 decode of weight-quant paths is slow).

Batching: B folds into the head dim for KV write/append (existing per-head kernels); attend
uses the batched read kernel. Decode weight matmul: B=1 -> GEMV, B>1 -> GEMM(M=B). Output
sweep computed from ONE decode trajectory (to max L_out) via prefix integration.

Usage: CUDA_VISIBLE_DEVICES=0 python tests/harness_batchsweep.py [--reps 15]
"""
import argparse, sys, os, json, gc, subprocess
import numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ms_lib import ops  # noqa: F401
from ms_lib.pack import pack_weight, pack_weight_mxint8
OPS = torch.ops.msaq


class Cfg:                         # Llama-3.1-8B
    hidden = 4096; n_head = 32; n_kv = 8; head_dim = 128; inter = 14336
    vocab = 128256; eps = 1e-5; theta = 500000.0; layers = 32


def cuda(a): return torch.from_numpy(a).cuda()


class QLinear:
    def __init__(self, OUT, K, path, u, gs, seed):
        rng = np.random.default_rng(seed)
        W = (rng.standard_normal((OUT, K)) * 0.02).astype(np.float32)
        self.OUT, self.K, self.nb, self.path, self.u, self.gs = OUT, K, K // 32, path, u, gs
        if path == "bf16":
            self.Wt = torch.from_numpy(W).to(torch.bfloat16).cuda().t().contiguous()
        elif path.startswith("mxint8"):
            p = pack_weight_mxint8(W); self.s, self.qw = cuda(p["scale_exp"]), cuda(p["qweight"])
            self.qwc = cuda(p["qweight_cm"])   # column-major twin for the wide-load kernels
        else:
            p = pack_weight(W, u, gs); self.s = cuda(p["scale_exp"])
            self.up, self.sh = cuda(p["upper"]), cuda(p["shared"])
            self.upc, self.shc = cuda(p["upper_cm"]), cuda(p["shared_cm"])

    def gemm(self, X):                                   # X [M,K] bf16 -> [M,OUT]
        p, M = self.path, X.shape[0]
        if p == "bf16":          return X @ self.Wt
        # PREFILL (compute-bound GEMM): dequant weight ONCE -> bf16 [K,OUT], then cuBLAS X@Wd.
        # Ties bf16 (the fused per-tile dequant starved the tensor cores ~11% -> ~4x slower).
        if p == "mxint8_wonly" or p == "mxint8_wa":
            return X @ OPS.mxint8_dequant_bf16(self.s, self.qwc, self.OUT, self.K, self.nb)
        if p == "msaq_wonly" or p == "msaq_wa":
            return X @ OPS.ms_dequant_bf16(self.s, self.upc, self.shc, self.OUT, self.K, self.nb, self.u, self.gs)

    def gemv(self, x):                                   # x [K] bf16 -> [OUT]
        p = self.path
        if p == "bf16":          return x @ self.Wt
        if p == "mxint8_wonly":  return OPS.mxint8_gemv_wide(x, self.s, self.qwc, self.OUT, self.nb)
        if p == "mxint8_wa":     return OPS.mxint8_wa_gemv_wide(x, self.s, self.qwc, self.OUT, self.nb)
        if p == "msaq_wonly":    return OPS.wonly_gemv_wide(x, self.s, self.upc, self.shc, self.OUT, self.nb, self.u, self.gs)
        if p == "msaq_wa":       return OPS.wa_gemv(x, self.s, self.upc, self.shc, self.OUT, self.nb, self.u, self.gs)

    def fwd(self, X):                                    # decode: [B,K]->[B,OUT]
        B, p = X.shape[0], self.path
        if B == 1: return self.gemv(X[0])[None]          # B=1 -> wide GEMV
        # B>1 W-only -> batched-decode GEMV (amortize weight read over B); else GEMM
        if p == "msaq_wonly":   return OPS.wonly_gemv_batched(X, self.s, self.upc, self.shc, B, self.OUT, self.nb, self.u, self.gs)
        if p == "mxint8_wonly": return OPS.mxint8_gemv_batched_wide(X, self.s, self.qwc, B, self.OUT, self.nb)
        if p == "msaq_wa":      return OPS.wa_gemv_batched(X, self.s, self.upc, self.shc, B, self.OUT, self.nb, self.u, self.gs)
        if p == "mxint8_wa":    return OPS.mxint8_wa_gemv_batched_wide(X, self.s, self.qwc, B, self.OUT, self.nb)
        return self.gemm(X)                              # bf16 (cuBLAS)


def rmsnorm(x):
    xf = x.float()
    return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + Cfg.eps)).to(x.dtype)

def rope_tables(L, dev):
    hd = Cfg.head_dim
    inv = 1.0 / (Cfg.theta ** (torch.arange(0, hd, 2, device=dev).float() / hd))
    f = torch.outer(torch.arange(L, device=dev).float(), inv)
    return torch.cos(f), torch.sin(f)

def rope_pos(pos, dev):
    hd = Cfg.head_dim
    inv = 1.0 / (Cfg.theta ** (torch.arange(0, hd, 2, device=dev).float() / hd))
    f = torch.outer(torch.tensor([float(pos)], device=dev), inv)
    return torch.cos(f), torch.sin(f)

def rope_bthd(x, cos, sin):                              # x [B,L,H,hd], cos/sin [L,hd/2]
    x1, x2 = x[..., 0::2], x[..., 1::2]
    c, s = cos[None, :, None, :], sin[None, :, None, :]
    return torch.stack([x1 * c - x2 * s, x1 * s + x2 * c], -1).flatten(-2).to(x.dtype)

def silu_mlp(g, u): return (F.silu(g.float()) * u.float()).to(g.dtype)


class KVCache:
    """Batched cache; planes [B,Hkv,nb,Lcap,*]. write/append fold B into head dim."""
    def __init__(self, path, B, Lcap, u, gs):
        H, hd, nb = Cfg.n_kv, Cfg.head_dim, Cfg.head_dim // 32
        self.path, self.B, self.Lcap, self.u, self.gs, self.nb = path, B, Lcap, u, gs, nb
        z = lambda *s, dt=torch.int8: torch.zeros(s, dtype=dt, device="cuda")
        if path == "bf16":
            self.K = z(B, H, Lcap, hd, dt=torch.bfloat16); self.V = z(B, H, Lcap, hd, dt=torch.bfloat16)
        elif path.startswith("mxint8"):
            self.ks = z(B, H, nb, Lcap); self.kq = z(B, H, nb, Lcap, 32)
            self.vs = z(B, H, nb, Lcap); self.vq = z(B, H, nb, Lcap, 32)
        else:
            UB, SB = 32 * (8 - u) // 8, ((32 // gs) * u + 7) // 8
            mk = lambda: (z(B, H, nb, Lcap), z(B, H, nb, Lcap, UB, dt=torch.uint8), z(B, H, nb, Lcap, SB, dt=torch.uint8))
            self.ks, self.ku, self.kh = mk(); self.vs, self.vu, self.vh = mk()

    def write_prefill(self, Kr, Vr):                     # Kr,Vr [B,Hkv,Lp,hd] bf16
        B, H, hd, nb, Lp = self.B, Cfg.n_kv, Cfg.head_dim, self.nb, Kr.shape[2]
        BH = B * H
        if self.path == "bf16":
            self.K[:, :, :Lp].copy_(Kr); self.V[:, :, :Lp].copy_(Vr); return
        Kf, Vf = Kr.reshape(BH, Lp, hd).contiguous(), Vr.reshape(BH, Lp, hd).contiguous()
        if self.path.startswith("mxint8"):
            ks, kq = OPS.mxint8_kv_write(Kf, BH, Lp, hd, nb); vs, vq = OPS.mxint8_kv_write(Vf, BH, Lp, hd, nb)
            self.ks.view(BH, nb, self.Lcap)[:, :, :Lp].copy_(ks.view(BH, nb, Lp))
            self.kq.view(BH, nb, self.Lcap, 32)[:, :, :Lp].copy_(kq.view(BH, nb, Lp, 32))
            self.vs.view(BH, nb, self.Lcap)[:, :, :Lp].copy_(vs.view(BH, nb, Lp))
            self.vq.view(BH, nb, self.Lcap, 32)[:, :, :Lp].copy_(vq.view(BH, nb, Lp, 32)); return
        ks, ku, kh = OPS.kv_write(Kf, BH, Lp, hd, nb, self.u, self.gs)
        vs, vu, vh = OPS.kv_write(Vf, BH, Lp, hd, nb, self.u, self.gs)
        UB, SB = ku.shape[-1], kh.shape[-1]
        for dst, src in ((self.ks, ks), (self.vs, vs)):
            dst.view(BH, nb, self.Lcap)[:, :, :Lp].copy_(src.view(BH, nb, Lp))
        for dst, src, W in ((self.ku, ku, UB), (self.kh, kh, SB), (self.vu, vu, UB), (self.vh, vh, SB)):
            dst.view(BH, nb, self.Lcap, W)[:, :, :Lp].copy_(src.view(BH, nb, Lp, W))

    def append(self, k_new, v_new, pos):                 # k_new,v_new [B,Hkv,hd]
        B, H, hd, nb, Lc = self.B, Cfg.n_kv, Cfg.head_dim, self.nb, self.Lcap
        BH = B * H
        if self.path == "bf16":
            self.K[:, :, pos].copy_(k_new); self.V[:, :, pos].copy_(v_new); return
        kf, vf = k_new.reshape(BH, hd).contiguous(), v_new.reshape(BH, hd).contiguous()
        if self.path.startswith("mxint8"):
            OPS.mxint8_kv_append(kf, self.ks.view(BH, nb, Lc), self.kq.view(BH, nb, Lc, 32), BH, hd, nb, pos, Lc)
            OPS.mxint8_kv_append(vf, self.vs.view(BH, nb, Lc), self.vq.view(BH, nb, Lc, 32), BH, hd, nb, pos, Lc); return
        UB, SB = self.ku.shape[-1], self.kh.shape[-1]
        OPS.kv_append(kf, self.ks.view(BH, nb, Lc), self.ku.view(BH, nb, Lc, UB), self.kh.view(BH, nb, Lc, SB), BH, hd, nb, pos, Lc, self.u, self.gs)
        OPS.kv_append(vf, self.vs.view(BH, nb, Lc), self.vu.view(BH, nb, Lc, UB), self.vh.view(BH, nb, Lc, SB), BH, hd, nb, pos, Lc, self.u, self.gs)

    def attend(self, q):                                 # q [B,Hq,hd] -> [B,Hq,hd]
        B, Hq, Hkv, hd, nb, Lk, Lc = self.B, Cfg.n_head, Cfg.n_kv, Cfg.head_dim, self.nb, self.pos + 1, self.Lcap
        if self.path == "bf16":
            g = Hq // Hkv
            k = self.K[:, :, :Lk].repeat_interleave(g, 1); v = self.V[:, :, :Lk].repeat_interleave(g, 1)
            o = F.scaled_dot_product_attention(q[:, :, None, :], k, v)   # [B,Hq,1,hd]
            return o[:, :, 0, :].to(torch.bfloat16)
        if self.path.startswith("mxint8"):
            return OPS.mxint8_kv_decode_batched(q, self.ks, self.kq, self.vs, self.vq, B, Hq, Hkv, Lk, hd, nb, Lc)
        return OPS.kv_decode_attention_batched(q, self.ks, self.ku, self.kh, self.vs, self.vu, self.vh,
                                               B, Hq, Hkv, Lk, hd, nb, self.u, self.gs, Lc)


class Model:
    def __init__(self, w_path, u, gs):
        C = Cfg; self.w_path = w_path
        L = lambda OUT, K, sd: QLinear(OUT, K, w_path, u, gs, sd)
        self.wq = L(C.n_head * C.head_dim, C.hidden, 1); self.wk = L(C.n_kv * C.head_dim, C.hidden, 2)
        self.wv = L(C.n_kv * C.head_dim, C.hidden, 3); self.wo = L(C.hidden, C.n_head * C.head_dim, 4)
        self.wg = L(C.inter, C.hidden, 5); self.wu = L(C.inter, C.hidden, 6); self.wd = L(C.hidden, C.inter, 7)
        rng = np.random.default_rng(8)
        self.embed = torch.from_numpy((rng.standard_normal((C.vocab, C.hidden)) * 0.02).astype(np.float32)).to(torch.bfloat16).cuda()

    def prefill(self, ids, caches):                      # ids [B,Lp]
        C = Cfg; B, Lp = ids.shape
        x = self.embed[ids]                              # [B,Lp,H]
        cos, sin = rope_tables(Lp, x.device); g = C.n_head // C.n_kv
        for li in range(C.layers):
            h = rmsnorm(x).reshape(B * Lp, C.hidden)
            q = self.wq.gemm(h).view(B, Lp, C.n_head, C.head_dim)
            k = self.wk.gemm(h).view(B, Lp, C.n_kv, C.head_dim)
            v = self.wv.gemm(h).view(B, Lp, C.n_kv, C.head_dim)
            q = rope_bthd(q, cos, sin); k = rope_bthd(k, cos, sin)
            caches[li].write_prefill(k.transpose(1, 2).contiguous(), v.transpose(1, 2).contiguous())
            caches[li].pos = Lp - 1
            qa = q.transpose(1, 2); ka = k.transpose(1, 2).repeat_interleave(g, 1); va = v.transpose(1, 2).repeat_interleave(g, 1)
            ao = F.scaled_dot_product_attention(qa, ka, va, is_causal=True)   # [B,Hq,Lp,hd]
            ao = ao.transpose(1, 2).reshape(B * Lp, C.n_head * C.head_dim).to(torch.bfloat16)
            x = (x.reshape(B * Lp, C.hidden) + self.wo.gemm(ao)).reshape(B, Lp, C.hidden)
            h = rmsnorm(x).reshape(B * Lp, C.hidden)
            x = (x.reshape(B * Lp, C.hidden) + self.wd.gemm(silu_mlp(self.wg.gemm(h), self.wu.gemm(h)))).reshape(B, Lp, C.hidden)
        return x[:, -1, :].contiguous()                  # [B,H]

    def decode_step(self, x, cos, sin, pos, caches):     # x [B,H]
        C = Cfg; B = x.shape[0]
        for li in range(C.layers):
            h = rmsnorm(x)
            q = self.wq.fwd(h).view(B, C.n_head, C.head_dim)
            k = self.wk.fwd(h).view(B, C.n_kv, C.head_dim)
            v = self.wv.fwd(h).view(B, C.n_kv, C.head_dim)
            q = rope_bthd(q[:, None], cos, sin)[:, 0]; k = rope_bthd(k[:, None], cos, sin)[:, 0]
            caches[li].append(k, v, pos); caches[li].pos = pos
            ao = caches[li].attend(q).reshape(B, C.n_head * C.head_dim)
            x = x + self.wo.fwd(ao)
            h = rmsnorm(x)
            x = x + self.wd.fwd(silu_mlp(self.wg.fwd(h), self.wu.fwd(h)))
        return x


CKPT_OFF = [1, 128, 512, 1024, 2048, 3880]               # decode-step offsets to sample TPOT at

def run_scenario(w_path, kv_path, u, gs, B, L_in, L_out, reps=30):
    torch.cuda.synchronize(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    m = Model(w_path, u, gs); Lcap = L_in + L_out
    caches = [KVCache(kv_path, B, Lcap, u, gs) for _ in range(Cfg.layers)]
    ids = torch.randint(0, Cfg.vocab, (B, L_in), device="cuda")
    # PREFILL: W-only uses the pipelined-WMMA tile (MS_TILE_CFG=11); W+A keeps auto. Popped for decode.
    if "wonly" in w_path: os.environ["MS_TILE_CFG"] = "11"
    else: os.environ.pop("MS_TILE_CFG", None)
    _fast = os.environ.get("MS_FAST") == "1"             # high-B sweep: long prefill keeps P0 hot
    for _ in range(1 if _fast else 2): m.prefill(ids, caches)   # warmup (alloc / cuBLAS autotune)
    torch.cuda.synchronize()
    _tt = []
    for _ in range(1 if _fast else 3):                   # min-of-3: single-shot TTFT was ~2x noisy (B=1)
        t0 = torch.cuda.Event(True); t1 = torch.cuda.Event(True)
        t0.record(); m.prefill(ids, caches); t1.record(); torch.cuda.synchronize()
        _tt.append(t0.elapsed_time(t1))
    ttft = min(_tt)
    os.environ.pop("MS_TILE_CFG", None)                  # decode: M-adaptive default

    offs = sorted({o for o in CKPT_OFF if o <= L_out} | {L_out})
    x_static = torch.zeros(B, Cfg.hidden, dtype=torch.bfloat16, device="cuda")
    gtpot = {}
    for off in offs:
        pos = min(L_in + off - 1, Lcap - 1)
        cos, sin = rope_pos(pos, "cuda")
        for c in caches: c.pos = pos
        s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3): m.decode_step(x_static, cos, sin, pos, caches)
        torch.cuda.current_stream().wait_stream(s)
        gr = torch.cuda.CUDAGraph()
        with torch.cuda.graph(gr): m.decode_step(x_static, cos, sin, pos, caches)
        for _ in range(20 if _fast else 60): gr.replay()   # ramp mem clock to P0 (decode is BW-bound;
        torch.cuda.synchronize()                 # a cold worker starts in a low-power P-state)
        best = float("inf")                       # min over windows = steady-state P0 (drop throttle dips)
        for _ in range(1 if _fast else 3):
            e0 = torch.cuda.Event(True); e1 = torch.cuda.Event(True); e0.record()
            for _ in range(reps): gr.replay()
            e1.record(); torch.cuda.synchronize()
            best = min(best, e0.elapsed_time(e1) / reps)
        gtpot[off] = best; del gr
    peak = torch.cuda.max_memory_allocated() / 1e9
    del m, caches; gc.collect(); torch.cuda.empty_cache()
    return dict(ttft=ttft, curve=sorted(gtpot.items()), peak_gb=peak)


def integ(curve, n):                                     # ms summed over decode steps 1..n
    cks = [c for c, _ in curve]; vals = [v for _, v in curve]
    return float(np.interp(np.arange(1, n + 1), cks, vals).sum())


# scopes: (name, weight-style, kv-quant). S5 = W+A + KV (full quant).
SCENARIOS = [("S1 W-only", "wonly", False), ("S2 W+A", "wa", False),
             ("S3 KV-only", None, True), ("S4 W-only+KV", "wonly", True),
             ("S5 W+A+KV", "wa", True)]


def build_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--u", type=int, default=4); ap.add_argument("--gs", type=int, default=2)
    ap.add_argument("--reps", type=int, default=30)
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--wpath"); ap.add_argument("--kvpath"); ap.add_argument("--tag", default="")
    ap.add_argument("--B", type=int, default=1); ap.add_argument("--lin", type=int, default=1024)
    ap.add_argument("--lout", type=int, default=512)
    ap.add_argument("--perscope", action="store_true")   # use PERSCOPE_CFG (robust u/gs per scope)
    ap.add_argument("--out", default="harness_batchsweep_results.jsonl")
    return ap.parse_args()


def worker(a):
    try:
        r = run_scenario(a.wpath, a.kvpath, a.u, a.gs, a.B, a.lin, a.lout, a.reps)
        print("RESULT " + json.dumps(r), flush=True)
    except torch.cuda.OutOfMemoryError:
        print("RESULT " + json.dumps({"oom": True}), flush=True)
    except RuntimeError as e:
        if "out of memory" in str(e).lower(): print("RESULT " + json.dumps({"oom": True}), flush=True)
        else: raise


def spawn(a, w, kv, u, gs, B, lin, lout, tag):
    cmd = [sys.executable, "-u", os.path.abspath(__file__), "--worker", "--wpath", w, "--kvpath", kv,
           "--u", str(u), "--gs", str(gs), "--B", str(B), "--lin", str(lin), "--lout", str(lout),
           "--reps", str(a.reps), "--tag", tag]
    out = subprocess.run(cmd, capture_output=True, text=True)
    for line in out.stdout.splitlines():
        if line.startswith("RESULT "): return json.loads(line[len("RESULT "):])
    return {"err": out.stderr[-500:]}


def variants(u=4, gs=2):                                 # (fmt,u,gs); only MSAQ uses u/gs (mx/bf ignore)
    return [("bf16", u, gs), ("mxint8", u, gs), ("msaq", u, gs)]

# per-scope max-aggressive robust config (scope_uvgs_results.md; S3 uses the u=4 nibble since robust)
PERSCOPE_CFG = {"S1 W-only": (3, 16), "S2 W+A": (2, 8), "S3 KV-only": (4, 2),
                "S4 W-only+KV": (2, 8), "S5 W+A+KV": (2, 8)}


def fmt_paths(fmt, wstyle, kvq):
    w = "bf16" if wstyle is None else (f"{fmt}_{wstyle}" if fmt != "bf16" else "bf16")
    kv = (fmt if kvq else "bf16") if fmt != "bf16" else "bf16"
    return w, kv


def _metrics(r, lout):                                   # (prefill, decode, total) ms, or None if OOM/err
    if r.get("oom") or not r.get("curve"): return None
    pre = r["ttft"]; dec = integ(r["curve"], lout)
    return (pre, dec, pre + dec)


def _ratiorow(bf, mx, ms, lout):                         # 9 ratios: per metric (msaq/mx, msaq/bf, mx/bf)
    Mb, Mx, Ms = _metrics(bf, lout), _metrics(mx, lout), _metrics(ms, lout)
    cells = []
    for i in range(3):                                   # prefill, decode, total
        sm = "OOM" if Ms is None else f"{Ms[i]/Mx[i]:.2f}" if Mx else "—"
        sb = "OOM" if Ms is None else f"{Ms[i]/Mb[i]:.2f}" if Mb else "—"
        xb = "OOM" if Mx is None else f"{Mx[i]/Mb[i]:.2f}" if Mb else "—"
        cells += [sm, sb, xb]
    return cells


def _hdr(axis):
    sub = f"{'mq/mx':>7}{'mq/bf':>7}{'mx/bf':>7}"
    return (f"{axis:>6} | {'PREFILL':^21} | {'DECODE':^21} | {'TOTAL':^21}\n"
            f"{'':>6} | {sub:^21} | {sub:^21} | {sub:^21}")
def _print_row(label, cells):
    g = lambda i: f"{cells[i]:>7}{cells[i+1]:>7}{cells[i+2]:>7}"
    print(f"{label:>6} | {g(0):^21} | {g(3):^21} | {g(6):^21}", flush=True)


def main():
    a = build_args()
    if a.worker: worker(a); return
    outp = os.path.join(os.path.dirname(os.path.abspath(__file__)), a.out)
    jf = open(outp, "w")
    BATCHES = [1, 8, 32, 64, 128, 256]
    LOUTS = [128, 512, 1024, 2048, 3880]
    cfgdesc = "per-scope robust (S1 u3/gs16, S2/S4/S5 u2/gs8, S3 u4/gs2)" if a.perscope else f"u{a.u}/gs{a.gs}"
    scope_cfg = lambda scn: PERSCOPE_CFG[scn] if a.perscope else (a.u, a.gs)
    print(f"[harness_batchsweep] {torch.cuda.get_device_name(0)} | MSAQ {cfgdesc} | "
          f"Llama-3.1-8B {Cfg.layers}L | ratios <1 = faster (mq=msaq, mx=mxint8, bf=bf16)", flush=True)

    def emit(scn, wstyle, kvq, fmt, u, gs, B, lin, lout):
        w, kv = fmt_paths(fmt, wstyle, kvq)
        r = spawn(a, w, kv, u, gs, B, lin, lout, f"{scn}/{fmt}")
        r.update(scope=scn, fmt=fmt, B=B, lin=lin, lout=lout)
        jf.write(json.dumps(r) + "\n"); jf.flush()
        return r

    # ---- batch sweep at canonical (1024, 512) ----
    print("\n==================== BATCH SWEEP  (L_in=1024, L_out=512) ====================", flush=True)
    for scn, wstyle, kvq in SCENARIOS:
        cu, cg = scope_cfg(scn)
        print(f"\n--- {scn} (MSAQ u{cu}/gs{cg}) ---", flush=True); print(_hdr("B"), flush=True)
        for B in BATCHES:
            row = {f: emit(scn, wstyle, kvq, f, u, gs, B, 1024, 512) for f, u, gs in variants(cu, cg)}
            _print_row(str(B), _ratiorow(row["bf16"], row["mxint8"], row["msaq"], 512))

    # ---- output sweep at fixed B=8 (L_in=1024, L_out varies; bf16 baseline must fit) ----
    OB = 8
    print(f"\n==================== OUTPUT SWEEP  (L_in=1024, B={OB}) ====================", flush=True)
    for scn, wstyle, kvq in SCENARIOS:
        cu, cg = scope_cfg(scn)
        rows = {f: emit(scn, wstyle, kvq, f, u, gs, OB, 1024, 3880) for f, u, gs in variants(cu, cg)}
        print(f"\n--- {scn} (MSAQ u{cu}/gs{cg}) ---", flush=True); print(_hdr("L_out"), flush=True)
        for lo in LOUTS:
            _print_row(str(lo), _ratiorow(rows["bf16"], rows["mxint8"], rows["msaq"], lo))
    jf.close()
    print(f"\n[harness_batchsweep] wrote {outp}", flush=True)


if __name__ == "__main__":
    main()
