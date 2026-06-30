"""Free-index two-level residual — recovery-rate (rho) gate.

Idea: native MX base (FP4/FP6/INT4/INT6) + a correction that approximates the residual R=W-W_hat by a
GLOBAL scalar codebook, where each element's codebook INDEX is DERIVED FROM the base encoding (free, no
storage) instead of from the residual. correction[elem] = codebook[ index(base_elem) ] * ulp[elem].

We do NOT judge absolute accuracy — we judge the RECOVERY RATE against two baselines:
  zero   = base only, no correction               (lower bound)
  oracle = best K-entry scalar VQ of the residual  (upper bound; Lloyd-Max, nearest assignment)
  rho = (QSNR(cand) - QSNR(zero)) / (QSNR(oracle) - QSNR(zero))
rho~0 => index carries no residual info (H1 rejected for that index); rho->1 => free index ~= oracle.

Hypotheses:  H1 some free index carries residual info (info-theoretically only via base cell-curvature
bias in mantissa low bits, since tau carries only scale and RTN residual value ~ indep of kept bits).
H2 coarser base => more recoverable (rho FP4/INT4 > FP6/INT6).

Design decisions fixed in code (per spec caveats):
- C6 normalization: codebook operates on the ulp-NORMALIZED residual rn = r/ulp (so index ②/⑥ are
  cell-relative, not scale-mixed). ulp = exact local grid step (FP) / 1 (INT, scaled domain).
- Codebook is learned IN-DISTRIBUTION per (tensor, base) — so rho measures the index's pure information
  content (an upper bound on any real global/per-scope table; C3 deferred).
- Indices ①②③④ are FREE (from base); ⑤⑥⑦ are STORED references. ⑦ (VQ)=oracle assignment => rho=1
  by construction (sanity). Sign guard B adds a stored residual-sign bit (flagged).

Run:  CUDA_VISIBLE_DEVICES=0 .venv/bin/python precision/index_residual.py
"""
import torch
from mxfp6_verify import _fp6_grid
from em_sharing import _base_encode, qsnr, BASES, _load_layers, BLOCK

DEV = "cuda" if torch.cuda.is_available() else "cpu"


# ----------------------------------------------------------------- base decode: codes, residual, ulp
def base_residual(W, base):
    """Return (y, code, r, ulp) in the per-block-scaled domain. code=quantized base, r=y-code residual,
    ulp=local grid step per element (FP exact via grid; INT=1)."""
    spec = BASES[base]
    xf = W.reshape(-1, BLOCK).to(torch.float32)
    s, snap = _base_encode(xf, spec)
    y = xf / s
    code = snap(y)
    r = y - code
    if spec["kind"] == "int":
        ulp = torch.ones_like(code)
    else:
        grid = _fp6_grid(spec["eb"], spec["mb"])              # sorted nonneg magnitudes
        steps = torch.diff(grid)                              # gap above each grid point
        a = code.abs()
        idx = torch.bucketize(a, grid, right=False).clamp(0, grid.numel() - 2)
        ulp = steps[idx]                                      # step at the element's base magnitude
    return y, code, r, ulp, s, spec


# ----------------------------------------------------------------------------------- index functions
def _qbucket(v, K):
    """Equal-population bucketing of v into K buckets via global quantiles (on a sample for speed)."""
    flat = v.reshape(-1)
    n = flat.numel()
    samp = flat[torch.randint(0, n, (min(n, 1_000_000),), device=flat.device)] if n > 1_000_000 else flat
    qs = torch.quantile(samp.float(), torch.linspace(0, 1, K + 1, device=flat.device)[1:-1])
    return torch.bucketize(v, qs).clamp(0, K - 1)


def index_fn(name, K, y, code, r, ulp, spec):
    """Return per-element bucket in [0,K-1] for the named index. FREE: tau, mant, rank, taumant."""
    if name == "tau":                                        # ① exponent shift (scale only)
        q = torch.floor(torch.log2(y.abs().clamp(min=1e-30)))
        return _qbucket(q, K)
    if name == "mant":                                       # ② base mantissa low bits (ulp-normalized)
        if spec["kind"] == "fp":
            # mantissa position = (|code| / ulp) integer part's low bits (cell identity)
            mi = torch.round(code.abs() / ulp).to(torch.int64)
        else:
            mi = torch.round(code).abs().to(torch.int64)     # INT code magnitude
        return (mi % K).clamp(0, K - 1)
    if name == "rank":                                       # ③ magnitude rank within block
        order = y.abs().argsort(-1).argsort(-1)              # 0..BLOCK-1 rank per row
        return (order * K // BLOCK).clamp(0, K - 1).reshape(y.shape)
    if name == "taumant":                                    # ④ tau × mantissa-low-bit
        bt = index_fn("tau", max(2, K // 2), y, code, r, ulp, spec)
        bm = index_fn("mant", 2, y, code, r, ulp, spec)
        return (bt * 2 + bm).clamp(0, K - 1)
    if name == "sign":                                       # ⑤ residual sign (STORED, K=2)
        return (r > 0).to(torch.int64)
    if name == "rhi":                                        # ⑥ residual high bits (STORED)
        return _qbucket(r / ulp, K)
    raise ValueError(name)


# ---------------------------------------------------------------------------- codebook + reconstruct
def _bucket_mean(vals, idx, K):
    sums = torch.zeros(K, device=vals.device).scatter_add_(0, idx, vals)
    cnts = torch.zeros(K, device=vals.device).scatter_add_(0, idx, torch.ones_like(vals))
    return sums / cnts.clamp(min=1.0)


def candidate_corr(rn, idx, K, sign_guard, r):
    """Learn per-bucket codebook on normalized residual rn and return per-element correction (in rn
    units). sign_guard: off=bucket mean; B=split bucket by residual-sign (stored +1b); A=zero buckets
    whose sign is mixed (|mean|/mean|.| < 0.34)."""
    fi = idx.reshape(-1); rf = rn.reshape(-1)
    if sign_guard == "B":
        sgn = (r.reshape(-1) > 0).to(torch.int64)
        comb = fi * 2 + sgn
        ent = _bucket_mean(rf, comb, K * 2)
        return ent[comb].reshape(rn.shape)
    ent = _bucket_mean(rf, fi, K)
    if sign_guard == "A":
        absmean = _bucket_mean(rf.abs(), fi, K)
        keep = (ent.abs() / absmean.clamp(min=1e-30)) >= 0.34   # drop sign-cancelled buckets
        ent = ent * keep
    return ent[fi].reshape(rn.shape)


def lloyd1d(rn, K, iters=12):
    """1D Lloyd-Max on normalized residual -> oracle per-element correction (nearest of K levels)."""
    flat = rn.reshape(-1)
    n = flat.numel()
    samp = flat[torch.randint(0, n, (min(n, 1_000_000),), device=flat.device)] if n > 1_000_000 else flat
    c = torch.quantile(samp.float(), torch.linspace(0, 1, 2 * K + 1, device=flat.device)[1::2])
    c = c.contiguous()
    for _ in range(iters):
        mids = (c[1:] + c[:-1]) / 2
        a = torch.bucketize(flat, mids).clamp(0, K - 1)
        cnts = torch.zeros(K, device=flat.device).scatter_add_(0, a, torch.ones_like(flat))
        nc = _bucket_mean(flat, a, K)
        c = torch.where(cnts > 0, nc, c)                      # keep centroid if cluster empty
    mids = (c[1:] + c[:-1]) / 2
    a = torch.bucketize(rn.reshape(-1), mids).clamp(0, K - 1)
    return c[a].reshape(rn.shape)


def recon_qsnr(W, base, corr_rn, y, code, ulp, s):
    recon = ((code + corr_rn * ulp) * s).reshape(W.shape)
    return qsnr(W, recon)


# ---------------------------------------------------------------------------------------------- sweep
FREE = ["tau", "mant", "rank", "taumant"]
STORED = ["sign", "rhi"]
SYMBOL = {"tau": "① τ", "mant": "② mant-lo", "rank": "③ rank", "taumant": "④ τ×mant",
          "sign": "⑤ r-sign*", "rhi": "⑥ r-hi*", "vq": "⑦ VQ*"}


def _rho(W, base, nm, K, sg, y, code, r, ulp, s, spec, q_zero, orac):
    corr = candidate_corr(r / ulp, index_fn(nm, K, y, code, r, ulp, spec), K, sg, r)
    q = recon_qsnr(W, base, corr, y, code, ulp, s)
    denom = orac - q_zero
    return (q - q_zero) / denom if abs(denom) > 1e-6 else 0.0


def sweep_tensor(name, W):
    W = W.to(DEV).float()
    lines = [f"\n### {name}  shape={tuple(W.shape)}"]
    for base in ["FP4", "INT4", "FP6", "INT6"]:
        y, code, r, ulp, s, spec = base_residual(W, base)
        rn = r / ulp
        q_zero = qsnr(W, (code * s).reshape(W.shape))
        orac = {K: recon_qsnr(W, base, lloyd1d(rn, K), y, code, ulp, s) for K in (2, 4, 8, 16)}
        lines.append(f"\n**{base}** (zero QSNR = {q_zero:.2f} dB; oracle Δ per K below)")
        lines.append("| index | K=2 | K=4 | K=8 | K=16 |")
        lines.append("|" + "---|" * 5)
        lines.append("| **oracle** (ΔdB) | " + " | ".join(f"{orac[K]-q_zero:+.2f}" for K in (2, 4, 8, 16)) + " |")
        # MAIN TABLE: pure-free rho (sign-guard OFF; no stored bits beyond the index itself)
        for nm in FREE + STORED + ["vq"]:
            cells = []
            for K in (2, 4, 8, 16):
                if nm == "vq":
                    q = recon_qsnr(W, base, lloyd1d(rn, K), y, code, ulp, s)
                    rho = (q - q_zero) / (orac[K] - q_zero) if abs(orac[K] - q_zero) > 1e-6 else 0.0
                else:
                    rho = _rho(W, base, nm, K, "off", y, code, r, ulp, s, spec, q_zero, orac[K])
                cells.append(f"{rho:.2f}")
            lines.append(f"| {SYMBOL[nm]} | " + " | ".join(cells) + " |")
        # SIGN-GUARD B uplift for FREE indices (B adds +1 STORED residual-sign bit -> not free anymore)
        b2 = []
        for nm in FREE:
            r2 = _rho(W, base, nm, 2, "B", y, code, r, ulp, s, spec, q_zero, orac[2])
            b2.append(f"{SYMBOL[nm]}={r2:.2f}")
        lines.append(f"_sign-guard B (+1 stored bit), K=2 ρ:_ " + ", ".join(b2))
    return "\n".join(lines)


def main():
    torch.manual_seed(0)
    out = []
    for name, W in _load_layers().items():
        out.append(sweep_tensor(name, W))
    print("\n".join(out))


if __name__ == "__main__":
    main()
