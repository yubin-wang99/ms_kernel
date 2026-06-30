"""Full VQ as the stored 2nd-level residual (free-index successor).

The free-index gate (index_residual.md) rejected H1: no base-derived index recovers the residual, and
only STORED bits do. The strongest stored option is VQ. But per-element SCALAR VQ (the index_residual
oracle) costs log2(K) b/elem + a full per-element second GEMM (caveat C2) — no better than just adding
mantissa bits. The serious question is VECTOR VQ: one stored index per group of g elements selecting a
g-dim residual codeword, storage = log2(K)/g b/elem (the cheap sub-bit regime that scalar VQ can't
reach). It only pays if the residual VECTOR has exploitable intra-group STRUCTURE.

This file answers, on real Llama-3.1-8B weights, in the ulp-normalized residual domain (C6):
  (1) STRUCTURE: PCA energy concentration of the g-dim residual + adjacent-element correlation.
      White residual (top-PC energy ~ 1/g) => no structure => vector VQ cannot beat scalar at iso-bits.
  (2) RECOVERY: vector-VQ QSNR gain over `zero` (base only) across (g, K) at fractional storage, vs
      per-element scalar VQ at >=1 b/elem. ΔdB and rho = ΔdB(VQ) / ΔdB(scalar @ matched/ref bits).

Run:  CUDA_VISIBLE_DEVICES=0 .venv/bin/python precision/vq_residual.py
"""
import torch
from em_sharing import qsnr, _load_layers, BLOCK
from index_residual import base_residual

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def _vq_fit_assign(X, K, iters=10, fit_cap=300_000):
    """1-level vector k-means (Lloyd). Fit codebook on a sample, assign ALL rows (chunked).
    X: [N, g] zero-mean residual vectors. Returns reconstruction [N, g]."""
    N, g = X.shape
    fit = X[torch.randint(0, N, (min(N, fit_cap),), device=X.device)] if N > fit_cap else X
    c = fit[torch.randint(0, fit.shape[0], (K,), device=X.device)].clone()
    for _ in range(iters):
        d = (fit * fit).sum(1, keepdim=True) - 2 * fit @ c.t() + (c * c).sum(1)[None]
        a = d.argmin(1)
        sums = torch.zeros(K, g, device=X.device).index_add_(0, a, fit)
        cnts = torch.zeros(K, device=X.device).index_add_(0, a, torch.ones_like(a, dtype=fit.dtype))
        nc = sums / cnts[:, None].clamp(min=1)
        c = torch.where(cnts[:, None] > 0, nc, c)
    out = torch.empty_like(X)
    for i in range(0, N, 2_000_000):                                   # chunked nearest assignment
        xb = X[i:i + 2_000_000]
        d = (xb * xb).sum(1, keepdim=True) - 2 * xb @ c.t() + (c * c).sum(1)[None]
        out[i:i + 2_000_000] = c[d.argmin(1)]
    return out


def structure_diag(rn_block, g):
    """rn_block: [rows, BLOCK] normalized residual. Reshape to g-dim group vectors and report PCA energy
    concentration (top-1/2/4 of g) + mean adjacent-element correlation. Residual is ~zero-mean."""
    X = rn_block.reshape(-1, g).float()
    X = X - X.mean(0, keepdim=True)
    cov = (X.t() @ X) / X.shape[0]
    ev = torch.linalg.eigvalsh(cov).flip(0).clamp(min=0)               # descending eigenvalues
    tot = ev.sum().clamp(min=1e-30)
    frac = (ev / tot)
    eff_rank = (tot ** 2 / (ev * ev).sum().clamp(min=1e-30)) / g       # participation ratio / g
    # adjacent correlation across the group axis
    a, b = X[:, :-1], X[:, 1:]
    ac = ((a * b).mean() / (a.std() * b.std()).clamp(min=1e-30)).item()
    return frac, eff_rank.item(), ac


def main():
    torch.manual_seed(0)
    BASES = ["FP4", "INT4", "FP6", "INT6"]
    # (g, K) grid -> storage log2(K)/g b/elem
    VQ_CFGS = [(g, K) for g in (4, 8, 16, 32) for K in (4, 16, 64, 256)]
    print("# Full VQ residual — structure + recovery on Llama-3.1-8B weights\n")
    for name, W in _load_layers().items():
        W = W.to(DEV).float()
        print(f"\n## {name}  shape={tuple(W.shape)}")
        for base in BASES:
            y, code, r, ulp, s, spec = base_residual(W, base)
            rn = r / ulp
            q_zero = qsnr(W, (code * s).reshape(W.shape))

            # --- structure diagnostic (g=8 representative; report 1/g white reference) ---
            frac, effr, ac = structure_diag(rn, 8)
            print(f"\n### {base}  (zero QSNR {q_zero:.2f} dB)")
            print(f"  structure (g=8): top-1/2/4 PC energy = "
                  f"{frac[0]*100:.1f}% / {(frac[:2].sum())*100:.1f}% / {(frac[:4].sum())*100:.1f}%  "
                  f"(white ref top-1 = {100/8:.1f}%); eff_rank/g = {effr:.2f}; adj-corr = {ac:+.3f}")

            # --- scalar VQ reference (per-element, g=1): the C2-expensive ceiling at >=1 b/elem ---
            def recon_q(corr): return qsnr(W, ((code + corr * ulp) * s).reshape(W.shape))
            sca = {}
            for bits, K in ((1, 2), (2, 4)):
                corr = _vq_fit_assign(rn.reshape(-1, 1), K).reshape(rn.shape)
                sca[bits] = recon_q(corr) - q_zero
            print(f"  scalar VQ (per-elem): Δ@1b = {sca[1]:+.2f} dB,  Δ@2b = {sca[2]:+.2f} dB")

            # --- vector VQ Pareto: ΔdB over zero vs storage ---
            rows = []
            for g, K in VQ_CFGS:
                stor = (K.bit_length() - 1) / g                       # log2(K)/g
                corr = _vq_fit_assign(rn.reshape(-1, g), K).reshape(rn.shape)
                rows.append((stor, recon_q(corr) - q_zero, g, K))
            rows.sort()
            # Pareto front (max ΔdB at <= storage)
            print(f"  vector VQ (storage b/elem -> ΔdB, g/K):")
            best = -1e9
            for stor, d, g, K in rows:
                mark = ""
                if d > best + 1e-6:
                    best = d; mark = " ★"
                print(f"    {stor:5.3f} b -> {d:+6.2f} dB  (g{g}/K{K}){mark}")


if __name__ == "__main__":
    main()
