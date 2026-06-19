"""Accuracy probe: does channel-major V (MSAQ block along TOKEN = reduction axis)
degrade quantization vs token-major V (block along head_dim, ~per-token / KIVI-aligned)?
Pure numpy pack/dequant; no kernel build. Both MSAQ-s and MXINT8.
"""
import numpy as np
from ms_lib.pack import (pack_weight, pack_weight_mxint8,
                         dequant_weight, dequant_weight_mxint8)

def rel(a, b): return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-12))

def make_V(Ltok, D, seed, token_var, chan_var=1.3):
    """V[token, d]: per-token norm variation (token_var) + mild per-channel outliers."""
    rng = np.random.default_rng(seed)
    base = rng.standard_normal((Ltok, D)).astype(np.float32)
    tscale = np.exp(rng.standard_normal((Ltok, 1)) * np.log(token_var)).astype(np.float32)
    cscale = np.exp(rng.standard_normal((1, D)) * np.log(chan_var)).astype(np.float32)
    return base * tscale * cscale

def emsaq(M, u, gs): return rel(dequant_weight(pack_weight(M.astype(np.float32), u, gs)), M)
def emx(M):          return rel(dequant_weight_mxint8(pack_weight_mxint8(M.astype(np.float32))), M)

if __name__ == "__main__":
    Ltok, D = 4096, 128
    print("V dequant rel_fro. token-major = MSAQ block over head_dim (per-token-ish, "
          "current/KIVI-aligned).  channel-major = block over 32 tokens (along reduction).\n")
    for tv in (1.0, 3.0, 10.0):
        V = make_V(Ltok, D, seed=0, token_var=tv)     # [token, d]
        Vc = np.ascontiguousarray(V.T)                # [d, token]  (channel-major)
        print(f"--- token_var={tv}  (1=no token-norm spread, 10=strong) ---")
        for u in (4, 3, 2):
            print(f"  MSAQ u{u}: token-major {emsaq(V, u, 8):.3e}  |  "
                  f"channel-major {emsaq(Vc, u, 8):.3e}  "
                  f"(x{emsaq(Vc,u,8)/max(emsaq(V,u,8),1e-12):.2f})")
        print(f"  MXINT8 : token-major {emx(V):.3e}  |  channel-major {emx(Vc):.3e}  "
              f"(x{emx(Vc)/max(emx(V),1e-12):.2f})")
