# tests/conftest.py
#
# Shared pytest scaffolding for the per-scope test modules (test_w / test_wa /
# test_kv). Keeps the three files DRY: one place for the seeded RNG, the (u,gs)
# sweep, the rel_fro metric, and the "skip if the compiled CUDA backend is not
# importable" logic.
#
# Two tiers of test, mirroring the established methodology:
#   * NumPy-only (numerics + packing roundtrip + oracle self-consistency) — run
#     anywhere, no torch / no GPU. These certify pack.py + reference.py.
#   * Kernel-vs-oracle — require torch + CUDA + a built ms_lib.ops backend; they
#     SKIP cleanly otherwise (so `pytest` is green on a CPU dev box and only
#     exercises the kernels on the RTX 3090 after `python setup.py ...`).

import numpy as np
import pytest

U_VALUES = [2, 3, 4]
GS_VALUES = [2, 4, 8]
CONFIGS = [(u, gs) for u in U_VALUES for gs in GS_VALUES]

# rel_fro pass threshold for a correct bf16 / int8 kernel vs the fp64 oracle
# (fed the SAME bf16-rounded inputs). The original diagnostic lands ~0.3-1%.
REL_FRO_TOL = 2e-2


# --- GPU / backend availability ----------------------------------------------
try:
    import torch  # noqa: F401
    _HAS_TORCH = True
    _HAS_CUDA = torch.cuda.is_available()
except Exception:
    _HAS_TORCH = False
    _HAS_CUDA = False

try:
    from ms_lib import ops as _ops
    _HAS_OPS = _ops.available()
except Exception:
    _ops = None
    _HAS_OPS = False

requires_kernel = pytest.mark.skipif(
    not (_HAS_TORCH and _HAS_CUDA and _HAS_OPS),
    reason="compiled CUDA backend unavailable (build with `python setup.py build_ext "
           "--inplace` on a CUDA GPU); NumPy-oracle tests still run.",
)


@pytest.fixture
def rng():
    return np.random.default_rng(11)


@pytest.fixture(params=CONFIGS, ids=[f"u{u}_gs{gs}" for (u, gs) in CONFIGS])
def cfg(request):
    """Parametrize a test across the full u x gs sweep."""
    return request.param


# --- metrics / helpers -------------------------------------------------------
def rel_fro(a, b):
    """Relative Frobenius error ||a-b|| / ||b||  (the kernel-vs-oracle metric)."""
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    return float(np.linalg.norm((a - b).ravel()) / (np.linalg.norm(b.ravel()) + 1e-12))


def bf16np(arr):
    """f32 view of the bf16-rounded array == exactly what a bf16 kernel consumes.
    The oracle must be fed THIS so the comparison isolates kernel error, not the
    precision gap. Requires torch (only used by kernel-vs-oracle tests)."""
    import torch
    return torch.from_numpy(np.asarray(arr)).to(torch.bfloat16).float().numpy()