# ms_lib/ops.py
#
# PyTorch wrappers around the compiled CUDA / CUTLASS kernels.
#
# The C++ side (csrc/pybind.cpp) registers ops under the `msaq` namespace via
# TORCH_LIBRARY, so once the extension module is imported they are reachable as
# torch.ops.msaq.wonly_gemv / wonly_gemm / wa_gemm / kv_decode_attention.
#
# torch.ops schemas take plain tensors + ints (not Python dicts), so each
# wrapper here marshals a pack dict (from ms_lib.pack) into its SoA planes:
#   scale_exp : int8   upper/shared : uint8   + the (OUT/K/NB/u/gs) meta ints.
# This is the exact device-resident layout the kernels index — keep the packed
# planes ON the GPU across timed iterations (moving them H2D per call is the
# ~48x measurement artifact noted in the kernel methodology).
#
# `available()` reports whether the compiled backend imported; the test suite
# and benchmark use it to skip GPU work cleanly on a CPU dev box.

import os

_IMPORT_ERROR = None
try:
    import torch
    import ms_cuda  # noqa: F401  — side effect: registers torch.ops.msaq.*
    _OPS = torch.ops.msaq
    _AVAILABLE = True
except Exception as e:                  # torch missing, or extension not built
    _IMPORT_ERROR = e
    _AVAILABLE = False
    _OPS = None


def available() -> bool:
    """True iff the compiled `ms_cuda` extension imported and registered ops."""
    return _AVAILABLE


def _require():
    if not _AVAILABLE:
        raise RuntimeError(
            "ms_cuda backend not available "
            f"({type(_IMPORT_ERROR).__name__}: {_IMPORT_ERROR}). "
            "Build it on a CUDA GPU:  python setup.py build_ext --inplace"
        )


# --- pack dict -> device tensors ---------------------------------------------
def _weight_planes(p, device):
    """(scale_exp int8 [nb,OUT], upper u8 [nb,UB,OUT], shared u8 [nb,SB,OUT])."""
    return (
        torch.from_numpy(p["scale_exp"]).to(device),
        torch.from_numpy(p["upper"]).to(device),
        torch.from_numpy(p["shared"]).to(device),
    )


def _kv_planes(p, device):
    """(scale_exp int8 [H,nb,L], upper u8 [H,nb,UB,L], shared u8 [H,nb,SB,L])."""
    return (
        torch.from_numpy(p["scale_exp"]).to(device),
        torch.from_numpy(p["upper"]).to(device),
        torch.from_numpy(p["shared"]).to(device),
    )


# --- ops ---------------------------------------------------------------------
def wonly_gemv(p, x_bf16):
    """W-only decode GEMV. p from pack_weight; x_bf16 [K] -> y [OUT] bf16.
    All u route to the wide-load (column-major) kernel."""
    _require()
    dev = x_bf16.device
    s = torch.from_numpy(p["scale_exp"]).to(dev)
    if "upper_cm" in p:
        up_cm = torch.from_numpy(p["upper_cm"]).to(dev)
        sh_cm = torch.from_numpy(p["shared_cm"]).to(dev)
        return _OPS.wonly_gemv_wide(x_bf16, s, up_cm, sh_cm,
                                    int(p["OUT"]), int(p["nb"]), int(p["u"]), int(p["gs"]))
    up = torch.from_numpy(p["upper"]).to(dev)
    sh = torch.from_numpy(p["shared"]).to(dev)
    return _OPS.wonly_gemv(x_bf16, s, up, sh,
                           int(p["OUT"]), int(p["nb"]),
                           int(p["u"]), int(p["gs"]))


def wa_gemv(p, x_bf16):
    """W+A decode GEMV. p from pack_weight; x_bf16 [K] -> y [OUT] bf16. Activation
    quantized to MSAQ-s on the fly (pre-pass), weight wide-load int-dot."""
    _require()
    dev = x_bf16.device
    s = torch.from_numpy(p["scale_exp"]).to(dev)
    up_cm = torch.from_numpy(p["upper_cm"]).to(dev)
    sh_cm = torch.from_numpy(p["shared_cm"]).to(dev)
    return _OPS.wa_gemv(x_bf16, s, up_cm, sh_cm,
                        int(p["OUT"]), int(p["nb"]), int(p["u"]), int(p["gs"]))


def wonly_gemm(p, X_bf16):
    """W-only prefill GEMM. p from pack_weight; X_bf16 [M,K] -> Y [M,OUT] bf16.
    Column-major wide-load unpack (coalesced; ~1.2-1.26x over the row-major path).
    MS_GEMM_ROWMAJOR=1 forces the legacy row-major kernel (A/B)."""
    _require()
    dev = X_bf16.device
    s = torch.from_numpy(p["scale_exp"]).to(dev)
    M = int(X_bf16.shape[0])
    args = (M, int(p["OUT"]), int(p["K"]), int(p["nb"]), int(p["u"]), int(p["gs"]))
    if os.environ.get("MS_GEMM_ROWMAJOR") == "1":
        s2, up, sh = _weight_planes(p, dev)
        return _OPS.wonly_gemm(X_bf16, s2, up, sh, *args)
    upc = torch.from_numpy(p["upper_cm"]).to(dev)
    shc = torch.from_numpy(p["shared_cm"]).to(dev)
    return _OPS.wonly_gemm_cm(X_bf16, s, upc, shc, *args)


def wa_gemm(p, X_bf16):
    """W+A GEMM (also serves decode at M=1). X_bf16 [M,K] -> Y [M,OUT] bf16.
    Weight unpacked to int8; activation quantized to MXINT8 on the fly (IMMA)."""
    _require()
    s, up, sh = _weight_planes(p, X_bf16.device)
    return _OPS.wa_gemm(X_bf16, s, up, sh,
                        int(X_bf16.shape[0]), int(p["OUT"]), int(p["K"]), int(p["nb"]),
                        int(p["u"]), int(p["gs"]))


def kv_decode_attention(q_bf16, pK, pV):
    """Fused-dequant flash-decode attention. q_bf16 [H,D] (one decode step);
    pK,pV from pack_kv. Returns [H,D] bf16."""
    _require()
    dev = q_bf16.device
    ks, ku, kh = _kv_planes(pK, dev)
    vs, vu, vh = _kv_planes(pV, dev)
    return _OPS.kv_decode_attention(q_bf16, ks, ku, kh, vs, vu, vh,
                                    int(q_bf16.shape[0]), int(pK["H"]), int(pK["L"]), int(pK["D"]),
                                    int(pK["nb"]), int(pK["u"]), int(pK["gs"]))


# --- plain MXINT8 baselines (pack from pack_*_mxint8: scale_exp + qweight) ----
def _mxint8_weight_planes(p, device):
    return (torch.from_numpy(p["scale_exp"]).to(device),
            torch.from_numpy(p["qweight"]).to(device))


def mxint8_gemv(p, x_bf16):
    _require()
    s, qw = _mxint8_weight_planes(p, x_bf16.device)
    return _OPS.mxint8_gemv(x_bf16, s, qw, int(p["OUT"]), int(p["nb"]))


def mxint8_wa_gemv(p, x_bf16):
    _require()
    s, qw = _mxint8_weight_planes(p, x_bf16.device)
    return _OPS.mxint8_wa_gemv(x_bf16, s, qw, int(p["OUT"]), int(p["nb"]))


def mxint8_gemm(p, X_bf16):
    _require()
    s, qw = _mxint8_weight_planes(p, X_bf16.device)
    return _OPS.mxint8_gemm(X_bf16, s, qw,
                            int(X_bf16.shape[0]), int(p["OUT"]), int(p["K"]), int(p["nb"]))


def mxint8_wa_gemm(p, X_bf16):
    _require()
    s, qw = _mxint8_weight_planes(p, X_bf16.device)
    return _OPS.mxint8_wa_gemm(X_bf16, s, qw,
                               int(X_bf16.shape[0]), int(p["OUT"]), int(p["K"]), int(p["nb"]))


def mxint8_kv_decode(q_bf16, pK, pV):
    _require()
    dev = q_bf16.device
    ks = torch.from_numpy(pK["scale_exp"]).to(dev)
    kq = torch.from_numpy(pK["qweight"]).to(dev)
    vs = torch.from_numpy(pV["scale_exp"]).to(dev)
    vq = torch.from_numpy(pV["qweight"]).to(dev)
    return _OPS.mxint8_kv_decode(q_bf16, ks, kq, vs, vq,
                                 int(q_bf16.shape[0]), int(pK["H"]), int(pK["L"]), int(pK["D"]), int(pK["nb"]))
