#!/usr/bin/env python3
# setup.py  —  build the MSAQ-signed CUDA/CUTLASS kernels into one extension.
#
#   python setup.py build_ext --inplace      # produces ms_cuda*.so next to this
#
# Produces a single `ms_cuda` module; importing it registers torch.ops.msaq.*
# (see csrc/pybind.cpp). ms_lib.ops imports it lazily.
#
# Target: RTX 3090 = Ampere GA102 = sm_86.
#
# Optional environment toggles:
#   CUTLASS_DIR=/path/to/cutlass   add CUTLASS headers to the include path (only
#                                  needed once wa_gemm.cu adopts the CUTLASS
#                                  tensor-core path; the current baselines build
#                                  WITHOUT it).
#   ENABLE_PROFILING=1             compile in the clock64() micro-profiling spans
#                                  (ms_utils.cuh). Off by default — it perturbs
#                                  scheduling; use cuda.Event + ncu for real ms.

import os
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

HERE = os.path.dirname(os.path.abspath(__file__))
CSRC = os.path.join(HERE, "csrc")

include_dirs = [CSRC, os.path.join(CSRC, "core")]
cutlass_dir = os.environ.get("CUTLASS_DIR")
if cutlass_dir:
    include_dirs.append(os.path.join(cutlass_dir, "include"))

nvcc_flags = ["-O3", "-gencode", "arch=compute_86,code=sm_86", "--use_fast_math",
              "--expt-extended-lambda"]   # device lambdas for the (u,gs)-specialized unpack helper
cxx_flags = ["-O3"]
if os.environ.get("ENABLE_PROFILING") == "1":
    nvcc_flags.append("-DENABLE_PROFILING")
    cxx_flags.append("-DENABLE_PROFILING")

setup(
    name="ms_cuda",
    ext_modules=[
        CUDAExtension(
            name="ms_cuda",
            sources=[
                os.path.join(CSRC, "pybind.cpp"),
                os.path.join(CSRC, "w_gemv.cu"),
                os.path.join(CSRC, "wa_gemm.cu"),
                os.path.join(CSRC, "kv_attention.cu"),
                os.path.join(CSRC, "rotate.cu"),
                os.path.join(CSRC, "mxint8.cu"),
            ],
            include_dirs=include_dirs,
            extra_compile_args={"cxx": cxx_flags, "nvcc": nvcc_flags},
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
