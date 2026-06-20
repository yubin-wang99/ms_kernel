# Porting the project to another server

The CUDA extension (`ms_cuda`) is **compiled on the machine it runs on** — the `.so` is
tied to the exact GPU arch + CUDA toolkit + torch ABI, so you **never copy the build
artifacts**; you copy the source and rebuild.

## 1. Transfer the files

### Option A — git clone (recommended, cleanest)
Everything is pushed to the remote, so on the new server:
```bash
git clone git@github.com:yubin-wang99/ms_kernel.git
cd ms_kernel
```
(needs SSH key / GitHub access on the new server.)

### Option B — scp / rsync (your usual flow)
Copy the repo but **exclude regenerated artifacts** (`.so`, `build/`, caches). rsync is
easiest because of `--exclude`:
```bash
rsync -av --exclude='*.so' --exclude='build/' --exclude='__pycache__/' \
          --exclude='.pytest_cache/' --exclude='.git/' --exclude='.claude/' \
   /home/yubin/ssnf/ms/  user@newhost:/path/to/ms/
```
Or with scp via a tarball:
```bash
cd /home/yubin/ssnf && tar czf ms.tgz \
   --exclude='*.so' --exclude='build' --exclude='__pycache__' \
   --exclude='.pytest_cache' --exclude='.claude' ms
scp ms.tgz user@newhost:/path/to/   # then: tar xzf ms.tgz on the new host
```

**What must go** (all source): `csrc/` (`.cu`, `.cuh`, `.cpp`), `ms_lib/` (`.py`),
`tests/` (`.py`), `setup.py`, `requirements.txt`, `pytest.ini`, the `*.md` docs.
**What to skip** (regenerated / machine-specific): `ms_cuda*.so`, `build/`,
`__pycache__/`, `.pytest_cache/`, `.claude/`. (The `tests/*.jsonl` sweep data and
`results.md` are optional — keep if you want the recorded numbers, else they regenerate.)

## 2. Prerequisites on the new server
- NVIDIA GPU + driver. This project targets **RTX 3090 = sm_86**. Other GPUs need a
  one-line arch change (step 4): A100=sm_80, 4090=sm_89, H100=sm_90.
- **CUDA toolkit with `nvcc`** on PATH, version matching torch's CUDA build (we use 11.8).
  Check: `nvcc --version`.
- conda (or any Python 3.10 env manager).

## 3. Environment
```bash
conda create -n ssnf_env python=3.10 -y
conda activate ssnf_env
# torch built for the target's CUDA (cu118 here); pick the matching wheel:
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu118
pip install numpy==1.24.4 ninja pytest
```
(`ninja` is optional but makes the build ~10x faster; without it setup.py uses distutils.)

## 4. GPU arch (only if NOT an RTX 3090 / sm_86)
Edit `setup.py`, the `nvcc_flags` line:
```python
nvcc_flags = ["-O3", "-gencode", "arch=compute_86,code=sm_86", "--use_fast_math"]
```
Replace `86` with your arch (e.g. `80` for A100, `89` for RTX 4090, `90` for H100).
Find it with: `python -c "import torch; print(torch.cuda.get_device_capability(0))"`.

## 5. Build the extension
```bash
python setup.py build_ext --inplace      # produces ms_cuda*.so next to setup.py
```

## 6. Run
The package `ms_lib` has no `__init__.py` (namespace package), so **run from the repo
root with the root on PYTHONPATH**:
```bash
# correctness tests
CUDA_VISIBLE_DEVICES=0 python -m pytest tests/test_kv.py tests/test_w.py tests/test_wa.py -q

# end-to-end harness (3 models x u x gs sweep)
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python tests/harness.py            # full sweep
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python tests/harness.py --models llama31_8b --us 4 --gss 8

# KV microbenchmarks
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python tests/kv_lever_bench.py
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python tests/kv_batch_bench.py
```

## 7. Profiling (optional)
ncu needs admin perf-counter access (`sudo`, or NVreg_RestrictProfilingToAdminUsers=0).
Invoke the real binary if the `bin/ncu` shim is broken:
```bash
sudo -E env PATH="$PATH" CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. \
   <conda>/nsight-compute/<ver>/ncu --target-processes all -k "regex:<kernel>" -c 1 \
   --metrics dram__bytes_read.sum,... python tests/kv_ncu_driver.py
```

## Quick checklist
1. clone/rsync source (no `.so`/`build/`) → 2. conda env + torch(matching CUDA)+numpy+ninja
→ 3. fix `setup.py` arch if not sm_86 → 4. `python setup.py build_ext --inplace`
→ 5. `pytest tests/` to verify → 6. run harness with `PYTHONPATH=.`.
