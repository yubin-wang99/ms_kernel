"""One-shot driver for ncu: a few warm iters then ONE timed call of the chosen kernel.
Usage: KV_WHICH=msaq|mx KV_U=4 KV_LK=4680 KV_MODE=wide|gqa python tests/kv_ncu_driver.py
"""
import os, sys, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ms_lib import ops; assert ops.available()
import kv_lever_bench as B  # reuse setup

which = os.environ.get("KV_WHICH", "msaq")
u  = int(os.environ.get("KV_U", "4"))
Lk = int(os.environ.get("KV_LK", "4680"))
H  = int(os.environ.get("KV_H", "8"))
Hkv= int(os.environ.get("KV_HKV", "8"))
torch.cuda.init()
d = B.setup(u=u, H=H, Hkv=Hkv, Lk=Lk, D=128)
fn = d[which]
for _ in range(20): fn()
torch.cuda.synchronize()
fn()   # the call ncu profiles (filter by kernel-name regex)
torch.cuda.synchronize()
