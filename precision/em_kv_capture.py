"""Capture real K/V tensors from Llama-3.1-8B (CPU forward, GPU-independent) for EM_sharing KV study.
K: rotated by H128 along head_dim D (matching kv_ladder q_K), block axis=D. Also raw K.
V: transposed so token axis T is last, block axis=T. Layers 0 + 16."""
import os, torch, torch.nn.functional as F
os.environ.setdefault("HF_HUB_OFFLINE","1"); os.environ.setdefault("HF_DATASETS_OFFLINE","1")
torch.set_num_threads(64)
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
MODEL=os.environ.get("MSAQ_MODEL","NousResearch/Meta-Llama-3.1-8B")
OUT="/home/yubin/ms_kernel/precision"

def hadamard(n):
    H=torch.ones(1,1)
    while H.shape[0]<n: H=torch.cat([torch.cat([H,H],1),torch.cat([H,-H],1)],0)
    return H
HD=hadamard(128).float()

cap={}; LAYERS={0,16}; _n=[0]
_real=F.scaled_dot_product_attention
def patch(q,k,v,*a,**kw):
    li=_n[0]; _n[0]+=1
    if li in LAYERS and li not in cap:
        kk=k[0].float(); vv=v[0].float()          # [H,S,D]
        kr=kk@HD
        cap[li]={"K_rot":kr.reshape(-1,128).clone(),
                 "K_raw":kk.reshape(-1,128).clone(),
                 "V_tok":vv.transpose(-1,-2).reshape(-1,vv.shape[1]).clone()}  # [H*D, S]
    return _real(q,k,v,*a,**kw)
F.scaled_dot_product_attention=patch

tok=AutoTokenizer.from_pretrained(MODEL)
model=AutoModelForCausalLM.from_pretrained(MODEL,dtype=torch.float32,attn_implementation="sdpa").eval()
try: wt=load_dataset("wikitext","wikitext-2-raw-v1",split="test")
except Exception: wt=load_dataset("Salesforce/wikitext","wikitext-2-raw-v1",split="test")
ids=tok("\n\n".join(t for t in wt["text"] if t.strip()),return_tensors="pt").input_ids[:,:256]
with torch.no_grad(): model(ids)
torch.save(cap, f"{OUT}/kv_cap.pt")
for li in sorted(cap):
    for k,v in cap[li].items(): print(f"L{li} {k}: {tuple(v.shape)}", flush=True)
print("DONE saved", f"{OUT}/kv_cap.pt", flush=True)
