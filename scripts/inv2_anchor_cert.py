"""Anchor cert on ictl's mod-novel wins (0.986 = the fdiv standard;
reading vs arriving decides whether the pair-data effect promotes)."""
import os, sys, json
for k,v in [("ALG2","1"),("ALG_FTYPES","8"),("ALG_HW","512"),("ALG_DUP","1")]: os.environ.setdefault(k,v)
sys.path.insert(0,'.'); sys.path.insert(0,'scripts')
import numpy as np
from phase1_algebra_head import T_ALG, build_params, forward, decode, sent_indices, TOKENIZER_JSON
from beacon_closing_arm import recompute_states
from tokenizers import Tokenizer
from tinygrad import Tensor, dtypes
from tinygrad.nn.state import safe_load
tok=Tokenizer.from_file(TOKENIZER_JSON)
rng=np.random.RandomState(93000); rows=[]
while len(rows)<8:
    K=int(rng.choice([3,4,5,6,7])); A,B=int(rng.randint(2,12)),int(rng.randint(2,12))
    C=A*B; g=C%K
    if C>300 or g==0: continue
    rows.append(f"Consider the numbers a, b, c, d. a is {A}. b is {B}. a times b equals c. After dividing c by {K}, d is what remains. What is d?")
p=build_params(0); sd=safe_load(".cache/g25v2_inv_ctl.safetensors")
for k in p: p[k].assign(sd[k].to(p[k].device).cast(p[k].dtype)).realize()
hits=0; mass=[]
for t in rows:
    ids=np.zeros((8,T_ALG),np.int32); msk=np.zeros((8,T_ALG),np.float32); snt=np.zeros((8,T_ALG),np.int32)
    e=tok.encode(t); Ln=min(len(e.ids),T_ALG)
    ids[0,:Ln]=e.ids[:Ln]; msk[0,:Ln]=1.0
    snt[0]=sent_indices(t,list(e.offsets),msk[0])
    o=forward(p,Tensor(recompute_states(ids).astype(np.float32),dtype=dtypes.float),
              Tensor(msk,dtype=dtypes.float),Tensor(snt,dtype=dtypes.int))
    keys=("pres","ftype","op","islit","dig","args","res","query","fat")+(("sel",) if "sel" in o else ())+(("dup",) if "dup" in o else ())
    onp={k:o[k].realize().numpy() for k in keys}
    facs=decode({k:onp[k][0] for k in onp if k!="fat"})[0]
    mj=[j for j,f in enumerate(facs) if f.get("ftype")=="mod"]
    if not mj: continue
    a=onp["fat"][0,mj[0]]; a=a/max(a.sum(),1e-9)
    m=float(a[(snt[0]==4)&(msk[0]>0)].sum()); mass.append(m)
    hits+=int(snt[0][a.argmax()]==4)
print(f"[cert:ictl] mod slot decoded {len(mass)}/8; anchored-to-mod-sentence {hits}/8; span mass mean {np.mean(mass) if mass else float('nan'):.3f}")
