"""nazare_bolton.py — titration item 5: the focused wave onto the mature
from-birth champion. Refusal-triggered two-forward: pass-1 masked read;
events = rel slots whose args argmax CHANGED pass-1 vs pass-0 (the jre
rebinding criterion, ftype==rel gate); pass-2 with gmod focus (events 1.0
/ bg 0.05). Placebo: same count, random slots. Meter: converts + wrongs."""
import os, sys, json
sys.path.insert(0,'.'); sys.path.insert(0,'scripts')
import numpy as np
from phase1_algebra_head import (build_params, forward, load_alg, decode,
                                 build_slot_masks, T_ALG, L_FAC)
from repair_replace_swap import solve_forced
from tinygrad import Tensor, dtypes
from tinygrad.nn.state import safe_load
samples, states, tokmask, gold, sent = load_alg("test")
p=build_params(0); sd=safe_load(os.environ["ALG_CKPT"])
for k in p: p[k].assign(sd[k].to(p[k].device).cast(p[k].dtype)).realize()
K=("pres","ftype","op","islit","dig","args","res","query")
def fwd(slp, sm=None, gm=None):
    o=forward(p,Tensor(states[slp].astype(np.float32),dtype=dtypes.float),
                Tensor(tokmask[slp].astype(np.float32),dtype=dtypes.float),
                Tensor(sent[slp].astype(np.int32),dtype=dtypes.int),
                slot_mask=None if sm is None else Tensor(sm,dtype=dtypes.float),
                gmod=None if gm is None else Tensor(gm,dtype=dtypes.float))
    ex=tuple(k for k in ("sel","dup","sgn") if k in o)
    return {k:o[k].realize().numpy() for k in K+ex},(o["args"].realize().numpy())
# find refusals over the first 600 rows (masked two-pass read = deployment form)
ref=[]; base={}
for s0 in range(0,600,8):
    sl=list(range(s0,s0+8))
    o0,_=fwd(sl)
    mk=build_slot_masks({k:o0[k] for k in ("fat","args","res") if k in o0} if "fat" in o0 else None,None) if False else None
    # masked pass: rebuild via the standard recipe
    o0f=forward(p,Tensor(states[sl].astype(np.float32),dtype=dtypes.float),
                Tensor(tokmask[sl].astype(np.float32),dtype=dtypes.float),
                Tensor(sent[sl].astype(np.int32),dtype=dtypes.int))
    o0n={k:o0f[k].realize().numpy() for k in ("fat","args","res")}
    m=build_slot_masks(o0n,sent[sl])
    o1,a1=fwd(sl,sm=m)
    for bi,ri in enumerate(sl):
        facs,q=decode({k:o1[k][bi] for k in o1})
        ans=solve_forced(facs,q,samples[ri])
        if ans is None:
            ref.append((ri,m[bi:bi+1]))
        base[ri]=ans
print(f"[naz] refusals in first 600: {len(ref)}",flush=True)
gold_ans={i:samples[i]["solution"][samples[i]["query_var"]] for i in range(len(samples))}
rng=np.random.default_rng(41)
res={"FOCUSED":0,"PLACEBO":0}; wrong={"FOCUSED":0,"PLACEBO":0}
for ri,m in ref[:120]:
    sl=[ri]*8
    o0,a0=fwd(sl)
    o1,a1=fwd(sl,sm=np.repeat(m,8,0))
    # events: rel slots whose args argmax changed between passes
    ev=np.zeros(L_FAC,np.float32)
    for j in range(L_FAC):
        if o1["pres"][0,j]>0 and o1["ftype"][0,j].argmax()==0:
            if a0[0,j].argmax()!=a1[0,j].argmax(): ev[j]=1.0
    nev=int(ev.sum())
    if nev==0: continue
    for arm in ("FOCUSED","PLACEBO"):
        g=np.full((8,L_FAC,1),0.05,np.float32)
        if arm=="FOCUSED": idx=np.where(ev>0)[0]
        else: idx=rng.choice(L_FAC,nev,replace=False)
        g[:,idx,:]=1.0
        o2,_=fwd(sl,sm=np.repeat(m,8,0),gm=g)
        facs,q=decode({k:o2[k][0] for k in o2})
        a=solve_forced(facs,q,samples[ri])
        if a is not None:
            if a==gold_ans[ri]: res[arm]+=1
            else: wrong[arm]+=1
print(f"[naz] FOCUSED converts {res['FOCUSED']} wrongs {wrong['FOCUSED']}  |  PLACEBO converts {res['PLACEBO']} wrongs {wrong['PLACEBO']}",flush=True)
print("== NAZARE BOLTON COMPLETE ==",flush=True)
