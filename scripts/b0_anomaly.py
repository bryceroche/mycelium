"""b0_anomaly.py — does breaths[0] twin the silent state? Three-way:
twin holds (=> pass_econ had a bug; recheck inline); twin differs
(instrument fault at the exposure); partial (mechanism: pre-boundary
rebinding)."""
import os, sys, json
os.environ["ALG_BREATH"]="3"
for k,v in [("ALG2","1"),("ALG_FTYPES","8"),("ALG_HW","512"),("ALG_DUP","1"),("ALG_WIDE","1"),("ALG_INV","1")]: os.environ.setdefault(k,v)
sys.path.insert(0,'.'); sys.path.insert(0,'scripts')
import numpy as np
from phase1_algebra_head import T_ALG, build_params, forward, L_FAC, build_slot_masks, sent_indices, TOKENIZER_JSON
from beacon_closing_arm import recompute_states
from tokenizers import Tokenizer
from tinygrad import Tensor, dtypes
from tinygrad.nn.state import safe_load
tok=Tokenizer.from_file(TOKENIZER_JSON)
p=build_params(0); sd=safe_load(".cache/g51_whisper.safetensors")
for k in p: p[k].assign(sd[k].to(p[k].device).cast(p[k].dtype)).realize()
def ap(a,d,j):
    if d[j]>0:
        a0=int(np.argmax(a[j])); return (a0,a0)
    return tuple(sorted(np.argsort(-a[j])[:2].tolist()))
L="abcdefghij"
def fixture_mint(nd, n=15, seed=96000):
    rng=np.random.RandomState(seed+nd); rows=[]
    while len(rows)<n:
        op="add" if rng.rand()<0.5 else "mul"
        x=int(rng.randint(2,60)) if op=="add" else int(rng.randint(2,13))
        gg=x+x if op=="add" else x*x
        if gg>300: continue
        gv=[int(rng.randint(2,90)) for _ in range(nd)]
        dv=nd; res=nd+1
        w="{a} plus another {a} makes {c}." if op=="add" else "{a} lots of {a} make {c}."
        sents=[f"{L[i]} is {gv[i]}." for i in range(nd)]+[f"{L[dv]} is {x}.", w.format(a=L[dv],c=L[res])]
        rows.append(r"".join([]) or {"text":f"Consider the numbers {', '.join(L[:res+1])}. "+" ".join(sents)+f" What is {L[res]}?","dv":dv,"op":op})
    return rows
rows4=fixture_mint(4)
twin_diffs=0; slots_checked=0; rec=0; tot=0
for s0 in range(0,15,8):
    ch=rows4[s0:s0+8]
    ids=np.zeros((8,T_ALG),np.int32); msk=np.zeros((8,T_ALG),np.float32); snt=np.zeros((8,T_ALG),np.int32)
    for i,r in enumerate(ch):
        e=tok.encode(r["text"]); Ln=min(len(e.ids),T_ALG)
        ids[i,:Ln]=e.ids[:Ln]; msk[i,:Ln]=1.0
        snt[i]=sent_indices(r["text"],list(e.offsets),msk[i])
    tr=Tensor(recompute_states(ids).astype(np.float32),dtype=dtypes.float)
    tk=Tensor(msk,dtype=dtypes.float); se=Tensor(snt,dtype=dtypes.int)
    o0=forward(p,tr,tk,se)
    sa=o0["args"].realize().numpy(); sdp=o0["dup"].realize().numpy(); sp=o0["pres"].realize().numpy()
    o0n={"fat":o0["fat"].realize().numpy(),"args":sa,"res":o0["res"].realize().numpy()}
    mk=build_slot_masks(o0n, snt)
    oe=forward(p,tr,tk,se,slot_mask=Tensor(mk,dtype=dtypes.float))
    b0=oe["breaths"][0]
    ba=b0["args"].realize().numpy(); bd=b0["dup"].realize().numpy()
    fa=oe["args"].realize().numpy(); fd=oe["dup"].realize().numpy(); fp=oe["pres"].realize().numpy()
    for i,r in enumerate(ch):
        for j in range(L_FAC):
            if sp[i,j]<=0: continue
            slots_checked+=1
            if ap(sa[i],sdp[i],j)!=ap(ba[i],bd[i],j): twin_diffs+=1
        jre=-1
        for j in range(L_FAC):
            if fp[i,j]>0 and fd[i,j]>0 and int(np.argmax(fa[i,j]))==r["dv"]: jre=j; break
        if jre>=0:
            tot+=1
            rec+= (ap(ba[i],bd[i],jre)!=ap(fa[i],fd[i],jre))
print(f"[b0] twin check: {twin_diffs} argpair diffs / {slots_checked} silent-on slots (0 = twin holds)",flush=True)
print(f"[b0] inline recompute b0-vs-final rebound recall: {rec}/{tot}",flush=True)
