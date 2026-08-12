"""pass_econ.py — the pass-economics read (the word given): can the
event field come from ONE engaged forward's own trajectory
(breath-1 vs final argpairs) instead of the cross-grain pair?
15/15-class recall = the three-forward organism collapses to two."""
import os, sys, json
os.environ["ALG_BREATH"]="3"
for k,v in [("ALG2","1"),("ALG_FTYPES","8"),("ALG_HW","512"),("ALG_DUP","1"),("ALG_WIDE","1"),("ALG_INV","1")]: os.environ.setdefault(k,v)
os.environ.setdefault("ALG_TEST",".cache/algebra_nl_bigtest.jsonl"); os.environ.setdefault("ALG_TEST_NAME","bigtest")
sys.path.insert(0,'.'); sys.path.insert(0,'scripts')
import numpy as np
from phase1_algebra_head import T_ALG, build_params, forward, load_alg, L_FAC, build_slot_masks, sent_indices, TOKENIZER_JSON
from beacon_closing_arm import recompute_states
from tokenizers import Tokenizer
from tinygrad import Tensor, dtypes
from tinygrad.nn.state import safe_load
tok=Tokenizer.from_file(TOKENIZER_JSON)
p=build_params(0); sd=safe_load(".cache/g51_whisper.safetensors")
for k in p: p[k].assign(sd[k].to(p[k].device).cast(p[k].dtype)).realize()
def argpair_np(args_row,dup_row,j):
    if dup_row[j]>0:
        a0=int(np.argmax(args_row[j])); return (a0,a0)
    return tuple(sorted(np.argsort(-args_row[j])[:2].tolist()))
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
        rows.append({"text":f"Consider the numbers {', '.join(L[:res+1])}. "+" ".join(sents)+f" What is {L[res]}?","dv":dv,"op":op})
    return rows
rows4=fixture_mint(4); rec=0; tot=0; SP=[]
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
    o0n={k2:o0[k2].realize().numpy() for k2 in ("fat","args","res")}
    mk=build_slot_masks(o0n, snt)
    oe=forward(p,tr,tk,se,slot_mask=Tensor(mk,dtype=dtypes.float))
    assert "breaths" in oe, "breaths not exposed"
    b0=oe["breaths"][0]; bf=oe   # breath-1 heads vs final heads
    b0a=b0["args"].realize().numpy(); b0d=b0["dup"].realize().numpy() if "dup" in b0 else np.zeros_like(b0a[...,0])
    bfa=oe["args"].realize().numpy(); bfd=oe["dup"].realize().numpy()
    bfp=oe["pres"].realize().numpy(); bft=oe["ftype"].realize().numpy()
    end={"args":bfa,"dup":bfd}
    for i,r in enumerate(ch):
        ev=np.zeros(L_FAC)
        for j in range(L_FAC):
            if bfp[i,j]<=0: continue
            if argpair_np(b0a[i],b0d[i],j)!=argpair_np(bfa[i],bfd[i],j): ev[j]=1.0
        # the rebound slot per the engaged decode
        jre=-1
        for j in range(L_FAC):
            if bfp[i,j]>0 and bft[i,j].argmax()==0 \
               and bfd[i,j]>0 and int(np.argmax(bfa[i,j]))==r["dv"]: jre=j; break
        if jre>=0:
            tot+=1; rec+= bool(ev[jre]>0); SP.append(int(ev.sum()))
print(f"[pass-econ] INTRA-PASS events (breath1 vs final): rebound recall {rec}/{tot}  sparsity median {int(np.median(SP)) if SP else -1}",flush=True)
json.dump({"recall":rec,"tot":tot,"sparsity":SP},open('.cache/pass_econ.json','w'))
