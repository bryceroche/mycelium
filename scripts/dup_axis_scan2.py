import os, sys, json
for k,v in [("ALG2","1"),("ALG_FTYPES","8"),("ALG_HW","512"),("ALG_DUP","1")]: os.environ.setdefault(k,v)
sys.path.insert(0,'.'); sys.path.insert(0,'scripts')
import numpy as np
from phase1_algebra_head import T_ALG, build_params, forward, decode, sent_indices, TOKENIZER_JSON
from beacon_closing_arm import recompute_states
from tokenizers import Tokenizer
from tinygrad import Tensor, dtypes
from tinygrad.nn.state import safe_load
L="abcdefghij"
tok=Tokenizer.from_file(TOKENIZER_JSON)
def mint(nd, n=15, seed=96000):
    rng=np.random.RandomState(seed+nd); rows=[]
    while len(rows)<n:
        op="add" if rng.rand()<0.5 else "mul"
        x=int(rng.randint(2,60)) if op=="add" else int(rng.randint(2,13))
        gold=x+x if op=="add" else x*x
        if gold>300: continue
        gv=[int(rng.randint(2,90)) for _ in range(nd)]
        dv=nd; res=nd+1
        w="{a} plus another {a} makes {c}." if op=="add" else "{a} lots of {a} make {c}."
        sents=[f"{L[i]} is {gv[i]}." for i in range(nd)]+[f"{L[dv]} is {x}.", w.format(a=L[dv],c=L[res])]
        rows.append({"text":f"Consider the numbers {', '.join(L[:res+1])}. "+" ".join(sents)+f" What is {L[res]}?","dv":dv,"op":op})
    return rows
p=build_params(0); sd=safe_load(os.environ["CK"])
for k in p: p[k].assign(sd[k].to(p[k].device).cast(p[k].dtype)).realize()
def one(t):
    ids=np.zeros((8,T_ALG),np.int32); msk=np.zeros((8,T_ALG),np.float32); snt=np.zeros((8,T_ALG),np.int32)
    e=tok.encode(t); Ln=min(len(e.ids),T_ALG)
    ids[0,:Ln]=e.ids[:Ln]; msk[0,:Ln]=1.0
    snt[0]=sent_indices(t,list(e.offsets),msk[0])
    _tr=Tensor(recompute_states(ids).astype(np.float32),dtype=dtypes.float)
    _tk=Tensor(msk,dtype=dtypes.float); _se=Tensor(snt,dtype=dtypes.int)
    out=forward(p,_tr,_tk,_se)
    if os.environ.get("TWO_PASS")=="1" and "W_bo" in p:
        from phase1_algebra_head import build_slot_masks
        _o0={k:out[k].realize().numpy() for k in ("fat","args","res")}
        _mk=build_slot_masks(_o0, snt)
        out=forward(p,_tr,_tk,_se,slot_mask=Tensor(_mk,dtype=dtypes.float))
    keys=("pres","ftype","op","islit","dig","args","res","query")+(("sel",) if "sel" in out else ())+(("dup",) if "dup" in out else ())
    o={k:out[k].realize().numpy() for k in keys}
    return decode({k:o[k][0] for k in o})[0]
res={}
for nd in (0,1,2,4):
    rows=mint(nd); mis=0
    for r in rows:
        facs=one(r["text"])
        ok=any(f.get("ftype")=="rel" and f.get("args")==[r["dv"],r["dv"]] and f.get("op")==r["op"] for f in facs)
        mis+=(not ok)
    res[nd]=mis
    print(f"[scan] distractors={nd}: misbind {mis}/15",flush=True)
json.dump(res,open('.cache/dup_axis_scan2.json','w'))
