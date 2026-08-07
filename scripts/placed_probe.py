import os, sys, json
for k,v in [("ALG2","1"),("ALG_FTYPES","8"),("ALG_HW","512"),("ALG_DUP","1")]: os.environ.setdefault(k,v)
sys.path.insert(0,'.'); sys.path.insert(0,'scripts')
import numpy as np
from collections import Counter
from phase1_algebra_head import T_ALG, build_params, forward, decode, sent_indices, TOKENIZER_JSON
from beacon_closing_arm import recompute_states
from tta_views import permuted_view
from tta_alg2_dials import solve2
from tokenizers import Tokenizer
from tinygrad import Tensor, dtypes
from tinygrad.nn.state import safe_load
tok=Tokenizer.from_file(TOKENIZER_JSON)
T=json.load(open('.cache/aug_table_v3.json'))["licensed"]
CANDS=[("mod2","The leftover after dividing {a} by {k} is {b}.","leftover"),
       ("mod3","{a} divided by {k} has {b} left at the end.","left at the end"),
       ("fdiv2","Sharing {a} across {k} people leaves {b} apiece.","apiece"),
       ("pct2","{p} percent taken from a hundred parts of {b2} is {p2}.","hundred parts")]
for _,_,g in CANDS: assert not any(g in e["fmt"] for e in T), g
print("[guard] all candidates ABSENT")
def mint(fmt, kind, n=8, seed=95000):
    rng=np.random.RandomState(seed); rows=[]
    while len(rows)<n:
        if kind.startswith("mod"):
            K=int(rng.choice([3,4,5,6,7])); A,B=int(rng.randint(2,12)),int(rng.randint(2,12))
            C=A*B; g=C%K
            if C>300 or g==0: continue
            s=fmt.format(a="c",k=K,b="d")
            rows.append({"text":f"Consider the numbers a, b, c, d. a is {A}. b is {B}. a times b equals c. {s} What is d?","q":3,"gold":g})
        elif kind.startswith("fdiv"):
            K=int(rng.choice([2,3,4,5,6])); B=int(rng.randint(2,12)); A=K*B
            if A>300: continue
            s=fmt.format(a="a",k=K,b="b")
            rows.append({"text":f"Consider the numbers a, b. a is {A}. {s} What is b?","q":1,"gold":B})
        else:
            p=int(rng.choice([10,20,25,50])); b2=int(rng.randint(2,12))*20; g=p*b2//100
            if b2>300 or not(1<=g<=300): continue
            s=fmt.format(p=p,b2="a",p2="b")
            rows.append({"text":f"Consider the numbers a, b. a is {b2}. {s} What is b?","q":1,"gold":g})
    return rows
def load(ck):
    p=build_params(0); sd=safe_load(ck)
    for k in p: p[k].assign(sd[k].to(p[k].device).cast(p[k].dtype)).realize()
    return p
def one(p,t):
    ids=np.zeros((8,T_ALG),np.int32); msk=np.zeros((8,T_ALG),np.float32); snt=np.zeros((8,T_ALG),np.int32)
    e=tok.encode(t); Ln=min(len(e.ids),T_ALG)
    ids[0,:Ln]=e.ids[:Ln]; msk[0,:Ln]=1.0
    snt[0]=sent_indices(t,list(e.offsets),msk[0])
    out=forward(p,Tensor(recompute_states(ids).astype(np.float32),dtype=dtypes.float),
                Tensor(msk,dtype=dtypes.float),Tensor(snt,dtype=dtypes.int))
    keys=("pres","ftype","op","islit","dig","args","res","query")+(("sel",) if "sel" in out else ())+(("dup",) if "dup" in out else ())
    o={k:out[k].realize().numpy() for k in keys}
    return decode({k:o[k][0] for k in o})[0]
def quorum(p,rows,seed):
    nq=0
    for j,r in enumerate(rows):
        vt=[r["text"]]+[permuted_view(r["text"],seed+40*j+k) for k in range(1,5)]
        ans=[solve2(one(p,t),r["q"],{"n_vars":24,"m":300}) for t in vt]
        nn=[a for a in ans if a is not None]
        c=Counter(nn).most_common(1); plur,cnt=c[0] if c else (None,0)
        nq+=(cnt>=3 and plur==r["gold"])
    return nq
v120=load(".cache/g24_v120.safetensors")
sel=None
for name,fmt,_ in CANDS:
    rows=mint(fmt,name)
    nq=quorum(v120,rows,95100)
    print(f"[scan:v120] {name} {nq}/8",flush=True)
    if sel is None and 2<=nq<=6: sel=(name,fmt,rows,nq)
if sel is None:
    print("NO CANDIDATE IN BAND — placement fails, banked as such"); sys.exit(0)
name,fmt,rows,base=sel
print(f"[SELECTED] {name} (v120 {base}/8) — reading arms",flush=True)
for arm,ck in (("ictl",".cache/g25v2_inv_ctl.safetensors"),("iinv",".cache/g25v2_inv_inv.safetensors")):
    p=load(ck); nq=quorum(p,rows,95100)
    print(f"[{arm}] {name} {nq}/8",flush=True)
