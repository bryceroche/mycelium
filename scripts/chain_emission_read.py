"""chain_emission_read.py — door #56's primary: held-out cascade prose
(fresh seed), macro-slot factor-exact + ANSWER through expansion."""
import os, sys, json
for k,v in [("ALG2","1"),("ALG_FTYPES","9"),("ALG_HW","512"),("ALG_DUP","1"),("ALG_WIDE","1")]: os.environ.setdefault(k,v)
sys.path.insert(0,'.'); sys.path.insert(0,'scripts')
import numpy as np, re
from phase1_algebra_head import T_ALG, build_params, forward, decode, sent_indices, TOKENIZER_JSON
from beacon_closing_arm import recompute_states
from tta_alg2_dials import solve2
from tokenizers import Tokenizer
from tinygrad import Tensor, dtypes
from tinygrad.nn.state import safe_load
tok=Tokenizer.from_file(TOKENIZER_JSON)
p=build_params(0); sd=safe_load(os.environ.get("EM_CK",".cache/g56_chain.safetensors"))
for k in p: p[k].assign(sd[k].to(p[k].device).cast(p[k].dtype)).realize()
L="abcdefghijklmnopqrstuvwx"
FORMS=["The product of {xs} is {r}.","Multiplying {xs} together gives {r}.","{r} is the product of {xs}."]
GIV=["{v} is {n}.","It is known that {v} is {n}.","{v} has the value {n}."]
def xs_phrase(vs): return ", ".join(vs[:-1])+f" and {vs[-1]}" if len(vs)>2 else f"{vs[0]} and {vs[1]}"
rng=np.random.RandomState(77000)   # FRESH seed — held out
fe=0; ans=0; n=100; rowsb=[]; AUT=[0]
for _ in range(n):
    if os.environ.get("INBAND")=="1":
        k=int(rng.randint(3,5)); nd=0
    else:
        k=int(rng.randint(3,6)); nd=int(rng.randint(0,4))
    while True:
        vals=[int(rng.randint(2,5)) for _ in range(k)]
        prod=1
        for v in vals: prod*=v
        if prod<=300: break
    gv=[int(rng.randint(2,90)) for _ in range(nd)]
    nv=nd+k+1; xs=list(range(nd,nd+k)); res=nd+k
    sents=[GIV[rng.randint(3)].format(v=L[i],n=gv[i]) for i in range(nd)]
    sents+=[GIV[rng.randint(3)].format(v=L[nd+i],n=vals[i]) for i in range(k)]
    if os.environ.get("SEQ_READ")=="1":
        PAIR=["The product of {a} and {b} is {c}.","Multiplying {a} by {b} gives {c}.","{c} is {a} multiplied by {b}.","{a} times {b} makes {c}."]
        ts=[nv+t for t in range(k-2)]; nv=nv+k-2
        cs=[]; acc=xs[0]
        for t,v in enumerate(xs[1:]):
            tgt=res if t==k-2 else ts[t]
            cs.append(PAIR[rng.randint(4)].format(a=L[acc],b=L[v],c=L[tgt])); acc=tgt
        ms=" ".join(cs)
    else:
        ms=FORMS[rng.randint(3)].format(xs=xs_phrase([L[v] for v in xs]),r=L[res])
    pos=rng.randint(3); ins=0 if pos==0 else (len(sents)//2 if pos==1 else len(sents))
    body=sents[:ins]+[ms]+sents[ins:]
    text=f"Consider the numbers {', '.join(L[:nv])}. "+" ".join(body)+f" What is {L[res]}?"
    rowsb.append((text,xs,res,prod,nv))
for s0 in range(0,n,8):
    ch=rowsb[s0:s0+8]
    ids=np.zeros((8,T_ALG),np.int32); msk=np.zeros((8,T_ALG),np.float32); snt=np.zeros((8,T_ALG),np.int32)
    for i,(t,_,_,_,_) in enumerate(ch):
        e=tok.encode(t); Ln=min(len(e.ids),T_ALG)
        ids[i,:Ln]=e.ids[:Ln]; msk[i,:Ln]=1.0
        snt[i]=sent_indices(t,list(e.offsets),msk[i])
    o=forward(p,Tensor(recompute_states(ids).astype(np.float32),dtype=dtypes.float),
              Tensor(msk,dtype=dtypes.float),Tensor(snt,dtype=dtypes.int))
    keys=("pres","ftype","op","islit","dig","args","res","query")+(("sel",) if "sel" in o else ())+(("dup",) if "dup" in o else ())+(("sgn",) if "sgn" in o else ())+(("dig2",) if "dig2" in o else ())
    onp={k2:o[k2].realize().numpy() for k2 in keys}
    for i,(t,xs,res,prod,nv_) in enumerate(ch):
        facs,q=decode({k2:onp[k2][i] for k2 in onp})
        hit=any(f.get("name")=="CHAIN_MUL" and sorted(f.get("xs",[]))==xs and f.get("result")==res for f in facs)
        fe+=hit
        a=solve2(facs,q,{"n_vars":nv_,"m":300})
        ans+=(a==prod)
        if os.environ.get("AUTOPSY")=="1" and hit and a!=prod and AUT[0]<10:
            AUT[0]+=1
            print(f"--- fail #{AUT[0]}: gold prod={prod} answered={a} xs={xs} res={res}",flush=True)
            for f in facs: print(f"    {f}",flush=True)
print(f"[emission] held-out cascades: macro factor-exact {fe}/{n}  ANSWER {ans}/{n}",flush=True)
