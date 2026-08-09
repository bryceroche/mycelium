import os, sys, json
for k,v in [("ALG2","1"),("ALG_FTYPES","8"),("ALG_HW","512"),("ALG_DUP","1"),("ALG_WIDE","1")]: os.environ.setdefault(k,v)
sys.path.insert(0,'.'); sys.path.insert(0,'scripts')
import numpy as np
from phase1_algebra_head import T_ALG, build_params, forward, sent_indices, TOKENIZER_JSON
from beacon_closing_arm import recompute_states
from tokenizers import Tokenizer
from tinygrad import Tensor, dtypes
from tinygrad.nn.state import safe_load
L="abcdefghij"
tok=Tokenizer.from_file(TOKENIZER_JSON)
CK=os.environ["CK"]
p=build_params(0); sd=safe_load(CK)
for k in p: p[k].assign(sd[k].to(p[k].device).cast(p[k].dtype)).realize()
def fw(t):
    ids=np.zeros((8,T_ALG),np.int32); msk=np.zeros((8,T_ALG),np.float32); snt=np.zeros((8,T_ALG),np.int32)
    e=tok.encode(t); Ln=min(len(e.ids),T_ALG)
    ids[0,:Ln]=e.ids[:Ln]; msk[0,:Ln]=1.0
    snt[0]=sent_indices(t,list(e.offsets),msk[0])
    o=forward(p,Tensor(recompute_states(ids).astype(np.float32),dtype=dtypes.float),
              Tensor(msk,dtype=dtypes.float),Tensor(snt,dtype=dtypes.int))
    return {k:o[k].realize().numpy() for k in ("pres","ftype","op","dup","sgn","sel","args") if k in o}
def dup_rows(nd,n,seed):
    rng=np.random.RandomState(seed); out=[]
    while len(out)<n:
        op="add" if rng.rand()<0.5 else "mul"
        x=int(rng.randint(2,60)) if op=="add" else int(rng.randint(2,13))
        g=x+x if op=="add" else x*x
        if g>300: continue
        gv=[int(rng.randint(2,90)) for _ in range(nd)]
        dv=nd; res=nd+1
        w="{a} plus another {a} makes {c}." if op=="add" else "{a} lots of {a} make {c}."
        sents=[f"{L[i]} is {gv[i]}." for i in range(nd)]+[f"{L[dv]} is {x}.", w.format(a=L[dv],c=L[res])]
        out.append({"t":f"Consider the numbers {', '.join(L[:res+1])}. "+" ".join(sents)+f" What is {L[res]}?","dv":dv,"op":op})
    return out
def relslot(o):
    js=[j for j in range(24) if o["pres"][0,j]>0 and o["ftype"][0,j].argmax()==0]
    return (max(js,key=lambda q:o["dup"][0,q]) if js else None)
# member 1+2: dup misbind + bit fire + op acc on nd=1 novel; adjacent nd=6
for nd in (1,6):
    mis=0; fire=0; opok=0; n=0
    for r in dup_rows(nd,15,96000+nd):
        o=fw(r["t"]); j=relslot(o)
        if j is None: mis+=1; continue
        n+=1
        d=o["dup"][0,j]>0
        fire+=int(d)
        a0=int(np.argmax(o["args"][0,j]))
        ok=d and a0==r["dv"]
        if not ok: mis+=1
        opok+=int(o["op"][0,j].argmax()==(1 if r["op"]=="mul" else 0))
    print(f"[dup nd={nd}] misbind {mis}/15  bit-fire {fire}/{n}  op-acc {opok}/{n}",flush=True)
# member 3: ftype mod-vs-fdiv @ nd=2; adjacent nd=6
rng=np.random.RandomState(98300)
for ndx in (2,6):
    ok=0; n=0
    rng2=np.random.RandomState(98300+ndx)
    while n<20:
        K=int(rng2.choice([3,4,5,6,7])); A=int(rng2.randint(10,290))
        if A%K==0: continue
        gv=[int(rng2.randint(2,90)) for _ in range(ndx)]
        av=ndx; res=ndx+1
        sents=[f"{L[i]} is {gv[i]}." for i in range(ndx)]+[f"{L[av]} is {A}.", f"After dividing {L[av]} by {K}, {L[res]} is what remains."]
        t=f"Consider the numbers {', '.join(L[:res+1])}. "+" ".join(sents)+f" What is {L[res]}?"
        o=fw(t); n+=1
        js=[j for j in range(24) if o["pres"][0,j]>0 and o["ftype"][0,j].argmax() not in (0,1)]
        ok+=int(bool(js) and o["ftype"][0,js[-1]].argmax()==4)
    print(f"[mod nd={ndx}] ftype-mod acc {ok}/{n}",flush=True)
# sentinels
rng=np.random.RandomState(98500); s_ok=0
for i in range(20):
    x=int(rng.randint(2,200)); d=int(rng.randint(2,90))
    o=fw(f"Consider the numbers a, b. a is {d}. b is -{x}. What is b?")
    js=[j for j in range(24) if o["pres"][0,j]>0 and o["ftype"][0,j].argmax()==1]
    s_ok+=int(bool(js) and o["sgn"][0,js[-1]]>0)
print(f"[sentinel sgn-neg] {s_ok}/20",flush=True)
sel_ok=0
for i in range(20):
    a,b=int(rng.randint(2,90)),int(rng.randint(2,90))
    if a==b: continue
    o=fw(f"Consider the numbers a, b, c, d, e. d is 7. e is 9. a is {a}. b is {b}. c is the smaller of a and b. What is c?")
    js=[j for j in range(24) if o["pres"][0,j]>0 and o["ftype"][0,j].argmax()==3]
    sel_ok+=int(bool(js) and o["sel"][0,js[0]].argmax()==1)
print(f"[sentinel sel-smaller] {sel_ok}/20",flush=True)
