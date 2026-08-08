import os, sys, json
for k,v in [("ALG2","1"),("ALG_FTYPES","8"),("ALG_HW","512"),("ALG_DUP","1"),("ALG_WIDE","1"),("ALG_DUPPTR","1"),("ALG_INV","1")]: os.environ.setdefault(k,v)
sys.path.insert(0,'.'); sys.path.insert(0,'scripts')
import numpy as np
from phase1_algebra_head import T_ALG, build_params, forward, sent_indices, TOKENIZER_JSON
from beacon_closing_arm import recompute_states
from tokenizers import Tokenizer
from tinygrad import Tensor, dtypes
from tinygrad.nn.state import safe_load
L="abcdefghij"
tok=Tokenizer.from_file(TOKENIZER_JSON)
def mint(dup, n, seed):
    rng=np.random.RandomState(seed); rows=[]
    while len(rows)<n:
        op="add" if rng.rand()<0.5 else "mul"
        x=int(rng.randint(2,60)) if op=="add" else int(rng.randint(2,13))
        d1=int(rng.randint(2,90))
        if dup:
            gold=x+x if op=="add" else x*x
            if gold>300: continue
            w="{a} plus another {a} makes {c}." if op=="add" else "{a} lots of {a} make {c}."
            body=w.format(a="b",c="c")
        else:
            y=int(rng.randint(2,60)) if op=="add" else int(rng.randint(2,13))
            gold=x+y if op=="add" else x*y
            if gold>300: continue
            body=(f"b plus a makes c." if op=="add" else "b times a makes c.") if False else (f"The sum of b and a is c." if op=="add" else f"The product of b and a is c.")
        rows.append(f"Consider the numbers a, b, c. a is {d1}. b is {x}. {body} What is c?")
    return rows
p=build_params(0); sd=safe_load(".cache/g29_dp_dupptr.safetensors")
for k in p: p[k].assign(sd[k].to(p[k].device).cast(p[k].dtype)).realize()
def states_and_logits(texts):
    X=[]; DL=[]
    for t in texts:
        ids=np.zeros((8,T_ALG),np.int32); msk=np.zeros((8,T_ALG),np.float32); snt=np.zeros((8,T_ALG),np.int32)
        e=tok.encode(t); Ln=min(len(e.ids),T_ALG)
        ids[0,:Ln]=e.ids[:Ln]; msk[0,:Ln]=1.0
        snt[0]=sent_indices(t,list(e.offsets),msk[0])
        o=forward(p,Tensor(recompute_states(ids).astype(np.float32),dtype=dtypes.float),
                  Tensor(msk,dtype=dtypes.float),Tensor(snt,dtype=dtypes.int))
        onp={k:o[k].realize().numpy() for k in ("pres","ftype","dup","fst_s")}
        js=[j for j in range(24) if onp["pres"][0,j]>0 and onp["ftype"][0,j].argmax()==0]
        if not js: continue
        j=max(js,key=lambda q: onp["dup"][0,q])
        X.append(onp["fst_s"][0,j]); DL.append(float(onp["dup"][0,j]))
    return np.array(X), np.array(DL)
Xd,Ld=states_and_logits(mint(True,100,97000))
Xn,Ln_=states_and_logits(mint(False,100,97500))
print(f"collected dup {len(Xd)} nondup {len(Xn)} | bit logit: dup-rows mean {Ld.mean():.2f} (fire {(Ld>0).mean():.2f})  nondup mean {Ln_.mean():.2f}")
n=min(len(Xd),len(Xn)); Xd,Xn=Xd[:n],Xn[:n]
h=n//2
Xtr=np.vstack([Xd[:h],Xn[:h]]); ytr=np.array([1]*h+[0]*h)
Xte=np.vstack([Xd[h:],Xn[h:]]); yte=np.array([1]*(n-h)+[0]*(n-h))
mu,sg=Xtr.mean(0),Xtr.std(0)+1e-6
Xtr=(Xtr-mu)/sg; Xte=(Xte-mu)/sg
w=np.linalg.lstsq(Xtr.T@Xtr+10.0*np.eye(Xtr.shape[1]), Xtr.T@(2*ytr-1), rcond=None)[0]
acc=((Xte@w>0).astype(int)==yte).mean()
print(f"[probe] held-out acc {acc:.3f} (n={len(yte)})")
json.dump({"acc":float(acc),"dup_fire":float((Ld>0).mean())},open('.cache/bit_detection.json','w'))
