import os, sys, json
for k,v in [("ALG2","1"),("ALG_FTYPES","8"),("ALG_HW","512"),("ALG_DUP","1"),("ALG_WIDE","1"),("ALG_DUPPTR","1"),("ALG_INV","1")]: os.environ.setdefault(k,v)
os.environ.setdefault("ALG_TEST",".cache/algebra_nl_bigtest.jsonl"); os.environ.setdefault("ALG_TEST_NAME","bigtest")
sys.path.insert(0,'.'); sys.path.insert(0,'scripts')
import numpy as np, json as _j, re
from phase1_algebra_head import T_ALG, build_params, forward, sent_indices, TOKENIZER_JSON, load_alg, L_FAC
from beacon_closing_arm import recompute_states
from tokenizers import Tokenizer
from tinygrad import Tensor, dtypes
from tinygrad.nn.state import safe_load
L="abcdefghij"
tok=Tokenizer.from_file(TOKENIZER_JSON)
T=_j.load(open('.cache/aug_table_v3.json'))["licensed"]
dupfmt=[e["fmt"] for e in T if e["construction"]=="dup"]
NOVEL={"add":"{a} plus another {a} makes {c}.","mul":"{a} lots of {a} make {c}."}
p=build_params(0); sd=safe_load(".cache/g30_inst_install.safetensors")
for k in p: p[k].assign(sd[k].to(p[k].device).cast(p[k].dtype)).realize()
def slotstate(t, want_dup):
    ids=np.zeros((8,T_ALG),np.int32); msk=np.zeros((8,T_ALG),np.float32); snt=np.zeros((8,T_ALG),np.int32)
    e=tok.encode(t); Ln=min(len(e.ids),T_ALG)
    ids[0,:Ln]=e.ids[:Ln]; msk[0,:Ln]=1.0
    snt[0]=sent_indices(t,list(e.offsets),msk[0])
    o=forward(p,Tensor(recompute_states(ids).astype(np.float32),dtype=dtypes.float),
              Tensor(msk,dtype=dtypes.float),Tensor(snt,dtype=dtypes.int))
    onp={k:o[k].realize().numpy() for k in ("pres","ftype","dup","fst_s")}
    js=[j for j in range(24) if onp["pres"][0,j]>0 and onp["ftype"][0,j].argmax()==0]
    if not js: return None
    j=max(js,key=lambda q: onp["dup"][0,q])
    return onp["fst_s"][0,j]
def mint_cell(nd, novel, n, seed):
    rng=np.random.RandomState(seed); X=[]
    while len(X)<n:
        op="add" if rng.rand()<0.5 else "mul"
        x=int(rng.randint(2,60)) if op=="add" else int(rng.randint(2,13))
        g=x+x if op=="add" else x*x
        if g>300: continue
        gv=[int(rng.randint(2,90)) for _ in range(nd)]
        dv=nd; res=nd+1
        if novel: fmt=NOVEL[op]
        else:
            cands=[m for m in dupfmt if any(w in m for w in ("times","*","square","multiplied","x ","roduct"))==(op=="mul")]
            fmt=cands[rng.randint(len(cands))]
        sents=[f"{L[i]} is {gv[i]}." for i in range(nd)]+[f"{L[dv]} is {x}.", fmt.format(a=L[dv],c=L[res])]
        t=f"Consider the numbers {', '.join(L[:res+1])}. "+" ".join(sents)+f" What is {L[res]}?"
        s=slotstate(t,True)
        if s is not None: X.append(s)
    return np.array(X)
CELLS={}
sd_=98000
for nd in (0,1,2,4):
    for novel in (False,True):
        CELLS[(nd,novel)]=mint_cell(nd,novel,40,sd_); sd_+=13
        print(f"[cell] nd={nd} novel={novel}: {len(CELLS[(nd,novel)])}",flush=True)
# natural bigtest rel slots (positives arg_dup=1, negatives=0)
samples, states, tokmask, gold, sent = load_alg("test")
POS=[]; NEG=[]
rng=np.random.RandomState(99000)
order=rng.permutation(len(samples))
for ri in order:
    if len(NEG)>=400 and len(POS)>=60: break
    ids=np.zeros((8,T_ALG),np.int32)
    tr=Tensor(states[ri:ri+1].repeat(8,0).astype(np.float32) if False else np.repeat(states[ri:ri+1],8,axis=0).astype(np.float32),dtype=dtypes.float)
    tk=Tensor(np.repeat(tokmask[ri:ri+1],8,axis=0).astype(np.float32),dtype=dtypes.float)
    se=Tensor(np.repeat(sent[ri:ri+1],8,axis=0).astype(np.int32),dtype=dtypes.int)
    o=forward(p,tr,tk,se)
    onp={k:o[k].realize().numpy() for k in ("pres","ftype","fst_s")}
    for j in range(L_FAC):
        if gold["presence"][ri,j]<=0: continue
        if gold["ftype"][ri,j]!=0: continue
        _hot=(gold["args"][ri,j]>0.5).sum()
        if _hot==1:
            if len(POS)<60: POS.append(onp["fst_s"][0,j])
        elif len(NEG)<400: NEG.append(onp["fst_s"][0,j])
POS=np.array(POS); NEG=np.array(NEG)
print(f"[natural] bigtest dup slots {len(POS)}  non-dup rel slots {len(NEG)}",flush=True)
# assemble train/test: half of each cell + half of naturals
Xtr=[];ytr=[];test_sets={}
for k2,X in CELLS.items():
    h=len(X)//2
    Xtr.append(X[:h]); ytr += [1]*h
    test_sets[f"dup nd={k2[0]} novel={k2[1]}"]=(X[h:],1)
hp=len(POS)//2; hn=len(NEG)//2
if len(POS): Xtr.append(POS[:hp]); ytr+=[1]*hp
Xtr.append(NEG[:hn]); ytr+=[0]*hn
if len(POS): test_sets["bigtest dup (natural)"]=(POS[hp:],1)
test_sets["bigtest rel (false-fire)"]=(NEG[hn:],0)
Xtr=np.vstack(Xtr); ytr=np.array(ytr)
mu,sg=Xtr.mean(0),Xtr.std(0)+1e-6
Xs=(Xtr-mu)/sg
w=np.linalg.lstsq(Xs.T@Xs+10.0*np.eye(Xs.shape[1]), Xs.T@(2*ytr-1), rcond=None)[0]
W=(w/sg).astype(np.float32); b=np.float32(-(w*mu/sg).sum())
sd2=dict(sd)
from tinygrad import Tensor as _T
sd2["h_dup"]=_T(W.reshape(-1,1)); sd2["h_dup_b"]=_T(np.array([b],np.float32))
from tinygrad.nn.state import safe_save as _ss
_ss(sd2, ".cache/g30_transplant2.safetensors")
print("[fold] broad map -> g30_transplant2.safetensors",flush=True)
res={}
for name,(X,lab) in test_sets.items():
    if len(X)==0: continue
    pr=((X-mu)/sg)@w>0
    acc=(pr==bool(lab)).mean() if lab else (pr==False).mean()
    res[name]=float(acc)
    print(f"[test] {name}: {'acc' if lab else 'reject'} {acc:.3f} (n={len(X)})",flush=True)
json.dump(res,open('.cache/detector_coverage.json','w'))
