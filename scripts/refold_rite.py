"""refold_rite.py — THE REFOLD RITE (2026-08-10; the law's final form:
REFOLD AFTER ANY TRAINING, full stop). Re-derives the broad-gold dup
detector on the CANDIDATE's own waist and folds it into h_dup at the
gap-midpoint theta — headroom-first (distributions must not touch or
the rite ABORTS; the fold is never forced). Faithful to the v5 form
(manifest mechanism line). CK in -> CK_OUT saved only on headroom."""
import os, sys, json
for k,v in [("ALG2","1"),("ALG_FTYPES","8"),("ALG_HW","512"),("ALG_DUP","1"),("ALG_WIDE","1"),("ALG_INV","1")]: os.environ.setdefault(k,v)
os.environ.setdefault("ALG_TEST",".cache/algebra_nl_bigtest.jsonl"); os.environ.setdefault("ALG_TEST_NAME","bigtest")
sys.path.insert(0,'.'); sys.path.insert(0,'scripts')
import numpy as np, json as _j, re
from phase1_algebra_head import T_ALG, build_params, forward, sent_indices, TOKENIZER_JSON, load_alg, L_FAC
from beacon_closing_arm import recompute_states
from tokenizers import Tokenizer
from tinygrad import Tensor, dtypes
from tinygrad.nn.state import safe_load, safe_save
CK=os.environ["CK"]; CKO=os.environ["CK_OUT"]
L="abcdefghij"
tok=Tokenizer.from_file(TOKENIZER_JSON)
T=_j.load(open('.cache/aug_table_v3.json'))["licensed"]
dupfmt=[e["fmt"] for e in T if e["construction"]=="dup"]
NOVEL={"add":"{a} plus another {a} makes {c}.","mul":"{a} lots of {a} make {c}."}
p=build_params(0); sd=safe_load(CK)
for k in p: p[k].assign(sd[k].to(p[k].device).cast(p[k].dtype)).realize()
def rel_states(t):
    ids=np.zeros((8,T_ALG),np.int32); msk=np.zeros((8,T_ALG),np.float32); snt=np.zeros((8,T_ALG),np.int32)
    e=tok.encode(t); Ln=min(len(e.ids),T_ALG)
    ids[0,:Ln]=e.ids[:Ln]; msk[0,:Ln]=1.0
    snt[0]=sent_indices(t,list(e.offsets),msk[0])
    o=forward(p,Tensor(recompute_states(ids).astype(np.float32),dtype=dtypes.float),
              Tensor(msk,dtype=dtypes.float),Tensor(snt,dtype=dtypes.int))
    onp={k:o[k].realize().numpy() for k in ("pres","ftype","dup","fst_s")}
    js=[j for j in range(24) if onp["pres"][0,j]>0 and onp["ftype"][0,j].argmax()==0]
    return onp, js
def slotstate(t):
    onp,js=rel_states(t)
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
        s=slotstate(t)
        if s is not None: X.append(s)
    return np.array(X)
CELLS={}; sd_=98000
for nd in (0,1,2,4):
    for novel in (False,True):
        CELLS[(nd,novel)]=mint_cell(nd,novel,40,sd_); sd_+=13
        print(f"[cell] nd={nd} novel={novel}: {len(CELLS[(nd,novel)])}",flush=True)
samples, states, tokmask, gold, sent = load_alg("test")
POS=[]; NEG=[]
rng=np.random.RandomState(99000)
for ri in rng.permutation(len(samples)):
    if len(NEG)>=400 and len(POS)>=60: break
    tr=Tensor(states[ri:ri+1].repeat(8,axis=0).astype(np.float32),dtype=dtypes.float)
    tk=Tensor(tokmask[ri:ri+1].repeat(8,axis=0).astype(np.float32),dtype=dtypes.float)
    se=Tensor(sent[ri:ri+1].repeat(8,axis=0).astype(np.int32),dtype=dtypes.int)
    o=forward(p,tr,tk,se)
    onp={k:o[k].realize().numpy() for k in ("pres","ftype","fst_s")}
    for j in range(L_FAC):
        if gold["presence"][ri,j]<=0 or gold["ftype"][ri,j]!=0: continue
        _hot=(gold["args"][ri,j]>0.5).sum()
        if _hot==1:
            if len(POS)<60: POS.append(onp["fst_s"][0,j])
        elif len(NEG)<400: NEG.append(onp["fst_s"][0,j])
POS=np.array(POS); NEG=np.array(NEG)
print(f"[natural] dup slots {len(POS)}  non-dup rel slots {len(NEG)}",flush=True)
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
score=lambda X: ((X-mu)/sg)@w
# HEADROOM-FIRST: wild forty rel slots vs cell holdout positives
def int_answer(a):
    s=str(a).strip().replace("$","").replace(",","")
    return int(s) if re.fullmatch(r"-?\d+",s) else None
h=[_j.loads(l) for l in open(".cache/math_harvest_v0.jsonl")]
wild=[x["problem"] for x in h if x["level"]=="Level 4" and len(x["problem"])<300
      and "asy]" not in x["problem"]
      and all(int(n)<=300 for n in re.findall(r"\d+",x["problem"]))
      and int_answer(x["answer"]) is not None and 0<=int_answer(x["answer"])<=300][:40]
WS=[]; WIDX=[]
for wi,t in enumerate(wild):
    onp,js=rel_states(t)
    for j in js: WS.append(onp["fst_s"][0,j]); WIDX.append(wi)
WS=np.array(WS) if WS else np.zeros((0,Xtr.shape[1]))
WIDX=np.array(WIDX,np.int32)
cell_hold=np.vstack([X[len(X)//2:] for X in CELLS.values()])
sw=score(WS) if len(WS) else np.zeros(0)
for r in np.argsort(-sw)[:5]:
    print(f"[wild-rank] score {sw[r]:+.3f}  prob#{WIDX[r]}  {wild[WIDX[r]][:90]}",flush=True)
excl=set(int(x) for x in os.environ.get("WILD_EXCLUDE","").split(",") if x.strip())
if excl: print(f"[purity] excluding wild problems {sorted(excl)} (eye-audited GENUINE dup constructions — true fires, not false; the negative pool measures false-fire only)",flush=True)
keep=np.array([i for i in range(len(WS)) if int(WIDX[i]) not in excl])
cmin=float(score(cell_hold).min()); wmax=float(sw[keep].max()) if len(keep) else float("-inf")
print(f"[headroom] cell-min {cmin:.3f}  wild-max {wmax:.3f}  (wild rel slots n={len(keep)}/{len(WS)})",flush=True)
sc_cells={f"nd={k2[0]} novel={k2[1]}": score(X[len(X)//2:]) for k2,X in CELLS.items()}
for nm,scs in sc_cells.items():
    q=np.percentile(scs,[0,10,50,90])
    print(f"[dist] {nm}: min {q[0]:+.3f}  p10 {q[1]:+.3f}  med {q[2]:+.3f}  p90 {q[3]:+.3f}",flush=True)
swk=sw[keep] if len(keep) else np.zeros(0)
print(f"[dist] wild(kept): p90 {np.percentile(swk,90):+.3f}  p99 {np.percentile(swk,99):+.3f}  max {swk.max():+.3f}",flush=True)
if not (cmin > wmax):
    print("[ABORT] distributions touch — the fold is refused (headroom law)",flush=True); sys.exit(3)
theta=(cmin+wmax)/2.0
W=(w/sg).astype(np.float32); b=np.float32(-(w*mu/sg).sum()-theta)
sd2=dict(sd)
sd2["h_dup"]=Tensor(W.reshape(-1,1)); sd2["h_dup_b"]=Tensor(np.array([b],np.float32))
safe_save(sd2, CKO)
print(f"[fold] gap-midpoint theta {theta:.4f} -> {CKO}",flush=True)
res={"theta":theta,"cell_min":cmin,"wild_max":wmax}
for name,(X,lab) in test_sets.items():
    if len(X)==0: continue
    pr=score(X)>theta
    acc=(pr==bool(lab)).mean() if lab else (pr==False).mean()
    res[name]=float(acc)
    print(f"[test@theta] {name}: {'acc' if lab else 'reject'} {acc:.3f} (n={len(X)})",flush=True)
json.dump(res,open('.cache/refold_g35.json','w'))
