"""calibrated_rite.py — THE CALIBRATED REFOLD RITE (2026-08-10).
The instrument law: an instrument that fails its known-good control
produces no verdicts, only readings. Pools per the adoption
instrument (door #26): EVAL CELLS = the promotion fixture's own mint
(dup_axis_scan2 novel formats, seeds 96000+nd); NEGATIVES =
in-register bigtest rel slots at scale (the deployed chain's
admitted material — the mouth guards the wild; the fold's false-fire
duty is over ADMITTED slots only); TRAIN = broad-gold aug-table
surfaces + natural negatives (cure-class final form). Headroom-first;
gap-midpoint theta; fold only on CK_OUT."""
import os, sys, json
for k,v in [("ALG2","1"),("ALG_FTYPES","8"),("ALG_HW","512"),("ALG_DUP","1"),("ALG_WIDE","1"),("ALG_INV","1")]: os.environ.setdefault(k,v)
os.environ.setdefault("ALG_TEST",".cache/algebra_nl_bigtest.jsonl"); os.environ.setdefault("ALG_TEST_NAME","bigtest")
sys.path.insert(0,'.'); sys.path.insert(0,'scripts')
import numpy as np, json as _j
from phase1_algebra_head import T_ALG, build_params, forward, sent_indices, TOKENIZER_JSON, load_alg, L_FAC
from beacon_closing_arm import recompute_states
from tokenizers import Tokenizer
from tinygrad import Tensor, dtypes
from tinygrad.nn.state import safe_load, safe_save
CK=os.environ["CK"]; CKO=os.environ.get("CK_OUT")
L="abcdefghij"
tok=Tokenizer.from_file(TOKENIZER_JSON)
T=_j.load(open('.cache/aug_table_v3.json'))["licensed"]
dupfmt=[e["fmt"] for e in T if e["construction"]=="dup"]
NOVEL={"add":"{a} plus another {a} makes {c}.","mul":"{a} lots of {a} make {c}."}
p=build_params(0); sd=safe_load(CK)
for k in p: p[k].assign(sd[k].to(p[k].device).cast(p[k].dtype)).realize()
def dup_slotstate(t):
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
# --- EVAL CELLS: the promotion fixture's exact mint (dup_axis_scan2) ---
def fixture_mint(nd, n=15, seed=96000):
    rng=np.random.RandomState(seed+nd); rows=[]
    while len(rows)<n:
        op="add" if rng.rand()<0.5 else "mul"
        x=int(rng.randint(2,60)) if op=="add" else int(rng.randint(2,13))
        gold=x+x if op=="add" else x*x
        if gold>300: continue
        gv=[int(rng.randint(2,90)) for _ in range(nd)]
        dv=nd; res=nd+1
        w=NOVEL[op]
        sents=[f"{L[i]} is {gv[i]}." for i in range(nd)]+[f"{L[dv]} is {x}.", w.format(a=L[dv],c=L[res])]
        rows.append(f"Consider the numbers {', '.join(L[:res+1])}. "+" ".join(sents)+f" What is {L[res]}?")
    return rows
FIX={}
for nd in (0,1,2,4):
    X=[dup_slotstate(t) for t in fixture_mint(nd)]
    FIX[nd]=np.array([x for x in X if x is not None])
    print(f"[fixture] nd={nd}: {len(FIX[nd])}/15 slot states",flush=True)
# --- TRAIN CELLS: broad-gold aug-table surfaces (fresh seeds) ---
def train_mint(nd, n, seed):
    rng=np.random.RandomState(seed); X=[]
    while len(X)<n:
        op="add" if rng.rand()<0.5 else "mul"
        x=int(rng.randint(2,60)) if op=="add" else int(rng.randint(2,13))
        g=x+x if op=="add" else x*x
        if g>300: continue
        gv=[int(rng.randint(2,90)) for _ in range(nd)]
        dv=nd; res=nd+1
        if rng.rand()<0.3: fmt=NOVEL[op]
        else:
            cands=[m for m in dupfmt if any(w in m for w in ("times","*","square","multiplied","x ","roduct"))==(op=="mul")]
            fmt=cands[rng.randint(len(cands))]
        sents=[f"{L[i]} is {gv[i]}." for i in range(nd)]+[f"{L[dv]} is {x}.", fmt.format(a=L[dv],c=L[res])]
        s=dup_slotstate(f"Consider the numbers {', '.join(L[:res+1])}. "+" ".join(sents)+f" What is {L[res]}?")
        if s is not None: X.append(s)
    return np.array(X)
TR=[train_mint(nd,40,77000+nd) for nd in (0,1,2,4)]
print(f"[train-cells] {sum(len(x) for x in TR)}",flush=True)
# --- NEGATIVES: in-register bigtest rel slots at scale ---
samples, states, tokmask, gold, sent = load_alg("test")
NEG=[]
rng=np.random.RandomState(99000)
order=rng.permutation(len(samples))
for s0 in range(0,len(order),8):
    if len(NEG)>=4000: break
    sl=order[s0:s0+8]
    pad=8-len(sl); slp=np.concatenate([sl,sl[:1].repeat(pad)]) if pad else sl
    o=forward(p,Tensor(states[slp].astype(np.float32),dtype=dtypes.float),
              Tensor(tokmask[slp].astype(np.float32),dtype=dtypes.float),
              Tensor(sent[slp].astype(np.int32),dtype=dtypes.int))
    onp={k:o[k].realize().numpy() for k in ("fst_s",)}
    for bi,ri in enumerate(sl):
        for j in range(L_FAC):
            if gold["presence"][ri,j]<=0 or gold["ftype"][ri,j]!=0: continue
            if (gold["args"][ri,j]>0.5).sum()!=2: continue
            NEG.append(onp["fst_s"][bi,j])
NEG=np.array(NEG)
print(f"[neg] in-register rel slots {len(NEG)}",flush=True)
hn=len(NEG)//2
Xtr=np.vstack(TR+[NEG[:hn]]); ytr=np.array([1]*sum(len(x) for x in TR)+[0]*hn)
mu,sg=Xtr.mean(0),Xtr.std(0)+1e-6
Xs=(Xtr-mu)/sg
w=np.linalg.lstsq(Xs.T@Xs+10.0*np.eye(Xs.shape[1]), Xs.T@(2*ytr-1), rcond=None)[0]
score=lambda X: ((X-mu)/sg)@w
cmin=min(float(score(X).min()) for X in FIX.values())
negev=score(NEG[hn:]); nmax=float(negev.max())
for nd,X in FIX.items():
    s_=score(X); print(f"[fix-dist] nd={nd}: min {s_.min():+.3f}  med {np.median(s_):+.3f}",flush=True)
print(f"[neg-dist] p99 {np.percentile(negev,99):+.3f}  max {nmax:+.3f}  (n={len(negev)})",flush=True)
print(f"[headroom] fixture-min {cmin:.3f}  neg-max {nmax:.3f}",flush=True)
res={"ck":CK,"cell_min":cmin,"neg_max":nmax,"gap":cmin-nmax}
json.dump(res,open(os.environ.get("OUT_JSON",".cache/calibrated_rite_last.json"),'w'))
if not (cmin > nmax):
    print("[NO-HEADROOM] distributions touch on this waist (calibrated pools)",flush=True); sys.exit(3)
theta=(cmin+nmax)/2.0
fpr=float((negev>theta).mean())
print(f"[fold-ready] theta {theta:.4f}  holdout-FPR {fpr:.5f}",flush=True)
if CKO:
    W=(w/sg).astype(np.float32); b=np.float32(-(w*mu/sg).sum()-theta)
    sd2=dict(sd); sd2["h_dup"]=Tensor(W.reshape(-1,1)); sd2["h_dup_b"]=Tensor(np.array([b],np.float32))
    safe_save(sd2, CKO)
    print(f"[fold] -> {CKO}",flush=True)
