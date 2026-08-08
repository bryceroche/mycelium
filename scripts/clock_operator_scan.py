import os, sys, json, re
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
def mint(nd, n=15, seed_base=96000):
    rng=np.random.RandomState(seed_base+nd); rows=[]
    while len(rows)<n:
        op="add" if rng.rand()<0.5 else "mul"
        x=int(rng.randint(2,60)) if op=="add" else int(rng.randint(2,13))
        gold=x+x if op=="add" else x*x
        if gold>300: continue
        gv=[int(rng.randint(2,90)) for _ in range(nd)]
        dv=nd; res=nd+1
        w="{a} plus another {a} makes {c}." if op=="add" else "{a} lots of {a} make {c}."
        sents=[f"{L[i]} is {gv[i]}." for i in range(nd)]+[f"{L[dv]} is {x}.", w.format(a=L[dv],c=L[res])]
        rows.append({"text":f"Consider the numbers {', '.join(L[:res+1])}. "+" ".join(sents)+f" What is {L[res]}?","dv":dv,"op":op,"rel":w.format(a=L[dv],c=L[res])})
    return rows
p=build_params(0); sd=safe_load(".cache/g24_v120.safetensors")
for k in p: p[k].assign(sd[k].to(p[k].device).cast(p[k].dtype)).realize()
def fwd(ids,msk,snt):
    out=forward(p,Tensor(recompute_states(ids).astype(np.float32),dtype=dtypes.float),
                Tensor(msk,dtype=dtypes.float),Tensor(snt,dtype=dtypes.int))
    return {k:out[k].realize().numpy() for k in ("pres","ftype","args","dup") if k in out}
def analyze(r):
    t=r["text"]; e=tok.encode(t); n=min(len(e.ids),T_ALG); offs=list(e.offsets)
    rel_start=t.find(r["rel"]); rel_end=rel_start+len(r["rel"])
    ids=np.zeros((8,T_ALG),np.int32); msk=np.zeros((8,T_ALG),np.float32)
    ids[0,:n]=e.ids[:n]; msk[0,:n]=1.0
    snt=np.zeros((8,T_ALG),np.int32); snt[0]=sent_indices(t,offs,msk[0])
    full=fwd(ids,msk,snt)
    js=[j for j in range(24) if full["pres"][0,j]>0 and full["ftype"][0,j].argmax()==0]
    if not js: return None
    j=max(js,key=lambda q: full["dup"][0,q])
    final=tuple(np.argsort(-full["args"][0,j])[:2])
    correct = set(final)=={r["dv"]} or final[0]==r["dv"]==final[1] or (final[0]==r["dv"] and full["dup"][0,j]>0)
    rel_tok0=next(i for i,(cs,ce) in enumerate(offs[:n]) if ce>rel_start)
    rel_tokN=max(i for i,(cs,ce) in enumerate(offs[:n]) if ce>cs and cs<rel_end)
    traj=[]
    for cut in range(rel_tok0+1, n+1):
        ids2=np.zeros((8,T_ALG),np.int32); m2=np.zeros((8,T_ALG),np.float32)
        ids2[0,:cut]=e.ids[:cut]; m2[0,:cut]=1.0
        s2=np.zeros((8,T_ALG),np.int32); s2[0]=snt[0]
        o=fwd(ids2,m2,s2)
        traj.append(tuple(np.argsort(-o["args"][0,j])[:2]))
    settle=None
    for i in range(len(traj)):
        if all(tt==final for tt in traj[i:]): settle=rel_tok0+1+i; break
    if settle is None: return None
    # operators at settle (text/graph-geometric)
    dv_let=L[r["dv"]]
    char_at=offs[min(settle,n-1)][0]
    pre=t[:char_at]
    a_cnt=len(re.findall(r'\b'+dv_let+r'\b', pre))                      # (a) visible mentions of dup var
    b_deg=sum(1 for s_ in pre.split(". ") if re.search(r'\b'+dv_let+r'\b',s_))  # (b) sentences touching it
    given=f"{dv_let} is "
    c_bound=1.0 if given in pre and "." in pre[pre.find(given):] else 0.0 # (c) arg bound?
    d_comp=(settle-rel_tok0)/max(rel_tokN-rel_tok0,1)                    # (d) own-sentence completion
    return {"correct":bool(correct),"a":a_cnt,"b":b_deg,"c":c_bound,"d":min(d_comp,1.0)}
H=[]; P=[]
for r in mint(0,seed_base=96000): 
    x=analyze(r)
    if x and x["correct"]: H.append(x)
for r in mint(1,seed_base=96000):
    x=analyze(r)
    if x: P.append(x)
print(f"populations: healthy {len(H)}  premature {len(P)}")
from scipy.stats import mannwhitneyu
for opk in ("a","b","c","d"):
    hv=[x[opk] for x in H]; pv=[x[opk] for x in P]
    try:
        u,pp=mannwhitneyu(hv,pv,alternative="greater"); auc=u/(len(hv)*len(pv))
    except Exception: auc,pp=float("nan"),1
    print(f"[op {opk}] healthy mean {np.mean(hv):.3f}  premature mean {np.mean(pv):.3f}  AUC {auc:.3f} p={pp:.3g}")
