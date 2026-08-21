"""vote_wall.py — the wall on the rescue stack (registered): 5 independent
conductor runs per row (orig + 4 perms), >=3/5 banks, 5/5 certifies."""
import os, sys, json, re, random
sys.path.insert(0,'.'); sys.path.insert(0,'scripts')
import numpy as np
from phase1_algebra_head import build_params, forward, load_alg, decode, T_ALG, TOKENIZER_JSON, sent_indices
from repair_replace_swap import solve_forced
from beacon_closing_arm import recompute_states
from mycelium.csp_domains import problem_from_algebra2
from mycelium.csp_core import solve_symbolic
from tokenizers import Tokenizer
from tinygrad import Tensor, dtypes
from tinygrad.nn.state import safe_load

samples, states, tokmask, gold, sent = load_alg("test")
res=[int(r["idx"]) for r in json.load(open('.cache/residue_census.json'))["rows"]]
tok=Tokenizer.from_file(TOKENIZER_JSON)
p=build_params(0); sd=safe_load(os.environ["ALG_CKPT"])
for k in p: p[k].assign(sd[k].to(p[k].device).cast(p[k].dtype)).realize()
K=("pres","ftype","op","islit","dig","args","res","query")
def parse_texts(texts):
    m=len(texts)
    ids=np.zeros((m,T_ALG),np.int32); msk=np.zeros((m,T_ALG),np.float32); snt=np.zeros((m,T_ALG),np.int32)
    for li,t in enumerate(texts):
        e=tok.encode(t)
        if len(e.ids)>T_ALG: continue
        ids[li,:len(e.ids)]=e.ids; msk[li,:len(e.ids)]=1.0
        snt[li]=sent_indices(t,list(e.offsets),msk[li])
    sts=recompute_states(ids)
    out=[]
    for s0 in range(0,m,8):
        sl=list(range(s0,min(s0+8,m))); pad=8-len(sl); slp=sl+sl[:1]*pad
        o=forward(p,Tensor(sts[slp].astype(np.float32),dtype=dtypes.float),
                    Tensor(msk[slp].astype(np.float32),dtype=dtypes.float),
                    Tensor(snt[slp].astype(np.int32),dtype=dtypes.int))
        ex=tuple(k2 for k2 in ("sel","dup","sgn") if k2 in o)
        onp={k2:o[k2].realize().numpy() for k2 in K+ex}
        for bi in range(len(sl)): out.append(decode({k2:onp[k2][bi] for k2 in onp}))
    return out
def diagnose(facs,q):
    gv={f["var"]:f["value"] for f in facs if f.get("ftype")=="given"}
    def fv(f):
        if f.get("ftype") in ("rel","sel"): return list(f["args"])+[f["result"]]
        if f.get("ftype")=="mod": return [f["var"],f["result"]]
        return [f.get("var",0)]
    try:
        nv=max([24]+[v+1 for f in facs for v in fv(f)]+[q+1])
        return solve_symbolic(problem_from_algebra2(nv,facs,gv,300),budget=100_000,seed=0)["status"]
    except Exception: return "malformed"
def rescue(facs,q,smp,text):
    a=solve_forced(facs,q,smp)
    if a is not None: return a
    d=diagnose(facs,q)
    if d=="unsat":
        idxs=[i for i,f in enumerate(facs) if f.get("ftype") in ("rel","given")][:20]
        for i in idxs:
            f2=[f for j,f in enumerate(facs) if j!=i]
            a=solve_forced(f2,q,smp)
            if a is not None:
                try:
                    from mycelium.campaign_db import record_core
                    record_core(str(smp.get("text","")),os.environ.get("ALG_TEST_NAME","?"),"unsat",facs[i],[],"2026-08-20")
                except Exception: pass
                return a
        return None
    parts=re.split(r"(?<=\.)\s+",text.strip())
    if len(parts)<=3: return None
    t=" ".join([parts[0]]+sorted(parts[1:-1],key=len)+[parts[-1]])
    f2,q2=parse_texts([t])[0]
    return solve_forced(f2,q2,smp)
def views(text,ri):
    parts=re.split(r"(?<=\.)\s+",text.strip())
    out=[text]
    for k in range(1,5):
        if len(parts)<=3: out.append(text); continue
        mid=parts[1:-1]; random.Random(1000*k+ri).shuffle(mid)
        out.append(" ".join([parts[0]]+mid+[parts[-1]]))
    return out
ga={i:samples[i]["solution"][samples[i]["query_var"]] for i in res}
banked=0; certified=0; wrong_banked=0; abstain=0
for ri in res:
    vt=views(samples[ri]["text"],ri)
    parses=parse_texts(vt)
    ans=[]
    for vi,(facs,q) in enumerate(parses):
        ans.append(rescue(facs,q,samples[ri],vt[vi]))
    forced=[a for a in ans if a is not None]
    if not forced: abstain+=1; continue
    top=max(set(forced),key=forced.count)
    if forced.count(top)>=3:
        banked+=1
        if forced.count(top)==5: certified+=1
        if top!=ga[ri]: wrong_banked+=1
    else: abstain+=1
print(f"[wall] banked {banked}/74 (certified 5/5: {certified})  WRONG-BANKED {wrong_banked}  abstain {abstain}",flush=True)
print("== VOTE WALL COMPLETE ==",flush=True)
