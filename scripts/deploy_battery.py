"""deploy_battery.py — THE DEPLOYMENT BATTERY (registered): the walled
rescue stack on the deployed gate, full bigtest. Straight path untouched;
refusals -> conductor -> wall. Bars: wrongs-added ~0; net gain reported."""
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
n=len(samples)
tok=Tokenizer.from_file(TOKENIZER_JSON)
p=build_params(0); sd=safe_load(os.environ["ALG_CKPT"])
for k in p: p[k].assign(sd[k].to(p[k].device).cast(p[k].dtype)).realize()
K=("pres","ftype","op","islit","dig","args","res","query")
def parse_states(sts,msk,snt,m):
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
def parse_texts(texts):
    m=len(texts)
    ids=np.zeros((m,T_ALG),np.int32); msk=np.zeros((m,T_ALG),np.float32); snt=np.zeros((m,T_ALG),np.int32)
    for li,t in enumerate(texts):
        e=tok.encode(t)
        if len(e.ids)>T_ALG: continue
        ids[li,:len(e.ids)]=e.ids; msk[li,:len(e.ids)]=1.0
        snt[li]=sent_indices(t,list(e.offsets),msk[li])
    return parse_states(recompute_states(ids),msk,snt,m)
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
    if diagnose(facs,q)=="unsat":
        idxs=[i for i,f in enumerate(facs) if f.get("ftype") in ("rel","given")][:20]
        for i in idxs:
            a=solve_forced([f for j,f in enumerate(facs) if j!=i],q,smp)
            if a is not None:
                try:                    # audit #7: the conflict store hears
                    from mycelium.campaign_db import record_core   # every organ
                    record_core(str(smp.get("text","")),os.environ.get("ALG_TEST_NAME","?"),
                                "unsat",facs[i],[],"2026-08-20")
                except Exception: pass
                return a
        return None
    parts=re.split(r"(?<=\.)\s+",text.strip())
    if len(parts)<=3: return None
    t=" ".join([parts[0]]+sorted(parts[1:-1],key=len)+[parts[-1]])
    f2,q2=parse_texts([t])[0]
    return solve_forced(f2,q2,smp)
print("[deploy] straight pass ...",flush=True)
straight=parse_states(states,tokmask,sent,n)
ga={i:samples[i]["solution"][samples[i]["query_var"]] for i in range(n)}
base_ans={}; refusals=[]
for i in range(n):
    facs,q=straight[i]
    a=solve_forced(facs,q,samples[i])
    base_ans[i]=a
    if a is None: refusals.append(i)
base_right=sum(1 for i in range(n) if base_ans[i]==ga[i])
base_wrong=sum(1 for i in range(n) if base_ans[i] is not None and base_ans[i]!=ga[i])
print(f"[deploy] baseline: right {base_right} wrong {base_wrong} refusals {len(refusals)}",flush=True)
add_right=0; add_wrong=0; banked=0
for ct,ri in enumerate(refusals):
    parts=re.split(r"(?<=\.)\s+",samples[ri]["text"].strip())
    vt=[samples[ri]["text"]]
    for k in range(1,5):
        if len(parts)<=3: vt.append(samples[ri]["text"]); continue
        mid=parts[1:-1]; random.Random(1000*k+ri).shuffle(mid)
        vt.append(" ".join([parts[0]]+mid+[parts[-1]]))
    parses=parse_texts(vt)
    ans=[rescue(f,q,samples[ri],vt[vi]) for vi,(f,q) in enumerate(parses)]
    forced=[a for a in ans if a is not None]
    if not forced: continue
    top=max(set(forced),key=forced.count)
    if forced.count(top)>=3:
        banked+=1
        if top==ga[ri]: add_right+=1
        else: add_wrong+=1
    if ct%50==0: print(f"[deploy] rescue {ct}/{len(refusals)} (banked {banked})",flush=True)
print(f"[deploy] RESCUE: banked {banked}  +right {add_right}  +WRONG {add_wrong}",flush=True)
print(f"[deploy] NET ANSWER: {base_right+add_right}/{n} (was {base_right})  wrongs total {base_wrong+add_wrong} (was {base_wrong})",flush=True)
print("== DEPLOY BATTERY COMPLETE ==",flush=True)
