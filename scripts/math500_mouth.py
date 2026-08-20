"""math500_mouth.py — THE TRUE CHAIN (mouth first).
From: math500_run.py — THE MEASUREMENT (registered): the full v2 stack on MATH-500.
Original doc: conductor_v2.py — THE PROACTIVE CONDUCTOR (registered; bars: net>=1352,
wrongs<=35): the rough map routes BEFORE parsing — wildness = corr(position,
length) of middle sentences (canonical mints run short->long; pinned
threshold: wild if corr <= 0.0, conservative lane). Wild -> len_asc lane;
else straight. Refusals -> the deployed reactive rescue (diagnose -> route
-> five-view wall). Lane census reported."""
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

import json as _json
_rows=[_json.loads(l) for l in open('.cache/math500_test.jsonl')]
samples=[{"text":r["problem"],"n_vars":24,"m":300,"solution":None,"query_var":None,
          "answer":r.get("answer"),"level":r.get("level"),"subject":r.get("subject")} for r in _rows]
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
def wildness(text):
    parts=re.split(r"(?<=\.)\s+",text.strip())
    mid=parts[1:-1]
    if len(mid)<4: return 1.0
    L=np.array([len(x) for x in mid],float)
    pos=np.arange(len(mid),dtype=float)
    if L.std()<1e-6: return 1.0
    return float(np.corrcoef(pos,L)[0,1])
def lenasc(text):
    parts=re.split(r"(?<=\.)\s+",text.strip())
    if len(parts)<=3: return text
    return " ".join([parts[0]]+sorted(parts[1:-1],key=len)+[parts[-1]])
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
def rescue(facs,q,smp,lane_text):
    a=solve_forced(facs,q,smp)
    if a is not None: return a
    if diagnose(facs,q)=="unsat":
        for i in [i for i,f in enumerate(facs) if f.get("ftype") in ("rel","given")][:20]:
            a=solve_forced([f for j,f in enumerate(facs) if j!=i],q,smp)
            if a is not None: return a
        return None
    t=lenasc(lane_text)
    if t==lane_text: return None
    f2,q2=parse_texts([t])[0]
    return solve_forced(f2,q2,smp)
# THE MOUTH (the chain's first link): pooled trunk kNN vs the family bank,
# length-corrected (coef), refuse above thr — refused rows NEVER parse.
_mz=np.load('.cache/recognition_mouth_gen41.npz')
_bank=_mz["bank"].astype(np.float32); _coef=_mz["coef"].astype(np.float64); _mthr=float(_mz["thr_knn"])
def mouth_pass(texts):
    m=len(texts); out=np.zeros(m,bool)
    for c0 in range(0,m,64):
        ch=texts[c0:c0+64]
        ids=np.zeros((len(ch),T_ALG),np.int32); msk=np.zeros((len(ch),T_ALG),np.float32)
        for li,t in enumerate(ch):
            e=tok.encode(t)
            if len(e.ids)>T_ALG: continue
            ids[li,:len(e.ids)]=e.ids; msk[li,:len(e.ids)]=1.0
        sts=recompute_states(ids)
        pooled=(np.asarray(sts)*msk[:,:,None]).sum(1)/np.maximum(msk.sum(1)[:,None],1)
        pooled=pooled/np.maximum(np.linalg.norm(pooled,axis=1,keepdims=True),1e-9)
        L=np.maximum(msk.sum(1).astype(np.float64),1)
        d=np.sort(1.0-pooled@_bank.T,axis=1)[:,:8].mean(1)
        res=d-(_coef[0]+_coef[1]/L)
        out[c0:c0+len(ch)]=res<=_mthr
    return out
_passed=mouth_pass([samples[i]["text"] for i in range(n)])
PASS=[i for i in range(n) if _passed[i]]
print(f"[m500] THE MOUTH: passed {len(PASS)}/500 (refused {n-len(PASS)} as out-of-register)",flush=True)
W_THR=0.0
wild=[i for i in PASS if wildness(samples[i]["text"])<=W_THR]
print(f"[m500] lane census: wild {len(wild)}/{n} (thr {W_THR}) -> len_asc lane; rest straight",flush=True)
lane_text={i:(lenasc(samples[i]["text"]) if i in set(wild) else samples[i]["text"]) for i in range(n)}
print("[m500] lane parses ...",flush=True)
all_parses=parse_texts([lane_text[i] for i in PASS])
lane_parse={i:all_parses[pi] for pi,i in enumerate(PASS)}
import re as _re
def _ga(i):
    a=str(samples[i].get("answer","")).strip()
    return int(a) if _re.fullmatch(r"-?\d+",a) else None
ga={i:_ga(i) for i in range(n)}
base={i:None for i in range(n)}; refusals=[]
for i in PASS:
    facs,q=lane_parse[i]
    a=solve_forced(facs,q,samples[i])
    base[i]=a
    if a is None: refusals.append(i)
r0=sum(1 for i in range(n) if base[i] is not None and ga[i] is not None and base[i]==ga[i])
w0=sum(1 for i in range(n) if base[i] is not None and (ga[i] is None or base[i]!=ga[i]))
print(f"[m500] lane baseline: right {r0} wrong {w0} refusals {len(refusals)}",flush=True)
ar=0; aw=0; bk=0
for ct,ri in enumerate(refusals):
    src=lane_text[ri]
    parts=re.split(r"(?<=\.)\s+",src.strip())
    vt=[src]
    for k in range(1,5):
        if len(parts)<=3: vt.append(src); continue
        mid=parts[1:-1]; random.Random(1000*k+ri).shuffle(mid)
        vt.append(" ".join([parts[0]]+mid+[parts[-1]]))
    parses=parse_texts(vt)
    ans=[rescue(f,q,samples[ri],vt[vi]) for vi,(f,q) in enumerate(parses)]
    forced=[a for a in ans if a is not None]
    if not forced: continue
    top=max(set(forced),key=forced.count)
    if forced.count(top)>=3:
        bk+=1
        if ga[ri] is not None and top==ga[ri]: ar+=1
        else: aw+=1
    if ct%50==0: print(f"[m500] rescue {ct}/{len(refusals)} (banked {bk})",flush=True)
print(f"[m500] RESCUE: banked {bk} +right {ar} +WRONG {aw}",flush=True)
print(f"[m500] MATH-500: right {r0+ar}/500  wrong {w0+aw}  banked {sum(1 for i in range(n) if base[i] is not None)+bk}",flush=True)
from collections import Counter
lv=Counter()
for i in range(n):
    if base[i] is not None and ga[i] is not None and base[i]==ga[i]: lv[samples[i]["level"]]+=1
print(f"[m500] rights by level (straight only): {dict(lv)}",flush=True)
import json as _j2
_j2.dump({"base":{str(i):base[i] for i in range(n) if base[i] is not None}},open('.cache/math500_results.json','w'))
print("== MATH500 COMPLETE ==",flush=True)
