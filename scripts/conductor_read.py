"""conductor_read.py — THE CONDUCTOR + THE REFUTATION TREE (registered):
solver diagnoses each refusal (unsat vs underdetermined), conductor routes
by symptom (contradiction -> deletion-core amputation; starvation ->
len_asc re-read), crossed dispatch = the placebo. Gate ckpt, residue-74."""
import os, sys, json, re
sys.path.insert(0,'.'); sys.path.insert(0,'scripts')
import numpy as np
from phase1_algebra_head import build_params, forward, load_alg, decode, T_ALG, TOKENIZER_JSON, sent_indices
from repair_replace_swap import solve_forced
from beacon_closing_arm import recompute_states
from mycelium.csp_domains import problem_from_algebra2
from mycelium.csp_core import solve_symbolic
from tokenizers import Tokenizer

samples, states, tokmask, gold, sent = load_alg("test")
rc=json.load(open('.cache/residue_census.json'))["rows"]
res=[int(r["idx"]) for r in rc]
tok=Tokenizer.from_file(TOKENIZER_JSON)
p=build_params(0)
from tinygrad import Tensor, dtypes
from tinygrad.nn.state import safe_load
sd=safe_load(os.environ["ALG_CKPT"])
for k in p: p[k].assign(sd[k].to(p[k].device).cast(p[k].dtype)).realize()
K=("pres","ftype","op","islit","dig","args","res","query")
def parse_rows(sts,msk,snt,rows):
    out={}
    for s0 in range(0,len(rows),8):
        sl=list(range(s0,min(s0+8,len(rows)))); pad=8-len(sl); slp=sl+sl[:1]*pad
        o=forward(p,Tensor(sts[slp].astype(np.float32),dtype=dtypes.float),
                    Tensor(msk[slp].astype(np.float32),dtype=dtypes.float),
                    Tensor(snt[slp].astype(np.int32),dtype=dtypes.int))
        ex=tuple(k2 for k2 in ("sel","dup","sgn") if k2 in o)
        onp={k2:o[k2].realize().numpy() for k2 in K+ex}
        for bi,li in enumerate(sl): out[rows[li]]=decode({k2:onp[k2][bi] for k2 in onp})
    return out
base=parse_rows(states[res],tokmask[res],sent[res],res)
def diagnose(facs,q):
    gv={f["var"]:f["value"] for f in facs if f.get("ftype")=="given"}
    def fv(f):
        if f.get("ftype") in ("rel","sel"): return list(f["args"])+[f["result"]]
        if f.get("ftype")=="mod": return [f["var"],f["result"]]
        return [f.get("var",0)]
    try:
        nv=max([24]+[v+1 for f in facs for v in fv(f)]+[q+1])
        r=solve_symbolic(problem_from_algebra2(nv,facs,gv,300),budget=100_000,seed=0)
        return r["status"], nv, gv
    except Exception: return "malformed", 24, gv
def amputate(facs,q,smp):
    idxs=[i for i,f in enumerate(facs) if f.get("ftype") in ("rel","given")][:20]
    for i in idxs:
        f2=[f for j,f in enumerate(facs) if j!=i]
        a=solve_forced(f2,q,smp)
        if a is not None: return a
    return None
def lenasc_reread(ri):
    parts=re.split(r"(?<=\.)\s+",samples[ri]["text"].strip())
    if len(parts)<=3: return None
    t=" ".join([parts[0]]+sorted(parts[1:-1],key=len)+[parts[-1]])
    e=tok.encode(t)
    if len(e.ids)>T_ALG: return None
    ids=np.zeros((1,T_ALG),np.int32); msk=np.zeros((1,T_ALG),np.float32); snt=np.zeros((1,T_ALG),np.int32)
    ids[0,:len(e.ids)]=e.ids; msk[0,:len(e.ids)]=1.0
    snt[0]=sent_indices(t,list(e.offsets),msk[0])
    sts=recompute_states(ids)
    o=parse_rows(sts,msk,snt,[ri])
    facs,q=o[ri]
    return solve_forced(facs,q,samples[ri])
ga={i:samples[i]["solution"][samples[i]["query_var"]] for i in res}
diag={}
for ri in res:
    facs,q=base[ri]
    diag[ri]=diagnose(facs,q)[0]
from collections import Counter
print(f"[conductor] refusal taxonomy: {Counter(diag.values())}",flush=True)
for arm,route in (("ROUTED",{"unsat":"amp","solved":"len","malformed":"len"}),
                  ("CROSSED",{"unsat":"len","solved":"amp","malformed":"amp"})):
    conv=0; wrong=0
    for ri in res:
        facs,q=base[ri]
        r=route.get(diag[ri],"len")
        a=amputate(facs,q,samples[ri]) if r=="amp" else lenasc_reread(ri)
        if a is not None:
            if a==ga[ri]: conv+=1
            else: wrong+=1
    print(f"[conductor {arm}] converts {conv}/74  wrongs {wrong}",flush=True)
print("== CONDUCTOR COMPLETE ==",flush=True)
