"""alternation_read.py — the two jaws as a loop (registered): refusal ->
gac-propagate -> derived knowns as canonical sentences -> re-parse ->
re-solve. Placebo: same vars, scrambled values. Gate ckpt, residue-74."""
import os, sys, json, re
sys.path.insert(0,'.'); sys.path.insert(0,'scripts')
import numpy as np
from phase1_algebra_head import build_params, forward, load_alg, decode, T_ALG, L_FAC
from repair_replace_swap import solve_forced
from beacon_closing_arm import recompute_states
from mycelium.csp_domains import problem_from_algebra2
from mycelium.csp_core import make_initial_state, gac_propagate, UNASSIGNED
from tokenizers import Tokenizer
from phase1_algebra_head import TOKENIZER_JSON, sent_indices
from tinygrad import Tensor, dtypes
from tinygrad.nn.state import safe_load

samples, states, tokmask, gold, sent = load_alg("test")
res=[r["idx"] for r in json.load(open('.cache/residue_census.json'))["rows"]]
tok=Tokenizer.from_file(TOKENIZER_JSON)
p=build_params(0); sd=safe_load(os.environ["ALG_CKPT"])
for k in p: p[k].assign(sd[k].to(p[k].device).cast(p[k].dtype)).realize()
K=("pres","ftype","op","islit","dig","args","res","query")
def fwd_rows(sts,msk,snt,rows):
    out={}
    for s0 in range(0,len(rows),8):
        sl=list(range(s0,min(s0+8,len(rows)))); pad=8-len(sl); slp=sl+sl[:1]*pad
        o=forward(p,Tensor(sts[slp].astype(np.float32),dtype=dtypes.float),
                    Tensor(msk[slp].astype(np.float32),dtype=dtypes.float),
                    Tensor(snt[slp].astype(np.int32),dtype=dtypes.int))
        ex=tuple(k for k in ("sel","dup","sgn") if k in o)
        onp={k:o[k].realize().numpy() for k in K+ex}
        for bi,li in enumerate(sl):
            out[rows[li]]=decode({k:onp[k][bi] for k in onp})
    return out
base=fwd_rows(states[res],tokmask[res],sent[res],res)
def derive(facs,q,smp):
    gv={f["var"]:f["value"] for f in facs if f.get("ftype")=="given"}
    def fv(f):
        if f.get("ftype") in ("rel","sel"): return list(f["args"])+[f["result"]]
        if f.get("ftype")=="mod": return [f["var"],f["result"]]
        return [f.get("var",0)]
    try:
        nv=max([smp["n_vars"]]+[v+1 for f in facs for v in fv(f)]+[q+1])
        pr=problem_from_algebra2(nv,facs,gv,smp["m"])
        st=gac_propagate(make_initial_state(pr))
        out={}
        for v in range(nv):
            if v in gv: continue
            if st.values[v]!=UNASSIGNED: out[v]=int(st.values[v])
            elif len(st.domains[v])==1: out[v]=int(tuple(st.domains[v])[0])
        return {v:x for v,x in out.items() if 0<=x<=300}
    except Exception: return {}
def rerun(texts,rows):
    m=len(rows)
    ids=np.zeros((m,T_ALG),np.int32); msk=np.zeros((m,T_ALG),np.float32); snt=np.zeros((m,T_ALG),np.int32)
    for li,t in enumerate(texts):
        e=tok.encode(t)
        if len(e.ids)>T_ALG: continue
        ids[li,:len(e.ids)]=e.ids; msk[li,:len(e.ids)]=1.0
        snt[li]=sent_indices(t,list(e.offsets),msk[li])
    sts=recompute_states(ids)
    return fwd_rows(sts,msk,snt,rows)
for arm in ("REAL","PLACEBO"):
    inj_rows=[]; inj_texts=[]
    for ri in res:
        facs,q=base[ri]; smp=samples[ri]
        if solve_forced(facs,q,smp) is not None: continue
        der=derive(facs,q,{"n_vars":24,"m":300})
        if not der: continue
        rng=np.random.default_rng(ri)
        parts=re.split(r"(?<=\.)\s+",smp["text"].strip())
        adds=[]
        for v,x in sorted(der.items())[:6]:
            val=x if arm=="REAL" else int(rng.integers(1,300))
            adds.append(f"{chr(ord('a')+v)} is {val}.")
        txt=" ".join(parts[:-1]+adds+[parts[-1]])
        inj_rows.append(ri); inj_texts.append(txt)
    if not inj_rows:
        print(f"[alt {arm}] no rows with derivations"); continue
    out2=rerun(inj_texts,inj_rows)
    conv=[ri for ri in inj_rows if solve_forced(*out2[ri],samples[ri])==samples[ri]["solution"][samples[ri]["query_var"]]]
    print(f"[alt {arm}] injectable {len(inj_rows)}/74  CONVERTS {len(conv)}  {sorted(conv)[:12]}")
print("== ALTERNATION READ COMPLETE ==")
