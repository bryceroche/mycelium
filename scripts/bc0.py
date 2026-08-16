"""bc0.py — the Grand Vision rung BC0 (registered): iterated text-keyed
re-matching across settle rounds; arms TDM/FDM/BOTH/PLACEBO via ARM env.
Bars: hold>=0.98; arm-minus-placebo differential >=4 on residue-74."""
import os, sys, json, re
sys.path.insert(0,'.'); sys.path.insert(0,'scripts')
import numpy as np
from phase1_algebra_head import build_params, forward, load_alg, decode, T_ALG, L_FAC
from repair_replace_swap import solve_forced
from tinygrad import Tensor, dtypes
from tinygrad.nn.state import safe_load

ARM=os.environ.get("ARM","BOTH"); R=3 if ARM!="FDM" else 1
samples, states, tokmask, gold, sent = load_alg("test")
rc=json.load(open('.cache/residue_census.json'))["rows"]
res=[int(r["idx"]) for r in rc]
census=json.load(open('.cache/miss_census_gen41.json')); miss=set(int(i) for i in census["miss_idx"])
rng=np.random.default_rng(41)
solved=sorted(int(x) for x in rng.choice([i for i in range(len(samples)) if i not in miss],240,replace=False))
rows=res+solved
p=build_params(0); sd=safe_load(os.environ["ALG_CKPT"])
for k in p: p[k].assign(sd[k].to(p[k].device).cast(p[k].dtype)).realize()
KEYS=("pres","ftype","op","islit","dig","args","res","query")
def fwd(slp,pm=None):
    o=forward(p,Tensor(states[slp].astype(np.float32),dtype=dtypes.float),
                Tensor(tokmask[slp].astype(np.float32),dtype=dtypes.float),
                Tensor(sent[slp].astype(np.int32),dtype=dtypes.int),
                pmask=None if pm is None else Tensor(pm.astype(np.float32),dtype=dtypes.float))
    ex=tuple(k for k in ("sel","dup","sgn") if k in o)
    return {k:o[k].realize().numpy() for k in KEYS+ex},o["fat"].realize().numpy()
def match(o,fat,bi,ri,rnd):
    txt=samples[ri]["text"]; parts=re.split(r"(?<=\.)\s+",txt.strip())
    lits=[set(int(m) for m in re.findall(r"\d+",pp)) for pp in parts]
    sn=sent[ri]; mk=tokmask[ri]>0
    smax=int(sn[mk].max()) if mk.any() else 0
    gsl=sorted(j for j in range(L_FAC) if o["pres"][bi,j]>0 and o["ftype"][bi,j].argmax()==1)
    vals={}
    for j in gsl:
        dg=o["dig"][bi,j].argmax(-1) if o["dig"][bi,j].ndim==2 else o["dig"][bi,j]
        try: vals[j]=int("".join(str(int(d)) for d in dg))
        except Exception: vals[j]=-1
    litsent=[ss for ss in range(min(len(parts),smax+1)) if lits[ss]]
    aslot={}; asent=set()
    for j in gsl:
        cand=[ss for ss in litsent if ss not in asent and vals.get(j,-1) in lits[ss]]
        if len(cand)==1: aslot[j]=cand[0]; asent.add(cand[0])
    rem=[j for j in gsl if j not in aslot]; rems=[ss for ss in litsent if ss not in asent]
    if ARM in ("FDM","BOTH") and rem:      # phase-cost fallback: avoid mod-6
        for j in rem:                      # collision with slot's neighbor
            if not rems: break
            prev=aslot.get(j-1,-99)
            best=min(rems,key=lambda ss:(1 if prev>=0 and ss%6==prev%6 else 0,abs(ss-(j))))
            aslot[j]=best; rems.remove(best)
    else:
        for j,ss in zip(rem,rems): aslot[j]=ss
    if ARM=="PLACEBO" and len(aslot)>1:
        ks=sorted(aslot); vs=[aslot[k] for k in ks]
        aslot={k:vs[(i+1+rnd)%len(vs)] for i,k in enumerate(ks)}
    pmrow=np.zeros((1,L_FAC,T_ALG),np.float32)
    n=0
    for j,ss in aslot.items():
        selm=(sn==ss)&mk
        if not selm.any(): continue
        pmrow[0,j][~selm]=-1e9; n+=1
    return pmrow,n
out={}; nmask=0
for s0 in range(0,len(rows),8):
    sl=rows[s0:s0+8]; pad=8-len(sl); slp=sl+sl[:1]*pad
    o,fat=fwd(slp); pm=np.zeros((8,1,L_FAC,T_ALG),np.float32)
    ans={}
    for bi,ri in enumerate(sl):
        facs,q=decode({k:o[k][bi] for k in o}); ans[ri]=solve_forced(facs,q,samples[ri])
    for rnd in range(R):
        act=False
        for bi,ri in enumerate(sl):
            if ans[ri] is not None: continue
            pmrow,n=match(o,fat,bi,ri,rnd); pm[bi]=pmrow; nmask+=n; act=True
        if not act: break
        o,fat=fwd(slp,pm)
        for bi,ri in enumerate(sl):
            if ans[ri] is None:
                facs,q=decode({k:o[k][bi] for k in o}); ans[ri]=solve_forced(facs,q,samples[ri])
    for ri in sl: out[ri]=ans[ri]
ga={i:samples[i]["solution"][samples[i]["query_var"]] for i in rows}
hold=sum(1 for i in solved if out[i]==ga[i])
conv=sorted(i for i in res if out[i]==ga[i])
print(f"[BC0 {ARM}] hold {hold}/240 = {hold/240:.4f}  residue converts {len(conv)}/74  {conv}")
json.dump({"arm":ARM,"conv":conv,"hold":hold},open(f'.cache/bc0_{ARM}.json','w'))
