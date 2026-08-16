"""shuffle_a0.py — THE SHUFFLE ORGAN, RUNG A0 (registered; bars pinned):
neural proposes (pass-1 fat = affinity), symbolic disposes (1:1 greedy
matching of given slots to sentences), pass-2 reads under imposed route
masks. Zero training. Bars: solved hold >=0.98 (240 sample); residue-74
conversions >=5 claims contact, 0-4 reported as null. Deployment-honest:
pass-1 decode identifies given slots; no gold in the loop."""
import os, sys, json
sys.path.insert(0,'.'); sys.path.insert(0,'scripts')
import numpy as np
from phase1_algebra_head import build_params, forward, load_alg, decode, T_ALG, L_FAC
from repair_replace_swap import solve_forced
from tinygrad import Tensor, dtypes
from tinygrad.nn.state import safe_load

samples, states, tokmask, gold, sent = load_alg("test")
rc=json.load(open('.cache/residue_census.json'))["rows"]
res=[r["idx"] for r in rc]
census=json.load(open('.cache/miss_census_gen41.json')); miss=set(int(i) for i in census["miss_idx"])
rng=np.random.default_rng(41)
solved=sorted(rng.choice([i for i in range(len(samples)) if i not in miss],240,replace=False))
rows=[int(x) for x in res]+[int(x) for x in solved]
p=build_params(0); sd=safe_load(os.environ["ALG_CKPT"])
for k in p: p[k].assign(sd[k].to(p[k].device).cast(p[k].dtype)).realize()
KEYS=("pres","ftype","op","islit","dig","args","res","query")
def fwd(slp, pm=None):
    o=forward(p,Tensor(states[slp].astype(np.float32),dtype=dtypes.float),
                Tensor(tokmask[slp].astype(np.float32),dtype=dtypes.float),
                Tensor(sent[slp].astype(np.int32),dtype=dtypes.int),
                pmask=None if pm is None else Tensor(pm.astype(np.float32),dtype=dtypes.float))
    ex=tuple(k for k in ("sel","dup","sgn") if k in o)
    return {k:o[k].realize().numpy() for k in KEYS+ex},o["fat"].realize().numpy()
n_reroute=0; n_masked=0
out1={}; out2={}
for s0 in range(0,len(rows),8):
    sl=rows[s0:s0+8]; pad=8-len(sl); slp=sl+sl[:1]*pad
    o1,fat=fwd(slp)
    pm=np.zeros((8,1,L_FAC,T_ALG),np.float32)
    for bi,ri in enumerate(sl):
        facs,q=decode({k:o1[k][bi] for k in o1})
        out1[ri]=solve_forced(facs,q,samples[ri])
        gslots=[j for j in range(L_FAC) if o1["pres"][bi,j]>0 and o1["ftype"][bi,j].argmax()==1]
        sn=sent[ri]; mk=tokmask[ri]>0
        smax=int(sn[mk].max()) if mk.any() else 0
        if smax<2: continue
        aff=np.zeros((len(gslots),smax+1),np.float32)
        for gi,j in enumerate(gslots):
            for ss in range(smax+1):
                selm=(sn==ss)&mk
                aff[gi,ss]=fat[bi,j][selm].sum()
        order=np.dstack(np.unravel_index(np.argsort(-aff,axis=None),aff.shape))[0]
        aslot={}; asent=set()
        for gi,ss in order:
            if gi in aslot or ss in asent: continue
            aslot[int(gi)]=int(ss); asent.add(int(ss))
            if len(aslot)==min(len(gslots),smax+1): break
        for gi,j in enumerate(gslots):
            if gi not in aslot: continue
            ss=aslot[gi]
            if int(np.argmax(aff[gi]))!=ss: n_reroute+=1
            selm=(sn==ss)&mk
            pm[bi,0,j][~selm]=-1e9; n_masked+=1
    o2,_=fwd(slp,pm)
    for bi,ri in enumerate(sl):
        facs,q=decode({k:o2[k][bi] for k in o2})
        out2[ri]=solve_forced(facs,q,samples[ri])
ga={i:samples[i]["solution"][samples[i]["query_var"]] for i in rows}
hold=sum(1 for i in solved if out2[i]==ga[i])
base_hold=sum(1 for i in solved if out1[i]==ga[i])
conv=[i for i in res if out2[i]==ga[i]]
ref=[r["idx"] for r in rc if r["mode"]=="unforced"]
convr=[i for i in conv if i in set(ref)]
print(f"[A0] masked given-slots {n_masked}  REROUTED by matching {n_reroute}")
print(f"[A0] SOLVED HOLD {hold}/240 = {hold/240:.4f} (bar >=0.98; pass-1 baseline {base_hold}/240)")
print(f"[A0] RESIDUE-74 CONVERSIONS {len(conv)}/74 (floor >=5)  — from refusal-65: {len(convr)}")
print(f"[A0] {'CONTACT' if len(conv)>=5 else 'NULL'} | hold {'PASS' if hold/240>=0.98 else 'FAIL'}")
json.dump({"conv":sorted(conv),"hold":hold,"reroutes":n_reroute},open('.cache/shuffle_a0.json','w'))
