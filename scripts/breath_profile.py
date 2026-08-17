"""breath_profile.py — per-breath decodability (pinned sheet read) +
overlap gap-over-floor. Acquittal decisive / conviction provisional
(the ordering predates the overlap collapse; flat = ambiguous)."""
import os, sys, json
sys.path.insert(0,'.'); sys.path.insert(0,'scripts')
import numpy as np
from phase1_algebra_head import (build_params, forward, load_alg, decode,
                                 build_slot_masks, T_ALG, L_FAC)
from tinygrad import Tensor, dtypes
from tinygrad.nn.state import safe_load
CK=os.environ["PR_CK"]; TAG=os.path.basename(CK).split(".")[0]
samples, states, tokmask, gold, sent = load_alg("test")
rc=json.load(open('.cache/residue_census.json'))["rows"]
rows=[int(r["idx"]) for r in rc]+sorted(int(x) for x in np.random.default_rng(41).choice(1500,76,replace=False))
p=build_params(0); sd=safe_load(CK)
for k in p: p[k].assign(sd[k].to(p[k].device).cast(p[k].dtype)).realize()
def gk(f):
    if f["ftype"]=="rel": return ("rel",f["op"],tuple(sorted(f["args"])),f["result"])
    if f["ftype"]=="given": return ("given",f["var"],int(f["value"]))
    return (f["ftype"],)
K=("pres","ftype","op","islit","dig","args","res","query")
acc={}; fl=[]
for s0 in range(0,len(rows),8):
    sl=rows[s0:s0+8]; pad=8-len(sl); slp=sl+sl[:1]*pad
    ts=Tensor(states[slp].astype(np.float32),dtype=dtypes.float)
    tk=Tensor(tokmask[slp].astype(np.float32),dtype=dtypes.float)
    se=Tensor(sent[slp].astype(np.int32),dtype=dtypes.int)
    ls=Tensor(gold["lsent"][slp].astype(np.float32),dtype=dtypes.float) if os.environ.get("ALG_LSENT")=="1" and "lsent" in gold else None
    o1=forward(p,ts,tk,se,lsent=ls)
    o1n={k:o1[k].realize().numpy() for k in ("fat","args","res")}
    mk=build_slot_masks(o1n,sent[slp])
    o=forward(p,ts,tk,se,slot_mask=Tensor(mk,dtype=dtypes.float),lsent=ls)
    stages=[e for e in o.get("_early",[])]+[o]
    fat=o["fat"].realize().numpy()
    for si,st in enumerate(stages):
        onp={k:(st[k] if k in st else o[k]).realize().numpy() for k in K}
        for bi,ri in enumerate(sl):
            facs,q=decode({k:onp[k][bi] for k in onp})
            gs={};ds={}
            for f in samples[ri]["factors"]: gs[gk(f)]=gs.get(gk(f),0)+1
            for f in facs:
                kk=gk({"ftype":f.get("ftype"),"op":f.get("op"),"args":f.get("args",[]),"result":f.get("result"),"var":f.get("var"),"value":f.get("value",0)}) if f.get("ftype") in ("rel","given") else ("x",)
                ds[kk]=ds.get(kk,0)+1
            m=sum(min(gs.get(k2,0),ds.get(k2,0)) for k2 in gs)/max(sum(gs.values()),1)
            acc.setdefault(si,[]).append(m)
    for bi,ri in enumerate(sl):   # floor: permuted-attention chance overlap
        F=fat[bi][:8]; F=F/(np.linalg.norm(F,axis=1,keepdims=True)+1e-9)
        rngp=np.random.default_rng(ri)
        Fp=np.stack([rngp.permutation(f) for f in F])
        C=Fp@Fp.T; fl.append(np.mean([C[a,b] for a in range(8) for b in range(a+1,8)]))
prof=[float(np.mean(acc[s])) for s in sorted(acc)]
print(f"[profile {TAG}] per-breath factor-match: "+" -> ".join(f"{x:.3f}" for x in prof))
print(f"[profile {TAG}] permuted-attention overlap floor: {np.mean(fl):.4f}")
