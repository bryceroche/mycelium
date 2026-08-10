import os, sys, json, re
for k,v in [("ALG2","1"),("ALG_FTYPES","8"),("ALG_HW","512"),("ALG_DUP","1"),("ALG_WIDE","1")]: os.environ.setdefault(k,v)
os.environ.setdefault("ALG_TEST",".cache/algebra_nl_bigtest.jsonl"); os.environ.setdefault("ALG_TEST_NAME","bigtest")
sys.path.insert(0,'.'); sys.path.insert(0,'scripts')
import numpy as np
from collections import Counter
from phase1_algebra_head import T_ALG, build_params, forward, decode, load_alg
from tta_alg2_dials import solve2
from tinygrad import Tensor, dtypes
from tinygrad.nn.state import safe_load
samples, states, tokmask, gold, sent = load_alg("test")
p=build_params(0); sd=safe_load(".cache/g23v5.safetensors")
for k in p: p[k].assign(sd[k].to(p[k].device).cast(p[k].dtype)).realize()
POWG=10**np.arange(gold["digits"].shape[-1]-1,-1,-1)
POWM=10**np.arange(6,-1,-1)
stations=Counter(); n_miss=0
for s0 in range(0,len(samples),8):
    sl=np.arange(s0,min(s0+8,len(samples)))
    pad=8-len(sl); slp=np.concatenate([sl,sl[:1].repeat(pad)]) if pad else sl
    tr=Tensor(states[slp].astype(np.float32),dtype=dtypes.float)
    tk=Tensor(tokmask[slp].astype(np.float32),dtype=dtypes.float)
    se=Tensor(sent[slp].astype(np.int32),dtype=dtypes.int)
    o=forward(p,tr,tk,se)
    keys=("pres","ftype","op","islit","dig","args","res","query")+(("sel",) if "sel" in o else ())+(("dup",) if "dup" in o else ())+(("sgn",) if "sgn" in o else ())
    onp={k:o[k].realize().numpy() for k in keys}
    for bi,ri in enumerate(sl):
        smp=samples[ri]
        facs,q=decode({k:onp[k][bi] for k in onp})
        a=solve2(facs,q,{"n_vars":24,"m":300})
        goldans=smp["solution"][smp["query_var"]]
        if a==goldans: continue
        n_miss+=1
        # first failing station by slot-grade gold comparison
        st=None
        for j in range(24):
            gp=gold["presence"][ri,j]
            pp=onp["pres"][bi,j]>0
            if gp>0 and not pp: st="missing-slot"; break
            if gp<=0 and pp: st="ghost-slot"; break
            if gp<=0: continue
            if onp["ftype"][bi,j].argmax()!=gold["ftype"][ri,j]: st="ftype"; break
            if gold["ftype"][ri,j]==0:
                if onp["op"][bi,j].argmax()!=gold["op"][ri,j]: st="op"; break
                gset=set(np.where(gold["args"][ri,j]>0.5)[0].tolist())
                if len(gset)==1:
                    if int(np.argmax(onp.get("dargs",onp["args"])[bi,j] if "dargs" in onp else onp["args"][bi,j])) not in gset: st="args-dup"; break
                else:
                    if set(np.argsort(-onp["args"][bi,j])[:2].tolist())!=gset: st="args"; break
            else:
                gd=int(gold["digits"][ri,j] @ POWG)
                dd=int((onp["dig"][bi,j].argmax(-1) * POWM).sum())
                if dd!=gd: st="digits"; break
            if onp["res"][bi,j].argmax()!=gold["res"][ri,j]: st="res"; break
        if st is None:
            if int(onp["query"][bi].argmax())!=int(gold["query"][ri]): st="query"
            else: st="graph-ok-solve-wrong"
        stations[st]+=1
    if s0%400==0: print(f"  [{s0}]",flush=True)
print(f"[stations] misses {n_miss}")
print("[histogram]", dict(stations.most_common()))
json.dump(dict(stations),open('.cache/tail_stations.json','w'))
