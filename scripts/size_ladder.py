import os, sys, json, re
for k,v in [("ALG2","1"),("ALG_FTYPES","8"),("ALG_HW","512"),("ALG_DUP","1"),("ALG_WIDE","1"),("ALG_INV","1")]: os.environ.setdefault(k,v)
os.environ.setdefault("ALG_TEST",".cache/algebra_nl_bigtest.jsonl"); os.environ.setdefault("ALG_TEST_NAME","bigtest")
sys.path.insert(0,'.'); sys.path.insert(0,'scripts')
import numpy as np
from phase1_algebra_head import T_ALG, build_params, forward, decode, load_alg
from tta_alg2_dials import solve2
from tinygrad import Tensor, dtypes
from tinygrad.nn.state import safe_load
samples, states, tokmask, gold, sent = load_alg("test")
CK=os.environ["CK"]
p=build_params(0); sd=safe_load(CK)
for k in p: p[k].assign(sd[k].to(p[k].device).cast(p[k].dtype)).realize()
ORD=re.compile(r"(first|second|third|fourth|fifth) number")
HX=[];HY=[];PB={"pass":([],[]),"miss":([],[])}
for s0 in range(0,len(samples),8):
    sl=np.arange(s0,min(s0+8,len(samples)))
    pad=8-len(sl); slp=np.concatenate([sl,sl[:1].repeat(pad)]) if pad else sl
    tr=Tensor(states[slp].astype(np.float32),dtype=dtypes.float)
    tk=Tensor(tokmask[slp].astype(np.float32),dtype=dtypes.float)
    se=Tensor(sent[slp].astype(np.int32),dtype=dtypes.int)
    o=forward(p,tr,tk,se)
    keys=("pres","ftype","op","islit","dig","args","res","query","fst_s")+(("sel",) if "sel" in o else ())+(("dup",) if "dup" in o else ())+(("sgn",) if "sgn" in o else ())
    onp={k:o[k].realize().numpy() for k in keys}
    for bi,ri in enumerate(sl):
        smp=samples[ri]; t=smp["text"]
        facs,q=decode({k:onp[k][bi] for k in onp if k!="fst_s"})
        a=solve2(facs,q,{"n_vars":24,"m":300})
        hit=(a==smp["solution"][smp["query_var"]])
        is_ord=bool(ORD.search(t))
        for j in range(24):
            if gold["presence"][ri,j]<=0 or gold["ftype"][ri,j]!=0: continue
            gs=sorted(np.where(gold["args"][ri,j]>0.5)[0].tolist())
            if len(gs)!=2: continue
            if is_ord:
                g_="pass" if hit else "miss"
                PB[g_][0].append(onp["fst_s"][bi,j]); PB[g_][1].append(gs[0])
            elif hit and len(HX)<1000:
                HX.append(onp["fst_s"][bi,j]); HY.append(gs[0])
HX=np.array(HX);HY=np.array(HY)
mu,sg=HX.mean(0),HX.std(0)+1e-6
Y=np.eye(24)[HY]*2-1
W=np.linalg.lstsq(((HX-mu)/sg).T@((HX-mu)/sg)+10.0*np.eye(HX.shape[1]), ((HX-mu)/sg).T@Y, rcond=None)[0]
for g_ in ("pass","miss"):
    X=np.array(PB[g_][0]); yv=np.array(PB[g_][1])
    pr=(((X-mu)/sg)@W).argmax(1)
    print(f"[ladder {g_}] slots {len(X)} acc {(pr==yv).mean():.3f}")
