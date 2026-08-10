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
p=build_params(0); sd=safe_load(".cache/g23v5.safetensors")
for k in p: p[k].assign(sd[k].to(p[k].device).cast(p[k].dtype)).realize()
ORD=re.compile(r"(first|second|third|fourth|fifth) number")
SUB=re.compile(r"minus|difference|exceeds|reduced|away from|less than|fewer|decreased")
FEAT={"miss":[], "pass":[]}
PX={"miss":[], "pass":[]}; PY={"miss":[], "pass":[]}
HX=[]; HY=[]
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
        ng=sum(1 for f in smp["factors"] if f.get("ftype")=="given")
        nf=len(smp["factors"])
        n_ord=len(ORD.findall(t)); n_sub=len(SUB.findall(t))
        if is_ord:
            g_="pass" if hit else "miss"
            FEAT[g_].append((ng,nf,len(t),n_ord,n_sub))
            for j in range(24):
                if gold["presence"][ri,j]<=0 or gold["ftype"][ri,j]!=0: continue
                gs=sorted(np.where(gold["args"][ri,j]>0.5)[0].tolist())
                if len(gs)!=2: continue
                PX[g_].append(onp["fst_s"][bi,j]); PY[g_].append(gs[0])
        elif hit:
            for j in range(24):
                if gold["presence"][ri,j]<=0 or gold["ftype"][ri,j]!=0: continue
                gs=sorted(np.where(gold["args"][ri,j]>0.5)[0].tolist())
                if len(gs)!=2 or len(HX)>=1000: continue
                HX.append(onp["fst_s"][bi,j]); HY.append(gs[0])
    if s0%400==0: print(f"  [{s0}]",flush=True)
for g_ in ("miss","pass"):
    F=np.array(FEAT[g_])
    print(f"[feat {g_}] n={len(F)} givens {F[:,0].mean():.2f} factors {F[:,1].mean():.2f} len {F[:,2].mean():.0f} ordinals {F[:,3].mean():.2f} subs {F[:,4].mean():.2f}")
HX=np.array(HX);HY=np.array(HY)
mu,sg=HX.mean(0),HX.std(0)+1e-6
Y=np.eye(24)[HY]*2-1
W=np.linalg.lstsq(((HX-mu)/sg).T@((HX-mu)/sg)+10.0*np.eye(HX.shape[1]), ((HX-mu)/sg).T@Y, rcond=None)[0]
for g_ in ("pass","miss"):
    X=np.array(PX[g_]); yv=np.array(PY[g_])
    pr=(((X-mu)/sg)@W).argmax(1)
    print(f"[probe {g_}] slots {len(X)} first-arg acc {(pr==yv).mean():.3f}")
