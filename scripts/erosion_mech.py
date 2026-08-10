import os, sys, json, re
for k,v in [("ALG2","1"),("ALG_FTYPES","8"),("ALG_HW","512"),("ALG_DUP","1"),("ALG_WIDE","1"),("ALG_INV","1")]: os.environ.setdefault(k,v)
os.environ.setdefault("ALG_TEST",".cache/algebra_nl_bigtest.jsonl"); os.environ.setdefault("ALG_TEST_NAME","bigtest")
sys.path.insert(0,'.'); sys.path.insert(0,'scripts')
import numpy as np
from phase1_algebra_head import T_ALG, build_params, forward, load_alg
from tinygrad import Tensor, dtypes
from tinygrad.nn.state import safe_load
samples, states, tokmask, gold, sent = load_alg("test")
p=build_params(0); sd=safe_load(".cache/g23v5.safetensors")
for k in p: p[k].assign(sd[k].to(p[k].device).cast(p[k].dtype)).realize()
ORD=re.compile(r"(first|second|third|fourth|fifth) number")
HX=[];HY=[];ROWS=[]
for s0 in range(0,len(samples),8):
    sl=np.arange(s0,min(s0+8,len(samples)))
    pad=8-len(sl); slp=np.concatenate([sl,sl[:1].repeat(pad)]) if pad else sl
    tr=Tensor(states[slp].astype(np.float32),dtype=dtypes.float)
    tk=Tensor(tokmask[slp].astype(np.float32),dtype=dtypes.float)
    se=Tensor(sent[slp].astype(np.int32),dtype=dtypes.int)
    o=forward(p,tr,tk,se)
    onp={k:o[k].realize().numpy() for k in ("pres","ftype","args","fat","fst_s")}
    for bi,ri in enumerate(sl):
        smp=samples[ri]; t=smp["text"]
        is_ord=bool(ORD.search(t))
        for j in range(24):
            if gold["presence"][ri,j]<=0 or gold["ftype"][ri,j]!=0: continue
            gs=sorted(np.where(gold["args"][ri,j]>0.5)[0].tolist())
            if len(gs)!=2: continue
            if is_ord:
                a=onp["fat"][bi,j]; a=a/max(a.sum(),1e-9)
                ent=float(-(a[a>1e-9]*np.log(a[a>1e-9])).sum())
                nf=len(smp["factors"]); ng=sum(1 for f in smp["factors"] if f.get("ftype")=="given")
                # mention distance: slot's fspan center to gold arg's nearest mention span (chars)
                md=-1.0
                m=smp.get("mentions",{}).get(str(gs[0]))
                fs=gold["fspan"][ri,j]
                if m is not None and fs.sum()>0:
                    # token center -> approx char via row text len scaling is crude; use token idx distance to mention token idx via offsets unavailable here; fallback: min |span_start - argmax(fspan token)*4|
                    ftok=int(np.argmax(fs))
                    md=min(abs(sp[0]-ftok*4) for sp in m)
                ROWS.append((onp["fst_s"][bi,j], gs[0], ent, nf, ng, md, len(t)))
            elif len(HX)<1000:
                HX.append(onp["fst_s"][bi,j]); HY.append(gs[0])
    if s0%400==0: print(f"  [{s0}]",flush=True)
HX=np.array(HX);HY=np.array(HY)
mu,sg=HX.mean(0),HX.std(0)+1e-6
Y=np.eye(24)[HY]*2-1
W=np.linalg.lstsq(((HX-mu)/sg).T@((HX-mu)/sg)+10.0*np.eye(HX.shape[1]), ((HX-mu)/sg).T@Y, rcond=None)[0]
X=np.array([r[0] for r in ROWS]); yv=np.array([r[1] for r in ROWS])
S=((X-mu)/sg)@W
true_s=S[np.arange(len(yv)),yv]
S2=S.copy(); S2[np.arange(len(yv)),yv]=-1e9
margin=true_s-S2.max(1)
ent=np.array([r[2] for r in ROWS]); nf=np.array([r[3] for r in ROWS]); ng=np.array([r[4] for r in ROWS])
md=np.array([r[5] for r in ROWS]); tl=np.array([r[6] for r in ROWS])
from scipy.stats import spearmanr
print(f"[n slots] {len(margin)}  margin mean {margin.mean():.3f}")
for name,f in (("dilution(entropy)",ent),("contention(n_factors)",nf),("givens",ng),("mention_dist",md),("text_len",tl)):
    ok=f>=0
    rho,pv=spearmanr(margin[ok],f[ok])
    print(f"[cand {name}] spearman(margin, feat) {rho:+.3f} p={pv:.2g}")
