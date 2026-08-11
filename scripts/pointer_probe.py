import os, sys, json
for k,v in [("ALG2","1"),("ALG_FTYPES","8"),("ALG_HW","512"),("ALG_DUP","1"),("ALG_WIDE","1"),("ALG_INV","1")]: os.environ.setdefault(k,v)
os.environ.setdefault("ALG_TEST",".cache/algebra_nl_bigtest.jsonl"); os.environ.setdefault("ALG_TEST_NAME","bigtest")
sys.path.insert(0,'.'); sys.path.insert(0,'scripts')
import numpy as np
from phase1_algebra_head import T_ALG, build_params, forward, decode, load_alg
from tta_alg2_dials import solve2
from tinygrad import Tensor, dtypes
from tinygrad.nn.state import safe_load
samples, states, tokmask, gold, sent = load_alg("test")
p=build_params(0); sd=safe_load(os.environ.get("PROBE_CKPT",".cache/g23v5.safetensors"))
for k in p: p[k].assign(sd[k].to(p[k].device).cast(p[k].dtype)).realize()
TX=[]; TY=[]; HX=[]; HY=[]
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
        smp=samples[ri]
        facs,q=decode({k:onp[k][bi] for k in onp if k!="fst_s"})
        a=solve2(facs,q,{"n_vars":24,"m":300})
        hit=(a==smp["solution"][smp["query_var"]])
        for j in range(24):
            if gold["presence"][ri,j]<=0 or gold["ftype"][ri,j]!=0: continue
            gset=sorted(np.where(gold["args"][ri,j]>0.5)[0].tolist())
            if len(gset)!=2: continue
            pred=sorted(np.argsort(-onp["args"][bi,j])[:2].tolist())
            wrong=(pred!=gset)
            x=onp["fst_s"][0 if False else bi,j] if onp["fst_s"].shape[0]>bi else onp["fst_s"][0,j]
            x=onp["fst_s"][bi,j]
            if (not hit) and wrong and len(TX)<400: TX.append(x); TY.append(gset[0])
            elif hit and not wrong and len(HX)<800: HX.append(x); HY.append(gset[0])
    if s0%400==0: print(f"  [{s0}] tail {len(TX)} healthy {len(HX)}",flush=True)
TX=np.array(TX);TY=np.array(TY);HX=np.array(HX);HY=np.array(HY)
print(f"[pools] tail-wrong-arg slots {len(TX)}  healthy {len(HX)}",flush=True)
ht=len(TX)//2
Xtr=np.vstack([HX,TX[:ht]]); ytr=np.concatenate([HY,TY[:ht]])
mu,sg=Xtr.mean(0),Xtr.std(0)+1e-6
Xs=(Xtr-mu)/sg
Y=np.eye(24)[ytr]*2-1
W=np.linalg.lstsq(Xs.T@Xs+10.0*np.eye(Xs.shape[1]), Xs.T@Y, rcond=None)[0]
pr=(((TX[ht:]-mu)/sg)@W).argmax(1)
acc=(pr==TY[ht:]).mean()
prh=(((HX-mu)/sg)@W).argmax(1)
print(f"[probe] held-out TAIL first-arg acc {acc:.3f} (n={len(TY)-ht}) | healthy train-fit {(prh==HY).mean():.3f}")
json.dump({"tail_acc":float(acc),"n":int(len(TY)-ht)},open(os.environ.get("PROBE_OUT",".cache/pointer_probe.json"),"w"))
