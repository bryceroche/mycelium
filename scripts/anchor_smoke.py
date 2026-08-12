"""anchor_smoke.py — FORM (A)'s object smoke (inference-only): pass-1
fst states at HIGH-CONFIDENCE slots pinned as pass-2 anchors. Both
populations from birth (the denominator's lesson): the 233 + 400
passers."""
import os, sys, json
for k,v in [("ALG2","1"),("ALG_FTYPES","8"),("ALG_HW","512"),("ALG_DUP","1"),("ALG_WIDE","1"),("ALG_INV","1")]: os.environ.setdefault(k,v)
os.environ.setdefault("ALG_TEST",".cache/algebra_nl_bigtest.jsonl"); os.environ.setdefault("ALG_TEST_NAME","bigtest")
sys.path.insert(0,'.'); sys.path.insert(0,'scripts')
import numpy as np
from phase1_algebra_head import T_ALG, build_params, forward, decode, load_alg, L_FAC
from tta_alg2_dials import solve2
from tinygrad import Tensor, dtypes
from tinygrad.nn.state import safe_load
samples, states, tokmask, gold, sent = load_alg("test")
base=json.load(open('.cache/miss_census_gen41.json'))
miss=set(base["miss_idx"])
rng=np.random.RandomState(7)
passers=[i for i in range(len(samples)) if i not in miss]
ROWS=sorted(miss)+list(rng.choice(passers,400,replace=False))
p=build_params(0); sd=safe_load(json.load(open('.cache/GENERATION.json'))["parser_ckpt"])
for k in p: p[k].assign(sd[k].to(p[k].device).cast(p[k].dtype)).realize()
PRES_G=float(os.environ.get("ANCH_PRES","4.0")); ARG_G=float(os.environ.get("ANCH_ARG","4.0"))
res={"miss":[0,0],"pass":[0,0]}; n_anch=[]
for s0 in range(0,len(ROWS),8):
    sl=np.array(ROWS[s0:s0+8])
    pad=8-len(sl); slp=np.concatenate([sl,sl[:1].repeat(pad)]) if pad else sl
    tr=Tensor(states[slp].astype(np.float32),dtype=dtypes.float)
    tk=Tensor(tokmask[slp].astype(np.float32),dtype=dtypes.float)
    se=Tensor(sent[slp].astype(np.int32),dtype=dtypes.int)
    o1=forward(p,tr,tk,se)
    keys=("pres","ftype","op","islit","dig","args","res","query","fst_s")+(("sel",) if "sel" in o1 else ())+(("dup",) if "dup" in o1 else ())+(("sgn",) if "sgn" in o1 else ())
    onp={k2:o1[k2].realize().numpy() for k2 in keys}
    anch=np.zeros((8,L_FAC,512),np.float32); am=np.zeros((8,L_FAC,1),np.float32)
    for bi in range(8):
        for j in range(L_FAC):
            if onp["pres"][bi,j]>PRES_G:
                a=np.sort(onp["args"][bi,j])[::-1]
                if len(a)>2 and (a[1]-a[2])>ARG_G:
                    anch[bi,j]=onp["fst_s"][bi,j]; am[bi,j]=1.0
    n_anch.append(am.sum()/8)
    o2=forward(p,tr,tk,se,anchor=Tensor(anch,dtype=dtypes.float),amask=Tensor(am,dtype=dtypes.float))
    onp2={k2:o2[k2].realize().numpy() for k2 in keys if k2!="fst_s"}
    for bi,ri in enumerate(sl):
        g_=samples[ri]["solution"][samples[ri]["query_var"]]
        f1,q1=decode({k2:onp[k2][bi] for k2 in onp if k2!="fst_s"})
        f2,q2=decode({k2:onp2[k2][bi] for k2 in onp2})
        a1=solve2(f1,q1,{"n_vars":24,"m":300}); a2=solve2(f2,q2,{"n_vars":24,"m":300})
        pop="miss" if ri in miss else "pass"
        if a1!=g_ and a2==g_: res[pop][0]+=1
        if a1==g_ and a2!=g_: res[pop][1]+=1
print(f"[anchor] mean anchored slots/row {np.mean(n_anch):.1f}")
print(f"[anchor] MISS pop: converts {res['miss'][0]}  regressions {res['miss'][1]} (of {len(miss)})")
print(f"[anchor] PASS pop: converts {res['pass'][0]}  regressions {res['pass'][1]} (of 400)")
json.dump(res,open('.cache/anchor_smoke.json','w'))
