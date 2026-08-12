import os, sys, json
os.environ["ALG_BREATH"]="3"
for k,v in [("ALG2","1"),("ALG_FTYPES","8"),("ALG_HW","512"),("ALG_DUP","1"),("ALG_WIDE","1"),("ALG_INV","1")]: os.environ.setdefault(k,v)
os.environ.setdefault("ALG_TEST",".cache/algebra_nl_bigtest.jsonl"); os.environ.setdefault("ALG_TEST_NAME","bigtest")
sys.path.insert(0,'.'); sys.path.insert(0,'scripts')
import numpy as np
from phase1_algebra_head import build_params, forward, decode, load_alg, L_FAC, build_slot_masks
from tta_alg2_dials import solve2
from tinygrad import Tensor, dtypes
from tinygrad.nn.state import safe_load
samples, states, tokmask, gold, sent = load_alg("test")
base=json.load(open('.cache/miss_census_gen41.json')); om=set(base["miss_idx"])
rng=np.random.RandomState(7)
ROWS=list(rng.choice([i for i in range(len(samples)) if i not in om],150,replace=False))
p=build_params(0); sd=safe_load(".cache/g51_whisper.safetensors")
for k in p: p[k].assign(sd[k].to(p[k].device).cast(p[k].dtype)).realize()
def argpair(oo,bi,j):
    if oo["dup"][bi,j]>0:
        a0=int(np.argmax(oo["args"][bi,j])); return (a0,a0)
    return tuple(sorted(np.argsort(-oo["args"][bi,j])[:2].tolist()))
BG=float(os.environ.get("BG_AUTH","0.05"))
reg={"uni":0,"foc":0}
for s0 in range(0,len(ROWS),8):
    sl=np.array(ROWS[s0:s0+8])
    pad=8-len(sl); slp=np.concatenate([sl,sl[:1].repeat(pad)]) if pad else sl
    tr=Tensor(states[slp].astype(np.float32),dtype=dtypes.float)
    tk=Tensor(tokmask[slp].astype(np.float32),dtype=dtypes.float)
    se=Tensor(sent[slp].astype(np.int32),dtype=dtypes.int)
    o0=forward(p,tr,tk,se)
    o0n={k2:o0[k2].realize().numpy() for k2 in ("fat","args","res")}
    mk=build_slot_masks(o0n, sent[slp])
    oe=forward(p,tr,tk,se,slot_mask=Tensor(mk,dtype=dtypes.float))
    keys=("pres","ftype","op","islit","dig","args","res","query")+(("sel",) if "sel" in oe else ())+(("dup",) if "dup" in oe else ())+(("sgn",) if "sgn" in oe else ())
    en={k2:oe[k2].realize().numpy() for k2 in keys}
    sn={k2:o0[k2].realize().numpy() for k2 in keys}
    gm=np.full((8,L_FAC,1),BG,np.float32)
    for bi in range(8):
        for j in range(L_FAC):
            if en["pres"][bi,j]>0 and sn["pres"][bi,j]>0 and argpair(en,bi,j)!=argpair(sn,bi,j): gm[bi,j]=1.0
    of=forward(p,tr,tk,se,slot_mask=Tensor(mk,dtype=dtypes.float),gmod=Tensor(gm,dtype=dtypes.float))
    fn={k2:of[k2].realize().numpy() for k2 in keys}
    for bi,ri in enumerate(sl):
        g_=samples[ri]["solution"][samples[ri]["query_var"]]
        for nm,oo in (("uni",en),("foc",fn)):
            facs,q=decode({k2:oo[k2][bi] for k2 in oo})
            if solve2(facs,q,{"n_vars":24,"m":300})!=g_: reg[nm]+=1
print(f"[focus cost] passers(150) wrong: uniform-engaged {reg['uni']}  FOCUSED(bg={BG}) {reg['foc']}",flush=True)
