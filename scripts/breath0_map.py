"""breath0_map.py — the MLIR gut's rent, now the selective gate's
design input: does BREATH-0 state predict which of the 233 the organ
converts? Features: breath-0 pooled row state (g50, the strongest
arm). Target: union of converts across g50/g50r/g51."""
import os, sys, json
os.environ["ALG_BREATH"]="3"
for k,v in [("ALG2","1"),("ALG_FTYPES","8"),("ALG_HW","512"),("ALG_DUP","1"),("ALG_WIDE","1")]: os.environ.setdefault(k,v)
os.environ.setdefault("ALG_TEST",".cache/algebra_nl_bigtest.jsonl"); os.environ.setdefault("ALG_TEST_NAME","bigtest")
sys.path.insert(0,'.'); sys.path.insert(0,'scripts')
import numpy as np
from phase1_algebra_head import build_params, forward, load_alg
from tinygrad import Tensor, dtypes
from tinygrad.nn.state import safe_load
samples, states, tokmask, gold, sent = load_alg("test")
base=json.load(open('.cache/miss_census_gen41.json'))
om=set(base["miss_idx"])
conv=set()
for f in ("miss_census_g50.json","miss_census_g50r.json","miss_census_g51.json"):
    nm=set(json.load(open('.cache/'+f))["miss_idx"]); conv|= (om-nm)
ROWS=sorted(om)
y=np.array([1 if i in conv else 0 for i in ROWS])
print(f"[map] the 233: union converts {y.sum()} / {len(y)}")
p=build_params(0); sd=safe_load(".cache/g50_boot.safetensors")
for k in p: p[k].assign(sd[k].to(p[k].device).cast(p[k].dtype)).realize()
X=[]
for s0 in range(0,len(ROWS),8):
    sl=np.array(ROWS[s0:s0+8])
    pad=8-len(sl); slp=np.concatenate([sl,sl[:1].repeat(pad)]) if pad else sl
    o=forward(p,Tensor(states[slp].astype(np.float32),dtype=dtypes.float),
              Tensor(tokmask[slp].astype(np.float32),dtype=dtypes.float),
              Tensor(sent[slp].astype(np.int32),dtype=dtypes.int))
    b0=o["breaths"][0]
    feats=[]
    for k2 in ("pres","dup","op","res"):
        if k2 in b0: feats.append(b0[k2].realize().numpy().reshape(8,-1))
    arr=np.concatenate(feats,axis=1)
    for bi in range(len(sl)): X.append(arr[bi])
X=np.array(X)
n=len(X); tr=np.arange(n)%2==0; te=~tr
mu,sg=X[tr].mean(0),X[tr].std(0)+1e-6
Xs=(X-mu)/sg
w=np.linalg.lstsq(Xs[tr].T@Xs[tr]+10.0*np.eye(X.shape[1]), Xs[tr].T@(2*y[tr]-1), rcond=None)[0]
sc=Xs[te]@w
pos=sc[y[te]==1]; neg=sc[y[te]==0]
auc=float((pos[:,None]>neg[None,:]).mean()) if len(pos) and len(neg) else float("nan")
print(f"[map] BREATH-0 -> convert-membership AUC {auc:.3f} (holdout: {int(y[te].sum())} pos / {int((1-y[te]).sum())} neg)")
