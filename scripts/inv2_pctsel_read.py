"""pct/sel sub-valve support (2026-08-07; inv2's owed reads): pct at 7
and sel at 8 phrasings are both sub-valve. If the pair-data effect is
real, ictl beats iinv/v120 on guard-verified never-licensed forms."""
import os, sys, json
for k,v in [("ALG2","1"),("ALG_FTYPES","8"),("ALG_HW","512"),("ALG_DUP","1")]: os.environ.setdefault(k,v)
sys.path.insert(0,'.'); sys.path.insert(0,'scripts')
import numpy as np
from collections import Counter
from phase1_algebra_head import T_ALG, build_params, forward, decode, sent_indices, TOKENIZER_JSON
from beacon_closing_arm import recompute_states
from tta_views import permuted_view
from tta_alg2_dials import solve2
from tokenizers import Tokenizer
from tinygrad import Tensor, dtypes
from tinygrad.nn.state import safe_load
tok=Tokenizer.from_file(TOKENIZER_JSON)
T=json.load(open('.cache/aug_table_v3.json'))["licensed"]
for probe in ("amounts to","bigger one"):
    assert not any(probe in e["fmt"] for e in T), probe
print("[guard] pct/sel probes ABSENT from licensed table")
rng=np.random.RandomState(94000)
P=[]  # pct rows
while len(P)<8:
    p=int(rng.choice([10,20,25,50])); b2=int(rng.randint(2,12))*20
    g=p*b2//100
    if b2>300 or g<1 or g>300: continue
    P.append({"text":f"Consider the numbers a, b. a is {b2}. {p} percent of a amounts to b. What is b?","q":1,"gold":g})
S=[]  # sel rows
while len(S)<8:
    a,b=int(rng.randint(2,90)),int(rng.randint(2,90))
    if a==b: continue
    S.append({"text":f"Consider the numbers a, b, c. a is {a}. b is {b}. Of a and b, the bigger one is c. What is c?","q":2,"gold":max(a,b)})
def load(ck):
    p=build_params(0); sd=safe_load(ck)
    for k in p: p[k].assign(sd[k].to(p[k].device).cast(p[k].dtype)).realize()
    return p
def one(p,t):
    ids=np.zeros((8,T_ALG),np.int32); msk=np.zeros((8,T_ALG),np.float32); snt=np.zeros((8,T_ALG),np.int32)
    e=tok.encode(t); Ln=min(len(e.ids),T_ALG)
    ids[0,:Ln]=e.ids[:Ln]; msk[0,:Ln]=1.0
    snt[0]=sent_indices(t,list(e.offsets),msk[0])
    out=forward(p,Tensor(recompute_states(ids).astype(np.float32),dtype=dtypes.float),
                Tensor(msk,dtype=dtypes.float),Tensor(snt,dtype=dtypes.int))
    keys=("pres","ftype","op","islit","dig","args","res","query")+(("sel",) if "sel" in out else ())+(("dup",) if "dup" in out else ())
    o={k:out[k].realize().numpy() for k in keys}
    return decode({k:o[k][0] for k in o})[0]
res={}
for name,ck in (("ictl",".cache/g25v2_inv_ctl.safetensors"),("iinv",".cache/g25v2_inv_inv.safetensors"),("v120",".cache/g24_v120.safetensors")):
    p=load(ck); out={}
    for lab,rows in (("pct",P),("sel",S)):
        nq=0
        for j,r in enumerate(rows):
            vt=[r["text"]]+[permuted_view(r["text"],94100+40*j+k) for k in range(1,5)]
            ans=[solve2(one(p,t),r["q"],{"n_vars":24,"m":300}) for t in vt]
            nn=[a for a in ans if a is not None]
            c=Counter(nn).most_common(1); plur,cnt=c[0] if c else (None,0)
            nq+=(cnt>=3 and plur==r["gold"])
        out[lab]=nq
    res[name]=out
    print(f"[{name}] pct-novel {out['pct']}/8   sel-novel {out['sel']}/8",flush=True)
json.dump(res,open('.cache/inv2_pctsel.json','w'))
