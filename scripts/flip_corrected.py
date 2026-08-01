"""flip_corrected.py — THE CORRECTED FLIP INSTRUMENT (2026-08-02).
Own-vs-own per ckpt, one transform vintage. Bars RELATIVE (0.75xB /
0.5xB, B = g22's re-based rate). arm-vs-g22 = augmentation;
arm-vs-arm = sink."""
import sys, os, json
sys.path.insert(0,'.'); sys.path.insert(0,'scripts')
os.environ.setdefault("ALG2","1"); os.environ.setdefault("ALG_FTYPES","8")
os.environ.setdefault("ALG_HW","512"); os.environ.setdefault("ALG_DUP","1")
import numpy as np
from collections import Counter
from phase1_algebra_head import T_ALG, build_params, forward, decode, sent_indices, TOKENIZER_JSON
from beacon_closing_arm import recompute_states
from tta_views import permuted_view
from tta_alg2_dials import solve2
from binding_invariance_read import transform
from tokenizers import Tokenizer
from tinygrad import Tensor, dtypes
from tinygrad.nn.state import safe_load
tok = Tokenizer.from_file(TOKENIZER_JSON)
recs=[json.loads(l) for l in open('.cache/wild_ledger_v1.jsonl')]
ans=[r for r in recs if r["tier"]=="answered"]
h=[json.loads(l) for l in open('.cache/math_harvest_v0.jsonl')]
items=[]
for j,r in enumerate(ans):
    t=h[r["harvest_idx"]]["problem"]
    nt,kind=transform(t)
    if nt: items.append((j,t,nt))
print(f"[flip] transformable items: {len(items)}/124 (one vintage, post-rename-fix)")

def parse_batch(p, texts):
    n=len(texts); N=((n+7)//8)*8
    ids=np.zeros((N,T_ALG),np.int32); msk=np.zeros((N,T_ALG),np.float32); snt=np.zeros((N,T_ALG),np.int32)
    for i,t in enumerate(texts):
        e=tok.encode(t); Ln=min(len(e.ids),T_ALG)
        ids[i,:Ln]=e.ids[:Ln]; msk[i,:Ln]=1.0
        snt[i]=sent_indices(t,list(e.offsets),msk[i])
    st=recompute_states(ids)
    out_r=[]
    for s0 in range(0,N,8):
        out=forward(p,Tensor(st[s0:s0+8].astype(np.float32),dtype=dtypes.float),
                    Tensor(msk[s0:s0+8].astype(np.float32),dtype=dtypes.float),
                    Tensor(snt[s0:s0+8].astype(np.int32),dtype=dtypes.int))
        keys=("pres","ftype","op","islit","dig","args","res","query")+(("sel",) if "sel" in out else ())+(("dup",) if "dup" in out else ())
        o={k:out[k].realize().numpy() for k in keys}
        for bi in range(8):
            if s0+bi<n: out_r.append(decode({k:o[k][bi] for k in o}))
    return out_r

def quorum_of(p, text, base):
    vt=[text]+[permuted_view(text,base+k) for k in range(1,5)]
    a=[solve2(f,q,{"n_vars":24,"m":300}) for f,q in parse_batch(p,vt)]
    nn=[x for x in a if x is not None]
    c=Counter(nn).most_common(1)
    return c[0] if c else (None,0)

results={}
for name,ck in (("g22",".cache/g22.safetensors"),("vlow",".cache/g24_vlow.safetensors"),("vfull",".cache/g24_vfull.safetensors")):
    p=build_params(0); sd=safe_load(ck)
    for k in p: p[k].assign(sd[k].to(p[k].device).cast(p[k].dtype)).realize()
    flips=stable=uncounted=0
    for idx,(j,t,nt) in enumerate(items):
        po,co=quorum_of(p,t,101000+10*j)
        if co<3: uncounted+=1; continue
        pt,ct=quorum_of(p,nt,102000+10*j)
        if ct>=3 and pt==po: stable+=1
        else: flips+=1
        if (idx+1)%25==0: print(f"  [{name} {idx+1}/{len(items)}]",flush=True)
    n=flips+stable
    rate=flips/max(n,1)
    results[name]={"flips":flips,"stable":stable,"uncounted":uncounted,"rate":rate}
    print(f"[{name}] flip {flips}/{n} = {rate:.1%}  (uncounted {uncounted})",flush=True)
B=results["g22"]["rate"]
print(f"\n=== RE-BASED BASELINE B = {B:.1%} (bars: PRIMARY <= {0.75*B:.1%}, STRONG <= {0.5*B:.1%}) ===")
for a in ("vlow","vfull"):
    r=results[a]["rate"]
    v="STRONG" if r<=0.5*B else "PRIMARY" if r<=0.75*B else "no bar met"
    print(f"[AUGMENTATION: {a} vs g22] {r:.1%} vs {B:.1%} -> {v}")
d=results["vfull"]["rate"]-results["vlow"]["rate"]
print(f"[SINK: vfull vs vlow] {results['vfull']['rate']:.1%} vs {results['vlow']['rate']:.1%} (delta {d:+.1%})")
json.dump(results,open('.cache/flip_corrected.json','w'),indent=1)
print("[saved] .cache/flip_corrected.json")
