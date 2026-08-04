"""wild_signal_pricing.py — THE WILD PRICING (2026-08-04; pins in
ledger). Three reads, per-row banked: coincidence decomposition
(bigtest 200 re-run), modal-agreement + min-mass on the wild 124."""
import os, sys, json
os.environ.setdefault("ALG2","1"); os.environ.setdefault("ALG_FTYPES","8")
os.environ.setdefault("ALG_HW","512"); os.environ.setdefault("ALG_DUP","1")
os.environ.setdefault("ALG_WIDE","1")
sys.path.insert(0,'.'); sys.path.insert(0,'scripts')
import numpy as np
from collections import Counter
from phase1_algebra_head import T_ALG, build_params, forward, decode, sent_indices, TOKENIZER_JSON
from beacon_closing_arm import recompute_states
from tta_views import permuted_view
from tta_alg2_dials import solve2
from mycelium.canonicalizer import canonical_digest
from tokenizers import Tokenizer
from tinygrad import Tensor, dtypes
from tinygrad.nn.state import safe_load
tok=Tokenizer.from_file(TOKENIZER_JSON)

def load(ck):
    p=build_params(0); sd=safe_load(ck)
    for k in p: p[k].assign(sd[k].to(p[k].device).cast(p[k].dtype)).realize()
    return p

def parse_batch(p, texts, want_mass=False):
    n=len(texts); N=((n+7)//8)*8
    ids=np.zeros((N,T_ALG),np.int32); msk=np.zeros((N,T_ALG),np.float32); snt=np.zeros((N,T_ALG),np.int32)
    for i,t in enumerate(texts):
        e=tok.encode(t); Ln=min(len(e.ids),T_ALG)
        ids[i,:Ln]=e.ids[:Ln]; msk[i,:Ln]=1.0
        snt[i]=sent_indices(t,list(e.offsets),msk[i])
    st=recompute_states(ids)
    out=[]
    for s0 in range(0,N,8):
        kw={}
        if want_mass:
            o0=forward(p,Tensor(st[s0:s0+8].astype(np.float32),dtype=dtypes.float),
                       Tensor(msk[s0:s0+8],dtype=dtypes.float),Tensor(snt[s0:s0+8],dtype=dtypes.int))
            from phase1_algebra_head import build_slot_masks
            onp0={k:o0[k].realize().numpy() for k in ("fat","args","res")}
            mk=build_slot_masks(onp0,snt[s0:s0+8])
            kw["slot_mask"]=Tensor(mk,dtype=dtypes.float)
        o=forward(p,Tensor(st[s0:s0+8].astype(np.float32),dtype=dtypes.float),
                  Tensor(msk[s0:s0+8],dtype=dtypes.float),Tensor(snt[s0:s0+8],dtype=dtypes.int),**kw)
        keys=["pres","ftype","op","islit","dig","sgn","args","res","query"]
        if "sel" in o: keys.append("sel")
        if "dup" in o: keys.append("dup")
        if want_mass: keys.append("cmt_m")
        onp={k:o[k].realize().numpy() for k in keys}
        for bi in range(8):
            if s0+bi<n:
                d=decode({k:onp[k][bi] for k in onp if k!="cmt_m"})
                mass=None
                if want_mass:
                    pm=[float(onp["cmt_m"][bi,j]) for j in range(24) if float(onp["pres"][bi,j])>0]
                    mass=min(pm) if pm else None
                out.append((d,mass))
    return out

from scipy.stats import mannwhitneyu
def auc(x,y):
    x=np.array(x); y=np.array(y,bool)
    a,b=x[y],x[~y]
    if not len(a) or not len(b): return float("nan"),1.0
    u,pv=mannwhitneyu(a,b,alternative="greater")
    return u/(len(a)*len(b)),pv

# ---- 1. decomposition re-run (per-row banked) ----
gate=load('.cache/g23.safetensors')
rows=[json.loads(l) for l in open('.cache/algebra_nl_bigtest.jsonl')][:200]
perrow=[]
for j,r in enumerate(rows):
    vt=[r["text"]]+[permuted_view(r["text"],103950+10*j+k) for k in range(1,5)]
    parsed=parse_batch(gate,vt)
    digs=[]
    for (f,q),_ in parsed:
        try: digs.append(canonical_digest(f,q,n_vars=24))
        except Exception: digs.append("ERR")
    modal=Counter(digs).most_common(1)[0][1]
    ans=[solve2(f,q,{"n_vars":24,"m":300}) for (f,q),_ in parsed]
    nn=[a for a in ans if a is not None]; ac=Counter(nn).most_common(1)
    ok=bool(ac) and ac[0][0]==r["solution"][r["query_var"]] and ac[0][1]>=3
    perrow.append({"modal":modal,"agree":ac[0][1] if ac else 0,"correct":bool(ok)})
coinc=[r for r in perrow if r["agree"]>r["modal"]]
cc=sum(1 for r in coinc if r["correct"])
print(f"[decomp] coincidence rows {len(coinc)}: correct {cc}, wrong {len(coinc)-cc}",flush=True)

# ---- 2+3. the wild 124 ----
recs=[json.loads(l) for l in open('.cache/wild_ledger_v1.jsonl')]
h=[json.loads(l) for l in open('.cache/math_harvest_v0.jsonl')]
wild=[(h[r["harvest_idx"]]["problem"], bool(r["correct"])) for r in recs if r["tier"]=="answered"]
modal_w=[]; lab=[]
for j,(t,ok) in enumerate(wild):
    vt=[t]+[permuted_view(t,103960+10*j+k) for k in range(1,5)]
    parsed=parse_batch(gate,vt)
    digs=[]
    for (f,q),_ in parsed:
        try: digs.append(canonical_digest(f,q,n_vars=24))
        except Exception: digs.append("ERR")
    modal_w.append(Counter(digs).most_common(1)[0][1]); lab.append(ok)
a6,p6=auc(modal_w,lab)
print(f"[wild:modal] AUC={a6:.3f} p={p6:.3g} (n={len(lab)}, {sum(lab)}/{len(lab)-sum(lab)})",flush=True)
os.environ["ALG_BREATH"]="3"; os.environ["ALG_RINGS"]="1"
rp=load('.cache/g24_rings_rings.safetensors')
mass_w=[]
for j,(t,ok) in enumerate(wild):
    d=parse_batch(rp,[t],want_mass=True)[0]
    mass_w.append(d[1] if d[1] is not None else 0.0)
mw=np.array(mass_w)
print(f"[wild:mass] APPLICABILITY: wild min-mass mean {mw.mean():.4f} p50 {np.median(mw):.4f} "
      f"(in-register correct ~1.00, wrong ~0.96)",flush=True)
am,pm_=auc(mass_w,lab)
print(f"[wild:mass] AUC={am:.3f} p={pm_:.3g}",flush=True)
json.dump({"decomp":{"coinc":len(coinc),"correct":cc},
           "perrow_bigtest":perrow,
           "wild":{"modal":modal_w,"mass":mass_w,"correct":lab},
           "auc_modal":float(a6),"auc_mass":float(am)},
          open(".cache/wild_signal_pricing.json","w"),indent=0)
print("[saved] .cache/wild_signal_pricing.json",flush=True)
