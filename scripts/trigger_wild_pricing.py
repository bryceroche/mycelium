"""trigger_wild_pricing.py — the licensed trigger at the frontier
(2026-08-04; pins in ledger: distribution first, no bar, survey n)."""
import os, sys, json, re
os.environ["ALG_BREATH"]="3"; os.environ["ALG_RINGS"]="1"
os.environ.setdefault("ALG2","1"); os.environ.setdefault("ALG_FTYPES","8")
os.environ.setdefault("ALG_HW","512"); os.environ.setdefault("ALG_DUP","1")
os.environ.setdefault("ALG_WIDE","1")
sys.path.insert(0,'.'); sys.path.insert(0,'scripts')
import numpy as np
from tinygrad import Tensor, dtypes
from tinygrad.nn.state import safe_load
from phase1_algebra_head import (build_params, forward, decode, build_slot_masks,
                                 sent_indices, TOKENIZER_JSON, T_ALG, L_FAC)
from tokenizers import Tokenizer
tok=Tokenizer.from_file(TOKENIZER_JSON)
recs=[json.loads(l) for l in open('.cache/wild_ledger_v1.jsonl')]
h=[json.loads(l) for l in open('.cache/math_harvest_v0.jsonl')]
wild=[(h[r["harvest_idx"]]["problem"],bool(r["correct"])) for r in recs if r["tier"]=="answered"]
p=build_params(0); sd=safe_load('.cache/g24_rings_rings.safetensors')
for k in p: p[k].assign(sd[k].to(p[k].device).cast(p[k].dtype)).realize()
THR=0.3648
res=[]
for t,ok in wild:
    e=tok.encode(t); Ln=min(len(e.ids),T_ALG)
    ids=np.zeros((8,T_ALG),np.int32); msk=np.zeros((8,T_ALG),np.float32); snt=np.zeros((8,T_ALG),np.int32)
    ids[0,:Ln]=e.ids[:Ln]; msk[0,:Ln]=1.0
    snt[0]=sent_indices(t,list(e.offsets),msk[0])
    st_=__import__('beacon_closing_arm').recompute_states(ids)
    o0=forward(p,Tensor(st_.astype(np.float32),dtype=dtypes.float),Tensor(msk,dtype=dtypes.float),Tensor(snt,dtype=dtypes.int))
    onp0={k:o0[k].realize().numpy() for k in ("fat","args","res")}
    mk=build_slot_masks(onp0,snt)
    o=forward(p,Tensor(st_.astype(np.float32),dtype=dtypes.float),Tensor(msk,dtype=dtypes.float),Tensor(snt,dtype=dtypes.int),slot_mask=Tensor(mk,dtype=dtypes.float))
    keys=["pres","ftype","op","islit","dig","sgn","args","res","query","fat"]
    if "sel" in o: keys.append("sel")
    if "dup" in o: keys.append("dup")
    onp={k:o[k].realize().numpy() for k in keys}
    facs,q=decode({k:onp[k][0] for k in onp if k!="fat"})
    offs=list(e.offsets)
    worst=1.0; nlit=0
    for f in facs:
        if f.get("ftype")!="given": continue
        j=None
        for jj in range(L_FAC):
            if onp["pres"][0,jj]>0 and onp["res"][0,jj].argmax()==f["var"] and onp["ftype"][0,jj].argmax()==1:
                j=jj; break
        if j is None: continue
        a=onp["fat"][0,j]; a=a/max(a.sum(),1e-9)
        vs=str(abs(int(f["value"])))
        hits=[mm.span() for mm in re.finditer(re.escape(vs),t)]
        mask=np.zeros(T_ALG,bool)
        for ti,(cs,ce) in enumerate(offs[:T_ALG]):
            if ce<=cs: continue
            for (hs,he) in hits:
                if cs<he and ce>hs: mask[ti]=True
        if not mask.any(): continue
        rmask=np.zeros(T_ALG,bool)
        for mm in re.finditer(r"\d+",t):
            if mm.group()==vs: continue
            hs,he=mm.span()
            for ti,(cs,ce) in enumerate(offs[:T_ALG]):
                if ce>cs and cs<he and ce>hs: rmask[ti]=True
        rmask&=~mask
        if not rmask.any(): continue
        own=float(a[mask].sum()); rival=float(a[rmask].sum())
        prox=own/(own+rival+1e-9); nlit+=1
        worst=min(worst,prox)
    res.append({"correct":ok,"worst_prox":worst if nlit else None,"nlit":nlit})
r=[x for x in res if x["worst_prox"] is not None]
pw=np.array([x["worst_prox"] for x in r]); y=np.array([x["correct"] for x in r])
print(f"[wild-trigger] readable rows n={len(r)} ({int(y.sum())} correct / {int((~y).sum())} wrong); no-literal/no-rival rows {len(res)-len(r)}")
print(f"[wild-trigger] DISTRIBUTION: correct p50 {np.median(pw[y]):.3f}  wrong p50 {np.median(pw[~y]):.3f} (in-register correct p50 was ~high)")
flag=pw<THR
print(f"[wild-trigger] flag@{THR}: FP on correct {flag[y].mean():.1%}  catch on wrong {flag[~y].mean():.1%}")
from scipy.stats import mannwhitneyu
u,pv=mannwhitneyu(pw[y],pw[~y],alternative="greater")
print(f"[wild-trigger] AUC(worst-proxy)={u/(y.sum()*(~y).sum()):.3f} p={pv:.3g}")
json.dump({"rows":res,"fp":float(flag[y].mean()),"catch":float(flag[~y].mean())},
          open(".cache/trigger_wild_pricing.json","w"),indent=0)
print("[saved] .cache/trigger_wild_pricing.json")
