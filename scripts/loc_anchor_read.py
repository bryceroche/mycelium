"""loc_anchor_read.py — ORGAN 1 (2026-08-04; pins in ledger). Anchor =
final-breath fat mass inside gold fspan, per present slot; banks top-k
anchor tokens per slot for the reverse gear."""
import os, sys, json
os.environ["ALG_BREATH"]="3"; os.environ["ALG_RINGS"]="1"
os.environ.setdefault("ALG2","1"); os.environ.setdefault("ALG_FTYPES","8")
os.environ.setdefault("ALG_HW","512"); os.environ.setdefault("ALG_DUP","1")
os.environ.setdefault("ALG_WIDE","1")
os.environ.setdefault("ALG_TEST",".cache/algebra_nl_bigtest.jsonl")
os.environ.setdefault("ALG_TEST_NAME","bigtest")
sys.path.insert(0,'.'); sys.path.insert(0,'scripts')
import numpy as np
from tinygrad import Tensor, dtypes
from tinygrad.nn.state import safe_load
from phase1_algebra_head import (build_params, forward, load_alg,
                                 build_slot_masks, L_FAC)

samples, states, tokmask, gold, sent = load_alg("test")
p=build_params(0); sd=safe_load('.cache/g24_rings_rings.safetensors')
for k in p: p[k].assign(sd[k].to(p[k].device).cast(p[k].dtype)).realize()
n=len(samples)
ok_mass=[]; bad_mass=[]; anchors=[]
for s0 in range(0,n,8):
    sl=np.arange(s0,min(s0+8,n)); pad=8-len(sl)
    sl_p=np.concatenate([sl,sl[:1].repeat(pad)]) if pad else sl
    t_tr=Tensor(states[sl_p].astype(np.float32),dtype=dtypes.float)
    t_tk=Tensor(tokmask[sl_p].astype(np.float32),dtype=dtypes.float)
    t_se=Tensor(sent[sl_p].astype(np.int32),dtype=dtypes.int)
    o0=forward(p,t_tr,t_tk,t_se)
    onp0={k:o0[k].realize().numpy() for k in ("fat","args","res")}
    mk=build_slot_masks(onp0,sent[sl_p])
    o=forward(p,t_tr,t_tk,t_se,slot_mask=Tensor(mk,dtype=dtypes.float))
    fat=o["fat"].realize().numpy()      # (B, L_FAC, T)
    ft=o["ftype"].realize().numpy().argmax(-1)
    rs=o["res"].realize().numpy().argmax(-1)
    pres=o["pres"].realize().numpy()
    for bi,i in enumerate(sl):
        i=int(i); row_anch=[]
        for j in range(L_FAC):
            if gold["presence"][i,j]<=0: continue
            span=gold["fspan"][i,j]>0
            if not span.any(): continue
            a=fat[bi,j]; a=a/max(a.sum(),1e-9)
            m=float(a[span].sum())
            correct=(ft[bi,j]==gold["ftype"][i,j]) and (rs[bi,j]==gold["res"][i,j])
            (ok_mass if correct else bad_mass).append(m)
            row_anch.append({"slot":j,"inspan":round(m,4),
                             "topk":[int(t) for t in np.argsort(-a)[:8]]})
        anchors.append(row_anch)
mo,mb=np.array(ok_mass),np.array(bad_mass)
from scipy.stats import mannwhitneyu
u,pv=mannwhitneyu(mo,mb,alternative="greater")
auc=u/(len(mo)*len(mb))
print(f"[anchor] correct slots n={len(mo)}: in-span mass mean {mo.mean():.3f} p50 {np.median(mo):.3f}")
print(f"[anchor] wrong slots  n={len(mb)}: in-span mass mean {mb.mean():.3f} p50 {np.median(mb):.3f}")
print(f"[anchor] informative? AUC={auc:.3f} p={pv:.2e} (pinned: directional, p<0.05)")
trust=float(np.median(np.concatenate([mo,mb])))
print(f"[anchor] TRUST (median in-span mass, all present slots): {trust:.3f}")
json.dump({"mean_ok":float(mo.mean()),"mean_bad":float(mb.mean()),
           "auc":float(auc),"p":float(pv),"trust_p50":trust,
           "n_ok":len(mo),"n_bad":len(mb)},
          open(".cache/loc_anchor_read.json","w"),indent=1)
with open(".cache/loc_anchors_bigtest.jsonl","w") as f:
    for r in anchors: f.write(json.dumps(r)+"\n")
print("[saved] loc_anchor_read.json + loc_anchors_bigtest.jsonl (organ-2's artifact)")
