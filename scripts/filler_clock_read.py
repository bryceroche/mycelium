"""filler_clock_read.py — #179's CONSTITUENCY READ (2026-08-08): does
the completion gate move SPATIAL prematurity? Mis-anchor rate (gold
inspan<0.5) on bigtest filler rows (gap>=3), cure vs control. Unmoved
= the second clock's constituency is MEASURED; moved = one clock
serves both grains."""
import os, sys, json
for k,v in [("ALG2","1"),("ALG_FTYPES","8"),("ALG_HW","512"),("ALG_DUP","1"),("ALG_WIDE","1")]: os.environ.setdefault(k,v)
os.environ["ALG_RINGS"]="1"; os.environ["ALG_BREATH"]="3"; os.environ["ALG_BEXIT"]="1"
os.environ.setdefault("ALG_TEST",".cache/algebra_nl_bigtest.jsonl"); os.environ.setdefault("ALG_TEST_NAME","bigtest")
sys.path.insert(0,'.'); sys.path.insert(0,'scripts')
import numpy as np
from phase1_algebra_head import T_ALG, build_params, forward, load_alg, L_FAC, build_slot_masks, tails_of
from tinygrad import Tensor, dtypes
from tinygrad.nn.state import safe_load
samples, states, tokmask, gold, sent = load_alg("test")
def ns(t):
    c,i=1,t.find(". ")
    while i!=-1: c+=1; i=t.find(". ",i+1)
    return c
gaps=np.array([ns(s["text"])-len(s["factors"]) for s in samples])
idx=np.where(gaps>=3)[0]
print(f"filler rows: {len(idx)}")
def run(name, ck, clock):
    os.environ["ALG_CLOCK"]="1" if clock else "0"
    if clock: os.environ["ALG_CLOCK_FLOOR"]="0.3"
    p=build_params(0); sd=safe_load(ck)
    for k in p: p[k].assign(sd[k].to(p[k].device).cast(p[k].dtype)).realize()
    mis=0; tot=0
    for s0 in range(0,len(idx),8):
        sl=idx[s0:s0+8]; pad=8-len(sl)
        slp=np.concatenate([sl,sl[:1].repeat(pad)]) if pad else sl
        tr=Tensor(states[slp].astype(np.float32),dtype=dtypes.float)
        tk=Tensor(tokmask[slp].astype(np.float32),dtype=dtypes.float)
        se=Tensor(sent[slp].astype(np.int32),dtype=dtypes.int)
        tl=Tensor(tails_of(sent[slp]),dtype=dtypes.float) if clock else None
        o0=forward(p,tr,tk,se,tail=tl)
        mk=Tensor(build_slot_masks({k:o0[k].realize().numpy() for k in ("fat","args","res")},sent[slp]),dtype=dtypes.float)
        o=forward(p,tr,tk,se,slot_mask=mk,tail=tl)
        fat=o["fat"].realize().numpy(); pres=o["pres"].realize().numpy()
        for bi,ri in enumerate(sl):
            for j in range(L_FAC):
                if pres[bi,j]<=0: continue
                fs=gold["fspan"][ri,j]
                if fs.sum()<=0: continue
                a=fat[bi,j]; a=a/max(a.sum(),1e-9)
                tot+=1
                if float(a[fs>0].sum())<0.5: mis+=1
        if s0%160==0: print(f"  [{name} {s0}/{len(idx)}]",flush=True)
    print(f"[{name}] mis-anchored slots {mis}/{tot} ({mis/max(tot,1):.3f})",flush=True)
    return mis,tot
run("c2ctl",".cache/g28_p2_c2ctl.safetensors",False)
run("cure",".cache/g28_p2_cure.safetensors",True)
