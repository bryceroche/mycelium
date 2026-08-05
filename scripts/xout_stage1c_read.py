"""xout_stage1c_read.py — STAGE-1a: ARM C ONLY (2026-08-05; declared:
delta-corrective +1-sentence re-read on revoked slots, implemented as a
pass-2 GLOBAL snt shift (snt-1, clip 0) — per-slot gating comes FREE
from ring mass (committed slots are anchor-locked; only released slots
move). A/B (anchor-bias) need bank-level surgery — deferred, reason
banked. Bars per registration, applied to C; ckpt = g24 rings."""
import os, sys, json
os.environ.setdefault("ALG2","1"); os.environ.setdefault("ALG_FTYPES","8")
os.environ.setdefault("ALG_HW","512"); os.environ.setdefault("ALG_DUP","1")
os.environ.setdefault("ALG_WIDE","1")
os.environ["ALG_BREATH"]="3"; os.environ["ALG_RINGS"]="1"
os.environ["ALG_XOUT"]="1"; os.environ["ALG_XARM"]="dump"
os.environ.setdefault("ALG_TEST",".cache/algebra_nl_bigtest.jsonl")
os.environ.setdefault("ALG_TEST_NAME","bigtest")
sys.path.insert(0,"."); sys.path.insert(0,"scripts")
import numpy as np
from phase1_algebra_head import build_params, forward, load_alg, L_FAC, build_slot_masks
from tinygrad import Tensor, dtypes
from tinygrad.nn.state import safe_load
samples, states, tokmask, gold, sent = load_alg("test")
n = states.shape[0]
def nsent(t):
    c,i=1,t.find(". ")
    while i!=-1: c+=1; i=t.find(". ",i+1)
    return c
gaps=np.array([nsent(s["text"])-len(s["factors"]) for s in samples])
p=build_params(0); sd=safe_load(".cache/g24_rings_rings.safetensors")
assert set(sd.keys())==set(p.keys())
for k in p: p[k].assign(sd[k].to(p[k].device).cast(p[k].dtype)).realize()
cnt={c:{"same_wrong":0,"new_right":0,"new_wrong":0} for c in ("clean","filler")}
for s0 in range(0,n,8):
    sl=np.arange(s0,min(s0+8,n)); pad=8-len(sl)
    slp=np.concatenate([sl,sl[:1].repeat(pad)]) if pad else sl
    tr=Tensor(states[slp].astype(np.float32),dtype=dtypes.float)
    tk=Tensor(tokmask[slp].astype(np.float32),dtype=dtypes.float)
    se=Tensor(sent[slp].astype(np.int32),dtype=dtypes.int)
    o0=forward(p,tr,tk,se)
    onp0={k:o0[k].realize().numpy() for k in ("fat","args","res")}
    mk=Tensor(build_slot_masks(onp0,sent[slp]),dtype=dtypes.float)
    o1=forward(p,tr,tk,se,slot_mask=mk)
    ft1=o1["ftype"].realize().numpy().argmax(-1); rs1=o1["res"].realize().numpy().argmax(-1)
    gft=gold["ftype"][slp]; grs=gold["res"][slp]; prs=gold["presence"][slp]
    ok1=(ft1==gft)&(rs1==grs)
    rv=(prs*(1.0-ok1.astype(np.float32))).astype(np.float32)
    se_shift=Tensor(np.maximum(sent[slp]-1,0).astype(np.int32),dtype=dtypes.int)
    o2=forward(p,tr,tk,se_shift,slot_mask=mk,revoke=Tensor(rv,dtype=dtypes.float))
    ft2=o2["ftype"].realize().numpy().argmax(-1); rs2=o2["res"].realize().numpy().argmax(-1)
    ok2=(ft2==gft)&(rs2==grs)
    for bi,ri in enumerate(sl):
        cat="clean" if gaps[ri]==2 else "filler"
        for j in range(L_FAC):
            if rv[bi,j]<=0: continue
            if ft2[bi,j]==ft1[bi,j] and rs2[bi,j]==rs1[bi,j]: cnt[cat]["same_wrong"]+=1
            elif ok2[bi,j]: cnt[cat]["new_right"]+=1
            else: cnt[cat]["new_wrong"]+=1
    if s0%400==0: print(f"[C] {s0}/{n}",flush=True)
for cat in ("filler","clean"):
    c=cnt[cat]; tot=sum(c.values()) or 1
    print(f"[C][{cat}] n={tot} same-wrong {c['same_wrong']/tot:.3f} new-right {c['new_right']/tot:.3f} new-wrong {c['new_wrong']/tot:.3f}",flush=True)
json.dump(cnt,open(".cache/xout_stage1c.json","w"),indent=1)
print("[saved] .cache/xout_stage1c.json")
