import os, sys, json
for k,v in [("ALG2","1"),("ALG_FTYPES","8"),("ALG_HW","512"),("ALG_DUP","1"),("ALG_WIDE","1")]: os.environ.setdefault(k,v)
os.environ["ALG_BREATH"]="3"; os.environ["ALG_RINGS"]="1"; os.environ["ALG_XOUT"]="1"; os.environ["ALG_XARM"]="dump"
os.environ.setdefault("ALG_TEST",".cache/algebra_nl_bigtest.jsonl"); os.environ.setdefault("ALG_TEST_NAME","bigtest")
sys.path.insert(0,"."); sys.path.insert(0,"scripts")
import numpy as np
from phase1_algebra_head import build_params, forward, load_alg, L_FAC, build_slot_masks
from tinygrad import Tensor, dtypes
from tinygrad.nn.state import safe_load
samples, states, tokmask, gold, sent = load_alg("test")
n=states.shape[0]
def ns(t):
    c,i=1,t.find(". ")
    while i!=-1: c+=1; i=t.find(". ",i+1)
    return c
gaps=np.array([ns(s["text"])-len(s["factors"]) for s in samples])
p=build_params(0); sd=safe_load(".cache/g24_rings_rings.safetensors")
for k in p: p[k].assign(sd[k].to(p[k].device).cast(p[k].dtype)).realize()
cnt={"same_wrong":0,"new_right":0,"new_wrong":0}; kept=0
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
    fat=o1["fat"].realize().numpy()
    gft=gold["ftype"][slp]; grs=gold["res"][slp]; prs=gold["presence"][slp]
    ok1=(ft1==gft)&(rs1==grs)
    rv=(prs*(1.0-ok1.astype(np.float32))).astype(np.float32)
    rvT=Tensor(rv,dtype=dtypes.float)
    oA=forward(p,tr,tk,Tensor(np.maximum(sent[slp]-1,0).astype(np.int32),dtype=dtypes.int),slot_mask=mk,revoke=rvT)
    oB=forward(p,tr,tk,Tensor(np.maximum(sent[slp]-2,0).astype(np.int32),dtype=dtypes.int),slot_mask=mk,revoke=rvT)
    fA={k:oA[k].realize().numpy().argmax(-1) for k in ("ftype","res")}
    fB={k:oB[k].realize().numpy().argmax(-1) for k in ("ftype","res")}
    for bi,ri in enumerate(sl):
        if gaps[ri]<3: continue
        prev_d=None
        for j in range(L_FAC):
            if rv[bi,j]<=0: continue
            fs=gold["fspan"][ri,j]
            d=None
            if fs.sum()>0:
                gs=int(np.round(sent[ri][fs>0].mean()))
                d=int(sent[ri][fat[bi,j].argmax()])-gs
            if prev_d==-1: ft2,rs2=fA["ftype"][bi,j],fA["res"][bi,j]
            elif prev_d==-2: ft2,rs2=fB["ftype"][bi,j],fB["res"][bi,j]
            else: kept+=1; prev_d=d; continue
            if ft2==ft1[bi,j] and rs2==rs1[bi,j]: cnt["same_wrong"]+=1
            elif ft2==gft[bi,j] and rs2==grs[bi,j]: cnt["new_right"]+=1
            else: cnt["new_wrong"]+=1
            prev_d=d
tot=sum(cnt.values()) or 1
print(f"[blade] steered {tot} kept(no-prev/-other-delta) {kept} | same-wrong {cnt['same_wrong']/tot:.3f} new-right {cnt['new_right']/tot:.3f} new-wrong {cnt['new_wrong']/tot:.3f} ratio {cnt['new_right']/max(cnt['new_wrong'],1):.2f}")
json.dump(cnt,open(".cache/xout_blade_oracle.json","w"))
