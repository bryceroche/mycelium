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
acc={"same_wrong":0,"new_right":0,"new_wrong":0}; rej=0; tot=0
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
    m1=o1["cmt_m"].realize().numpy()
    gft=gold["ftype"][slp]; grs=gold["res"][slp]; prs=gold["presence"][slp]
    ok1=(ft1==gft)&(rs1==grs)
    rv=(prs*(1.0-ok1.astype(np.float32))).astype(np.float32)
    se2=Tensor(np.maximum(sent[slp]-1,0).astype(np.int32),dtype=dtypes.int)
    o2=forward(p,tr,tk,se2,slot_mask=mk,revoke=Tensor(rv,dtype=dtypes.float))
    ft2=o2["ftype"].realize().numpy().argmax(-1); rs2=o2["res"].realize().numpy().argmax(-1)
    m2=o2["cmt_m"].realize().numpy(); ok2=(ft2==gft)&(rs2==grs)
    for bi,ri in enumerate(sl):
        if gaps[ri]<3: continue
        for j in range(L_FAC):
            if rv[bi,j]<=0: continue
            tot+=1
            if m2[bi,j]<=m1[bi,j]: rej+=1; continue
            if ft2[bi,j]==ft1[bi,j] and rs2[bi,j]==rs1[bi,j]: acc["same_wrong"]+=1
            elif ok2[bi,j]: acc["new_right"]+=1
            else: acc["new_wrong"]+=1
na=sum(acc.values()) or 1
print(f"[D][filler] revoked {tot} accepted {na} ({na/max(tot,1):.3f}) rejected {rej}")
print(f"[D][accepted] same-wrong {acc['same_wrong']/na:.3f} new-right {acc['new_right']/na:.3f} new-wrong {acc['new_wrong']/na:.3f}")
json.dump({"acc":acc,"rej":rej,"tot":tot},open(".cache/xout_armD.json","w"))
