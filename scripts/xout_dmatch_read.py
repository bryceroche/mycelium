import os, sys, json, re
for k,v in [("ALG2","1"),("ALG_FTYPES","8"),("ALG_HW","512"),("ALG_DUP","1"),("ALG_WIDE","1")]: os.environ.setdefault(k,v)
os.environ["ALG_BREATH"]="3"; os.environ["ALG_RINGS"]="1"; os.environ["ALG_XOUT"]="1"; os.environ["ALG_XARM"]="dump"
os.environ.setdefault("ALG_TEST",".cache/algebra_nl_bigtest.jsonl"); os.environ.setdefault("ALG_TEST_NAME","bigtest")
sys.path.insert(0,"."); sys.path.insert(0,"scripts")
import numpy as np
from phase1_algebra_head import build_params, forward, load_alg, L_FAC, T_ALG, build_slot_masks
from tinygrad import Tensor, dtypes
from tinygrad.nn.state import safe_load
samples, states, tokmask, gold, sent = load_alg("test")
n=states.shape[0]
def sents_of(t):
    out=[]; st=0
    i=t.find(". ")
    while i!=-1: out.append(t[st:i+1]); st=i+2; i=t.find(". ",i+1)
    out.append(t[st:]); return out
def is_fill(s):
    if re.search(r"\d",s): return False
    if re.search(r"\b[a-z]\b",s): return False
    return True
gaps=[]; fills=[]
for s in samples:
    ss=sents_of(s["text"]); gaps.append(len(ss)-len(s["factors"]))
    fills.append([1 if (i>0 and is_fill(x)) else 0 for i,x in enumerate(ss)])
gaps=np.array(gaps)
p=build_params(0); sd=safe_load(".cache/g24_rings_rings.safetensors")
for k in p: p[k].assign(sd[k].to(p[k].device).cast(p[k].dtype)).realize()
cnt={"same_wrong":0,"new_right":0,"new_wrong":0}
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
    o2a=forward(p,tr,tk,Tensor(np.maximum(sent[slp]-1,0).astype(np.int32),dtype=dtypes.int),slot_mask=mk,revoke=rvT)
    o2b=forward(p,tr,tk,Tensor(np.maximum(sent[slp]-2,0).astype(np.int32),dtype=dtypes.int),slot_mask=mk,revoke=rvT)
    fa={k:(o2a[k].realize().numpy().argmax(-1)) for k in ("ftype","res")}
    fb={k:(o2b[k].realize().numpy().argmax(-1)) for k in ("ftype","res")}
    for bi,ri in enumerate(sl):
        if gaps[ri]<3: continue
        fl=fills[ri]
        for j in range(L_FAC):
            if rv[bi,j]<=0: continue
            s_att=int(sent[ri][fat[bi,j].argmax()])
            kf=sum(fl[:min(s_att+1,len(fl))])
            if kf==0: cnt["same_wrong"]+=1; continue   # keep pass1 = unchanged wrong
            ft2,rs2=(fa["ftype"][bi,j],fa["res"][bi,j]) if kf==1 else (fb["ftype"][bi,j],fb["res"][bi,j])
            if ft2==ft1[bi,j] and rs2==rs1[bi,j]: cnt["same_wrong"]+=1
            elif ft2==gft[bi,j] and rs2==grs[bi,j]: cnt["new_right"]+=1
            else: cnt["new_wrong"]+=1
tot=sum(cnt.values()) or 1
print(f"[dM][filler] n={tot} same-wrong {cnt['same_wrong']/tot:.3f} new-right {cnt['new_right']/tot:.3f} new-wrong {cnt['new_wrong']/tot:.3f} ratio {cnt['new_right']/max(cnt['new_wrong'],1):.2f}")
json.dump(cnt,open(".cache/xout_dmatch.json","w"))
