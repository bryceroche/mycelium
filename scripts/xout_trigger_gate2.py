"""Trigger gate re-fire (2026-08-05) — the LICENSED proxy read from
trigger_license_v2.py through the authority door: own-value vs
rival-number attention mass, flag prox<0.5. Bars stand as pinned."""
import os, sys, json, re
for k,v in [("ALG2","1"),("ALG_FTYPES","8"),("ALG_HW","512"),("ALG_DUP","1"),("ALG_WIDE","1")]: os.environ.setdefault(k,v)
os.environ["ALG_BREATH"]="3"; os.environ["ALG_RINGS"]="1"; os.environ["ALG_XOUT"]="1"; os.environ["ALG_XARM"]="dump"
os.environ.setdefault("ALG_TEST",".cache/algebra_nl_bigtest.jsonl"); os.environ.setdefault("ALG_TEST_NAME","bigtest")
sys.path.insert(0,"."); sys.path.insert(0,"scripts")
import numpy as np
from phase1_algebra_head import build_params, forward, load_alg, L_FAC, T_ALG, N_DIG, build_slot_masks, TOKENIZER_JSON
from tokenizers import Tokenizer
from tinygrad import Tensor, dtypes
from tinygrad.nn.state import safe_load
tok=Tokenizer.from_file(TOKENIZER_JSON)
samples, states, tokmask, gold, sent = load_alg("test")
n=states.shape[0]
def ns(t):
    c,i=1,t.find(". ")
    while i!=-1: c+=1; i=t.find(". ",i+1)
    return c
gaps=np.array([ns(s["text"])-len(s["factors"]) for s in samples])
OFFS=[tok.encode(s["text"]).offsets for s in samples]
p=build_params(0); sd=safe_load(".cache/g24_rings_rings.safetensors")
for k in p: p[k].assign(sd[k].to(p[k].device).cast(p[k].dtype)).realize()
POW=10**np.arange(N_DIG-1,-1,-1)
res={"flag":{"same_wrong":0,"new_right":0,"new_wrong":0},"unflag":0,"nonlit":0,"noriv":0}
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
    fat=o1["fat"].realize().numpy(); dg=o1["dig"].realize().numpy().argmax(-1)
    il=o1["islit"].realize().numpy()
    gft=gold["ftype"][slp]; grs=gold["res"][slp]; prs=gold["presence"][slp]
    ok1=(ft1==gft)&(rs1==grs)
    rv=(prs*(1.0-ok1.astype(np.float32))).astype(np.float32)
    se2=Tensor(np.maximum(sent[slp]-1,0).astype(np.int32),dtype=dtypes.int)
    o2=forward(p,tr,tk,se2,slot_mask=mk,revoke=Tensor(rv,dtype=dtypes.float))
    ft2=o2["ftype"].realize().numpy().argmax(-1); rs2=o2["res"].realize().numpy().argmax(-1)
    ok2=(ft2==gft)&(rs2==grs)
    for bi,ri in enumerate(sl):
        if gaps[ri]<3: continue
        txt=samples[ri]["text"]; offs=OFFS[ri]
        for j in range(L_FAC):
            if rv[bi,j]<=0: continue
            if il[bi,j]<=0: res["nonlit"]+=1; continue
            vs=str(int(dg[bi,j] @ POW))
            a=fat[bi,j]; a=a/max(a.sum(),1e-9)
            hits=[m.span() for m in re.finditer(re.escape(vs),txt)]
            mask=np.zeros(T_ALG,bool)
            for ti,(cs,ce) in enumerate(offs[:T_ALG]):
                if ce<=cs: continue
                for hs,he in hits:
                    if cs<he and ce>hs: mask[ti]=True
            own=float(a[mask].sum()) if mask.any() else 0.0
            rmask=np.zeros(T_ALG,bool)
            for m in re.finditer(r"\d+",txt):
                if m.group()==vs: continue
                hs,he=m.span()
                for ti,(cs,ce) in enumerate(offs[:T_ALG]):
                    if ce>cs and cs<he and ce>hs: rmask[ti]=True
            rmask&=~mask
            if not rmask.any(): res["noriv"]+=1; continue
            prox=own/(own+float(a[rmask].sum())+1e-9)
            if prox>=0.5: res["unflag"]+=1; continue
            if ft2[bi,j]==ft1[bi,j] and rs2[bi,j]==rs1[bi,j]: res["flag"]["same_wrong"]+=1
            elif ok2[bi,j]: res["flag"]["new_right"]+=1
            else: res["flag"]["new_wrong"]+=1
f=res["flag"]; nf=sum(f.values()) or 1
print(f"[T2][coverage] flagged {nf} unflag {res['unflag']} nonlit {res['nonlit']} no-rival {res['noriv']}")
print(f"[T2][flagged] same-wrong {f['same_wrong']/nf:.3f} new-right {f['new_right']/nf:.3f} new-wrong {f['new_wrong']/nf:.3f}")
json.dump(res,open(".cache/xout_trigger_gate2.json","w"))
