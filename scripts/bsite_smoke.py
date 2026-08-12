"""bsite_smoke.py — the (B)-site lawfulness smoke: per-slot gate
modulation INSIDE the loop (events 1.0 / others 0.2) vs uniform
engaged. Scrambling = the site inherits the mixing poison; holding =
the site is lawful and event-focused breathing may be designed."""
import os, sys, json
os.environ["ALG_BREATH"]="3"
for k,v in [("ALG2","1"),("ALG_FTYPES","8"),("ALG_HW","512"),("ALG_DUP","1"),("ALG_WIDE","1"),("ALG_INV","1")]: os.environ.setdefault(k,v)
os.environ.setdefault("ALG_TEST",".cache/algebra_nl_bigtest.jsonl"); os.environ.setdefault("ALG_TEST_NAME","bigtest")
sys.path.insert(0,'.'); sys.path.insert(0,'scripts')
import numpy as np
from phase1_algebra_head import T_ALG, build_params, forward, decode, load_alg, L_FAC, build_slot_masks, sent_indices, TOKENIZER_JSON
from beacon_closing_arm import recompute_states
from tokenizers import Tokenizer
from tinygrad import Tensor, dtypes
from tinygrad.nn.state import safe_load
tok=Tokenizer.from_file(TOKENIZER_JSON)
p=build_params(0); sd=safe_load(os.environ.get("CK_OVERRIDE",".cache/g51_whisper.safetensors"))
for k in p: p[k].assign(sd[k].to(p[k].device).cast(p[k].dtype)).realize()
def argpair(oo,bi,j):
    if oo["dup"][bi,j]>0:
        a0=int(np.argmax(oo["args"][bi,j])); return (a0,a0)
    return tuple(sorted(np.argsort(-oo["args"][bi,j])[:2].tolist()))
L="abcdefghij"
def fixture_mint(nd, n=15, seed=96000):
    rng=np.random.RandomState(seed+nd); rows=[]
    while len(rows)<n:
        op="add" if rng.rand()<0.5 else "mul"
        x=int(rng.randint(2,60)) if op=="add" else int(rng.randint(2,13))
        gg=x+x if op=="add" else x*x
        if gg>300: continue
        gv=[int(rng.randint(2,90)) for _ in range(nd)]
        dv=nd; res=nd+1
        w="{a} plus another {a} makes {c}." if op=="add" else "{a} lots of {a} make {c}."
        sents=[f"{L[i]} is {gv[i]}." for i in range(nd)]+[f"{L[dv]} is {x}.", w.format(a=L[dv],c=L[res])]
        rows.append({"text":f"Consider the numbers {', '.join(L[:res+1])}. "+" ".join(sents)+f" What is {L[res]}?","dv":dv,"op":op})
    return rows
rows4=fixture_mint(4)
mis={"uni":0,"mod":0}
for s0 in range(0,15,8):
    ch=rows4[s0:s0+8]
    ids=np.zeros((8,T_ALG),np.int32); msk=np.zeros((8,T_ALG),np.float32); snt=np.zeros((8,T_ALG),np.int32)
    for i,r in enumerate(ch):
        e=tok.encode(r["text"]); Ln=min(len(e.ids),T_ALG)
        ids[i,:Ln]=e.ids[:Ln]; msk[i,:Ln]=1.0
        snt[i]=sent_indices(r["text"],list(e.offsets),msk[i])
    tr=Tensor(recompute_states(ids).astype(np.float32),dtype=dtypes.float)
    tk=Tensor(msk,dtype=dtypes.float); se=Tensor(snt,dtype=dtypes.int)
    o0=forward(p,tr,tk,se)
    o0n={k2:o0[k2].realize().numpy() for k2 in ("fat","args","res")}
    mk=build_slot_masks(o0n, snt)
    oe=forward(p,tr,tk,se,slot_mask=Tensor(mk,dtype=dtypes.float))
    keys=("pres","ftype","op","islit","dig","args","res","query")+(("sel",) if "sel" in oe else ())+(("dup",) if "dup" in oe else ())+(("sgn",) if "sgn" in oe else ())
    en={k2:oe[k2].realize().numpy() for k2 in keys}
    sn={k2:o0[k2].realize().numpy() for k2 in keys}
    gm=np.full((8,L_FAC,1),float(os.environ.get("BG_AUTH","0.2")),np.float32)
    for bi in range(8):
        for j in range(L_FAC):
            if en["pres"][bi,j]>0 and sn["pres"][bi,j]>0 and argpair(en,bi,j)!=argpair(sn,bi,j):
                gm[bi,j]=1.0
    om=forward(p,tr,tk,se,slot_mask=Tensor(mk,dtype=dtypes.float),gmod=Tensor(gm,dtype=dtypes.float))
    mn={k2:om[k2].realize().numpy() for k2 in keys}
    for i,r in enumerate(ch):
        for nm,oo in (("uni",en),("mod",mn)):
            facs=decode({k2:oo[k2][i] for k2 in oo})[0]
            ok=any(f.get("ftype")=="rel" and f.get("args")==[r["dv"],r["dv"]] and f.get("op")==r["op"] for f in facs)
            mis[nm]+=(not ok)
print(f"[bsite nd4] misbind: uniform-engaged {mis['uni']}/15  FOCUSED(bg={os.environ.get('BG_AUTH','0.2')}) {mis['mod']}/15",flush=True)
json.dump(mis,open('.cache/bsite_smoke.json','w'))
