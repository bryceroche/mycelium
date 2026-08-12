"""reconcile.py — the reconciliation read: ONE fixture row, both
pipelines' quantities slot by slot, both fetch orders (lazy-realize
contamination is a named suspect)."""
import os, sys, json
os.environ["ALG_BREATH"]="3"
for k,v in [("ALG2","1"),("ALG_FTYPES","8"),("ALG_HW","512"),("ALG_DUP","1"),("ALG_WIDE","1"),("ALG_INV","1")]: os.environ.setdefault(k,v)
sys.path.insert(0,'.'); sys.path.insert(0,'scripts')
import numpy as np
from phase1_algebra_head import T_ALG, build_params, forward, L_FAC, build_slot_masks, sent_indices, TOKENIZER_JSON
from beacon_closing_arm import recompute_states
from tokenizers import Tokenizer
from tinygrad import Tensor, dtypes
from tinygrad.nn.state import safe_load
tok=Tokenizer.from_file(TOKENIZER_JSON)
p=build_params(0); sd=safe_load(".cache/g51_whisper.safetensors")
for k in p: p[k].assign(sd[k].to(p[k].device).cast(p[k].dtype)).realize()
def ap(a,d,j):
    if d[j]>0:
        a0=int(np.argmax(a[j])); return (a0,a0)
    return tuple(sorted(np.argsort(-a[j])[:2].tolist()))
L="abcdefghij"
rng=np.random.RandomState(96004)
# row 0 of the nd4 mint (deterministic)
op="add" if rng.rand()<0.5 else "mul"
x=int(rng.randint(2,60)) if op=="add" else int(rng.randint(2,13))
g=x+x if op=="add" else x*x
gv=[int(rng.randint(2,90)) for _ in range(4)]
dv=4; res=5
w="{a} plus another {a} makes {c}." if op=="add" else "{a} lots of {a} make {c}."
sents=[f"{L[i]} is {gv[i]}." for i in range(4)]+[f"{L[dv]} is {x}.", w.format(a=L[dv],c=L[res])]
text=f"Consider the numbers {', '.join(L[:res+1])}. "+" ".join(sents)+f" What is {L[res]}?"
print(f"[row] op={op} text[:80]={text[:80]}",flush=True)
ids=np.zeros((8,T_ALG),np.int32); msk=np.zeros((8,T_ALG),np.float32); snt=np.zeros((8,T_ALG),np.int32)
e=tok.encode(text); Ln=min(len(e.ids),T_ALG)
ids[0,:Ln]=e.ids[:Ln]; msk[0,:Ln]=1.0
snt[0]=sent_indices(text,list(e.offsets),msk[0])
tr=Tensor(recompute_states(ids).astype(np.float32),dtype=dtypes.float)
tk=Tensor(msk,dtype=dtypes.float); se=Tensor(snt,dtype=dtypes.int)
for order in ("EARLY","LATE"):
    o0=forward(p,tr,tk,se)
    if order=="EARLY":
        sa=o0["args"].realize().numpy().copy(); sdp=o0["dup"].realize().numpy().copy(); sp=o0["pres"].realize().numpy().copy()
    o0n={"fat":o0["fat"].realize().numpy(),"args":o0["args"].realize().numpy(),"res":o0["res"].realize().numpy()}
    mk=build_slot_masks(o0n, snt)
    oe=forward(p,tr,tk,se,slot_mask=Tensor(mk,dtype=dtypes.float))
    fa=oe["args"].realize().numpy(); fd=oe["dup"].realize().numpy(); fp=oe["pres"].realize().numpy()
    if order=="LATE":
        sa=o0["args"].realize().numpy(); sdp=o0["dup"].realize().numpy(); sp=o0["pres"].realize().numpy()
    print(f"--- fetch order {order} ---",flush=True)
    for j in range(L_FAC):
        if sp[0,j]>0 or fp[0,j]>0:
            print(f"  j={j:2d} sil(pres{sp[0,j]:+6.1f} dup{sdp[0,j]:+5.1f} ap{ap(sa[0],sdp[0],j)})  fin(pres{fp[0,j]:+6.1f} dup{fd[0,j]:+5.1f} ap{ap(fa[0],fd[0],j)})",flush=True)
