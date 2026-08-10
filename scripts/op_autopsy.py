"""op_autopsy.py — THE OP-GRAIN AUTOPSY (2026-08-10, the word given).
The nd=0 fixture's failing four: rel slot forms with args [0,0]; the
scan's misbind is op-mismatch. This read prints, per row: gold op,
decoded op, and the op head's raw logit at the found slot — margins
decide the fork (diet line / op-head cure / scope conversation)."""
import os, sys, json
for k,v in [("ALG2","1"),("ALG_FTYPES","8"),("ALG_HW","512"),("ALG_DUP","1"),("ALG_WIDE","1"),("ALG_INV","1")]: os.environ.setdefault(k,v)
sys.path.insert(0,'.'); sys.path.insert(0,'scripts')
import numpy as np
from phase1_algebra_head import T_ALG, build_params, forward, decode, sent_indices, TOKENIZER_JSON
from beacon_closing_arm import recompute_states
from tokenizers import Tokenizer
from tinygrad import Tensor, dtypes
from tinygrad.nn.state import safe_load
L="abcdefghij"
tok=Tokenizer.from_file(TOKENIZER_JSON)
def fixture_mint(nd, n=15, seed=96000):
    rng=np.random.RandomState(seed+nd); rows=[]
    while len(rows)<n:
        op="add" if rng.rand()<0.5 else "mul"
        x=int(rng.randint(2,60)) if op=="add" else int(rng.randint(2,13))
        gold=x+x if op=="add" else x*x
        if gold>300: continue
        gv=[int(rng.randint(2,90)) for _ in range(nd)]
        dv=nd; res=nd+1
        w="{a} plus another {a} makes {c}." if op=="add" else "{a} lots of {a} make {c}."
        sents=[f"{L[i]} is {gv[i]}." for i in range(nd)]+[f"{L[dv]} is {x}.", w.format(a=L[dv],c=L[res])]
        rows.append({"text":f"Consider the numbers {', '.join(L[:res+1])}. "+" ".join(sents)+f" What is {L[res]}?","op":op,"x":x})
    return rows
ND=int(os.environ.get("ND","0"))
ROWS=fixture_mint(ND)
def read_model(name, ck):
    p=build_params(0); sd=safe_load(ck)
    for k in p: p[k].assign(sd[k].to(p[k].device).cast(p[k].dtype)).realize()
    print(f"--- {name} ---",flush=True)
    bad=[]
    for i,r in enumerate(ROWS):
        t=r["text"]
        ids=np.zeros((8,T_ALG),np.int32); msk=np.zeros((8,T_ALG),np.float32); snt=np.zeros((8,T_ALG),np.int32)
        e=tok.encode(t); Ln=min(len(e.ids),T_ALG)
        ids[0,:Ln]=e.ids[:Ln]; msk[0,:Ln]=1.0
        snt[0]=sent_indices(t,list(e.offsets),msk[0])
        o=forward(p,Tensor(recompute_states(ids).astype(np.float32),dtype=dtypes.float),
                  Tensor(msk,dtype=dtypes.float),Tensor(snt,dtype=dtypes.int))
        keys=("pres","ftype","op","islit","dig","args","res","query","sel","dup")
        onp={k:o[k].realize().numpy() for k in keys if k in o}
        facs=decode({k:onp[k][0] for k in onp})[0]
        rel=[f for f in facs if f.get("ftype")=="rel" and f.get("args")==[ND,ND]]
        if not rel:
            print(f"[row{i:2d}] gold={r['op']:3s} NO-REL-SLOT",flush=True); bad.append(i); continue
        f0=rel[0]; j=None
        # find the slot index for the op logit: match by decode order — re-derive: pres-on rel slots
        js=[jj for jj in range(24) if onp["pres"][0,jj]>0 and onp["ftype"][0,jj].argmax()==0]
        oplog=[f"add{onp['op'][0,jj][0]:+.2f}/mul{onp['op'][0,jj][1]:+.2f}/m{onp['op'][0,jj][1]-onp['op'][0,jj][0]:+.2f}" for jj in js]
        dec=f0.get("op"); ok=dec==r["op"]
        if not ok: bad.append(i)
        print(f"[row{i:2d}] gold={r['op']:3s} decoded={dec:3s} {'OK ' if ok else 'MISS'} op={oplog}",flush=True)
    print(f"[{name}] op-miss rows: {bad}",flush=True)
    return bad
b1=read_model("gate",".cache/g23v5.safetensors")
b2=read_model(os.environ.get("OPAUT_NAME","g39"),os.environ.get("OPAUT_CK",".cache/g39_op_refold.safetensors"))
print(f"[fork-data] gate-miss {b1}  g38-miss {b2}  ops={[r['op'] for r in ROWS]}",flush=True)
