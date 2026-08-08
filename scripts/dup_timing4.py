"""dup_timing_read.py — THE DISCRIMINATING READ (2026-08-07): on the
scan's one-distractor rows (15/15 misbind), prefix-forward through the
rel sentence and time the dup slot's args commitment vs the operand
token (the repeated letter's 2nd occurrence). settle BEFORE operand =
PREMATURE (staging cures); settle at/after with wrong binding = LOST
CONTEST (strength cures)."""
import os, sys, json, re
for k,v in [("ALG2","1"),("ALG_FTYPES","8"),("ALG_HW","512"),("ALG_DUP","1")]: os.environ.setdefault(k,v)
sys.path.insert(0,'.'); sys.path.insert(0,'scripts')
import numpy as np
from phase1_algebra_head import T_ALG, build_params, forward, decode, sent_indices, TOKENIZER_JSON, build_slot_masks, tails_of
from beacon_closing_arm import recompute_states
from tokenizers import Tokenizer
from tinygrad import Tensor, dtypes
from tinygrad.nn.state import safe_load
L="abcdefghij"
tok=Tokenizer.from_file(TOKENIZER_JSON)
rng=np.random.RandomState(96001); rows=[]
while len(rows)<15:
    op="add" if rng.rand()<0.5 else "mul"
    x=int(rng.randint(2,60)) if op=="add" else int(rng.randint(2,13))
    gold=x+x if op=="add" else x*x
    if gold>300: continue
    gv=[int(rng.randint(2,90))]
    dv=1; res=2
    w="{a} plus another {a} makes {c}." if op=="add" else "{a} lots of {a} make {c}."
    sents=[f"a is {gv[0]}.", f"{L[dv]} is {x}.", w.format(a=L[dv],c=L[res])]
    rows.append({"text":f"Consider the numbers a, b, c. "+" ".join(sents)+f" What is c?","dv":dv,"op":op,"rel":w.format(a=L[dv],c=L[res])})
p=build_params(0); sd=safe_load(os.environ["CK"])
for k in p: p[k].assign(sd[k].to(p[k].device).cast(p[k].dtype)).realize()
def fwd(ids,msk,snt):
    _st=Tensor(recompute_states(ids).astype(np.float32),dtype=dtypes.float)
    _tk=Tensor(msk,dtype=dtypes.float); _sn=Tensor(snt,dtype=dtypes.int)
    _tl=Tensor(tails_of(snt),dtype=dtypes.float) if int(os.environ.get("ALG_CLOCK","0")) else None
    _o0=forward(p,_st,_tk,_sn,tail=_tl)
    _mk=build_slot_masks({k:_o0[k].realize().numpy() for k in ("fat","args","res")},snt)
    out=forward(p,_st,_tk,_sn,slot_mask=Tensor(_mk,dtype=dtypes.float),tail=_tl)
    keys=("pres","ftype","args","dup")
    return {k:out[k].realize().numpy() for k in keys if k in out}
prem=0; lost=0; other=0
for r in rows:
    t=r["text"]; e=tok.encode(t); n=min(len(e.ids),T_ALG)
    offs=list(e.offsets)
    rel_start=t.find(r["rel"])
    # operand = 2nd occurrence of dup letter within the rel sentence
    li=[m.start() for m in re.finditer(r'\b'+L[r["dv"]]+r'\b', t[rel_start:])]
    op_char=rel_start+li[1]
    op_tok=next(i for i,(cs,ce) in enumerate(offs[:n]) if cs<=op_char<ce)
    rel_tok0=next(i for i,(cs,ce) in enumerate(offs[:n]) if ce>rel_start)
    ids=np.zeros((8,T_ALG),np.int32); msk=np.zeros((8,T_ALG),np.float32); snt=np.zeros((8,T_ALG),np.int32)
    ids[0,:n]=e.ids[:n]; msk[0,:n]=1.0
    snt[0]=sent_indices(t,offs,msk[0])
    full=fwd(ids,msk,snt)
    # dup slot = present rel slot with max dup logit
    js=[j for j in range(24) if full["pres"][0,j]>0 and full["ftype"][0,j].argmax()==0]
    if not js: other+=1; continue
    j=max(js,key=lambda q: full["dup"][0,q] if "dup" in full else full["args"][0,q].max())
    final=tuple(np.argsort(-full["args"][0,j])[:2])
    traj=[]
    for cut in range(rel_tok0+1, n+1):
        ids2=np.zeros((8,T_ALG),np.int32); m2=np.zeros((8,T_ALG),np.float32)
        ids2[0,:cut]=e.ids[:cut]; m2[0,:cut]=1.0
        s2=np.zeros((8,T_ALG),np.int32); s2[0]=snt[0]
        o=fwd(ids2,m2,s2)
        traj.append(tuple(np.argsort(-o["args"][0,j])[:2]))
    # settle = first cut index after which binding stays == final
    settle=None
    for i in range(len(traj)):
        if all(tt==final for tt in traj[i:]): settle=rel_tok0+1+i; break
    if settle is None: other+=1; continue
    if settle<=op_tok: prem+=1
    else: lost+=1
    print(f"  row: settle_tok {settle} operand_tok {op_tok} -> {'PREMATURE' if settle<=op_tok else 'LOST'}",flush=True)
print(f"[timing] PREMATURE {prem}  LOST {lost}  other {other} / 15")
json.dump({"premature":prem,"lost":lost,"other":other},open('.cache/dup_timing4.json','w'))
