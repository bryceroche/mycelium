"""formation_autopsy.py — THE ND=4 SLOT-FORMATION AUTOPSY (2026-08-10,
the word given; audit-before-diet: the missing slots get slot-level
autopsies before any corpus line claims the fix). The 15 fixture rows
(seed 96000+4, deterministic) through gate/g35/g36; per row: does a
rel slot with args=[4,4] form; if not, WHAT formed where it should be
— absent (presence off) / misrouted (ftype != rel; to what, at what
margin) / captured (rel exists but bound elsewhere). Margins tell
near-miss from confident misroute."""
import os, sys, json
for k,v in [("ALG2","1"),("ALG_FTYPES","8"),("ALG_HW","512"),("ALG_DUP","1"),("ALG_WIDE","1"),("ALG_INV","1")]: os.environ.setdefault(k,v)
sys.path.insert(0,'.'); sys.path.insert(0,'scripts')
import numpy as np
from phase1_algebra_head import T_ALG, build_params, forward, decode, sent_indices, TOKENIZER_JSON
from beacon_closing_arm import recompute_states
from tokenizers import Tokenizer
from tinygrad import Tensor, dtypes
from tinygrad.nn.state import safe_load
L="abcdefghij"; FT=["rel","given","mod","sel","pct","fdiv","macro","frac"]
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
        rows.append({"text":f"Consider the numbers {', '.join(L[:res+1])}. "+" ".join(sents)+f" What is {L[res]}?","op":op})
    return rows
ROWS=fixture_mint(int(os.environ.get("AUTOPSY_ND","4")))
def softmax(x):
    e=np.exp(x-x.max()); return e/e.sum()
def read_model(ck):
    p=build_params(0); sd=safe_load(ck)
    for k in p: p[k].assign(sd[k].to(p[k].device).cast(p[k].dtype)).realize()
    out=[]
    for r in ROWS:
        t=r["text"]
        ids=np.zeros((8,T_ALG),np.int32); msk=np.zeros((8,T_ALG),np.float32); snt=np.zeros((8,T_ALG),np.int32)
        e=tok.encode(t); Ln=min(len(e.ids),T_ALG)
        ids[0,:Ln]=e.ids[:Ln]; msk[0,:Ln]=1.0
        snt[0]=sent_indices(t,list(e.offsets),msk[0])
        o=forward(p,Tensor(recompute_states(ids).astype(np.float32),dtype=dtypes.float),
                  Tensor(msk,dtype=dtypes.float),Tensor(snt,dtype=dtypes.int))
        keys=("pres","ftype","op","islit","dig","args","res","query","sel","dup")
        onp={k:o[k].realize().numpy() for k in keys if k in o}
        facs=decode({k:onp[k][0] for k in onp})[0] if True else None
        # cured? a rel factor with args [4,4]
        _nd=int(os.environ.get("AUTOPSY_ND","4"))
        cured=any(f.get("ftype")=="rel" and f.get("args")==[_nd,_nd] for f in facs)
        pres=onp["pres"][0]; on=[j for j in range(24) if pres[j]>0]
        ftv=onp["ftype"][0]
        ftypes=[FT[int(ftv[j].argmax())] for j in on]
        rels=[j for j in on if FT[int(ftv[j].argmax())]=="rel"]
        # the would-be dup slot: among ALL 24, the one with max dup logit
        jd=int(np.argmax(onp["dup"][0])) if "dup" in onp else -1
        sm=softmax(ftv[jd]); order=np.argsort(-sm)
        diag={
          "cured":bool(cured), "n_on":len(on),
          "ftype_census":{ft:ftypes.count(ft) for ft in set(ftypes)},
          "rel_slots":[{"j":int(j),"args":next((f.get("args") for f in facs if f.get("j",None)==j), None)} for j in rels],
          "dupmax_slot":{"j":jd,"pres_logit":float(pres[jd]),
                         "ftype_top":FT[int(order[0])],"p_top":float(sm[order[0]]),
                         "p_rel":float(sm[0]),"rel_rank":int(np.where(order==0)[0][0])},
          "rel_args_all":[f.get("args") for f in facs if f.get("ftype")=="rel"],
        }
        out.append(diag)
    return out
RES={}
import os as _os
_cks=[("gate",".cache/g23v5.safetensors"),("g35",".cache/g35_size8x_refold.safetensors"),("g36",".cache/g36_freeze8x_refold.safetensors")]
if _os.environ.get("AUTOPSY_CKS"):
    _cks=[tuple(x.split(":",1)) for x in _os.environ["AUTOPSY_CKS"].split(",")]
for name,ck in _cks:
    RES[name]=read_model(ck)
    cured=[i for i,d in enumerate(RES[name]) if d["cured"]]
    print(f"[{name}] cured rows: {len(cured)}/15  {cured}",flush=True)
for name in [n for n,_ in _cks if n!="gate"]:
    print(f"--- {name} failures ---",flush=True)
    for i,d in enumerate(RES[name]):
        if d["cured"]: continue
        g=RES["gate"][i]; dm=d["dupmax_slot"]
        mode=("ABSENT" if dm["pres_logit"]<=0 else
              ("MISROUTED" if dm["ftype_top"]!="rel" else "MISBOUND"))
        print(f"[{name} row{i:2d}] op={ROWS[i]['op']:3s} mode={mode:9s} n_on={d['n_on']}(gate {g['n_on']}) "
              f"pres={dm['pres_logit']:+.2f} ftype_top={dm['ftype_top']}@{dm['p_top']:.2f} "
              f"p_rel={dm['p_rel']:.2f} rel_args={d['rel_args_all']}",flush=True)
for name,_ in _cks:
    print(f"[fails] {name}: {sorted(i for i,d in enumerate(RES[name]) if not d['cured'])}",flush=True)
json.dump({k:[{kk:vv for kk,vv in d.items()} for d in v] for k,v in RES.items()},
          open('.cache/formation_autopsy.json','w'), default=str)
print("[banked] .cache/formation_autopsy.json",flush=True)
