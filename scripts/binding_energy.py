"""binding_energy.py — the Nazaré queue-head's first step (the word
given): the per-slot BINDING-ENERGY field (args movement between
grains) on g51. Two locations tested: (1) fixture — does the
max-energy slot coincide with the slot the loop REBINDS? (2) the
233 — does row-level energy predict the organ's converts (AUC vs
the banked union-convert set)?"""
import os, sys, json
os.environ["ALG_BREATH"]="3"
for k,v in [("ALG2","1"),("ALG_FTYPES","8"),("ALG_HW","512"),("ALG_DUP","1"),("ALG_WIDE","1"),("ALG_INV","1")]: os.environ.setdefault(k,v)
os.environ.setdefault("ALG_TEST",".cache/algebra_nl_bigtest.jsonl"); os.environ.setdefault("ALG_TEST_NAME","bigtest")
sys.path.insert(0,'.'); sys.path.insert(0,'scripts')
import numpy as np
from phase1_algebra_head import T_ALG, build_params, forward, load_alg, L_FAC, build_slot_masks, sent_indices, TOKENIZER_JSON
from beacon_closing_arm import recompute_states
from tokenizers import Tokenizer
from tinygrad import Tensor, dtypes
from tinygrad.nn.state import safe_load
tok=Tokenizer.from_file(TOKENIZER_JSON)
p=build_params(0); sd=safe_load(".cache/g51_whisper.safetensors")
for k in p: p[k].assign(sd[k].to(p[k].device).cast(p[k].dtype)).realize()
def grains(tr,tk,se,snt_np):
    o0=forward(p,tr,tk,se)
    o0n={k2:o0[k2].realize().numpy() for k2 in ("fat","args","res")}
    mk=build_slot_masks(o0n, snt_np)
    oe=forward(p,tr,tk,se,slot_mask=Tensor(mk,dtype=dtypes.float))
    sn={k2:o0[k2].realize().numpy() for k2 in ("pres","ftype","args","dup")}
    en={k2:oe[k2].realize().numpy() for k2 in ("pres","ftype","args","dup")}
    return sn,en
def argpair(oo,bi,j):
    if oo["dup"][bi,j]>0:
        a0=int(np.argmax(oo["args"][bi,j])); return (a0,a0)
    return tuple(sorted(np.argsort(-oo["args"][bi,j])[:2].tolist()))
def energy(sn,en,bi):
    """arm EVENTS: decode-grade rebinding (dup-aware argpair change),
    presence-gated both grains; arm PGE: presence-gated distribution
    movement. Returns (events_bool[L], pge[L])."""
    ev=np.zeros(L_FAC); pg=np.zeros(L_FAC)
    for j in range(L_FAC):
        if en["pres"][bi,j]<=0: continue
        if sn["pres"][bi,j]>0 and argpair(en,bi,j)!=argpair(sn,bi,j): ev[j]=1.0
        pa=np.exp(sn["args"][bi,j]-sn["args"][bi,j].max()); pa/=pa.sum()
        pb=np.exp(en["args"][bi,j]-en["args"][bi,j].max()); pb/=pb.sum()
        pg[j]=float(np.abs(pa-pb).sum())
    return ev,pg
# (1) fixture localization
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
rows4=fixture_mint(4); hits=0; tot=0; EV_HIT=[]
for s0 in range(0,15,8):
    ch=rows4[s0:s0+8]
    ids=np.zeros((8,T_ALG),np.int32); msk=np.zeros((8,T_ALG),np.float32); snt=np.zeros((8,T_ALG),np.int32)
    for i,r in enumerate(ch):
        e=tok.encode(r["text"]); Ln=min(len(e.ids),T_ALG)
        ids[i,:Ln]=e.ids[:Ln]; msk[i,:Ln]=1.0
        snt[i]=sent_indices(r["text"],list(e.offsets),msk[i])
    tr=Tensor(recompute_states(ids).astype(np.float32),dtype=dtypes.float)
    sn,en=grains(tr,Tensor(msk,dtype=dtypes.float),Tensor(snt,dtype=dtypes.int),snt)
    for i,r in enumerate(ch):
        ev,pg=energy(sn,en,i)
        jmax=int(np.argmax(pg))
        # the rebound slot: engaged dup-routed rel pointing [dv,dv]
        jre=-1
        for j in range(L_FAC):
            if en["pres"][i,j]>0 and en["ftype"][i,j].argmax()==0 and en["dup"][i,j]>0 \
               and int(np.argmax(en["args"][i,j]))==r["dv"]:
                jre=j; break
        if jre>=0:
            tot+=1; hits+= (jmax==jre)
            EV_HIT.append((bool(ev[jre]>0), int(ev.sum())))
print(f"[PGE fixture] max presence-gated energy == rebound slot: {hits}/{tot}",flush=True)
rec=sum(1 for h,_ in EV_HIT if h); spars=[n for _,n in EV_HIT]
print(f"[EVENTS fixture] rebound slot IS an event: {rec}/{len(EV_HIT)}  events/row median {int(np.median(spars))}",flush=True)
# (2) the 233: row energy vs union converts
samples, states, tokmask, gold, sent = load_alg("test")
base=json.load(open('.cache/miss_census_gen41.json'))
om=set(base["miss_idx"])
conv=set()
for f in ("miss_census_g50.json","miss_census_g50r.json","miss_census_g51.json"):
    nm=set(json.load(open('.cache/'+f))["miss_idx"]); conv|=(om-nm)
ROWS=sorted(om); EN=[]
for s0 in range(0,len(ROWS),8):
    sl=np.array(ROWS[s0:s0+8])
    pad=8-len(sl); slp=np.concatenate([sl,sl[:1].repeat(pad)]) if pad else sl
    tr=Tensor(states[slp].astype(np.float32),dtype=dtypes.float)
    sn,en=grains(tr,Tensor(tokmask[slp].astype(np.float32),dtype=dtypes.float),
                 Tensor(sent[slp].astype(np.int32),dtype=dtypes.int),sent[slp])
    for bi in range(len(sl)):
        ev,pg=energy(sn,en,bi)
        EN.append((ev.sum(), pg.max()))
EN=np.array(EN); y=np.array([1 if r in conv else 0 for r in ROWS])
for col,name in ((0,"EVENT-count"),(1,"PGE-max")):
    pos=EN[y==1,col]; neg=EN[y==0,col]
    auc=float((pos[:,None]>neg[None,:]).mean()+0.5*(pos[:,None]==neg[None,:]).mean())
    print(f"[{name} 233] -> convert AUC {auc:.3f}",flush=True)
json.dump({"fixture_pge":hits,"fixture_tot":tot,"ev_recall":rec,"ev_sparsity":spars},open('.cache/binding_energy2.json','w'))
