"""crossgrain_smoke.py — the cross-grain anchor smoke (the word given):
ENGAGED-pass states (the loop's completions) pinned into the SILENT
read on g51_whisper. The anchors carry what silence cannot make.
Reads: the nd4 fixture (silent 15/15 vs engaged 0/15 — the widest
grain gap) + the 233 + 300 passers (cost from birth)."""
import os, sys, json
os.environ["ALG_BREATH"]="3"
for k,v in [("ALG2","1"),("ALG_FTYPES","8"),("ALG_HW","512"),("ALG_DUP","1"),("ALG_WIDE","1"),("ALG_INV","1")]: os.environ.setdefault(k,v)
os.environ.setdefault("ALG_TEST",".cache/algebra_nl_bigtest.jsonl"); os.environ.setdefault("ALG_TEST_NAME","bigtest")
sys.path.insert(0,'.'); sys.path.insert(0,'scripts')
import numpy as np
from phase1_algebra_head import T_ALG, build_params, forward, decode, load_alg, L_FAC, build_slot_masks, sent_indices, TOKENIZER_JSON
from beacon_closing_arm import recompute_states
from tta_alg2_dials import solve2
from tokenizers import Tokenizer
from tinygrad import Tensor, dtypes
from tinygrad.nn.state import safe_load
tok=Tokenizer.from_file(TOKENIZER_JSON)
p=build_params(0); sd=safe_load(".cache/g51_whisper.safetensors")
for k in p: p[k].assign(sd[k].to(p[k].device).cast(p[k].dtype)).realize()
PRES_G=float(os.environ.get("XG_PRES","2.0")); ARG_G=float(os.environ.get("XG_ARG","0")); DIFF_G=float(os.environ.get("XG_DIFF","0")); ROW_G=float(os.environ.get("XG_ROW","0"))
def both_grains(tr,tk,se,snt_np):
    o0=forward(p,tr,tk,se)                                   # silent (bare)
    o0n={k2:o0[k2].realize().numpy() for k2 in ("fat","args","res")}
    mk=build_slot_masks(o0n, snt_np)
    oe=forward(p,tr,tk,se,slot_mask=Tensor(mk,dtype=dtypes.float))   # engaged
    keys=("pres","ftype","op","islit","dig","args","res","query","fst_s")+(("sel",) if "sel" in oe else ())+(("dup",) if "dup" in oe else ())+(("sgn",) if "sgn" in oe else ())
    en={k2:oe[k2].realize().numpy() for k2 in keys}
    sn={k2:o0[k2].realize().numpy() for k2 in keys}
    anch=np.zeros((8,L_FAC,512),np.float32); am=np.zeros((8,L_FAC,1),np.float32)
    for bi in range(8):
        if ROW_G>0:
            _reloc=any((en["pres"][bi,jj]-sn["pres"][bi,jj])>ROW_G for jj in range(L_FAC))
            if not _reloc: continue        # no relocation -> row untouched
        for j in range(L_FAC):
            _g_ok = (en["pres"][bi,j] - sn["pres"][bi,j]) > DIFF_G if DIFF_G>0 \
                else en["pres"][bi,j]>PRES_G
            if _g_ok:                                        # the loop formed it
                if ARG_G>0:
                    a_=np.sort(en["args"][bi,j])[::-1]
                    if len(a_)>2 and (a_[1]-a_[2])<=ARG_G: continue
                anch[bi,j]=en["fst_s"][bi,j]; am[bi,j]=1.0
    oa=forward(p,tr,tk,se,anchor=Tensor(anch,dtype=dtypes.float),amask=Tensor(am,dtype=dtypes.float))  # silent + anchors
    an={k2:oa[k2].realize().numpy() for k2 in keys if k2!="fst_s"}
    return sn,en,an,am
# --- READ 1: the nd4 fixture (scan-grade) ---
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
        rows.append({"text":f"Consider the numbers {', '.join(L[:res+1])}. "+" ".join(sents)+f" What is {L[res]}?","op":op,"dv":dv})
    return rows
rows4=fixture_mint(4)
mis={"sil":0,"anch":0,"eng":0}
for s0 in range(0,15,8):
    ch=rows4[s0:s0+8]
    ids=np.zeros((8,T_ALG),np.int32); msk=np.zeros((8,T_ALG),np.float32); snt=np.zeros((8,T_ALG),np.int32)
    for i,r in enumerate(ch):
        e=tok.encode(r["text"]); Ln=min(len(e.ids),T_ALG)
        ids[i,:Ln]=e.ids[:Ln]; msk[i,:Ln]=1.0
        snt[i]=sent_indices(r["text"],list(e.offsets),msk[i])
    tr=Tensor(recompute_states(ids).astype(np.float32),dtype=dtypes.float)
    tk=Tensor(msk,dtype=dtypes.float); se=Tensor(snt,dtype=dtypes.int)
    sn,en,an,am=both_grains(tr,tk,se,snt)
    for i,r in enumerate(ch):
        for nm,oo in (("sil",sn),("eng",en),("anch",an)):
            facs=decode({k2:oo[k2][i] for k2 in oo if k2!="fst_s"})[0]
            ok=any(f.get("ftype")=="rel" and f.get("args")==[r["dv"],r["dv"]] and f.get("op")==r["op"] for f in facs)
            mis[nm]+= (not ok)
print(f"[xgrain nd4] misbind: silent {mis['sil']}/15  engaged {mis['eng']}/15  SILENT+ANCHORS {mis['anch']}/15",flush=True)
# --- READ 2: the 233 + passers (cost from birth) ---
samples, states, tokmask, gold, sent = load_alg("test")
base=json.load(open('.cache/miss_census_gen41.json'))
missset=set(base["miss_idx"])
rng=np.random.RandomState(7)
passers=[i for i in range(len(samples)) if i not in missset]
ROWS=sorted(missset)+list(rng.choice(passers,150,replace=False))
res={"miss":[0,0],"pass":[0,0]}
for s0 in range(0,len(ROWS),8):
    sl=np.array(ROWS[s0:s0+8])
    pad=8-len(sl); slp=np.concatenate([sl,sl[:1].repeat(pad)]) if pad else sl
    tr=Tensor(states[slp].astype(np.float32),dtype=dtypes.float)
    tk=Tensor(tokmask[slp].astype(np.float32),dtype=dtypes.float)
    se=Tensor(sent[slp].astype(np.int32),dtype=dtypes.int)
    sn,en,an,am=both_grains(tr,tk,se,sent[slp])
    for bi,ri in enumerate(sl):
        g_=samples[ri]["solution"][samples[ri]["query_var"]]
        f1,q1=decode({k2:sn[k2][bi] for k2 in sn if k2!="fst_s"})
        f2,q2=decode({k2:an[k2][bi] for k2 in an})
        a1=solve2(f1,q1,{"n_vars":24,"m":300}); a2=solve2(f2,q2,{"n_vars":24,"m":300})
        pop="miss" if ri in missset else "pass"
        if a1!=g_ and a2==g_: res[pop][0]+=1
        if a1==g_ and a2!=g_: res[pop][1]+=1
print(f"[xgrain 233] MISS: converts {res['miss'][0]} regress {res['miss'][1]} | PASS(150): converts {res['pass'][0]} regress {res['pass'][1]}",flush=True)
json.dump({"nd4":mis,"pops":res},open(os.environ.get('XG_OUT','.cache/crossgrain_smoke.json'),'w'))
