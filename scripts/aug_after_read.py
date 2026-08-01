"""aug_after_read.py — PHASE A (2026-08-01): the decisive cheap reads.
(1) the dup held-out-config fixture (the 24%) and (2) the fdiv
varied-surface fixture (the 0/8) under g22/vlow/vfull — held-out-of-
table renderings by construction (mint_iso's sum/product phrasing IS
in the table now! CHECK: iso used 'The sum of {a} and {a}' — sum-self
IS licensed... the fixture is no longer held-out-of-table for dup.
HONEST HANDLING: dup read via a NOVEL phrasing neither table nor pool
contains ('Adding {a} to itself gives {c}'); fdiv varied ('Dividing c
by K gives d') IS in the table (dividing licensed) — fdiv eval switches
to a held-out form ('{a} split into {k} equal parts gives {b}').
THE RECURSION GUARD applied at read time: eval templates verified
absent from the licensed table before reading. (3) slot emission +
misbinding via bench_rung2b per arm. Notes: mouth-native deferred to
entourage (input-side; moves only when the bank rebuilds). Flip rate =
phase B."""
import sys, os, json, subprocess, time
sys.path.insert(0,'.'); sys.path.insert(0,'scripts')
os.environ.setdefault("ALG2","1"); os.environ.setdefault("ALG_FTYPES","8")
os.environ.setdefault("ALG_HW","512"); os.environ.setdefault("ALG_DUP","1")
while subprocess.run(["systemctl","--user","is-active","aug-fire.service"],
                     capture_output=True,text=True).stdout.strip()=="active":
    time.sleep(30)
import numpy as np, re
from collections import Counter
from phase1_algebra_head import T_ALG, build_params, forward, decode, sent_indices, TOKENIZER_JSON
from beacon_closing_arm import recompute_states
from tta_views import permuted_view
from tta_alg2_dials import solve2
from tokenizers import Tokenizer
from tinygrad import Tensor, dtypes
from tinygrad.nn.state import safe_load
tok = Tokenizer.from_file(TOKENIZER_JSON)
L = "abcdefghij"
TABLE = json.load(open('.cache/aug_table_v1.json'))["licensed"]
for probe in ("Adding {a} to itself", "split into"):
    assert not any(probe.split("{")[0].strip() in e["fmt"] for e in TABLE), probe
print("[guard] eval templates verified ABSENT from the licensed table")

def load(ck):
    p=build_params(0); sd=safe_load(ck)
    for k in p: p[k].assign(sd[k].to(p[k].device).cast(p[k].dtype)).realize()
    return p
def one(p, text):
    ids=np.zeros((8,T_ALG),np.int32); msk=np.zeros((8,T_ALG),np.float32); snt=np.zeros((8,T_ALG),np.int32)
    e=tok.encode(text); Ln=min(len(e.ids),T_ALG)
    ids[0,:Ln]=e.ids[:Ln]; msk[0,:Ln]=1.0
    snt[0]=sent_indices(text,list(e.offsets),msk[0])
    out=forward(p,Tensor(recompute_states(ids).astype(np.float32),dtype=dtypes.float),
                Tensor(msk,dtype=dtypes.float),Tensor(snt,dtype=dtypes.int))
    keys=("pres","ftype","op","islit","dig","args","res","query")+(("sel",) if "sel" in out else ())+(("dup",) if "dup" in out else ())
    o={k:out[k].realize().numpy() for k in keys}
    return decode({k:o[k][0] for k in o})[0]

def dup_novel(n, seed):
    rng=np.random.RandomState(seed); rows=[]
    while len(rows)<n:
        op="add" if rng.rand()<0.5 else "mul"
        x=int(rng.randint(2,60)) if op=="add" else int(rng.randint(2,13))
        nd=int(rng.randint(2,5)); gv=[int(rng.randint(2,90)) for _ in range(nd)]
        gold=x+x if op=="add" else x*x
        if gold>300: continue
        dv=nd; res=nd+1
        w = "Adding {a} to itself gives {c}." if op=="add" else "Multiplying {a} by itself gives {c}."
        order=list(range(nd)); rng.shuffle(order)
        sents=[f"{L[i]} is {gv[i]}." for i in order]+[f"{L[dv]} is {x}.", w.format(a=L[dv],c=L[res])]
        facs=[{"ftype":"given","var":i,"value":gv[i]} for i in range(nd)]+[{"ftype":"given","var":dv,"value":x},{"ftype":"rel","op":op,"args":[dv,dv],"result":res}]
        if solve2(facs,res,{"n_vars":24,"m":300})!=gold: continue
        rows.append({"text":f"Consider the numbers {', '.join(L[:res+1])}. "+" ".join(sents)+f" What is {L[res]}?","dv":dv,"op":op})
    return rows

def fdiv_novel(n, seed):
    rng=np.random.RandomState(seed); rows=[]
    while len(rows)<n:
        K=int(rng.choice([2,3,4,5,6,7])); A,B=int(rng.randint(2,12)),int(rng.randint(2,12))
        C=A*B
        if C%K or C>300: continue
        text=(f"Consider the numbers a, b, c, d. a is {A}. b is {B}. "
              f"a times b equals c. c split into {K} equal parts gives d. What is d?")
        facs=[{"ftype":"given","var":0,"value":A},{"ftype":"given","var":1,"value":B},
              {"ftype":"rel","op":"mul","args":[0,1],"result":2},
              {"ftype":"fdiv","var":2,"k":K,"result":3}]
        rows.append({"text":text,"facs":facs,"q":3,"gold":C//K})
    return rows

DUP=dup_novel(60,88100); FDV=fdiv_novel(8,88200)
res={}
for name,ck in (("g22",".cache/g22.safetensors"),("vlow",".cache/g24_vlow.safetensors"),("vfull",".cache/g24_vfull.safetensors")):
    p=load(ck)
    dm=sum(1 for r in DUP if not any(f.get("ftype")=="rel" and f.get("args")==[r["dv"],r["dv"]] and f.get("op")==r["op"] for f in one(p,r["text"])))
    fq=0
    for j,r in enumerate(FDV):
        vt=[r["text"]]+[permuted_view(r["text"],88300+20*j+k) for k in range(1,5)]
        ans=[]
        for t in vt: 
            facs=one(p,t)
            q=r["q"]
            ans.append(solve2(facs,q,{"n_vars":24,"m":300}))
        nn=[a for a in ans if a is not None]
        c=Counter(nn).most_common(1); plur,cnt=c[0] if c else (None,0)
        fq+=(cnt>=3 and plur==r["gold"])
    res[name]={"dup_novel_misbind":f"{dm}/60","fdiv_novel_quorum":f"{fq}/8"}
    print(f"[{name}] dup-novel misbind {dm}/60   fdiv-novel quorum {fq}/8",flush=True)
json.dump(res,open('.cache/aug_after_A.json','w'),indent=1)
print("[saved] .cache/aug_after_A.json — phase B (flip rate, 2b slot emission) rides next")
