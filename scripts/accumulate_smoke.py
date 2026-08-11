"""accumulate_smoke.py — the ACCUMULATE spec's object-decision smoke
(inference-only): pass-2 conditioned on pass-1's committed factors
(decoded GIVENS restated canonically, appended as text). Does any of
the 233 convert? Null pre-written: flat = the fourth no-transfer at
the charter grain (deep structure unresponsive to committed context
in EITHER direction)."""
import os, sys, json
for k,v in [("ALG2","1"),("ALG_FTYPES","8"),("ALG_HW","512"),("ALG_DUP","1"),("ALG_WIDE","1")]: os.environ.setdefault(k,v)
os.environ.setdefault("ALG_TEST",".cache/algebra_nl_bigtest.jsonl"); os.environ.setdefault("ALG_TEST_NAME","bigtest")
sys.path.insert(0,'.'); sys.path.insert(0,'scripts')
import numpy as np
from phase1_algebra_head import T_ALG, build_params, forward, decode, load_alg, sent_indices, TOKENIZER_JSON
from beacon_closing_arm import recompute_states
from tta_alg2_dials import solve2
from tokenizers import Tokenizer
from tinygrad import Tensor, dtypes
from tinygrad.nn.state import safe_load
tok=Tokenizer.from_file(TOKENIZER_JSON)
samples, states, tokmask, gold, sent = load_alg("test")
base=json.load(open('.cache/miss_census_gen41.json'))
ROWS=sorted(base["miss_idx"]) if os.environ.get("FULL_POP")!="1" else list(range(len(samples)))
p=build_params(0); sd=safe_load(json.load(open('.cache/GENERATION.json'))["parser_ckpt"])
for k in p: p[k].assign(sd[k].to(p[k].device).cast(p[k].dtype)).realize()
L="abcdefghijklmnopqrstuvwx"
def run_pass(texts):
    outs=[]
    for s0 in range(0,len(texts),8):
        ch=texts[s0:s0+8]
        ids=np.zeros((8,T_ALG),np.int32); msk=np.zeros((8,T_ALG),np.float32); snt=np.zeros((8,T_ALG),np.int32)
        for i,t in enumerate(ch):
            e=tok.encode(t); Ln=min(len(e.ids),T_ALG)
            ids[i,:Ln]=e.ids[:Ln]; msk[i,:Ln]=1.0
            snt[i]=sent_indices(t,list(e.offsets),msk[i])
        o=forward(p,Tensor(recompute_states(ids).astype(np.float32),dtype=dtypes.float),
                  Tensor(msk,dtype=dtypes.float),Tensor(snt,dtype=dtypes.int))
        keys=("pres","ftype","op","islit","dig","args","res","query")+(("sel",) if "sel" in o else ())+(("dup",) if "dup" in o else ())+(("sgn",) if "sgn" in o else ())
        onp={k2:o[k2].realize().numpy() for k2 in keys}
        for i in range(len(ch)):
            outs.append(decode({k2:onp[k2][i] for k2 in onp}))
    return outs
texts1=[samples[r]["text"] for r in ROWS]
p1=run_pass(texts1)
conv=0; reg=0; texts2=[]; kept=[]
for i,r in enumerate(ROWS):
    facs,q=p1[i]
    givens=[f for f in facs if f.get("ftype")=="lit" or f.get("ftype")=="given"]
    parts=[f"It is known that {L[f['var']]} is {f['value']}." for f in givens
           if isinstance(f.get('var'),int) and 0<=f.get('var',99)<24 and isinstance(f.get('value'),int)]
    if os.environ.get("REL_COMMITS")=="1":
        for f in facs:
            if f.get("ftype")=="rel" and len(set(f.get("args",[])))==2 and f.get("result") is not None:
                a,b=sorted(f["args"]); c=f["result"]
                if 0<=a<24 and 0<=b<24 and 0<=c<24 and c not in (a,b):
                    if f.get("op")=="add": parts.append(f"{L[a]} and {L[b]} together make {L[c]}.")
                    elif f.get("op")=="mul": parts.append(f"The product of {L[a]} and {L[b]} is {L[c]}.")
    add=" ".join(parts)
    texts2.append(samples[r]["text"]+" "+add if add else samples[r]["text"])
    kept.append(add)
p2=run_pass(texts2)
recs=[]
for i,r in enumerate(ROWS):
    g_=samples[r]["solution"][samples[r]["query_var"]]
    facs1,q1=p1[i]; facs2,q2=p2[i]
    a1=solve2(facs1,q1,{"n_vars":24,"m":300}); a2=solve2(facs2,q2,{"n_vars":24,"m":300})
    if a1!=g_ and a2==g_: conv+=1
    if a1==g_ and a2!=g_: reg+=1
    ol=int(tokmask[r].sum()); al=len(tok.encode(kept[i]).ids) if kept[i] else 0
    recs.append({"r":int(r),"p1":bool(a1==g_),"p2":bool(a2==g_),"ol":ol,"al":al})
print(f"[smoke] converts {conv}  regressions {reg}",flush=True)
import numpy as _np
R=[x for x in recs if x["p1"] and not x["p2"]]
P=[x for x in recs if x["p1"]]
for lo,hi,name in ((0,250,"IN-window (ol+al<=250)"),(250,9999,"OVER-window (ol+al>250)")):
    pp=[x for x in P if lo < x["ol"]+x["al"] <= hi]
    rr=[x for x in pp if not x["p2"]]
    print(f"[discriminator] {name}: regress {len(rr)}/{len(pp)} = {len(rr)/max(len(pp),1):.3f}",flush=True)
json.dump({"converts":conv,"regressions":reg,"recs":recs},open(os.environ.get("SMOKE_OUT",".cache/accumulate_smoke.json"),"w"))
