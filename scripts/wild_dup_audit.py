import os, sys, json
for k,v in [("ALG2","1"),("ALG_FTYPES","8"),("ALG_HW","512"),("ALG_DUP","1"),("ALG_WIDE","1"),("ALG_INV","1")]: os.environ.setdefault(k,v)
os.environ.setdefault("ALG_TEST",".cache/algebra_nl_bigtest.jsonl"); os.environ.setdefault("ALG_TEST_NAME","bigtest")
sys.path.insert(0,'.'); sys.path.insert(0,'scripts')
import numpy as np, json as _j, re
from phase1_algebra_head import T_ALG, build_params, forward, sent_indices, TOKENIZER_JSON
from beacon_closing_arm import recompute_states
from tokenizers import Tokenizer
from tinygrad import Tensor, dtypes
from tinygrad.nn.state import safe_load
tok=Tokenizer.from_file(TOKENIZER_JSON)
p=build_params(0); sd=safe_load(os.environ["CK"])
for k in p: p[k].assign(sd[k].to(p[k].device).cast(p[k].dtype)).realize()
def int_answer(a):
    s=str(a).strip().replace("$","").replace(",","")
    return int(s) if re.fullmatch(r"-?\d+",s) else None
h=[_j.loads(l) for l in open(".cache/math_harvest_v0.jsonl")]
wild=[x["problem"] for x in h if x["level"]=="Level 4" and len(x["problem"])<300
      and "asy]" not in x["problem"]
      and all(int(n)<=300 for n in re.findall(r"\d+",x["problem"]))
      and int_answer(x["answer"]) is not None and 0<=int_answer(x["answer"])<=300][:40]
# rebuild the SAME detector the rite fit (same seeds/pipeline)? Too heavy;
# instead reuse the model's own dup logit as a cross-check AND report which
# wild problems have rel slots at all, with texts, for eye audit.
rows=[]
for wi,t in enumerate(wild):
    ids=np.zeros((8,T_ALG),np.int32); msk=np.zeros((8,T_ALG),np.float32); snt=np.zeros((8,T_ALG),np.int32)
    e=tok.encode(t); Ln=min(len(e.ids),T_ALG)
    ids[0,:Ln]=e.ids[:Ln]; msk[0,:Ln]=1.0
    snt[0]=sent_indices(t,list(e.offsets),msk[0])
    o=forward(p,Tensor(recompute_states(ids).astype(np.float32),dtype=dtypes.float),
              Tensor(msk,dtype=dtypes.float),Tensor(snt,dtype=dtypes.int))
    onp={k:o[k].realize().numpy() for k in ("pres","ftype","dup")}
    for j in range(24):
        if onp["pres"][0,j]>0 and onp["ftype"][0,j].argmax()==0:
            rows.append((float(onp["dup"][0,j]), wi, j, t[:110].replace("\n"," ")))
rows.sort(reverse=True)
for r in rows[:8]: print(f"[wild-slot] duplogit {r[0]:+.3f}  prob#{r[1]} slot{r[2]}  {r[3]}")
print(f"[n] wild rel slots {len(rows)} over {len(wild)} problems")
