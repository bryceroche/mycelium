"""paraphrase_collapse_read.py — first light (2026-08-04; pins in
ledger). 200 bigtest rows x 5 views under g23: canonical identity
across views. Seed base 103950 (adjacent claim, flip family)."""
import os, sys, json
os.environ.setdefault("ALG2","1"); os.environ.setdefault("ALG_FTYPES","8")
os.environ.setdefault("ALG_HW","512"); os.environ.setdefault("ALG_DUP","1")
os.environ.setdefault("ALG_WIDE","1")
sys.path.insert(0,'.'); sys.path.insert(0,'scripts')
import numpy as np
from collections import Counter
from phase1_algebra_head import T_ALG, build_params, forward, decode, sent_indices, TOKENIZER_JSON
from beacon_closing_arm import recompute_states
from tta_views import permuted_view
from tta_alg2_dials import solve2
from mycelium.canonicalizer import canonical_digest
from tokenizers import Tokenizer
from tinygrad import Tensor, dtypes
from tinygrad.nn.state import safe_load
tok=Tokenizer.from_file(TOKENIZER_JSON)
rows=[json.loads(l) for l in open('.cache/algebra_nl_bigtest.jsonl')][:200]
p=build_params(0); sd=safe_load('.cache/g23.safetensors')
for k in p: p[k].assign(sd[k].to(p[k].device).cast(p[k].dtype)).realize()

def parse_batch(texts):
    n=len(texts); N=((n+7)//8)*8
    ids=np.zeros((N,T_ALG),np.int32); msk=np.zeros((N,T_ALG),np.float32); snt=np.zeros((N,T_ALG),np.int32)
    for i,t in enumerate(texts):
        e=tok.encode(t); Ln=min(len(e.ids),T_ALG)
        ids[i,:Ln]=e.ids[:Ln]; msk[i,:Ln]=1.0
        snt[i]=sent_indices(t,list(e.offsets),msk[i])
    st=recompute_states(ids)
    out=[]
    for s0 in range(0,N,8):
        o=forward(p,Tensor(st[s0:s0+8].astype(np.float32),dtype=dtypes.float),
                  Tensor(msk[s0:s0+8],dtype=dtypes.float),
                  Tensor(snt[s0:s0+8],dtype=dtypes.int))
        keys=("pres","ftype","op","islit","dig","sgn","args","res","query")+(("sel",) if "sel" in o else ())+(("dup",) if "dup" in o else ())
        onp={k:o[k].realize().numpy() for k in keys}
        for bi in range(8):
            if s0+bi<n: out.append(decode({k:onp[k][bi] for k in onp}))
    return out

full=0; modal=[]; res=[]
for j,r in enumerate(rows):
    t=r["text"]
    vt=[t]+[permuted_view(t,103950+10*j+k) for k in range(1,5)]
    parsed=parse_batch(vt)
    digs=[]
    for f,q in parsed:
        try: digs.append(canonical_digest(f,q,n_vars=24))
        except Exception: digs.append("ERR")
    c=Counter(digs).most_common(1)[0][1]
    modal.append(c)
    if c==5: full+=1
    ans=[solve2(f,q,{"n_vars":24,"m":300}) for f,q in parsed]
    nn=[a for a in ans if a is not None]
    ac=Counter(nn).most_common(1)
    ok = bool(ac) and ac[0][0]==r["solution"][r["query_var"]] and ac[0][1]>=3
    ans_agree = ac[0][1] if ac else 0
    res.append({"modal":c,"correct":ok,"ans_agree":ans_agree})
    if (j+1)%50==0: print(f"  [{j+1}/200]",flush=True)
m=np.array(modal)
print(f"[collapse] full(5/5-identical): {full}/200 = {full/200:.1%}  modal mean {m.mean():.2f}/5")
ok=[r for r in res if r["correct"]]; bad=[r for r in res if not r["correct"]]
print(f"[collapse] modal | correct rows {np.mean([r['modal'] for r in ok]):.2f}  wrong rows {np.mean([r['modal'] for r in bad]):.2f}")
coinc=sum(1 for r in res if r["ans_agree"]>r["modal"])
print(f"[collapse] ANSWER-COINCIDENCE rows (answers agree beyond graph identity): {coinc}/200")
json.dump({"full":full,"modal_mean":float(m.mean()),
           "modal_correct":float(np.mean([r['modal'] for r in ok])),
           "modal_wrong":float(np.mean([r['modal'] for r in bad])) if bad else None,
           "answer_coincidence":coinc},
          open(".cache/paraphrase_collapse.json","w"),indent=1)
print("[saved] .cache/paraphrase_collapse.json")
