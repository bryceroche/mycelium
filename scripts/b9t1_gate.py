"""b9t1_gate.py — THE VOTE GATE on book9 t1's 25 (the books' law: 5-view
vote >=3 + the answer key; the gate judges the surgeon's prose too)."""
import os, sys, json, re, random
sys.path.insert(0,'.'); sys.path.insert(0,'scripts')
import numpy as np
from phase1_algebra_head import build_params, forward, load_alg, decode, T_ALG, TOKENIZER_JSON, sent_indices
from repair_replace_swap import solve_forced
from beacon_closing_arm import recompute_states
from tokenizers import Tokenizer
from tinygrad import Tensor, dtypes
from tinygrad.nn.state import safe_load
samples, states, tokmask, gold, sent = load_alg("test")
tok=Tokenizer.from_file(TOKENIZER_JSON)
p=build_params(0); sd=safe_load(os.environ["ALG_CKPT"])
for k in p: p[k].assign(sd[k].to(p[k].device).cast(p[k].dtype)).realize()
K=("pres","ftype","op","islit","dig","args","res","query")
def parse_texts(texts):
    m=len(texts)
    ids=np.zeros((m,T_ALG),np.int32); msk=np.zeros((m,T_ALG),np.float32); snt=np.zeros((m,T_ALG),np.int32)
    for li,t in enumerate(texts):
        e=tok.encode(t)
        if len(e.ids)>T_ALG: continue
        ids[li,:len(e.ids)]=e.ids; msk[li,:len(e.ids)]=1.0
        snt[li]=sent_indices(t,list(e.offsets),msk[li])
    sts=recompute_states(ids)
    out=[]
    for s0 in range(0,m,8):
        sl=list(range(s0,min(s0+8,m))); pad=8-len(sl); slp=sl+sl[:1]*pad
        o=forward(p,Tensor(sts[slp].astype(np.float32),dtype=dtypes.float),
                    Tensor(msk[slp].astype(np.float32),dtype=dtypes.float),
                    Tensor(snt[slp].astype(np.int32),dtype=dtypes.int))
        ex=tuple(k2 for k2 in ("sel","dup","sgn") if k2 in o)
        onp={k2:o[k2].realize().numpy() for k2 in K+ex}
        for bi in range(len(sl)): out.append(decode({k2:onp[k2][bi] for k2 in onp}))
    return out
rows=[json.loads(l) for l in open('.cache/book9_t1_batch1.jsonl')]+\
     [json.loads(l) for l in open('.cache/book9_t1_batch23.jsonl')]
passed=[]
for r in rows:
    parts=re.split(r"(?<=\.)\s+",r["prose"].strip())
    vt=[r["prose"]]
    for k in range(1,5):
        mid=parts[1:-1]; random.Random(1000*k+r["src_idx"]).shuffle(mid)
        vt.append(" ".join([parts[0]]+mid+[parts[-1]]))
    parses=parse_texts(vt)
    ans=[solve_forced(f,q,{"n_vars":24,"m":300}) for f,q in parses]
    hits=sum(1 for a in ans if a==r["answer"])
    ok=hits>=3
    print(f"[gate {r['src_idx']}] views-correct {hits}/5 {'PASS' if ok else 'FAIL'}")
    if ok: r["gate"]=f"{hits}/5"; passed.append(r)
with open('.cache/book9_t1_gated.jsonl','w') as f:
    for r in passed: f.write(json.dumps(r)+"\n")
print(f"[gate] GATED: {len(passed)}/{len(rows)} enter the diet pool")
print("== GATE COMPLETE ==")
