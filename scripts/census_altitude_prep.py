"""census_altitude_prep.py — THE STRATA CUT (2026-08-02, driving toward
gut #123 via #69's assigned split: 'Claude Code cuts the strata
mechanically; Bryce's hands take it from there').

Pool: the 46 wild lies (answered AND wrong under the deployed g22 chain,
wild_ledger_v1). This is the December-relevant lie population — the
pool amendment vs #69's original strata spec (zone/species/vintage) is
NOTED for Bryce's acceptance: one vintage (g22, cleaner), stratified by
what the corrected cut says matters (trained-verbatim vs never-trained,
level). The annotation itself WAITS on Bryce's hand + the tighten pass
(#69's law: spec before specimens).

Captures per specimen: text, level/subject, trained-verbatim (sha ∩
deployed mix), gold, per-view answers, quorum answer, and the VIEW-0
PARSED GRAPH (what the chain believed) — everything the altitude
judgment (schema-level extraction vs arith3-level assembly) needs on
one sheet. Seed base 103000 (claimed in doors.VIEW_SEED_BASES).
"""
import sys, os, json, hashlib
sys.path.insert(0,'.'); sys.path.insert(0,'scripts')
os.environ.setdefault("ALG2","1"); os.environ.setdefault("ALG_FTYPES","8")
os.environ.setdefault("ALG_HW","512"); os.environ.setdefault("ALG_DUP","1")
import numpy as np
from collections import Counter
from phase1_algebra_head import T_ALG, build_params, forward, decode, sent_indices, TOKENIZER_JSON
from beacon_closing_arm import recompute_states
from tta_views import permuted_view
from tta_alg2_dials import solve2
from tokenizers import Tokenizer
from tinygrad import Tensor, dtypes
from tinygrad.nn.state import safe_load
tok = Tokenizer.from_file(TOKENIZER_JSON)

mix_shas=set()
for l in open('.cache/gen22_mix.jsonl'):
    mix_shas.add(hashlib.sha256(json.loads(l)['text'].encode()).hexdigest())
recs=[json.loads(l) for l in open('.cache/wild_ledger_v1.jsonl')]
h=[json.loads(l) for l in open('.cache/math_harvest_v0.jsonl')]
lies=[r for r in recs if r['tier']=='answered' and not r['correct']]
print(f"[strata] wild lies: {len(lies)}")

p=build_params(0); sd=safe_load('.cache/g22.safetensors')
for k in p: p[k].assign(sd[k].to(p[k].device).cast(p[k].dtype)).realize()

def parse_batch(texts):
    n=len(texts); N=((n+7)//8)*8
    ids=np.zeros((N,T_ALG),np.int32); msk=np.zeros((N,T_ALG),np.float32); snt=np.zeros((N,T_ALG),np.int32)
    for i,t in enumerate(texts):
        e=tok.encode(t); Ln=min(len(e.ids),T_ALG)
        ids[i,:Ln]=e.ids[:Ln]; msk[i,:Ln]=1.0
        snt[i]=sent_indices(t,list(e.offsets),msk[i])
    st=recompute_states(ids)
    out_r=[]
    for s0 in range(0,N,8):
        out=forward(p,Tensor(st[s0:s0+8].astype(np.float32),dtype=dtypes.float),
                    Tensor(msk[s0:s0+8].astype(np.float32),dtype=dtypes.float),
                    Tensor(snt[s0:s0+8].astype(np.int32),dtype=dtypes.int))
        keys=("pres","ftype","op","islit","dig","args","res","query")+(("sel",) if "sel" in out else ())+(("dup",) if "dup" in out else ())
        o={k:out[k].realize().numpy() for k in keys}
        for bi in range(8):
            if s0+bi<n: out_r.append(decode({k:o[k][bi] for k in o}))
    return out_r

rows=[]
for idx,r in enumerate(lies):
    t=h[r['harvest_idx']]['problem']
    views=[t]+[permuted_view(t,103000+10*idx+k) for k in range(1,5)]
    parsed=parse_batch(views)
    answers=[solve2(f,q,{"n_vars":24,"m":300}) for f,q in parsed]
    nn=[a for a in answers if a is not None]
    c=Counter(nn).most_common(1)
    f0,q0=parsed[0]
    rows.append({
        "harvest_idx": r['harvest_idx'], "text": t,
        "level": r['level'], "subject": h[r['harvest_idx']].get('subject',''),
        "trained_verbatim": hashlib.sha256(t.encode()).hexdigest() in mix_shas,
        "gold": r['gold'],
        "view_answers": answers, "quorum_answer": c[0][0] if c else None,
        "quorum_count": c[0][1] if c else 0,
        "graph_view0": {"factors": f0, "query": int(q0) if q0 is not None else None},
        "altitude": None,  # Bryce's hand: "schema" | "assembly" | "other"
    })
    if (idx+1)%10==0: print(f"  [{idx+1}/{len(lies)}]",flush=True)

with open('.cache/census_altitude_specimens.jsonl','w') as f:
    for r in rows: f.write(json.dumps(r)+"\n")
strata=Counter((r['trained_verbatim'],r['level']) for r in rows)
print("[strata] (trained,level):",dict(strata))
print(f"[saved] .cache/census_altitude_specimens.jsonl  n={len(rows)}")
