"""form_assemble13.py — G51 THE MIXTURE (2026-08-21, word given): deeds
demoted to BASE (machine-L1 = rehearsal substrate, x1), diet = human
surgery + machine-L2 rescue deeds x DIET_REPS. Positive-presence guards
on both demotion and rescue inclusion. Solutions re-derived (key-lawful).
"""
import sys, os, json
sys.path.insert(0,'.'); sys.path.insert(0,'scripts')
os.environ.setdefault("ALG2","1"); os.environ.setdefault("ALG_FTYPES","8")
os.environ.setdefault("ALG_HW","512"); os.environ.setdefault("ALG_DUP","1")
os.environ.setdefault("ALG_WIDE","1")
import numpy as np
from phase1_algebra_head import T_ALG, sent_indices, build_gold
import phase1_algebra_head as PH
from beacon_closing_arm import recompute_states
from mycelium.era import mix_sha16
import glob as _g


import re as _re
_CL=_re.compile(r"\\. |, | and |; |\\? ")
def clause_indices(text, offs, mask_row):
    from phase1_algebra_head import SENT_MAX
    if len(_re.split(r"(?<=\\.)\\s+",text.strip()))>1:
        return sent_indices(text,offs,mask_row)
    bounds=[m.end()-1 for m in _CL.finditer(text)]
    out=np.zeros((T_ALG,),np.int32)
    ntk=int(mask_row.sum())
    arr=np.asarray(offs[:min(ntk,T_ALG)],dtype=np.int64)
    if len(arr) and bounds:
        idx=np.searchsorted(np.asarray(bounds,dtype=np.int64),arr[:,0],"right")
        out[:len(arr)]=np.minimum(idx,SENT_MAX-1)
    return out

files=sorted(_g.glob('.cache/book*_t*_batch*.jsonl'))
assert not any('t7self' in f for f in files), \
    "DEMOTION NOT DONE: t7self deeds still in the diet glob — rename to .cache/base_t7self_deeds.jsonl first"
assert any('t8rescue' in f for f in files), \
    "POSITIVE PRESENCE: rescue batch .cache/book10_t8rescue_batch1.jsonl absent from diet glob"
assert os.path.exists('.cache/base_t7self_deeds.jsonl'), "base deeds file missing"

def load_rows(fs):
    return [json.loads(l) for f in fs for l in open(f) if l.strip()]

def to_train(rows,gen):
    out=[]
    for r in rows:
        out.append({"text":r["original"],"factors":r["factors"],"query_var":r["query"],
                    "n_vars":24,"m":300,"solution":None,"decisions":0,"mentions":{},
                    "gen":gen})
    from mycelium.csp_domains import problem_from_algebra3 as problem_from_algebra2
    from mycelium.csp_core import solve_symbolic
    for t,src in zip(out,rows):
        gv={f["var"]:f["value"] for f in t["factors"] if f["ftype"]=="given"}
        res=solve_symbolic(problem_from_algebra2(24,t["factors"],gv,300),budget=200_000,seed=0)
        assert res["status"]=="solved"
        t["solution"]=[int(res["assignment"][v]) for v in range(24)]
        assert t["solution"][t["query_var"]]==src["answer"]
    return out

deeds=to_train(load_rows(['.cache/base_t7self_deeds.jsonl']),"b10deed")
diet=to_train(load_rows(files),"b11diet")
rows8=[json.loads(l) for l in open('.cache/form_mix8.jsonl')]
rng=np.random.RandomState(41)
keep=np.sort(rng.choice(96100, 25000, replace=False))
REPS=int(os.environ.get("DIET_REPS","3"))
ND, NU = len(deeds), len(diet)
mix=[rows8[i] for i in keep]+deeds+[r for _ in range(REPS) for r in diet]
with open('.cache/form_mix13.jsonl','w') as f:
    for r in mix: f.write(json.dumps(r)+"\n")
n=len(mix)
print(f"[assemble13] base 25000 + deeds {ND} (x1) + diet {NU}x{REPS}={REPS*NU} = {n}"
      f" (diet mass {REPS*NU})", flush=True)
b8=np.load('.cache/phase1_alg_states_form8_states.npy', mmap_mode='r')
out=np.lib.format.open_memmap('.cache/phase1_alg_states_form13_states.npy', mode='w+',
                              dtype=np.float16, shape=(n,T_ALG,2048))
CH=2048
for s0 in range(0,25000,CH):
    sl=keep[s0:min(s0+CH,25000)]
    out[s0:s0+len(sl)]=b8[sl]
samples, ids2, mask, offsets = PH.tokenize('.cache/form_mix13.jsonl')
NUQ=ND+NU
uids=np.zeros((NUQ,T_ALG),np.int32)
for li in range(NUQ): uids[li]=ids2[25000+li]
sts=np.asarray(recompute_states(uids)).astype(np.float16)
for li in range(ND):
    assert mix[25000+li]["text"]==deeds[li]["text"]
    out[25000+li]=sts[li]
for rep in range(REPS):
    base=25000+ND+rep*NU
    for li in range(NU):
        assert mix[base+li]["text"]==diet[li]["text"]
        out[base+li]=sts[ND+li]
out.flush(); del out
gold=build_gold(samples, offsets)
sent=np.stack([clause_indices(s["text"],o,mask[i]) for i,(s,o) in enumerate(zip(samples,offsets))])
n_clause=sum(1 for s2 in samples if len(_re.split(r"(?<=\.)\s+",s2["text"].strip()))<=1)
assert n_clause>0, "POSITIVE PRESENCE: no single-sentence rows — clause grains never engaged"
print(f"[assemble13] clause-grain rows: {n_clause}/{len(samples)}", flush=True)
np.savez('.cache/phase1_alg_states_form13.npz', tokmask=mask.astype(np.uint8),
         sent=sent.astype(np.int8), mix_sha=mix_sha16('.cache/form_mix13.jsonl'),
         **{f"g_{k}":v for k,v in gold.items()})
print(f"[assemble13] STITCHED base+deeds+diet + gold + sha ({n} rows)", flush=True)
