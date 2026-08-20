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
rows8=[json.loads(l) for l in open('.cache/form_mix8.jsonl')]
b9=[json.loads(l) for l in open('.cache/book9_t1_batch1.jsonl')]+[json.loads(l) for l in open('.cache/book9_t1_batch23.jsonl')]
import copy as _cp
_b=[]
for r in b9:
    _b.append({"text":r["original"],"factors":r["factors"],"query_var":r["query"],
               "n_vars":24,"m":300,"solution":None,"decisions":0,"mentions":{},
               "gen":"b9t1"})
# solutions re-derived (key-lawful) at assembly
from mycelium.csp_domains import problem_from_algebra3 as problem_from_algebra2
from mycelium.csp_core import solve_symbolic
for r,src in zip(_b,b9):
    gv={f["var"]:f["value"] for f in r["factors"] if f["ftype"]=="given"}
    res=solve_symbolic(problem_from_algebra2(24,r["factors"],gv,300),budget=200_000,seed=0)
    assert res["status"]=="solved"
    r["solution"]=[int(res["assignment"][v]) for v in range(24)]
    assert r["solution"][r["query_var"]]==src["answer"]
b9=_b
rng=np.random.RandomState(41)
keep=np.sort(rng.choice(96100, 25000, replace=False))
NB9=len(b9)
mix=[rows8[i] for i in keep]+[r for _ in range(10) for r in b9]
with open('.cache/form_mix9.jsonl','w') as f:
    for r in mix: f.write(json.dumps(r)+"\n")
n=len(mix)
print(f"[assemble9] sampled-base mix: 25000 + {10*NB9} = {n} (share {250/n:.3%}, reps 10)", flush=True)
b8=np.load('.cache/phase1_alg_states_form8_states.npy', mmap_mode='r')
out=np.lib.format.open_memmap('.cache/phase1_alg_states_form9_states.npy', mode='w+', dtype=np.float16, shape=(n,T_ALG,2048))
CH=2048
for s0 in range(0,25000,CH):
    sl=keep[s0:min(s0+CH,25000)]
    out[s0:s0+len(sl)]=b8[sl]
samples, ids2, mask, offsets = PH.tokenize('.cache/form_mix9.jsonl')
uids=np.zeros((NB9,T_ALG),np.int32)
for li in range(NB9): uids[li]=ids2[25000+li]
sts=np.asarray(recompute_states(uids)).astype(np.float16)
for rep in range(10):
    base=25000+rep*NB9
    for li in range(NB9):
        assert mix[base+li]["text"]==b9[li]["text"]
        out[base+li]=sts[li]
out.flush(); del out
gold=build_gold(samples, offsets)
sent=np.stack([sent_indices(s["text"],o,mask[i]) for i,(s,o) in enumerate(zip(samples,offsets))])
np.savez('.cache/phase1_alg_states_form9.npz', tokmask=mask.astype(np.uint8),
         sent=sent.astype(np.int8), mix_sha=mix_sha16('.cache/form_mix9.jsonl'),
         **{f"g_{k}":v for k,v in gold.items()})
print(f"[assemble9] STITCHED 96100 + computed 25x10 + gold + sha ({n} rows)", flush=True)
