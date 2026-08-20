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
rows=[json.loads(l) for l in open('.cache/form_mix9.jsonl')]
n=len(rows)
b8=np.load('.cache/phase1_alg_states_form8_states.npy', mmap_mode='r')  # 96,100
out=np.lib.format.open_memmap('.cache/phase1_alg_states_form9_states.npy', mode='w+', dtype=np.float16, shape=(n,T_ALG,2048))
CH=4096
for s0 in range(0,96100,CH): out[s0:min(s0+CH,96100)]=b8[s0:min(s0+CH,96100)]
# the 250 new rows: 25 unique texts x 10 reps -> compute 25, tile
samples, ids2, mask, offsets = PH.tokenize('.cache/form_mix9.jsonl')
uniq_texts=[rows[96100+i]["text"] for i in range(25)]
uids=np.zeros((25,T_ALG),np.int32)
for li in range(25):
    uids[li]=ids2[96100+li]
sts=np.asarray(recompute_states(uids)).astype(np.float16)
for rep in range(10):
    base=96100+rep*25
    for li in range(25):
        assert rows[base+li]["text"]==uniq_texts[li]
        out[base+li]=sts[li]
out.flush(); del out
gold=build_gold(samples, offsets)
sent=np.stack([sent_indices(s["text"],o,mask[i]) for i,(s,o) in enumerate(zip(samples,offsets))])
np.savez('.cache/phase1_alg_states_form9.npz', tokmask=mask.astype(np.uint8),
         sent=sent.astype(np.int8), mix_sha=mix_sha16('.cache/form_mix9.jsonl'),
         **{f"g_{k}":v for k,v in gold.items()})
print(f"[assemble9] STITCHED 96100 + computed 25x10 + gold + sha ({n} rows)", flush=True)
