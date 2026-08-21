import sys
sys.exit("RETIRED (2026-08-20 audit #3): stitch-source memmaps (form2-7) were "
         "reclaimed; regenerate via trunk recompute before reviving this script")
import sys, os, json
os.environ.setdefault("ALG2","1"); os.environ.setdefault("ALG_FTYPES","9")
os.environ.setdefault("ALG_HW","512"); os.environ.setdefault("ALG_DUP","1")
os.environ.setdefault("ALG_WIDE","1"); os.environ.setdefault("ALG_VALATT","1")
import numpy as np
from phase1_algebra_head import T_ALG, sent_indices, build_gold
import phase1_algebra_head as PH
from mycelium.era import mix_sha16
rows=[json.loads(l) for l in open('.cache/form_mix8.jsonl')]
n=len(rows)
b3=np.load('.cache/phase1_alg_states_form3_states.npy', mmap_mode='r')     # 94,100
b7=np.load('.cache/phase1_alg_states_form7_states.npy', mmap_mode='r')     # 102,100 (dv = last 2,000)
out=np.lib.format.open_memmap('.cache/phase1_alg_states_form8_states.npy', mode='w+', dtype=np.float16, shape=(n,T_ALG,2048))
CH=4096
for s0 in range(0,94100,CH): out[s0:min(s0+CH,94100)]=b3[s0:min(s0+CH,94100)]
out[94100:96100]=b7[100100:102100]
out.flush(); del out
samples, ids2, mask, offsets = PH.tokenize('.cache/form_mix8.jsonl')
gold=build_gold(samples, offsets)
sent=np.stack([sent_indices(s["text"],o,mask[i]) for i,(s,o) in enumerate(zip(samples,offsets))])
np.savez('.cache/phase1_alg_states_form8.npz', tokmask=mask.astype(np.uint8),
         sent=sent.astype(np.int8), mix_sha=mix_sha16('.cache/form_mix8.jsonl'),
         **{f"g_{k}":v for k,v in gold.items()})
print("[assemble8] STITCHED (no trunk recompute) + gold + sha", flush=True)
