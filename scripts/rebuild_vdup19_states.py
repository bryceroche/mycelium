import sys, os, json
sys.path.insert(0,'.'); sys.path.insert(0,'scripts')
os.environ.setdefault("ALG2","1"); os.environ.setdefault("ALG_FTYPES","8")
os.environ.setdefault("ALG_HW","512"); os.environ.setdefault("ALG_DUP","1")
import numpy as np
from phase1_algebra_head import T_ALG, sent_indices, TOKENIZER_JSON, build_gold
import phase1_algebra_head as PH
from beacon_closing_arm import recompute_states
from tokenizers import Tokenizer
tok = Tokenizer.from_file(TOKENIZER_JSON)
mixp=".cache/augfire_vdup19_mix.jsonl"
rows=[json.loads(l) for l in open(mixp)]
changed=json.load(open(".cache/augfire_vdup19_changed.json"))
base=np.load(".cache/phase1_alg_states_g22_states.npy", mmap_mode="r")
npyp=".cache/phase1_alg_states_vdup19_states.npy"
out=np.lib.format.open_memmap(npyp, mode="w+", dtype=np.float16, shape=base.shape)
CH=4096
for s0 in range(0, base.shape[0], CH):
    out[s0:min(s0+CH,base.shape[0])]=base[s0:min(s0+CH,base.shape[0])]
for s0 in range(0, len(changed), 8):
    idxs=changed[s0:s0+8]
    ids=np.zeros((8,T_ALG),np.int32)
    for i,ridx in enumerate(idxs):
        e=tok.encode(rows[ridx]["text"]); Ln=min(len(e.ids),T_ALG)
        ids[i,:Ln]=e.ids[:Ln]
    st=recompute_states(ids).astype(np.float16)
    for i,ridx in enumerate(idxs): out[ridx]=st[i]
    if (s0//8)%150==0: print(f"  [patch {s0}/{len(changed)}]",flush=True)
out.flush(); del out
samples, ids2, mask, offsets = PH.tokenize(mixp)
gold=build_gold(samples, offsets)
sent=np.stack([sent_indices(s["text"],o,mask[i]) for i,(s,o) in enumerate(zip(samples,offsets))])
np.savez(".cache/phase1_alg_states_vdup19.npz", tokmask=mask.astype(np.uint8),
         sent=sent.astype(np.int8), **{f"g_{k}":v for k,v in gold.items()})
print("[rebuild] vdup19 states + gold staged", flush=True)
