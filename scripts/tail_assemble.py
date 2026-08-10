import sys, os, json
sys.path.insert(0,'.'); sys.path.insert(0,'scripts')
os.environ.setdefault("ALG2","1"); os.environ.setdefault("ALG_FTYPES","8")
os.environ.setdefault("ALG_HW","512"); os.environ.setdefault("ALG_DUP","1")
os.environ.setdefault("ALG_WIDE","1")   # manifest era — the 3-vs-7 lesson
import numpy as np
from phase1_algebra_head import T_ALG, sent_indices, TOKENIZER_JSON, build_gold
import phase1_algebra_head as PH
from beacon_closing_arm import recompute_states
from tokenizers import Tokenizer
tok=Tokenizer.from_file(TOKENIZER_JSON)
rows=[json.loads(l) for l in open('.cache/tail_mix.jsonl')]
base=np.load('.cache/phase1_alg_states_gen23_states.npy', mmap_mode='r')
n=len(rows); nb=base.shape[0]
out=np.lib.format.open_memmap('.cache/phase1_alg_states_tail_states.npy', mode='w+', dtype=np.float16, shape=(n,T_ALG,2048))
CH=4096
for s0 in range(0,nb,CH): out[s0:min(s0+CH,nb)]=base[s0:min(s0+CH,nb)]
for s0 in range(nb,n,8):
    idxs=list(range(s0,min(s0+8,n)))
    ids=np.zeros((8,T_ALG),np.int32)
    for i,ri in enumerate(idxs):
        e=tok.encode(rows[ri]["text"]); Ln=min(len(e.ids),T_ALG)
        ids[i,:Ln]=e.ids[:Ln]
    st=recompute_states(ids).astype(np.float16)
    for i,ri in enumerate(idxs): out[ri]=st[i]
out.flush(); del out
samples, ids2, mask, offsets = PH.tokenize('.cache/tail_mix.jsonl')
gold=build_gold(samples, offsets)
sent=np.stack([sent_indices(s["text"],o,mask[i]) for i,(s,o) in enumerate(zip(samples,offsets))])
np.savez('.cache/phase1_alg_states_tail.npz', tokmask=mask.astype(np.uint8),
         sent=sent.astype(np.int8), **{f"g_{k}":v for k,v in gold.items()})
print("[aim] states + gold staged (WIDE era)", flush=True)
