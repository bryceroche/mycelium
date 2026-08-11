"""gold_restage3.py — rebuild form3 gold npz WITH ALG_REF (g_refvar in;
states untouched; sha stamp riding)."""
import sys, os, json
os.environ["ALG_REF"]="1"
for k,v in [("ALG2","1"),("ALG_FTYPES","8"),("ALG_HW","512"),("ALG_DUP","1"),("ALG_WIDE","1")]: os.environ.setdefault(k,v)
sys.path.insert(0,'.'); sys.path.insert(0,'scripts')
import numpy as np
from phase1_algebra_head import T_ALG, sent_indices, build_gold
import phase1_algebra_head as PH
from mycelium.era import mix_sha16
samples, ids2, mask, offsets = PH.tokenize('.cache/form_mix3.jsonl')
gold=build_gold(samples, offsets)
assert "refvar" in gold and (gold["refvar"]>=0).sum()>50000, "refvar gold thin"
sent=np.stack([sent_indices(s["text"],o,mask[i]) for i,(s,o) in enumerate(zip(samples,offsets))])
np.savez('.cache/phase1_alg_states_form3.npz', tokmask=mask.astype(np.uint8),
         sent=sent.astype(np.int8), mix_sha=mix_sha16('.cache/form_mix3.jsonl'),
         **{f"g_{k}":v for k,v in gold.items()})
print(f"[restage] form3 gold + refvar ({int((gold['refvar']>=0).sum())} site-tokens) + sha")
