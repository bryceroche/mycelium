import sys, os, json
sys.path.insert(0,'.'); sys.path.insert(0,'scripts')
os.environ.setdefault("ALG2","1"); os.environ.setdefault("ALG_FTYPES","8")
os.environ.setdefault("ALG_HW","512"); os.environ.setdefault("ALG_DUP","1")
os.environ.setdefault("ALG_WIDE","1")  # manifest era — the 3-vs-7 lesson
import numpy as np
from phase1_algebra_head import T_ALG, sent_indices, TOKENIZER_JSON, build_gold
import phase1_algebra_head as PH
from beacon_closing_arm import recompute_states
from tokenizers import Tokenizer
tok = Tokenizer.from_file(TOKENIZER_JSON)
mixp=".cache/augfire_vdup19_mix.jsonl"
rows=[json.loads(l) for l in open(mixp)]
changed=json.load(open(".cache/augfire_vdup19_changed.json"))
samples, ids2, mask, offsets = PH.tokenize(mixp)
gold=build_gold(samples, offsets)
sent=np.stack([sent_indices(s["text"],o,mask[i]) for i,(s,o) in enumerate(zip(samples,offsets))])
np.savez(".cache/phase1_alg_states_vdup19.npz", tokmask=mask.astype(np.uint8),
         sent=sent.astype(np.int8), **{f"g_{k}":v for k,v in gold.items()})
print("[rebuild] gold npz re-staged at WIDE era", flush=True)
