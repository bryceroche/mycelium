"""bank mean-pooled trunk states + L for the 143 golds (probe eval side)."""
import os, sys, json, glob
os.environ.update({"DEV": "AMD", "ALG2": "1", "ALG_FTYPES": "9", "ALG_DUP": "1",
                   "ALG_HW": "512", "ALG_WIDE": "1",
                   "ALG_TEST": ".cache/algebra_nl_bigtest.jsonl",
                   "ALG_TEST_NAME": "bigtest"})
sys.path.insert(0, '.'); sys.path.insert(0, 'scripts')
import numpy as np
from phase1_algebra_head import T_ALG, TOKENIZER_JSON, load_alg
from beacon_closing_arm import recompute_states
from tokenizers import Tokenizer
_ = load_alg("test")
tok = Tokenizer.from_file(TOKENIZER_JSON)
byid = {}
for f in sorted(glob.glob('.cache/book*_t*_batch*.jsonl')):
    for l in open(f): r = json.loads(l); byid[r["src_idx"]] = r
for l in open('.cache/book12_anchor_batch1.jsonl'):
    r = json.loads(l); byid[r["src_idx"]] = r
sk = set(json.load(open('.cache/book12_anchor_skips.json')))
rows = [v for k, v in sorted(byid.items()) if k not in sk]
P = np.zeros((len(rows), 2048), np.float32); L = np.zeros(len(rows), np.float32)
for s0 in range(0, len(rows), 8):
    sl = rows[s0:s0 + 8]
    ids = np.zeros((8, T_ALG), np.int32); msk = np.zeros((8, T_ALG), np.float32)
    for li, r in enumerate(sl):
        e = tok.encode(r["original"])
        if len(e.ids) > T_ALG: continue
        ids[li, :len(e.ids)] = e.ids; msk[li, :len(e.ids)] = 1.0
    sts = np.asarray(recompute_states(ids)).astype(np.float32)
    m = msk[:, :, None]
    pool = (sts * m).sum(1) / np.maximum(m.sum(1), 1)
    for li in range(len(sl)):
        P[s0 + li] = pool[li]; L[s0 + li] = msk[li].sum()
np.save('.cache/probe_P_gold143.npy', P); np.save('.cache/probe_L_gold143.npy', L)
print(f"[pgf] banked {len(rows)} gold features", flush=True)
