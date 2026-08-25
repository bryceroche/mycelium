"""opc_read.py — LEVER 3 POST-FIRE READ (2026-08-25, word given; bars
PINNED pre-fire): the count head's op-multisets vs the chain baseline.
BARS: KILL if exact-row rate <= 6/143 (the chain-decode baseline, banked
pre-fire); program-advance >= 15/143. Rescue variant (single, registered):
fst-pool instead of waist-pool. Downstream: enumeration on opc multisets
vs chain baselines (wv 4/16, held 2/16 coverage; lies dominated unique).
Phase A (GPU): gsb227_opc opc argmax counts on 143 golds (grade) + the
180-row fixture (bank -> .cache/opc_ops.json). Phase B (CPU): enum.
"""
import os, sys, json, re, glob
os.environ.setdefault("ALG_MINE_BREATHS", "0")
os.environ.update({"DEV": "AMD", "ALG2": "1", "ALG_FTYPES": "9",
                   "ALG_DUP": "1", "ALG_HW": "512", "ALG_WIDE": "1",
                   "ALG_BREATH": "7", "ALG_NOTEBOOK": "1", "ALG_SIXWAVE": "1",
                   "ALG_OPCOUNT": "1",
                   "ALG_TEST": ".cache/algebra_nl_bigtest.jsonl",
                   "ALG_TEST_NAME": "bigtest"})
sys.path.insert(0, '.'); sys.path.insert(0, 'scripts')
import numpy as np
from collections import Counter
from phase1_algebra_head import (build_params, forward, T_ALG, TOKENIZER_JSON,
                                 sent_indices, load_alg, OPC_CLASSES, _opc_meta)
from tokenizers import Tokenizer
from tinygrad import Tensor, dtypes
from tinygrad.nn.state import safe_load
from beacon_closing_arm import recompute_states
from enum_assembly import reachable

_ = load_alg("test")
tok = Tokenizer.from_file(TOKENIZER_JSON)
NUM = re.compile(r"(?<![\d.])(\d+)(?![\d.])")

def corpus143():
    byid = {}
    for f in sorted(glob.glob('.cache/book*_t*_batch*.jsonl')):
        for l in open(f):
            r = json.loads(l); byid[r["src_idx"]] = r
    for l in open('.cache/book12_anchor_batch1.jsonl'):
        r = json.loads(l); byid[r["src_idx"]] = r
    sk = set(json.load(open('.cache/book12_anchor_skips.json')))
    return [v for k, v in sorted(byid.items()) if k not in sk]

def decode_counts(p, rows):
    out = []
    for s0 in range(0, len(rows), 8):
        sl = rows[s0:s0 + 8]
        ids = np.zeros((8, T_ALG), np.int32)
        msk = np.zeros((8, T_ALG), np.float32)
        snt = np.zeros((8, T_ALG), np.int32)
        for li, r in enumerate(sl):
            e = tok.encode(r["original"])
            if len(e.ids) > T_ALG: continue
            ids[li, :len(e.ids)] = e.ids; msk[li, :len(e.ids)] = 1.0
            snt[li] = sent_indices(r["original"], list(e.offsets), msk[li])
        sts = np.asarray(recompute_states(ids)).astype(np.float32)
        o = forward(p, Tensor(sts, dtype=dtypes.float),
                    Tensor(msk, dtype=dtypes.float),
                    Tensor(snt.astype(np.int32), dtype=dtypes.int))
        opc = o["opc"].realize().numpy()          # (8, n_cls, cap+1)
        for li in range(len(sl)):
            cnt = opc[li].argmax(-1)
            out.append(Counter({c: int(k) for c, k in zip(OPC_CLASSES, cnt)
                                if k > 0}))
    return out

def main():
    p = build_params(0)
    sd = safe_load('.cache/gsb227_opc.safetensors')
    assert set(sd.keys()) == set(p.keys()), \
        f"gsb227_opc key mismatch: {len(set(sd)-set(p))}/{len(set(p)-set(sd))}"
    for k in p:
        p[k].assign(sd[k].to(p[k].device).cast(p[k].dtype)).realize()

    gold = corpus143()
    dec = decode_counts(p, gold)
    ex = 0; f1s = []
    for d, r in zip(dec, gold):
        g = Counter({c: k for c, k in zip(OPC_CLASSES, _opc_meta(r)) if k > 0})
        if d == g: ex += 1
        inter = sum((d & g).values())
        f1s.append(2 * inter / max(sum(d.values()) + sum(g.values()), 1))
    print(f"[opc] EXACT-ROW multiset rate: {ex}/{len(gold)} "
          f"(chain baseline 6/143; KILL <=6; advance >=15)", flush=True)
    print(f"[opc] F1 mean {np.mean(f1s):.3f} median {np.median(f1s):.3f} "
          f"(chain 0.550/0.571)", flush=True)

    aro = open('scripts/audition_read_one.py').read()
    ns = {"json": json, "glob": glob, "np": np}
    exec(aro[aro.index("def fixtures():"):aro.index("rows = fixtures()")], ns)
    rows = ns["fixtures"]()
    fdec = decode_counts(p, rows)
    json.dump([{"tag": r["tag"], "ops": sorted(d.elements())}
               for r, d in zip(rows, fdec)], open('.cache/opc_ops.json', 'w'))
    T = {t: {"n": 0, "go": 0, "cover": 0, "uniq": 0, "ur": 0, "ul": 0}
         for t in ("wv", "held", "cen")}
    for r, d in zip(rows, fdec):
        tag = r["tag"]
        if tag.startswith("anc"): continue
        tag = tag if tag in T else "cen"
        t = T[tag]; t["n"] += 1
        cops = sorted(d.elements())
        ops = [l for l in cops if l in ("add", "sub", "mul", "sq", "fr")]
        if len(ops) != len(cops) or not ops: continue
        nums = [int(m.group(1)) for m in NUM.finditer(r["original"])]
        if not nums or len(nums) > 8 or len(ops) > 6: continue
        t["go"] += 1
        roots, _ = reachable(nums, ops)
        key = r["answer"]
        if key in roots: t["cover"] += 1
        if len(roots) == 1:
            t["uniq"] += 1
            if key in roots: t["ur"] += 1
            else: t["ul"] += 1
    for tag, t in T.items():
        print(f"[opc {tag}] n={t['n']} enum-on {t['go']} coverage "
              f"{t['cover']}/{t['go']} unique {t['uniq']} "
              f"(right {t['ur']} lies {t['ul']})", flush=True)

if __name__ == "__main__":
    main()
