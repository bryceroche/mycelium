"""gen_deep_ladders.py — THE DEPTH FRONTIER's corpora (2026-09-01, word).
(1) TRAIN: 24k ladders, depth UNIFORM 2..16 (deep rungs favor forward
form to fit 24 vars) -> .cache/ladder_deep24k.jsonl + form_mix10 =
form_mix8 + deep ladders (share ~20%, dose declared).
(2) EVAL: the depth-curve fixture — 150 rows per depth bucket
{2,4,6,8,10,12,14,16}, HELD-OUT seed -> .cache/deepval.jsonl.
Both through the standard roundtrip/uniqueness gate; density asserted.
"""
import json
import random
import sys

sys.path.insert(0, 'scripts')
from algebra_nl_gen import gen_ladder, render, roundtrip

M = 300


def make(rng, depth):
    inv = 0.25 if depth <= 8 else 0.10   # deep rows lean forward to fit 24
    n_vars, factors, sol, query = gen_ladder(rng, depth, M, inverse_p=inv)
    if n_vars > 24:
        return None
    text, gfactors, mentions = render(rng, n_vars, factors, query)
    ok, decisions = roundtrip(n_vars, gfactors, M, sol)
    if not ok:
        return None
    return {"n_vars": n_vars, "m": M, "text": text, "factors": gfactors,
            "mentions": mentions, "query_var": query, "solution": sol,
            "decisions": decisions, "gen": {"ladder": depth}}


def density(rows):
    ds = []
    for r in rows:
        giv = {f["var"] for f in r["factors"] if f["ftype"] == "given"}
        slots = [set(f.get("args", [])) | {f.get("result")}
                 for f in r["factors"] if f["ftype"] == "rel"]
        det, fired = set(giv), 0
        for _ in range(20):
            for vv in slots:
                unk = [v for v in vv if v not in det]
                if len(unk) == 1:
                    det.add(unk[0]); fired += 1
        ds.append(fired / max(len(slots), 1))
    return sum(ds) / len(ds)


# --- train corpus ---
rng = random.Random(1601)
train, rej = [], 0
while len(train) < 24000:
    d = rng.randint(2, 16)
    r = make(rng, d)
    if r is None:
        rej += 1
        continue
    train.append(r)
dtr = density(train)
assert dtr >= 0.85, dtr
with open('.cache/ladder_deep24k.jsonl', 'w') as fh:
    for r in train:
        fh.write(json.dumps(r) + "\n")
depths = {}
for r in train:
    depths[r["gen"]["ladder"]] = depths.get(r["gen"]["ladder"], 0) + 1
print(f"[deep-train] 24000 rows ({rej} rej), density {dtr:.3f}, "
      f"depths {dict(sorted(depths.items()))}")
base = [l for l in open('.cache/form_mix8.jsonl')]
mix = base + [json.dumps(r) + "\n" for r in train]
random.Random(13).shuffle(mix)
with open('.cache/form_mix10.jsonl', 'w') as fh:
    fh.writelines(mix)
print(f"[mix] form_mix10: {len(mix)} rows; deep-ladder share "
      f"{24000 / len(mix) * 100:.1f}%")

# --- eval fixture (held-out seed) ---
rng = random.Random(7707)
ev, rej = [], 0
for d in (2, 4, 6, 8, 10, 12, 14, 16):
    got = 0
    while got < 150:
        r = make(rng, d)
        if r is None:
            rej += 1
            continue
        r["gen"]["bucket"] = d
        ev.append(r); got += 1
dev = density(ev)
assert dev >= 0.85, dev
with open('.cache/deepval.jsonl', 'w') as fh:
    for r in ev:
        fh.write(json.dumps(r) + "\n")
print(f"[deepval] {len(ev)} rows ({rej} rej), density {dev:.3f} — the "
      f"depth-curve fixture (held seed 7707)")
