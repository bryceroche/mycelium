"""fingerpost.py — THE CONSENSUS GATE (the instantia crucis, coded).
Pure CPU over audition jsons (which carry answers + canonical graph
digests). Per row: collect (digest, answer) from solving readers;
EMIT iff >=MIN_WITNESS readers share an identical digest AND span
>=MIN_AXES distinct lineages (the unrelated-witnesses clause — same-
lineage readers count once); else REFUSE. Reports rights/LIES/refusals
vs keys: the zero-lie read at ensemble grain. Mouth precondition is
assumed upstream (fixtures are register-appropriate by construction).
Env: FP_HIRES (comma ids; default = recruit_round1.json hires),
FP_MIN_W (3), FP_MIN_AXES (2).
"""
import os, json
from collections import defaultdict

def main():
    if os.environ.get("FP_HIRES"):
        hires = os.environ["FP_HIRES"].split(",")
    else:
        hires = json.load(open('.cache/recruit_round1.json'))["hires"]
    pool = {r["id"]: r for r in json.load(open('docs/reader_pool.json'))}
    reads = {c: json.load(open(f'.cache/audition_{c}.json'))["rows"]
             for c in hires}
    n = len(reads[hires[0]])
    MIN_W = int(os.environ.get("FP_MIN_W", "3"))
    MIN_AX = int(os.environ.get("FP_MIN_AXES", "2"))
    emit = right = lies = refuse = 0
    by_tag = defaultdict(lambda: [0, 0, 0])
    for i in range(n):
        groups = defaultdict(list)
        for c in hires:
            row = reads[c][i]
            if row["got"] is not None and row["dig"] is not None:
                groups[(row["dig"], row["got"])].append(c)
        verdict = None
        for (dig, got), members in sorted(groups.items(),
                                          key=lambda kv: -len(kv[1])):
            axes = set(pool[c]["lineage"] for c in members)
            if len(members) >= MIN_W and len(axes) >= MIN_AX:
                verdict = got; break
        tag = reads[hires[0]][i]["tag"]; key = reads[hires[0]][i]["key"]
        if verdict is None:
            refuse += 1; by_tag[tag][2] += 1
        else:
            emit += 1
            if verdict == key: right += 1; by_tag[tag][0] += 1
            else: lies += 1; by_tag[tag][1] += 1
    print(f"[fingerpost] n={n} EMIT {emit} (right {right} LIES {lies}) "
          f"REFUSE {refuse}   [witness>={MIN_W}, axes>={MIN_AX}]", flush=True)
    for t, (r, l, f) in sorted(by_tag.items()):
        print(f"  {t}: right {r} lies {l} refuse {f}", flush=True)

if __name__ == "__main__":
    main()
