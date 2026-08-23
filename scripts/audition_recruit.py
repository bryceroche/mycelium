"""audition_recruit.py — THE RECRUITER (the covering loop's hire step).
Phase 1 (GPU, sequential): audition_read_one per done candidate ->
.cache/audition_{ID}.json. Phase 2 (CPU): failure-vector disagreement
matrix -> greedy max-min hire under the competence floor -> the
UNCOVERED RESIDUE report (round 2's design input). The diversity fence
holds: hiring reads MEASURED behavior only.
"""
import os, sys, json, glob, subprocess
import numpy as np

def main():
    pool = json.load(open('docs/reader_pool.json'))
    done = [r["id"] for r in pool if os.path.exists(f'.cache/pool_{r["id"]}.done')]
    print(f"[recruit] candidates done: {len(done)}", flush=True)
    for cid in done:
        if not os.path.exists(f'.cache/audition_{cid}.json'):
            r = subprocess.run([".venv/bin/python3", "scripts/audition_read_one.py"],
                               env={**os.environ, "CAND_ID": cid},
                               capture_output=True, text=True)
            print(r.stdout.strip().split("\n")[-1] if r.stdout else
                  f"[audition {cid}] FAILED: {r.stderr[-200:]}", flush=True)
    reads = {}
    for cid in done:
        try:
            reads[cid] = json.load(open(f'.cache/audition_{cid}.json'))["rows"]
        except Exception:
            pass
    ids = sorted(reads)
    if len(ids) < 2:
        print("[recruit] not enough auditions"); return
    n = len(reads[ids[0]])
    # behavior vector: per row, the emitted answer (None = refuse)
    beh = {c: [r["got"] for r in reads[c]] for c in ids}
    rights = {c: sum(1 for r in reads[c] if r["got"] == r["key"]) for c in ids}
    floor = max(2, int(np.median(list(rights.values()))) // 2)
    eligible = [c for c in ids if rights[c] >= floor]
    print(f"[recruit] competence floor {floor}: {len(eligible)}/{len(ids)} eligible", flush=True)
    def dis(a, b):
        return sum(1 for x, y in zip(beh[a], beh[b]) if x != y) / n
    D = {(a, b): dis(a, b) for a in eligible for b in eligible if a < b}
    # greedy max-min hire
    if not D: print("[recruit] no pairs"); return
    first = max(eligible, key=lambda c: rights[c])
    hires = [first]
    while len(hires) < min(16, len(eligible)):
        best = max((c for c in eligible if c not in hires),
                   key=lambda c: min(D.get((min(c, h2), max(c, h2)), 0)
                                     for h2 in hires))
        hires.append(best)
    print(f"[recruit] HIRED {len(hires)}: {hires}", flush=True)
    # the uncovered residue: rows no hire reads right
    resid = [i for i in range(n)
             if not any(reads[c][i]["got"] == reads[c][i]["key"] for c in hires)]
    union_right = n - len(resid)
    tags = {}
    for i in resid:
        tags[reads[ids[0]][i]["tag"]] = tags.get(reads[ids[0]][i]["tag"], 0) + 1
    print(f"[recruit] UNION-RIGHT {union_right}/{n}  RESIDUE {len(resid)} "
          f"(by tag: {tags})", flush=True)
    json.dump({"hires": hires, "rights": rights,
               "matrix": {f"{a}|{b}": v for (a, b), v in D.items()},
               "residue_rows": resid, "union_right": union_right},
              open('.cache/recruit_round1.json', 'w'))

if __name__ == "__main__":
    main()
