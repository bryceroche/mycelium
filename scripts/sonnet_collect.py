"""sonnet_collect.py — join Sonnet's returns to the queue rows, run THE ADMISSION
GATE (scripts/admit_annotation.py: the rulebook + the solver forcing the key),
write the admitted silver split (versioned apart) and the human-audit sample.
usage: sonnet_collect.py returns.jsonl [version=sonnet_v1] [audit_frac=0.1]"""
import json, random, sys, os
sys.path.insert(0, "."); sys.path.insert(0, "scripts")
from admit_annotation import admit, key_of
ret_path = sys.argv[1]; version = sys.argv[2] if len(sys.argv) > 2 else "sonnet_v1"; frac = float(sys.argv[3]) if len(sys.argv) > 3 else 0.1
pk = json.load(open(os.environ.get("PACKET", ".cache/annotation_packet_top300.json"))); by_id = {r["id"]: r for r in pk["rows"]}
rets = []; refused = {}
for line in open(ret_path):
    line = line.strip()
    if not line: continue
    try: o = json.loads(line)
    except Exception: refused["unparseable"] = refused.get("unparseable", 0) + 1; continue
    if "refuse" in o: refused["sonnet:" + str(o["refuse"])[:40]] = refused.get("sonnet:" + str(o["refuse"])[:40], 0) + 1; continue
    r = by_id.get(o.get("id"))
    if r is None: refused["unknown_id"] = refused.get("unknown_id", 0) + 1; continue
    # spans and mentions arrive as EXACT SUBSTRINGS; convert to offsets (every occurrence for mentions,
    # the first for a factor's span); a substring not found is dropped (the gate sees the row without it)
    def offs(text, subs, all_occ):
        out = []
        for sub in subs or []:
            if not isinstance(sub, str) or not sub: continue
            i = text.find(sub)
            while i >= 0:
                out.append([i, i + len(sub)])
                if not all_occ: break
                i = text.find(sub, i + 1)
        return out
    facs = []
    for f in (o.get("factors") or []):
        f = dict(f)
        if isinstance(f.get("spans"), list) and f["spans"] and isinstance(f["spans"][0], str): f["spans"] = offs(r["text"], f["spans"], False)
        facs.append(f)
    ments = {k: (offs(r["text"], v, True) if v and isinstance(v[0], str) else v) for k, v in (o.get("mentions") or {}).items()}
    rets.append({"text": r["text"], "answer_field": r["answer_field"], "n_vars": o.get("n_vars"), "query_var": o.get("query_var"), "factors": facs, "mentions": ments, "decisions": 0, "solution": [],
                 "gen": {"src": "gsm8k", "pool": "queue_top300", "queue_rank": r.get("queue_rank"), "fingerpost_unstable": r.get("fingerpost_unstable")}})
adm, why = admit(rets, version)
random.Random(0).shuffle(adm); n_audit = max(1, int(round(frac * len(adm)))) if adm else 0
with open(f".cache/silver_{version}.jsonl", "w") as f:
    for r in adm: f.write(json.dumps(r) + "\n")
with open(f".cache/silver_{version}_audit.jsonl", "w") as f:
    for r in adm[:n_audit]: f.write(json.dumps(r) + "\n")
print(f"[collect] returns {len(rets)} annotated + {sum(refused.values())} refused/unusable {refused}")
print(f"[collect] ADMITTED {len(adm)} / {len(rets)} -> .cache/silver_{version}.jsonl (silver, versioned apart); gate refusals {why}; audit sample {n_audit} -> .cache/silver_{version}_audit.jsonl")
print("[collect] the answer key gated every admitted row; nothing here is gold; a human audits the sample before any diet mix is declared (dose law: share-of-mix AND reps-per-unique)")
