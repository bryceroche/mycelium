"""bank_enriched_fixture.py — CPU pass: re-derive per-item injected facts
from banked graphs, bank the named fixture roundtrip_enriched.jsonl
(annotated texts + facts + no-facts ids). Zero GPU."""
import sys, json, re
sys.path.insert(0, "."); sys.path.insert(0, "scripts")
from collections import Counter
from roundtrip_read import forced_facts, letters_of, rows, gold, lat, bank
abst = []
for i, v in enumerate(lat):
    nn = [a for a in v if a is not None]
    c = Counter(nn).most_common(1)
    if (c[0][1] if c else 0) < 3: abst.append(i)
out = open(".cache/roundtrip_enriched.jsonl", "w")
nf = []
for i in abst:
    r = rows[i]; m = r.get("m", 60); letters = letters_of(r["text"])
    facts = {}
    for v in bank[i]:
        for L, val in forced_facts(v["factors"], r["text"], m, letters):
            facts[L] = val
    if not facts: nf.append(i)
    ann = r["text"] + (" " + " ".join(f"It is known that {L} is {val}." for L, val in sorted(facts.items())) if facts else "")
    out.write(json.dumps({"item": i, "text_enriched": ann, "facts": facts, "gold": gold[i]}) + "\n")
out.close()
print(f"[fixture] {len(abst)} rows banked -> .cache/roundtrip_enriched.jsonl | no-facts ids ({len(nf)}): {nf}")
