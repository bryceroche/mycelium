"""gsm8k_chain_post.py — the drafts of gsm8k_wild_extract.py into the silver row form (2026-09-17):
text := the question; answer_field := "#### key"; every given gets the SPAN of its numeral in the text
(first unused word-bounded occurrence — the extractor's anchor law guarantees one); gen stamped
src gsm8k / silver chain_v2. Rows whose given cannot be located are refused (counted).
usage: gsm8k_chain_post.py drafts.jsonl out.jsonl"""
import json, re, sys
rows = [json.loads(l) for l in open(sys.argv[1])]; out = []; ref = 0
for r in rows:
    text = r["original"]; used = []; ok = True; facs = []
    for f in r["factors"]:
        f = dict(f)
        if f["ftype"] == "given" and f.get("spans"):   # tier-2 lexicon givens carry their word span already
            used.append(tuple(f["spans"][0])); facs.append(f); continue
        if f["ftype"] == "given":
            v = str(f["value"]); hit = None
            for m in re.finditer(r"(?<![\w.])" + re.escape(v) + r"(?![\w]|\.\d)", text):
                if not any(a <= m.start() < b for a, b in used): hit = (m.start(), m.end()); break
            if hit is None:   # numerals with thousands separators
                for m in re.finditer(r"(?<![\w.])" + re.escape(f"{f['value']:,}") + r"(?![\w]|\.\d)", text):
                    if not any(a <= m.start() < b for a, b in used): hit = (m.start(), m.end()); break
            if hit is None: ok = False; break
            used.append(hit); f["spans"] = [list(hit)]
        facs.append(f)
    if not ok: ref += 1; continue
    out.append({"text": text, "factors": facs, "n_vars": r["n_vars"], "query_var": r["query_var"], "m": r.get("m", 10000), "mentions": {}, "decisions": 0, "solution": [],
                "answer_field": f"#### {r['answer']}", "key": int(r["answer"]), "gen": {"src": "gsm8k", "src_idx": r.get("src_idx"), "silver": "chain_v2", "tier": ("lexicon" if any(x.get("hidden") for x in facs) else "anchored"), "admitted_by": "gsm8k_wild_extract:key-propagation"}})
with open(sys.argv[2], "w") as f:
    for r in out: f.write(json.dumps(r) + "\n")
print(f"[chain-post] {len(rows)} drafts -> {len(out)} rows with given spans ({ref} refused: a given's numeral not located)")
