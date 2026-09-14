"""THE OPERATOR LEXICON, rung 1 / THE RELATION-CUE ALIGNER v3 (2026-09-14): on
a gsm8k pen row, an operator cue phrase in the text becomes the RELATION
factor's span — but only when the assignment is unambiguous: the row has
exactly one relation of the cue's class (add-class cues: total, altogether,
in all, combined, sum of, more than, plus, added; mul-class cues: each, per,
times as, twice, double, triple, times) and exactly one cue of that class in
the text. The span is the cue phrase itself. Given factors keep their v2
numeral spans (input = the v2 file). usage: relation_cue_aligner.py src dst"""
import sys, json, re
ADD = ["altogether", "in all", "in total", "total", "combined", "sum of", "more than", "plus", "added", "together"]
MUL = ["times as many", "times as much", "times as", "each", "per", "twice", "double", "triple", "times"]
src, dst = sys.argv[1], sys.argv[2]
n_pen = n_rows = n_rel = n_al = 0; by_cue = {}
with open(dst, "w") as out:
    for l in open(src):
        d = json.loads(l); g = d.get("gen")
        if isinstance(g, dict) and g.get("src") == "gsm8k":
            n_pen += 1; t = (d.get("text") or ""); tl = t.lower(); touched = False
            rels = [f for f in d.get("factors", []) if f.get("ftype") == "rel" and not f.get("spans")]
            for cls, cues in (("add", ADD), ("mul", MUL)):
                rc = [f for f in rels if f.get("op") == cls]
                if len(rc) != 1: continue
                hits = []
                for c in cues:
                    for m in re.finditer(r"\b" + re.escape(c) + r"\b", tl):
                        hits.append((m.start(), m.end(), c))
                # collapse overlapping cues ("times as many" contains "times")
                hits.sort(); merged = []
                for h in hits:
                    if merged and h[0] < merged[-1][1]: continue
                    merged.append(h)
                if len(merged) != 1: continue
                s, e, c = merged[0]; rc[0]["spans"] = [[s, e]]; n_al += 1; touched = True; by_cue[c] = by_cue.get(c, 0) + 1
            n_rel += len(rels)
            if touched:
                n_rows += 1; d.setdefault("gen", {})["aligned_rel"] = "relation-cue-v3-2026-09-14"
        out.write(json.dumps(d) + "\n")
print(f"[rel-cue] pen rows {n_pen}: rows with >=1 relation anchored {n_rows} ({n_rows / max(n_pen, 1):.2f}); relations {n_rel}, anchored {n_al} ({n_al / max(n_rel, 1):.2f}); by cue: {dict(sorted(by_cue.items(), key=lambda kv: -kv[1]))}")
