"""THE VALUE ALIGNER (2026-09-13, the order of attack): the gsm8k pen rows carry
factor gold but no spans. For every `given` (var v, value x), find x as a whole
number in the text (digits; optional thousands commas; optional leading $;
optional trailing %) and write mentions[v] = its char spans — the given's
variable is named by its value in prose. Rows/values without a digit match
(word numbers, derived values) are left as they were. Writes a NEW jsonl beside
the source and prints the coverage. usage: value_aligner.py src.jsonl dst.jsonl"""
import sys, json, re
src, dst = sys.argv[1], sys.argv[2]
n_rows = n_pen = n_pen_touched = n_giv = n_al = n_multi = 0
with open(dst, "w") as out:
    for l in open(src):
        d = json.loads(l); n_rows += 1
        g = d.get("gen"); pen = isinstance(g, dict) and g.get("src") == "gsm8k" and not d.get("mentions")
        if pen:
            n_pen += 1; t = d.get("text") or ""; ment = {}
            for f in d.get("factors", []):
                if f.get("ftype") != "given" or "value" not in f: continue
                n_giv += 1; x = f["value"]
                if isinstance(x, float) and x != int(x): continue
                xs = str(int(x)); alts = {xs, f"{int(x):,}"}
                spans = []
                for a in alts:
                    for m in re.finditer(r"(?<![\d.,])\$?" + re.escape(a) + r"(?:%|(?=[^\d.,]|$))", t):
                        s, e = m.start(), m.end()
                        if t[s] == "$": s += 1
                        if t[e - 1] == "%": e -= 1
                        spans.append([s, e])
                if spans:
                    n_al += 1; n_multi += int(len(spans) > 1)
                    ment.setdefault(str(f["var"]), []).extend(sorted(set(map(tuple, spans))) and [list(x) for x in sorted(set(map(tuple, spans)))])
            if ment:
                d["mentions"] = ment; n_pen_touched += 1
                d.setdefault("gen", {})["aligned"] = "value-aligner-2026-09-13"
        out.write(json.dumps(d) + "\n")
print(f"[value-aligner] rows {n_rows}; pen rows {n_pen}, touched {n_pen_touched} ({n_pen_touched / max(n_pen, 1):.2f}); givens {n_giv}, aligned {n_al} ({n_al / max(n_giv, 1):.2f}), multi-occurrence {n_multi} -> {dst}")
