"""THE VALUE ALIGNER v2 (2026-09-14): the numeral is the GIVEN FACTOR's span (its
slot's attention target), not the variable's mention (v1's error: the variable is
the quantity; the numeral is its value). For every given (value x) in a gsm8k pen
row, x as a whole number in the text -> f["spans"] = [[s, e], ...]; build_gold's
fspan for that slot (and valspan under ALG_VALATT) become the numeral's tokens.
Relation factors stay unanchored. usage: value_aligner_v2.py src.jsonl dst.jsonl"""
import sys, json, re
src, dst = sys.argv[1], sys.argv[2]
n_pen = n_touch = n_giv = n_al = 0
with open(dst, "w") as out:
    for l in open(src):
        d = json.loads(l); g = d.get("gen")
        if isinstance(g, dict) and g.get("src") == "gsm8k" and not d.get("mentions"):
            n_pen += 1; t = d.get("text") or ""; touched = False
            for f in d.get("factors", []):
                if f.get("ftype") != "given" or "value" not in f or f.get("spans"): continue
                n_giv += 1; x = f["value"]
                if float(x) != int(float(x)): continue
                xi = int(float(x)); spans = []
                for a in {str(xi), f"{xi:,}"}:
                    for m in re.finditer(r"(?<![\d.,])\$?" + re.escape(a) + r"(?:%|(?=[^\d.,]|$))", t):
                        s, e = m.start(), m.end()
                        if t[s] == "$": s += 1
                        if t[e - 1] == "%": e -= 1
                        spans.append([s, e])
                if spans:
                    f["spans"] = sorted({tuple(x) for x in spans}); f["spans"] = [list(x) for x in f["spans"]]; n_al += 1; touched = True
            if touched:
                n_touch += 1; d.setdefault("gen", {})["aligned"] = "value-aligner-v2-2026-09-14"
        out.write(json.dumps(d) + "\n")
print(f"[value-aligner-v2] pen rows {n_pen}, touched {n_touch}; givens {n_giv}, anchored as factor spans {n_al} ({n_al / max(n_giv, 1):.2f}) -> {dst}")
