"""eq_sets_extract.py — SVAMP and ASDiv into the dialect (2026-09-17, word given). Each problem carries an
equation over the text's numbers; a shunting-yard parse gives binary steps; givens anchor to a numeral in
the text (or a lexicon word span: tier "lexicon"), sub/div re-encode per the grammar (x-y=d -> add(y,d)=x;
a/b=c -> mul(c,b)=a; the unknown as an argument), every value an integer in [0, M], the propagated query
must equal the dataset's answer. Rows in the silver form with given spans, gen.src = svamp | asdiv.
usage: eq_sets_extract.py out.jsonl"""
import json, re, sys, collections
sys.path.insert(0, "."); from mycelium import lexicon as L
M = 9999
TOK = re.compile(r"\d+\.?\d*|[-+*/()]")

def to_rpn(expr):
    prec = {"+": 1, "-": 1, "*": 2, "/": 2}; out = []; st = []
    for t in TOK.findall(expr):
        if t[0].isdigit(): out.append(float(t))
        elif t == "(": st.append(t)
        elif t == ")":
            while st and st[-1] != "(": out.append(st.pop())
            if not st: raise ValueError("paren")
            st.pop()
        else:
            while st and st[-1] != "(" and prec[st[-1]] >= prec[t]: out.append(st.pop())
            st.append(t)
    while st:
        if st[-1] == "(": raise ValueError("paren")
        out.append(st.pop())
    return out

def build(text, expr, key):
    nums = {int(x.replace(",", "")) for x in re.findall(r"\d[\d,]*", text) if float(x.replace(",", "")) == int(float(x.replace(",", "")))}
    lex = L.constants(text); lexused = []; facs = []; by_val = {}; nv = [0]
    def new_var(): nv[0] += 1; return nv[0] - 1
    def given(v):
        if v in by_val: return by_val[v]
        if v != int(v) or not (0 <= v <= M): return None
        v = int(v)
        if v == 1 or v in nums:
            i = new_var(); f = {"ftype": "given", "var": i, "value": v}
            for m in re.finditer(r"(?<![\w.])" + re.escape(str(v)) + r"(?![\w]|\.\d)", text):   # the numeral's span (first unused) — the resampler needs it
                if not any(a <= m.start() < b for a, b in lexused): lexused.append((m.start(), m.end())); f["spans"] = [[m.start(), m.end()]]; break
            facs.append(f); by_val[v] = i; return i
        for s, e, val in lex:
            if val == v and not any(a <= s < b for a, b in lexused):
                lexused.append((s, e)); i = new_var(); facs.append({"ftype": "given", "var": i, "value": v, "spans": [[s, e]], "hidden": "lexicon"}); by_val[v] = i; return i
        return None
    st = []
    for t in to_rpn(expr):
        if isinstance(t, float): st.append(("num", t)); continue
        b = st.pop(); a = st.pop()
        va = a[1] if a[0] == "var" else given(a[1]); vb = b[1] if b[0] == "var" else given(b[1])
        if va is None or vb is None: return None, "unanchored"
        x = a[2] if a[0] == "var" else a[1]; y = b[2] if b[0] == "var" else b[1]
        r = {"+": x + y, "-": x - y, "*": x * y, "/": (x / y if y else None)}[t]
        if r is None or r != int(r) or not (0 <= r <= M): return None, "non-integer"
        i = new_var()
        if t == "+": facs.append({"ftype": "rel", "op": "add", "args": [va, vb], "result": i})
        elif t == "*": facs.append({"ftype": "rel", "op": "mul", "args": [va, vb], "result": i})
        elif t == "-": facs.append({"ftype": "rel", "op": "add", "args": [vb, i], "result": va})   # x - y = d  ->  add(y, d) = x
        else: facs.append({"ftype": "rel", "op": "mul", "args": [vb, i], "result": va})           # a / b = c  ->  mul(c, b) = a
        st.append(("var", i, int(r)))
    if len(st) != 1 or st[0][0] != "var": return None, "expr"
    if st[0][2] != key: return None, "key-mismatch"
    if nv[0] > 24 or len(facs) > 24: return None, "capacity"
    return {"factors": facs, "n_vars": nv[0], "query_var": st[0][1]}, None

def rows_svamp():
    for p in json.load(open(".cache/eqsets/SVAMP.json")):
        yield (p["Body"].strip() + " " + p["Question"].strip()).replace("  ", " "), p["Equation"], p["Answer"], "svamp"
def rows_asdiv():
    import xml.etree.ElementTree as ET
    for pr in ET.parse(".cache/eqsets/ASDiv.xml").getroot().iter("Problem"):
        f = pr.findtext("Formula") or ""; a = (pr.findtext("Answer") or "").split(" ")[0]
        yield (pr.findtext("Body") or "").strip() + " " + (pr.findtext("Question") or "").strip(), f.split("=")[0], a, "asdiv"

if __name__ == "__main__":
    out = []; rej = collections.Counter(); n = 0
    for text, eq, ans, src in list(rows_svamp()) + list(rows_asdiv()):
        n += 1
        try: key = float(str(ans).replace(",", "")); key = int(key) if key == int(key) else None
        except Exception: key = None
        if key is None or not (0 <= key <= M): rej["answer"] += 1; continue
        try: d, why = build(text, eq, key)
        except Exception as e: d, why = None, "expr"
        if d is None: rej[why] += 1; continue
        out.append({"text": text, "factors": d["factors"], "n_vars": d["n_vars"], "query_var": d["query_var"], "m": 10000, "mentions": {}, "decisions": 0, "solution": [],
                    "answer_field": f"#### {key}", "key": key, "gen": {"src": src, "silver": "eqsets_v1", "tier": ("lexicon" if any(f.get("hidden") for f in d["factors"]) else "anchored"), "admitted_by": "eq_sets_extract:key-propagation"}})
    with open(sys.argv[1], "w") as f:
        for r in out: f.write(json.dumps(r) + "\n")
    print(f"[eqsets] {n} problems -> {len(out)} rows ({collections.Counter(r['gen']['src'] for r in out)}); rejects {dict(rej.most_common())}")
