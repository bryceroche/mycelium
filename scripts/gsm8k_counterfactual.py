"""gsm8k_counterfactual.py -- zero-GPU: THE GSM8K COUNTERFACTUAL FIXTURE (2026-10-23; Bryce's word; ledger 13:24 registration).
Overfit the TEXT, score on COUNTERFACTUALS whose gold is recomputed: a memorizer returns the old intermediates, a binder
recomputes. GSM8K-train only (wild holdout texts excluded); GSM8K-test untouched. The answer field IS read (harvest gold, not
pen-written; no LLM sees any of it).
TRACE      each answer's <<expr=result>> steps -> an executable graph. Every operand is sourced by VALUE: a question literal
           (unique value), a prior result, or a CONSTANT (a frozen literal, a number word in the question, or an implicit
           constant in IMPLICIT). Anything else -> rejected 'unsourced'. Ambiguity (value matches two sources) -> rejected.
           Every step must re-evaluate to its stated result; '####' must equal a trace result.
NUMERALS   every trace-used question literal gets a new value of similar magnitude; the trace is re-executed; a draw is kept
           only if every intermediate keeps the original's type (non-negative; integer where integer; <= 2 decimals where so).
           Times / ordinals / years / fractions / definitional phrases ('60 minutes in an hour') are FROZEN (counted).
NAMES      spaCy PERSON first names -> same-gender gazetteer names (NLTK lists decide gender; both-lists names kept);
           titles and post-title surnames kept; never inside a hyphenated token; applied to question AND answer.
ANCHORS    value memorizer (masked-text key -> ORIGINAL intermediates) and graph memorizer (-> original graph re-executed).
Out: .cache/gsm8k_cf_train.jsonl, .cache/gsm8k_cf_census.{json,txt}, .cache/gsm8k_cf_handcheck40.txt"""
import ast, collections, glob, json, math, operator, os, random, re, sys

import pyarrow.parquet as pq

R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
C = lambda f: os.path.join(R, ".cache", f)
OUTJ, OUTC, OUTT, OUTH = C("gsm8k_cf_train.jsonl"), C("gsm8k_cf_census.json"), C("gsm8k_cf_census.txt"), C("gsm8k_cf_handcheck40.txt")
SMOKE = int(os.environ.get("CF_SMOKE", "0"))            # >0: first N problems, writes nothing
K = 3                                                   # numeral counterfactuals per problem
if not SMOKE:
    for p in (OUTJ, OUTC, OUTT, OUTH):
        assert not os.path.exists(p), f"refuse overwrite {p}"

WORDS = dict(one=1, two=2, three=3, four=4, five=5, six=6, seven=7, eight=8, nine=9, ten=10, eleven=11, twelve=12,
             fifteen=15, twenty=20, thirty=30, forty=40, fifty=50, sixty=60, hundred=100, thousand=1000, twice=2, double=2,
             doubles=2, doubled=2, triple=3, tripled=3, triples=3, thrice=3, half=2, halves=2, quarter=4, quarters=4,
             third=3, thirds=3, dozen=12, dozens=12, pair=2, pairs=2, couple=2, gross=144)
UNITW = dict(week=7, weeks=7, hour=60, hours=60, minute=60, minutes=60, day=24, days=24, year=12, years=12, month=30, months=30,
             percent=100, cents=100, cent=100, dollar=100, dollars=100, feet=12, foot=12, inches=12, yard=3, yards=3, weekend=2)
IMPLICIT = {0, 1, 2, 3, 4, 5, 7, 10, 12, 24, 30, 31, 52, 60, 100, 365, 1000, 0.5, 0.25}   # common implicit constants
UNIT = r"(?:minutes?|hours?|days?|weeks?|months?|years?|seconds?|inches|feet|eggs|items|ounces|cents|pages|cups)"
DEFN = re.compile(rf"\b(\d+)\s+{UNIT}\s+(?:in|per|make up|equals?)\s+(?:a|an|one|each|1|every)\b", re.I)
NUM = re.compile(r"(?<![\w.,/])(\d{1,3}(?:,\d{3})+|\d+)(\.\d+)?(?![\w/]|\.\d)")

# ---------------------------------------------------------------- expressions
OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv, ast.Pow: operator.pow}

def parse_expr(s):
    """-> (template with {i} holes, [operand values]) or None. Operands are the numeric constants of the expression."""
    s = s.replace("x", "*").replace("X", "*").replace("×", "*").replace("÷", "/").replace("^", "**").replace("$", "")
    s = re.sub(r"(?<=\d),(?=\d{3}\b)", "", s)
    if not re.fullmatch(r"[\d.\s+\-*/()]+", s):
        return None
    try:
        tree = ast.parse(s.strip(), mode="eval")
    except SyntaxError:
        return None
    vals = []
    def walk(n):
        if isinstance(n, ast.Expression):
            return walk(n.body)
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
            vals.append(float(n.value)); return "{%d}" % (len(vals) - 1)
        if isinstance(n, ast.BinOp) and type(n.op) in OPS:
            return "(%s %s %s)" % (walk(n.left), {ast.Add: "+", ast.Sub: "-", ast.Mult: "*", ast.Div: "/", ast.Pow: "**"}[type(n.op)], walk(n.right))
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, (ast.USub, ast.UAdd)):
            return "(%s%s)" % ("-" if isinstance(n.op, ast.USub) else "+", walk(n.operand))
        raise ValueError
    try:
        return walk(tree), vals
    except ValueError:
        return None

def run(tmpl, vals):
    def ev(n):
        if isinstance(n, ast.Expression): return ev(n.body)
        if isinstance(n, ast.Constant): return n.value
        if isinstance(n, ast.BinOp):
            a, b = ev(n.left), ev(n.right)
            if isinstance(n.op, ast.Div) and b == 0: raise ZeroDivisionError
            if isinstance(n.op, ast.Pow) and abs(b) > 6: raise OverflowError
            return OPS[type(n.op)](a, b)
        if isinstance(n, ast.UnaryOp): return -ev(n.operand) if isinstance(n.op, ast.USub) else ev(n.operand)
        raise ValueError
    return ev(ast.parse(tmpl.format(*[repr(float(v)) for v in vals]), mode="eval"))

near = lambda a, b: abs(a - b) <= 1e-6 * max(1.0, abs(b)) + 1e-9
isint = lambda v: abs(v - round(v)) < 1e-6
ndec = lambda v: 0 if isint(v) else (1 if isint(v * 10) else (2 if isint(v * 100) else 9))

# ---------------------------------------------------------------- question literals
def literals(q):
    """-> list of dicts {span, val, txt, frozen(reason|None)}"""
    out, defn = [], {m.start(1) for m in DEFN.finditer(q)}
    for m in NUM.finditer(q):
        txt = m.group(0); v = float(txt.replace(",", "")); a, b = m.span()
        pre, post = q[max(0, a - 2):a], q[b:b + 6].lower()
        fr = None
        if re.match(r"\s*(?:am|pm|a\.m|p\.m|o'clock)\b|:\d", post) or pre.endswith(":"): fr = "time"
        elif re.match(r"(?:st|nd|rd|th)\b", post): fr = "ordinal"
        elif not txt.count(".") and 1900 <= v <= 2100 and "," not in txt: fr = "year"
        elif a in defn: fr = "definitional"
        elif post.startswith("/") or pre.endswith("/"): fr = "fraction"
        out.append(dict(span=(a, b), val=v, txt=txt, frozen=fr, pct=post.lstrip().startswith(("%", "percent"))))
    return out

def fmt_like(old_txt, v):
    dec = len(old_txt.split(".")[1]) if "." in old_txt else 0
    s = f"{v:.{dec}f}"
    if "," in old_txt:
        ip, _, fp = s.partition(".")
        s = f"{int(ip):,}" + ("." + fp if fp else "")
    return s

def draw(lit, rng):
    v, txt = lit["val"], lit["txt"]
    dec = len(txt.split(".")[1]) if "." in txt else 0
    if lit["pct"]:
        step = 5 if v % 5 == 0 else 1
        return float(rng.choice([x for x in range(step, 100, step) if x != v]))
    if dec:
        lo, hi = max(1, int(v * 0.5 * 10 ** dec)), max(2, int(v * 1.5 * 10 ** dec))
        return round(rng.randint(lo, hi) / 10 ** dec, dec)
    v = int(v)
    if v <= 10:
        return float(rng.choice([x for x in range(2, 13) if x != v]))
    step = next(s for s in (1000, 100, 10, 5, 1) if v % s == 0 and v // s >= 2)
    lo, hi = max(1, math.ceil(v * 0.5 / step)), max(2, math.floor(v * 1.5 / step))
    return float(rng.randint(lo, hi) * step)

# ---------------------------------------------------------------- names
nl = lambda f: {l.strip() for l in open(C("names_corpus_dl/names/" + f)) if l.strip()}
NM, NF = nl("male.txt"), nl("female.txt")
GZ = json.load(open(C("names_gazetteer.json")))
TITLES = {"Mr", "Mrs", "Ms", "Miss", "Dr", "Mx", "Prof", "Sir", "Madam", "Aunt", "Uncle", "Grandma", "Grandpa", "Coach", "Officer"}

def name_map(doc, text_all, rng):
    mp, kept = {}, 0
    for ent in doc.ents:
        if ent.label_ != "PERSON":
            continue
        for t in ent:
            n = t.text
            if not re.fullmatch(r"[A-Z][a-z]+", n) or n in mp: continue
            if t.i > 0 and doc[t.i - 1].text.rstrip(".") in TITLES or "-" in (doc[t.i - 1].text if t.i else "") or \
               (t.i + 1 < len(doc) and doc[t.i + 1].text == "-"):
                continue
            g = "M" if n in NM and n not in NF else ("F" if n in NF and n not in NM else None)
            if g is None:
                kept += 1; continue
            used = set(mp.values())
            pool = [x for x in GZ[g] if x not in used and not re.search(rf"\b{x}\b", text_all)]
            mp[n] = rng.choice(pool)
    return mp, kept

def apply_names(s, mp):
    for a, b in mp.items():
        s = re.sub(rf"(?<![\w-]){a}(?![\w-])", b, s)
    return s

# ---------------------------------------------------------------- main
def build_problem(q, ans, rng):
    steps = re.findall(r"<<([^=<>]+)=([^<>]+)>>", ans)
    fin = re.search(r"####\s*([-\d.,]+)", ans)
    if not steps: return "no_trace", None
    if not fin: return "no_final", None
    final = float(fin.group(1).replace(",", ""))
    lits = literals(q)
    toks = [w.lower() for w in re.findall(r"[A-Za-z]+", q)]
    words = {WORDS[w] for w in toks if w in WORDS}                      # number words: a literal of equal value is ambiguous
    units = {UNITW[w] for w in toks if w in UNITW}                      # unit constants: allowed sources only
    free = [l for l in lits if not l["frozen"]]
    fvals = collections.Counter(l["val"] for l in free)
    consts = {l["val"] for l in lits if l["frozen"]} | set(map(float, words | units)) | set(map(float, IMPLICIT))
    if re.search(r"%|percent", q, re.I): consts |= {0.01}               # the percent factor (smoke fix 10-23)
    trace, results = [], []
    for expr, res in steps:
        pe = parse_expr(expr)
        try:
            r = float(res.strip().replace(",", "").replace("$", "").rstrip("."))
        except ValueError:
            return "bad_result", None
        if pe is None: return "unparseable", None
        tmpl, vals = pe
        try:
            if not near(run(tmpl, vals), r): return "self_inconsistent", None
        except (ZeroDivisionError, OverflowError, ValueError):
            return "unparseable", None
        src = []
        for v in vals:
            cand = []
            if fvals.get(v): cand.append(("lit", v))
            prior = [j for j, pr in enumerate(results) if near(pr, v)]
            if prior: cand.append(("res", prior[-1]))
            if not cand and any(near(v, c) for c in consts): cand.append(("const", v))
            if not cand: return "unsourced", None
            if len(cand) > 1: return "ambiguous_lit_vs_result", None
            if cand[0][0] == "lit":
                if fvals[v] > 1: return "ambiguous_dup_literal", None
                if v in words or v in {l["val"] for l in lits if l["frozen"]}: return "ambiguous_lit_vs_word", None
            src.append(cand[0])
        trace.append(dict(expr=expr.strip(), tmpl=tmpl, src=src, res=r))
        results.append(r)
    fi = [j for j, r in enumerate(results) if near(r, final)]
    if not fi: return "final_not_in_trace", None
    used = sorted({s[1] for st in trace for s in st["src"] if s[0] == "lit"})
    if not used: return "no_literal_used", None
    lit_by_val = {l["val"]: l for l in free}
    out = []
    seen = set()
    for _ in range(300):
        if len(out) == K: break
        new = {}
        for v in used:
            nv = draw(lit_by_val[v], rng)
            new[v] = nv
        if len(set(new.values())) < len(new) or tuple(sorted(new.items())) in seen: continue
        if any(nv in fvals for nv in new.values()): continue   # distinct from the question's other literals (smoke fix 10-23:
                                                               # also barring implicit constants emptied small-literal pools)
        seen.add(tuple(sorted(new.items())))
        nres, ok = [], True
        for st, r0 in zip(trace, results):
            vals = [new[s[1]] if s[0] == "lit" else (nres[s[1]] if s[0] == "res" else s[1]) for s in st["src"]]
            try:
                r = run(st["tmpl"], vals)
            except (ZeroDivisionError, OverflowError, ValueError):
                ok = False; break
            if (r0 >= 0 and r < -1e-9) or (isint(r0) and not isint(r)) or (ndec(r0) <= 2 and ndec(r) > 2) or abs(r) > 1e9:
                ok = False; break
            nres.append(round(r) if isint(r) else round(r, 6))
        if not ok: continue
        qn, off = q, 0
        for l in free:
            if l["val"] in new:
                a, b = l["span"]; s = fmt_like(l["txt"], new[l["val"]])
                qn = qn[:a + off] + s + qn[b + off:]; off += len(s) - (b - a)
        tcf = [dict(expr=st["tmpl"].format(*[fmt_num(new[s[1]] if s[0] == "lit" else (nres[s[1]] if s[0] == "res" else s[1]))
                                             for s in st["src"]]), res=nres[j]) for j, st in enumerate(trace)]
        out.append(dict(question_cf=qn, literal_map={fmt_num(k): fmt_num(v) for k, v in new.items()}, trace_cf=tcf,
                        final_cf=nres[fi[-1]]))
    if not out: return "no_valid_draw", None
    frozen = collections.Counter(l["frozen"] for l in lits if l["frozen"])
    return "ok", dict(trace=[dict(expr=t["expr"], res=t["res"]) for t in trace], final=final, final_step=fi[-1],
                      n_lit_used=len(used), frozen=dict(frozen), cfs=out,
                      src_kinds=collections.Counter(s[0] for t in trace for s in t["src"]))

def fmt_num(v):
    return str(int(round(v))) if isint(v) else f"{v:g}"

def mask(q):
    return NUM.sub("#", q)

def main():
    norm = lambda s: re.sub(r"\s+", " ", s.strip().lower())
    fixtures = set()
    for f in glob.glob(C("wild_*holdout*.jsonl")):
        fixtures |= {norm(json.loads(l)["text"]) for l in open(f)}
    t = pq.read_table(glob.glob(C("gsm8k/datasets--openai--gsm8k/snapshots/*/main/train-*.parquet"))[0]).to_pydict()
    items = [(i, q, a) for i, (q, a) in enumerate(zip(t["question"], t["answer"])) if norm(q) not in fixtures]
    n_all = len(t["question"]); n_excl = n_all - len(items)
    if SMOKE: items = items[:SMOKE]
    import spacy
    nlp = spacy.load("en_core_web_sm")
    rng = random.Random(1023)
    census, rows = collections.Counter(), []
    src_tot, frozen_tot, mem_same, mem_n, graph_ok = collections.Counter(), collections.Counter(), 0, 0, 0
    docs = nlp.pipe([q for _, q, _ in items], batch_size=64)
    names_swapped = names_kept = 0
    for (i, q, a), doc in zip(items, docs):
        why, P = build_problem(q, a, rng)
        census[why] += 1
        if P is None: continue
        src_tot.update(P["src_kinds"]); frozen_tot.update(P["frozen"])
        mp, kept = name_map(doc, q + " " + a, rng)
        names_kept += kept; names_swapped += len(mp)
        for c, cf in enumerate(P["cfs"]):
            old = [s["res"] for s in P["trace"]]; new = [s["res"] for s in cf["trace_cf"]]
            mem_same += sum(near(n, o) for n, o in zip(new, old)); mem_n += len(old)
            graph_ok += 1                                                         # by construction; asserted below
            assert mask(cf["question_cf"]) == mask(q), (i, q, cf["question_cf"])
            row = dict(gsm8k_train_idx=i, cf_id=c, kind="num", question=q, question_cf=cf["question_cf"],
                       trace=P["trace"], trace_cf=cf["trace_cf"], final=P["final"], final_cf=cf["final_cf"],
                       literal_map=cf["literal_map"], frozen=P["frozen"], name_map={})
            rows.append(row)
            if c == 0 and mp:
                rows.append(dict(row, kind="num+name", question_cf=apply_names(cf["question_cf"], mp), name_map=mp))
    elig = len(items)
    ok = census["ok"]
    F1 = ok / elig
    F3 = mem_same / max(1, mem_n)
    res = dict(train_total=n_all, eligible=elig, fixture_excluded=n_excl,
               census=dict(census), F1_yield=F1, F3_value_memorizer=F3, intermediates_scored=mem_n,
               cf_rows=len(rows), name_rows=sum(r["kind"] == "num+name" for r in rows), names_swapped=names_swapped,
               names_kept_unknown_gender=names_kept, operand_sources=dict(src_tot), frozen_literals=dict(frozen_tot),
               graph_memorizer=1.0)
    print(json.dumps(res, indent=1))
    if SMOKE:
        for r in rows[:6]:
            print("\n", r["question"], "\n->", r["question_cf"], "\n", r["trace"], "\n", r["trace_cf"])
        return
    with open(OUTJ, "w") as f:
        for r in rows: f.write(json.dumps(r) + "\n")
    json.dump(res, open(OUTC, "w"), indent=1)
    with open(OUTT, "w") as f:
        f.write(f"GSM8K COUNTERFACTUAL FIXTURE census\n{json.dumps(res, indent=1)}\n"
                f"BARS: F1 yield {F1:.3f} vs >= 0.50 -> {'PASS' if F1 >= 0.5 else 'FAIL'}; "
                f"F3 value memorizer {F3:.3f} vs <= 0.15 -> {'PASS' if F3 <= 0.15 else 'FAIL'}; F2 = hand-check (40 rows)\n")
    hr = random.Random(4023)
    pick = hr.sample([r for r in rows if r["kind"] == "num"], 30) + hr.sample([r for r in rows if r["kind"] == "num+name"], 10)
    with open(OUTH, "w") as f:
        f.write("HAND-CHECK (F2): each counterfactual must be sensible English-world AND its recomputed gold correct. "
                "Mark OK / BAD + reason.\n\n")
        for n, r in enumerate(pick):
            f.write(f"#{n} [{r['kind']}] train {r['gsm8k_train_idx']} cf {r['cf_id']}\nQ: {r['question_cf']}\n")
            for s in r["trace_cf"]: f.write(f"   {s['expr']} = {fmt_num(s['res'])}\n")
            f.write(f"   FINAL {fmt_num(r['final_cf'])}   (orig final {fmt_num(r['final'])}; names {r['name_map']})\nVERDICT: \n\n")
    print("wrote", OUTJ, OUTC, OUTT, OUTH)

if __name__ == "__main__":
    main()
