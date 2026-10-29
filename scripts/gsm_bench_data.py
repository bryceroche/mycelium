"""gsm_bench_data.py -- THE GSM8K READER BENCH, the data (ledger 2026-10-29, Bryce "word given for both"). Zero-GPU, CPU only.
The reader arms of the NL binding bench (scripts/nl_reader.py) moved onto GSM8K TEXT. The task is a POINTER PROGRAM: the reader emits
up to KMAX binary steps (op in + - * /, two operands); an operand is a POINTER to a numeral token of the question, to an earlier step,
or to a constant. No number crosses into the reader (every digit numeral is the token <num>); the executor reads the values off the
text and runs the program, so a counterfactual question is graded by re-running the SAME pointers on its new numerals.
GOLD = the harvest trace: every <<expr=res>> is parsed (gsm8k_counterfactual.parse_expr) and binarized (post-order BinOps); an operand
value's VALID SET = numeral tokens of that value + earlier steps of that value; a constant only when neither exists AND it is licensed
by the text (gsm8k_counterfactual_v2.licensed). Unsourced, unary, power, > KMAX steps, or last step != final -> excluded (census).
Splits (written to .cache/gsmb_data.json; GSM8K text, not committed):
  TRAIN   GSM8K-train, representable, wild holdout texts excluded (the CF fixture's exclusion); overfit by design (note 11)
  ORIG    the originals of the CF fixture's problems (train items: the fit read)
  CFNUM   .cache/gsm8k_cf2_train.jsonl rows kind=num (numerals changed; gold final_cf) -- a SANITY read: with numerals masked the token
          sequence is the original's, so a graph memorizer scores its ORIG fit here
  CFNAME  rows kind=num+name (numerals AND first names changed): the reader-vs-graph separation the fixture memo names
  TEST    GSM8K-test, representable (gold from its own trace): THE HONEST GENERALIZATION READ
Names: spaCy PERSON first-name tokens ([A-Z][a-z]+); gender for the tree from the names corpus (male.txt / female.txt)."""
import collections, glob, json, os, re, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gsm8k_counterfactual as V1
import gsm8k_counterfactual_v2 as V2

OUT = V1.C("gsmb_data.json")
KMAX, LQMAX = 10, 240
NUMS = r"\$?(?:\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?|\.\d+)"
TOK = re.compile(NUMS + r"|[A-Za-z]+(?:'[a-z]+)?|[^\sA-Za-z\d]")
NUMRX = re.compile(NUMS)
CONSTS = [0.01, 0.1, 0.25, 0.5, 0.75, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 16, 20, 24, 30, 31, 36, 40, 50, 52, 60, 100, 144, 365, 1000]
OPS = "+-*/"


def tokenize(q, names):
    """-> toks (model words), vals (float or None per token: pointer-eligible numerals + number words), isn, sents (sentence id)."""
    toks, vals, isn, sid, s = [], [], [], [], 0
    for m in TOK.finditer(q):
        w = m.group(0)
        if NUMRX.fullmatch(w):
            toks.append("<num>"); vals.append(float(w.lstrip("$").replace(",", ""))); isn.append(False)
        else:
            lw = w.lower(); toks.append(lw); vals.append(float(V1.WORDS[lw]) if lw in V1.WORDS else None); isn.append(w in names)
        sid.append(s)
        if w in ".?!": s += 1
    return toks, vals, isn, sid


def binarize(tmpl, vals):
    """template '({0} + ({1} * {2}))' -> post-order list of (op, ('v', operand index) | ('n', node index))."""
    import ast
    nodes = []
    def walk(n):
        if isinstance(n, ast.Expression): return walk(n.body)
        if isinstance(n, ast.Constant): return ("v", float(n.value))
        if isinstance(n, ast.BinOp) and type(n.op) in (ast.Add, ast.Sub, ast.Mult, ast.Div):
            a, b = walk(n.left), walk(n.right)
            nodes.append(({ast.Add: "+", ast.Sub: "-", ast.Mult: "*", ast.Div: "/"}[type(n.op)], a, b)); return ("n", len(nodes) - 1)
        raise ValueError
    walk(ast.parse(tmpl.format(*[repr(v) for v in vals]), mode="eval"))
    return nodes


def program(q, ans, names):
    """-> (reason, item | None). item: toks, vals, isn, sid, prog = [(op, validA, validB)], valid = {'t': [tok idx], 's': [step], 'c': [const idx]}."""
    steps = re.findall(r"<<([^=<>]+)=([^<>]+)>>", ans); fin = re.search(r"####\s*([-\d.,]+)", ans)
    if not steps or not fin: return "no_trace", None
    final = float(fin.group(1).replace(",", ""))
    toks, vals, isn, sid = tokenize(q, names)
    if len(toks) > LQMAX: return "too_long", None
    lic = V2.licensed(q); prog, res, census_note = [], [], []
    for expr, r in steps:
        pe = V1.parse_expr(expr)
        if pe is None: return "unparseable", None
        try:
            nodes = binarize(*pe)
            if not V1.near(V1.run(*pe), float(r.strip().replace(",", "").replace("$", "").rstrip("."))): return "self_inconsistent", None
        except (ValueError, SyntaxError, ZeroDivisionError, OverflowError):
            return "unary_or_pow", None
        if not nodes: census_note.append("bare_step_skipped"); continue      # '<<5=5>>' restates a value; later uses source it
        gid = {}                                                        # expression-local node -> global step index
        for nk, (op, a, b) in enumerate(nodes):
            sets, opv = [], []
            for kind, v in (a, b):
                if kind == "n":
                    sets.append(dict(t=[], s=[gid[v]], c=[])); opv.append(res[gid[v]]); continue
                t = [i for i, x in enumerate(vals) if x is not None and V1.near(x, v)]
                s_ = [j for j, x in enumerate(res) if V1.near(x, v)]
                c = [k for k, x in enumerate(CONSTS) if V1.near(x, v) and any(V1.near(v, y) for y in lic)] if not (t or s_) else []
                if not (t or s_ or c):
                    der = derived(toks, vals, v)                        # a/b fraction or p% written as a decimal in the trace
                    if der is None: return "unsourced", None
                    prog.append(der); res.append(v); census_note.append("derived"); s_ = [len(res) - 1]
                sets.append(dict(t=t, s=s_, c=c)); opv.append(v)
            res.append({"+": opv[0] + opv[1], "-": opv[0] - opv[1], "*": opv[0] * opv[1], "/": opv[0] / opv[1] if opv[1] else float("nan")}[op])
            prog.append((op, sets[0], sets[1])); gid[nk] = len(res) - 1
    hit = [j for j, x in enumerate(res) if V1.near(x, final)]
    if not hit: return "final_not_in_trace", None
    prog = prog[:hit[-1] + 1]                                           # steps after the final's step are restatements / checks
    if len(prog) > KMAX: return "too_many_steps", None
    return "ok", dict(toks=toks, vals=vals, isn=isn, sid=sid, prog=prog, final=final)


def derived(toks, vals, v):
    """v unsourced: a fraction 'a / b' or a percent 'p %' of the text, as one inserted division step (op, A, B)."""
    for i in range(len(toks) - 2):
        if toks[i] == toks[i + 2] == "<num>" and toks[i + 1] == "/" and vals[i + 2] and V1.near(vals[i] / vals[i + 2], v):
            return ("/", dict(t=[i], s=[], c=[]), dict(t=[i + 2], s=[], c=[]))
    for i in range(len(toks) - 1):
        if toks[i] == "<num>" and toks[i + 1] in ("%", "percent") and V1.near(vals[i] / 100, v):
            return ("/", dict(t=[i], s=[], c=[]), dict(t=[], s=[], c=[CONSTS.index(100)]))
    return None


def execute(prog_pred, vals):
    """prog_pred: [(op, ('t'|'s'|'c', idx), (...))] -> final value or None."""
    out = []
    try:
        for op, a, b in prog_pred:
            av, bv = [vals[i] if k == "t" else (out[i] if k == "s" else CONSTS[i]) for k, i in (a, b)]
            if av is None or bv is None: return None
            out.append({"+": av + bv, "-": av - bv, "*": av * bv, "/": av / bv}[op])
    except (ZeroDivisionError, IndexError):
        return None
    return out[-1] if out else None


def main():
    assert not os.path.exists(OUT), f"refusing to overwrite {OUT}"
    import pyarrow.parquet as pq, spacy
    nlp = spacy.load("en_core_web_sm")
    norm = lambda s: re.sub(r"\s+", " ", s.strip().lower())
    fixtures = set()
    for f in glob.glob(V1.C("wild_*holdout*.jsonl")): fixtures |= {norm(json.loads(l)["text"]) for l in open(f)}
    rd = lambda sp: pq.read_table(glob.glob(V1.C(f"gsm8k/datasets--openai--gsm8k/snapshots/*/main/{sp}-*.parquet"))[0]).to_pydict()
    tr, te = rd("train"), rd("test")
    cf = [json.loads(l) for l in open(V1.C("gsm8k_cf2_train.jsonl"))]
    texts = list(tr["question"]) + list(te["question"]) + [r["question_cf"] for r in cf]
    names = {}
    for q, doc in zip(texts, nlp.pipe(texts, batch_size=128)):
        names[q] = sorted({t.text for e in doc.ents if e.label_ == "PERSON" for t in e if re.fullmatch(r"[A-Z][a-z]+", t.text)})
    census = collections.defaultdict(collections.Counter); D = collections.defaultdict(list)
    def add(split, q, a, extra):
        why, it = program(q, a, set(names[q]))
        census[split][why] += 1
        if it: it.update(extra, names=names[q]); D[split].append(it)
    for i, (q, a) in enumerate(zip(tr["question"], tr["answer"])):
        if norm(q) in fixtures: census["TRAIN"]["wild_fixture"] += 1; continue
        add("TRAIN", q, a, dict(gid=i))
    trq = {it["gid"]: it for it in D["TRAIN"]}
    for i, (q, a) in enumerate(zip(te["question"], te["answer"])): add("TEST", q, a, dict(gid=i))
    for r in cf:
        sp = "CFNUM" if r["kind"] == "num" else "CFNAME"; base = trq.get(r["gsm8k_train_idx"])
        if base is None: census[sp]["base_not_representable"] += 1; continue
        toks, vals, isn, sid = tokenize(r["question_cf"], set(names[r["question_cf"]]))
        if len(toks) > LQMAX: census[sp]["too_long"] += 1; continue
        pk = lambda st, e: ("t", st["t"][e]) if st["t"] else (("s", st["s"][e]) if st["s"] else ("c", st["c"][e]))
        if not all(V1.near(execute([(o, pk(a, e), pk(b, e)) for o, a, b in base["prog"]], vals) or 0, float(r["final_cf"])) for e in (0, -1)):
            census[sp]["ambiguous_gold_set"] += 1; continue          # a valid set holds a member the counterfactual tells apart
        census[sp]["ok"] += 1
        D[sp].append(dict(toks=toks, vals=vals, isn=isn, sid=sid, prog=base["prog"], final=float(r["final_cf"]), gid=r["gsm8k_train_idx"],
                          cf=r["id"], names=names[r["question_cf"]]))
    D["ORIG"] = [trq[g] for g in sorted({it["gid"] for it in D["CFNUM"] + D["CFNAME"]})]
    # sanity: the gold program executes to the final on every representable item (gold pointers = first valid of each set)
    pick = lambda st: ("t", st["t"][0]) if st["t"] else (("s", st["s"][0]) if st["s"] else ("c", st["c"][0]))
    bad = {sp: sum(not V1.near(execute([(o, pick(a), pick(b)) for o, a, b in it["prog"]], it["vals"]) or 0, it["final"]) for it in D[sp])
           for sp in D}
    lines = [f"{sp:7s} n {len(D[sp]):5d} | " + ", ".join(f"{k} {v}" for k, v in census[sp].most_common()) +
             f" | gold-pick exec mismatch {bad[sp]}" for sp in ("TRAIN", "ORIG", "CFNUM", "CFNAME", "TEST")]
    st = [len(it["prog"]) for it in D["TRAIN"]]
    lines.append(f"TRAIN steps hist {sorted(collections.Counter(st).items())} | tokens mean {sum(len(i['toks']) for i in D['TRAIN']) / len(st):.1f}")
    nn = [len(it["names"]) > 0 for it in D["TEST"]]; lines.append(f"TEST items with >= 1 PERSON name {sum(nn)}/{len(nn)}")
    print("\n".join(lines)); open(V1.C("gsmb_census.txt"), "w").write("\n".join(lines) + "\n")
    json.dump(D, open(OUT, "w"))


if __name__ == "__main__":
    main()
