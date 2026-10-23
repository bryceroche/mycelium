"""gsm8k_counterfactual_v2.py -- zero-GPU: THE COUNTERFACTUAL FIXTURE v2 (2026-10-23; Bryce's word; ledger 13:33 registration).
v1 (gsm8k_counterfactual.py) is the disclosed preview: its IMPLICIT whitelist sourced DERIVED values as constants (3/40 gold errors).
v2 = (a) LICENSED constants + no bare-constant steps, (b) a HAIKU PROPOSER chooses new values / FROZEN per literal (offline data
generation, the 10-22 ruling; question text only), (c) a BLIND HAIKU SOLVER answers each counterfactual from its own program;
a row is ACCEPTED only when that answer equals the executor's final. GOLD = THE EXECUTOR'S TRACE, ALWAYS (Haiku never writes gold).
Stages (all write under .cache/cf2/):
  requests  -> problems.jsonl + prop_in_XX.jsonl (proposer batches: id + question with trace-used literals marked [A=48])
  execute   -> reads prop_out_XX.jsonl; builds K counterfactuals per problem (haiku set, else rule draw honoring Haiku's freezes)
               -> candidates.jsonl + solve_in_XX.jsonl (id + plain counterfactual question only)
  accept    -> reads solve_out_XX.jsonl; agreement filter -> .cache/gsm8k_cf2_train.jsonl + census + handcheck40"""
import collections, glob, json, os, random, re, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gsm8k_counterfactual as V1                                      # helpers only (parse/run/literals/draw/names)
from gsm8k_counterfactual import C, near, isint, ndec, fmt_num, fmt_like, mask, WORDS

D = C("cf2")
K = 2                                                                   # numeral counterfactuals per problem
PROP_B, SOLVE_B = 150, 63
LICENSE = [({60}, r"minute|hour|second|mph"), ({24}, r"hour|day|daily"), ({7}, r"\bdays?\b|week"),
           ({12}, r"dozen|month|year|\bfoot\b|\bfeet\b|inch"), ({52}, r"week|year|annual"), ({365}, r"year|\bdays?\b|annual"),
           ({30, 31}, r"month|\bdays?\b"), ({100, 0.01}, r"%|percent|\bcents?\b|dollar|century"),
           ({1000}, r"thousand|gram|kilo|lit(?:er|re)|millil|met(?:er|re)|\bkg\b|\bml\b|\bkm\b"), ({0.5}, r"half|halves"),
           ({0.25}, r"quarter"), ({0.75}, r"three[- ]quarter"), ({3}, r"triple|thrice|third"), ({4}, r"quarter|fourth"),
           ({10}, r"\bten\b|tenth|dime|decade"), ({5}, r"nickel|fifth"), ({16}, r"\bpounds?\b|ounce")]

def licensed(q):
    out = {0.0, 1.0, 2.0}
    for vals, rx in LICENSE:
        if re.search(rx, q, re.I):
            out |= set(map(float, vals))
    return out

def source(q, ans):
    """v2 sourcing -> (reason, problem dict | None)."""
    steps = re.findall(r"<<([^=<>]+)=([^<>]+)>>", ans)
    fin = re.search(r"####\s*([-\d.,]+)", ans)
    if not steps: return "no_trace", None
    if not fin: return "no_final", None
    final = float(fin.group(1).replace(",", ""))
    lits = V1.literals(q)
    toks = [w.lower() for w in re.findall(r"[A-Za-z]+", q)]
    words = {float(WORDS[w]) for w in toks if w in WORDS}
    free = [l for l in lits if not l["frozen"]]
    fvals = collections.Counter(l["val"] for l in free)
    frozen_vals = {l["val"] for l in lits if l["frozen"]}
    consts = frozen_vals | words | licensed(q)
    trace, results = [], []
    for expr, res in steps:
        pe = V1.parse_expr(expr)
        try:
            r = float(res.strip().replace(",", "").replace("$", "").rstrip("."))
        except ValueError:
            return "bad_result", None
        if pe is None: return "unparseable", None
        tmpl, vals = pe
        try:
            if not near(V1.run(tmpl, vals), r): return "self_inconsistent", None
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
                if v in words or v in frozen_vals: return "ambiguous_lit_vs_word", None
            src.append(cand[0])
        if re.fullmatch(r"\(?[+-]?\{0\}\)?", tmpl) and src[0][0] == "const":
            return "bare_constant_step", None
        trace.append(dict(expr=expr.strip(), tmpl=tmpl, src=src, res=r))
        results.append(r)
    fi = [j for j, r in enumerate(results) if near(r, final)]
    if not fi: return "final_not_in_trace", None
    used = sorted({s[1] for st in trace for s in st["src"] if s[0] == "lit"})
    if not used: return "no_literal_used", None
    letters = {v: chr(65 + i) for i, v in enumerate(used)} if len(used) <= 26 else None
    if letters is None: return "too_many_literals", None
    lit_rows = [dict(letter=letters[l["val"]], val=l["val"], txt=l["txt"], span=l["span"], pct=l["pct"])
                for l in free if l["val"] in letters]
    marked, off = q, 0
    for l in sorted(lit_rows, key=lambda x: x["span"]):
        a, b = l["span"]; s = f"[{l['letter']}={l['txt']}]"
        marked = marked[:a + off] + s + marked[b + off:]; off += len(s) - (b - a)
    return "ok", dict(trace=trace, final=final, final_step=fi[-1], lits=lit_rows, other_free=sorted(fvals), marked=marked)

def execute(P, sets, rng):
    """-> list of counterfactual dicts (<= K). sets: Haiku's proposals (letter -> value | 'FROZEN')."""
    lits = {l["letter"]: l for l in P["lits"]}
    frozen = set()
    for s in sets or []:
        frozen |= {k for k, v in s.items() if isinstance(v, str) and v.upper() == "FROZEN"}
    trace, results = P["trace"], [t["res"] for t in P["trace"]]
    others = set(P["other_free"])

    def check(newv):                                                    # newv: letter -> value
        if any(near(newv[k], lits[k]["val"]) for k in newv if k not in frozen): return None
        changed = [k for k in newv if k not in frozen]
        if not changed: return None
        if any(newv[k] in others for k in changed) or len({newv[k] for k in changed}) < len(changed): return None
        for k in changed:
            if ndec(newv[k]) > (len(lits[k]["txt"].split(".")[1]) if "." in lits[k]["txt"] else 0): return None
            if lits[k]["pct"] and not 0 < newv[k] <= 100: return None
        byval = {lits[k]["val"]: newv[k] for k in newv}
        nres = []
        for st, r0 in zip(trace, results):
            vals = [byval[s[1]] if s[0] == "lit" else (nres[s[1]] if s[0] == "res" else s[1]) for s in st["src"]]
            try:
                r = V1.run(st["tmpl"], vals)
            except (ZeroDivisionError, OverflowError, ValueError):
                return None
            if (r0 >= 0 and r < -1e-9) or (isint(r0) and not isint(r)) or (ndec(r0) <= 2 and ndec(r) > 2) or abs(r) > 1e9:
                return None
            nres.append(round(r) if isint(r) else round(r, 6))
        return nres

    out, seen = [], set()
    def add(newv, srcname):
        key = tuple(sorted(newv.items()))
        if key in seen: return
        nres = check(newv)
        if nres is None: return
        seen.add(key)
        qn, off = P["question"], 0
        for l in sorted(P["lits"], key=lambda x: x["span"]):
            a, b = l["span"]; s = fmt_like(l["txt"], newv[l["letter"]])
            qn = qn[:a + off] + s + qn[b + off:]; off += len(s) - (b - a)
        byval = {lits[k]["val"]: newv[k] for k in newv}
        tcf = [dict(expr=st["tmpl"].format(*[fmt_num(byval[s[1]] if s[0] == "lit" else (nres[s[1]] if s[0] == "res" else s[1]))
                                             for s in st["src"]]), res=nres[j]) for j, st in enumerate(trace)]
        out.append(dict(question_cf=qn, source=srcname, frozen=sorted(frozen),
                        literal_map={lits[k]["txt"]: fmt_like(lits[k]["txt"], newv[k]) for k in newv if k not in frozen},
                        trace_cf=tcf, final_cf=nres[P["final_step"]]))
    for s in sets or []:
        if len(out) == K: break
        try:
            newv = {k: (lits[k]["val"] if k in frozen else float(str(s[k]).replace(",", "").replace("$", ""))) for k in lits}
        except (KeyError, ValueError):
            continue
        add(newv, "haiku")
    for _ in range(300):
        if len(out) == K: break
        add({k: (lits[k]["val"] if k in frozen else V1.draw(dict(val=lits[k]["val"], txt=lits[k]["txt"], pct=lits[k]["pct"]), rng))
             for k in lits}, "rule")
    return out

def ans_eq(a, b):
    try:
        a = float(str(a).replace(",", "").replace("$", ""))
    except ValueError:
        return False
    return near(a, b) or abs(a - b) < 0.005

def stage_requests():
    os.makedirs(D, exist_ok=True)
    assert not os.path.exists(os.path.join(D, "problems.jsonl")), "refuse overwrite"
    import pyarrow.parquet as pq
    norm = lambda s: re.sub(r"\s+", " ", s.strip().lower())
    fixtures = set()
    for f in glob.glob(C("wild_*holdout*.jsonl")):
        fixtures |= {norm(json.loads(l)["text"]) for l in open(f)}
    t = pq.read_table(glob.glob(C("gsm8k/datasets--openai--gsm8k/snapshots/*/main/train-*.parquet"))[0]).to_pydict()
    census, probs = collections.Counter(), []
    for i, (q, a) in enumerate(zip(t["question"], t["answer"])):
        if norm(q) in fixtures:
            census["fixture_excluded"] += 1; continue
        why, P = source(q, a)
        census[why] += 1
        if P: probs.append(dict(P, id=f"p{i}", gsm8k_train_idx=i, question=q))
    with open(os.path.join(D, "problems.jsonl"), "w") as f:
        for P in probs: f.write(json.dumps(P) + "\n")
    for b in range(0, len(probs), PROP_B):
        with open(os.path.join(D, f"prop_in_{b // PROP_B:02d}.jsonl"), "w") as f:
            for P in probs[b:b + PROP_B]:
                f.write(json.dumps(dict(id=P["id"], question=P["marked"], letters=[l["letter"] for l in P["lits"]])) + "\n")
    json.dump(dict(census), open(os.path.join(D, "census_requests.json"), "w"), indent=1)
    print(json.dumps(dict(census), indent=1), "\nproblems", len(probs), "proposer batches", -(-len(probs) // PROP_B))

def stage_execute():
    assert not os.path.exists(os.path.join(D, "candidates.jsonl")), "refuse overwrite"
    probs = [json.loads(l) for l in open(os.path.join(D, "problems.jsonl"))]
    props = {}
    for f in sorted(glob.glob(os.path.join(D, "prop_out_*.jsonl"))):
        for l in open(f):
            l = l.strip()
            if not l: continue
            try:
                r = json.loads(l)
                props[r["id"]] = r.get("sets") or []
            except (json.JSONDecodeError, KeyError):
                pass
    rng = random.Random(1033)
    cands, census = [], collections.Counter()
    census["proposals_missing"] = sum(P["id"] not in props for P in probs)
    for P in probs:
        cfs = execute(P, props.get(P["id"]), rng)
        census["problems_with_cf"] += bool(cfs)
        for c, cf in enumerate(cfs):
            census["cf_" + cf["source"]] += 1
            cands.append(dict(id=f"{P['id']}_c{c}", pid=P["id"], gsm8k_train_idx=P["gsm8k_train_idx"], cf_id=c,
                              question=P["question"], trace=[dict(expr=t["expr"], res=t["res"]) for t in P["trace"]],
                              final=P["final"], **cf))
    with open(os.path.join(D, "candidates.jsonl"), "w") as f:
        for r in cands: f.write(json.dumps(r) + "\n")
    order = list(range(len(cands))); random.Random(7).shuffle(order)          # solver batches mix problems
    for b in range(0, len(order), SOLVE_B):
        with open(os.path.join(D, f"solve_in_{b // SOLVE_B:03d}.jsonl"), "w") as f:
            for k in order[b:b + SOLVE_B]:
                f.write(json.dumps(dict(id=cands[k]["id"], question=cands[k]["question_cf"])) + "\n")
    json.dump(dict(census), open(os.path.join(D, "census_execute.json"), "w"), indent=1)
    print(json.dumps(dict(census), indent=1), "\ncandidates", len(cands), "solver batches", -(-len(cands) // SOLVE_B))

def stage_accept():
    OUTJ, OUTC, OUTT, OUTH = (C(f) for f in ("gsm8k_cf2_train.jsonl", "gsm8k_cf2_census.json", "gsm8k_cf2_census.txt",
                                             "gsm8k_cf2_handcheck40.txt"))
    for p in (OUTJ, OUTC, OUTT, OUTH):
        assert not os.path.exists(p), f"refuse overwrite {p}"
    cands = [json.loads(l) for l in open(os.path.join(D, "candidates.jsonl"))]
    sol = {}
    for f in sorted(glob.glob(os.path.join(D, "solve_out_*.jsonl"))):
        for l in open(f):
            l = l.strip()
            if not l: continue
            try:
                r = json.loads(l); sol[r["id"]] = r.get("answer")
            except (json.JSONDecodeError, KeyError):
                pass
    import spacy
    nlp = spacy.load("en_core_web_sm")
    rng = random.Random(4033)
    census, rows, by_src = collections.Counter(), [], collections.Counter()
    acc_p, mem_same, mem_n = set(), 0, 0
    docs = {}
    for r in cands:
        a = sol.get(r["id"])
        if a is None:
            census["solver_missing"] += 1; continue
        ok = ans_eq(a, r["final_cf"])
        census["agree" if ok else "disagree"] += 1
        by_src[(r["source"], ok)] += 1
        if not ok: continue
        acc_p.add(r["pid"])
        old = [s["res"] for s in r["trace"]]; new = [s["res"] for s in r["trace_cf"]]
        mem_same += sum(near(n, o) for n, o in zip(new, old)); mem_n += len(old)
        assert mask(r["question_cf"]) == mask(r["question"])
        row = dict(r, kind="num", solver_answer=a, name_map={})
        rows.append(row)
        if r["cf_id"] == 0:
            if r["pid"] not in docs: docs[r["pid"]] = nlp(r["question"])
            mp, kept = V1.name_map(docs[r["pid"]], r["question"] + " " + " ".join(s["expr"] for s in r["trace"]), rng)
            census["names_kept_unknown_gender"] += kept
            if mp:
                rows.append(dict(row, kind="num+name", question_cf=V1.apply_names(r["question_cf"], mp), name_map=mp))
    probs = sum(1 for _ in open(os.path.join(D, "problems.jsonl")))
    F1, F3 = len(acc_p), mem_same / max(1, mem_n)
    res = dict(problems_sourced=probs, candidates=len(cands), census=dict(census),
               agreement=census["agree"] / max(1, census["agree"] + census["disagree"]),
               agreement_by_source={f"{s}_{'agree' if o else 'disagree'}": n for (s, o), n in by_src.items()},
               F1_accepted_problems=F1, F3_value_memorizer=F3, intermediates_scored=mem_n, rows=len(rows),
               name_rows=sum(r["kind"] == "num+name" for r in rows), graph_memorizer=1.0)
    print(json.dumps(res, indent=1))
    with open(OUTJ, "w") as f:
        for r in rows: f.write(json.dumps(r) + "\n")
    json.dump(res, open(OUTC, "w"), indent=1)
    with open(OUTT, "w") as f:
        f.write(f"GSM8K COUNTERFACTUAL FIXTURE v2 census\n{json.dumps(res, indent=1)}\nBARS: F1 {F1} accepted problems vs >= 1500 -> "
                f"{'PASS' if F1 >= 1500 else 'FAIL'}; F3 {F3:.3f} vs <= 0.15 -> {'PASS' if F3 <= 0.15 else 'FAIL'}; "
                f"F2 = hand-check (40 rows; >= 38 OK and <= 1 gold error; borderline = BAD)\n")
    hr = random.Random(5033)
    pick = hr.sample([r for r in rows if r["kind"] == "num"], 30) + hr.sample([r for r in rows if r["kind"] == "num+name"], 10)
    with open(OUTH, "w") as f:
        f.write("HAND-CHECK (v2 F2): sensible English-world AND recomputed gold correct. OK / BAD (+reason); borderline = BAD.\n\n")
        for n, r in enumerate(pick):
            f.write(f"#{n} [{r['kind']}/{r['source']}] train {r['gsm8k_train_idx']} cf {r['cf_id']}\nORIG: {r['question']}\n"
                    f"Q: {r['question_cf']}\n")
            for s in r["trace_cf"]: f.write(f"   {s['expr']} = {fmt_num(s['res'])}\n")
            f.write(f"   FINAL {fmt_num(r['final_cf'])}   (frozen {r['frozen']}; names {r['name_map']})\nVERDICT: \n\n")
    print("wrote", OUTJ, OUTC, OUTT, OUTH)

if __name__ == "__main__":
    {"requests": stage_requests, "execute": stage_execute, "accept": stage_accept}[sys.argv[1]]()
