"""step_blame_census.py -- CAN GSM8K'S STEP VALUES GRADE A GRAPH STEP BY STEP? (2026-10-16, Bryce: "step level gradient
and step level lookup table... give credit for correct intermediate results"; zero GPU.)
Wild rows are verbatim GSM8K TRAIN questions (302/311); the dataset's rationale carries every intermediate value in its
<<a op b = c>> annotations. STEP SET S(row) = the calc results c (+ the #### key).
For each head (LV_DUMP of wild, positional frame: var j = the var introduced at slot j), the predicted graph is rebuilt from
the dump (given digits; rel op + args (+ the dup bit) + result) and solved by the June core (the reading_order ladder).
  SOLVED / FINAL-RIGHT     the graph has a solution / the gold query slot carries the key
  STEP CREDIT              per predicted relation: the value of the var it INTRODUCES is in S (a creditable step)
  BLAME on FINAL-WRONG     #predicted relations whose introduced value is NOT in S: 0 (blame invisible) / 1 (ISOLATED:
                           exactly one bad step) / 2+ (diffuse)
CONTROLS (honesty of the credit signal):
  GOLD CEILING             the same credit on the GOLD graphs (how often a true step's value is in S at all)
  CHANCE                   credit against ANOTHER row's S (rotated by one): how often a value is creditable by accident
Out: .cache/step_blame_census.txt"""
import collections, glob, json, pickle, re, sys

sys.path.insert(0, "."); sys.path.insert(0, "scripts")
import pyarrow.parquet as pq  # noqa: E402

from admit_annotation import solve_ladder  # noqa: E402
from mycelium.csp_domains import problem_from_algebra3  # noqa: E402

norm = lambda s: re.sub(r"\s+", " ", s).strip()
snap = glob.glob(".cache/gsm8k/datasets--openai--gsm8k/snapshots/*/main/")[0]
T = pq.read_table(snap + "train-00000-of-00001.parquet").to_pydict()
RAT = {norm(q): a for q, a in zip(T["question"], T["answer"])}
CALC = re.compile(r"<<[^<>=]+=([\d,\.\-]+)>>")
OUT = []


def say(s=""):
    print(s); OUT.append(s)


def step_set(answer):
    s = set()
    for c in CALC.findall(answer):
        try:
            v = float(c.replace(",", ""))
            if v == int(v): s.add(int(v))
        except ValueError:
            pass
    k = answer.split("####")[-1].strip().replace(",", "")
    if re.fullmatch(r"-?\d+", k): s.add(int(k))
    return s


def solve_all(nv, facs, m=300):
    gv = {f["var"]: int(f["value"]) for f in facs if f["ftype"] == "given"}
    m0 = int(min(10001, max(300, 2 * max([abs(v) for v in gv.values()] + [1]), m)))
    res, _ = solve_ladder(lambda m: problem_from_algebra3(nv, facs, gv, m), m0, budget=5000, wall=10)
    return res.get("assignment") if res.get("status") == "solved" else None


def intro_vars(facs):
    """per relation index: the var it introduces (first not-yet-seen of args + [result]); None if none."""
    seen, out = set(), []
    for f in facs:
        if f["ftype"] == "given":
            seen.add(f["var"]); continue
        vs = list(f["args"]) + [f["result"]]
        new = [v for v in vs if v not in seen]
        out.append(new[0] if new else None); seen |= set(vs)
    return out


def pred_graph(slots):
    facs, bad = [], 0
    for s in slots:   # (j, gft, gop, gargs, gres, gdig, pft, pop, pargs, pres, pdig, ppres, pdup)
        j, pft, pop, pargs, pres, pdig, ppres, pdup = s[0], s[6], s[7], s[8], s[9], s[10], s[11], s[12]
        if not ppres:
            continue
        if pft == 1:
            facs.append({"ftype": "given", "var": j, "value": int("".join(str(int(d)) for d in pdig))})
        elif pft == 0:
            a = list(pargs) * 2 if (len(pargs) == 1 and pdup) else list(pargs)
            if len(a) != 2:
                bad += 1; continue
            facs.append({"ftype": "rel", "op": ("add", "mul")[pop], "args": a, "result": pres})
        else:
            bad += 1
    return facs, bad


def credit(facs, asg, S):
    vals = []
    for f, v in zip([f for f in facs if f["ftype"] != "given"], intro_vars(facs)):
        if v is not None and asg is not None and v < len(asg):
            vals.append(int(asg[v]) in S)
    return vals


def main(tags):
    rows = [json.loads(l) for l in open(".cache/wild_admitted_holdout.jsonl")]
    Ss = [step_set(RAT[norm(r["text"])]) if norm(r["text"]) in RAT else None for r in rows]
    have = [i for i, s in enumerate(Ss) if s]
    say(f"wild rows with a GSM8K train rationale: {len(have)}/{len(rows)}")
    C = collections.Counter()
    for i in have:
        r = rows[i]; asg = solve_all(int(r.get("n_vars", 24)), r["factors"])
        c = credit(r["factors"], asg, Ss[i]); c2 = credit(r["factors"], asg, Ss[have[(have.index(i) + 1) % len(have)]])
        C["solved"] += asg is not None; C["steps"] += len(c); C["cred"] += sum(c); C["chance"] += sum(c2)
        C["rows_all"] += bool(c) and all(c)
    say(f"GOLD CEILING: solved {C['solved']}/{len(have)} | relation steps {C['steps']} | creditable (value in S) "
        f"{C['cred'] / max(C['steps'], 1):.3f} | CHANCE (another row's S) {C['chance'] / max(C['steps'], 1):.3f} | rows with every "
        f"step creditable {C['rows_all']}/{len(have)}")
    for tag in tags:
        by = collections.defaultdict(list)
        for t in pickle.load(open(f".cache/dump_wild_{tag}.pkl", "rb")):
            by[t[0]].append(t[1:])
        C = collections.Counter()
        for i in have:
            slots = sorted(by[i], key=lambda s: s[0]); r = rows[i]
            facs, bad = pred_graph(slots)
            C["rows"] += 1
            if bad:
                C["unbuildable"] += 1; continue
            asg = solve_all(24, facs, int(r.get("m", 300)))
            if asg is None:
                C["unsolved"] += 1; continue
            C["solved"] += 1
            key = int(r["solution"][r["query_var"]]) if r.get("solution") else None
            right = key is not None and r["query_var"] < len(asg) and int(asg[r["query_var"]]) == key
            c = credit(facs, asg, Ss[i]); c2 = credit(facs, asg, Ss[have[(have.index(i) + 1) % len(have)]])
            C["steps"] += len(c); C["cred"] += sum(c); C["chance"] += sum(c2)
            tag2 = "R" if right else "W"
            C["final_" + tag2] += 1; C["steps_" + tag2] += len(c); C["cred_" + tag2] += sum(c)
            if not right:
                nb = len(c) - sum(c)
                C["blame_" + ("0" if nb == 0 else "1" if nb == 1 else "2+")] += 1
        n = C["rows"]; w = max(C["final_W"], 1)
        say(f"\n== {tag}: rows {n} | unbuildable {C['unbuildable']} | unsolved {C['unsolved']} | solved {C['solved']} "
            f"(FINAL-RIGHT {C['final_R']}, FINAL-WRONG {C['final_W']})")
        say(f"   STEP CREDIT: all solved graphs {C['cred'] / max(C['steps'], 1):.3f} of {C['steps']} steps (CHANCE "
            f"{C['chance'] / max(C['steps'], 1):.3f}) | on FINAL-RIGHT {C['cred_R'] / max(C['steps_R'], 1):.3f} | on FINAL-WRONG "
            f"{C['cred_W'] / max(C['steps_W'], 1):.3f} ({C['cred_W']} creditable steps inside wrong answers)")
        say(f"   BLAME on FINAL-WRONG: 0 bad steps {C['blame_0'] / w:.3f} | ISOLATED (exactly 1) {C['blame_1'] / w:.3f} | "
            f"2+ {C['blame_2+'] / w:.3f}")
    open(".cache/step_blame_census.txt", "w").write("\n".join(OUT) + "\n")


if __name__ == "__main__":
    main(sys.argv[1:] or ["SRK3_241", "ROK3_241"])
