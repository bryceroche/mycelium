"""prime_tables.py -- THE PRIME READS R1 + R2 (2026-10-18; bars pinned in the ledger 11:25 before running). Zero GPU.
R1 THE NL PRIME TABLE: hidden constants. Source A = gold givens of the unique verbatim GSM8K rows of form_mix_real whose
value is neither a numeral nor a number word of the text; Source B = GSM8K train calculator operands that are neither a
question numeral / number word nor an earlier step's result. Each hidden value gets its most associated question lemmas
(lift over the base rate of that lemma across questions).
R2 THE MATH PRIME TABLE: motifs on the same gold graphs. 2-motifs = two non-given factors sharing a var; 3-motifs =
connected triples of non-given factors, typed by a brute-force canonical form.
Out: .cache/prime_tables.txt + .cache/prime_tables.json"""
import collections, glob, itertools, json, re
import pyarrow.parquet as pq
import spacy

OUT = []


def say(s=""):
    print(s, flush=True); OUT.append(s)


NUM = re.compile(r"(?<![\w])\$?(\d[\d,]*(?:\.\d+)?)")
WORDS = {"zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
         "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16,
         "seventeen": 17, "eighteen": 18, "nineteen": 19, "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
         "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90, "hundred": 100, "thousand": 1000}
# fraction / multiplier words name BOTH the factor and its reciprocal-side integer ('half' -> 2 and 0.5)
FRAC = {"half": (2, 0.5), "halves": (2, 0.5), "twice": (2,), "double": (2,), "doubled": (2,), "doubles": (2,),
        "triple": (3,), "tripled": (3,), "thrice": (3,), "third": (3, 1 / 3), "thirds": (3, 1 / 3),
        "quarter": (4, 0.25), "quarters": (4, 0.25), "fourth": (4, 0.25), "fourths": (4, 0.25), "fifth": (5, 0.2),
        "fifths": (5, 0.2), "tenth": (10, 0.1), "percent": (100,), "%": (100,), "quadruple": (4,)}


def val(s):
    try:
        return float(s.replace(",", ""))
    except ValueError:
        return None


def text_values(t):
    """every value the text states: numerals, number words (incl. 'twenty-five'), fraction/multiplier words, a/b."""
    vs = {val(m.group(1)) for m in NUM.finditer(t)} - {None}
    low = t.lower()
    for w in re.findall(r"[a-z]+(?:-[a-z]+)?", low):
        parts = w.split("-")
        if all(p in WORDS for p in parts):
            vs.add(float(sum(WORDS[p] for p in parts)))
        for p in parts:
            if p in WORDS:
                vs.add(float(WORDS[p]))
            if p in FRAC:
                vs.update(float(x) for x in FRAC[p])
    if "%" in t:
        vs.add(100.0)
    for a, b in re.findall(r"(\d+)\s*/\s*(\d+)", t):
        vs.update((float(a), float(b)))
    return vs


def close(v, S):
    return any(abs(v - s) < 1e-6 * max(1, abs(v)) for s in S)


nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])
lem = lambda doc: {t.lemma_.lower() for t in doc if t.is_alpha and len(t) > 1}


def lift_table(hidden, docs_lemmas, nq, top_vals=25, top_lem=4):
    """hidden: list of (value, qidx). Lift of lemma w for value v = P(w | v hidden in q) / P(w)."""
    base = collections.Counter(w for L in docs_lemmas for w in L)
    by_v = collections.defaultdict(list)
    for v, q in hidden:
        by_v[v].append(q)
    rows = []
    for v, qs in sorted(by_v.items(), key=lambda kv: -len(kv[1]))[:top_vals]:
        qs_u = set(qs); c = collections.Counter(w for q in qs_u for w in docs_lemmas[q])
        sc = [(c[w] / len(qs_u) / (base[w] / nq), w, c[w]) for w in c if c[w] >= max(3, 0.1 * len(qs_u))]
        sc.sort(reverse=True)
        rows.append((v, len(qs), len(qs_u), [(w, round(l, 1), n) for l, w, n in sc[:top_lem]]))
    return rows


def fmt_v(v):
    return str(int(v)) if v == int(v) else f"{v:g}"


# ---------------- R1 source A: gold givens ----------------
gold, seen = [], set()
for l in open(".cache/form_mix_real.jsonl"):
    r = json.loads(l); g = r.get("gen") or {}
    if g.get("src") == "gsm8k" and not g.get("resample") and r["text"] not in seen:
        seen.add(r["text"]); gold.append(r)
docsA = [lem(d) for d in nlp.pipe([r["text"] for r in gold], batch_size=256)]
hidA, nGiv = [], 0
for qi, r in enumerate(gold):
    tv = text_values(r["text"])
    for f in r["factors"]:
        if f["ftype"] == "given":
            nGiv += 1
            if not close(float(f["value"]), tv):
                hidA.append((float(f["value"]), qi))
mA = collections.Counter(v for v, _ in hidA)
cov20A = sum(c for _, c in mA.most_common(20)) / max(1, len(hidA))
say(f"R1 SOURCE A (gold): {len(gold)} unique GSM8K rows | givens {nGiv} | HIDDEN givens {len(hidA)} ({len(hidA) / nGiv:.3f}) "
    f"in {len({q for _, q in hidA})} rows | distinct hidden values {len(mA)} | TOP-20 share {cov20A:.3f}")
for v, n, nu, L in lift_table(hidA, docsA, len(gold)):
    say(f"  {fmt_v(v):>8s}  n {n:4d} (rows {nu:4d}) | " + "  ".join(f"{w} x{l} ({c})" for w, l, c in L))
vA = "VOID" if not hidA else "CLOSED TABLE" if cov20A >= 0.80 else "OPEN" if cov20A < 0.60 else "MIXED (between the bars)"
say(f"R1 VERDICT (pinned frame, source A): {vA}")

# ---------------- R1 source B: GSM8K train calculator steps ----------------
snap = glob.glob(".cache/gsm8k/datasets--openai--gsm8k/snapshots/*/main/")[0]
T = pq.read_table(snap + "train-00000-of-00001.parquet").to_pydict()
docsB = [lem(d) for d in nlp.pipe(T["question"], batch_size=256)]
OPND = re.compile(r"(?<![\d.])(\d+(?:\.\d+)?)")
hidB, nOp, nDer = [], 0, 0
for qi, (q, a) in enumerate(zip(T["question"], T["answer"])):
    tv, prior = text_values(q), set()
    for m in re.finditer(r"<<([^<>=]*)=([^<>]*)>>", a):
        lhs, rhs = m.group(1), m.group(2)
        for o in OPND.findall(lhs.replace(",", "")):
            v = float(o); nOp += 1
            if close(v, tv):
                continue
            if close(v, prior):
                nDer += 1; continue
            hidB.append((v, qi))
        r_ = val(rhs.strip())
        if r_ is not None:
            prior.add(r_)
mB = collections.Counter(v for v, _ in hidB)
cov20B = sum(c for _, c in mB.most_common(20)) / max(1, len(hidB))
say(f"\nR1 SOURCE B (GSM8K train calculator operands): {len(T['question'])} problems | operands {nOp} | derived (an earlier "
    f"result) {nDer} ({nDer / nOp:.3f}) | HIDDEN {len(hidB)} ({len(hidB) / nOp:.3f}) | distinct {len(mB)} | TOP-20 share {cov20B:.3f}")
for v, n, nu, L in lift_table(hidB, docsB, len(T["question"])):
    say(f"  {fmt_v(v):>8s}  n {n:5d} (rows {nu:5d}) | " + "  ".join(f"{w} x{l} ({c})" for w, l, c in L))
say(f"  (source B is reported, no bar: its 'hidden' also holds operands of results never written as a step and rounding)")

# ---------------- R2 motifs ----------------
def graph(r):
    giv = {f["var"] for f in r["factors"] if f["ftype"] == "given"}
    F = []
    for f in r["factors"]:
        if f["ftype"] == "rel":
            F.append((f["op"], tuple(f["args"]), f["result"]))
        elif f["ftype"] == "fdiv":
            F.append(("fdiv", (f["var"],), f["result"]))
    kind = lambda v: "K" if v in giv else "Q" if v == r["query_var"] else "I"
    return F, kind


def canon(fs, kind):
    best = None
    for perm in itertools.permutations(fs):
        for flips in itertools.product((0, 1), repeat=len(perm)):
            lab, parts = {}, []
            for (op, args, res), fl in zip(perm, flips):
                a = list(args)[::-1] if (fl and op in ("add", "mul")) else list(args)
                if fl and op not in ("add", "mul"):
                    break
                ids = []
                for v in a + [res]:
                    if v not in lab:
                        lab[v] = len(lab)
                    ids.append(f"{lab[v]}{kind(v)}")
                parts.append(f"{op}({','.join(ids[:-1])})={ids[-1]}")
            else:
                s = " ; ".join(parts)
                best = s if best is None or s < best else best
    return best


M2, M3, nfac, fan = collections.Counter(), collections.Counter(), [], collections.Counter()
for r in gold:
    F, kind = graph(r)
    nfac.append(len(F))
    occ = collections.defaultdict(set)
    for i, (op, args, res) in enumerate(F):
        for v in list(args) + [res]:
            occ[v].add(i)
    for v, s in occ.items():
        fan[min(len(s), 4)] += 1
    adj = collections.defaultdict(set)
    for v, s in occ.items():
        for i, j in itertools.combinations(sorted(s), 2):
            adj[i].add(j); adj[j].add(i)
    for i, j in {(min(i, j), max(i, j)) for i in adj for j in adj[i]}:
        M2[canon([F[i], F[j]], kind)] += 1
    trip = set()
    for i in adj:
        for j in adj[i]:
            for k in adj[i] | adj[j]:
                if len({i, j, k}) == 3:
                    trip.add(tuple(sorted((i, j, k))))
    for t in trip:
        M3[canon([F[x] for x in t], kind)] += 1


def curve(M):
    c = sorted(M.values(), reverse=True); n = sum(c)
    return n, {k: sum(c[:k]) / n for k in (1, 5, 12, 25, 50, 100)}


n2, c2 = curve(M2); n3, c3 = curve(M3)
say(f"\nR2: {len(gold)} gold graphs | non-given factors per graph mean {sum(nfac) / len(nfac):.2f} (max {max(nfac)}) | "
    f"var fan-out (factors touching a var) 1:{fan[1]} 2:{fan[2]} 3:{fan[3]} 4+:{fan[4]}")
say(f"  2-MOTIFS: {n2} occurrences, {len(M2)} types | top-k cover " + " ".join(f"{k}:{v:.3f}" for k, v in c2.items()))
for s, c in M2.most_common(15):
    say(f"    {c / n2:.3f}  {s}")
say(f"  3-MOTIFS: {n3} occurrences, {len(M3)} types | top-k cover " + " ".join(f"{k}:{v:.3f}" for k, v in c3.items()))
for s, c in M3.most_common(15):
    say(f"    {c / n3:.3f}  {s}")
v2 = "FEW PRIMES" if c3[12] >= 0.80 else "MANY" if c3[12] < 0.60 else "BETWEEN"
say(f"R2 VERDICT (pinned frame, top-12 3-motif cover {c3[12]:.3f}): {v2}")
say("  (var kinds: K known/given, Q query, I internal; add(a,b)=c covers both a+b=c and c-b=a; mul covers x and /)")
open(".cache/prime_tables.txt", "w").write("\n".join(OUT) + "\n")
json.dump({"hiddenA": {fmt_v(v): c for v, c in mA.most_common()}, "hiddenB": {fmt_v(v): c for v, c in mB.most_common(200)},
           "motif2": dict(M2.most_common()), "motif3": dict(M3.most_common(300))}, open(".cache/prime_tables.json", "w"))
