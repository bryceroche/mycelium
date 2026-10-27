"""order_table.py -- THE MASTER ORDER TABLE (ledger 2026-10-26 19:27, word given; zero GPU, CPU only).
The problem text is never reordered: this table maps the order the GIVENS appear in the text (text_pos) to the order
the equations consume them (eq_slot = rank in gold factor order = derivation order), one row per given, over the
chassis training mix + the wild holdout (split='holdout': a measurement fixture, never trained on).
Custody: reads `text`, `factors`, `mentions`, `gen` only -- never solution fields. owner/owner_kind are the governor
census's spaCy heuristic (an offline LABEL, never gold, never a runtime stage).
  .venv/bin/python3 scripts/order_table.py   -> .cache/order_table.sqlite + .cache/order_table_census.txt"""
import collections, hashlib, json, os, re, sqlite3, sys, time
import xml.etree.ElementTree as ET

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import pyarrow.parquet as pq
import spacy
import governor_census as G
from resample_silver import rewrite

DB, TXT = ".cache/order_table.sqlite", ".cache/order_table_census.txt"
MIXES = [(".cache/form_mix_pm35c.jsonl", "train"), (G.HOLDOUT, "holdout")]
SNAP = ".cache/gsm8k/datasets--openai--gsm8k/snapshots/740312add88f781978c0658806c59bc2815b9866/main/"
norm = lambda s: " ".join(s.split())
tmpl = lambda s: norm(re.sub(r"\d[\d,]*", "#", s))   # digit-blind: resample_silver rewrites digit strings only
UNITS = {w: i for i, w in enumerate("zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen "
                                    "fifteen sixteen seventeen eighteen nineteen".split())}
TENS = {w: 10 * i for i, w in enumerate("_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()) if w != "_"}
SCALE = {"hundred": 100, "thousand": 1000}
NUMW = "|".join(sorted(list(UNITS) + list(TENS) + list(SCALE), key=len, reverse=True))
SPELL_RE = re.compile(rf"\b(?:{NUMW})(?:(?:[\s-]+|\s+and\s+)(?:{NUMW}))*\b", re.I)
DIGIT_RE = re.compile(r"(?<![\w.])-?\d[\d,]*(?:\.\d+)?(?![\w])")   # a sign only when not glued to a word ("6-liter")
# multiplicative words carry a given lexically (the rulebook's lexical explicitation of knowns); fraction denominators
# only after a count word ("a third", "two-thirds"), so ordinals ("the third day") never match
MULT = {"twice": 2, "double": 2, "doubled": 2, "doubles": 2, "thrice": 3, "triple": 3, "tripled": 3, "triples": 3,
        "quadruple": 4, "quadrupled": 4, "half": 2, "halves": 2, "halved": 2, "dozen": 12, "dozens": 12}
DENOM = {"third": 3, "quarter": 4, "fourth": 4, "fifth": 5, "sixth": 6, "seventh": 7, "eighth": 8, "ninth": 9, "tenth": 10}
MULT_RE = re.compile(r"\b(" + "|".join(MULT) + r")\b", re.I)
DENOM_RE = re.compile(r"\b(?:a|an|one|two|three|four|five|six|seven|eight|nine|\d+)[\s-]+(" + "|".join(DENOM) + r")s?\b", re.I)


def spelled_value(phrase):
    total, cur = 0, 0
    for w in re.split(r"[\s-]+", phrase.lower()):
        if w == "and":
            continue
        if w in UNITS: cur += UNITS[w]
        elif w in TENS: cur += TENS[w]
        elif w == "hundred": cur = max(cur, 1) * 100
        elif w == "thousand": total += max(cur, 1) * 1000; cur = 0
    return total + cur


def span_value(s):
    s = s.strip().strip("$")
    if re.fullmatch(r"-?\d[\d,]*", s): return int(s.replace(",", ""))
    if SPELL_RE.fullmatch(s): return spelled_value(s)
    return None


def scan(text):
    """-> [(char_start, value, kind)] in text order; kind in numeral | lexical."""
    out = []
    for m in DIGIT_RE.finditer(text):
        t = m.group().replace(",", "")
        if "." not in t: out.append((m.start(), int(t), "numeral"))
    for m in SPELL_RE.finditer(text):
        neg = m.start() > 0 and text[m.start() - 1] == "-" and (m.start() < 2 or not text[m.start() - 2].isalnum())
        out.append((m.start() - neg, -spelled_value(m.group()) if neg else spelled_value(m.group()), "numeral"))
    for m in MULT_RE.finditer(text):
        out.append((m.start(), MULT[m.group(1).lower()], "lexical"))
    for m in DENOM_RE.finditer(text):
        out.append((m.start(1), DENOM[m.group(1).lower()], "lexical"))
    return sorted(out)
sha = lambda s: hashlib.sha1(s.encode()).hexdigest()[:16]


def sources():
    """normalized text -> (dataset, problem_idx); plus per-dataset id -> normalized text for src_idx verification."""
    by_text, by_id, by_tmpl = {}, {}, {}
    for sp in ("train", "test"):
        for i, q in enumerate(pq.read_table(SNAP + f"{sp}-00000-of-00001.parquet").column("question").to_pylist()):
            by_text.setdefault(norm(q), (f"gsm8k-{sp}", str(i))); by_id[(f"gsm8k-{sp}", str(i))] = norm(q)
            by_tmpl.setdefault(tmpl(q), []).append((f"gsm8k-{sp}", str(i)))
    for p in json.load(open(".cache/eqsets/SVAMP.json")):   # eq_sets_extract.rows_svamp's text, verbatim
        t = norm((p["Body"].strip() + " " + p["Question"].strip()).replace("  ", " ")); by_text.setdefault(t, ("svamp", p["ID"]))
        by_tmpl.setdefault(tmpl(t), []).append(("svamp", p["ID"]))
    for pr in ET.parse(".cache/eqsets/ASDiv.xml").getroot().iter("Problem"):
        t = norm((pr.findtext("Body") or "").strip() + " " + (pr.findtext("Question") or "").strip())
        by_text.setdefault(t, ("asdiv", pr.attrib["ID"])); by_tmpl.setdefault(tmpl(t), []).append(("asdiv", pr.attrib["ID"]))
    return by_text, by_id, by_tmpl


def provenance(r, mix_row, by_text, by_id, by_tmpl):
    gen = r.get("gen") if isinstance(r.get("gen"), dict) else {}
    src = gen.get("src") or (r["gen"] if isinstance(r.get("gen"), str) else None)
    if src not in ("gsm8k", "svamp", "asdiv"):
        name = src or gen.get("shape") or gen.get("diet") or ("ladder" if "ladder" in gen else "pairs" if "n_pairs" in gen else "nogen")
        return f"mint:{name}", str(mix_row), "synthetic"
    texts = [(norm(r["text"]), "exact")]
    subs = gen.get("resample", {}).get("subs")
    if subs:
        texts.append((norm(rewrite(r["text"], {v: k for k, v in subs.items()})[0]), "unsubstituted"))
    if src == "gsm8k" and "src_idx" in gen:
        for t, how in texts:
            for sp in ("train", "test"):
                if by_id.get((f"gsm8k-{sp}", str(gen["src_idx"]))) == t:
                    return f"gsm8k-{sp}", str(gen["src_idx"]), f"src_idx-verified/{how}"
    for t, how in texts:
        if t in by_text and by_text[t][0].startswith(src):
            return by_text[t][0], by_text[t][1], how
    hits = [h for h in by_tmpl.get(tmpl(r["text"]), []) if h[0].startswith(src)]
    if src == "gsm8k" and "src_idx" in gen and any(h[1] == str(gen["src_idx"]) for h in hits):
        h = next(h for h in hits if h[1] == str(gen["src_idx"])); return h[0], h[1], "src_idx-verified/template"
    if len(hits) == 1:
        return hits[0][0], hits[0][1], "template"
    if len(hits) > 1:
        return hits[0][0].split("-")[0] + ("-?" if src == "gsm8k" else ""), None, "template-ambiguous"
    return (src if src != "gsm8k" else "gsm8k-?"), None, "none"


def tok_at(doc, ch):
    return next((t for t in doc if t.idx <= ch < t.idx + len(t.text)), None)


def locate(r, doc, cands):
    """cands = scan(text). -> gl, {var: (char_start, how)}; how in mention | value | lexical | unstated."""
    gl = [(f["var"], f["value"]) for f in r["factors"] if f["ftype"] == "given"]
    out, used = {}, set()
    for var, val in gl:                                         # pass 1: the stamper's spans
        for s, e in r.get("mentions", {}).get(str(var), []):
            if span_value(r["text"][s:e]) == val and s not in used:
                out[var] = (s, "mention"); used.add(s); break
    for kind, how in (("numeral", "value"), ("lexical", "lexical")):   # pass 2 numerals, pass 3 multiplicative words
        for var, val in gl:
            if var in out:
                continue
            ch = next((c for c, v, k in cands if v == val and k == kind and c not in used), None)
            if ch is not None:
                out[var] = (ch, how); used.add(ch)
    for var, _ in gl:
        out.setdefault(var, (None, "unstated"))
    return gl, out


if __name__ == "__main__":
    assert not os.path.exists(DB), "refusing to overwrite a banked table"
    t0 = time.time(); by_text, by_id, by_tmpl = sources()
    nlp = spacy.load("en_core_web_sm", disable=["ner"])
    rows = [(mix, split, i, json.loads(l)) for mix, split in MIXES for i, l in enumerate(open(mix))]
    texts = sorted({r["text"] for *_, r in rows})
    docs = dict(zip(texts, nlp.pipe(texts, batch_size=256, n_process=8)))
    print(f"[order] {len(rows)} rows, {len(texts)} unique texts parsed in {time.time() - t0:.0f}s", flush=True)
    con = sqlite3.connect(DB)
    con.execute("""CREATE TABLE order_map (dataset TEXT, problem_idx TEXT, provenance TEXT, split TEXT, mix TEXT, row INTEGER,
                   text_sha TEXT, var INTEGER, value INTEGER, text_pos INTEGER, char_start INTEGER, eq_slot INTEGER,
                   owner TEXT, owner_kind TEXT, locate TEXT, multi_occ INTEGER)""")   # locate: mention | value | lexical | unstated
    con.execute("""CREATE TABLE order_rows (dataset TEXT, problem_idx TEXT, provenance TEXT, split TEXT, mix TEXT, row INTEGER,
                   text_sha TEXT, n_given INTEGER, n_located INTEGER, identical INTEGER)""")
    C = collections.Counter()
    for mix, split, i, r in rows:
        ds, pid, prov = provenance(r, i, by_text, by_id, by_tmpl); doc = docs[r["text"]]; sl = list(doc.sents)
        cands = scan(r["text"]); gl, loc = locate(r, doc, cands)
        occ = collections.Counter(v for _, v, _ in cands); need = collections.Counter(v for _, v in gl)
        located = sorted((loc[v][0], v) for v, _ in gl if loc[v][0] is not None)
        tpos = {v: k for k, (_, v) in enumerate(located)}
        for eq, (var, val) in enumerate(gl):
            c0, how = loc[var]; tok = tok_at(doc, c0) if c0 is not None else None
            ch = G.governor_chain(doc, sl, tok) if tok is not None else None
            con.execute("INSERT INTO order_map VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                        (ds, pid, prov, split, mix, i, sha(r["text"]), var, val, tpos.get(var), c0,
                         eq, ch["owner_text"] if ch else None, ch["owner_kind"] if ch else None, how, int(occ[val] > need[val])))
            C[(split, ds.split(":")[0], "loc_" + how)] += 1
        eq_order = [v for v, _ in gl if loc[v][0] is not None]
        ident = int(eq_order == [v for _, v in located])
        con.execute("INSERT INTO order_rows VALUES (?,?,?,?,?,?,?,?,?,?)",
                    (ds, pid, prov, split, mix, i, sha(r["text"]), len(gl), len(located), ident))
        k = (split, ds.split(":")[0]); C[k + ("rows",)] += 1; C[k + ("ident",)] += ident; C[k + ("prov_" + prov.split("/")[0],)] += 1
    con.commit()
    out = open(TXT, "w")

    def say(s):
        print(s, flush=True); out.write(s + "\n")
    say(f"MASTER ORDER TABLE (ledger 2026-10-26 19:27): {DB}; {len(rows)} rows; {time.time() - t0:.0f}s")
    groups = sorted({k[:2] for k in C})
    for g in groups:
        n = C[g + ("rows",)]; ng = sum(C[g + ("loc_" + h,)] for h in ("mention", "value", "lexical", "unstated"))
        if not n:
            continue
        prov = ", ".join(f"{p} {C[g + ('prov_' + p,)]}" for p in ("src_idx-verified", "exact", "unsubstituted", "template", "template-ambiguous", "none", "synthetic") if C[g + ("prov_" + p,)])
        say(f"  {g[0]:7s} {g[1]:12s} rows {n:6d} | mention order = derivation order {C[g + ('ident',)] / n:.3f} | givens {ng}: "
            f"mention {C[g + ('loc_mention',)] / ng:.3f} value {C[g + ('loc_value',)] / ng:.3f} lexical {C[g + ('loc_lexical',)] / ng:.3f} unstated {C[g + ('loc_unstated',)] / ng:.3f} | provenance: {prov}")
    for split in ("train", "holdout"):
        q = con.execute("SELECT COUNT(*), SUM(identical) FROM order_rows WHERE split=?", (split,)).fetchone()
        g = con.execute("SELECT COUNT(*), SUM(locate!='unstated'), SUM(multi_occ) FROM order_map WHERE split=?", (split,)).fetchone()
        say(f"  ALL {split}: rows {q[0]}, identical {q[1] / q[0]:.3f}; givens {g[0]}, located {g[1] / g[0]:.3f}, multi_occ {g[2] / g[0]:.3f}")
    leak = con.execute("SELECT COUNT(DISTINCT problem_idx) FROM order_rows WHERE split='train' AND dataset='gsm8k-test'").fetchone()[0]
    say(f"  READ: training rows whose source is GSM8K TEST: {leak} distinct problems")
