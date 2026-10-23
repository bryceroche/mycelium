"""prev_subject_check.py -- THE GSM8K PREVIOUS-SUBJECT CHECK (2026-10-23; Bryce's word; ledger 16:23). Zero GPU.
How often does T2k's role rule ('a pronoun's antecedent is an entity of the PREVIOUS sentence: its subject if
pronoun-compatible, else its next compatible noun phrase') hold on real GSM8K-train pronouns?
  sample   150 third-person pronouns from GSM8K-TRAIN questions (seed 1023; wild-fixture texts excluded; answer field never
           read). Writes the Haiku input (text + [[pronoun]] only), the coordinator's blind gold-30 sheet, and the rule's
           predictions (HIDDEN from both, .cache/prevsubj_rule.jsonl).
  score    reads Haiku's labels + the coordinator's gold-30 -> validity gate (>= 26/30) and R1 / R2 / R3.
Matching: an antecedent is identified by its HEAD word (last alphabetic token of the span, lowercased); a label and a
prediction agree when their head words match. R1 location = the sentence of the nearest preceding occurrence of the
antecedent's head word."""
import glob, json, os, random, re, sys

R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
C = lambda f: os.path.join(R, ".cache", f)
F_IN, F_RULE, F_GOLD, F_HAIKU, F_OUT = (C(f) for f in ("prevsubj_haiku_in.jsonl", "prevsubj_rule.jsonl", "prevsubj_gold30.txt",
                                                       "prevsubj_haiku_out.jsonl", "prevsubj_check.txt"))
PRONS = {"he": "m", "him": "m", "his": "m", "she": "f", "her": "f", "they": "pl", "them": "pl", "their": "pl", "it": "n", "its": "n"}
PERSON_M = set("man boy father dad brother son uncle grandpa grandfather husband king mr gentleman nephew".split())
PERSON_F = set("woman girl mother mom mum sister daughter aunt grandma grandmother wife queen mrs ms lady niece".split())
PERSON_ANY = set("person teacher student friend farmer baker chef doctor driver kid child worker player owner customer "
                 "neighbor coach manager clerk artist author".split())
nl = lambda f: {l.strip() for l in open(C("names_corpus_dl/names/" + f)) if l.strip()}
NM, NF = nl("male.txt"), nl("female.txt")
head = lambda s: (re.findall(r"[A-Za-z]+", s or "") or [""])[-1].lower()


def compat(chunk, cls):
    """is a spaCy noun chunk compatible with the pronoun class?"""
    r = chunk.root; w = r.text; lw = w.lower()
    if r.pos_ == "PRON":
        return False
    plural = r.tag_ in ("NNS", "NNPS") or " and " in chunk.text
    if cls == "pl":
        return plural
    if plural:
        return False
    name_m, name_f = w in NM and w not in NF, w in NF and w not in NM
    person = r.pos_ == "PROPN" or lw in PERSON_M | PERSON_F | PERSON_ANY
    if cls == "m":
        return name_m or lw in PERSON_M or (person and not name_f and lw not in PERSON_F)
    if cls == "f":
        return name_f or lw in PERSON_F or (person and not name_m and lw not in PERSON_M)
    return not person                                                       # 'it': a non-person singular


def rule(doc, tok):
    sents = list(doc.sents); si = next(i for i, s in enumerate(sents) if s.start <= tok.i < s.end)
    if si == 0:
        return None, "first_sentence", 0
    prev = sents[si - 1]; cls = PRONS[tok.text.lower()]
    chunks = [c for c in doc.noun_chunks if prev.start <= c.start < prev.end]
    subj = [c for c in chunks if c.root.dep_ in ("nsubj", "nsubjpass")]
    order = subj[:1] + [c for c in chunks if c not in subj[:1]]
    ok = [c for c in order if compat(c, cls)]
    return (ok[0].text if ok else None), ("subject" if ok and subj and ok[0] == subj[0] else ("other_np" if ok else "none")), len(ok)


def sample():
    for p in (F_IN, F_RULE, F_GOLD):
        assert not os.path.exists(p), f"refuse overwrite {p}"
    import pyarrow.parquet as pq, spacy
    norm = lambda s: re.sub(r"\s+", " ", s.strip().lower())
    fixtures = set()
    for f in glob.glob(C("wild_*holdout*.jsonl")):
        fixtures |= {norm(json.loads(l)["text"]) for l in open(f)}
    qs = pq.read_table(glob.glob(C("gsm8k/datasets--openai--gsm8k/snapshots/*/main/train-*.parquet"))[0], columns=["question"]).column("question").to_pylist()
    qs = [q for q in qs if norm(q) not in fixtures]
    rnd = random.Random(1023); pool = rnd.sample(qs, 600)
    nlp = spacy.load("en_core_web_sm")
    cands = [(q, doc, t) for q, doc in zip(pool, nlp.pipe(pool, batch_size=64)) for t in doc if t.text.lower() in PRONS and t.pos_ == "PRON"]
    pick = rnd.sample(range(len(cands)), 150)
    rows = []
    for n, k in enumerate(pick):
        q, doc, t = cands[k]
        pred, kind, ncomp = rule(doc, t)
        si = next(i for i, s in enumerate(doc.sents) if s.start <= t.i < s.end)
        rows.append(dict(id=f"ps{n}", text=q, pron=t.text, off=t.idx, sent=si, rule_pred=pred, rule_kind=kind, prev_ncompat=ncomp,
                         sent_bounds=[(s.start_char, s.end_char) for s in doc.sents]))
    mark = lambda r: r["text"][:r["off"]] + "[[" + r["pron"] + "]]" + r["text"][r["off"] + len(r["pron"]):]
    with open(F_RULE, "w") as f:
        for r in rows: f.write(json.dumps(r) + "\n")
    with open(F_IN, "w") as f:
        for r in rows: f.write(json.dumps(dict(id=r["id"], text=mark(r))) + "\n")
    g = random.Random(4).sample(rows, 30)
    with open(F_GOLD, "w") as f:
        f.write("GOLD-30 (coordinator, blind to Haiku and to the rule): for each [[pronoun]] write the antecedent as it appears in "
                "the text (the entity's name or noun phrase), or NONE (pleonastic/generic/no antecedent).\n\n")
        for r in g: f.write(f"{r['id']}\n{mark(r)}\nANTECEDENT: \n\n")
    print("candidates", len(cands), "sampled", len(rows), "first-sentence pronouns", sum(r["rule_kind"] == "first_sentence" for r in rows))


def score():
    assert not os.path.exists(F_OUT), "refuse overwrite"
    rows = {json.loads(l)["id"]: json.loads(l) for l in open(F_RULE)}
    hk = {json.loads(l)["id"]: json.loads(l).get("antecedent") for l in open(F_HAIKU) if l.strip()}
    gold = dict(re.findall(r"^(ps\d+)\n.*?\nANTECEDENT: (.*)$", open(F_GOLD).read(), re.M))
    isnone = lambda a: a is None or str(a).strip().upper() in ("", "NONE", "NULL")
    agree = sum((isnone(gold[k]) and isnone(hk.get(k))) or (not isnone(gold[k]) and not isnone(hk.get(k)) and head(gold[k]) == head(hk[k]))
                for k in gold)
    out = [f"VALIDITY GATE: Haiku vs coordinator gold-30: {agree}/{len(gold)} (bar >= 26) -> {'PASS' if agree >= 26 else 'VOID'}"]
    loc_counts, r2, r3 = {}, [0, 0], [0, 0]
    for k, r in rows.items():
        a = hk.get(k)
        if k not in hk:
            loc = "missing"
        elif isnone(a):
            loc = "none"
        else:
            hw = head(a); txt = r["text"][:r["off"]]
            pos = max((m.start() for m in re.finditer(r"\b" + re.escape(hw) + r"\b", txt, re.I)), default=-1)
            if pos < 0:
                loc = "not_found"
            else:
                ls = next(i for i, (s0, s1) in enumerate(r["sent_bounds"]) if s0 <= pos < s1 or (i == len(r["sent_bounds"]) - 1))
                loc = "same" if ls == r["sent"] else ("previous" if ls == r["sent"] - 1 else "earlier")
        loc_counts[loc] = loc_counts.get(loc, 0) + 1
        if k in hk:
            ok = (isnone(a) and r["rule_pred"] is None) or (not isnone(a) and r["rule_pred"] is not None and head(a) == head(r["rule_pred"]))
            r2[0] += ok; r2[1] += 1
            if r["prev_ncompat"] >= 2:
                r3[0] += ok; r3[1] += 1
    n = sum(v for kk, v in loc_counts.items() if kk != "missing")
    out.append("R1 antecedent location (nearest preceding mention): " + ", ".join(f"{kk} {v} ({v / max(1, n):.2f})" for kk, v in sorted(loc_counts.items())))
    out.append(f"R2 the rule's accuracy, all pronouns: {r2[0]}/{r2[1]} = {r2[0] / max(1, r2[1]):.3f}")
    out.append(f"R3 the rule's accuracy, role-like slice (>= 2 compatible candidates in the previous sentence): {r3[0]}/{r3[1]} = {r3[0] / max(1, r3[1]):.3f}")
    print("\n".join(out)); open(F_OUT, "w").write("\n".join(out) + "\n")


if __name__ == "__main__":
    {"sample": sample, "score": score}[sys.argv[1]]()
