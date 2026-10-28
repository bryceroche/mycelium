"""eq_dsl.py -- THE DSL HANDOFF, step 1: a fixed (unlearned) DSL reader + the census (ledger 2026-10-28, word given; zero GPU).
Bryce's design: the GSM8K text is untouched; the NL reader hands the breathing transformer ONLY discrete DSL symbols, so the
reasoner never sees English surface. This step builds a DETERMINISTIC DSL reader (spaCy en_core_web_sm + fixed rules; nothing
learned, so nothing memorized) that renders each located given mention as a code tuple, in text order:
  ent    the mention's sentence subject: a person -> E1, E2.. by first appearance; he/she/his/her/him -> E1 (GSM8K pronouns
         follow the protagonist, ledger 10-23); they/we/them -> EG; I/me/my -> EI; none -> E0
  obj    the counted noun (nummod head, else the next noun within 3 tokens) -> O1, O2.. by first appearance; '$' -> OMONEY
  verb   the governing verb's lemma if among the corpus's top VERB_TOP verbs, else VOTHER
  cues   flags in a +/-4 token window: per/each/every, times/twice/double/triple, half/third/quarter, more/less/fewer/than,
         percent, total/altogether/combined, left/remain/rest
  form   int / frac / dec / pct / money;  mag  log10 bucket of the value;  sent  sentence index;  q  in the question sentence
plus ONE query code: the question sentence's subject + asked noun + verb + cues. No word, name or number crosses.
Writes .cache/eq_dsl.json {split: {row: {"m": [mention code dicts], "q": query code dict}}} + .cache/eq_dsl_census.txt:
  how many problems share a DSL skeleton (the generalization available), whether a shared skeleton fixes the order (the
  DSL's determinacy), and how much of the holdout's skeletons the training set covers."""
import collections, json, math, os, re, sqlite3, sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ".")
import eq_slot_grade as EG

VERB_TOP = int(os.environ.get("DSL_VERB_TOP", "60"))
OUT, CENSUS = ".cache/eq_dsl.json", ".cache/eq_dsl_census.txt"
CUES = {"per": {"per", "each", "every", "apiece"}, "times": {"times", "twice", "double", "triple", "thrice"},
        "part": {"half", "third", "quarter", "fourth", "fifth", "fraction"}, "cmp": {"more", "less", "fewer", "than", "longer", "older"},
        "pct": {"percent", "%"}, "tot": {"total", "altogether", "combined", "together", "sum"}, "left": {"left", "remain", "remaining", "rest"}}
PRON1, PRONG, PRONI = {"he", "she", "his", "her", "him", "hers"}, {"they", "we", "them", "their", "us", "our"}, {"i", "me", "my"}


def problem_key(smp):
    g = smp.get("gen")
    if isinstance(g, dict) and "src_idx" in g: return f"{g.get('src')}:{g['src_idx']}"
    if isinstance(g, dict) and "pool" in g: return f"{g.get('src')}:pool:{json.dumps(g['pool'], sort_keys=True)}"
    return "mask:" + re.sub(r"\d+(\.\d+)?", "#", smp["text"].strip())


def subject(sent):
    for t in sent:
        if t.dep_ in ("nsubj", "nsubjpass") and t.head.dep_ == "ROOT": return t
    for t in sent:
        if t.dep_ in ("nsubj", "nsubjpass"): return t
    return None


def gov_verb(t):
    h = t
    for _ in range(8):
        if h.pos_ in ("VERB", "AUX") or h.dep_ == "ROOT": return h.lemma_.lower()
        if h.head is h: break
        h = h.head
    return "none"


def render(doc, cands, vals, verbs_ok):
    """cands: located char_starts (text order); vals: their values. -> (mention codes, query code)."""
    ents, objs = {}, {}
    sents = list(doc.sents); qs = next((i for i, s in enumerate(sents) if "?" in s.text or s[0].lower_ in ("how", "what")), len(sents) - 1)

    def ent_of(sent):
        s = subject(sent)
        if s is None: return "E0"
        w = s.lower_
        if w in PRON1: return "E1" if ents else "E0"
        if w in PRONG: return "EG"
        if w in PRONI: return "EI"
        if s.pos_ == "PROPN" or s.ent_type_ == "PERSON":
            key = s.lemma_.lower()
            if key not in ents: ents[key] = f"E{len(ents) + 1}"
            return ents[key]
        return "ENOUN"

    def obj_of(tok):
        if tok.i > 0 and doc[tok.i - 1].text == "$": return "OMONEY"
        if tok.dep_ == "nummod" and tok.head.pos_ in ("NOUN", "PROPN"): n = tok.head
        else: n = next((doc[j] for j in range(tok.i + 1, min(tok.i + 4, len(doc))) if doc[j].pos_ in ("NOUN",) and doc[j].sent == tok.sent), None)
        if n is None: return "O0"
        key = n.lemma_.lower()
        if key not in objs: objs[key] = f"O{len(objs) + 1}"
        return objs[key]

    def cues_of(lo, hi, sent):
        win = {doc[j].lower_ for j in range(max(lo, sent.start), min(hi, sent.end))}
        return tuple(sorted(k for k, ws in CUES.items() if win & ws))

    out = []
    for cs, v in zip(cands, vals):
        tok = next((t for t in doc if t.idx <= cs < t.idx + len(t.text)), None) or next(t for t in doc if t.idx >= cs)
        sent = tok.sent; si = next(i for i, s in enumerate(sents) if s.start <= tok.i < s.end)
        vb = gov_verb(tok); txt = tok.text
        form = ("money" if tok.i > 0 and doc[tok.i - 1].text == "$" else "pct" if tok.i + 1 < len(doc) and doc[tok.i + 1].lower_ in ("%", "percent")
                else "frac" if "/" in txt or tok.lower_ in CUES["part"] else "dec" if "." in txt else "int")
        fv = float(v) if not isinstance(v, str) else 0.0
        mag = "m0" if fv < 1 else f"m{min(4, int(math.log10(fv)) + 1)}"
        out.append(dict(ent=ent_of(sent), obj=obj_of(tok), verb=vb if vb in verbs_ok else "VOTHER", cues=cues_of(tok.i - 4, tok.i + 5, sent),
                        form=form, mag=mag, sent=si, q=int(si == qs)))
    qsent = sents[qs]
    qn = next((t for t in qsent if t.pos_ == "NOUN"), None)
    qobj = "O0" if qn is None else objs.get(qn.lemma_.lower(), "ONEW")
    qverb = next((t.lemma_.lower() for t in qsent if t.pos_ == "VERB"), "none")
    q = dict(ent=ent_of(qsent), obj=qobj, verb=qverb if qverb in verbs_ok else "VOTHER", cues=cues_of(qsent.start, qsent.end, qsent))
    return out, q


def skeleton(r):
    return (tuple((m["ent"], m["obj"], m["verb"], m["cues"], m["form"], m["mag"], m["sent"], m["q"]) for m in r["m"]),
            (r["q"]["ent"], r["q"]["obj"], r["q"]["verb"], r["q"]["cues"]))


if __name__ == "__main__":
    import spacy
    nlp = spacy.load("en_core_web_sm")
    c = sqlite3.connect(EG.DB); rows = {}
    for split in ("train", "holdout"):
        by = {}
        for r, eq, val, cs, loc in c.execute("select row,eq_slot,value,char_start,locate from order_map where split=? order by row,eq_slot", (split,)):
            if loc != "unstated": by.setdefault(r, []).append((eq, cs, val))
        texts = {i: json.loads(l) for i, l in enumerate(open(EG.MIX[split])) if i in by}
        rows[split] = (by, texts)
    # the verb vocabulary: governing-verb lemmas over the TRAINING texts only (fixed before any rendering)
    vc = collections.Counter()
    by, texts = rows["train"]; docs = {}
    for i, d in zip(texts, nlp.pipe((t["text"] for t in texts.values()), batch_size=256)):
        docs[("train", i)] = d
        for t in d:
            if t.like_num: vc[gov_verb(t)] += 1
    verbs_ok = {v for v, _ in vc.most_common(VERB_TOP)} - {"none"}
    by_h, texts_h = rows["holdout"]
    for i, d in zip(texts_h, nlp.pipe((t["text"] for t in texts_h.values()), batch_size=256)): docs[("holdout", i)] = d
    dsl = {}; n_fail = 0
    for split in ("train", "holdout"):
        by, texts = rows[split]; dsl[split] = {}
        for i, smp in texts.items():
            st = sorted(by[i], key=lambda s: s[1]); cands = [s[1] for s in st]; vals = [s[2] for s in st]
            try: m, q = render(docs[(split, i)], cands, vals, verbs_ok)
            except StopIteration: n_fail += 1; continue
            tgt_text_rank = [cands.index(s[1]) for s in sorted(by[i])]   # derivation order -> text rank (the eq_slot target)
            dsl[split][i] = dict(m=m, q=q, tgt=tgt_text_rank, key=problem_key(smp), src=(smp.get("gen") or {}).get("src") if isinstance(smp.get("gen"), dict) else None)
    json.dump({s: {str(k): v for k, v in d.items()} for s, d in dsl.items()}, open(OUT, "w"))
    # THE CENSUS
    L = []
    say = lambda s: (print(s, flush=True), L.append(s))
    tr = dsl["train"]; ho = dsl["holdout"]
    allm = [m for r in tr.values() for m in r["m"]]
    say(f"rendered train {len(tr)} rows, holdout {len(ho)} rows, render failures {n_fail} | verbs kept {len(verbs_ok)} "
        f"(top: {', '.join(v for v, _ in vc.most_common(12))})")
    say(f"field coverage (train mentions {len(allm)}): ent!=E0 {np.mean([m['ent'] != 'E0' for m in allm]):.3f} | obj!=O0 "
        f"{np.mean([m['obj'] != 'O0' for m in allm]):.3f} | verb kept {np.mean([m['verb'] != 'VOTHER' for m in allm]):.3f} | any cue "
        f"{np.mean([bool(m['cues']) for m in allm]):.3f}")
    gs = {}   # one row per GSM8K problem (the capped view)
    for i, r in sorted(tr.items()):
        if r["src"] == "gsm8k" and r["key"] not in gs: gs[r["key"]] = r
    sk = collections.defaultdict(list)
    for k, r in gs.items(): sk[skeleton(r)].append(r)
    shared = [v for v in sk.values() if len(v) >= 2]
    say(f"GSM8K train problems {len(gs)} -> distinct DSL skeletons {len(sk)} | problems in a shared skeleton {sum(map(len, shared))} "
        f"({sum(map(len, shared)) / len(gs):.3f}) over {len(shared)} skeletons")
    agree = [r["tgt"] == collections.Counter(tuple(x["tgt"]) for x in v).most_common(1)[0][0] for v in shared for r in v]
    agree = [r["tgt"] == list(collections.Counter(tuple(x["tgt"]) for x in v).most_common(1)[0][0]) for v in shared for r in v]
    say(f"DETERMINACY: within shared skeletons, problems whose order = the skeleton's majority order {np.mean(agree) if agree else float('nan'):.3f} "
        f"(n {len(agree)})")
    # coarser skeleton (drop mag + sent): how much structure repeats at the level of roles alone
    sk2 = collections.defaultdict(list)
    for r in gs.values(): sk2[tuple((m["ent"], m["obj"], m["verb"], m["cues"], m["form"], m["q"]) for m in r["m"])].append(r)
    sh2 = [v for v in sk2.values() if len(v) >= 2]
    ag2 = [r["tgt"] == list(collections.Counter(tuple(x["tgt"]) for x in v).most_common(1)[0][0]) for v in sh2 for r in v]
    say(f"coarse skeleton (no mag/sent/query): {len(sk2)} distinct | in shared {sum(map(len, sh2)) / len(gs):.3f} | determinacy {np.mean(ag2):.3f} (n {len(ag2)})")
    hk = [skeleton(r) in sk for r in ho.values()]
    hit = [r["tgt"] == list(collections.Counter(tuple(x["tgt"]) for x in sk[skeleton(r)]).most_common(1)[0][0]) for r in ho.values() if skeleton(r) in sk]
    say(f"HOLDOUT: skeleton seen in GSM8K train {np.mean(hk):.3f} ({sum(hk)}/{len(hk)}) | of those, train majority order = gold {np.mean(hit) if hit else float('nan'):.3f}")
    say(f"identity order on holdout {np.mean([r['tgt'] == list(range(len(r['tgt']))) for r in ho.values()]):.3f} | mean mentions "
        f"{np.mean([len(r['m']) for r in ho.values()]):.2f}")
    open(CENSUS, "w").write("\n".join(L) + "\n")
