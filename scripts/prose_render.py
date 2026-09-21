"""prose_render.py -- THE PROSE RENDERER (2026-09-21, word given after THE ARGUMENT-BINDING
CENSUS 2026-09-20 09:49: the arg-mention stamps reach only 56-66% of relation slots and the
wall is same-sentence argument SELECTION, not lexical reach -- 91.6% of wild arguments are
already anchored and the pointer misses them anyway). This renders GSM8K-shaped narratives
DIRECTLY FROM sampled factor graphs -- templates + a lexicon, never an LLM, never an external
call -- so every span (clause, value mention, argument re-mention, cue) is gold BY
CONSTRUCTION: recorded as character offsets while the text is assembled, never recovered by
re-parsing the finished string (the one exception, stated: cue_spans are found by scanning the
just-built clause for the census's own CUE_WORDS/CUE_PHRASES, reused verbatim from
scripts/args_census.py -- safe because the text is ours and the words are the ones we chose to
insert, not an ambiguous parse).

SCOPE (stated, deliberately narrower than the full dialect): given + rel(add, mul) factors
only, exactly 2 args per relation, THE POSITIONAL LAW holds BY CONSTRUCTION (factor j always
introduces variable j -- no canonicalization pass is needed or run). No mod/sel/pct/fdiv/
macro/frac in this pass (registered for a v2 if the census asks for one). Values and answers
are kept <= 300 (the books' cap) via bounded resampling, well under the rulebook's VALUE_CAP
(9999).

THE GATE: every row is a candidate until it passes admit_annotation.admit() unchanged (the
SAME rulebook + THE SAME CSP solver forcing the answer key that gates the pen/silver corpora --
no separate door is built for rendered rows). Since the graph is forward-evaluated by
construction the key is definitionally reachable, but the row must still clear the structural
rulebook (span bounds, capacity, pointers) and the solver's own domain ladder -- a row that
does not is refused and counted, exactly like any other source.

usage:
  .venv/bin/python3 scripts/prose_render.py --n 5000 --seed 41 --out .cache/render_v1_5k.jsonl
  .venv/bin/python3 scripts/prose_render.py --n 200 --seed 99 --out .cache/render_v1_val.jsonl
"""
import argparse
import json
import random
import re
import sys

sys.path.insert(0, ".")
sys.path.insert(0, "scripts")
from admit_annotation import admit  # noqa: E402
import args_census as census        # noqa: E402  (reuse CUE_WORDS/CUE_PHRASES verbatim)

VALUE_CAP_SOFT = 300     # the books' convention; the rulebook allows up to 9999
MAX_TRIES_GRAPH = 500    # a sampling attempt gives up and the caller reseeds

# --------------------------------------------------------------------------- lexicon

NAMES = [
    ("Tom", "m"), ("Jerry", "m"), ("Sam", "m"), ("Mike", "m"), ("Ben", "m"), ("Alex", "m"),
    ("Jack", "m"), ("Paul", "m"), ("Ray", "m"), ("Leo", "m"), ("Carlos", "m"), ("Kevin", "m"),
    ("Danny", "m"), ("Eric", "m"), ("Frank", "m"), ("Oscar", "m"),
    ("Maria", "f"), ("Anna", "f"), ("Julia", "f"), ("Emma", "f"), ("Sara", "f"), ("Lucy", "f"),
    ("Nina", "f"), ("Grace", "f"), ("Olivia", "f"), ("Ella", "f"), ("Kate", "f"), ("Rosa", "f"),
    ("Wendy", "f"), ("Diana", "f"), ("Carla", "f"), ("Ivy", "f"),
]
PRONOUNS = {"m": ("he", "him", "his"), "f": ("she", "her", "her")}  # (subject, object, possessive)

OBJECTS = [
    ("apple", "apples"), ("sticker", "stickers"), ("dollar", "dollars"), ("marble", "marbles"),
    ("book", "books"), ("cookie", "cookies"), ("pencil", "pencils"), ("stamp", "stamps"),
    ("card", "cards"), ("egg", "eggs"), ("cupcake", "cupcakes"), ("ticket", "tickets"),
    ("page", "pages"), ("balloon", "balloons"), ("candy", "candies"), ("shell", "shells"),
    ("coin", "coins"), ("flower", "flowers"), ("toy", "toys"), ("bead", "beads"),
    ("pen", "pens"), ("crayon", "crayons"), ("photo", "photos"), ("stone", "stones"),
]
CONTAINERS = [("box", "boxes"), ("bag", "bags"), ("basket", "baskets"), ("crate", "crates"), ("shelf", "shelves")]

ONES_W = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
          "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"]
TENS_W = {2: "twenty", 3: "thirty", 4: "forty", 5: "fifty", 6: "sixty", 7: "seventy", 8: "eighty", 9: "ninety"}

MUL_ADVERBS = {2: ["twice", "double"], 3: ["triple"], 4: ["quadruple"]}

DISTRACTOR_TEMPLATES = [
    "{name} also has a pet dog.",
    "{name} likes to read on weekends.",
    "It takes {name} ten minutes to walk to school.",
    "{name}'s friend has a red bicycle.",
    "The weather was sunny that day.",
    "{name} plays soccer every Saturday.",
    "{name} usually wakes up at seven.",
]


def number_word(v):
    """A spelled-out form for v, or None if v is too large to spell naturally."""
    if v == 12 and random.random() < 0.4:
        return "a dozen"
    if 0 <= v < 20:
        return ONES_W[v]
    if v < 100 and v % 10 == 0:
        return TENS_W[v // 10]
    if v < 100:
        return TENS_W[v // 10] + "-" + ONES_W[v % 10]
    return None


def value_text(v, word_prob=0.20):
    """Returns (text, is_word). Falls back to digits when no word form exists."""
    if random.random() < word_prob:
        w = number_word(v)
        if w is not None:
            return w, True
    return str(v), False


def mul_operand_text(v):
    """The multiplier's own rendering. mode='adverb': a bare word ('twice', 'triple') that
    already means '<v> times' on its own -- comparative_pieces must NOT also say 'times'.
    mode='times': a bare magnitude ('three', '5') that DOES need the template's own ' times'."""
    if v in MUL_ADVERBS and random.random() < 0.5:
        return random.choice(MUL_ADVERBS[v]), "adverb"
    w = number_word(v) if random.random() < 0.5 else None
    return (w if w is not None else str(v)), "times"


# --------------------------------------------------------------------------- graph sampling

def sample_graph(rng):
    """Builds a positional factor graph: factor j always introduces variable j.
    Returns (factors, values, query_var, key) where `factors` carry an internal
    `role` in {base, operand, together, comparative, continuation} consumed only
    by the renderer (stripped before the row is written)."""
    n_factors = rng.choice([4, 5, 5, 6, 6, 6, 7, 7, 8])
    factors, values, person_vars = [], [], []

    def add_base():
        v = rng.randint(2, 60)
        factors.append({"ftype": "given", "role": "base", "value": v})
        values.append(v)
        person_vars.append(len(factors) - 1)

    def try_together():
        if len(person_vars) < 2:
            return False
        a, b = rng.sample(person_vars, 2)
        r = values[a] + values[b]
        if not (0 < r <= VALUE_CAP_SOFT):
            return False
        factors.append({"ftype": "rel", "role": "together", "op": "add", "args": [a, b]})
        values.append(r)
        person_vars.append(len(factors) - 1)
        return True

    def try_comparative_or_continuation(remaining):
        style = "continuation" if (rng.random() < 0.20) else "comparative"
        op = rng.choices(["add", "mul"], weights=[0.6, 0.4])[0]
        if style == "continuation":
            op = "add"
            a = person_vars[-1]
        else:
            fresh_ok = person_vars and person_vars[-1] == len(factors) - 1
            a = person_vars[-1] if (fresh_ok and rng.random() < 0.6) else rng.choice(person_vars)
        av = values[a]
        if op == "add":
            bv = rng.randint(2, 30)
            r = av + bv
        else:
            bv = rng.choice([2, 3, 4, 5, 6])
            r = av * bv
        if not (0 < r <= VALUE_CAP_SOFT):
            return False
        factors.append({"ftype": "given", "role": "operand", "value": bv})
        values.append(bv)
        rel_idx = len(factors)
        factors.append({"ftype": "rel", "role": style, "op": op, "args": [a, rel_idx - 1]})
        values.append(r)
        person_vars.append(rel_idx)
        if style == "continuation":
            # 'a's identity is SUPERSEDED (same person, later time) -- retire the stale index so a
            # later 'together'/comparative draw can never combine a person with their own earlier
            # count (the double-counting this caught: "Sam has 56 ... finds 26 more" then a later
            # draw adding var(56) and var(82) together as if they were two different people).
            person_vars.remove(a)
        return True

    add_base()
    add_base()
    tries = 0
    while len(factors) < n_factors:
        tries += 1
        if tries > MAX_TRIES_GRAPH:
            raise RuntimeError("graph sampling exhausted its budget")
        remaining = n_factors - len(factors)
        if remaining == 1:
            if not (len(person_vars) >= 2 and rng.random() < 0.7 and try_together()):
                add_base()
            continue
        r = rng.random()
        if r < 0.10 and len(person_vars) >= 2:
            try_together()
        elif r < 0.32:
            add_base()
        else:
            try_comparative_or_continuation(remaining)

    # prefer asking about a COMPUTED quantity (never a value stated verbatim in the text --
    # a "base" given's question would be trivially answerable by reading its own sentence)
    computed = [i for i, f in enumerate(factors) if f["role"] in ("together", "comparative", "continuation")]
    candidates = computed or [i for i, f in enumerate(factors) if f["role"] != "operand"]
    weights = [5 if i == candidates[-1] else 1 for i in candidates]
    query_var = rng.choices(candidates, weights=weights)[0]
    key = values[query_var]
    return factors, values, query_var, key


# --------------------------------------------------------------------------- entities

def assign_entities(rng, factors):
    obj_sing, obj_plur = rng.choice(OBJECTS)
    pool = NAMES[:]
    rng.shuffle(pool)
    used = {"i": 0}

    def next_name():
        nm = pool[used["i"] % len(pool)]
        used["i"] += 1
        return nm

    entities = [None] * len(factors)
    for i, f in enumerate(factors):
        role = f["role"]
        if role == "operand":
            continue
        if role == "base":
            name, g = next_name()
            entities[i] = {"name": name, "gender": g, "kind": "person"}
        elif role == "together":
            entities[i] = {"name": "they", "gender": None, "kind": "group"}
        elif role == "continuation":
            a = f["args"][0]
            entities[i] = dict(entities[a])
        elif role == "comparative":
            name, g = next_name()
            entities[i] = {"name": name, "gender": g, "kind": "person"}
    return entities, obj_sing, obj_plur


# --------------------------------------------------------------------------- doc / span plumbing

class Doc:
    def __init__(self):
        self.text = ""

    def pos(self):
        return len(self.text)

    def add(self, s):
        a = len(self.text)
        self.text += s
        return (a, len(self.text))


def emit(doc, pieces):
    """pieces: [(text, tag_or_tags_or_None), ...]; tags may repeat across pieces.
    Returns (clause_span, spans) where spans: tag -> [[s, e], ...], clause_span
    trimmed of a single trailing space (never of interior punctuation)."""
    spans = {}
    start = doc.pos()
    for text, tag in pieces:
        s, e = doc.add(text)
        if tag:
            for t in (tag if isinstance(tag, (list, tuple)) else [tag]):
                spans.setdefault(t, []).append([s, e])
    end = doc.pos()
    while end > start and doc.text[end - 1] == " ":
        end -= 1
    return (start, end), spans


CUE_RE = None


def _cue_re():
    global CUE_RE
    if CUE_RE is None:
        words = sorted(census.CUE_WORDS, key=len, reverse=True)
        phrases = sorted(census.CUE_PHRASES, key=len, reverse=True)
        alts = [re.escape(p) for p in phrases] + [r"\b" + re.escape(w) + r"\b" for w in words]
        CUE_RE = re.compile("|".join(alts), re.I)
    return CUE_RE


def cue_spans_in(text, clause_start, clause_end):
    out = []
    for m in _cue_re().finditer(text[clause_start:clause_end]):
        out.append([clause_start + m.start(), clause_start + m.end()])
    return out


# --------------------------------------------------------------------------- sentence templates

def mention_piece(entity, tag, pronoun_rate=0.06, subj=False):
    """subj=True: the mention sits in subject position (he/she/they); subj=False:
    object position (him/her/them, e.g. after 'than' or 'as')."""
    if entity["kind"] == "group":
        return ("they" if subj else "them", tag)
    if entity["kind"] == "person" and random.random() < pronoun_rate:
        idx = 0 if subj else 1
        return (PRONOUNS[entity["gender"]][idx], tag)
    return (entity["name"], tag)


def base_pieces(name, value_piece, obj_plur):
    tmpl = random.choice([
        lambda: [(f"{name} has ", None), value_piece, (f" {obj_plur}", None)],
        lambda: [(f"{name} bought ", None), value_piece, (f" {obj_plur}", None)],
        lambda: [(f"{name} started with ", None), value_piece, (f" {obj_plur}", None)],
        lambda: [(f"{name} has collected ", None), value_piece, (f" {obj_plur} so far", None)],
        lambda: [(f"{name} owns ", None), value_piece, (f" {obj_plur}", None)],
    ])
    return tmpl()


def comparative_pieces(op, new_name, obj_plur, b_piece, a_mention_piece, b_mode="times"):
    if op == "add":
        tmpl = random.choice([
            lambda: [(f"{new_name} has ", None), b_piece, (" more ", None), (obj_plur, None), (" than ", None), a_mention_piece],
            lambda: [(f"{new_name} collected ", None), b_piece, (" additional ", None), (obj_plur, None), (" compared to ", None), a_mention_piece],
        ])
    else:
        obj_phrase = "as much " + obj_plur if obj_plur == "dollars" else "as many " + obj_plur
        middle = f" {obj_phrase} as " if b_mode == "adverb" else f" times {obj_phrase} as "
        tmpl = random.choice([
            lambda: [(f"{new_name} has ", None), b_piece, (middle, None), a_mention_piece],
            lambda: [(new_name, None), (" has ", None), b_piece, (middle, None), a_mention_piece],
        ])
    return tmpl()


def together_pieces(obj_plur, a_mention_piece, b_mention_piece):
    cue = random.choice(["together", "in total", "combined", "altogether"])
    tmpl = random.choice([
        lambda: [a_mention_piece, (" and ", None), b_mention_piece, (f" have some {obj_plur} {cue}", None)],
        lambda: [a_mention_piece, (" and ", None), b_mention_piece, (f" put their {obj_plur} {cue}", None)],
    ])
    return tmpl()


def continuation_pieces(op, subj_piece, obj_plur, b_piece, merged, plural=False):
    """merged=True: joins onto a preceding clause with ', and then ...' (lowercase);
    merged=False: opens its own sentence with 'Then ...' (capitalized). plural=True when the
    subject is a group ('they') and needs the plural verb form."""
    lead = "and then " if merged else "Then "
    find_v, collect_v = ("find", "collect") if plural else ("finds", "collects")
    if op == "add":
        return [(lead, None), subj_piece, (f" {find_v} ", None), b_piece, (f" more {obj_plur}", None)]
    return [(lead, None), subj_piece, (f" {collect_v} ", None), b_piece, (f" times as many more {obj_plur}", None)]


def question_pieces(entity, obj_plur):
    if entity["kind"] == "group":
        return f"How many {obj_plur} do they have in total?"
    return f"How many {obj_plur} does {entity['name']} have?"


# --------------------------------------------------------------------------- rendering

def render_row(rng, row_id):
    factors, values, query_var, key = sample_graph(rng)
    entities, obj_sing, obj_plur = assign_entities(rng, factors)
    n = len(factors)

    out_factors = [None] * n
    mentions = {}
    doc = Doc()
    done = [False] * n

    for i, f in enumerate(factors):
        if done[i]:
            continue
        role = f["role"]
        if role == "base":
            # look ahead: is this base immediately consumed as the fresh 'a' of the very next
            # comparative/continuation relation two slots later, OR as an arg of a 'together'
            # relation the very next slot over? then merge into one sentence (this is the
            # renderer's main same-sentence-argument lever -- see THE ARGUMENT-BINDING CENSUS).
            merged = False
            if i + 2 < n and factors[i + 1]["role"] == "operand" and factors[i + 2]["role"] in ("comparative", "continuation"):
                if factors[i + 2]["args"][0] == i:
                    merged = True
            elif i + 1 < n and factors[i + 1]["role"] == "together" and i in factors[i + 1]["args"]:
                merged = True
            entity = entities[i]
            txt, is_word = value_text(f["value"])
            val_piece = (txt, f"val:{i}")
            pieces = base_pieces(entity["name"], val_piece, obj_plur)
            if not merged:
                (cs, ce), spans = emit(doc, pieces)
                out_factors[i] = {"ftype": "given", "var": i, "value": f["value"],
                                   "spans": [[cs, ce]], "arg_spans": [], "cue_spans": []}
                mentions[i] = spans.get(f"val:{i}", [])
                doc.add(". ")
                done[i] = True
            else:
                # defer emission: fold into the relation's own merged sentence below
                out_factors[i] = {"_pending_base": True, "value": f["value"]}
        elif role == "operand":
            rel_i = i + 1
            rel = factors[rel_i]
            a = rel["args"][0]
            fresh = out_factors[a] is not None and out_factors[a].get("_pending_base")
            entity_r = entities[rel_i]
            if rel["op"] == "mul":
                b_txt, b_mode = mul_operand_text(f["value"])
            else:
                b_txt, _ = value_text(f["value"]); b_mode = "times"
            b_piece = (b_txt, [f"val:{i}", f"argm:{rel_i}:1"])
            a_entity = entities[a]
            if fresh:
                # a's own given clause was DEFERRED (see the "base" branch's lookahead); build it
                # now and fold it into the SAME physical sentence as the relation it feeds.
                a_val_txt, _ = value_text(factors[a]["value"])
                lead = base_pieces(a_entity["name"], (a_val_txt, f"val:{a}"), obj_plur)
                if rel["role"] == "continuation":
                    subj = mention_piece(a_entity, f"argm:{rel_i}:0", subj=True)
                    tail = continuation_pieces(rel["op"], subj, obj_plur, b_piece, merged=True, plural=(a_entity["kind"] == "group"))
                else:
                    a_mention = mention_piece(a_entity, f"argm:{rel_i}:0", subj=False)
                    tail = comparative_pieces(rel["op"], entity_r["name"], obj_plur, b_piece, a_mention, b_mode)
                joiner = ", " if rel["role"] == "continuation" else ", and "
                pieces = lead + [(joiner, None)] + tail
                (cs, ce), spans = emit(doc, pieces)
                doc.add(". ")
                out_factors[a] = {"ftype": "given", "var": a, "value": factors[a]["value"],
                                   "spans": [[cs, ce]], "arg_spans": [], "cue_spans": []}
                mentions[a] = spans.get(f"val:{a}", [])
                done[a] = True
            else:
                if rel["role"] == "continuation":
                    subj = mention_piece(a_entity, f"argm:{rel_i}:0", subj=True)
                    pieces = continuation_pieces(rel["op"], subj, obj_plur, b_piece, merged=False, plural=(a_entity["kind"] == "group"))
                else:
                    a_mention = mention_piece(a_entity, f"argm:{rel_i}:0", subj=False)
                    pieces = comparative_pieces(rel["op"], entity_r["name"], obj_plur, b_piece, a_mention, b_mode)
                (cs, ce), spans = emit(doc, pieces)
                doc.add(". ")
            out_factors[i] = {"ftype": "given", "var": i, "value": f["value"],
                               "spans": [[cs, ce]], "arg_spans": [], "cue_spans": []}
            mentions[i] = spans.get(f"val:{i}", [])
            arg0 = spans.get(f"argm:{rel_i}:0", [])
            arg1 = spans.get(f"argm:{rel_i}:1", [])
            cue = cue_spans_in(doc.text, cs, ce)
            out_factors[rel_i] = {"ftype": "rel", "op": rel["op"], "args": rel["args"], "result": rel_i,
                                   "spans": [[cs, ce]], "arg_spans": [arg0, arg1], "cue_spans": cue}
            done[i] = True
            done[rel_i] = True
        elif role == "together":
            a, b = f["args"]
            pend, pend_pos = (None, None)
            for pos, v in ((0, a), (1, b)):
                if out_factors[v] is not None and out_factors[v].get("_pending_base"):
                    pend, pend_pos = v, pos
                    break
            if pend is not None:
                other = b if pend_pos == 0 else a
                other_pos = 1 - pend_pos
                pend_entity = entities[pend]
                val_txt, _ = value_text(factors[pend]["value"])
                lead = base_pieces(pend_entity["name"], (val_txt, f"val:{pend}"), obj_plur)
                other_ment = mention_piece(entities[other], f"argm:{i}:{other_pos}", subj=False)
                pend_ment = mention_piece(pend_entity, f"argm:{i}:{pend_pos}", subj=True)
                pieces = lead + [(", and together with ", None), other_ment, (", ", None), pend_ment,
                                  (f" has some {obj_plur} combined", None)]
                (cs, ce), spans = emit(doc, pieces)
                doc.add(". ")
                out_factors[pend] = {"ftype": "given", "var": pend, "value": factors[pend]["value"],
                                      "spans": [[cs, ce]], "arg_spans": [], "cue_spans": []}
                mentions[pend] = spans.get(f"val:{pend}", [])
                done[pend] = True
            else:
                a_ment = mention_piece(entities[a], f"argm:{i}:0", subj=True)
                b_ment = mention_piece(entities[b], f"argm:{i}:1", subj=True)
                pieces = together_pieces(obj_plur, a_ment, b_ment)
                (cs, ce), spans = emit(doc, pieces)
                doc.add(". ")
            cue = cue_spans_in(doc.text, cs, ce)
            out_factors[i] = {"ftype": "rel", "op": "add", "args": [a, b], "result": i,
                               "spans": [[cs, ce]], "arg_spans": [spans.get(f"argm:{i}:0", []), spans.get(f"argm:{i}:1", [])],
                               "cue_spans": cue}
            done[i] = True
        # comparative / continuation are fully handled from the "operand" branch above
        # (rel_i is always operand_i + 1 by construction).

    # distractor sentence (~20% of rows), inserted after a random already-rendered clause
    if rng.random() < 0.20:
        dname = random.choice(NAMES)[0]
        dtxt = random.choice(DISTRACTOR_TEMPLATES).format(name=dname).rstrip(".")
        doc.add(dtxt + ". ")

    q_entity = entities[query_var]
    q_start = doc.pos()
    doc.text += question_pieces(q_entity, obj_plur)
    q_end = doc.pos()

    text = doc.text
    solution = [values[i] for i in range(n)]
    out = {
        "n_vars": n,
        "m": min(9999, max(300, 2 * max(values))),
        "text": text,
        "factors": out_factors,
        "mentions": {str(k): v for k, v in mentions.items() if v},
        "query_var": query_var,
        "solution": solution,
        "decisions": 0,
        "key": key,
        "gen": {"row_id": row_id},
    }
    return out


# --------------------------------------------------------------------------- driver

def generate(n, seed):
    rng = random.Random(seed)
    random.seed(seed)  # templates use the module-level `random` for choice()
    rows = []
    attempts = 0
    while len(rows) < n:
        attempts += 1
        try:
            row = render_row(rng, len(rows))
        except RuntimeError:
            continue
        assert 0 <= row["query_var"] < row["n_vars"]
        rows.append(row)
    return rows, attempts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=41)
    ap.add_argument("--out", default=".cache/render_v1_5k.jsonl")
    args = ap.parse_args()

    rows, attempts = generate(args.n, args.seed)
    admitted, why = admit(rows, version="render_v1", refused_out=(refused := []))
    for r in admitted:
        g = dict(r.get("gen") or {})
        g.update({"src": "render", "source": "render", "canonical": "positional", "renderer_version": "v1"})
        r["gen"] = g

    with open(args.out, "w") as fh:
        for r in admitted:
            fh.write(json.dumps(r) + "\n")

    refused_path = args.out.rsplit(".", 1)[0] + "_refused.jsonl"
    with open(refused_path, "w") as fh:
        for r in refused:
            fh.write(json.dumps(r) + "\n")

    print(f"[prose_render] sampled {len(rows)} graphs ({attempts} attempts, "
          f"{attempts - len(rows)} graph-sampling retries) -> admitted {len(admitted)} "
          f"({100.0 * len(admitted) / len(rows):.1f}%); refused {why} -> {args.out} "
          f"(refusals detailed in {refused_path})")


if __name__ == "__main__":
    main()
