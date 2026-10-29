"""nl_bench_gen.py -- THE NL BINDING BENCH, the generator + census (ledger 2026-10-29, Bryce: "word given for the whole queue" on
the reader-side organ test). Zero training. The organs' jobs moved to where they live: READING. Every story is generated from a GOLD
relation-DSL structure (.cache/rel_dsl_rows.jsonl: given/add/mul factors, custody-gold answers), so custody is the rel-DSL's.
A STORY: each var becomes a distinct (owner, item) quantity ("Ann's apples"); each factor one sentence from a paraphrase frame
(given 4 frames, add 4, mul 3); a final query sentence (2 frames). Values appear as NUM tokens (value kept aside, never read).
  PRONOUNS  a mention of an owner already named MAY become her/his/she/he/him (p 0.6) iff that owner is the MOST RECENT owner of its
            gender: resolution = gender agreement + recency, so a pronoun whose most recent owner overall is someone else NEEDS gender
  TITLES    the first mention of an owner carries Ms./Mr. with p 0.35: the only in-text gender evidence for a name never seen in training
  NAMES     a world of 90 names with fixed gender; TRAIN pool 60 (30 F / 30 M), HELD pool 30 (15 / 15) never in any training story
THE READER'S OUTPUT (what the arms predict; everything else is assembled symbolically): per mention (anchor = its item token): the
sentence op (given / add / mul / query), the mention's role (GIVEN / RES / ARG / QUERY) and its OWNER (a pointer to any occurrence of
the owner's name). The assembler builds the factor graph (var = (pointed name, item); a given's value = its sentence's NUM), forced
propagation (rel_dsl_census) solves it, and the story scores iff the query value equals the custody key.
SPLITS (structure = sdig; VAL = sha1(sdig) % 5 == 0, as the rel-DSL's): TRAIN = non-VAL structures with <= 8 factors (stories
regenerated fresh every batch); IN = VAL structures <= 8 factors, TRAIN names; HELD = the same structures, HELD names; LONG = all
structures with 9..12 factors (never trained at that length), TRAIN names. One fixed story per test structure.
Census: survival, the gold-label assembly must reproduce the key on every row (asserted), story lengths, pronoun rates, the share
of pronouns that NEED gender, title coverage per held name. Writes .cache/nlb_census.txt."""
import hashlib, json, os, sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ".")
import rel_dsl_census as RC

F_NAMES = ("Ann Beth Cara Dana Eva Fay Gina Hana Ida Jill Kate Lena Mia Nina Olga Pia Rosa Sara Tara Uma Vera Wren Xena Yara Zoe "
           "Alma Bree Cleo Dora Edie Flor Gwen Hope Iris June Kira Lila Mara Nell Opal Pearl Ruth Sage Tess Vida").split()
M_NAMES = ("Abe Ben Carl Dan Eli Fred Gus Hal Ivan Jack Kurt Leo Max Ned Otto Paul Rex Sam Tom Ugo Vic Walt Xavi Yuri Zack "
           "Amos Bert Cole Dean Emil Finn Glen Hugh Igor Joel Kent Luke Mark Noel Owen Pete Rudy Seth Ted Wade").split()
TRAIN_NAMES = F_NAMES[:30] + M_NAMES[:30]
HELD_NAMES = F_NAMES[30:] + M_NAMES[30:]
GENDER = {**{n: 0 for n in F_NAMES}, **{n: 1 for n in M_NAMES}}
ITEMS = "apples pencils stamps marbles books cards coins shells stickers beads cookies toys".split()
PRON = {"subj": ("she", "he"), "obj": ("her", "him"), "poss": ("her", "his")}
TITLE = ("ms.", "mr.")
# frames: P<k> = possessive mention of role k ("ann 's apples" / "her apples"); O<k> / B<k> = owner as subject / object; I<k> = item
FRAMES = {
    "given": ["O0 has NUM I0 .", "O0 owns NUM I0 .", "P0 number NUM .", "there are NUM I0 that belong to B0 ."],
    "add": ["P0 equal P1 plus P2 .", "together , P1 and P2 make P0 .", "P0 are as many as P1 and P2 combined .",
            "combining P1 with P2 gives P0 ."],
    "mul": ["P0 equal P1 times P2 .", "P0 are the product of P1 and P2 .", "multiplying P1 by P2 gives P0 ."],
    "query": ["how many I0 does O0 have ?", "what is the number of P0 ?"],
}
OPS = ("given", "add", "mul", "query"); ROLES = ("GIVEN", "RES", "ARG", "QUERY")
WORDS = sorted({w for fs in FRAMES.values() for f in fs for w in f.split() if not (w[0] in "POBI" and w[1:].isdigit())} |
               {"'s", "NUM", *PRON["subj"], *PRON["obj"], *PRON["poss"], *TITLE, *ITEMS})
VOCAB = ["<pad>"] + WORDS + F_NAMES + M_NAMES
WID = {w: i for i, w in enumerate(VOCAB)}
h = lambda s: int(hashlib.sha1(s.encode()).hexdigest(), 16)


def load_structures():
    """One row per structure (first train-fixture row), add/mul only, <= 12 factors; split by the rel-DSL's VAL rule."""
    first = {}
    for l in open(".cache/rel_dsl_rows.jsonl"):
        t = json.loads(l)
        if t["split"] != "train" or t["sdig"] in first: continue
        first[t["sdig"]] = t
    keep, drop = [], {"sub": 0, "long": 0}
    for t in sorted(first.values(), key=lambda t: h(t["sdig"])):
        if any(f["ftype"] == "rel" and f["op"] == "sub" for f in t["facs"]): drop["sub"] += 1; continue
        if len(t["facs"]) > 12: drop["long"] += 1; continue
        keep.append(t)
    isval = lambda t: h(t["sdig"]) % 5 == 0
    train = [t for t in keep if not isval(t) and len(t["facs"]) <= 8]
    test_in = [t for t in keep if isval(t) and len(t["facs"]) <= 8]
    long_ = [t for t in keep if len(t["facs"]) >= 9]
    return train, test_in, long_, drop, len(first)


def make_story(t, pool, rng, p_pron=0.6, p_title=0.35):
    """-> dict: tokens (words), is_name, anchors [(tok, sent, op, role, owner, item)], name_pos {owner: [toks]}, num (tok -> value),
    sents (per sentence: op, mention idxs), query mention idx, pronoun stats."""
    used = sorted({v for f in t["facs"] for v in RC.fvars(f)} | {t["query"]})
    k = int(rng.integers(2, min(5, len(used)) + 1))
    owners = list(rng.choice(pool, size=k, replace=False))
    var = {}; taken = set()
    for v in used:
        while True:
            o = owners[int(rng.integers(k))]; it = ITEMS[int(rng.integers(len(ITEMS)))]
            if (o, it) not in taken: taken.add((o, it)); var[v] = (o, it); break
    toks, isn, anchors, sents, nums = [], [], [], [], {}
    seen, last_of, last_any = set(), {}, [None]   # owners named so far; most recent owner per gender; most recent owner
    stats = dict(pron=0, need_gender=0, mentions=0); need = [False]

    def owner_toks(o, case):
        g = GENDER[o]
        if o in seen and last_of.get(g) == o and rng.random() < p_pron:
            need[0] = last_any[0] != o                                       # the most recent owner is the other gender
            stats["pron"] += 1; stats["need_gender"] += need[0]
            return [PRON[case][g]], False
        return ([TITLE[g]] if o not in seen and rng.random() < p_title else []) + [o], True

    def note(o):
        last_of[GENDER[o]] = o; last_any[0] = o; seen.add(o)

    plan = [(i, f) for i, f in enumerate(t["facs"])] + [(None, None)]
    for si, (fi, f) in enumerate(plan):
        if f is None: op, roles_v = "query", [t["query"]]
        elif f["ftype"] == "given": op, roles_v = "given", [f["var"]]
        else: op, roles_v = f["op"], [f["result"]] + list(f["args"])
        frame = FRAMES[op][int(rng.integers(len(FRAMES[op])))].split()
        sent = dict(op=op, ments=[], start=len(toks))
        pending = {}                      # role idx -> mention record awaiting its item / owner
        for w in frame:
            if w[0] in "POBI" and w[1:].isdigit():
                k_ = int(w[1:]); v = roles_v[k_]; o, it = var[v]
                role = {"given": "GIVEN", "query": "QUERY"}.get(op) or ("RES" if k_ == 0 else "ARG")
                if w[0] in "POB":
                    ot, named = owner_toks(o, {"P": "poss", "O": "subj", "B": "obj"}[w[0]])
                    for x in ot: toks.append(x); isn.append(x in GENDER)
                    note(o)
                    if w[0] == "P":
                        if named: toks.append("'s"); isn.append(False)
                        toks.append(it); isn.append(False)
                        anchors.append(dict(tok=len(toks) - 1, sent=si, op=op, role=role, owner=o, item=it, pron=not named,
                                            need=(not named) and need[0]))
                        sent["ments"].append(len(anchors) - 1)
                    else:
                        pending.setdefault(k_, {})["owner"] = (o, not named, (not named) and need[0])
                        if "tok" in pending[k_]:
                            a = anchors[pending[k_]["tok"]]; a["owner"], a["pron"], a["need"] = o, not named, (not named) and need[0]
                else:   # I<k>: the item token is the anchor
                    toks.append(it); isn.append(False)
                    ow = pending.get(k_, {}).get("owner")
                    anchors.append(dict(tok=len(toks) - 1, sent=si, op=op, role=role, owner=o, item=it, pron=bool(ow and ow[1]),
                                        need=bool(ow and ow[2])))
                    sent["ments"].append(len(anchors) - 1); pending.setdefault(k_, {})["tok"] = len(anchors) - 1
            elif w == "NUM":
                nums[len(toks)] = f["value"]; toks.append("NUM"); isn.append(False)
            else:
                toks.append(w); isn.append(False)
        sent["end"] = len(toks); sents.append(sent)
    stats["mentions"] = len(anchors)
    name_pos = {}
    for i, w in enumerate(toks):
        if w in GENDER: name_pos.setdefault(w, []).append(i)
    return dict(toks=toks, is_name=isn, anchors=anchors, sents=sents, nums=nums, name_pos=name_pos, stats=stats,
                gold=t["gold"], sdig=t["sdig"], nfac=len(t["facs"]))


def assemble(st, op_of, role_of, owner_of):
    """Symbolic assembly from per-anchor predictions (op_of / role_of: lists of labels; owner_of: list of name strings) -> answer."""
    facs, query = [], None; vid = {}
    key = lambda a: vid.setdefault((owner_of[a], st["anchors"][a]["item"]), len(vid))
    for s in st["sents"]:
        ms = s["ments"]
        if not ms: return None
        op = op_of[ms[0]]
        if op == "query": query = key(ms[0]); continue
        if op == "given":
            nt = [p for p in st["nums"] if s["start"] <= p < s["end"]]
            if len(nt) != 1: return None
            facs.append(dict(ftype="given", var=key(ms[0]), value=st["nums"][nt[0]])); continue
        res = [a for a in ms if role_of[a] == "RES"]
        if len(res) != 1: return None
        facs.append(dict(ftype="rel", op=op, args=[key(a) for a in ms if a != res[0]], result=key(res[0])))
    if query is None: return None
    val, _, _, err = RC.propagate(facs)
    return None if err else val.get(query)


def gold_labels(st):
    A = st["anchors"]
    return [a["op"] for a in A], [a["role"] for a in A], [a["owner"] for a in A]


def test_sets(seed=0):
    train, tin, lng, drop, n0 = load_structures()
    rng = np.random.default_rng(seed)
    S = dict(IN=[make_story(t, TRAIN_NAMES, rng) for t in tin], HELD=[make_story(t, HELD_NAMES, rng) for t in tin],
             LONG=[make_story(t, TRAIN_NAMES, rng) for t in lng])
    return train, S, drop, n0


if __name__ == "__main__":
    from fractions import Fraction
    train, S, drop, n0 = test_sets()
    out = open(".cache/nlb_census.txt", "w")

    def say(s):
        print(s, flush=True); out.write(s + "\n")
    say(f"structures {n0} | dropped: sub {drop['sub']}, > 12 factors {drop['long']} | TRAIN {len(train)} | IN/HELD {len(S['IN'])} | LONG {len(S['LONG'])}")
    rng = np.random.default_rng(1); tr_st = [make_story(t, TRAIN_NAMES, rng) for t in train]
    for name, sts in (("TRAIN(sample)", tr_st), *S.items()):
        ok = [assemble(st, *gold_labels(st)) == Fraction(st["gold"]) for st in sts]
        assert all(ok), f"{name}: gold assembly fails on {len(ok) - sum(ok)} rows"
        L = [len(st["toks"]) for st in sts]; P = sum(st["stats"]["pron"] for st in sts); M = sum(st["stats"]["mentions"] for st in sts)
        NG = sum(st["stats"]["need_gender"] for st in sts)
        say(f"{name:13s} n {len(sts):5d} | gold assembly reproduces the key on ALL rows | tokens mean {np.mean(L):.0f} max {max(L)} | "
            f"mentions {M} | pronouns {P} ({P / M:.3f}) | pronouns NEEDING gender {NG} ({NG / max(P, 1):.3f})")
    held = S["HELD"]; cov = {n: 0 for n in HELD_NAMES}; inprob = []
    for st in held:
        tl = {st["toks"][i + 1] for i, w in enumerate(st["toks"][:-1]) if w in TITLE}
        for n in tl: cov[n] += 1
        inprob += [a["owner"] in tl for a in st["anchors"] if a["pron"]]
    say(f"HELD titles: every held name titled somewhere in HELD: {all(c > 0 for c in cov.values())} (min {min(cov.values())} stories) | "
        f"pronoun mentions whose referent is titled IN-PROBLEM: {np.mean(inprob):.3f}")
    say(f"vocab {len(VOCAB)} (names {len(F_NAMES) + len(M_NAMES)}, held names never in training stories)")
    st = tr_st[3]; say("EXAMPLE: " + " ".join(st["toks"]))
