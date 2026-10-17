"""photobooth.py -- THE PHOTOBOOTH, distortion 1: FRESH NAMES (2026-10-16; Bryce's photobooth, word given; Opus's guards).
A view keeps the problem's math EXACTLY and changes only person names. Rule-based, local, no external calls.
NAMES      spaCy PERSON entities' capitalized alphabetic tokens; each distinct name maps to a fresh name of the SAME gender
           list (pronouns stay consistent) that TOKENIZES IDENTICALLY IN LENGTH under the Llama tokenizer, both with a
           leading space and at text start (Opus's control: a vote flip must not be the trunk reacting to odd tokenization).
           A name outside both gender lists is LEFT UNCHANGED (counted). A fresh name never already occurs in the text.
THE HONESTY CHECK (a lying generator teaches lying): per view, (1) the numeral multiset is identical; (2) the gold graph is
           untouched and still solves to the stored key; (3) the Llama token count is identical (so the trunk frame, tokmask
           and sentence indices are unchanged); (4) no fresh name collides with a word already in the text.
           Any failure -> the view keeps the ORIGINAL text for that row and is counted (never a silent pass).
usage: photobooth.py IN.jsonl OUT_PREFIX K  ->  OUT_PREFIX_v{1..K}.jsonl + OUT_PREFIX.census.json"""
import collections, json, random, re, sys

sys.path.insert(0, "."); sys.path.insert(0, "scripts")
import spacy  # noqa: E402
from tokenizers import Tokenizer  # noqa: E402

MALE = ("James John Robert Michael William David Richard Joseph Thomas Charles Daniel Matthew Anthony Mark Paul Steven Andrew "
        "Kenneth Joshua Kevin Brian George Edward Ronald Timothy Jason Jeffrey Ryan Jacob Gary Nicholas Eric Jonathan Stephen "
        "Larry Justin Scott Brandon Benjamin Samuel Frank Gregory Raymond Alexander Patrick Jack Dennis Jerry Tyler Aaron Jose "
        "Adam Henry Nathan Douglas Zachary Peter Kyle Walter Ethan Jeremy Harold Keith Christian Roger Noah Gerald Carl Terry "
        "Sean Austin Arthur Lawrence Jesse Dylan Bryan Joe Jordan Billy Bruce Albert Willie Gabriel Logan Alan Juan Wayne Roy "
        "Ralph Randy Eugene Vincent Russell Elijah Louis Bobby Philip Johnny Tom Bob Sam Ben Tim Jim Mike Dan Max Leo Jake Luke "
        "Carlos Luis Omar Ali Raj Tony Greg Ted Ray Ron Ken Matt Nick Chris Josh Alex Ian Owen Hugo Oscar Felix Marco Ivan").split()
FEMALE = ("Mary Patricia Jennifer Linda Elizabeth Barbara Susan Jessica Sarah Karen Lisa Nancy Betty Margaret Sandra Ashley "
          "Kimberly Emily Donna Michelle Carol Amanda Dorothy Melissa Deborah Stephanie Rebecca Sharon Laura Cynthia Kathleen "
          "Amy Angela Shirley Anna Brenda Pamela Emma Nicole Helen Samantha Katherine Christine Debra Rachel Carolyn Janet "
          "Catherine Maria Heather Diane Ruth Julie Olivia Joyce Virginia Victoria Kelly Lauren Christina Joan Evelyn Judith "
          "Megan Andrea Cheryl Hannah Jacqueline Martha Gloria Teresa Ann Sara Madison Frances Kathryn Janice Jean Abigail Alice "
          "Julia Judy Sophia Grace Denise Amber Doris Marilyn Danielle Beverly Isabella Theresa Diana Natalie Brittany Charlotte "
          "Marie Kayla Alexis Lori Jane Kate Lucy Rose Amy Eva Mia Ella Lily Zoe Nina Tina Sue Kim Jill Beth Meg Liz Ava "
          "Chloe Ruby Sofia Elena Rosa Leah Maya Iris Nora Vera Ruth Jade Ivy").split()
CAP = re.compile(r"^[A-Z][a-z]+$")


def main(src, prefix, K):
    tok = Tokenizer.from_file(".cache/llama-3.2-1b-weights/tokenizer.json")
    L = lambda s: len(tok.encode(s, add_special_tokens=False).ids)
    sig = lambda n: (L(" " + n), L(n))
    nlp = spacy.load("en_core_web_sm")
    from reading_order import _solve
    rows = [json.loads(l) for l in open(src)]
    docs = list(nlp.pipe([r["text"] for r in rows]))
    pools = {g: collections.defaultdict(list) for g in ("M", "F")}
    for g, names in (("M", MALE), ("F", FEMALE)):
        for n in dict.fromkeys(names):
            pools[g][sig(n)].append(n)
    gender = {**{n: "M" for n in MALE}, **{n: "F" for n in FEMALE}}
    C = collections.Counter()
    keys = []
    for r in rows:
        a, st = _solve(r) if r.get("solution") else (None, "no-key")
        keys.append(a)
    for v in range(1, K + 1):
        rng = random.Random(1000 + v)
        out = []
        for r, d, key in zip(rows, docs, keys):
            text = r["text"]
            names = list(dict.fromkeys(t.text for e in d.ents if e.label_ == "PERSON" for t in e if CAP.match(t.text)))
            words = set(re.findall(r"[A-Za-z]+", text))
            mp = {}
            for n in names:
                g = gender.get(n)
                if g is None:
                    C["name_unknown_gender_kept"] += 1; continue
                cands = [c for c in pools[g][sig(n)] if c != n and c not in words and c not in mp.values()]
                if not cands:
                    C["name_no_token_twin_kept"] += 1; continue
                mp[n] = rng.choice(cands)
            new = text
            for n, m in mp.items():
                new = re.sub(rf"\b{n}\b", m, new)
            ok = True
            if sorted(re.findall(r"\d[\d,\.]*", new)) != sorted(re.findall(r"\d[\d,\.]*", text)):
                ok = False; C["FAIL_numerals"] += 1
            if L(new) != L(text):
                ok = False; C["FAIL_token_count"] += 1
            if any(re.search(rf"\b{m}\b", text) for m in mp.values()):
                ok = False; C["FAIL_collision"] += 1
            r2 = dict(r); r2["text"] = new if ok else text
            if ok and key is not None:
                a2, _ = _solve(r2)
                if a2 != key:
                    ok = False; C["FAIL_solve"] += 1; r2["text"] = text
            r2["gen"] = {**(r.get("gen") or {}), "photobooth": {"view": v, "names": mp if ok and new != text else {}}}
            C[f"v{v}_changed"] += ok and new != text
            out.append(r2)
        with open(f"{prefix}_v{v}.jsonl", "w") as fh:
            for r2 in out:
                fh.write(json.dumps(r2) + "\n")
    C["rows"] = len(rows); C["rows_with_person"] = sum(1 for d in docs if any(e.label_ == "PERSON" for e in d.ents))
    json.dump(dict(C), open(f"{prefix}.census.json", "w"), indent=1)
    print(json.dumps(dict(C), indent=1))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]))
