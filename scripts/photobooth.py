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
PB_WIDE=1 (2026-10-17, the training augmentation; default OFF so the banked wild views regenerate unchanged): wider name
           lists (EXTRA_M / EXTRA_F, + the NLTK gazetteer .cache/names_gazetteer.json); a name outside every list is KEPT (the 10-17 pronoun inference renamed spaCy false
           positives -- 'Doritos' -> Takeshi -- and was removed 10-18 after the audit). Names never match inside a hyphenated token.
           Titles (Mr, Mrs, Dr, ...) and any name that follows a title (a surname: 'Mrs. Hilt') are never swapped. The honesty check is unchanged.
usage: photobooth.py IN.jsonl OUT_PREFIX K  ->  OUT_PREFIX_v{1..K}.jsonl + OUT_PREFIX.census.json"""
import collections, json, os, random, re, sys

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
EXTRA_M = ("Ahmed Hassan Yusuf Ibrahim Tariq Kareem Malik Jamal Darius Andre Marcus Terrence Dwayne Rashad Hiroshi Kenji "
           "Takeshi Arjun Rohan Vikram Sanjay Ravi Amit Rahul Diego Mateo Santiago Javier Miguel Pedro Rafael Alejandro "
           "Fernando Pablo Lucas Liam Mason Oliver Elias Caleb Isaac Levi Ezra Silas Jasper Milo Theo Finn Kai Ryder Wyatt "
           "Colton Chase Blake Grant Trevor Derek Dean Neil Kurt Lance Brett Wade Troy Glenn Dale Cole Reid Seth Gavin "
           "Spencer Victor Simon Martin Stefan Anton Boris Dmitri Sergei Nikolai Lars Sven Erik Hans Klaus Pierre Jacques "
           "Luca Paolo Giorgio Kwame Kofi Chidi Emeka Tunde Femi Musa Sipho Thabo Mohammed Abdul Omari Jamie Rick Tyrone "
           "Desmond Reggie Earl Floyd Vernon Leon Lionel Mario Bruno Hector Ramon Enrique Andres Ricardo Emilio").split()
EXTRA_F = ("Fatima Aisha Layla Zara Yasmin Amira Leila Noor Priya Ananya Divya Kavya Meera Neha Pooja Riya Sana Yuki Akiko "
           "Sakura Hana Keiko Camila Valentina Lucia Gabriela Daniela Isabel Carmen Ana Paula Martina Chiara Giulia "
           "Francesca Ingrid Astrid Freya Greta Heidi Katya Olga Natasha Svetlana Irina Tanya Anya Amara Ngozi Chioma Zainab "
           "Adaeze Thandiwe Naledi Ayanna Imani Keisha Tamika Jasmine Aaliyah Harper Brooke Haley Claire Fiona Molly Holly "
           "Jenna Erin Tessa Wendy Gina Rita Carla Nadia Vanessa Monica Wanda Tara Kara Lena Mila Elsa Stella Hazel Violet "
           "Penelope Scarlett Aria Luna Bella Sadie Maggie Josie Rosie Annie Ellie Millie Paige Kendra Tiffany Crystal "
           "Brianna Gabrielle Mariana Ximena Paloma Inez Dolores Esther Miriam Naomi Tamar Shira").split()
TITLES = set("Mr Mrs Ms Miss Dr Mx Sir Madam Aunt Uncle Grandma Grandpa Mom Dad Mother Father Coach Professor Principal Captain".split())
CAP = re.compile(r"^[A-Z][a-z]+$")


SPAN_KEYS = ("spans", "arg_spans", "cue_spans", "role_cues", "arg_role_cues")


def substitute(text, mp):
    """one pass over whole-word name matches; returns (new text, pmap) with pmap[p] = the new position of old char p
    (len(text)+1 entries; inside a renamed name, the start maps to the new start and the end to the new end)."""
    if not mp:
        return text, None
    pat = re.compile(r"(?<![\w-])(" + "|".join(re.escape(n) for n in sorted(mp, key=len, reverse=True)) + r")(?![\w-])")  # hyphen-aware: never half of 'Jo-Bob'
    out, pmap, last = [], [], 0
    for m in pat.finditer(text):
        a, b = m.span(); rep = mp[m.group(1)]
        base = sum(len(x) for x in out)
        pmap.extend(base + i for i in range(a - last)); out.append(text[last:a])
        base = sum(len(x) for x in out)
        pmap.extend([base] + [base + len(rep)] * (b - a - 1)); out.append(rep); last = b
    base = sum(len(x) for x in out)
    pmap.extend(base + i for i in range(len(text) - last + 1)); out.append(text[last:])
    return "".join(out), pmap


def remap_spans(row, pmap):
    """THE SPAN CLAUSE (2026-10-17): every char-offset span a row carries (mentions; each factor's spans / arg_spans /
    cue_spans / role_cues / arg_role_cues) moves with the text. A span is a list whose first two items are ints [s, e, ...]."""
    def mv(o):
        if isinstance(o, list) and len(o) >= 2 and isinstance(o[0], int) and isinstance(o[1], int):
            return [pmap[o[0]], pmap[o[1]]] + o[2:]
        if isinstance(o, list):
            return [mv(x) for x in o]
        return o
    if isinstance(row.get("mentions"), dict):
        row["mentions"] = {k: mv(v) for k, v in row["mentions"].items()}
    row["factors"] = [{k: (mv(v) if k in SPAN_KEYS else v) for k, v in f.items()} for f in row["factors"]]


def main(src, prefix, K):
    tok = Tokenizer.from_file(".cache/llama-3.2-1b-weights/tokenizer.json")
    L = lambda s: len(tok.encode(s, add_special_tokens=False).ids)
    sig = lambda n: (L(" " + n), L(n))
    nlp = spacy.load("en_core_web_sm")
    wide = os.environ.get("PB_WIDE") == "1"
    male, female = (MALE + EXTRA_M, FEMALE + EXTRA_F) if wide else (MALE, FEMALE)
    if wide:  # (2026-10-18) + the NLTK first-name gazetteer (scripts/build_names_gazetteer.py); the hand lists win a conflict
        gz = json.load(open(".cache/names_gazetteer.json"))
        hand = set(male) | set(female)
        male, female = male + [n for n in gz["M"] if n not in hand], female + [n for n in gz["F"] if n not in hand]
    from reading_order import _solve
    rows = [json.loads(l) for l in open(src)]
    docs = list(nlp.pipe([r["text"] for r in rows]))
    pools = {g: collections.defaultdict(list) for g in ("M", "F")}
    for g, names in (("M", male), ("F", female)):
        for n in dict.fromkeys(names):
            pools[g][sig(n)].append(n)
    gender = {**{n: "M" for n in male}, **{n: "F" for n in female}}
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
            low = set(w.lower() for w in words)
            for n in names:
                if wide and n in TITLES:
                    C["name_title_kept"] += 1; continue
                if wide and re.search(rf"\b(?:{'|'.join(TITLES)})\.?\s+{n}\b", text):
                    C["name_surname_after_title_kept"] += 1; continue
                g = gender.get(n)
                # (2026-10-18, the audit) NO pronoun-inferred gender: spaCy PERSON false positives ('Doritos', 'Bronze
                # Status', 'Seattle Seahawks') took a gender from the text and were renamed. Only LISTED names swap.
                if g is None:
                    C["name_unknown_gender_kept"] += 1; continue
                pool = pools["M"][sig(n)] + pools["F"][sig(n)] if g == "N" else pools[g][sig(n)]
                cands = [c for c in pool if c != n and c not in words and c not in mp.values()]
                if not cands:
                    C["name_no_token_twin_kept"] += 1; continue
                mp[n] = rng.choice(cands)
            new, pmap = substitute(text, mp)
            ok = True
            if sorted(re.findall(r"\d[\d,\.]*", new)) != sorted(re.findall(r"\d[\d,\.]*", text)):
                ok = False; C["FAIL_numerals"] += 1
            if L(new) != L(text):
                ok = False; C["FAIL_token_count"] += 1
            if any(re.search(rf"\b{m}\b", text) for m in mp.values()):
                ok = False; C["FAIL_collision"] += 1
            r2 = dict(r); r2["text"] = new if ok else text
            if ok and new != text:
                remap_spans(r2, pmap)
            if ok and key is not None:
                a2, _ = _solve(r2)
                if a2 != key:
                    ok = False; C["FAIL_solve"] += 1; r2 = dict(r)
            r2["gen"] = {**(r.get("gen") or {}), "photobooth": {"view": v, "names": mp if ok and new != text else {}, "source_text": text, **({"wide": True} if wide else {})}}
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
