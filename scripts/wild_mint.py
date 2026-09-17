"""wild_mint.py — WILDED MINT (2026-09-17, word given): a mint TRAIN row with its rigid surface broken while
its parse stays: (a) every integer numeral <= 999 rewritten as cardinal words (the worded-mint transform;
the lexicon road's register), (b) a DISTRACTOR sentence with a non-quantity number prefixed (a year, a time,
a street number) — spans and mentions shifted exactly; factors / solution untouched. The row is the same
knot (a rep of its unique problem), stamped gen.wild. Library + CLI. usage: wild_mint.py in out [frac] [seed]"""
import json, re, sys, random
ONES = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"]
TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]
DISTRACTORS = ["In 2019, {name} moved to {n} Maple Street.", "It was {t} in the afternoon when {name} sat down.", "On the {d}th of March, {name} was thinking about this.",
               "{name} had lived at apartment {n} for years.", "At {t} that morning, {name} checked the calendar for {y}.", "The bus number {n} passed {name} on the way home."]
NAMES = ["Maria", "Tom", "Aisha", "Wei", "Luis", "Priya", "Noah", "Fatima", "Ken", "Olga"]

def words(n):
    if n < 20: return ONES[n]
    if n < 100: return TENS[n // 10] + ("" if n % 10 == 0 else "-" + ONES[n % 10])
    return ONES[n // 100] + " hundred" + ("" if n % 100 == 0 else " " + words(n % 100))

def reword(text):
    """numerals <= 999 -> words; returns (new_text, shift) with shift an exact offset map"""
    segs = []; out = []; last = 0
    for m in re.finditer(r"(?<![\w.])\d{1,3}(?![\w]|\.\d)", text):
        out.append(text[last:m.start()]); ns = sum(len(x) for x in out); w = words(int(m.group())); out.append(w)
        segs.append((m.start(), m.end(), ns, ns + len(w))); last = m.end()
    out.append(text[last:]); new = "".join(out)
    def shift(o):
        d = 0
        for os_, oe, ns, ne in segs:
            if o >= oe: d += (ne - ns) - (oe - os_)
            elif o > os_: return ns + min(o - os_, ne - ns)
        return o + d
    return new, shift

def wild(row, rng, do_words=True, do_distractor=True):
    r = json.loads(json.dumps(row)); text = r["text"]; shifts = []
    if do_words:
        text, sh = reword(text); shifts.append(sh)
    if do_distractor:
        pre = rng.choice(DISTRACTORS).format(name=rng.choice(NAMES), n=rng.randint(2, 98), t=f"{rng.randint(1, 12)}:{rng.randint(10, 59)}", d=rng.randint(2, 28), y=rng.randint(2001, 2030)) + " "
        text = pre + text; L = len(pre); shifts.append(lambda o, L=L: o + L)
    def sh(o):
        for f in shifts: o = f(o)
        return o
    for f in r["factors"]:
        if f.get("spans"): f["spans"] = [[sh(a), sh(b)] for a, b in f["spans"]]
    r["mentions"] = {k: [[sh(a), sh(b)] for a, b in v] for k, v in (r.get("mentions") or {}).items()}
    r["text"] = text; g = r.get("gen") if isinstance(r.get("gen"), dict) else {"src": str(r.get("gen"))}; g = dict(g); g["wild"] = {"words": do_words, "distractor": do_distractor}; r["gen"] = g
    return r

if __name__ == "__main__":
    src, dst = sys.argv[1], sys.argv[2]; frac = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0; rng = random.Random(int(sys.argv[4]) if len(sys.argv) > 4 else 0)
    rows = [json.loads(l) for l in open(src)]; out = []; n = 0; bad = 0
    for r in rows:
        if rng.random() < frac:
            w = wild(r, rng, do_words=rng.random() < 0.7, do_distractor=rng.random() < 0.7); n += 1
            for f, g in zip(r["factors"], w["factors"]):   # the offset audit: every span selects the same words (numerals excepted)
                for (a, b), (c, d) in zip(f.get("spans", []), g.get("spans", [])):
                    if re.sub(r"\d+|\b[a-z-]+\b", "#", r["text"][a:b]) != re.sub(r"\d+|\b[a-z-]+\b", "#", w["text"][c:d]) and not re.fullmatch(r"\d+", r["text"][a:b].strip()): bad += 1
            out.append(w)
        else: out.append(r)
    with open(dst, "w") as f:
        for r in out: f.write(json.dumps(r) + "\n")
    print(f"[wild-mint] {len(rows)} rows, {n} wilded (frac {frac}); span audit mismatches {bad} -> {dst}")
