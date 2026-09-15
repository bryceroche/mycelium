"""build_wordmint.py — THE WORDED MINT fixture (2026-09-15, word given): the
mint test rows (algebra_nl_test.jsonl, 300 rows the machine reads at 0.98)
with every integer numeral rewritten as its cardinal WORDS (12 -> twelve,
45 -> forty-five, 120 -> one hundred twenty); spans and mentions (character
offsets) shifted exactly; factors / solution / decisions untouched (the
parse is the same). A MEASUREMENT fixture, never diet: the open read's drop
vs mint = the lexicon's headroom; what the lexicon road recovers = its
verdict. usage: build_wordmint.py [in] [out]"""
import json, re, sys
ONES = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"]
TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]

def words(n):
    assert 0 <= n <= 999, n
    if n < 20: return ONES[n]
    if n < 100: return TENS[n // 10] + ("" if n % 10 == 0 else "-" + ONES[n % 10])
    return ONES[n // 100] + " hundred" + ("" if n % 100 == 0 else " " + words(n % 100))

def rewrite(text):
    """returns (new_text, shift(offset) -> new offset)"""
    out = []; last = 0; segs = []      # segs: (old_start, old_end, new_start, new_end)
    for m in re.finditer(r"(?<![\w.])\d{1,3}(?![\w]|\.\d)", text):
        if int(m.group()) > 999: continue
        out.append(text[last:m.start()]); ns = sum(len(x) for x in out); w = words(int(m.group())); out.append(w)
        segs.append((m.start(), m.end(), ns, ns + len(w))); last = m.end()
    out.append(text[last:]); new = "".join(out)
    def shift(o):
        d = 0
        for os_, oe, ns, ne in segs:
            if o >= oe: d = ne - oe - (os_ - os_) + (ns - os_) if False else ne - oe   # cumulative
            elif o > os_: return ns + min(o - os_, ne - ns)                                # inside a rewritten numeral
        # cumulative delta = sum over segments entirely before o
        d = sum((ne - ns) - (oe - os_) for os_, oe, ns, ne in segs if oe <= o)
        return o + d
    return new, shift

src = sys.argv[1] if len(sys.argv) > 1 else ".cache/algebra_nl_test.jsonl"; dst = sys.argv[2] if len(sys.argv) > 2 else ".cache/wordmint_test.jsonl"
n = 0; n_num = 0
with open(dst, "w") as f:
    for line in open(src):
        r = json.loads(line); new, sh = rewrite(r["text"]); n_num += len(re.findall(r"(?<![\w.])\d{1,3}(?![\w]|\.\d)", r["text"]))
        for fac in r.get("factors", []):
            if fac.get("spans"): fac["spans"] = [[sh(a), sh(b)] for a, b in fac["spans"]]
        if r.get("mentions"):
            r["mentions"] = {k: [[sh(a), sh(b)] for a, b in v] for k, v in r["mentions"].items()}
        r["text"] = new; r.setdefault("gen", {}); (r["gen"] if isinstance(r["gen"], dict) else {})["wordmint"] = True
        f.write(json.dumps(r) + "\n"); n += 1
print(f"[wordmint] {n} rows -> {dst}; {n_num} numerals rewritten as words ({n_num / n:.1f}/row)")
r = json.loads(open(dst).readline()); print("[wordmint] sample:", r["text"][:200]); print("[wordmint] spans check:", [r["text"][a:b] for a, b in r["factors"][0]["spans"]], "| mention 0:", [r["text"][a:b] for a, b in r["mentions"]["0"][:3]])
