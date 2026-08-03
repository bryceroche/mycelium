"""answer_space_retro_read.py — THE RETROACTIVE READ (2026-08-03, the
answer-space word given; pins banked BEFORE this runs — see ledger
'THE RETROACTIVE READ'S PINS'). Prices candidate dialect extensions
against MATH-500 addressability. Baseline must reproduce 197/500 or
no read fires. E5 (length 300→600) is a trunk lever, priced
SEPARATELY, never bundled. asy always excluded. MATH-500 is MEASURED,
never trained on — this is a lawful composition read (precedent: the
m500 composition audit)."""
import json, re
from fractions import Fraction

probs = [json.loads(l) for l in open('.cache/math500_test.jsonl')]
assert len(probs) == 500


def classify_answer(ans):
    """→ (kind, value-ish) where kind ∈ int/neg-int/big-int/rational/
    radical/other."""
    s = str(ans).strip().replace('$', '').replace('\\!', '').replace(',', '').strip()
    s = s.replace('\\dfrac', '\\frac').replace('\\tfrac', '\\frac')
    if re.fullmatch(r'-?\d+', s):
        v = int(s)
        if 0 <= v <= 300: return ('int', v)
        if -300 <= v < 0: return ('neg-int', v)
        return ('big-int', v)
    m = re.fullmatch(r'(-?)\\frac\{(-?\d+)\}\{(\d+)\}', s)
    if m:
        p = int(m.group(2)) * (-1 if m.group(1) == '-' else 1)
        return ('rational', Fraction(p, int(m.group(3))))
    m = re.fullmatch(r'(-?\d+)/(\d+)', s)
    if m:
        return ('rational', Fraction(int(m.group(1)), int(m.group(2))))
    m = re.fullmatch(r'-?\d+\.\d+', s)
    if m:
        return ('rational', Fraction(s))
    if re.fullmatch(r'-?\d*\s*\\sqrt\{?\d+\}?', s) or \
       re.fullmatch(r'-?\d+\s*[+-]\s*\d*\\sqrt\{?\d+\}?', s) or \
       re.fullmatch(r'\\frac\{-?\d*\\sqrt\{?\d+\}?\}\{\d+\}', s):
        return ('radical', s)
    return ('other', s)


def literals_ok(text, cap):
    return all(int(n) <= cap for n in re.findall(r'\d+', text))


rows = []
for p in probs:
    kind, val = classify_answer(p['answer'])
    rows.append({
        'kind': kind,
        'asy': 'asy]' in p['problem'],
        'len_ok300': len(p['problem']) < 300,
        'len_ok600': len(p['problem']) < 600,
        'lit300': literals_ok(p['problem'], 300),
        'lit1e6': literals_ok(p['problem'], 10**6),
        'rat_small': kind == 'rational' and abs(val.numerator) <= 300
                     and val.denominator <= 300 if kind == 'rational' else False,
        'neg_ok': kind == 'neg-int',
        'big_ok': kind == 'big-int' and abs(val) <= 10**6 if kind == 'big-int' else False,
    })

def count(admit_kinds_fn, lit_key='lit300', len_key='len_ok300'):
    n = 0
    for r in rows:
        if r['asy'] or not r[len_key] or not r[lit_key]:
            continue
        if admit_kinds_fn(r):
            n += 1
    return n

base = count(lambda r: r['kind'] == 'int')
print(f"[baseline] integer[0,300], len<300, lit<=300, no-asy: {base}/500")
assert base == 197, f"baseline {base} != 197 — NO READ FIRES (filter reconstruction failed)"

steps = [
    ("today (int 0..300)",            lambda r: r['kind'] == 'int', 'lit300', 'len_ok300'),
    ("+E1 NEG",                       lambda r: r['kind'] == 'int' or r['neg_ok'], 'lit300', 'len_ok300'),
    ("+E1+E2 RAT",                    lambda r: r['kind'] == 'int' or r['neg_ok'] or r['rat_small'], 'lit300', 'len_ok300'),
    ("+E1+E2+E3 BIG (lit<=1e6 too)",  lambda r: r['kind'] in ('int',) or r['neg_ok'] or r['rat_small'] or r['big_ok'], 'lit1e6', 'len_ok300'),
    ("+E1..E4 RAD",                   lambda r: r['kind'] in ('int', 'radical') or r['neg_ok'] or r['rat_small'] or r['big_ok'], 'lit1e6', 'len_ok300'),
]
res = {}
prev = None
for name, fn, litk, lenk in steps:
    n = count(fn, litk, lenk)
    gain = f" (+{n-prev})" if prev is not None else ""
    print(f"[ladder] {name}: {n}/500{gain}")
    res[name] = n
    prev = n

# E5 separately (never bundled): each rung re-read at len<600
print("\n[E5 LEN 300->600, priced separately per rung]")
for name, fn, litk, _ in steps:
    n = count(fn, litk, 'len_ok600')
    print(f"  {name} @ len<600: {n}/500 (+{n-res[name]} vs len<300)")

# single-extension marginal gains (not stacked)
print("\n[single-extension marginals from today]")
for nm, fn in [("E1 alone", lambda r: r['kind']=='int' or r['neg_ok']),
               ("E2 alone", lambda r: r['kind']=='int' or r['rat_small']),
               ("E4 alone", lambda r: r['kind'] in ('int','radical'))]:
    print(f"  {nm}: {count(fn)}/500 (+{count(fn)-197})")
n3 = count(lambda r: r['kind']=='int' or r['big_ok'], 'lit1e6')
print(f"  E3 alone (answers+literals to 1e6): {n3}/500 (+{n3-197})")

import collections
print("\n[answer-kind census, all 500]:", dict(collections.Counter(r['kind'] for r in rows)))
json.dump({"baseline": base, "ladder": res}, open('.cache/answer_space_retro_read.json','w'), indent=1)
print("[saved] .cache/answer_space_retro_read.json")
