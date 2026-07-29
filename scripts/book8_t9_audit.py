import json, re, string

rows = [json.loads(l) for l in open('/home/bryce/mycelium/.cache/book8_t9_prose_pairs_draft.jsonl')]

def letter_to_idx(l):
    return string.ascii_lowercase.index(l)

issues = []

for r in rows:
    src = r['gen']['src_idx']
    facs = r['factors']
    dia = r['gen']['dialect']
    qv = r['query_var']

    # split into sentences (roughly)
    body = dia
    m = re.match(r'Consider the numbers ([a-z, ]+)\. (.*)', dia)
    if not m:
        issues.append((src, 'no header match'))
        continue
    rest = m.group(2)
    sents = [s.strip() for s in re.split(r'(?<=[.?])\s+', rest) if s.strip()]

    # classify factors by a signature so we can match sentence-by-sentence in order
    # (units may have been shuffled in real gen, but here we wrote them in factor order)
    remaining = list(facs)
    parsed_ok = True
    for s in sents:
        if s.startswith('What is'):
            qm = re.match(r'What is ([a-z])\??\.?$', s)
            if not qm:
                issues.append((src, 'bad query sentence', s))
                continue
            qletter = qm.group(1)
            if letter_to_idx(qletter) != qv:
                issues.append((src, 'query letter mismatch', (qletter, qv)))
            continue
        gm = re.match(r'([a-z]) is (\d+)\.$', s)
        am = re.match(r'([a-z]) plus ([a-z]) equals ([a-z])\.$', s)
        em = re.match(r'([a-z]) exceeds ([a-z]) by ([a-z])\.$', s)
        mm = re.match(r'([a-z]) times ([a-z]) equals ([a-z])\.$', s)
        fm = re.match(r'When ([a-z]) is divided by (\d+), the quotient is ([a-z])\.$', s)
        pm = re.match(r'([a-z]) is (\d+) percent of ([a-z])\.$', s)
        matched = None
        if gm:
            v, val = letter_to_idx(gm.group(1)), int(gm.group(2))
            cand = [f for f in remaining if f['ftype']=='given' and f['var']==v and f['value']==val]
            matched = cand[0] if cand else None
        elif am:
            a,b,c = [letter_to_idx(x) for x in am.groups()]
            cand = [f for f in remaining if f['ftype']=='rel' and f['op']=='add' and f['args']==[a,b] and f['result']==c]
            matched = cand[0] if cand else None
        elif em:
            a,b,c = [letter_to_idx(x) for x in em.groups()]
            cand = [f for f in remaining if f['ftype']=='rel' and f['op']=='sub' and f['args']==[a,b] and f['result']==c]
            matched = cand[0] if cand else None
        elif mm:
            a,b,c = [letter_to_idx(x) for x in mm.groups()]
            cand = [f for f in remaining if f['ftype']=='rel' and f['op']=='mul' and f['args']==[a,b] and f['result']==c]
            matched = cand[0] if cand else None
        elif fm:
            a = letter_to_idx(fm.group(1)); k = int(fm.group(2)); q = letter_to_idx(fm.group(3))
            cand = [f for f in remaining if f['ftype']=='fdiv' and f['var']==a and f['k']==k and f['result']==q]
            matched = cand[0] if cand else None
        elif pm:
            a = letter_to_idx(pm.group(1)); p = int(pm.group(2)); b = letter_to_idx(pm.group(3))
            cand = [f for f in remaining if f['ftype']=='pct' and f['args']==[a,b] and f['p']==p]
            matched = cand[0] if cand else None
        else:
            issues.append((src, 'unrecognized sentence pattern', s))
            parsed_ok = False
            continue
        if matched is None:
            issues.append((src, 'sentence has no matching factor', s))
            parsed_ok = False
        else:
            remaining.remove(matched)
    if remaining:
        issues.append((src, 'unconsumed factors', remaining))

print(f"Checked {len(rows)} rows.")
print(f"Issues found: {len(issues)}")
for i in issues:
    print(i)
