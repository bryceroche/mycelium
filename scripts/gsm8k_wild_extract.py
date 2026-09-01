"""gsm8k_wild_extract.py — THE GSM8K WILD LANE, pilot (2026-08-31).
Machine-extracts factor-graph drafts from GSM8K train calculator
annotations (<<a op b = c>>). Custody: the annotation only PROPOSES;
the gate is mechanical — propagation must reach the dataset's #### key,
every given must anchor to a literal in the QUESTION text, all values
integer in [0, M], sub/div re-encoded per the grammar (x-y=d ->
add(y,d)=x; a/b=c -> mul(c,b)=a), n_vars<=24, factors<=24.
GSM8K test split NEVER touched. Output: .cache/gsm8k_wild_drafts.jsonl
+ a reject census. Zero GPU.
"""
import json, re, sys
from collections import Counter

M = 300
SNAP = ('.cache/gsm8k/datasets--openai--gsm8k/snapshots/'
        '740312add88f781978c0658806c59bc2815b9866/main/'
        'train-00000-of-00001.parquet')
CALC = re.compile(r'<<([^<>=]+)=([\d,\.]+)>>')
TERM = re.compile(r'^\s*([\d,]+(?:\.\d+)?)\s*$')


def parse_expr(expr):
    """left-to-right chain of +-*/ over numeric literals; None if not."""
    toks = re.split(r'([+\-*/])', expr.replace(',', ''))
    if len(toks) < 3 or len(toks) % 2 == 0:
        return None
    try:
        vals = [float(toks[i]) for i in range(0, len(toks), 2)]
    except ValueError:
        return None
    ops = [toks[i] for i in range(1, len(toks), 2)]
    return vals, ops


def extract(question, rationale, key):
    if re.search(r'\d\.\d', question):
        return None, 'decimal-in-question'
    qnums = set()
    for x in re.findall(r'\d[\d,]*', question):
        qnums.add(int(x.replace(',', '')))
    calcs = CALC.findall(rationale)
    if not calcs:
        return None, 'no-calc-annotations'
    nv = 0
    factors = []
    by_val = {}          # value -> var id (prefer derived results)

    def new_var(v):
        nonlocal nv
        i = nv
        nv += 1
        return i

    def get_input(v):
        """resolve a calc input: prior result, else anchored given."""
        if v in by_val:
            return by_val[v]
        if v != int(v) or not (0 <= v <= M):
            return None
        v = int(v)
        if v == 1 or v in qnums:
            i = new_var(v)
            factors.append({"ftype": "given", "var": i, "value": v})
            by_val[v] = i
            return i
        return None

    def emit(a, op, b, r):
        """one binary step; returns result var or None."""
        for v in (a, b, r):
            if v != int(v):
                return 'non-integer'
        if not (0 <= r <= M):
            return 'bounds'
        ia = get_input(a)
        if ia is None:
            return 'unanchored' if 0 <= a <= M and a == int(a) else 'bounds'
        if op in ('+', '*'):
            ib = get_input(b)
            if ib is None:
                return 'unanchored' if 0 <= b <= M and b == int(b) else 'bounds'
            ir = new_var(r)
            factors.append({"ftype": "rel", "op": "add" if op == '+' else "mul",
                            "args": [ia, ib], "result": ir})
        elif op == '-':
            if a - b != r:
                return 'calc-mismatch'
            ib = get_input(b)
            if ib is None:
                return 'unanchored' if 0 <= b <= M and b == int(b) else 'bounds'
            ir = new_var(r)
            # a - b = r  ->  add(b, r) = a
            factors.append({"ftype": "rel", "op": "add",
                            "args": [ib, ir], "result": ia})
        elif op == '/':
            if b == 0 or a / b != r:
                return 'calc-mismatch'
            ib = get_input(b)
            if ib is None:
                return 'unanchored' if 0 <= b <= M and b == int(b) else 'bounds'
            ir = new_var(r)
            # a / b = r  ->  mul(r, b) = a
            factors.append({"ftype": "rel", "op": "mul",
                            "args": [ir, ib], "result": ia})
        else:
            return 'op'
        by_val[r] = ir           # results shadow given values
        return ir

    for expr, res in calcs:
        p = parse_expr(expr)
        try:
            rv = float(res.replace(',', ''))
        except ValueError:
            return None, 'bad-result'
        if p is None:
            return None, 'expr-parse'
        vals, ops = p
        acc = vals[0]
        for k, op in enumerate(ops):
            b = vals[k + 1]
            if op == '+':   r = acc + b
            elif op == '-': r = acc - b
            elif op == '*': r = acc * b
            elif op == '/': r = acc / b if b else None
            else:           return None, 'op'
            if r is None:
                return None, 'div-zero'
            # intermediate of a chained expr = its own step
            err = emit(acc, op, b, r)
            if isinstance(err, str):
                return None, err
            acc = r
        if acc != rv:
            return None, 'calc-mismatch'
    if key not in by_val:
        return None, 'key-not-derived'
    if nv > 24 or len([f for f in factors if f["ftype"] == "rel"]) > 24:
        return None, 'slot-overflow'
    q = by_val[key]
    if any(f["ftype"] == "given" and f["var"] == q for f in factors):
        return None, 'answer-stuffing'
    return {"n_vars": nv, "m": M, "query_var": q, "factors": factors}, None


def solve_check(d, key):
    """forward-propagate; must pin query to key with no search."""
    val = {}
    for f in d["factors"]:
        if f["ftype"] == "given":
            val[f["var"]] = f["value"]
    changed = True
    while changed:
        changed = False
        for f in d["factors"]:
            if f["ftype"] != "rel":
                continue
            a, b, r = f["args"][0], f["args"][1], f["result"]
            ka, kb, kr = a in val, b in val, r in val
            if f["op"] == "add":
                if ka and kb and not kr: val[r] = val[a] + val[b]; changed = True
                elif ka and kr and not kb: val[b] = val[r] - val[a]; changed = True
                elif kb and kr and not ka: val[a] = val[r] - val[b]; changed = True
                elif ka and kb and kr and val[a] + val[b] != val[r]: return False
            else:
                if ka and kb and not kr: val[r] = val[a] * val[b]; changed = True
                elif ka and kr and not kb and val[a]:
                    if val[r] % val[a]: return False
                    val[b] = val[r] // val[a]; changed = True
                elif kb and kr and not ka and val[b]:
                    if val[r] % val[b]: return False
                    val[a] = val[r] // val[b]; changed = True
                elif ka and kb and kr and val[a] * val[b] != val[r]: return False
    return (len(val) == d["n_vars"] and val[d["query_var"]] == key
            and all(0 <= v <= M for v in val.values()))


import pyarrow.parquet as pq
t = pq.read_table(SNAP).to_pydict()
qs, ans = t['question'], t['answer']
out, rej = [], Counter()
for i in range(len(qs)):
    a = ans[i]
    tail = a.split('####')[-1].strip().replace(',', '')
    try:
        key = int(tail)
    except ValueError:
        rej['non-integer-key'] += 1
        continue
    if not (0 <= key <= M):
        rej['bounds'] += 1
        continue
    d, err = extract(qs[i], a, key)
    if d is None:
        rej[err] += 1
        continue
    if not solve_check(d, key):
        rej['solve-check'] += 1
        continue
    d.update({"src": "gsm8k_train", "src_idx": i,
              "original": qs[i], "answer": key})
    out.append(d)
with open('.cache/gsm8k_wild_drafts.jsonl', 'w') as fh:
    for d in out:
        fh.write(json.dumps(d) + '\n')
print(f"[gsm8k wild] {len(out)} drafts from {len(qs)} "
      f"({len(out) / len(qs) * 100:.1f}%) -> .cache/gsm8k_wild_drafts.jsonl")
print("[rejects]", dict(rej.most_common()))
import statistics
depths = [len([f for f in d['factors'] if f['ftype'] == 'rel']) for d in out]
print(f"[shape] rels/row: median {statistics.median(depths)}, "
      f"max {max(depths)}; n_vars median "
      f"{statistics.median([d['n_vars'] for d in out])}")
