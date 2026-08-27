"""funnel_v2.py — THE INDUSTRIAL FUNNEL v2 (2026-08-28; the faithfulness
guards constitutional): (G1) query DERIVED through >=1 op factor, never
a given; (G2) graph PRUNED to the query's ancestors — junk factors die;
(G3) every given value is an anchor-law surface number from the raw;
(G4) no given carries the gold answer unless gold itself appears in the
raw text; (G5) no degenerate factors (result among own args, duplicate
givens). Validation runs INSIDE the repair search; the full-text
faithfulness review remains constitutional and happens on this log.
Original v1 header follows for lineage: (2026-08-28):
bounded repair search on the NEAR tier + graph->dialect renderer +
bulk gate voting + span proposals. Output: a triage report; NOTHING
banks until reviewed (funnel_bank.py consumes approvals).
Repair operators (<=2 edits, key-gated, uniqueness via solve_forced):
value-fix (anchor-law surface numbers), op flip, factor drop, query
move. Renderer covers given/rel/fdiv; other kinds -> surgery lane.
"""
import json, sys, os, re, itertools
sys.path.insert(0, "."); sys.path.insert(0, "scripts")
os.environ.update({"DEV": "AMD", "ALG2": "1", "ALG_FTYPES": "8",
                   "ALG_DUP": "1", "ALG_HW": "512", "ALG_WIDE": "1"})
import numpy as np
from phase1_algebra_head import (T_ALG, build_params, forward, decode,
                                 sent_indices, TOKENIZER_JSON)
from beacon_closing_arm import recompute_states
from repair_replace_swap import solve_forced
from tta_views import permuted_view
from tta_alg2_dials import solve2
from tokenizers import Tokenizer
from tinygrad import Tensor, dtypes
from tinygrad.nn.state import safe_load
from collections import Counter

NUM = re.compile(r"(?<![\d.])(\d+)(?![\d.])")
LETT = "abcdefghijklmnopqrstuvwx"

def sf(facs, q):
    try:
        return solve_forced(facs, q, {"n_vars": 24, "m": 300})
    except Exception:
        return None


def ancestors(facs, q):
    prod = {}
    for f in facs:
        if f["ftype"] == "rel": prod[f["result"]] = f
        elif f["ftype"] == "fdiv": prod[f["result"]] = f
    keep = []; seen = set(); stack = [q]
    giv = {f["var"]: f for f in facs if f["ftype"] == "given"}
    while stack:
        v = stack.pop()
        if v in seen: continue
        seen.add(v)
        if v in prod:
            f = prod[v]; keep.append(f)
            if f["ftype"] == "rel": stack.extend(f["args"])
            else: stack.append(f["var"])
        elif v in giv:
            keep.append(giv[v])
    return keep[::-1]

def validate(facs, q, gold, nums):
    g = ancestors(facs, q)
    if not g: return None
    prod_q = any(f for f in g if f["ftype"] in ("rel", "fdiv")
                 and f.get("result") == q)
    if not prod_q: return None                       # G1: derived query
    seen_g = set()
    for f in g:
        if f["ftype"] == "given":
            if f["var"] in seen_g: return None       # G5: dup givens
            seen_g.add(f["var"])
            if f["value"] not in nums: return None   # G3: anchored givens
            if f["value"] == gold and gold not in nums:
                return None                          # G4: no planting
        if f["ftype"] == "rel" and f["result"] in f["args"]:
            return None                              # G5: degenerate
    if sf(g, q) != gold: return None                 # pruned graph re-keyed
    return g

def repairs(facs, q, nums):
    yield facs, q, 0
    edits = []
    for fi, f in enumerate(facs):
        if f["ftype"] == "given":
            for v in nums:
                if 0 <= v <= 300 and v != f["value"]:
                    g = [dict(x) for x in facs]; g[fi]["value"] = v
                    edits.append((g, q))
        if f["ftype"] == "rel":
            for op in ("add", "sub", "mul"):
                if op != f.get("op"):
                    g = [dict(x) for x in facs]; g[fi]["op"] = op
                    edits.append((g, q))
        g = [dict(x) for x in facs if x is not facs[fi]]
        if len(g) >= 1:
            edits.append((g, q))
    vs = {f.get("result", f.get("var")) for f in facs} - {None}
    for v in vs:
        if v != q: edits.append((facs, v))
    for g, qq in edits:
        yield g, qq, 1
    for (g1, q1), (g2, q2) in itertools.islice(
            itertools.combinations(edits, 2), 4000):
        if q1 == q and g1 is not facs:
            pass
        yield g2 if g1 is facs else g1, q2 if q1 == q else q1, 2

def render(facs, q):
    """graph -> canonical dialect; None if unrenderable (non-g/rel/fdiv)."""
    order = []
    def L_(v):
        if v not in order: order.append(v)
        return None
    for f in facs:
        if f["ftype"] == "given": L_(f["var"])
        elif f["ftype"] == "rel":
            for a in f["args"]: L_(a)
            L_(f["result"])
        elif f["ftype"] == "fdiv":
            L_(f["var"]); L_(f["result"])
        else:
            return None
    L_(q)
    if len(order) > 24: return None
    M = {v: LETT[i] for i, v in enumerate(order)}
    sents = []
    for f in facs:
        if f["ftype"] == "given":
            sents.append(f"{M[f['var']]} is {f['value']}.")
        elif f["ftype"] == "rel":
            a, b = f["args"][0], f["args"][1] if len(f["args"]) > 1 else f["args"][0]
            r = f["result"]
            if f.get("op") == "add":
                sents.append(f"{M[a]} plus {M[b]} equals {M[r]}.")
            elif f.get("op") == "sub":
                sents.append(f"{M[a]} exceeds {M[b]} by {M[r]}.")
            elif f.get("op") == "mul":
                sents.append(f"{M[a]} times {M[b]} equals {M[r]}.")
            else:
                return None
        elif f["ftype"] == "fdiv":
            k = f.get("k", f.get("divisor"))
            if k is None: return None
            sents.append(f"When {M[f['var']]} is divided by {k}, "
                         f"the quotient is {M[f['result']]}.")
    letters = ", ".join(LETT[i] for i in range(len(order)))
    return (f"Consider the numbers {letters}. " + " ".join(sents)
            + f" What is {M[q]}?")

LEX = {
 'addf': ['sum', 'total', 'plus', 'more than', 'increased', 'exceeds',
          'difference', 'less than', 'minus', 'subtract', 'left', 'remain',
          'older', 'younger', 'combined', 'altogether', 'gain', 'lost'],
 'mul': ['times', 'product', 'twice', 'double', 'tripl', 'multipl', 'each',
         'per', 'every', '\\times', 'cdot', 'ratio', 'rate', 'as many'],
 'fr': ['divided by', 'quotient', 'half', 'split', 'share', 'average',
        'mean', 'ratio', '\\div', 'frac', 'evenly', 'equally'],
}
def propose_spans(raw, facs):
    low = raw.lower(); out = []; claimed = []
    for f in facs:
        if f["ftype"] == "rel":
            cls = ('mul' if f.get("op") == "mul" else 'addf')
        elif f["ftype"] == "fdiv":
            cls = 'fr'
        else:
            continue
        for cue in LEX[cls]:
            i = low.find(cue)
            if i >= 0 and not any(i < b and i + len(cue) > a for a, b in claimed):
                claimed.append((i, i + len(cue)))
                out.append({"op": {'addf': 'addf', 'mul': 'mul', 'fr': 'fr'}[cls],
                            "span": [i, i + len(cue)], "cue": raw[i:i+len(cue)],
                            "source": "funnel-auto"})
                break
    return out

def main():
    Lz = json.load(open(".cache/book3_lanes.json"))
    BY = {l["idx"]: l for l in Lz}
    done = set()
    for l in open('.cache/book3.jsonl'):
        done.add(json.loads(l)['raw'])
    cands = [x for x in Lz if x.get('lane') == 'L2'
             and (x.get('problem') or x.get('raw', '')) not in done]
    tok = Tokenizer.from_file(TOKENIZER_JSON)
    p = build_params(0)
    sd = safe_load(".cache/g41_onemass_refold.safetensors")
    for k in p:
        p[k].assign(sd[k].to(p[k].device).cast(p[k].dtype)).realize()

    def parse_batch(texts):
        n = len(texts)
        N = ((n + 7) // 8) * 8
        ids = np.zeros((N, T_ALG), np.int32); msk = np.zeros((N, T_ALG), np.float32)
        snt = np.zeros((N, T_ALG), np.int32)
        for i, t in enumerate(texts):
            e = tok.encode(t)
            Ln = min(len(e.ids), T_ALG)
            ids[i, :Ln] = e.ids[:Ln]; msk[i, :Ln] = 1.0
            snt[i] = sent_indices(t, list(e.offsets), msk[i])
        st = recompute_states(ids)
        res = []
        for s0 in range(0, N, 8):
            out = forward(p, Tensor(st[s0:s0+8].astype(np.float32), dtype=dtypes.float),
                          Tensor(msk[s0:s0+8].astype(np.float32), dtype=dtypes.float),
                          Tensor(snt[s0:s0+8].astype(np.int32), dtype=dtypes.int))
            keys = ("pres","ftype","op","islit","dig","args","res","query") + \
                tuple(k2 for k2 in ("sel","dup","sgn") if k2 in out)
            o = {k2: out[k2].realize().numpy() for k2 in keys}
            for bi in range(8):
                if s0 + bi < n:
                    res.append(decode({k2: o[k2][bi] for k2 in o}))
        return res

    raws = [c.get('problem') or c.get('raw', '') for c in cands]
    parses = parse_batch(raws)
    fixed = []
    for c, raw, (facs, q) in zip(cands, raws, parses):
        gold = c['answer']
        nums = [int(m.group(1)) for m in NUM.finditer(raw)]
        found = None
        for g, qq, ne in repairs(facs, q, nums):
            if sf(g, qq) == gold:
                gv = validate(g, qq, gold, nums)
                if gv is not None:
                    found = (gv, qq, ne); break
        if found:
            dia = render(found[0], found[1])
            if dia and sf(found[0], found[1]) == gold:
                fixed.append((c['idx'], gold, dia, found[2], raw))
    print(f"[fr] repaired+rendered: {len(fixed)}/{len(cands)}", flush=True)
    banked = []
    for ti, (li, gold, dia, ne, raw) in enumerate(fixed):
        texts = [dia] + [permuted_view(dia, 90000 + 100*ti + k) for k in range(1, 5)]
        votes = []
        for facs, q in parse_batch(texts):
            a = None
            try: a = solve2(facs, q, {"n_vars": 24, "m": 300})
            except Exception: pass
            if a is not None: votes.append(a)
        top, cnt = (Counter(votes).most_common(1)[0] if votes else (None, 0))
        ok = cnt >= 3 and top == gold
        spans = propose_spans(raw, [f for f in parses[0][0]]) if False else \
            propose_spans(raw, [])
        print(f"  [{li:3d}] gold {gold:>4} edits {ne} | votes {votes} -> "
              f"{'GATE-OK' if ok else 'refuses'}", flush=True)
        print(f"      RAW: {raw[:180]}", flush=True)
        print(f"      DIA: {dia}", flush=True)
        if ok:
            banked.append(dict(lane_idx=li, raw=raw, dialect=dia, answer=gold,
                               m=300, lane="L2", book=3, tranche=12,
                               gate="funnel-repair+5view+key",
                               generation="41", op_spans=[]))
    json.dump(banked, open('.cache/funnel_v2_gateok.json', 'w'))
    print(f"[fr] GATE-OK awaiting review: {len(banked)} -> .cache/funnel_v2_gateok.json",
          flush=True)

if __name__ == "__main__":
    main()
