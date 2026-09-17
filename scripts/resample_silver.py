"""resample_silver.py — THE RESAMPLER (2026-09-16, word given): silver x K.
Each admitted silver row's STORY is kept word for word — every span and
mention re-offset exactly — and its resamplable GIVENS (those whose value
appears as a digit numeral in the text; hidden constants like "twice" -> 2
stay) take new values; the answer is recomputed by the June solver and each
copy must CERTIFY UNIQUE (doors.certify_unique) with every value an
integer <= VALUE_CAP. A numeral is rewritten at EVERY occurrence in the
text so the story stays consistent; a value shared by two givens is left
alone (ambiguous). Copies are stamped gen.resample = {parent, k, key:
"solver"} — their key is DERIVED from the parent's certified graph, so they
are silver-of-silver and count as REPS PER UNIQUE under the dose law.
usage: resample_silver.py in.jsonl out.jsonl K [seed]"""
import json, re, sys, random
sys.path.insert(0, "."); sys.path.insert(0, "scripts")
import signal
from admit_annotation import VALUE_CAP, solve_ladder, _Timeout, _alarm
from mycelium.csp_domains import problem_from_algebra3
from mycelium.csp_core import solve_symbolic
from mycelium.macros import expand_graph
from mycelium.doors import certify_unique

def build(factors, n_vars, m=None):
    fs, nv = expand_graph([dict(f) for f in factors], n_vars); gv = {f["var"]: f["value"] for f in fs if f["ftype"] == "given"}
    # the row-sized domain first (2x its largest given, floor 300); the ladder in solve_certified widens it
    if m is None: m = min(VALUE_CAP + 1, max(300, 2 * max([f["value"] for f in fs if f["ftype"] == "given"] + [1])))
    return problem_from_algebra3(nv, fs, gv, m)


def solve_certified(factors, n_vars, query_var, budget=5000, wall=2):
    """the solve + the uniqueness certificate under a SHORT wall clock per rung of the domain ladder
    (admit_annotation.solve_ladder): a solvable copy solves in ~0.03 s at 10^4 / ~0.2 s at 10^5, so the
    wall only ever bites UNSAT draws (a resampled given that feeds a division is rarely divisible) — at
    30 s per rung those draws made the resampler crawl at minutes per parent (2026-09-16). A copy that
    does not SOLVE and certify inside the wall is dropped (conservative — never a copy without a certificate)"""
    r, m = solve_ladder(lambda m: build(factors, n_vars, m), build_m0(factors, n_vars), budget=budget, wall=wall)
    if r.get("status") != "solved": return None
    asg = [int(x) for x in r["assignment"][:n_vars]]
    if any(not (0 <= v <= VALUE_CAP) for v in asg): return None
    try:
        old = signal.signal(signal.SIGALRM, _alarm); signal.setitimer(signal.ITIMER_REAL, wall)
        try:
            if not certify_unique(build(factors, n_vars, m), query_var, asg[query_var], budget): return None
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0); signal.signal(signal.SIGALRM, old)
    except _Timeout:
        return None
    return asg


def build_m0(factors, n_vars):
    fs, nv = expand_graph([dict(f) for f in factors], n_vars)
    return min(VALUE_CAP + 1, max(300, 2 * max([f["value"] for f in fs if f["ftype"] == "given"] + [1])))

def rewrite(text, subs):
    """subs: {old_digit_string: new_digit_string}; every word-bounded occurrence; returns (new_text, shift)"""
    segs = []; out = []; last = 0
    pat = re.compile(r"(?<![\w.])(" + "|".join(re.escape(k) for k in subs) + r")(?![\w]|\.\d)")
    for m in pat.finditer(text):
        out.append(text[last:m.start()]); ns = sum(len(x) for x in out); new = subs[m.group(1)]; out.append(new)
        segs.append((m.start(), m.end(), ns, ns + len(new))); last = m.end()
    out.append(text[last:]); new_text = "".join(out)
    def shift(o):
        for os_, oe, ns, ne in segs:
            if os_ < o < oe: return ns + min(o - os_, ne - ns)
        return o + sum((ne - ns) - (oe - os_) for os_, oe, ns, ne in segs if oe <= o)
    return new_text, shift

def resample(row, K, rng):
    text = row["text"]; facs = row["factors"]; n = row["n_vars"]; q = row["query_var"]
    givens = [f for f in facs if f["ftype"] == "given"]
    counts = {}
    for f in givens: counts[f["value"]] = counts.get(f["value"], 0) + 1
    res = []
    for f in givens:
        s = str(f["value"])
        if counts[f["value"]] != 1 or not f.get("spans"): continue
        sp = f["spans"][0]; span_text = text[sp[0]:sp[1]]
        if re.search(r"(?<![\w.])" + re.escape(s) + r"(?![\w]|\.\d)", span_text): res.append(f)
    if not res: return []
    copies = []; tries = 0
    while len(copies) < K and tries < 20 * K:
        if tries >= 12 and not copies: break   # a parent whose first 12 draws never solve is not going to (261 s for nothing, timed 2026-09-16)
        tries += 1; new_facs = [dict(f) for f in facs]; subs = {}
        for f in new_facs:
            if f["ftype"] == "given" and any(f["var"] == g["var"] for g in res):
                v = f["value"]
                if tries % 2:   # shape-preserving draw: v x a / b with b | v, small a, b — divisibility the graph relies on survives more often
                    bs = [b for b in (1, 2, 3, 4, 5, 6) if v % b == 0]; b = rng.choice(bs); a = rng.choice([x for x in (1, 2, 3, 4, 5, 6) if x != b])
                    nv_ = v * a // b
                else:           # uniform draw in [v/3, 3v]
                    lo, hi = max(1, v // 3), min(VALUE_CAP, max(v * 3, v + 5)); nv_ = rng.randint(lo, hi)
                if nv_ == v or not (1 <= nv_ <= VALUE_CAP): continue
                subs[str(v)] = str(nv_); f["value"] = nv_
        if not subs or len(set(subs.values())) != len(subs) or any(v in subs for v in subs.values()): continue
        asg = solve_certified(new_facs, n, q)
        if asg is None: continue
        new_text, sh = rewrite(text, subs)
        for f in new_facs:
            if f.get("spans"): f["spans"] = [[sh(a), sh(b)] for a, b in f["spans"]]
        ments = {k: [[sh(a), sh(b)] for a, b in v] for k, v in row.get("mentions", {}).items()}
        c = dict(row); c.update({"text": new_text, "factors": new_facs, "mentions": ments, "answer_field": f"#### {asg[q]}", "solution": asg, "m": VALUE_CAP + 1,
                                 "gen": {**row.get("gen", {}), "resample": {"parent_rank": row.get("gen", {}).get("queue_rank"), "k": len(copies) + 1, "key": "solver", "subs": subs}}})
        copies.append(c)
    return copies

if __name__ == "__main__":
    src, dst, K = sys.argv[1], sys.argv[2], int(sys.argv[3]); rng = random.Random(int(sys.argv[4]) if len(sys.argv) > 4 else 0)
    rows = [json.loads(l) for l in open(src)]; n_out = 0; n_par = 0
    # RS_RESUME=1: keep an existing dst and skip every parent that already has a copy in it (matched by the
    # story with numerals masked — the rewrite touches numerals only); the in-flight parent of a killed run
    # may keep fewer than K copies. The rng is re-seeded per parent from its index so a resume draws the same values.
    import os as _os; done = set()
    if _os.environ.get("RS_RESUME") and _os.path.exists(dst):
        for l in open(dst):
            c = json.loads(l); done.add(re.sub(r"\d+", "#", c["text"])); n_out += 1
        print(f"[resample] resume: {n_out} copies kept, {len(done)} parents already done", flush=True)
    with open(dst, "a" if done else "w") as f:
        for i, r in enumerate(rows):
            if re.sub(r"\d+", "#", r["text"]) in done: continue
            rng = random.Random(hash((i, sys.argv[4] if len(sys.argv) > 4 else "0")) & 0xffffffff)
            cs = resample(r, K, rng); n_par += bool(cs)
            for c in cs: f.write(json.dumps(c) + "\n"); n_out += 1
    print(f"[resample] {len(rows)} parents -> {n_out} certified copies ({n_par} parents resamplable; K={K}) -> {dst}")
    if n_out:
        c = json.loads(open(dst).readline()); print("[resample] sample:", c["gen"]["resample"]["subs"], "|", c["text"][:150], "| answer", c["answer_field"])
        # the offset audit on the first copy: every span/mention selects the same words as its parent (numerals excepted)
        p = next(r for r in rows if r.get("gen", {}).get("queue_rank") == c["gen"]["resample"]["parent_rank"]); bad = 0
        for fa, fb in zip(p["factors"], c["factors"]):
            for (s1, e1), (s2, e2) in zip(fa.get("spans", []), fb.get("spans", [])):
                if re.sub(r"\d+", "#", p["text"][s1:e1]) != re.sub(r"\d+", "#", c["text"][s2:e2]): bad += 1
        print(f"[resample] offset audit on the sample copy: {bad} span mismatches")
