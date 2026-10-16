"""reading_order.py -- THE READING-ORDER RE-SERIALIZATION (2026-10-16; Bryce's KV-table idea, word given for steps 1-3).

THE IDEA: a prose row's gold serializes its givens in DERIVATION order (each given just before the relation that
consumes it); a given slot's identity is then a planning fact the text does not show (the swap census, ledger
2026-10-16: gold given order == text-mention order on 135 / 289 wild rows). Re-serialize as two blocks:
  READING  -- every given, in TEXT-MENTION order (slot k = the k-th mentioned number): pure reading;
  PLANNING -- every other factor, in the row's original relative order; its argument pointers into the reading
              block ARE the mapping table (mention order -> the order the solution needs them).
Variables are relabeled by FIRST APPEARANCE over the new factor list (given: var; rel/sel: args then result;
mod/fdiv: var then result; pct: args; macro: x [y] result / xs result) -- the convention the diet already
follows (42.5k / 50.6k rows reproduce their own labels under it). The graph is unchanged up to relabeling: the
canon digest (the knot-ID door) and the solved query answer are checked on every row.

SCOPE: PROSE rows only. LETTERED rows (mint: "Consider the numbers a, b, c ..." / "Let e, b, f be whole numbers")
are OUT: a variable's identity is its letter, visible in the text, and the loader already span-sorts them.

THE TIE-BREAK RULES (pinned before any training reads them):
  R1 a given value bound to exactly one text numeral not claimed by another given -> that numeral's position;
  R2 a value mentioned MORE times than the givens carrying it -> the earliest unused mentions (flag MULTI);
     the flag records whether the row's given ORDER would change under the latest mentions (ORDER-SENSITIVE);
  R3 equal-valued givens -> keep their gold relative order among themselves (flag DUPVAL);
  R4 a given whose value no text numeral carries (implicit constants: 60 min/h, 'twice', 'dozen' unless spelled,
     percent's 100, ...) -> placed AFTER all bound givens, in gold order (flag UNBOUND);
  a row with spans on its givens uses the earliest span start instead of the numeral search (flag SPANS).

usage: reading_order.py IN.jsonl OUT.jsonl [--solve all|N] [--workers 16]
Writes OUT.jsonl (every row; in-scope rows re-serialized and stamped gen.canonical="reading", out-of-scope rows
verbatim) and OUT.jsonl.census.json. Hard-errors on any digest or answer mismatch.
"""
import argparse
import collections
import json
import multiprocessing as mp
import re
import sys

sys.path.insert(0, "."); sys.path.insert(0, "scripts")

LETTERED = re.compile(r"\b(Let|numbers|about|variables|quantities)\s+[a-z](, [a-z])+(,? and [a-z])?\b")   # a DECLARED letter list
UNITS = {"zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8,
         "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
         "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19}
TENS = {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80,
        "ninety": 90}
WORDS = {**UNITS, **TENS, "dozen": 12, "hundred": 100, "thousand": 1000}
NUM_RE = re.compile(r"\d[\d,]*(?:\.\d+)?|[A-Za-z]+(?:-[A-Za-z]+)?")


def numerals(text):
    """[(char_pos, value)] for every integer-valued numeral in text order: digit runs (commas stripped; decimals
    kept only when integral), spelled words 0-19 / tens / tens-units ('twenty-one'), 'dozen', 'hundred'."""
    out = []
    for m in NUM_RE.finditer(text):
        s = m.group()
        if s[0].isdigit():
            try:
                x = float(s.replace(",", ""))
            except ValueError:
                continue
            if x == int(x):
                out.append((m.start(), int(x)))
            continue
        w = s.lower()
        if "-" in w:
            a, b = w.split("-", 1)
            if a in TENS and b in UNITS and UNITS[b] > 0:
                out.append((m.start(), TENS[a] + UNITS[b]))
            continue
        if w in WORDS:
            out.append((m.start(), WORDS[w]))
    return out


def vars_of(f):
    t = f["ftype"]
    if t == "given":
        return [f["var"]]
    if t in ("rel", "sel"):
        return list(f["args"]) + [f["result"]]
    if t in ("mod", "fdiv"):
        return [f["var"], f["result"]]
    if t == "pct":
        return list(f["args"])
    if t == "macro":
        if f.get("name") == "CHAIN_MUL":
            return list(f["xs"]) + [f["result"]]
        return [f["x"]] + ([f["y"]] if "y" in f else []) + [f["result"]]
    raise ValueError(f"unknown ftype {t!r}")


def relabel_factor(f, mp_):
    g = dict(f)
    t = f["ftype"]
    if t == "given":
        g["var"] = mp_[f["var"]]
    elif t in ("rel", "sel"):
        g["args"] = [mp_[a] for a in f["args"]]; g["result"] = mp_[f["result"]]
    elif t in ("mod", "fdiv"):
        g["var"] = mp_[f["var"]]; g["result"] = mp_[f["result"]]
    elif t == "pct":
        g["args"] = [mp_[a] for a in f["args"]]
    elif t == "macro":
        if f.get("name") == "CHAIN_MUL":
            g["xs"] = [mp_[a] for a in f["xs"]]
        else:
            g["x"] = mp_[f["x"]]
            if "y" in f:
                g["y"] = mp_[f["y"]]
        g["result"] = mp_[f["result"]]
    return g


def in_scope(row):
    return not LETTERED.search(row["text"])


def mention_positions(row):
    """-> (positions: list aligned to the row's givens in gold order (None = unbound), flags: set)."""
    giv = [f for f in row["factors"] if f["ftype"] == "given"]
    flags = set()
    if giv and all(f.get("spans") for f in giv):
        flags.add("SPANS")
        return [min(s for s, _ in f["spans"]) for f in giv], flags
    nums = numerals(row["text"])
    by_val = collections.defaultdict(list)
    for p, v in nums:
        by_val[v].append(p)
    need = collections.Counter(int(f["value"]) for f in giv)
    pos, pos_late = [], []
    used = collections.Counter()
    for f in giv:
        v = int(f["value"])
        occ = by_val.get(v, [])
        if need[v] > 1:
            flags.add("DUPVAL")
        if len(occ) > need[v]:
            flags.add("MULTI")
        k = used[v]; used[v] += 1
        if k < len(occ):
            pos.append(occ[k])
            pos_late.append(occ[len(occ) - need[v] + k] if len(occ) >= need[v] else occ[k])
        else:
            pos.append(None); pos_late.append(None); flags.add("UNBOUND")
    if re.search(r"\$[^\d\s,.]|\\[a-zA-Z(\[]", row["text"]):     # TeX math, not a dollar amount
        flags.add("LATEX")
    if "MULTI" in flags and _order(giv, pos) != _order(giv, pos_late):
        flags.add("ORDER-SENSITIVE")
    return pos, flags


def _order(giv, pos):
    BIG = 10 ** 9
    return sorted(range(len(giv)), key=lambda k: (pos[k] is None, pos[k] if pos[k] is not None else BIG, k))


def reserialize(row):
    """-> (new_row, info). The givens in mention order, then every other factor in original order; variables
    relabeled by first appearance; mentions / query_var / solution carried through the relabel."""
    facs = row["factors"]
    gi = [i for i, f in enumerate(facs) if f["ftype"] == "given"]
    giv = [facs[i] for i in gi]
    pos, flags = mention_positions(row)
    order = _order(giv, pos)
    new_facs = [giv[k] for k in order] + [f for f in facs if f["ftype"] != "given"]
    seen = []
    for f in new_facs:
        for v in vars_of(f):
            if v not in seen:
                seen.append(v)
    n_vars = int(row.get("n_vars", 24))
    mp_ = {v: i for i, v in enumerate(seen)}
    rest = [v for v in range(n_vars) if v not in mp_]           # untouched vars keep their relative order
    for v in rest:
        mp_[v] = len(mp_)
    assert sorted(mp_.values()) == list(range(len(mp_))), "relabel is not a bijection"
    new = dict(row)
    new["factors"] = [relabel_factor(f, mp_) for f in new_facs]
    new["query_var"] = mp_[row["query_var"]]
    if row.get("mentions"):
        new["mentions"] = {str(mp_[int(k)]): v for k, v in row["mentions"].items()}
    if row.get("solution"):                                     # silver carries [] (its key lives in the harvest)
        sol = row["solution"]
        inv = {nv: ov for ov, nv in mp_.items()}
        assert all(v < len(sol) for v in seen), "solution shorter than the used variables"
        new["solution"] = [sol[inv[nv]] if inv[nv] < len(sol) else 0 for nv in range(max(len(sol), len(seen)))]
    gen = dict(row["gen"]) if isinstance(row.get("gen"), dict) else {}
    gen["canonical_was"] = gen.get("canonical")
    gen["canonical"] = "reading"
    gen["reading_flags"] = sorted(flags)
    new["gen"] = gen
    gold_order_was_text = order == list(range(len(giv)))
    return new, {"flags": flags, "already_text_order": gold_order_was_text, "n_giv": len(giv),
                 "perm": mp_}


def _digest(row):
    from mycelium.canonicalizer import canonical_digest
    return canonical_digest(row["factors"], row["query_var"], row.get("n_vars", 24))


def _solve(row):
    """the query answer through the June core (macros expanded first; the domain ladder)."""
    from admit_annotation import solve_ladder
    from mycelium.csp_domains import problem_from_algebra3
    from mycelium.macros import expand_graph
    facs, nv = expand_graph(row["factors"], int(row.get("n_vars", 24)))
    gv = {f["var"]: int(f["value"]) for f in facs if f["ftype"] == "given"}
    gmax = max([abs(v) for v in gv.values()] + [1])
    m0 = int(min(10001, max(300, 2 * gmax, int(row.get("m", 300)))))
    res, m = solve_ladder(lambda m: problem_from_algebra3(nv, facs, gv, m), m0, budget=5000, wall=10)
    if res.get("status") != "solved":
        return None, res.get("status")
    return int(res["assignment"][row["query_var"]]), "solved"


def _check(args):
    idx, old, new, do_solve = args
    d_old, d_new = _digest(old), _digest(new)
    out = {"idx": idx, "digest_ok": d_old == d_new}
    if do_solve:
        a_old, s_old = _solve(old)
        a_new, s_new = _solve(new)
        out.update(ans_old=a_old, ans_new=a_new, st_old=s_old, st_new=s_new)
    return out


def _key(row):
    sol, q = row.get("solution"), row["query_var"]
    return sol[q] if sol is not None and q < len(sol) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inp"); ap.add_argument("out")
    ap.add_argument("--solve", default="all", help="'all' or N (solve every N-th in-scope row; 0 = none)")
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    rows = [json.loads(l) for l in open(a.inp)]
    out_rows, jobs = [], []
    C = collections.Counter()
    every = 1 if a.solve == "all" else int(a.solve)
    for i, r in enumerate(rows):
        if not in_scope(r):
            C["out_of_scope_lettered"] += 1
            out_rows.append(r); continue
        new, info = reserialize(r)
        C["in_scope"] += 1
        C["already_text_order" if info["already_text_order"] else "reordered"] += 1
        if info["n_giv"] < 2:
            C["lt2_givens"] += 1
        for fl in info["flags"]:
            C["flag_" + fl] += 1
        if not info["flags"] - {"SPANS"}:
            C["clean_R1"] += 1
        out_rows.append(new)
        jobs.append((i, r, new, bool(every) and (C["in_scope"] - 1) % every == 0))
    with mp.Pool(a.workers) as pool:
        res = pool.map(_check, jobs, chunksize=64)
    bad_d = [x["idx"] for x in res if not x["digest_ok"]]
    solved = [x for x in res if "ans_old" in x]
    bad_a = [x for x in solved if x["st_old"] == "solved" and x["ans_old"] != x["ans_new"]]
    st = collections.Counter((x["st_old"], x["st_new"]) for x in solved)
    key_ok = sum(1 for x in solved if x["st_new"] == "solved"
                 and _key(rows[x["idx"]]) is not None and x["ans_new"] == _key(rows[x["idx"]]))
    no_key = sum(1 for x in solved if _key(rows[x["idx"]]) is None)
    census = {"input": a.inp, "output": a.out, "rows": len(rows), **C,
              "digest_mismatch": len(bad_d), "digest_mismatch_idx": bad_d[:20],
              "solved_checked": len(solved), "solve_status_pairs": {f"{k[0]}->{k[1]}": v for k, v in st.items()},
              "answer_mismatch": len(bad_a), "answer_mismatch_idx": [x["idx"] for x in bad_a[:20]],
              "new_answer_equals_key": key_ok, "no_key_in_row": no_key}
    print(json.dumps(census, indent=1))
    assert not bad_d, f"{len(bad_d)} digest mismatches -- the relabel changed a graph"
    assert not bad_a, f"{len(bad_a)} answer mismatches"
    with open(a.out, "w") as f:
        for r in out_rows:
            f.write(json.dumps(r) + "\n")
    with open(a.out + ".census.json", "w") as f:
        json.dump(census, f, indent=1)


if __name__ == "__main__":
    main()
