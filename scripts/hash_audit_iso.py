"""hash_audit_iso.py — THE ANTI-COLLISION AUDIT (2026-07-12, registered).

The scariest hash failure is semantic and inverse: ISOMORPHIC DUPLICATES
THAT HASH DIFFERENTLY. The mint dedups by exact text; two isomorphic
problems (same knot, different diagram — relabeled letters, permuted
factors) survive dedup, and across the train/test boundary that is
CONTAMINATION no string check can see, because the system was designed
to make surface variation cheap.

The invariant: canonical form under variable renaming + factor
reordering, VALUES INCLUDED (givens, k, p, op, sel — a problem's
identity includes its numbers). WL color refinement (6 rounds, bipartite
var/factor, commutative roles sorted) -> canonical digest; any
cross-boundary digest match is verified EXACTLY by backtracking over
color classes before it counts (WL can falsely merge; the audit cannot).

Reads: train = the current training mix; tests = the full battery.
Reports per-corpus: exact-text overlap AND verified isomorph count.
Zero = the hoped answer; nonzero = every bar since the mint gets a
footnote. The canonical digest doubles as the RIGHT problem ID going
forward (knot invariant, not diagram fingerprint).
"""
import json, sys, hashlib
from collections import defaultdict

TRAIN = ".cache/algebra_mixed8_train.jsonl"
TESTS = {
    "dag8test":  ".cache/dag8_test.jsonl",
    "dag7btest": ".cache/dag7b_test.jsonl",
    "dagtest":   ".cache/dag_test.jsonl",
    "bigtest":   ".cache/algebra_nl_bigtest.jsonl",
    "alg2test":  ".cache/algebra2_nl_test.jsonl",
    "alg4test":  ".cache/algebra4_nl_test.jsonl",
    "vtest":     ".cache/algv_test_verbose.jsonl",
}


def fdesc(f):
    """(kind-params, ((role, var), ...)) — semantic fields only."""
    ft = f["ftype"]
    if ft == "rel":
        return (("rel", f["op"]),
                (("a", f["args"][0]), ("a", f["args"][1]),
                 ("r", f["result"])))
    if ft == "given":
        return (("giv", int(f["value"])), (("v", f["var"]),))
    if ft == "mod":
        return (("mod", int(f["k"])), (("s", f["var"]), ("r", f["result"])))
    if ft == "fdiv":
        return (("fdv", int(f["k"])), (("s", f["var"]), ("r", f["result"])))
    if ft == "pct":
        return (("pct", int(f["p"])),
                (("p", f["args"][0]), ("b", f["args"][1])))
    if ft == "sel":
        return (("sel", f["sel"]),
                (("a", f["args"][0]), ("a", f["args"][1]),
                 ("r", f["result"])))
    raise ValueError(ft)


def canon(row):
    facs = [fdesc(f) for f in row["factors"]]
    nv = row["n_vars"]
    q = row["query_var"]
    col = {v: ("Q" if v == q else ".") for v in range(nv)}
    for _ in range(6):
        # factor colors from member var colors (commutative role 'a' sorted)
        fcols = []
        for kind, mem in facs:
            aa = tuple(sorted(col[m] for r, m in mem if r == "a"))
            rest = tuple((r, col[m]) for r, m in mem if r != "a")
            fcols.append((kind, aa, rest))
        # var colors from incident (factor color, role) multiset
        inc = defaultdict(list)
        for (kind, mem), fc in zip(facs, fcols):
            for r, m in mem:
                inc[m].append((fc, r))
        col = {v: (col[v], tuple(sorted(map(repr, inc[v]))))
               for v in range(nv)}
        # intern to ranks (keeps colors small)
        ranks = {c: i for i, c in enumerate(sorted(set(map(repr, col.values()))))}
        col = {v: ranks[repr(c)] for v, c in col.items()}
    sig = sorted(repr((kind,
                       tuple(sorted(col[m] for r, m in mem if r == "a")),
                       tuple((r, col[m]) for r, m in mem if r != "a")))
                 for kind, mem in facs)
    return hashlib.sha256(("|".join(sig) + f"#q{col[q]}").encode()).hexdigest(), col


def verify_iso(ra, rb):
    """Exact isomorphism check by backtracking (small graphs)."""
    fa, fb = [fdesc(f) for f in ra["factors"]], [fdesc(f) for f in rb["factors"]]
    if ra["n_vars"] != rb["n_vars"] or len(fa) != len(fb):
        return False
    if sorted(k for k, _ in fa) != sorted(k for k, _ in fb):
        return False
    qa, qb = ra["query_var"], rb["query_var"]
    bind = {qa: qb}

    def match(i, used_b):
        if i == len(fa):
            return True
        ka, mema = fa[i]
        for j, (kb, memb) in enumerate(fb):
            if j in used_b or kb != ka:
                continue
            # role-aligned member matching; commutative 'a' tries both orders
            aa = [m for r, m in mema if r == "a"]
            ab = [m for r, m in memb if r == "a"]
            oa = [(r, m) for r, m in mema if r != "a"]
            ob = [(r, m) for r, m in memb if r != "a"]
            if [r for r, _ in oa] != [r for r, _ in ob]:
                continue
            orders = [ab] if len(aa) < 2 else [ab, ab[::-1]]
            for ord_b in orders:
                trial = dict(bind)
                ok = True
                for va, vb in list(zip(aa, ord_b)) + \
                        [(m1, m2) for (_, m1), (_, m2) in zip(oa, ob)]:
                    if trial.get(va, vb) != vb or \
                       any(k != va and v == vb for k, v in trial.items()):
                        ok = False; break
                    trial[va] = vb
                if ok:
                    saved = bind.copy()
                    bind.clear(); bind.update(trial)
                    if match(i + 1, used_b | {j}):
                        return True
                    bind.clear(); bind.update(saved)
        return False
    return match(0, set())


train_rows = [json.loads(l) for l in open(TRAIN)]
train_by_digest = defaultdict(list)
train_texts = set()
for r in train_rows:
    train_by_digest[canon(r)[0]].append(r)
    train_texts.add(r["text"])
n_classes = len(train_by_digest)
n_multi = sum(1 for v in train_by_digest.values() if len(v) > 1)
print(f"[iso] train {len(train_rows)} rows -> {n_classes} canonical classes "
      f"({n_multi} classes with >1 member — within-train redundancy)")

total_iso = total_txt = 0
contaminated = []
for name, path in TESTS.items():
    rows = [json.loads(l) for l in open(path)]
    txt = sum(1 for r in rows if r["text"] in train_texts)
    iso = 0
    for ri, r in enumerate(rows):
        dg = canon(r)[0]
        if dg in train_by_digest and \
                any(verify_iso(r, m) for m in train_by_digest[dg]):
            iso += 1
            contaminated.append({"corpus": name, "row": ri, "digest": dg})
    total_iso += iso; total_txt += txt
    flag = "  <-- CONTAMINATION" if iso else ""
    print(f"  {name:10s}: n={len(rows):4d} | exact-text overlap {txt:3d} | "
          f"VERIFIED isomorphs in train {iso:3d}{flag}")
print(f"\n[iso] TOTAL cross-boundary: exact-text {total_txt}, "
      f"verified isomorphs {total_iso}")
json.dump(contaminated, open(".cache/iso_contamination.json", "w"))
print("[iso] exclusion list saved -> .cache/iso_contamination.json")
print("[iso] canonical digest = the knot-invariant problem ID going forward")
