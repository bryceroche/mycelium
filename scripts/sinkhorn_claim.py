"""sinkhorn_claim.py -- THE SINKHORN CLAIM READ (2026-09-24, zero-GPU). chain_acc.py's numeral mask picks,
independently per slot, the text numeral that maximizes that slot's own digit-head log-probability
(mycelium.rulebook.choose_legal). Independent picks let two GIVEN slots both claim the same numeral
(a COLLISION) while another numeral in the text goes unclaimed. THE CLAIM asks whether solving the row as
an ASSIGNMENT PROBLEM instead -- each given claims exactly one numeral, each numeral claimed by at most one
given, maximizing total log-probability -- recovers any of chain_acc's masked-wild misses.

Two solvers over the same affinity matrix A (givens x candidate numerals, log-prob under the RAW digit
head, pre-mask):
  - Sinkhorn: balance exp(A/T) on a square-padded matrix (dummy rows/cols absorb any givens/numerals a
    side is short of) and take each given's row-argmax over the REAL numeral columns, at T in {0.5, 1, 2}.
  - Hungarian: scipy.optimize.linear_sum_assignment on -A (the hard, exact optimum of the same objective).

Relation slots are NEVER touched (derived values are legitimately reused); every other non-relation slot
(mod/pct/fdiv/macro) keeps chain_acc's ordinary per-slot numeral mask exactly as before -- only GIVEN slots
(ftype argmax == 1, decode()'s own convention, scripts/phase1_algebra_head.py:6256-6260) get the claim's
reassignment.

SELF-GATE (checked before anything else is trusted): the baseline (mask alone, no reassignment) must
reproduce chain_acc's masked wild read on PMS8_241 -- CORRECT 14/311.

Zero-GPU: only `from phase1_algebra_head import _decode_slots, N_DIG` (never build_params/forward) --
mirrors scripts/nl_certifier.py and scripts/beam_oracle.py's import pattern. Run under DEV=CPU and the
exact family env (sets N_DIG=7 / L_FAC=24 / K_VARS=24 to match the dump).

Inputs:
  SK_DUMP     raw per-row head dump, default .cache/rawslots_wild_PMS8_241.pkl
  SK_ROWS     fixture with gold factors[], default .cache/wild_admitted_holdout.jsonl
  SK_WALL     solver wall per attempt (s), default 3 (chain_acc's default)
  SK_WORKERS  spawn-pool workers, default 4
  SK_SELFGATE expected baseline CORRECT count, default 14
Output: .cache/sinkhorn_claim_PMS8_241.txt (+ prints progress).
"""
import os
import sys
import re
import json
import time
import pickle

sys.path.insert(0, "."); sys.path.insert(0, "scripts")
import numpy as np
from scipy.optimize import linear_sum_assignment

DUMP = os.environ.get("SK_DUMP", ".cache/rawslots_wild_PMS8_241.pkl")
ROWS_PATH = os.environ.get("SK_ROWS", ".cache/wild_admitted_holdout.jsonl")
OUT_TXT = os.environ.get("SK_OUT", ".cache/sinkhorn_claim_" + os.path.basename(DUMP).replace("rawslots_wild_", "").replace(".pkl", "") + ".txt")
WALL = float(os.environ.get("SK_WALL", "3"))
WORKERS = int(os.environ.get("SK_WORKERS", "4"))
SELF_GATE_EXPECT = int(os.environ.get("SK_SELFGATE", "14"))
TS = (0.5, 1.0, 2.0)
VARIANTS = ["baseline"] + [f"sinkhorn_T{t}" for t in TS] + ["hungarian"]


# ----------------------------------------------------------------------------------------------------------
# THE SOLVE (chain_acc._solve_task, verbatim semantics -- reused, never reimplemented, per the beam-oracle
# precedent). Module-level + picklable for the spawn context.
# ----------------------------------------------------------------------------------------------------------
def _solve_task(t):
    import os as _os, sys as _sys
    _sys.path.insert(0, "."); _sys.path.insert(0, "scripts")
    from admit_annotation import solve_walled
    from alternator_bridge import problem_from_algebra3
    task_id, q, key, parse, gv, nvv = t
    wall = float(_os.environ.get("SK_WALL", "3"))
    gmax = max([int(v) for v in gv.values()] + [1]); m0 = int(min(10001, max(300, 2 * gmax, 2 * key)))
    res = {"status": "?"}
    for m in ([m0] + ([10000] if m0 < 10000 else [])):
        try:
            res = solve_walled(problem_from_algebra3(nvv, parse, gv, m), budget=5000, wall=wall)
        except Exception:
            return task_id, "unbuildable", None
        if res.get("status") == "solved":
            return task_id, "solved", int(res["assignment"][q])
    return task_id, res.get("status", "?"), None


def build_gv_nvv(parse, q):
    used = [f.get("var") for f in parse if f["ftype"] == "given"] + \
           [a for f in parse if f["ftype"] == "rel" for a in list(f["args"]) + [f["result"]]]
    nvv = max([q + 1] + [v + 1 for v in used if v is not None])
    gv = {f["var"]: f["value"] for f in parse if f["ftype"] == "given"}
    return gv, nvv


def parse_sig(parse):
    """the wheel's own memo signature (_wheel_memo_key), reused verbatim."""
    return tuple(sorted(json.dumps({k: v for k, v in f.items() if k != "_slot"}, sort_keys=True, default=str) for f in parse))


# ----------------------------------------------------------------------------------------------------------
# per-row mechanics
# ----------------------------------------------------------------------------------------------------------
def given_slots(row):
    """decode()'s own convention (phase1_algebra_head.py:6246,6249,6256-6260): present (pres > 0) and
    ftype argmax == 1 ("given"; argmax == 0 is "rel" and is never touched)."""
    out = []
    for j in range(row["pres"].shape[0]):
        if row["pres"][j] <= 0:
            continue
        if int(row["ftype"][j].argmax()) == 1:
            out.append(j)
    return out


def _lsm(x):
    x = x - x.max(-1, keepdims=True)
    return x - np.log(np.exp(x).sum(-1, keepdims=True))


def affinity_row(dig_logits, candidates, nd):
    """log-probability of each candidate numeral under this slot's raw (pre-mask) digit head --
    mycelium.rulebook.choose_legal's own scoring, exposed as a vector instead of an argmax."""
    from mycelium.rulebook import digits_of
    ls = _lsm(dig_logits)
    return np.array([sum(ls[d, dd] for d, dd in enumerate(digits_of(v, nd))) for v in candidates], dtype=np.float64)


def onehot_digit_logits(v, nd, like):
    """mycelium.rulebook.legal_digit_logits' own output format for a fixed value v."""
    from mycelium.rulebook import digits_of
    fake = np.full_like(like, -1e9)
    for d, dd in enumerate(digits_of(v, nd)):
        fake[d, dd] = 0.0
    return fake


def mask_nonrel(row, text):
    """chain_acc's CA_MASK=1 loop verbatim (mask every non-relation slot's digits with the greedy legal
    choice) -- the baseline every variant starts from; given slots get overwritten again below."""
    from mycelium.rulebook import legal_digit_logits
    row = dict(row); row["dig"] = row["dig"].copy()
    for j in range(row["ftype"].shape[0]):
        if int(row["ftype"][j].argmax()) == 0:
            continue
        fake = legal_digit_logits(row["dig"][j], text)
        if fake is not None:
            row["dig"][j] = fake
    return row


def sinkhorn_balance(A, T, iters=200, eps=1e-12):
    """Balance exp(A/T) on a square-padded matrix (dummy rows/cols filled well below any real affinity,
    so real mass is preferred but every row/col sum stays finite and > 0). Returns the balanced (G, M)
    block restricted to the real givens x real candidates."""
    G, M = A.shape
    if G == 0 or M == 0:
        return np.zeros((G, M))
    N = max(G, M)
    fill = float(A.min()) - 10.0
    Apad = np.full((N, N), fill, dtype=np.float64)
    Apad[:G, :M] = A
    K = np.exp((Apad - Apad.max()) / T) + eps
    u = np.ones(N); v = np.ones(N)
    for _ in range(iters):
        u = 1.0 / (K @ v)
        v = 1.0 / (K.T @ u)
    P = (u[:, None] * K) * v[None, :]
    return P[:G, :M]


def apply_given_assignment(base_row, gslots, values, nd):
    """base_row (already mask_nonrel'd) with every given slot's digits overwritten by the ONE-HOT legal
    encoding of its assigned value."""
    row = dict(base_row); row["dig"] = row["dig"].copy()
    for j, v in zip(gslots, values):
        row["dig"][j] = onehot_digit_logits(v, nd, row["dig"][j])
    return row


def main():
    from phase1_algebra_head import _decode_slots, N_DIG
    from mycelium.rulebook import legal_values

    t0 = time.time()
    recs = pickle.load(open(DUMP, "rb"))
    _limit = int(os.environ.get("SK_LIMIT", "0"))
    if _limit:
        recs = recs[:_limit]
    n = len(recs)
    assert N_DIG == np.asarray(recs[0]["dig"]).shape[1], (
        f"N_DIG={N_DIG} from the head but the dump's digits are {np.asarray(recs[0]['dig']).shape[1]} wide -- "
        f"run under the FAMILY ENV (ALG_WIDE=1 sets N_DIG=7); the 09-23 facts-census trap")
    nd = N_DIG
    gold_rows = None
    if os.path.exists(ROWS_PATH):
        gold_rows = [json.loads(l) for l in open(ROWS_PATH)]
    print(f"[sinkhorn-claim] {n} rows from {DUMP}, N_DIG={nd}, wall={WALL}s, workers={WORKERS}", flush=True)

    per_row = []          # one dict per row: i, key, q, gslots, collision(bool), variants: name -> {values, parse, sig}
    task_cache = {}        # (row_idx, sig) -> (q, key, parse, gv, nvv) for the solver pool, de-duplicated
    n_collision_rows = 0
    n_collision_changed = {v: 0 for v in VARIANTS if v != "baseline"}

    for ri, rec in enumerate(recs):
        i, text, key = rec["i"], rec["text"], int(rec["key"])
        row = {k: np.asarray(rec[k], dtype=np.float64) for k in ("pres", "ftype", "op", "dig", "args", "res") + (("dup",) if "dup" in rec else ())}
        q = int(np.asarray(rec["q"]).argmax())
        candidates = legal_values(text, nd)   # SAME enumeration the mask uses -- the candidate set is text-derived, shared by every given
        base_row = mask_nonrel(row, text)     # chain_acc's ordinary mask on every non-rel slot (mod/pct/fdiv/macro untouched further)
        gslots = given_slots(row)
        G, M = len(gslots), len(candidates)
        if G:
            A = np.stack([affinity_row(row["dig"][j], candidates, nd) for j in gslots])
        else:
            A = np.zeros((0, M))
        baseline_vals = [candidates[int(np.argmax(A[k]))] for k in range(G)] if G and M else [None] * G
        collision = G >= 2 and len(set(v for v in baseline_vals if v is not None)) < len([v for v in baseline_vals if v is not None])
        n_collision_rows += int(collision)

        rrec = dict(i=i, key=key, q=q, text=text, gslots=gslots, candidates=candidates, baseline_vals=baseline_vals, collision=collision, variants={})

        def register(name, values, base=base_row):
            vrow = apply_given_assignment(base, gslots, values, nd) if G else base
            parse = _decode_slots(vrow)
            sig = parse_sig(parse)
            gv, nvv = build_gv_nvv(parse, q)
            task_id = (ri, sig)
            if task_id not in task_cache:
                task_cache[task_id] = (q, key, parse, gv, nvv) if (key is not None and parse) else None
            rrec["variants"][name] = dict(values=values, sig=sig, task_id=task_id, parse=parse)

        register("baseline", baseline_vals)

        for T in TS:
            if G and M:
                P = sinkhorn_balance(A, T)
                vals = [candidates[int(np.argmax(P[k]))] for k in range(G)]
            else:
                vals = list(baseline_vals)
            register(f"sinkhorn_T{T}", vals)
            if collision and vals != baseline_vals:
                n_collision_changed[f"sinkhorn_T{T}"] += 1

        if G and M:
            row_ind, col_ind = linear_sum_assignment(-A)
            hmap = dict(zip(row_ind.tolist(), col_ind.tolist()))
            hvals = [candidates[hmap[k]] if k in hmap else baseline_vals[k] for k in range(G)]
        else:
            hvals = list(baseline_vals)
        register("hungarian", hvals)
        if collision and hvals != baseline_vals:
            n_collision_changed["hungarian"] += 1

        per_row.append(rrec)
        if (ri + 1) % 50 == 0 or ri + 1 == n:
            print(f"[sinkhorn-claim] decoded {ri + 1}/{n} rows ({time.time() - t0:.1f}s); "
                  f"{len(task_cache)} unique (row, parse) solves queued", flush=True)

    # ------------------------------------------------------------------------------------------------
    # solve every unique (row, parse) once, across all rows and variants, hang-proof spawn pool
    # ------------------------------------------------------------------------------------------------
    tasks = [(tid, *args) for tid, args in task_cache.items() if args is not None]
    t_decode = time.time() - t0
    print(f"[sinkhorn-claim] solving {len(tasks)} unique parses ({n * len(VARIANTS)} row-variant cells) "
          f"with {WORKERS} workers, wall={WALL}s/attempt ...", flush=True)
    import multiprocessing as mp
    out = {}
    t1 = time.time()
    with mp.get_context("spawn").Pool(WORKERS) as pool:
        it = pool.imap_unordered(_solve_task, tasks, chunksize=1)
        try:
            for k in range(len(tasks)):
                tid, st, val = it.next(timeout=WALL * 3 + 30)
                out[tid] = (st, val)
                if (k + 1) % 50 == 0 or k + 1 == len(tasks):
                    print(f"[sinkhorn-claim] solved {k + 1}/{len(tasks)} ({time.time() - t1:.1f}s)", flush=True)
        except mp.TimeoutError:
            print(f"[sinkhorn-claim] {len(tasks) - len(out)} parses never returned -- counted as refused", flush=True)

    def label_of(task_id, key):
        if task_id not in task_cache or task_cache[task_id] is None:
            return "refused", None
        st, val = out.get(task_id, ("hung", None))
        if st != "solved":
            return "refused", None
        return ("correct" if int(val) == int(key) else "wrong"), val

    for rrec in per_row:
        for name, v in rrec["variants"].items():
            lab, val = label_of(v["task_id"], rrec["key"])
            v["label"] = lab; v["value"] = val

    # ------------------------------------------------------------------------------------------------
    # tallies
    # ------------------------------------------------------------------------------------------------
    counts = {name: {"correct": 0, "wrong": 0, "refused": 0} for name in VARIANTS}
    for rrec in per_row:
        for name in VARIANTS:
            counts[name][rrec["variants"][name]["label"]] += 1

    baseline_correct = counts["baseline"]["correct"]

    # slot-level given-digit accuracy against the fixture's gold factors[] (positional law: factors[j] is slot j)
    digit_acc = {name: [0, 0] for name in VARIANTS}   # [right, total]
    if gold_rows is not None:
        for rrec in per_row:
            gold_facs = gold_rows[rrec["i"]].get("factors") or []
            for name in VARIANTS:
                vals = rrec["variants"][name]["values"]
                for j, v in zip(rrec["gslots"], vals):
                    if j < len(gold_facs) and gold_facs[j].get("ftype") == "given" and v is not None:
                        digit_acc[name][1] += 1
                        if int(v) == int(gold_facs[j].get("value")):
                            digit_acc[name][0] += 1

    # per-row transitions vs baseline
    transitions = {name: {} for name in VARIANTS if name != "baseline"}
    for rrec in per_row:
        bl = rrec["variants"]["baseline"]["label"]
        for name in VARIANTS:
            if name == "baseline":
                continue
            vl = rrec["variants"][name]["label"]
            key = f"{bl}->{vl}"
            transitions[name][key] = transitions[name].get(key, 0) + 1

    # ------------------------------------------------------------------------------------------------
    # report
    # ------------------------------------------------------------------------------------------------
    lines = []
    def P(s=""):
        print(s); lines.append(s)

    P("=" * 100)
    P(f"THE SINKHORN CLAIM READ -- {DUMP} ({n} rows)")
    P("=" * 100)
    P("")
    P(f"SELF-GATE: baseline (mask alone, no reassignment) CORRECT = {baseline_correct}/{n} "
      f"(expected {SELF_GATE_EXPECT}/{n}) -- {'PASS' if baseline_correct == SELF_GATE_EXPECT else 'FAIL'}")
    P("")
    P(f"{'variant':16s} {'correct':>8s} {'refused':>8s} {'wrong':>8s}")
    for name in VARIANTS:
        c = counts[name]
        P(f"{name:16s} {c['correct']:8d} {c['refused']:8d} {c['wrong']:8d}")
    P("")
    P(f"COLLISIONS: {n_collision_rows}/{n} rows had >=2 given slots claim the same numeral under the baseline mask.")
    for name in VARIANTS:
        if name == "baseline":
            continue
        P(f"  of those, {name} changed the given-value assignment on {n_collision_changed[name]}/{n_collision_rows} rows.")
    P("")
    P("PER-ROW TRANSITIONS vs baseline (label->label counts; only cells that moved are listed):")
    for name in VARIANTS:
        if name == "baseline":
            continue
        P(f"  {name}:")
        moved = {k: v for k, v in transitions[name].items() if k.split("->")[0] != k.split("->")[1]}
        if not moved:
            P("    (no row changed label)")
        for k, v in sorted(moved.items(), key=lambda kv: -kv[1]):
            P(f"    {k}: {v}")
    P("")
    if gold_rows is not None:
        P("SLOT-LEVEL GIVEN-DIGIT ACCURACY vs the fixture's gold factors[] (positional law: factors[j] == slot j):")
        for name in VARIANTS:
            right, total = digit_acc[name]
            frac = right / total if total else float("nan")
            P(f"  {name:16s} {right:4d}/{total:<4d} = {frac:.3f}")
    else:
        P(f"(no gold-factor fixture found at {ROWS_PATH} -- slot-level digit accuracy skipped)")
    P("")
    P(f"[timing] decode {t_decode:.1f}s, {len(tasks)} unique parses solved in {time.time() - t1:.1f}s, "
      f"total {time.time() - t0:.1f}s")

    with open(OUT_TXT, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[sinkhorn-claim] wrote {OUT_TXT}")


if __name__ == "__main__":
    main()
