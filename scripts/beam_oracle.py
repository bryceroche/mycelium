"""beam_oracle.py — THE BEAM-SEARCH ORACLE BOUND (2026-09-24, zero-GPU read).

QUESTION: chain_acc.py's headline masked read takes the TOP-1 decoded parse per
row. If a small symbolic search over the head's OWN near-ties (not fresh model
compute) is allowed to propose a handful of alternate parses per row, how many
more of the 311 wild rows could a perfect judge push from wrong/refused to
correct? This is an ORACLE bound (perfect judge), not a claim about any judge
that exists — it bounds what a picker built on top of this decode could ever
buy.

======================================================================
HOW _decode_slots TURNS row["args"] INTO ARGUMENTS (read of the source,
scripts/phase1_algebra_head.py) — this governs every branch this script makes.
======================================================================

decode() (phase1_algebra_head.py:6242) is the per-slot fact builder. For a
RELATION slot (ftype[j].argmax() == 0, "rel"; op = add/mul from op[j]):

    if "dup" in o_np and o_np["dup"][j] > 0:                 # ALG_DUP's arg-
        a0 = argmax(o_np["dargs"][j] if "dargs" in o_np      # multiplicity bit
                     else o_np["args"][j])                    # (x+x, 2*x forms)
        args = [a0, a0]
    else:
        args = sorted(top-2 indices of o_np["args"][j] by value)   # a SET —
                                                                     # add/mul
                                                                     # commute,
                                                                     # and the
                                                                     # final
                                                                     # pair is
                                                                     # re-sorted
                                                                     # by VAR
                                                                     # INDEX,
                                                                     # not by
                                                                     # which
                                                                     # logit
                                                                     # was
                                                                     # larger.

Under the family env used for this dump, "dargs" is never a KEY chain_acc
banks (KEYS = pres/ftype/op/dig/args/res(/dup)), so the dup branch also reads
argmax(args[j]) — confirmed against the pickle's field set.

_decode_slots (phase1_algebra_head.py:464) calls decode() once per PRESENT
slot j IN ISOLATION: every other slot's "pres" is forced to -1 first, so a
slot's own fields are the only signal that reaches its own fact and nothing
about slot j can leak into slot j' != j's decode. This means a branch that
only touches slot j's args[j] leaves every other slot's fact byte-identical
— exactly the isolation this script's branching wants.

THE ORACLE'S DECISION UNIT: because the two-argument case is a SET (order
doesn't survive to the final fact), swapping WHICH of the two top values is
"first" vs "second" changes nothing decodable — the only decision that can
change the decoded fact is WHICH TWO variables make the top-2. Given args[j]
sorted descending as v[0] >= v[1] >= v[2] >= ...:

  - non-dup slot: v[0]'s index is the free/undisputed member of the pair
    (nothing in decode() ever second-guesses it in isolation); the live
    question is whether the SECOND member is v[1]'s index (today's choice)
    or v[2]'s index (the runner-up). MARGIN = v[1] - v[2]. This is exactly
    the quantity _slot_margins() (phase1_algebra_head.py:418, "the weaker of
    the top-2 pointers") already banks for the wheel's nogood road — reused
    here verbatim as the branching margin, not reimplemented.
  - dup slot: the single argument is argmax(args[j]) = v[0]'s index; the
    live question is whether it should instead be v[1]'s index.
    MARGIN = v[0] - v[1].

A branch is realized by SWAPPING THE TWO VALUES at the decision's two
candidate positions in a private copy of args[j] (every other slot's row is
untouched, and slot j's res/op/ftype/dig are untouched too) so that decode()'s
own argsort/argmax — called through the REAL, UNMODIFIED _decode_slots import
— naturally produces the forced pick. No decode logic is reimplemented.

Decisions are scoped to RELATION slots only (ftype argmax == 0), per the task;
given/mod/pct/fdiv/sel/macro slots are read as the top-1 decoder already
decodes them (their arguments are single argmax picks over "args" or digit
fields the numeral mask already governs for givens).

======================================================================
BRANCHING / SOLVING
======================================================================
Per row: rank present relation slots' decisions by margin ascending. For
k in 0..K_MAX(=4), the "k lowest-margin decisions" are decisions_sorted[:k];
enumerate all 2**k combinations of {default, swapped} over exactly those k
decisions (higher-margin decisions always stay at their default choice).
Bit i of a branch's bitmask == 1 means decisions_sorted[i] is swapped. Every
row is expanded ONCE at k=k_max=min(4, n_decisions) (2**k_max <= 16
bitmasks); a branch's "k_needed" = bitmask.bit_length() (0 for the all-
default/top-1 branch) is the smallest k whose window contains it — so the
k=1..4 reports below are read off ONE decode+solve pass per row, not four.

Distinct parses (by the wheel's own memo signature: JSON of each fact minus
"_slot", sorted) are solved ONCE; duplicate bitmasks that decode to the same
parse point at the same solved result. Solving reuses chain_acc.py's own
_solve_task semantics verbatim (same domain-ladder rule, same 5000-decision
budget, same 3s wall) via a hang-proof spawn pool (imap_unordered, per-result
timeout, 4 workers) — this script imports chain_acc's/admit_annotation's/
alternator_bridge's functions, never reimplements them.

Self-gate: the k=0 (top-1, all-default) branch's correct count over all 311
rows must equal chain_acc's masked read (14) — this is checked before any
oracle number is trusted.

Inputs (zero-GPU; DEV=CPU; run under the exact family env — see NEXT_SESSION /
the task brief — so N_DIG/L_FAC/K_VARS match the dump):
  BO_DUMP   raw per-row head dump, default .cache/rawslots_wild_PMS8_241.pkl
  BO_KMAX   max branching depth, default 4
  BO_WALL   solver wall per attempt (s), default 3 (chain_acc's default)
  BO_SELFGATE  expected k=0 correct count, default 14
Outputs: .cache/beam_oracle_PMS8_241.txt (report) + .cache/beam_oracle_PMS8_241.pkl
(per-row-per-branch records for an offline picker study).
"""
import os
import sys
import json
import time
import pickle

sys.path.insert(0, "."); sys.path.insert(0, "scripts")
import numpy as np

DUMP = os.environ.get("BO_DUMP", ".cache/rawslots_wild_PMS8_241.pkl")
OUT_TXT = os.environ.get("BO_OUT", ".cache/beam_oracle_" + os.path.basename(DUMP).replace("rawslots_wild_", "").replace(".pkl", "") + ".txt")
OUT_PKL = OUT_TXT.replace(".txt", ".pkl")
K_MAX = int(os.environ.get("BO_KMAX", "4"))
WALL = float(os.environ.get("BO_WALL", "3"))
SELF_GATE_EXPECT = int(os.environ.get("BO_SELFGATE", "14"))
WORKERS = int(os.environ.get("BO_WORKERS", "4"))


def _solve_task(t):
    """chain_acc._solve_task, verbatim semantics: (task_id, q, key, parse, gv, nvv) -> (task_id, status, value)."""
    import os as _os, sys as _sys
    _sys.path.insert(0, "."); _sys.path.insert(0, "scripts")
    from admit_annotation import solve_walled
    from alternator_bridge import problem_from_algebra3
    task_id, q, key, parse, gv, nvv = t
    wall = float(_os.environ.get("BO_WALL", "3"))
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


def mask_row(row, text):
    """chain_acc's CA_MASK=1 numeral mask, applied to non-relation slots' digits only
    (relation slots carry op/args, never dig — the mask never touches args)."""
    from mycelium.rulebook import legal_digit_logits
    row = dict(row); row["dig"] = row["dig"].copy()
    for j in range(row["ftype"].shape[0]):
        if int(row["ftype"][j].argmax()) == 0:
            continue
        fake = legal_digit_logits(row["dig"][j], text)
        if fake is not None:
            row["dig"][j] = fake
    return row


def slot_decisions(row):
    """One branching decision per present RELATION slot (ftype argmax == 0), read from
    args[j] exactly as decode() reads it (see module docstring). Returns a list of dicts:
    slot, kind ("dup"|"arg2"), idx_a (today's position), idx_b (runner-up position), margin."""
    decs = []
    pres, ftype, args = row["pres"], row["ftype"], row["args"]
    dup = row.get("dup")
    for j in range(pres.shape[0]):
        if pres[j] <= 0 or int(ftype[j].argmax()) != 0:
            continue
        v = args[j]; order = np.argsort(-v)
        is_dup = dup is not None and dup[j] > 0
        if is_dup:
            a, b = int(order[0]), int(order[1])
        else:
            a, b = int(order[1]), int(order[2])
        margin = float(v[a] - v[b])
        decs.append(dict(slot=int(j), kind=("dup" if is_dup else "arg2"), idx_a=a, idx_b=b, margin=margin))
    return decs


def apply_flips(row, decisions, flip_slots):
    """row with args[j] value-swapped at (idx_a, idx_b) for every decision whose slot is
    in flip_slots — every other field and every other slot is untouched."""
    if not flip_slots:
        return row
    row = dict(row); row["args"] = row["args"].copy()
    for d in decisions:
        if d["slot"] in flip_slots:
            j, a, b = d["slot"], d["idx_a"], d["idx_b"]
            row["args"][j][a], row["args"][j][b] = row["args"][j][b], row["args"][j][a]
    return row


def parse_sig(parse):
    """the wheel's own memo signature (_wheel_memo_key), reused verbatim: JSON of each
    fact minus "_slot", sorted — an order-free identity for a decoded graph."""
    return tuple(sorted(json.dumps({k: v for k, v in f.items() if k != "_slot"}, sort_keys=True, default=str) for f in parse))


def build_gv_nvv(parse, q):
    """chain_acc's exact (gv, nvv) construction from a decoded parse."""
    used = [f.get("var") for f in parse if f["ftype"] == "given"] + \
           [a for f in parse if f["ftype"] == "rel" for a in list(f["args"]) + [f["result"]]]
    nvv = max([q + 1] + [v + 1 for v in used if v is not None])
    gv = {f["var"]: f["value"] for f in parse if f["ftype"] == "given"}
    return gv, nvv


def main():
    from phase1_algebra_head import _decode_slots, N_DIG
    t0 = time.time()
    recs = pickle.load(open(DUMP, "rb"))
    _limit = int(os.environ.get("BO_LIMIT", "0"))
    if _limit:
        recs = recs[:_limit]
    n = len(recs)
    assert N_DIG == np.asarray(recs[0]["dig"]).shape[1], (
        f"N_DIG={N_DIG} from the head but the dump's digits are {np.asarray(recs[0]['dig']).shape[1]} wide — "
        f"run under the FAMILY ENV (ALG_WIDE=1 sets N_DIG=7)")
    print(f"[beam-oracle] {n} rows from {DUMP}, K_MAX={K_MAX}, wall={WALL}s, workers={WORKERS}", flush=True)

    per_row = []          # one dict per row: i, key, decisions, branches (sig -> record)
    all_margins = []       # every decision's margin, every row (population "all decisions")
    tasks = []             # (task_id, q, key, parse, gv, nvv) for the solver pool
    n_branches_total = 0

    for ri, rec in enumerate(recs):
        i, text, key = rec["i"], rec["text"], rec["key"]
        row0 = {k: np.asarray(rec[k]) for k in ("pres", "ftype", "op", "dig", "args", "res") + (("dup",) if "dup" in rec else ())}
        q = int(np.asarray(rec["q"]).argmax())
        masked = mask_row(row0, text)
        decisions = slot_decisions(masked)
        decisions_sorted = sorted(decisions, key=lambda d: d["margin"])
        all_margins.extend(d["margin"] for d in decisions_sorted)
        k_max_row = min(K_MAX, len(decisions_sorted))
        top_decs = decisions_sorted[:k_max_row]

        branches = {}   # sig -> {"parse":..., "k_needed":int, "flip_lists":[[dec,...],...], "task_id":..}
        for bitmask in range(1 << k_max_row):
            flip_slots = {top_decs[b]["slot"] for b in range(k_max_row) if bitmask & (1 << b)}
            branch_row = apply_flips(masked, top_decs, flip_slots)
            parse = _decode_slots(branch_row)
            sig = parse_sig(parse)
            k_needed = bitmask.bit_length()
            flips = [top_decs[b] for b in range(k_max_row) if bitmask & (1 << b)]
            if sig not in branches:
                branches[sig] = dict(parse=parse, k_needed=k_needed, flip_lists=[flips], bitmasks=[bitmask])
                gv, nvv = build_gv_nvv(parse, q)
                task_id = (ri, sig)
                if key is not None and parse:
                    tasks.append((task_id, q, key, parse, gv, nvv))
                    branches[sig]["task_id"] = task_id
                else:
                    branches[sig]["task_id"] = None
                    branches[sig]["status"] = "refused"; branches[sig]["value"] = None
            else:
                b = branches[sig]
                b["k_needed"] = min(b["k_needed"], k_needed)
                b["flip_lists"].append(flips); b["bitmasks"].append(bitmask)
        n_branches_total += len(branches)
        per_row.append(dict(i=i, key=key, q=q, text=text, decisions=decisions_sorted, k_max_row=k_max_row, branches=branches))
        if (ri + 1) % 50 == 0 or ri + 1 == n:
            print(f"[beam-oracle] decoded {ri+1}/{n} rows, {len(tasks)} solve tasks so far ({n_branches_total} distinct branches)", flush=True)

    print(f"[beam-oracle] {len(tasks)} distinct (row,parse) solves queued (of up to {n * (1 << K_MAX)} possible) — solving...", flush=True)
    import multiprocessing as mp
    solved = {}
    with mp.get_context("spawn").Pool(WORKERS) as pool:
        it = pool.imap_unordered(_solve_task, tasks, chunksize=1)
        n_done = 0
        try:
            for _ in range(len(tasks)):
                task_id, st, val = it.next(timeout=WALL * 3 + 30)
                solved[task_id] = (st, val); n_done += 1
                if n_done % 200 == 0 or n_done == len(tasks):
                    print(f"[beam-oracle] solved {n_done}/{len(tasks)} ({time.time()-t0:.0f}s elapsed)", flush=True)
        except mp.TimeoutError:
            print(f"[beam-oracle] {len(tasks) - len(solved)} tasks never returned — counted as refused", flush=True)

    # attach solve results back onto branches
    for row in per_row:
        for sig, b in row["branches"].items():
            if b.get("task_id") is not None:
                st, val = solved.get(b["task_id"], ("hung", None))
                b["status"] = st; b["value"] = val
            b["correct"] = (b.get("status") == "solved" and b.get("value") == row["key"])
            b["n_facts"] = len(b["parse"])

    # ---------------- self-gate ----------------
    def branches_at_k(row, k):
        kk = min(k, row["k_max_row"])
        return [b for b in row["branches"].values() if b["k_needed"] <= kk]

    top1_correct = top1_wrong = top1_refused = 0
    for row in per_row:
        top1 = [b for b in row["branches"].values() if b["k_needed"] == 0][0]
        if top1["status"] == "solved" and top1["correct"]:
            top1_correct += 1
        elif top1["status"] == "solved":
            top1_wrong += 1
        else:
            top1_refused += 1
        row["top1_status"] = "correct" if (top1["status"] == "solved" and top1["correct"]) else ("wrong" if top1["status"] == "solved" else "refused")

    lines = []
    def P(s=""):
        print(s); lines.append(s)

    P("=" * 78); P("THE BEAM-SEARCH ORACLE BOUND"); P("=" * 78)
    P(f"dump: {DUMP} ({n} rows) | K_MAX={K_MAX} | wall={WALL}s | {len(tasks)} distinct solves ({time.time()-t0:.0f}s total)")
    P("")
    P(f"SELF-GATE (k=0, top-1, all-default branch): CORRECT {top1_correct} | wrong {top1_wrong} | refused {top1_refused}")
    gate_ok = (top1_correct == SELF_GATE_EXPECT)
    P(f"  expected CORRECT {SELF_GATE_EXPECT} (chain_acc's masked read) -> {'PASS' if gate_ok else 'MISMATCH — investigate before trusting anything below'}")
    P("")

    right_flip_margins = []
    for k in range(0, K_MAX + 1):
        any_right = recoverable = free_wins = 0
        choice_sizes = []
        for row in per_row:
            bs = branches_at_k(row, k)
            right = [b for b in bs if b["correct"]]
            solved_bs = [b for b in bs if b["status"] == "solved"]
            if right:
                any_right += 1
            if row["top1_status"] != "correct" and right:
                recoverable += 1
                choice_sizes.append(len(solved_bs))
                if len(solved_bs) == 1:
                    free_wins += 1
                if k == K_MAX:   # margin census done once, at the full window
                    for b in right:
                        for fl in b["flip_lists"]:
                            for d in fl:
                                right_flip_margins.append(d["margin"])
        mean_cs = float(np.mean(choice_sizes)) if choice_sizes else float("nan")
        med_cs = float(np.median(choice_sizes)) if choice_sizes else float("nan")
        P(f"k={k}: ORACLE BOUND {any_right}/{n} correct-somewhere | recoverable (top1 wrong/refused, oracle right) {recoverable} "
           f"| choice-set size mean/median {mean_cs:.2f}/{med_cs:.1f} | free wins (only one branch solves) {free_wins}")

    P("")
    am = np.asarray(all_margins, float)
    rm = np.asarray(right_flip_margins, float)
    def stats(x):
        if len(x) == 0:
            return "n=0"
        return f"n={len(x)} mean={x.mean():.3f} median={np.median(x):.3f} p25={np.percentile(x,25):.3f} p75={np.percentile(x,75):.3f} min={x.min():.3f} max={x.max():.3f}"
    P("MARGIN CENSUS (args-decision margins = v[weaker]-v[runner-up] or v[top1]-v[top2] for dup slots):")
    P(f"  all decisions, every row:            {stats(am)}")
    P(f"  decisions flipped in a RIGHT branch (recoverable rows, k=4 window): {stats(rm)}")
    P("")
    P("READING: the oracle bound is a perfect-judge ceiling on THIS decode + THESE branches, not a claim any picker")
    P("reaches it; free wins are rows where the branch set alone disambiguates (no judge needed); the margin census")
    P("checks whether the flips that fix a row are concentrated at low margins (validating the ranking) or spread out.")

    with open(OUT_TXT, "w") as f:
        f.write("\n".join(lines) + "\n")

    # per-branch records for an offline picker study
    records = []
    for row in per_row:
        for sig, b in row["branches"].items():
            records.append(dict(
                row=row["i"], key=row["key"], top1_status=row["top1_status"],
                parse=b["parse"], flips=b["flip_lists"][0], k_needed=b["k_needed"],
                status=b.get("status"), value=b.get("value"), correct=b["correct"],
                n_facts=b["n_facts"],
            ))
    pickle.dump(records, open(OUT_PKL, "wb"))
    print(f"[beam-oracle] wrote {OUT_TXT} + {OUT_PKL} ({len(records)} branch records, {time.time()-t0:.0f}s total)", flush=True)


if __name__ == "__main__":
    main()
