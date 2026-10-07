"""scripts/courtroom.py -- THE COURTROOM (2026-10-06, zero-GPU, delegate; word given
docs/phase1_skeleton_spec.md 2026-10-06 13:17).

Bryce's frame (the ledger's 13:13 registration): "a competition between opposing teams
determines which story the jury selects." The furniture already in the room, reused as
libraries, NEVER edited:
  - THE LAWYERS = the first look's candidates: scripts/combined_oracle.py's three-axis
    (args x type/op x values) candidate lattice, already materialized per row by
    scripts/picker/gen_candidates.py's generate() -- this script imports that function and,
    by default, REUSES the already-banked .cache/picker/cand_diet.pkl /
    cand_wild_<TAG>.pkl (same generator, same K_ARGS<=3/M_TYPEOP<=2/VAL_MAX<=4/CAND_MAX<=64
    settings as the panel picker's own run) rather than re-solving ~24k candidates for
    ~30 CPU-minutes; a --cand override or a fresh dump name triggers regeneration.
  - THE BAILIFF = the June solver's own verdict already attached to every branch by
    gen_candidates (status=="solved" is "consistent"; unique is recorded, never forced).
    The bailiff's verdict is FINAL: a branch the bailiff did not mark "solved" can never be
    the WINNER (ledger: "a rejected story can never win"); if top-1 itself is not solved and
    no OTHER branch is solved either, the row keeps top-1 untouched (refused downstream, as
    today -- no new refusal mechanism invented here).
  - EXHAUSTION: no first-success stop (the 2026-10-06 07:09 MCTS finding this script exists
    to NOT repeat) -- every bailiff survivor stands and enters cross-examination.
  - CROSS-EXAMINATION: pairwise, over the top <=8 survivors by fewest edits from top-1 (the
    combined_oracle "rank" field: args-bits + typeop-bits + value-variant-priority, already
    on every branch); three jurors vote on every pair (see THE THREE JURORS below); a story
    wins a pair iff a majority (>=2 of 3) of jurors favor it; a Copeland tally (wins minus
    losses over the pool, ties scoring 0) is each story's aggregate.
  - THE VERDICT WITH A MARGIN: the top-Copeland SURVIVOR replaces top-1 only if it clears tau
    over BOTH the runner-up survivor AND top-1 itself (top-1 participates in every pool as the
    incumbent even when it is not itself a survivor, so "beats top-1" is measured on the same
    Copeland scale as "beats the runner-up" -- one threshold, not two interpretations); a
    single-survivor pool has no runner-up, so that half of the margin is vacuously satisfied
    (reported as such, never silently assumed).

THE THREE JURORS (pairwise, A vs B; "evidence", never pass/fail per the Goodhart fence --
only their SIGN votes feed the majority, and only tau, tuned on the diet, is a free parameter):
  JUROR 1 (text certificates ON THE DIFFERENCE): nl_certifier.py / angles.nlc_certificates is
    ROW-LEVEL (every certificate is a mean over ALL givens or ALL relations in a parse, read
    straight off scripts/nl_certifier.py main()'s body, lines 100-131, and angles.py's verbatim
    port, lines 63-93) -- it has no native span/slot argument. THIS SCRIPT achieves a SPAN-
    RESTRICTED certificate for the subset of its six formulas that are PER-FACT means
    (value_present, no_double, cue_agree, derived_stated restricted_cert() below) by filtering
    to facts whose decode-time `_slot` id is in the discriminating-slot set (the positional law
    angles.diff_from_top1 / twin_flip_flag already rely on: a slot's fact never moves rows, only
    its content does) -- each restricted formula quotes the SAME source formula, just over a
    filtered fact list. The two remaining certificates (coverage, chain_reaches) are WHOLE-PARSE
    properties (a text-wide numeral-coverage fraction; a binary "does the derived chain reach
    the query variable") that do not restrict to a slot subset in any way that preserves their
    meaning -- for THOSE TWO, and whenever the differing slots carry no given/rel fact at all
    (a pct/mod/fdiv/sel/macro-only difference, which nlc_certificates' formulas do not cover),
    this script falls back to the ROW-LEVEL full-candidate certificate and takes its pairwise
    difference, exactly as the task brief's own stated escape permits ("if the certifier is
    row-level only, score the full candidate and take the pairwise difference -- state which").
    GRANULARITY ACHIEVED: SPAN-RESTRICTED when the differing slots include >=1 given or >=1 rel
    fact (the common case); ROW-LEVEL FALLBACK otherwise. Both counts are reported.
  JUROR 2 ("evidence ignored"): the unexplained-numeral count (numerals in the text claimed by
    NO given slot of the story), full-candidate, per the brief's own framing ("a story that
    leaves a numeral unexplained has ignored evidence") -- the PAIRWISE score is the plain
    difference between the two stories' counts (fewer wins).
  JUROR 3 (collisions): the count of given slots sharing a numeral with another given slot in
    the SAME story, full-candidate; pairwise difference, fewer wins.
  NEVER a fourth juror on the candidate's own log-likelihood -- the task brief and the 10-05
  12:48 panel-picker ablation both rule it out by name ("the model's own likelihood was a
  NET-NEGATIVE witness... the defendant testifying for itself").

Usage:
  env $PANEL_DECODE_ENV .venv/bin/python3 scripts/courtroom.py <rawslots.pkl> --tag <TAG> \\
      [--tau FLOAT] [--tune] [--k 8] [--cand PATH] [--fixture PATH] [--workers 8] [--limit N]
  (the family env, .cache/picker/panel_env.sh, is only STRICTLY needed on a cold-cache run that
  must call gen_candidates.generate() itself -- the default fast path reuses the already-banked
  .cache/picker/cand_{diet,wild_<TAG>}.pkl and needs no model-shaped env at all; it is sourced
  unconditionally in the chain script below for safety.)

--tune sweeps tau over the GIVEN rawslots.pkl (intended for the DIET fixture only -- the task's
own law: "Tune tau ONLY on the diet slice... wild is read ONCE at the end") and prints/writes the
sweep table; it does not itself pick a winner for a second file. --tau <chosen> then RUNS the
verdict on whichever rawslots.pkl is given (the diet, for the parity read, or the wild holdout,
for the one registered read) at that fixed tau.

Outputs: .cache/courtroom_<basename-without-rawslots_>.txt (+ .pkl per-row records).
"""
import os
import sys
import re
import json
import time
import pickle
import argparse
from collections import Counter

sys.path.insert(0, "."); sys.path.insert(0, "scripts"); sys.path.insert(0, "scripts/picker")
import numpy as np

import combined_oracle as CO     # BO/SK re-exports, typeop_decisions/value_variants/candidate_loglik (unused here)
import angles as A                # nlc_certificates, build_row_tables, twin_flip_flag, diff_from_top1-style helpers
import args_census as AC          # sentence_bounds/spans (reused by restricted_cert's span lookups, parity with angles)
import gen_candidates as GC       # scripts/picker/gen_candidates.py (the bailiff + exhaustion generator)

BO = CO.BO

NUM_RE = re.compile(r"(?<![\d.])\d{1,3}(?![\d])")


# ============================================================================================
# 0. CANDIDATE LOADING -- reuse the already-banked cand_*.pkl when it matches the dump; else
#    call gen_candidates.generate() (never editing it) to build one.
# ============================================================================================
def guess_tag(dump_path):
    b = os.path.basename(dump_path)
    m = re.search(r"rawslots_(?:wild|slicevalid2)_(.+)\.pkl$", b)
    return m.group(1) if m else "PMS8_241"


def guess_cand_path(dump_path, tag):
    b = os.path.basename(dump_path)
    if "wild" in b:
        p = f".cache/picker/cand_wild_{tag}.pkl"
    elif "slicevalid2" in b or "valid2" in b:
        p = ".cache/picker/cand_diet.pkl"
    else:
        p = f".cache/courtroom/cand_{tag}_{b}"
    return p


def guess_fixture_path(dump_path):
    b = os.path.basename(dump_path)
    if "wild" in b:
        return ".cache/wild_admitted_holdout_a.jsonl"
    return ".cache/form_pm35c_slice1024_valid2.jsonl"


def load_candidates(dump_path, cand_path, workers, limit, log):
    recs = pickle.load(open(dump_path, "rb"))
    if limit:
        recs = recs[:limit]
    n_dump = len(recs)
    reused = False
    if cand_path and os.path.exists(cand_path):
        cands = pickle.load(open(cand_path, "rb"))
        if limit:
            cands = cands[:limit]
        if len(cands) == n_dump and all(c["text"] == r["text"] for c, r in zip(cands, recs)):
            log(f"[courtroom] REUSED banked candidates: {cand_path} ({len(cands)} rows, parity-checked by text)")
            reused = True
        else:
            log(f"[courtroom] {cand_path} exists but does not match {dump_path} (len/text mismatch) -- regenerating")
    if not reused:
        os.makedirs(os.path.dirname(cand_path) or ".", exist_ok=True)
        log(f"[courtroom] generating candidates fresh ({dump_path} -> {cand_path}) via picker.gen_candidates ...")
        cands = GC.generate(dump_path=dump_path, out_pkl=cand_path, k_args=3, m_typeop=2, val_max=4,
                             cand_max=64, wall=3.0, workers=workers, run_uniq=True, limit=limit, log=log)
    return cands, reused


# ============================================================================================
# 1. DISCRIMINATING SLOTS (the positional law: a slot's fact never moves rows, only its content)
# ============================================================================================
def _val_of(f):
    return f.get("value", f.get("k", f.get("p", f.get("a"))))


def discriminating_slots(parse_a, parse_b):
    a_by = {f["_slot"]: f for f in parse_a}
    b_by = {f["_slot"]: f for f in parse_b}
    diff = set()
    for s in set(a_by) | set(b_by):
        fa, fb = a_by.get(s), b_by.get(s)
        if fa is None or fb is None:
            diff.add(s); continue
        if fa.get("ftype") != fb.get("ftype") or fa.get("op") != fb.get("op"):
            diff.add(s); continue
        if fa.get("ftype") == "rel" and fa.get("args") != fb.get("args"):
            diff.add(s); continue
        if _val_of(fa) != _val_of(fb):
            diff.add(s)
    return diff


# ============================================================================================
# 2. JUROR 1 -- the text certificate restricted to the discriminating slots (span-restricted
#    where the slots carry a given/rel fact; row-level pairwise difference otherwise).
# ============================================================================================
def restricted_cert(text, parse, diff_slots, asg=None):
    # NOTE: SAM.clause_of / SAM.build_intro_map index facts by their POSITION in `parse` (the
    # enumerate() index nl_certifier.py's own main() and angles.nlc_certificates both use), NOT
    # by the decode-time `_slot` id (0..23 in the raw head) -- a slot can be ABSENT from a given
    # candidate's decode entirely, so the parse list is always <= the slot count. Resolve each
    # differing slot's REL fact to its actual list position before calling clause_of.
    givens = [f for f in parse if f.get("ftype") == "given" and f.get("_slot") in diff_slots]
    rels = [(pos, f) for pos, f in enumerate(parse) if f.get("ftype") == "rel" and f.get("_slot") in diff_slots]
    if not givens and not rels:
        return None
    nums = [int(x) for x in NUM_RE.findall(text)]
    numset = set(nums)
    all_gvals = [int(f["value"]) for f in parse if f.get("ftype") == "given" and f.get("value") is not None]
    cnt = Counter(all_gvals)
    parts = []
    if givens:
        gvals = [int(f["value"]) for f in givens if f.get("value") is not None]
        if gvals:
            parts.append(float(np.mean([v in numset for v in gvals])))             # value_present, restricted
            parts.append(float(np.mean([1.0 if cnt[v] <= 1 else 0.0 for v in gvals])))  # no_double, restricted
    if rels:
        import stamp_arg_mentions as SAM
        bounds = AC.sentence_bounds(text); sspans = AC.sentence_spans(text, bounds)
        intro = SAM.build_intro_map(parse); memo = {}
        sol = list(asg) if asg is not None else []
        agree = []
        for j, f in rels:
            cl = SAM.clause_of(j, parse, text, bounds, sspans, sol, intro, memo)
            words = set(w.lower() for w in re.findall(r"[a-z']+", " ".join(text[a:b] for a, b in SAM.windows_of(cl, sspans)))) if cl is not None else set()
            cues = A.ADD_CUES if f.get("op") == "add" else A.MUL_CUES
            agree.append(1.0 if (words & cues) else 0.0)
        if agree:
            parts.append(float(np.mean(agree)))                                     # cue_agree, restricted
    return float(np.mean(parts)) if parts else None


def juror1_cert_diff(text, parse_a, asg_a, parse_b, asg_b, q, diff_slots):
    ra = restricted_cert(text, parse_a, diff_slots, asg_a)
    rb = restricted_cert(text, parse_b, diff_slots, asg_b)
    if ra is not None and rb is not None:
        return ra - rb, "span"
    ca = A.nlc_certificates(text, parse_a, asg_a, q)["cert"]
    cb = A.nlc_certificates(text, parse_b, asg_b, q)["cert"]
    return ca - cb, "row"


# ============================================================================================
# 3. JURORS 2 & 3 -- unexplained numerals / collisions, full-candidate, pairwise difference
# ============================================================================================
def unexplained_count(text, parse):
    nums = [int(x) for x in NUM_RE.findall(text)]
    gvals = set(int(f["value"]) for f in parse if f.get("ftype") == "given" and f.get("value") is not None)
    return sum(1 for n in nums if n not in gvals)


def collision_count(parse):
    gvals = [int(f["value"]) for f in parse if f.get("ftype") == "given" and f.get("value") is not None]
    return len(gvals) - len(set(gvals))


# ============================================================================================
# 4. THE PAIRWISE VOTE (majority of 3 signed jurors) + per-row Copeland tally
# ============================================================================================
def sign(x):
    return 0 if x == 0 else (1 if x > 0 else -1)


def pairwise_votes(text, cA, cB, q):
    """Returns (votes=[j1,j2,j3] each in {-1,0,1}, favoring A when positive, granularity_label)."""
    pa, pb = cA["parse"], cB["parse"]
    diff_slots = discriminating_slots(pa, pb)
    v1, gran = juror1_cert_diff(text, pa, cA.get("assignment"), pb, cB.get("assignment"), q, diff_slots)
    uA, uB = unexplained_count(text, pa), unexplained_count(text, pb)
    colA, colB = collision_count(pa), collision_count(pb)
    votes = [sign(v1), sign(uB - uA), sign(colB - colA)]
    return votes, gran, diff_slots


def decided_by(votes, idx):
    """True iff juror `idx` was NECESSARY for the pairwise majority direction (removing it would
    leave a tie or flip the sign) -- 'how often X decides the pair', stated precisely."""
    total = sum(votes)
    if total == 0:
        return False
    rest = total - votes[idx]
    return rest == 0 or sign(rest) != sign(total)


def cross_examine(text, pool_items, q, stats):
    """pool_items: list of (sig, branch). Returns {sig: copeland_score}, plus pairwise detail
    accumulated into `stats` (a dict of running counters, mutated in place)."""
    scores = {sig: 0 for sig, _ in pool_items}
    n = len(pool_items)
    for i in range(n):
        for j in range(i + 1, n):
            sig_a, ca = pool_items[i]; sig_b, cb = pool_items[j]
            votes, gran, diff_slots = pairwise_votes(text, ca, cb, q)
            total = sum(votes)
            stats["n_pairs"] += 1
            stats["gran_" + gran] += 1
            if total > 0:
                scores[sig_a] += 1; scores[sig_b] -= 1
            elif total < 0:
                scores[sig_a] -= 1; scores[sig_b] += 1
            else:
                stats["n_ties"] += 1
            for k, name in enumerate(("cert", "unexplained", "collision")):
                if decided_by(votes, k):
                    stats[f"decided_{name}"] += 1
            if diff_slots and any(f.get("_slot") in diff_slots and f.get("ftype") == "rel"
                                   for f in ca["parse"] + cb["parse"]):
                stats["n_pairs_with_rel_diff"] += 1
                if stats.get("_tables") is not None:
                    twin_flag, n_checked = A.twin_flip_flag(stats["_tables"], ca["parse"], cb["parse"])
                    if n_checked:
                        stats["n_pairs_rel_diff_twin_checked"] += 1
                        if twin_flag:
                            stats["n_pairs_rel_diff_twin"] += 1
    return scores


# ============================================================================================
# 5. THE BAILIFF + THE VERDICT WITH A MARGIN, per row
# ============================================================================================
def run_row(row, tau, k_max, stats):
    branches = row["branches"]
    top1_sig = row.get("top1_sig")
    top1 = branches.get(top1_sig) if top1_sig is not None else None
    if top1 is None:   # (should not happen per gen_candidates' own self-gate note; fall back to any (0,0,0)-tagged branch)
        for sig, b in branches.items():
            if (0, 0, 0) in b.get("tags", ()):
                top1_sig, top1 = sig, b; break

    survivors = [(sig, b) for sig, b in branches.items() if b.get("status") == "solved"]
    stats["n_candidates_hist"].append(len(branches))

    if not survivors:
        stats["stop_bailiff_empty"] += 1
        return dict(i=row["i"], key=row["key"], top1=top1, winner=top1, winner_sig=top1_sig,
                    verdict="bailiff_empty", pool_size=0, copeland=None, margin_runnerup=None, margin_top1=None)

    survivors.sort(key=lambda sb: sb[1].get("rank", 0))
    pool = survivors[:k_max]
    survivor_sigs_ordered = [sig for sig, _ in pool]   # already rank-ordered; a LIST (deterministic
                                                         # iteration) -- never a set (python's per-process
                                                         # string hash randomization makes set iteration
                                                         # order, and therefore any tie-break derived from
                                                         # it, non-reproducible run to run)
    if top1_sig is not None and top1_sig not in survivor_sigs_ordered and top1 is not None:
        pool = pool + [(top1_sig, top1)]   # the incumbent always participates as the hurdle

    scores = cross_examine(row["text"], pool, row["q"], stats)

    rank_of = {sig: b.get("rank", 0) for sig, b in pool}
    surv_scores = sorted(((sig, scores[sig]) for sig in survivor_sigs_ordered),
                          key=lambda sb: (-sb[1], rank_of[sb[0]], sb[0]))   # score desc, then fewest
                                                                            # edits from top-1, then the
                                                                            # signature itself (total order
                                                                            # on tuples-of-str) -- fully
                                                                            # deterministic, no hash-order
                                                                            # dependency anywhere
    winner_sig, winner_score = surv_scores[0]
    runnerup_score = surv_scores[1][1] if len(surv_scores) > 1 else None
    top1_score = scores.get(top1_sig) if top1_sig is not None else None
    winner = branches[winner_sig]

    margin_runnerup = (winner_score - runnerup_score) if runnerup_score is not None else None
    margin_top1 = (winner_score - top1_score) if top1_score is not None else None

    ok_runnerup = (margin_runnerup is None) or (margin_runnerup >= tau)
    ok_top1 = (margin_top1 is None) or (winner_sig == top1_sig) or (margin_top1 >= tau)

    if winner_sig == top1_sig:
        stats["stop_already_top1"] += 1
        verdict = "already_top1"
        out_winner, out_sig = top1, top1_sig
    elif ok_runnerup and ok_top1:
        stats["stop_replaced"] += 1
        verdict = "replaced"
        out_winner, out_sig = winner, winner_sig
    else:
        stats["stop_margin_refused"] += 1
        verdict = "margin_refused"
        out_winner, out_sig = top1, top1_sig

    return dict(i=row["i"], key=row["key"], top1=top1, winner=out_winner, winner_sig=out_sig,
                verdict=verdict, pool_size=len(pool), copeland=winner_score,
                margin_runnerup=margin_runnerup, margin_top1=margin_top1)


# ============================================================================================
# 6. MAIN
# ============================================================================================
def _status_of(c):
    if c is None:
        return "refused"
    return "correct" if c.get("correct") else ("refused" if c.get("status") != "solved" else "wrong")


def run_courtroom(dump_path, tag, cand_path, fixture_path, tau, k_max, workers, limit, log):
    cands, reused = load_candidates(dump_path, cand_path, workers, limit, log)
    tables_by_row = None
    if fixture_path and os.path.exists(fixture_path):
        fixture = [json.loads(l) for l in open(fixture_path)]
    else:
        fixture = None

    top1_correct = sum(1 for row in cands if row.get("top1_sig") is not None
                        and row["branches"][row["top1_sig"]].get("correct"))
    log(f"[courtroom] {dump_path} ({len(cands)} rows) | top1_correct={top1_correct} | tau={tau} k_max={k_max}")

    stats = dict(n_pairs=0, n_ties=0, gran_span=0, gran_row=0, n_candidates_hist=[],
                 stop_bailiff_empty=0, stop_already_top1=0, stop_replaced=0, stop_margin_refused=0,
                 decided_cert=0, decided_unexplained=0, decided_collision=0,
                 n_pairs_with_rel_diff=0, n_pairs_rel_diff_twin_checked=0, n_pairs_rel_diff_twin=0)

    results = []
    t0 = time.time()
    for ri, row in enumerate(cands):
        tbl = None
        if fixture is not None and row["i"] < len(fixture) and "factors" in fixture[row["i"]]:
            try:
                tbl = A.build_row_tables(fixture[row["i"]])
            except Exception:
                tbl = None
        stats["_tables"] = tbl
        res = run_row(row, tau, k_max, stats)
        results.append(res)
        if (ri + 1) % 100 == 0 or ri + 1 == len(cands):
            log(f"[courtroom] cross-examined {ri+1}/{len(cands)} rows ({time.time()-t0:.0f}s)")
    stats.pop("_tables", None)

    verdict_correct = verdict_wrong = verdict_refused = 0
    confusion = {}; fixes = []; regressions = []
    for res in results:
        t1_status = _status_of(res["top1"])
        w_status = "correct" if res["winner"] is not None and res["winner"].get("correct") else \
                   ("refused" if res["winner"] is None or res["winner"].get("status") != "solved" else "wrong")
        if w_status == "correct": verdict_correct += 1
        elif w_status == "wrong": verdict_wrong += 1
        else: verdict_refused += 1
        confusion[(t1_status, w_status)] = confusion.get((t1_status, w_status), 0) + 1
        if t1_status != "correct" and w_status == "correct":
            fixes.append(res["i"])
        if t1_status == "correct" and w_status != "correct":
            regressions.append(res["i"])

    return dict(dump_path=dump_path, tag=tag, tau=tau, k_max=k_max, n=len(cands),
                top1_correct=top1_correct, verdict_correct=verdict_correct, verdict_wrong=verdict_wrong,
                verdict_refused=verdict_refused, confusion=confusion, fixes=fixes, regressions=regressions,
                stats=stats, results=results, reused_cache=reused)


def write_report(out, tag, tau_sweep_table=None):
    lines = []
    def P(s=""):
        print(s); lines.append(s)

    P("=" * 94); P(f"THE COURTROOM -- {tag}"); P("=" * 94)
    if tau_sweep_table is not None:
        P("-" * 94); P("DIET TAU SWEEP (tuned on the diet fixture ONLY, per the task's law; wild read once at the end)"); P("-" * 94)
        P(f"  {'tau':>5s} {'rows_correct':>13s} {'regressions':>12s} {'fixes':>7s}")
        for tau, rows_correct, regr, fx in tau_sweep_table:
            P(f"  {tau:5.1f} {rows_correct:13d} {len(regr):12d} {len(fx):7d}")
        P("")
    return lines, P


def report_run(res, P, label):
    s = res["stats"]
    hist = np.asarray(s["n_candidates_hist"], float)
    P("-" * 94); P(f"{label}: {res['dump_path']} ({res['n']} rows) | tau={res['tau']} k_max={res['k_max']} "
                   f"| candidates cache {'REUSED' if res['reused_cache'] else 'GENERATED'}"); P("-" * 94)
    P(f"  TOP-1 baseline:        {res['top1_correct']}/{res['n']}")
    P(f"  COURTROOM VERDICT:     correct {res['verdict_correct']} | wrong {res['verdict_wrong']} | refused {res['verdict_refused']}")
    P("")
    P("  STOP-REASON HISTOGRAM:")
    for k in ("stop_bailiff_empty", "stop_already_top1", "stop_replaced", "stop_margin_refused"):
        P(f"    {k:22s} {s[k]}")
    P("")
    P("  CONFUSION (top1_status -> verdict_status):")
    for (ts, vs), cnt in sorted(res["confusion"].items()):
        P(f"    {ts:8s} -> {vs:8s} : {cnt}")
    P("")
    P(f"  PRECISION vs top-1: FIXES={len(res['fixes'])} rows={res['fixes']}")
    P(f"                      REGRESSIONS={len(res['regressions'])} rows={res['regressions']}")
    P("")
    P("  CANDIDATES/ROW (solver-call proxy; each = 1 solve call + 1 uniqueness call if solved):")
    if len(hist):
        P(f"    n={len(hist)} mean={hist.mean():.1f} median={np.median(hist):.1f} min={hist.min():.0f} "
          f"max={hist.max():.0f} at-cap(=64)={int((hist>=64).sum())}")
    P("")
    P("  CROSS-EXAMINATION (pairwise evidence statistics):")
    P(f"    pairs examined: {s['n_pairs']}  (ties: {s['n_ties']})")
    tot_gran = max(1, s["gran_span"] + s["gran_row"])
    P(f"    JUROR 1 granularity: span-restricted {s['gran_span']} ({s['gran_span']/tot_gran:.3f})  "
      f"row-level fallback {s['gran_row']} ({s['gran_row']/tot_gran:.3f})")
    P(f"    decisive pairs where juror X was NECESSARY for the majority (ties excluded, n={s['n_pairs']-s['n_ties']}):")
    denom = max(1, s["n_pairs"] - s["n_ties"])
    P(f"      cert (text certificates):  {s['decided_cert']}  ({s['decided_cert']/denom:.3f})")
    P(f"      unexplained numerals:      {s['decided_unexplained']}  ({s['decided_unexplained']/denom:.3f})")
    P(f"      collisions:                {s['decided_collision']}  ({s['decided_collision']/denom:.3f})")
    if s["n_pairs_with_rel_diff"]:
        tw_denom = max(1, s["n_pairs_rel_diff_twin_checked"])
        P(f"    pairs with a differing RELATION slot: {s['n_pairs_with_rel_diff']}  "
          f"(twin-checkable: {s['n_pairs_rel_diff_twin_checked']}, of which SAME-NOUN TWIN: "
          f"{s['n_pairs_rel_diff_twin']}, {s['n_pairs_rel_diff_twin']/tw_denom:.3f})")
    P("")


def main():
    ap = argparse.ArgumentParser(description="THE COURTROOM: bailiff + exhaustion + cross-examination + "
                                              "verdict-with-a-margin over the combined oracle's candidates.")
    ap.add_argument("rawslots", help=".cache/rawslots_{wild_<TAG>|slicevalid2_<TAG>}.pkl")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--tau", type=float, default=None)
    ap.add_argument("--tune", action="store_true", help="sweep tau on THIS dump (use the diet dump) and report")
    ap.add_argument("--allow-wild-tune", action="store_true",
                     help="override the wild-tune refusal below (never use this for a real read)")
    ap.add_argument("--k", type=int, default=8, dest="k_max")
    ap.add_argument("--cand", default=None)
    ap.add_argument("--fixture", default=None)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--tau-grid", default="0,1,2,3,4,5,6,7,8")
    ap.add_argument("--tune-rule-regr-max", type=int, default=5,
                     help="the stated rule: pick the smallest tau maximizing diet rows_correct "
                          "subject to diet regressions <= this (default 5, per the task's own fallback phrasing)")
    args = ap.parse_args()

    tag = args.tag or guess_tag(args.rawslots)
    cand_path = args.cand or guess_cand_path(args.rawslots, tag)
    fixture_path = args.fixture or guess_fixture_path(args.rawslots)
    b = os.path.basename(args.rawslots)
    if "wild" in b:
        out_base = f".cache/courtroom_{tag}"            # the primary deliverable's exact pinned name
    elif "slicevalid2" in b or "valid2" in b:
        out_base = f".cache/courtroom_diet_{tag}"        # the parity read -- never overwrites the wild file
    else:
        out_base = ".cache/courtroom_" + re.sub(r"^rawslots_", "", b).replace(".pkl", "")
    out_txt = out_base + ".txt"; out_pkl = out_base + ".pkl"

    lines = []
    def log(s=""):
        print(s, flush=True); lines.append(s)

    if not args.tune and args.tau is None:
        log("[courtroom] neither --tau nor --tune given -- nothing to do (the task pins tau tuning on the "
            "diet ONLY; pass --tune on the diet dump first, then --tau <chosen> here).")
        sys.exit(2)

    if args.tune and "wild" in b and not args.allow_wild_tune:
        # BUG AUDIT 2026-10-06 (item 1, key leakage): the "diet only" rule above was convention,
        # not code -- nothing stopped --tune from sweeping tau (which selects on verdict_correct,
        # itself gold-derived) against a wild dump. No real invocation ever did this (ledger-verified:
        # every --tune ran on slicevalid2), but refuse it at the code level rather than by discipline.
        log(f"[courtroom] REFUSED: --tune on a wild dump ({b}) would tune tau against wild's own gold "
            "verdicts -- the law pins tau-tuning to the diet slice only. Pass --allow-wild-tune to "
            "override (never for a real read).")
        sys.exit(2)

    tau_sweep_table = None
    chosen_tau = args.tau
    if args.tune:
        grid = [float(x) for x in args.tau_grid.split(",")]
        tau_sweep_table = []
        best = None
        for t in grid:
            res_t = run_courtroom(args.rawslots, tag, cand_path, fixture_path, t, args.k_max, args.workers, args.limit, log)
            tau_sweep_table.append((t, res_t["verdict_correct"], res_t["regressions"], res_t["fixes"]))
            log(f"[courtroom][tune] tau={t} rows_correct={res_t['verdict_correct']} regressions={len(res_t['regressions'])}")
            if len(res_t["regressions"]) <= args.tune_rule_regr_max:
                if best is None or res_t["verdict_correct"] > best[1]:
                    best = (t, res_t["verdict_correct"])
        if chosen_tau is None:
            if best is None:
                chosen_tau = max(grid)
                log(f"[courtroom][tune] NO tau on the grid kept diet regressions <= {args.tune_rule_regr_max} "
                    f"-- falling back to the largest grid tau ({chosen_tau}), the most conservative (fewest replacements)")
            else:
                chosen_tau = best[0]
                log(f"[courtroom][tune] RULE: smallest tau maximizing diet rows_correct subject to regressions <= "
                    f"{args.tune_rule_regr_max} -> tau={chosen_tau} (rows_correct={best[1]})")

    final = run_courtroom(args.rawslots, tag, cand_path, fixture_path, chosen_tau, args.k_max, args.workers, args.limit, log)

    rep_lines, P = write_report(out_txt, tag, tau_sweep_table)
    report_run(final, P, f"RUN ({os.path.basename(args.rawslots)})")

    P("=" * 94)
    P(f"LEDGER-READY SUMMARY: {os.path.basename(args.rawslots)} tau={chosen_tau} k_max={args.k_max} -> "
      f"rows={final['verdict_correct']}/{final['n']} vs top-1 {final['top1_correct']} "
      f"(fixes={len(final['fixes'])} regressions={len(final['regressions'])}); "
      f"candidates cache {'reused' if final['reused_cache'] else 'generated'} ({cand_path})")

    with open(out_txt, "w") as f:
        f.write("\n".join(rep_lines) + "\n")
    with open(out_pkl, "wb") as f:
        pickle.dump(dict(tag=tag, dump_path=args.rawslots, tau=chosen_tau, k_max=args.k_max,
                          tau_sweep_table=tau_sweep_table, final=final), f)
    print(f"[courtroom] wrote {out_txt} + {out_pkl}")


if __name__ == "__main__":
    main()
