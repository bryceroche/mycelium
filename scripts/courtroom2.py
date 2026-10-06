"""scripts/courtroom2.py -- THE CARICATURE JURY plugged into THE COURTROOM (2026-10-06, zero-GPU,
delegate; form 2 of docs/phase1_skeleton_spec.md 2026-10-06 13:35 "THE TRAINED JUROR").

scripts/courtroom.py has no juror hook to extend -- its three jurors are free functions baked
into pairwise_votes()/cross_examine(). Per the task's own fallback, this script therefore
IMPORTS courtroom.py whole (never edits it) and swaps ONLY the cross-examination step by
MONKEYPATCHING the module-level name `courtroom.pairwise_votes` before calling
courtroom.run_courtroom() -- courtroom.cross_examine() resolves `pairwise_votes` by a bare global
lookup against its own module's namespace at CALL time, so this is a real swap, not a shadow: every
other piece -- THE BAILIFF (solved-only survivors), EXHAUSTION (no first-success stop), THE
VERDICT WITH A MARGIN (Copeland + tau over the top<=k survivors union top-1), load_candidates,
write_report, report_run, guess_tag/guess_cand_path/guess_fixture_path -- runs EXACTLY as
courtroom.py wrote it, unmodified, imported.

THE SWAP: courtroom.pairwise_votes(text, cA, cB, q) -> (votes=[j1,j2,j3], gran, diff_slots) becomes
jury_features.score_pair()'s single TRAINED score, replicated into votes=[s,s,s] (sign of the
trained juror's decision_function) so courtroom.py's sum()/decided_by() machinery still works
unchanged -- the three "decided_cert/unexplained/collision" breakdown lines in courtroom's own
report are therefore NOT meaningful here (they all collapse to the one trained signal); this
script prints its own juror-specific lines instead.

Usage:
  DEV=CPU .venv/bin/python3 scripts/courtroom2.py .cache/rawslots_slicevalid2_PMS8_241.pkl --tune
  DEV=CPU .venv/bin/python3 scripts/courtroom2.py .cache/rawslots_wild_PMS8_241.pkl --tau <chosen>

Outputs (wild): .cache/jury_<TAG>.txt + .pkl (the task's pinned deliverable name).
Outputs (diet): .cache/jury_diet_<TAG>.{txt,pkl} (the tau-tune + parity read).
"""
import os
import sys
import time
import pickle
import argparse

sys.path.insert(0, "."); sys.path.insert(0, "scripts"); sys.path.insert(0, "scripts/picker")

os.environ.setdefault("DEV", "CPU")
assert os.environ["DEV"] == "CPU", "courtroom2: CPU-only, always -- the trunk embedding never touches the GPU"

import numpy as np

import courtroom as CT
import jury_features as JF
import jury_train as JT   # build_pairs() reused for the wild gold-known pairwise-accuracy measurement


# ================================================================================================
# THE SWAP
# ================================================================================================
def install_trained_juror(model, trunk_cache, stats_box):
    def trained_pairwise_votes(text, cA, cB, q):
        score, gran, diff_slots = JF.score_pair(text, cA, cB, q, trunk_cache, model)
        s = CT.sign(score)
        stats_box["n_scored"] += 1
        stats_box["score_sum"] += score
        return [s, s, s], gran, diff_slots
    CT.pairwise_votes = trained_pairwise_votes


# ================================================================================================
# WILD-PAIR MEASUREMENT (gold known): the SAME true-vs-wrong pair construction jury_train.py used
# on the diet, applied once to the wild dump's own candidates, scored by the trained juror (no
# retraining, no retuning -- this is a read, not a fit).
# ================================================================================================
def wild_pairwise_measurement(cand_path, model, trunk_cache, log):
    rows, pairs, pair_stats = JT.build_pairs(cand_path, log=log)
    by_sig = {row["i"]: row["branches"] for row in rows}
    n_correct = 0
    for ri, text, sig_a, sig_b, q in pairs:
        branches = by_sig[ri]
        cA, cB = branches[sig_a], branches[sig_b]   # A=true, B=wrong by construction
        score, _, _ = JF.score_pair(text, cA, cB, q, trunk_cache, model)
        if score > 0:
            n_correct += 1
    acc = n_correct / len(pairs) if pairs else float("nan")
    log(f"[courtroom2][wild-pairs] gold-known pairwise accuracy: {n_correct}/{len(pairs)} = {acc:.4f} "
        f"(rows contributing: {pair_stats['n_contrib']}; no-true-story rows excluded: {pair_stats['n_no_true']})")
    return acc, pair_stats


# ================================================================================================
# MAIN -- mirrors courtroom.py's own main() control flow (tune-on-diet, then one fixed-tau run),
# just against a different output basename and with the trained-juror measurement appended.
# ================================================================================================
def main():
    ap = argparse.ArgumentParser(description="THE CARICATURE JURY wired into courtroom.py's cross-examination step.")
    ap.add_argument("rawslots")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--tau", type=float, default=None)
    ap.add_argument("--tune", action="store_true")
    ap.add_argument("--k", type=int, default=8, dest="k_max")
    ap.add_argument("--cand", default=None)
    ap.add_argument("--fixture", default=None)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--tau-grid", default="0,1,2,3,4,5,6,7,8")
    ap.add_argument("--tune-rule-regr-max", type=int, default=5)
    ap.add_argument("--model", default=".cache/picker/jury_model.pkl")
    ap.add_argument("--wild-pair-measure", action="store_true",
                     help="also run the gold-known pairwise-accuracy measurement on THIS dump's own candidates")
    args = ap.parse_args()

    lines = []
    def log(s=""):
        print(s, flush=True); lines.append(s)

    tag = args.tag or CT.guess_tag(args.rawslots)
    cand_path = args.cand or CT.guess_cand_path(args.rawslots, tag)
    fixture_path = args.fixture or CT.guess_fixture_path(args.rawslots)
    b = os.path.basename(args.rawslots)
    if "wild" in b:
        out_base = f".cache/jury_{tag}"
    elif "slicevalid2" in b or "valid2" in b:
        out_base = f".cache/jury_diet_{tag}"
    else:
        import re
        out_base = ".cache/jury_" + re.sub(r"^rawslots_", "", b).replace(".pkl", "")
    out_txt, out_pkl = out_base + ".txt", out_base + ".pkl"

    log(f"[courtroom2] loading trained juror: {args.model}")
    model = pickle.load(open(args.model, "rb"))
    log(f"[courtroom2] juror CV accuracy (diet, from training): full={model['cv_acc_full']:.4f} "
        f"embed-only={model['cv_acc_embed_only']:.4f} struct-only={model['cv_acc_struct_only']:.4f} "
        f"| hand-jurors baseline (same diet pairs) = {model['hand_acc_all']:.4f}")

    # candidates must exist on disk already (courtroom.load_candidates will generate if missing,
    # but we need the row texts BEFORE that to pre-embed in one batched pass) -- if the cache is
    # missing just let courtroom's own loader build it first, then re-derive texts from it.
    if not (cand_path and os.path.exists(cand_path)):
        log(f"[courtroom2] {cand_path} missing -- delegating to courtroom.load_candidates (will call gen_candidates)")
        CT.load_candidates(args.rawslots, cand_path, args.workers, args.limit, log)
    cands = pickle.load(open(cand_path, "rb"))
    if args.limit:
        cands = cands[:args.limit]
    texts = sorted({row["text"] for row in cands})
    log(f"[courtroom2] pre-embedding {len(texts)} unique row texts through the frozen trunk (CPU, cached for the "
        f"whole run -- the tau sweep below pays this ONCE)...")
    t0 = time.time()
    trunk_cache = JF.embed_texts(texts, batch_size=32, log=log)
    log(f"[courtroom2] trunk pre-embedding done in {time.time()-t0:.0f}s")

    stats_box = {"n_scored": 0, "score_sum": 0.0}
    install_trained_juror(model, trunk_cache, stats_box)
    log("[courtroom2] SWAPPED: courtroom.pairwise_votes -> the trained caricature juror "
        "(bailiff/exhaustion/verdict-with-a-margin all reused from courtroom.py, unedited)")

    if not args.tune and args.tau is None:
        log("[courtroom2] neither --tau nor --tune given -- nothing to do"); sys.exit(2)

    tau_sweep_table = None
    chosen_tau = args.tau
    if args.tune:
        grid = [float(x) for x in args.tau_grid.split(",")]
        tau_sweep_table = []
        best = None
        for t in grid:
            res_t = CT.run_courtroom(args.rawslots, tag, cand_path, fixture_path, t, args.k_max, args.workers, args.limit, log)
            tau_sweep_table.append((t, res_t["verdict_correct"], res_t["regressions"], res_t["fixes"]))
            log(f"[courtroom2][tune] tau={t} rows_correct={res_t['verdict_correct']} regressions={len(res_t['regressions'])}")
            if len(res_t["regressions"]) <= args.tune_rule_regr_max:
                if best is None or res_t["verdict_correct"] > best[1]:
                    best = (t, res_t["verdict_correct"])
        if chosen_tau is None:
            if best is None:
                chosen_tau = max(grid)
                log(f"[courtroom2][tune] NO tau kept diet regressions <= {args.tune_rule_regr_max} -- "
                    f"falling back to the largest grid tau ({chosen_tau})")
            else:
                chosen_tau = best[0]
                log(f"[courtroom2][tune] RULE: smallest tau maximizing diet rows_correct subject to regressions <= "
                    f"{args.tune_rule_regr_max} -> tau={chosen_tau} (rows_correct={best[1]})")

    final = CT.run_courtroom(args.rawslots, tag, cand_path, fixture_path, chosen_tau, args.k_max, args.workers, args.limit, log)

    rep_lines, P = CT.write_report(out_txt, tag, tau_sweep_table)
    CT.report_run(final, P, f"RUN ({os.path.basename(args.rawslots)}) -- THE TRAINED CARICATURE JURY")
    P("  NOTE: the stop-reason / confusion / fixes / regressions above are courtroom.py's own verdict "
      "machinery, UNCHANGED; the 'decided_cert/unexplained/collision' breakdown line inside it is NOT "
      "meaningful here (all three collapse to the one trained juror's sign).")
    P("")

    wild_pair_acc = None
    if args.wild_pair_measure or "wild" in b:
        P("-" * 94); P("GOLD-KNOWN PAIRWISE ACCURACY (true-vs-wrong candidate pairs from THIS dump's own pool, "
                       "scored by the trained juror -- not retrained/retuned here, a pure read):"); P("-" * 94)
        wild_pair_acc, pair_stats = wild_pairwise_measurement(cand_path, model, trunk_cache, log)
        P(f"  accuracy = {wild_pair_acc:.4f}  (pairs={pair_stats['n_pairs_one_dir']}, "
          f"contributing rows={pair_stats['n_contrib']}, no-true-story rows={pair_stats['n_no_true']})")
        P("")

    P("=" * 94)
    P(f"LEDGER-READY SUMMARY: {os.path.basename(args.rawslots)} tau={chosen_tau} k_max={args.k_max} -> "
      f"rows={final['verdict_correct']}/{final['n']} vs top-1 {final['top1_correct']} "
      f"(fixes={len(final['fixes'])} regressions={len(final['regressions'])}); "
      f"gold-known pairwise accuracy={wild_pair_acc}")

    with open(out_txt, "w") as f:
        f.write("\n".join(rep_lines) + "\n")
    with open(out_pkl, "wb") as f:
        pickle.dump(dict(tag=tag, dump_path=args.rawslots, tau=chosen_tau, k_max=args.k_max,
                          tau_sweep_table=tau_sweep_table, final=final, wild_pair_acc=wild_pair_acc,
                          model_path=args.model), f)
    print(f"[courtroom2] wrote {out_txt} + {out_pkl}")


if __name__ == "__main__":
    main()
