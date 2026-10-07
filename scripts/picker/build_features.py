"""scripts/picker/build_features.py -- THE PANEL PICKER's feature table (2026-10-05, delegate).

Loads a gen_candidates.py output pickle (per-row candidate records) + the matching gold-factor fixture,
computes every angle in scripts/picker/angles.py for every candidate of every row, and writes a flat
feature table (list of dicts; one per candidate) to a pickle. Prints THE FEATURE LIST explicitly (the
task brief's requirement).

BUG FIX (2026-10-06, Goodhart-fence audit): twin_flip_flag/twin_n_checked (angles.build_row_tables /
angles.twin_flip_flag) are now DIAGNOSTIC-ONLY (written to each record's top level, never into `X`,
never in FEATURE_NAMES/ANGLE_GROUPS) because their computation is NOT key-free: angles.build_row_tables
-> lexical_identity_census.build_candidate_tables keys every non-given slot's lexical identity off
SAM.own_value(fac, solution) -- `solution` is the row's own CUSTODY-GOLD answer array (fixture_row["solution"]).
That means the "twin" angle, if it feeds the trained picker's score (as it did until this fix: it was in
FEATURE_NAMES and therefore in the model clf.decision_function() consults at wild_read.py's argmax), lets
the picker consult THIS ROW's own gold answer while CHOOSING among candidates for this same row -- the
Goodhart fence violation docs/CLAUDE.md and diagnostic_register.py both name (see
scripts/picker/angles.py's TWIN_FLAG_NOTE, which claimed "no gold at read time"; that claim was false for
the deployed wild pipeline, which passes .cache/wild_admitted_holdout_a.jsonl -- full factors+solution --
as `fixture_jsonl` at wild build-features time: .cache/picker/run_panel_picker.sh step 3). Contrast with
scripts/courtroom.py's own use of the SAME angles.twin_flip_flag: there it only increments a `stats`
counter for the printed report, never touching the Copeland `scores` that pick the winner -- that use
stays safe. Here the angle fed `X` directly, so it was live at selection time. The diet's own 5-fold CV
ablation already showed this angle net-NEGATIVE (-twin: 305/775 vs FULL 297/775, delta +8) even before
the leak was understood structurally, so dropping it from the feature set costs nothing it was buying.
The one registered wild read (.cache/panel_picker_PMS8_241.txt, 19/311, ledger c02ce884) was produced
with this feature live and should be treated as compromised; a fresh wild read under the corrected
feature set is a decision for Bryce (wild is read ONCE per the project's own law), not fired here.

Usage: .venv/bin/python3 scripts/picker/build_features.py <cand_pkl> <fixture_jsonl> <out_pkl>
"""
import sys
import json
import pickle

sys.path.insert(0, "."); sys.path.insert(0, "scripts")
import numpy as np

import angles as A

FEATURE_NAMES = [
    "status_solved", "status_inconsistent", "status_refused",
    "unique", "unique_known",
    "loglik", "loglik_centered", "loglik_rank",
    "graph_len", "graph_len_diff_top1",
    "nlc_value_present", "nlc_coverage", "nlc_no_double", "nlc_cue_agree", "nlc_chain_reaches",
    "nlc_derived_stated", "nlc_cert",
    "n_args_flip", "n_val_flip", "n_typeop_flip",
]
# twin_flip_flag/twin_n_checked are intentionally NOT in FEATURE_NAMES -- see the Goodhart-fence
# bug-fix note above. They are still computed (DIAG_KEYS below) for reporting/diagnostics only.

ANGLE_GROUPS = {
    "status": ["status_solved", "status_inconsistent", "status_refused"],
    "unique": ["unique", "unique_known"],
    "loglik": ["loglik", "loglik_centered", "loglik_rank"],
    "length": ["graph_len", "graph_len_diff_top1"],
    "nlc": ["nlc_value_present", "nlc_coverage", "nlc_no_double", "nlc_cue_agree", "nlc_chain_reaches",
            "nlc_derived_stated", "nlc_cert"],
    "flips": ["n_args_flip", "n_val_flip", "n_typeop_flip"],
}

DIAG_KEYS = ["twin_flip_flag", "twin_n_checked"]   # gold-derived; diagnostic-only, never in X


def status_bucket(status):
    if status == "solved":
        return "solved"
    if status == "unsat":
        return "inconsistent"
    return "refused"   # budget / timeout / unbuildable:* / hung / ? / None


def build(cand_pkl, fixture_jsonl, out_pkl, log=print):
    rows = pickle.load(open(cand_pkl, "rb"))
    fixture = [json.loads(l) for l in open(fixture_jsonl)] if fixture_jsonl else None
    out = []
    n_twin_tables_built = 0
    for row in rows:
        i, text, key, q = row["i"], row["text"], row["key"], row["q"]
        branches = row["branches"]
        if not branches:
            continue
        top1_sig = row.get("top1_sig")
        top1 = branches.get(top1_sig) if top1_sig is not None else None
        if top1 is None:
            # fall back: the (0,0,0)-tagged branch
            for sig, b in branches.items():
                if (0, 0, 0) in b["tags"]:
                    top1 = b; break
        top1_parse = top1["parse"] if top1 is not None else []
        top1_len = len(top1_parse)

        tables = None
        if fixture is not None and i < len(fixture) and "factors" in fixture[i]:
            try:
                tables = A.build_row_tables(fixture[i])
                n_twin_tables_built += 1
            except Exception:
                tables = None

        logliks = {sig: b.get("loglik", float("-inf")) for sig, b in branches.items()}
        finite = [v for v in logliks.values() if np.isfinite(v)]
        row_max_ll = max(finite) if finite else 0.0
        order = sorted(branches.keys(), key=lambda s: -logliks[s])
        rank_of = {sig: r for r, sig in enumerate(order)}

        for sig, b in branches.items():
            status = b.get("status")
            bucket = status_bucket(status)
            parse = b["parse"]
            ll = logliks[sig]
            ll_fin = ll if np.isfinite(ll) else (row_max_ll - 50.0)
            certs = A.nlc_certificates(text, parse, b.get("assignment"), q) if parse else {k: 0.0 for k in A.NLC_KEYS}
            n_args_flip, n_val_flip, n_typeop_flip = A.diff_from_top1(parse, top1_parse)
            twin_flag, twin_n = A.twin_flip_flag(tables, parse, top1_parse) if tables is not None else (False, 0)
            feat = dict(
                status_solved=1.0 if bucket == "solved" else 0.0,
                status_inconsistent=1.0 if bucket == "inconsistent" else 0.0,
                status_refused=1.0 if bucket == "refused" else 0.0,
                unique=1.0 if b.get("unique") is True else 0.0,
                unique_known=1.0 if status == "solved" else 0.0,
                loglik=ll_fin,
                loglik_centered=ll_fin - row_max_ll,
                loglik_rank=float(rank_of[sig]),
                graph_len=float(len(parse)),
                graph_len_diff_top1=float(len(parse) - top1_len),
                nlc_value_present=certs["value_present"], nlc_coverage=certs["coverage"],
                nlc_no_double=certs["no_double"], nlc_cue_agree=certs["cue_agree"],
                nlc_chain_reaches=certs["chain_reaches"], nlc_derived_stated=certs["derived_stated"],
                nlc_cert=certs["cert"],
                n_args_flip=float(n_args_flip), n_val_flip=float(n_val_flip), n_typeop_flip=float(n_typeop_flip),
            )
            out.append(dict(row=i, sig=sig, y=bool(b.get("correct")), status=status, value=b.get("value"),
                             key=key, is_top1=(sig == top1_sig), loglik_raw=ll, X=feat,
                             # DIAGNOSTIC ONLY (gold-derived -- see module docstring): never merged into X.
                             diag_twin_flip_flag=1.0 if twin_flag else 0.0, diag_twin_n_checked=float(twin_n)))
    pickle.dump(out, open(out_pkl, "wb"))
    log(f"[build-features] {len(rows)} rows -> {len(out)} candidate rows ({n_twin_tables_built} rows had twin tables) "
        f"-> {out_pkl}")
    log("[build-features] FEATURE LIST:")
    for name in FEATURE_NAMES:
        log(f"    {name}")
    log("[build-features] ANGLE GROUPS (for ablation):")
    for g, names in ANGLE_GROUPS.items():
        log(f"    {g}: {names}")
    log(f"[build-features] DIAGNOSTIC-ONLY keys (gold-derived, never in X/FEATURE_NAMES): {DIAG_KEYS}")
    return out


def main():
    cand_pkl, fixture_jsonl, out_pkl = sys.argv[1], sys.argv[2], sys.argv[3]
    build(cand_pkl, fixture_jsonl if fixture_jsonl != "-" else None, out_pkl)


if __name__ == "__main__":
    main()
