"""busreg_cell_compare.py -- THE BUS REGISTER's resolution-gate BAR (2026-09-22): the twin-cell
compare (role_twin_cell_compare.py's four-dump, paired-SE, control-subtracted machinery, reused
verbatim) over THE MIXED-BUCKET DECOMPOSITION's cells (mixed_bucket_census.py's process_row: the
inherited relation key, derived-from-k, k-is-a-relation) instead of the literal-key twin classes.

Cells (each an argument instance (row, j, pos) with >= 1 same-sentence competitor):
  DERIVED-ARG      k is itself a relation (a derived argument) -- 55-57% of the wild wall, the cell
                   the register's PROPAGATED identity is built for
  GIVEN-ONLY       k a given and every competitor a given -- the looked-up identity's cell (already
                   0.86-0.87 on wild; must HOLD)
  GIVEN-RELCOMP    k a given with >= 1 relation competitor
  DERIVED-COMP     a derived-from-k competitor is present (the same quantity at a later time)
  SAME-NOUN(inh) / DIFFERENT-NOUN(inh) / MIXED(inh)   the inherited-key classes
THE BAR (pinned before the read): DERIVED-ARG control-subtracted rise >= 2 SE, GIVEN-ONLY does
not fall by more than 1 SE. Everything else is reported, not judged.

usage: busreg_cell_compare.py <fixture.jsonl> <pre_treat.pkl> <post_treat.pkl> <pre_ctrl.pkl> <post_ctrl.pkl>
"""
import math
import sys

sys.path.insert(0, ".")
sys.path.insert(0, "scripts")

import twin_split_census as TSC
import mixed_bucket_census as MBC
from role_twin_cell_compare import load_rows, item_key, per_item_correct, paired_se

CELLS = (
    ("DERIVED-ARG", lambda r: r["k_is_rel"]),
    ("GIVEN-ONLY", lambda r: (not r["k_is_rel"]) and r["n_rel_comp"] == 0),
    ("GIVEN-RELCOMP", lambda r: (not r["k_is_rel"]) and r["n_rel_comp"] > 0),
    ("DERIVED-COMP", lambda r: r["derived"]),
    ("SAME-NOUN(inh)", lambda r: r["new"] == "SAME-NOUN"),
    ("DIFFERENT-NOUN(inh)", lambda r: r["new"] == "DIFFERENT-NOUN"),
    ("MIXED(inh)", lambda r: r["new"] == "MIXED/NO-KEY"),
    ("ALL", lambda r: True),
)


def main():
    fixture_path, pre_treat_p, post_treat_p, pre_ctrl_p, post_ctrl_p = sys.argv[1:6]
    rows = load_rows(fixture_path)
    recs = []
    for i, row in enumerate(rows):
        recs.extend(MBC.process_row(row, i))   # every rec has >= 1 competitor by construction
    dumps = {
        "pre_treat": TSC.load_dump_pargs(pre_treat_p),
        "post_treat": TSC.load_dump_pargs(post_treat_p),
        "pre_ctrl": TSC.load_dump_pargs(pre_ctrl_p),
        "post_ctrl": TSC.load_dump_pargs(post_ctrl_p),
    }
    per_item = {name: per_item_correct(recs, pargs) for name, pargs in dumps.items()}
    print(f"[busreg-cell-compare] fixture={fixture_path} rows={len(rows)} classified instances={len(recs)}")
    print()
    results = {}
    for name, pred in CELLS:
        keys = [item_key(r) for r in recs if pred(r)]
        common = [k for k in keys if all(k in per_item[d] for d in dumps)]
        n = len(common)
        if n == 0:
            print(f"[busreg-cell-compare] {name}: n=0, skipped")
            continue
        pre_t = [per_item["pre_treat"][k] for k in common]
        post_t = [per_item["post_treat"][k] for k in common]
        pre_c = [per_item["pre_ctrl"][k] for k in common]
        post_c = [per_item["post_ctrl"][k] for k in common]
        d_t, se_t, _, _ = paired_se(pre_t, post_t)
        d_c, se_c, _, _ = paired_se(pre_c, post_c)
        change = d_t - d_c
        se = math.sqrt(se_t ** 2 + se_c ** 2)
        z = change / se if se > 0 else float("nan")
        results[name] = dict(n=n, change=change, se=se, z=z)
        print(f"[busreg-cell-compare] {name} (n={n}):")
        print(f"    treat: pre={sum(pre_t)/n:.4f} post={sum(post_t)/n:.4f} delta={d_t:+.4f} (paired SE {se_t:.4f})")
        print(f"    ctrl:  pre={sum(pre_c)/n:.4f} post={sum(post_c)/n:.4f} delta={d_c:+.4f} (paired SE {se_c:.4f})")
        print(f"    CONTROL-SUBTRACTED change: {change:+.4f}  SE={se:.4f}  z={z:+.2f}")
        print()
    if "DERIVED-ARG" in results and "GIVEN-ONLY" in results:
        da = results["DERIVED-ARG"]; go = results["GIVEN-ONLY"]
        fires = da["z"] >= 2.0 and go["change"] >= -1.0 * go["se"]
        print("=" * 70)
        print(f"BAR: DERIVED-ARG control-subtracted rise >= 2 SE (got {da['z']:+.2f} SE) AND GIVEN-ONLY "
              f"does not fall by more than 1 SE (got {go['change']/go['se'] if go['se']>0 else float('nan'):+.2f} SE)")
        print(f"VERDICT: THE ARM {'FIRES' if fires else 'DOES NOT FIRE'}")
        print("=" * 70)


if __name__ == "__main__":
    main()
