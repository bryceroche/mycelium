"""role_twin_cell.py -- THE ROLE SIGNATURE's twin-cell metric (2026-09-22, step 1 gate (c) of
THE ROLE SIGNATURE organ). Reuses twin_split_census.py's helpers (build_recs/classify/
load_dump_pargs/CLASSES) UNCHANGED -- never reimplements the same-noun/different-noun split --
restricted to the 64+16-row mixed fixture (.cache/form_pm35_mix64r.jsonl +
form_pm35_mix64r_val.jsonl, rebuilt from pm35c so every row carries role_cues/arg_role_cues) and a
supplied dump pickle (produced by loop_val.py LV_DUMP=<path>, the same tuple format
twin_split_census.load_dump_pargs already reads).

usage: role_twin_cell.py <dump.pkl> [label]
"""
import json
import sys

sys.path.insert(0, ".")
sys.path.insert(0, "scripts")

import twin_split_census as TSC

# THE MERGED FIXTURE (76 rows): form_pm35_mix64r.jsonl (64) + _val (16),
# MINUS 4 rows whose pm35c-stamped "solution" field is empty (the
# custody-gold TEST-side check in load_alg indexes solution[query_var]
# unconditionally and IndexErrors on those 4 -- harmless on the TRAIN
# side, where solution is never gold, but this fixture is read as
# ALG_TEST for the loop_val dumps). MUST stay in the SAME row order the
# dump's row indices were built from (.cache/form_pm35_mix64r_all2.jsonl,
# the file --precompute'd under ALG_TEST_NAME=pm35mix64rall2).
FIXTURE_ALL = ".cache/form_pm35_mix64r_all2.jsonl"


def load_rows():
    return [json.loads(l) for l in open(FIXTURE_ALL)]


def main():
    dump_path = sys.argv[1]
    label = sys.argv[2] if len(sys.argv) > 2 else dump_path

    rows = load_rows()
    recs = TSC.build_recs(rows)
    pargs = TSC.load_dump_pargs(dump_path)

    classified = [r for r in recs if r["has_comp"]]
    n_nocomp = sum(1 for r in recs if not r["has_comp"])
    print(f"[twin-cell] {label}: {len(rows)} rows, {len(recs)} argument instances, "
          f"{n_nocomp} no-competitor (excluded), {len(classified)} classified")
    out = {}
    for c in TSC.CLASSES:
        xs = [r for r in classified if r["cls"] == c]
        corr = [r["v"] in pargs.get(r["row"], {}).get(r["j"], [])
                for r in xs if r["j"] in pargs.get(r["row"], {})]
        acc = sum(corr) / len(corr) if corr else float("nan")
        out[c] = (acc, len(corr), len(xs))
        print(f"[twin-cell] {label}  {c:16s} n={len(xs):4d} n_scored={len(corr):4d} "
              f"args-correct={acc:.4f}")
    print(f"[twin-cell] {label} SUMMARY same_noun={out['SAME-NOUN'][0]:.4f} "
          f"(n={out['SAME-NOUN'][1]}) different_noun={out['DIFFERENT-NOUN'][0]:.4f} "
          f"(n={out['DIFFERENT-NOUN'][1]})")


if __name__ == "__main__":
    main()
