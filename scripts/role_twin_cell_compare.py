"""role_twin_cell_compare.py -- THE ROLE SIGNATURE's twin-cell BAR (2026-09-22, the coordinator's
resolution of gate (c): the 76-row fixture was too small to resolve the same-noun/different-noun
bar; re-run on the 1024-row slice with the paired SE convention this codebase already uses
(scripts/paired_read.py's McNemar-style paired-difference SE) instead of independent-binomial SE.

Reuses twin_split_census.py's structural classification (build_recs/classify/CLASSES) UNCHANGED.
Takes FOUR dump pickles (all loop_val.py LV_DUMP=... outputs on the SAME fixture, SAME row order):
pre-treatment, post-treatment, pre-control, post-control. For each twin class (SAME-NOUN,
DIFFERENT-NOUN):
  - n = the class's population (shared across all 4 dumps -- a structural property of the rows,
    not the checkpoint).
  - p_pre/p_post per arm, and delta = p_post - p_pre with the PAIRED SE (paired_read.py's formula,
    applied here per-item within the class rather than aggregated over paired_read's LV_PER_SLOT
    convention -- same math, same source: se_paired = sqrt((b+c)/n - ((b-c)/n)^2) / sqrt(n) where
    b/c are the discordant counts between the pre and post reads).
  - the CONTROL-SUBTRACTED change (delta_treat - delta_ctrl), with SE = sqrt(se_treat^2 +
    se_ctrl^2) (the two arms are independently trained runs -- no pairing between them) and its z
    (= change / SE).

usage: role_twin_cell_compare.py <fixture.jsonl> <pre_treat.pkl> <post_treat.pkl> <pre_ctrl.pkl> <post_ctrl.pkl>
"""
import json
import math
import sys

sys.path.insert(0, ".")
sys.path.insert(0, "scripts")

import twin_split_census as TSC


def load_rows(path):
    return [json.loads(l) for l in open(path)]


def item_key(r):
    return (r["row"], r["j"], r["pos"])


def per_item_correct(recs, pargs):
    """bool per rec, in recs' own order -- v in the dump's top-2 predicted args for (row,j).
    None (not scored -- the dump has no entry for this row/j) is dropped, never counted wrong."""
    out = {}
    for r in recs:
        pr = pargs.get(r["row"], {}).get(r["j"])
        if pr is None:
            continue
        out[item_key(r)] = bool(r["v"] in pr)
    return out


def paired_se(a, b):
    """a, b: aligned bool arrays (same items, pre vs post). Returns (delta, se) via
    paired_read.py's McNemar-style formula, verbatim."""
    n = len(a)
    only_a = sum(1 for x, y in zip(a, b) if x and not y)   # right under 'a' (pre) only
    only_b = sum(1 for x, y in zip(a, b) if y and not x)   # right under 'b' (post) only
    mean_a = sum(a) / n
    mean_b = sum(b) / n
    d = mean_b - mean_a
    se = math.sqrt((only_a + only_b) / n - ((only_a - only_b) / n) ** 2) / math.sqrt(n)
    return d, se, only_a, only_b


def main():
    fixture_path, pre_treat_p, post_treat_p, pre_ctrl_p, post_ctrl_p = sys.argv[1:6]

    rows = load_rows(fixture_path)
    recs = TSC.build_recs(rows)
    classified = [r for r in recs if r["has_comp"]]

    dumps = {
        "pre_treat": TSC.load_dump_pargs(pre_treat_p),
        "post_treat": TSC.load_dump_pargs(post_treat_p),
        "pre_ctrl": TSC.load_dump_pargs(pre_ctrl_p),
        "post_ctrl": TSC.load_dump_pargs(post_ctrl_p),
    }
    per_item = {name: per_item_correct(classified, pargs) for name, pargs in dumps.items()}

    print(f"[twin-cell-compare] fixture={fixture_path} rows={len(rows)} "
          f"argument instances={len(recs)} classified={len(classified)}")
    print()

    results = {}
    for cls in TSC.CLASSES:
        xs = [r for r in classified if r["cls"] == cls]
        keys = [item_key(r) for r in xs]
        # only items scored in ALL FOUR dumps (should be ~all of them -- same fixture, same rows)
        common = [k for k in keys if all(k in per_item[name] for name in dumps)]
        n = len(common)
        if n == 0:
            print(f"[twin-cell-compare] {cls}: n=0, skipped")
            continue
        pre_t = [per_item["pre_treat"][k] for k in common]
        post_t = [per_item["post_treat"][k] for k in common]
        pre_c = [per_item["pre_ctrl"][k] for k in common]
        post_c = [per_item["post_ctrl"][k] for k in common]

        d_treat, se_treat, ta, tb = paired_se(pre_t, post_t)
        d_ctrl, se_ctrl, ca, cb = paired_se(pre_c, post_c)
        change = d_treat - d_ctrl
        se_change = math.sqrt(se_treat ** 2 + se_ctrl ** 2)
        z = change / se_change if se_change > 0 else float("nan")

        results[cls] = dict(n=n, p_pre_treat=sum(pre_t) / n, p_post_treat=sum(post_t) / n,
                            d_treat=d_treat, se_treat=se_treat,
                            p_pre_ctrl=sum(pre_c) / n, p_post_ctrl=sum(post_c) / n,
                            d_ctrl=d_ctrl, se_ctrl=se_ctrl,
                            change=change, se_change=se_change, z=z)
        print(f"[twin-cell-compare] {cls} (n={n}):")
        print(f"    treat: pre={sum(pre_t)/n:.4f} post={sum(post_t)/n:.4f} "
              f"delta={d_treat:+.4f} (paired SE {se_treat:.4f})")
        print(f"    ctrl:  pre={sum(pre_c)/n:.4f} post={sum(post_c)/n:.4f} "
              f"delta={d_ctrl:+.4f} (paired SE {se_ctrl:.4f})")
        print(f"    CONTROL-SUBTRACTED change: {change:+.4f}  SE={se_change:.4f}  "
              f"z={z:+.2f} ({change/se_change if se_change>0 else float('nan'):+.2f} SE)")
        print()

    if "SAME-NOUN" in results and "DIFFERENT-NOUN" in results:
        sn = results["SAME-NOUN"]; dn = results["DIFFERENT-NOUN"]
        fires = sn["z"] >= 2.0 and dn["change"] >= -1.0 * dn["se_change"]
        print("=" * 70)
        print(f"BAR: same-noun control-subtracted rise >= 2 SE (got {sn['z']:+.2f} SE) AND "
              f"different-noun does not fall by more than 1 SE "
              f"(got {dn['change']/dn['se_change'] if dn['se_change']>0 else float('nan'):+.2f} SE)")
        print(f"VERDICT: THE ARM {'FIRES' if fires else 'DOES NOT FIRE'}")
        print("=" * 70)


if __name__ == "__main__":
    main()
