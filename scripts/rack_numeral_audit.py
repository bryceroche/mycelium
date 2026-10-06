"""rack_numeral_audit.py — THE RACK's numeral-level audit (2026-10-06, zero GPU; delegated task Part B).

scripts/rack_dryness_census.py's EXACT dry census (the sidecar <dump>.rack.npz: the flags the body
ACTUALLY committed at its last consult) reports only a POSITIONAL precision — P(the committed slot j
decodes the row's gold slot j exactly right | committed). That number for RK_241 is 0.526 (458 cells,
0.223 of gold slots; .cache/rack_dryness_census_RK_241.txt). This script asks the NUMERAL question
instead, reusing that script's own audit convention (the v_at / v_in columns it prints for the
UN-TRAINED proxy test, e.g. value-in-gold=0.972 for given_unique on PMS8_241): of the SAME 458
committed (row, slot) cells (read from the sidecar directly, not the recomputed proxy), what share
claimed a numeral that is

  (a) the value of one of the row's gold GIVEN slots (ftype == 1), order-free — "is this even one of
      the numbers this row gives?" — and
  (b) the value of SOME gold literal slot of the row (ftype != 0: given/mod/sel/pct/fdiv/macro/frac),
      exactly rack_dryness_census.py's existing "value-in-gold" definition, generalized from the
      proxy's fired[] set to the sidecar's exact committed set.

plus the per-row histogram of dry-slot counts.

The committed cell's CLAIMED NUMERAL is not itself stored in the sidecar (only per-test boolean dry
flags) — it is read from the matching LV_DUMP tuple (.cache/dump_wild_RK_241.pkl), which carries the
model's own masked/legal decode for every GOLD-PRESENT slot (loop_val's LV_DUMP write happens after
the LV_LEGAL=num mask is applied, the same numeral the rack's own masked decode would see). A committed
cell with no gold slot at all (no dump tuple — 2 of 458 on this census) has no recoverable numeral
from this artifact; it is counted in the denominator as a miss for both audit columns (consistent with
rack_dryness_census.py's own ok.get(cell, False) convention for cells that fall outside the gold-present
bank) and reported separately so the gap is visible.

usage: rack_numeral_audit.py <dump.pkl.rack.npz> <dump.pkl> [--gold-npz states.npz] [--tag TAG] [--append path]
"""
import os
import sys
import argparse
import pickle
import collections

sys.path.insert(0, "."); sys.path.insert(0, "scripts")
import numpy as np

L_FAC = 24   # the family's factor-slot count (ALG2=1 ALG_FTYPES=9 ALG_DUP=1); asserted against the sidecar's own width below


def _parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("sidecar", help=".cache/dump_wild_<tag>.pkl.rack.npz")
    ap.add_argument("dump", help=".cache/dump_wild_<tag>.pkl (LV_DUMP tuples)")
    ap.add_argument("--gold-npz", default=".cache/phase1_alg_states_wildhold.npz")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--append", default=None, help="append the report under a dated heading to this file (default: .cache/rack_dryness_census_<tag>.txt)")
    return ap.parse_args()


def main():
    a = _parse_args()
    tag = a.tag or os.path.basename(a.dump).replace("dump_wild_", "").replace(".pkl", "")
    append_path = a.append or f".cache/rack_dryness_census_{tag}.txt"

    z = np.load(a.sidecar, allow_pickle=True)
    rows = z["rows"]; dry = z["dry"]; tests = [str(x) for x in z["tests"]]
    assert dry.shape[1] >= L_FAC, (dry.shape, L_FAC)
    # THE COMMITTED (row, slot) FLAGS, exactly as rack_dryness_census.py's "THE DRY CENSUS (EXACT...)"
    # UNION row reads them: dry[:, :L_FAC] > 0.5 (the full L_TOT-wide array; scratch slots L_FAC.. are
    # never set by a dryness test, asserted below rather than silently trusted).
    assert float(dry[:, L_FAC:].max(initial=0.0)) == 0.0, "a scratch slot (j >= L_FAC) is marked dry — the L_FAC assumption is wrong for this body"
    cells = [(int(rows[k]), j) for k in range(len(rows)) for j in range(L_FAC) if dry[k, j] > 0.5]
    n_cells = len(cells)
    print(f"[numeral-audit] {tag}: {n_cells} committed (row, slot) cells from {os.path.basename(a.sidecar)} (tests={tests})", flush=True)

    recs = pickle.load(open(a.dump, "rb"))
    # LV_DUMP tuple: (i, j, gft, gop, gargs, gres, gdig, pft, pop, pargs, pres_, pdig, ppres, pdup)
    value_of = {}
    for t in recs:
        i, j = int(t[0]), int(t[1])
        pdig = t[11]
        value_of[(i, j)] = int("".join(str(int(d)) for d in pdig))
    n_no_dump = sum(1 for c in cells if c not in value_of)
    print(f"[numeral-audit] {n_no_dump}/{n_cells} committed cells have NO LV_DUMP entry (no gold slot there at all) — scored as a miss on both audit columns below", flush=True)

    gz = np.load(a.gold_npz)
    assert gz["g_presence"].shape[1] == L_FAC, (gz["g_presence"].shape, L_FAC)
    n_rows_fixture = gz["g_presence"].shape[0]

    def gval(i, j):
        return int("".join(str(int(x)) for x in gz["g_digits"][i, j]))

    # per-row gold GIVEN values (ftype == 1) and gold LITERAL values (ftype != 0), precomputed once
    given_vals = collections.defaultdict(set)   # row -> {gold given numerals}
    lit_vals = collections.defaultdict(set)     # row -> {gold literal numerals, any ftype != 0}
    for i in range(n_rows_fixture):
        for j in range(L_FAC):
            if gz["g_presence"][i, j] < 0.5:
                continue
            ft = int(gz["g_ftype"][i, j])
            if ft == 0:
                continue
            v = gval(i, j)
            lit_vals[i].add(v)
            if ft == 1:
                given_vals[i].add(v)

    n_given_match = n_lit_match = 0
    rows_with_dry = collections.Counter()   # row -> dry-slot count (the histogram)
    for k in range(len(rows)):
        rows_with_dry[int(rows[k])] = int((dry[k, :L_FAC] > 0.5).sum())

    for (i, j) in cells:
        v = value_of.get((i, j))
        if v is None:
            continue  # no dump entry: already counted as a miss (n_no_dump); contributes 0 to both match counts
        if v in given_vals.get(i, ()):
            n_given_match += 1
        if v in lit_vals.get(i, ()):
            n_lit_match += 1

    prec_given = n_given_match / n_cells if n_cells else float("nan")
    prec_lit = n_lit_match / n_cells if n_cells else float("nan")

    hist = collections.Counter(rows_with_dry.values())
    # rows in the fixture that never appear in `rows` (0 consults reached, or genuinely 0 dry slots
    # throughout) count as 0 dry slots too, for an honest per-row histogram over the WHOLE wild holdout.
    seen_rows = set(int(r) for r in rows)
    hist[0] += max(0, n_rows_fixture - len(seen_rows))
    max_dry_row = max(rows_with_dry, key=lambda r: rows_with_dry[r]) if rows_with_dry else None

    lines = []
    lines.append(f"THE NUMERAL-LEVEL AUDIT — {tag} ({os.path.basename(a.sidecar)} x {os.path.basename(a.dump)}; {n_cells} committed cells, the sidecar's exact UNION)")
    lines.append(f"  (positional precision for the same 458 cells, from the dry census above: 0.526 — this is the numeral question instead)")
    lines.append(f"  committed cells with NO dump entry (no gold slot there): {n_no_dump}/{n_cells} (scored as a miss below)")
    lines.append(f"  numeral-level precision vs gold GIVENS only (ftype==1, any slot of the row): {n_given_match}/{n_cells} = {prec_given:.3f}")
    lines.append(f"  numeral-level precision vs gold ANY literal slot  (ftype!=0, any slot of the row): {n_lit_match}/{n_cells} = {prec_lit:.3f}")
    lines.append(f"  (of the {n_cells - n_no_dump} cells WITH a dump entry: given-match {n_given_match}/{n_cells - n_no_dump} = {n_given_match / max(n_cells - n_no_dump, 1):.3f}; "
                 f"any-slot-match {n_lit_match}/{n_cells - n_no_dump} = {n_lit_match / max(n_cells - n_no_dump, 1):.3f})")
    lines.append(f"  per-row dry-slot count histogram (over all {n_rows_fixture} wild holdout rows; 0 = row never committed a slot):")
    for k in sorted(hist):
        lines.append(f"    {k:2d} dry slots: {hist[k]:4d} rows")
    if max_dry_row is not None:
        lines.append(f"  max dry slots in one row: {rows_with_dry[max_dry_row]} (row {max_dry_row})")
    lines.append(f"  rows with >=1 dry slot: {sum(1 for v in rows_with_dry.values() if v >= 1)} / {n_rows_fixture}")

    txt = "\n".join(lines)
    print(txt)

    from datetime import datetime
    stamp = os.popen("date '+%Y-%m-%d %H:%M'").read().strip()
    heading = f"\n\n### {stamp} — THE NUMERAL-LEVEL AUDIT (delegated task Part B; zero GPU)\n"
    with open(append_path, "a") as f:
        f.write(heading + txt + "\n")
    print(f"\n[numeral-audit] appended -> {append_path}")


if __name__ == "__main__":
    main()
