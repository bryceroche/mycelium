"""rack_numeral_audit.py — THE RACK's numeral-level audit (2026-10-06, delegated task Part B;
CORRECTED 2026-10-07 after THE DECISIVE FOLLOW-UP found the original version's numeral source
contaminated — ledger 2026-10-07 00:15/the hammerhead census file's same-dated heading).

scripts/rack_dryness_census.py's EXACT dry census (the sidecar <dump>.rack.npz: the flags the body
ACTUALLY committed at its last consult) reports only a POSITIONAL precision — P(the committed slot j
decodes the row's gold slot j exactly right | committed). That number for RK_241 is 0.526 (458 cells,
0.223 of gold slots; .cache/rack_dryness_census_RK_241.txt) — CONFIRMED CLEAN (re-run 2026-10-07): it
comes from verdict_from_ps(ps_legal_wild_RK_241.npz), a POSITIONAL verdict computed inside a SEPARATE,
fully LV_LEGAL=num masked loop_val.py run; it never touches a claimed numeral at all, let alone the
contaminated one below. This script asks the NUMERAL question instead: of the sidecar's committed
(row, slot) cells, what share claimed a numeral that is (a) the value of one of the row's gold GIVEN
slots (ftype == 1), order-free, or (b) the value of SOME gold literal slot of the row (ftype != 0).

THE BUG (found and fixed 2026-10-07): the ORIGINAL version of this script read the committed cell's
claimed numeral from the matching LV_DUMP tuple's `pdig` field (e.g. .cache/dump_wild_RK_241.pkl) on
the stated assumption that "loop_val's LV_DUMP write happens after the LV_LEGAL=num mask is applied" —
TRUE only when the loop_val.py RUN that wrote the dump was itself invoked with LV_LEGAL=num. The chain
that produced dump_wild_RK_241.pkl (.cache/rack_chain_RK.sh:18, the "open" read: LV_DUMP=... with NO
LV_LEGAL) was NOT; the masked/legal read is a SEPARATE invocation (line 20: LV_LEGAL=num, LV_PER_SLOT=
..., no LV_DUMP) that writes no dump. So the old `pdig` is the body's RAW, UNMASKED final-breath digit
argmax — a DIFFERENT number from the numeral the rack's own commit decision (rack_dry_row, which ALWAYS
masks internally, unconditionally) actually used. Reproduced end to end on row 1 ("James spends 40
years teaching... His partner... 10 years less..."; gold 40/10, the text's only two legal numerals):
the old pdig read 30/20 (illegal — not even present in the text, impossible for given_unique to have
certified by its own definition); the body's REAL consult-1/2 decode (re-run directly, this script)
gives 40/10 — both correct. The OLD column is kept below as "raw-dump (void)", for the record, never
used for a claim.

THE FIX: for every committed cell, RE-DERIVE the masked numeral by re-running the body's real two-
consult cycle (--ckpt, the full env the body was trained/read under — the caller's responsibility, as
every chain script in this family does) and calling rack_dry_row (scripts/phase1_algebra_head.py,
THE SAME unmodified organ the live rack commit uses, with its own unconditional internal mask) at
breath 2 (consult 1) and breath 4 (consult 2, fed consult 1's facts3/cert3/rack3 — loop_val.py's own
_consult3 pattern). A cell's corrected numeral is consult-1's claim if given_unique fired there,
else consult-2's (the monotone union, matching rack_pack/the sidecar's own "j in d1 or j in d2"
exactly). A REPRODUCTION CONSISTENCY CHECK is reported: does this re-derived dry set match the
sidecar's own committed set exactly (it should — same maths, same checkpoint)?

usage: rack_numeral_audit.py <dump.pkl.rack.npz> <dump.pkl> --ckpt .cache/sharp_RK_241.safetensors
         [--rows .cache/wild_admitted_holdout.jsonl] [--gold-npz states.npz] [--tag TAG]
         [--append path] [--batch 8]
       (the caller sets the body's FULL env first — FAM + SURF8 + HIER + ALG_ALT3=1 ALG_CERT=2.0
        ALG_RACK=1 ALG_RACK_TESTS=... ALG_RACK_FREEZE=... — exactly as the chain that trained/read it)
"""
import os
import sys
import json
import argparse
import pickle
import collections

sys.path.insert(0, "."); sys.path.insert(0, "scripts")
os.environ.setdefault("DEV", "CPU")
import numpy as np

L_FAC = 24   # the family's factor-slot count (ALG2=1 ALG_FTYPES=9 ALG_DUP=1); asserted against the sidecar's own width below


def _parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("sidecar", help=".cache/dump_wild_<tag>.pkl.rack.npz")
    ap.add_argument("dump", help=".cache/dump_wild_<tag>.pkl (LV_DUMP tuples) — kept for the OLD 'raw-dump (void)' column and the no-dump-entry stat")
    ap.add_argument("--ckpt", required=True, help="the checkpoint the sidecar was committed under (e.g. .cache/sharp_RK_241.safetensors) — re-run for the CORRECTED masked numeral")
    ap.add_argument("--rows", default=".cache/wild_admitted_holdout.jsonl")
    ap.add_argument("--gold-npz", default=".cache/phase1_alg_states_wildhold.npz")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--append", default=None, help="append the report under a dated heading to this file (default: .cache/rack_dryness_census_<tag>.txt)")
    ap.add_argument("--batch", type=int, default=8)
    return ap.parse_args()


def compute_masked_claims(ckpt, rows_path, tests, batch=8):
    """Re-run the body's real two-consult cycle over every row and return, per (row, slot) that
    given_unique (or any test in `tests`) certifies at EITHER consult: the corrected masked numeral
    (consult-1's claim if it fired there, else consult-2's — the monotone union), which consult fired
    it (1 or 2), and the full re-derived dry set (for the consistency check against the sidecar)."""
    import phase1_algebra_head as H
    from phase1_algebra_head import (build_params, forward, build_slot_masks, alt2_fact_buf,
                                     rack_dry_row, rack_pack, certifier_bias,
                                     T_ALG, K_VARS, sent_indices, TOKENIZER_JSON,
                                     ALG_CERT, ALG_CERT_IMPLIED)
    from beacon_closing_arm import recompute_states
    from tinygrad import Tensor, dtypes
    from tinygrad.nn.state import safe_load
    from tokenizers import Tokenizer

    fixture = [json.loads(l) for l in open(rows_path)]
    n = len(fixture)
    tok = Tokenizer.from_file(TOKENIZER_JSON)
    p = build_params(0)
    sd = safe_load(ckpt)
    assert set(sd) == set(p), f"ckpt/head param mismatch: {set(sd) ^ set(p)}"
    for k in p:
        p[k].assign(sd[k].to(p[k].device).cast(p[k].dtype)).realize()
    KEYS = ("pres", "ftype", "op", "dig", "args", "res") + (("dup",) if "h_dup" in p else ())

    masked_value = {}      # (i, j) -> corrected numeral (the monotone union's claim)
    fired_consult = {}     # (i, j) -> 1 or 2 (which consult first certified it)
    reproduced_dry = set()  # (i, j) re-derived as dry by this re-run, any test in `tests`

    t_wall = __import__("time").time()
    for s0 in range(0, n, batch):
        sl = np.arange(s0, min(s0 + batch, n))
        pad = batch - len(sl)
        sl_p = np.concatenate([sl, sl[:1].repeat(pad)]) if pad else sl
        texts = [fixture[int(i)]["text"] for i in sl_p]
        ids = np.zeros((len(sl_p), T_ALG), np.int32)
        msk = np.zeros((len(sl_p), T_ALG), np.float32)
        snt = np.zeros((len(sl_p), T_ALG), np.int32)
        for bi, t in enumerate(texts):
            e = tok.encode(t); L = min(len(e.ids), T_ALG)
            ids[bi, :L] = e.ids[:L]; msk[bi, :L] = 1.0
            snt[bi] = sent_indices(t, list(e.offsets), msk[bi])
        sts = recompute_states(ids)
        ts = Tensor(np.ascontiguousarray(sts), dtype=dtypes.half)
        tk = Tensor(msk, dtype=dtypes.float)
        se = Tensor(snt.astype(np.int32), dtype=dtypes.int)
        nv = np.array([fixture[int(i)].get("n_vars", K_VARS) for i in sl_p])
        ma = np.array([fixture[int(i)].get("m", 300) for i in sl_p])

        o0 = forward(p, ts, tk, se)
        onp0 = {k: o0[k].realize().numpy() for k in ("fat", "args", "res")}
        mk = build_slot_masks(onp0, snt.astype(np.int32))
        mk_t = Tensor(mk, dtype=dtypes.float)

        # CONSULT 1 (breath 2): THE SAME forward the live rack commit sees, stop_after=2
        oa3 = forward(p, ts, tk, se, slot_mask=mk_t, stop_after=2)
        onp3 = {k: oa3[k].realize().numpy() for k in KEYS}
        fb3 = alt2_fact_buf(onp3, snt.astype(np.int32), nv, ma)
        f3_t = Tensor(fb3, dtype=dtypes.float)
        rows3 = [{k: onp3[k][bi] for k in onp3} for bi in range(len(sl_p))]
        c3_t = None
        if ALG_CERT or ALG_CERT_IMPLIED:
            c3_t = Tensor(certifier_bias(rows3, fb3, texts, T_ALG, ALG_CERT, ALG_CERT_IMPLIED), dtype=dtypes.float)
        r3_np = rack_pack(rows3, fb3, texts, T_ALG, tests, prev=None)
        r3_t = Tensor(r3_np, dtype=dtypes.float)
        d1_all = [rack_dry_row(rows3[bi], fb3[bi], texts[bi], T_ALG, tests) for bi in range(len(sl_p))]

        # CONSULT 2 (breath 4): fed consult 1's facts3/cert3/rack3 — loop_val.py's own pattern
        ob3 = forward(p, ts, tk, se, slot_mask=mk_t, stop_after=4, facts3=f3_t, cert3=c3_t, rack3=r3_t)
        onp4 = {k: ob3[k].realize().numpy() for k in KEYS}
        fb5 = alt2_fact_buf(onp4, snt.astype(np.int32), nv, ma)
        rows5 = [{k: onp4[k][bi] for k in onp4} for bi in range(len(sl_p))]
        d2_all = [rack_dry_row(rows5[bi], fb5[bi], texts[bi], T_ALG, tests) for bi in range(len(sl_p))]

        for bi, i in enumerate(sl):
            i = int(i)
            d1 = d1_all[bi]; d2 = d2_all[bi]
            for t in tests:
                for j, (v, spans) in d1[t].items():
                    masked_value[(i, j)] = v; fired_consult[(i, j)] = 1; reproduced_dry.add((i, j))
                for j, (v, spans) in d2[t].items():
                    if (i, j) not in masked_value:   # consult 1's claim wins (the monotone union)
                        masked_value[(i, j)] = v; fired_consult[(i, j)] = 2
                    reproduced_dry.add((i, j))
        print(f"[numeral-audit] re-run batch {s0}-{min(s0 + batch, n)}/{n} ({__import__('time').time() - t_wall:.0f}s elapsed)", flush=True)

    return masked_value, fired_consult, reproduced_dry


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
    # THE OLD (VOID) SOURCE: the dump's RAW, possibly-unmasked final-breath digit argmax — kept ONLY
    # for the record (the "raw-dump (void)" column below), never used for a claim.
    value_of_raw = {}
    for t in recs:
        i, j = int(t[0]), int(t[1])
        pdig = t[11]
        value_of_raw[(i, j)] = int("".join(str(int(d)) for d in pdig))
    n_no_dump = sum(1 for c in cells if c not in value_of_raw)
    print(f"[numeral-audit] {n_no_dump}/{n_cells} committed cells have NO LV_DUMP entry at all", flush=True)

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

    def score(value_of):
        n_gm = n_lm = 0; n_missing = 0
        for (i, j) in cells:
            v = value_of.get((i, j))
            if v is None:
                n_missing += 1; continue
            if v in given_vals.get(i, ()):
                n_gm += 1
            if v in lit_vals.get(i, ()):
                n_lm += 1
        d = max(n_cells, 1)
        return dict(n_given_match=n_gm, n_lit_match=n_lm, n_missing=n_missing,
                   prec_given=n_gm / d, prec_lit=n_lm / d)

    old = score(value_of_raw)

    # THE FIX: re-run the body's real two-consult cycle to get the MASKED numeral per committed cell.
    print(f"[numeral-audit] re-running the body's two-consult cycle ({a.ckpt}) to re-derive the masked claim for {n_cells} committed cells...", flush=True)
    masked_value, fired_consult, reproduced_dry = compute_masked_claims(a.ckpt, a.rows, tuple(tests), batch=a.batch)
    new = score(masked_value)
    n_unresolved = sum(1 for c in cells if c not in masked_value)
    n_by_consult1 = sum(1 for c in cells if fired_consult.get(c) == 1)
    n_by_consult2 = sum(1 for c in cells if fired_consult.get(c) == 2)

    # THE REPRODUCTION CONSISTENCY CHECK: does the re-derived dry set match the sidecar's own
    # committed set exactly (same checkpoint, same maths — it should)?
    cells_set = set(cells)
    n_match = len(cells_set & reproduced_dry)
    n_sidecar_only = len(cells_set - reproduced_dry)
    n_repro_only = len(reproduced_dry - cells_set)

    hist = collections.Counter()   # PER-ROW HISTOGRAM — UNCHANGED (the sidecar's own dry flags, exactly as before)
    rows_with_dry = collections.Counter()
    for k in range(len(rows)):
        rows_with_dry[int(rows[k])] = int((dry[k, :L_FAC] > 0.5).sum())
    hist = collections.Counter(rows_with_dry.values())
    seen_rows = set(int(r) for r in rows)
    hist[0] += max(0, n_rows_fixture - len(seen_rows))
    max_dry_row = max(rows_with_dry, key=lambda r: rows_with_dry[r]) if rows_with_dry else None

    lines = []
    lines.append(f"THE NUMERAL-LEVEL AUDIT (CORRECTED) — {tag} ({os.path.basename(a.sidecar)} x {os.path.basename(a.dump)}; {n_cells} committed cells, the sidecar's exact UNION)")
    lines.append(f"  (positional precision for the same cells, from the dry census: 0.526 — re-confirmed CLEAN 2026-10-07, verdict_from_ps never touches a claimed numeral)")
    lines.append(f"  BUG FOUND + FIXED (2026-10-07): the OLD column below read the numeral from the dump's RAW/unmasked pdig (rack_chain_RK.sh's LV_DUMP invocation never set LV_LEGAL=num); THE NEW column re-derives it by re-running the body's real two-consult cycle and calling rack_dry_row directly, exactly as the live commit does.")
    lines.append("")
    lines.append(f"  OLD (raw-dump, VOID — never use this for a claim): given-match {old['n_given_match']}/{n_cells} = {old['prec_given']:.3f}; any-literal-match {old['n_lit_match']}/{n_cells} = {old['prec_lit']:.3f}; no-dump-entry {old['n_missing']}")
    lines.append(f"  NEW (corrected, masked re-derivation): given-match {new['n_given_match']}/{n_cells} = {new['prec_given']:.3f}; any-literal-match {new['n_lit_match']}/{n_cells} = {new['prec_lit']:.3f}; unresolved (neither consult's re-run fired there) {n_unresolved}")
    lines.append(f"  of the {n_cells} committed cells: {n_by_consult1} claimed at consult 1 (breath 2), {n_by_consult2} first claimed at consult 2 (breath 4) — the monotone union's own split")
    lines.append(f"  REPRODUCTION CONSISTENCY: the re-derived dry set vs. the sidecar's own committed set — match {n_match}/{n_cells} ({n_sidecar_only} sidecar-only, {n_repro_only} reproduction-only; 0/0 expected, same checkpoint + maths)")
    lines.append(f"  committed cells with NO LV_DUMP entry at all (the old column's separate miss class): {n_no_dump}/{n_cells}")
    lines.append(f"  per-row dry-slot count histogram (over all {n_rows_fixture} wild holdout rows; 0 = row never committed a slot) — UNCHANGED from the sidecar's own flags:")
    for k in sorted(hist):
        lines.append(f"    {k:2d} dry slots: {hist[k]:4d} rows")
    if max_dry_row is not None:
        lines.append(f"  max dry slots in one row: {rows_with_dry[max_dry_row]} (row {max_dry_row})")
    lines.append(f"  rows with >=1 dry slot: {sum(1 for v in rows_with_dry.values() if v >= 1)} / {n_rows_fixture}")
    lines.append("")
    old_gap = 0.90 - old["prec_given"]; new_gap = 0.90 - new["prec_given"]
    lines.append(f"VERDICT: the reported 0.686 is {'FULLY' if new['prec_given'] >= 0.90 else ('MOSTLY' if new_gap < 0.3 * old_gap else 'PARTIALLY' if new_gap < old_gap else 'NOT')} explained by the masking bug "
                 f"(old {old['prec_given']:.3f} -> new {new['prec_given']:.3f}; gap to 0.90 shrinks {old_gap:.3f} -> {new_gap:.3f})")

    txt = "\n".join(lines)
    print(txt)

    from datetime import datetime
    stamp = os.popen("date '+%Y-%m-%d %H:%M'").read().strip()
    heading = f"\n\n### {stamp} — THE NUMERAL-LEVEL AUDIT, CORRECTED (the masking bug fixed; re-scored all {n_cells} committed cells)\n"
    with open(append_path, "a") as f:
        f.write(heading + txt + "\n")
    print(f"\n[numeral-audit] appended -> {append_path}")


if __name__ == "__main__":
    main()
