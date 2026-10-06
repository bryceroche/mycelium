"""rack_dryness_census.py — THE DRYNESS CENSUS (2026-10-05, zero GPU; ledger 16:13 "WORD GIVEN: THE RACK").

ADMISSION OF A DRYNESS TEST (pinned before this census was run): a test enters THE RACK's commit only
if its PRECISION on right slots — P(the slot's decoded factor is right | the test fires) — is >= 0.90
on the banked wild dumps; a dish "dry" one time in ten poisons the row. The census is banked
whichever way it falls (.cache/rack_dryness_census_<tag>.txt).

THE TESTS (scripts/phase1_algebra_head.py: RACK_TESTS_ALL / rack_dry_row — ONE implementation, the
one the head's rack3/rack5 ports are built from; a check must call its organ):
  given_unique          (a) a given slot whose decoded numeral occurs ONCE in the text and no other
                            decoded slot claims it (a text-side certificate)
  relation_implied      (b) a relation slot whose result variable the solver forced to a value the
                            text carries, that variable stated by NO decoded given (derived)
  relation_implied_any  (b') as (b) without the derived condition
  given_any             (c) the baseline: the numeral appears in the text at all, collisions allowed
  judge                 (d) the row-level consistency judge (solved + unique; combined_oracle.py's own
                            doors) as the known-safe reference: P(row's answer == key | solved+unique)

RIGHT per slot = the banked masked read's own per-slot verdict (.cache/ps_legal_wild_<tag>.npz: the
positional, LV_LEGAL=num `ok` loop_val writes — the ledger's "right slot"); a decoded slot with no
gold factor at that position is WRONG. Gold comes from the read's own comparison against the
fixture's g_* arrays (custody_gold.py's door: the harvest key, never a pen field); this script never
reads a solution field. Without a ps file (the diet dump) the verdict is recomputed loop_val's way
from --gold-npz (the fixture's states npz, g_* arrays) under the numeral mask.

DUMP FORMATS (auto-detected):
  rawslots_*.pkl   list of per-row dicts (i, text, q, pres, ftype, op, dig, args, res, dup, key) —
                   every slot's RAW heads (chain_acc's CA_RAWDUMP): the EXACT inputs the live rack sees
                   (modulo the breath: the consult decodes at breath 2 / 4, the dump at the last).
  dump_*.pkl       loop_val's LV_DUMP: one tuple per GOLD slot (i, j, gold..., pred argmaxes). Heads
                   are RECONSTRUCTED as one-hot logits for the listed slots only (slots the model
                   decoded where gold has none are invisible here; the facts' confidence gates pass by
                   construction) — an approximation, stated in the report.
  <dump>.rack.npz  the SIDECAR loop_val writes under ALG_RACK=1 beside LV_DUMP: the dry flags the
                   model ACTUALLY committed (rack5, the last consult's union) per test. When present,
                   THE DRY CENSUS (the arm's bar: share of slots committed by the last breath + their
                   precision) is reported EXACTLY from it, beside the recomputed proxy.

FACTS: alt2_fact_buf (the head's own vectorized commit + alternator_bridge.ping, the consult's exact
path) on the dump's heads — the solver's forced values per variable, walled per row.

usage: rack_dryness_census.py <dump.pkl> [--tag TAG] [--rows fixture.jsonl] [--ps ps.npz]
                              [--gold-npz states.npz] [--out path] [--no-judge] [--workers N]
Family env: set here by default (DEV=CPU + the N_DIG=7 / 9-way ftype family; the chain calls this
script bare) — an inherited env wins, and the dump's digit width is asserted against the head's.
"""
import os
import sys
import json
import time
import pickle
import argparse
import collections

for _k, _v in (("DEV", "CPU"), ("ALG2", "1"), ("ALG_FTYPES", "9"), ("ALG_DUP", "1"), ("ALG_WIDE", "1"),
               ("ALG_HW", "512"), ("ALG_FACTS_ROW_WALL", "5")):
    os.environ.setdefault(_k, _v)
sys.path.insert(0, "."); sys.path.insert(0, "scripts")
import numpy as np


def _parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("dump")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--rows", default=None, help="the fixture jsonl (texts, n_vars, m, query_var); default: wild_admitted_holdout")
    ap.add_argument("--ps", default=None, help="per-slot verdict npz (rows, slots, ok); default .cache/ps_legal_wild_<tag>.npz when it exists")
    ap.add_argument("--gold-npz", default=None, help="fixture states npz with g_* arrays (the verdict recomputed loop_val's way; raw dumps only)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--no-judge", action="store_true", help="skip the row-level consistency judge (the solver pool)")
    ap.add_argument("--workers", type=int, default=4)
    return ap.parse_args()


def _tag_of(path):
    b = os.path.basename(path).replace(".pkl", "")
    for pre in ("rawslots_wild_", "dump_wild_", "rawslots_", "dump_"):
        if b.startswith(pre):
            return b[len(pre):]
    return b


# ----------------------------------------------------------------------------------------------------------
# the dump -> per-row heads (the consult's `onp` convention), texts, q, key
# ----------------------------------------------------------------------------------------------------------
def load_raw(recs, H):
    rows = []
    for r in recs:
        row = {k: np.asarray(r[k], dtype=np.float32) for k in ("pres", "ftype", "op", "dig", "args", "res") + (("dup",) if "dup" in r else ())}
        rows.append(dict(i=int(r["i"]), text=r["text"], q=int(np.asarray(r["q"]).argmax()), key=(int(r["key"]) if r.get("key") is not None else None), heads=row))
    return rows


def load_tuples(recs, fixture, H):
    """LV_DUMP tuples -> one-hot heads for the listed (gold) slots; absent slots pres = -6 (never decoded)."""
    by_row = collections.defaultdict(list)
    for t in recs:
        by_row[int(t[0])].append(t)
    nft = int(os.environ.get("ALG_FTYPES", "9")); nd = H.N_DIG; L = H.L_FAC; K = H.K_VARS
    rows = []
    for i in sorted(by_row):
        heads = dict(pres=np.full(L, -6.0, np.float32), ftype=np.full((L, nft), -6.0, np.float32), op=np.full((L, 2), -6.0, np.float32),
                     dig=np.full((L, nd, 10), -6.0, np.float32), args=np.full((L, K), -6.0, np.float32), res=np.full((L, K), -6.0, np.float32),
                     dup=np.full(L, -1.0, np.float32))
        for (i_, j, gft, gop, gargs, gres, gdig, pft, pop, pargs, pres_, pdig, ppres, pdup) in by_row[i]:
            heads["pres"][j] = 6.0 if ppres else -6.0
            heads["ftype"][j, pft] = 6.0; heads["op"][j, pop] = 6.0; heads["res"][j, pres_] = 6.0
            for d, dd in enumerate(pdig):
                heads["dig"][j, d, dd] = 6.0
            heads["args"][j, pargs[0]] = 6.0
            if len(pargs) > 1:
                heads["args"][j, pargs[1]] = 5.0
            heads["dup"][j] = 1.0 if pdup else -1.0
        fx = fixture[i]
        rows.append(dict(i=i, text=fx["text"], q=int(fx["query_var"]), key=None, heads=heads))
    return rows


# ----------------------------------------------------------------------------------------------------------
# RIGHT per slot: the ps npz, or loop_val's positional comparison recomputed from g_* under the numeral mask
# ----------------------------------------------------------------------------------------------------------
def verdict_from_ps(path):
    z = np.load(path)
    ok = {(int(r), int(c)): bool(o) for r, c, o in zip(z["rows"], z["slots"], z["ok"])}
    return ok


def verdict_from_tuples(recs):
    """loop_val's `ok` recomputed from the LV_DUMP tuple's own fields (gold vs pred argmaxes)."""
    ok = {}
    for (i, j, gft, gop, gargs, gres, gdig, pft, pop, pargs, pres_, pdig, ppres, pdup) in recs:
        good = ppres and pft == gft and pres_ == gres
        if gft == 0:
            gset = set(gargs)
            good = good and pop == gop and ((len(gset) == 1 and pdup and pargs[0] in gset) or (len(gset) == 2 and set(pargs) == gset))
        else:
            good = good and list(pdig) == list(gdig)
        ok[(int(i), int(j))] = bool(good)
    return ok


def verdict_from_gold(rows, gz, H):
    from mycelium.rulebook import legal_digit_logits
    ok = {}
    for r in rows:
        i = r["i"]; onp = r["heads"]; text = r["text"]
        for j in range(H.L_FAC):
            if gz["g_presence"][i, j] < 0.5:
                continue
            dig = onp["dig"][j]
            if gz["g_ftype"][i, j] != 0 and int(onp["ftype"][j].argmax()) != 0:
                fake = legal_digit_logits(dig, text)
                if fake is not None:
                    dig = fake
            f_pres = bool(onp["pres"][j] > 0)
            f_ftype = int(onp["ftype"][j].argmax()) == int(gz["g_ftype"][i, j])
            f_res = int(onp["res"][j].argmax()) == int(gz["g_res"][i, j])
            good = f_pres and f_ftype and f_res
            if gz["g_ftype"][i, j] == 0:
                gset = set(np.where(gz["g_args"][i, j] > .5)[0].tolist())
                f_op = int(onp["op"][j].argmax()) == int(gz["g_op"][i, j])
                if len(gset) == 1 and "dup" in onp:
                    f_args = bool(onp["dup"][j] > 0) and int(np.argmax(onp["args"][j])) in gset
                else:
                    f_args = set(np.argsort(-onp["args"][j])[:2].tolist()) == gset
                good = good and f_op and f_args
            else:
                good = good and bool((dig.argmax(-1) == gz["g_digits"][i, j]).all())
            ok[(i, j)] = bool(good)
    return ok


# ----------------------------------------------------------------------------------------------------------
# (d) the row-level consistency judge: combined_oracle's own doors (solve + certify_unique), spawn pool
# ----------------------------------------------------------------------------------------------------------
def judge_rows(rows, parses, keys, workers):
    import multiprocessing as mp
    from combined_oracle import _solve_task, _uniqueness_task
    from sinkhorn_claim import build_gv_nvv
    tasks = []
    for r, parse in zip(rows, parses):
        if keys.get(r["i"]) is None or not parse:
            continue
        gv, nvv = build_gv_nvv(parse, r["q"])
        tasks.append((r["i"], r["q"], keys[r["i"]], parse, gv, nvv))
    ctx = mp.get_context("spawn")
    t0 = time.time()
    with ctx.Pool(workers) as pool:
        solved = {tid: (st, val, m) for tid, st, val, m in pool.imap_unordered(_solve_task, tasks)}
    print(f"[rack-census] judge: {len(tasks)} rows solved in {time.time()-t0:.0f}s", flush=True)
    utasks = [(tid, q, parse, gv, nvv, solved[tid][2], solved[tid][1]) for (tid, q, key, parse, gv, nvv) in tasks if solved.get(tid, ("?",))[0] == "solved"]
    t0 = time.time()
    with ctx.Pool(workers) as pool:
        uniq = {tid: u for tid, u in pool.imap_unordered(_uniqueness_task, utasks)}
    print(f"[rack-census] judge: {len(utasks)} solved rows certified in {time.time()-t0:.0f}s", flush=True)
    out = {}
    for (tid, q, key, parse, gv, nvv) in tasks:
        st, val, m = solved.get(tid, ("?", None, None))
        out[tid] = dict(status=st, value=val, unique=(uniq.get(tid) is True), correct=(st == "solved" and val == key))
    return out


def main():
    a = _parse_args()
    import phase1_algebra_head as H
    from phase1_algebra_head import rack_dry_row, RACK_TESTS_ALL, _decode_slots, alt2_fact_buf
    from mycelium.rulebook import legal_digit_logits
    tag = a.tag or _tag_of(a.dump)
    out_path = a.out or f".cache/rack_dryness_census_{tag}.txt"
    rows_path = a.rows or ".cache/wild_admitted_holdout.jsonl"
    fixture = [json.loads(l) for l in open(rows_path)]
    recs = pickle.load(open(a.dump, "rb"))
    assert recs, f"empty dump {a.dump}"
    raw = isinstance(recs[0], dict)
    if raw:
        assert H.N_DIG == np.asarray(recs[0]["dig"]).shape[1], (H.N_DIG, np.asarray(recs[0]["dig"]).shape)
        rows = load_raw(recs, H)
    else:
        assert H.N_DIG == len(recs[0][6]), (H.N_DIG, len(recs[0][6]))
        rows = load_tuples(recs, fixture, H)
    keys = {}
    for r in rows:
        fx = fixture[r["i"]]
        from mycelium.custody_gold import row_gold   # the custody door: the harvest/gsm8k key by TEXT identity; never a pen field
        _k = int(row_gold(fx))
        assert r["key"] is None or r["key"] == _k, f"row {r['i']}: the dump's key {r['key']} != the custody key {_k}"
        r["key"] = _k; keys[r["i"]] = _k
        assert fx["text"] == r["text"], f"row {r['i']}: the dump's text is not the fixture's (wrong --rows?)"
    ps_path = a.ps or f".cache/ps_legal_wild_{tag}.npz"
    if os.path.exists(ps_path):
        ok = verdict_from_ps(ps_path); verdict_src = ps_path
    elif not raw:
        # an LV_DUMP without its ps twin: the dump's own gold comparison (loop_val's positional read on the
        # OPEN digits it dumped — exact for the open read it came from, not the masked one)
        ok = verdict_from_tuples(recs); verdict_src = a.dump + " (the dump's own gold comparison, open digits)"
    else:
        assert a.gold_npz, f"no per-slot verdict: {ps_path} missing and no --gold-npz (raw dump)"
        ok = verdict_from_gold(rows, np.load(a.gold_npz), H); verdict_src = a.gold_npz + " (recomputed, loop_val's positional comparison under the numeral mask)"
    gold_present = set(ok.keys())
    gz_path = a.gold_npz or (".cache/phase1_alg_states_wildhold.npz" if "wild" in rows_path else None)
    gz = np.load(gz_path) if (gz_path and os.path.exists(gz_path)) else None   # THE AUDIT COLUMNS' gold (g_* arrays): what the certificate certifies vs what the positional read scores
    if gz is not None:
        assert gz["g_presence"].shape[0] == len(fixture), (gz["g_presence"].shape, len(fixture))
    n = len(rows)
    print(f"[rack-census] {tag}: {n} rows from {a.dump} ({'raw heads' if raw else 'LV_DUMP tuples, heads reconstructed'}); verdict from {verdict_src}", flush=True)

    # the facts: the consult's own path (alt2_fact_buf -> ping), batched over the rows
    onp = {k: np.stack([r["heads"][k] for r in rows]) for k in rows[0]["heads"]}
    nv = np.array([fixture[r["i"]].get("n_vars", H.K_VARS) for r in rows]); ma = np.array([fixture[r["i"]].get("m", 300) for r in rows])
    t0 = time.time()
    facts = alt2_fact_buf(onp, np.zeros((n, H.T_ALG), np.int32), nv, ma)
    print(f"[rack-census] facts: {float((facts[..., 0].sum(1) > 0).mean()):.3f} of rows with a known variable, {facts[..., 0].sum(1).mean():.2f} known/row ({time.time()-t0:.0f}s)", flush=True)

    # the dryness tests, per row; decoded-present slots counted the way the rack sees them (the masked decode)
    tests = RACK_TESTS_ALL
    fired = {t: [] for t in tests}            # (i, j, value)
    n_decoded = n_decoded_given = n_decoded_rel = 0
    parses = []
    for bi, r in enumerate(rows):
        d = rack_dry_row(r["heads"], facts[bi], r["text"], H.T_ALG, tests)
        masked = dict(r["heads"]); masked["dig"] = r["heads"]["dig"].copy()
        for j in range(H.L_FAC):
            if int(masked["ftype"][j].argmax()) != 0:
                fake = legal_digit_logits(masked["dig"][j], r["text"])
                if fake is not None:
                    masked["dig"][j] = fake
        parse = _decode_slots(masked); parses.append(parse)
        n_decoded += len(parse); n_decoded_given += sum(f["ftype"] == "given" for f in parse); n_decoded_rel += sum(f["ftype"] == "rel" for f in parse)
        for t in tests:
            for j, (v, spans) in d[t].items():
                fired[t].append((r["i"], j, v))
    lines = [f"THE DRYNESS CENSUS — {tag} ({os.path.basename(a.dump)}; {n} rows; {'raw heads' if raw else 'LV_DUMP tuples: heads reconstructed one-hot for gold slots only — decoded-but-gold-absent slots invisible (precision optimistic)'})",
             f"verdict (RIGHT per slot): {verdict_src}",
             f"decoded present slots {n_decoded} (given {n_decoded_given}, rel {n_decoded_rel}); gold slots {len(gold_present)}; facts known/row {facts[..., 0].sum(1).mean():.2f}",
             "", f"{'test':24s} {'fires':>6s} {'right':>6s} {'PRECISION':>10s} {'cov/decoded':>12s} {'cov/gold':>9s} {'rows':>5s}  admitted(>=0.90)"]
    summary = {}
    lines[-1] += "   | AUDIT: value@slot  value-in-gold" if gz is not None else ""
    def _gval(i, j):
        return int("".join(str(int(x)) for x in gz["g_digits"][i, j]))
    for t in tests:
        F = fired[t]; nf = len(F); nr = sum(ok.get((i, j), False) for i, j, _ in F)
        pool = n_decoded_given if t.startswith("given") else n_decoded_rel
        prec = nr / nf if nf else float("nan")
        summary[t] = (nf, nr, prec)
        audit = ""
        if gz is not None and nf:
            # value@slot: the gold slot j is a literal carrying this very numeral (the binding right, whatever the rest of the factor);
            # value-in-gold: SOME gold literal on the row carries it (content right, order free — the matched read's criterion)
            v_at = sum(1 for i, j, v in F if gz["g_presence"][i, j] > 0.5 and int(gz["g_ftype"][i, j]) != 0 and _gval(i, j) == v)
            v_in = sum(1 for i, j, v in F if any(gz["g_presence"][i, jj] > 0.5 and int(gz["g_ftype"][i, jj]) != 0 and _gval(i, jj) == v for jj in range(H.L_FAC)))
            audit = f"   | {v_at / nf:10.3f} {v_in / nf:13.3f}"
        lines.append(f"{t:24s} {nf:6d} {nr:6d} {prec:10.3f} {nf / max(pool, 1):12.3f} {nf / max(len(gold_present), 1):9.3f} {len({i for i, _, _ in F}):5d}  {'YES' if (nf and prec >= 0.90) else 'no':3s}{audit}")
    # the union the default commit would make (given_unique + relation_implied), per row coverage
    adm = [t for t in ("given_unique", "relation_implied") if summary[t][0] and summary[t][2] >= 0.90]
    U = {(i, j) for t in adm for i, j, _ in fired[t]}
    if U:
        lines.append(f"{'UNION ' + '+'.join(adm):24s} {len(U):6d} {sum(ok.get(c, False) for c in U):6d} {sum(ok.get(c, False) for c in U) / len(U):10.3f} {len(U) / max(n_decoded, 1):12.3f} {len(U) / max(len(gold_present), 1):9.3f} {len({i for i, _ in U}):5d}  (the commit: slots dry by the last consult, recomputed from this dump's decode)")
    # wrong dry slots: what poisons (per test, the first few)
    for t in tests:
        bad = [(i, j, v) for i, j, v in fired[t] if not ok.get((i, j), False)][:6]
        if bad:
            lines.append(f"  wrong-but-dry {t}: " + "; ".join(f"row {i} slot {j} value {v}{' (no gold slot)' if (i, j) not in gold_present else ''}" for i, j, v in bad))
    # the sidecar: the flags the model actually committed (exact DRY CENSUS)
    side = a.dump + ".rack.npz"
    if os.path.exists(side):
        z = np.load(side, allow_pickle=True)
        srows = z["rows"]; dry = z["dry"]; names = [str(x) for x in z["tests"]]; per = z["per_test"]
        lines.append(""); lines.append(f"THE DRY CENSUS (EXACT, from the sidecar {os.path.basename(side)}: the flags the body committed at its last consult)")
        for ti, t in enumerate(names + ["UNION"]):
            fl = per[:, :, ti] if t != "UNION" else dry
            cells = [(int(srows[k]), j) for k in range(len(srows)) for j in range(fl.shape[1]) if fl[k, j] > 0.5]
            nr = sum(ok.get(c, False) for c in cells)
            lines.append(f"  {t:22s} committed {len(cells):5d} ({len(cells) / max(len(gold_present), 1):.3f} of gold slots, {len(cells) / max(n_decoded, 1):.3f} of decoded) precision {nr / max(len(cells), 1):.3f}  {'BAR MET' if (cells and nr / len(cells) >= 0.90) else ('KILL (< 0.80)' if (cells and nr / len(cells) < 0.80) else 'bar missed')}")
    else:
        lines.append(""); lines.append("(no sidecar <dump>.rack.npz: the DRY CENSUS above is the proxy recomputed from the dump's final decode; a body read under ALG_RACK=1 with LV_DUMP writes the exact one)")
    # (d) the judge
    if not a.no_judge:
        J = judge_rows(rows, parses, keys, a.workers)
        su = [i for i, j in J.items() if j["status"] == "solved" and j["unique"]]
        so = [i for i, j in J.items() if j["status"] == "solved"]
        lines.append("")
        lines.append(f"{'judge: solved+unique':24s} {len(su):6d} {sum(J[i]['correct'] for i in su):6d} {sum(J[i]['correct'] for i in su) / max(len(su), 1):10.3f}  (row-level; {len(so)} solved, of which correct {sum(J[i]['correct'] for i in so)}; top-1 correct {sum(j['correct'] for j in J.values())}/{len(J)})")
    lines.append(f"ADMITTED (precision >= 0.90, non-empty): {', '.join(adm) if adm else 'NONE'}   [relation_implied_any {'YES' if (summary['relation_implied_any'][0] and summary['relation_implied_any'][2] >= 0.90) else 'no'}; given_any {'YES' if (summary['given_any'][0] and summary['given_any'][2] >= 0.90) else 'no'}]")
    txt = "\n".join(lines) + "\n"
    open(out_path, "w").write(txt); print(txt); print(f"[rack-census] -> {out_path}")


if __name__ == "__main__":
    main()
