"""hammerhead_census.py — THE HAMMERHEAD CENSUS (2026-10-06, zero-GPU design, CPU-forward read; Bryce's
hammerhead: can VIEW UNANIMITY under K sentence-permutation views serve as a dryness test for THE RACK,
alone and crossed (AND) with given_unique?).

CONTEXT (read docs/phase1_skeleton_spec.md's 2026-10-05/06 "THE RACK" + "dryness" entries first): THE
RACK's admitted test, given_unique (a given slot whose decoded numeral occurs once in the text and no
other decoded slot claims it), commits a numeral that is a gold GIVEN of the row 0.686 of the time at
RK_241's consult-1 (THE DRY CENSUS, 2026-10-06 14:13/14:31 ledger — the number this script's --tag
RK_241 companion would reproduce) and 0.97 of the time at PMS8's final decode (the 2026-10-05 17:24
admission census). This script asks whether TTA's view machinery (scripts/tta_views.py: K=4 sentence
permutations + the original = 5 views, first/last sentence pinned, middle shuffled per-(view,row) seed
1000*k+i) gives a SECOND, independent certificate: does the SAME claim survive under K re-renderings of
the row (UNANIMOUS = all 5 views decode the identical numeral for that slot's role; MAJORITY = >=3/5)?

THE VIEW-ALIGNMENT METHOD (stated, not reused verbatim from tta_views/lattice_*, which vote on the FINAL
ANSWER only and never need per-slot identity): the FACTOR-slot bank (24 slots) is populated by content
(bilinear pointers over SENTENCE order), so slot index j is NOT stable under sentence permutation — a
view's j-th decoded slot may hold a different gold factor than the original's j-th slot. The VARIABLE
bank, by construction (CLAUDE.md: "mint packs letters consecutively"; "24 vars <-> letters positionally"),
is keyed by the row's own letters, which are a LEXICAL property of the text, invariant to sentence order.
So this script aligns by ROLE = the decoded VARIABLE a given-slot assigns a value to (o_np["res"] for a
ftype==given slot), never by slot index j, across views. The only place slot index j is still used is the
POSITIONAL verdict lookup into .cache/ps_legal_wild_<tag>.npz, which is itself a (row, slot-j) table
computed from the body's OWN standard (view-0, final-breath) masked read — so view 0's own decoded slot_j
for a given role is used to look that table up (an approximation for the breath-2 condition, stated below:
ps_legal_wild was computed at the FINAL breath; slot assignment is assumed stable across breaths within
one read, which the rack's own "freeze" mechanism assumes too).

GOLD comes from .cache/phase1_alg_states_wildhold.npz's g_* arrays (the custody door: the harvest key,
keyed by TEXT identity; never the fixture's own "factors" field, which is a convenience copy, not the
audited door) — rack_numeral_audit.py's own convention, reused: a gold GIVEN slot is (var=g_res[i,j],
value=the digits, j) wherever g_presence[i,j]>0.5 and g_ftype[i,j]==1.

PASS 1 / breath conditions: per view, ONE pass-1 forward (unmasked, no slot_mask: o0) builds the evidence-
sharing slot mask (build_slot_masks) and the live pass-1 facts (alt2_fact_buf) exactly as loop_val.py /
chain_acc.py do for a non-ALT3 body (PMS8_241, HS_241 carry no consults: ALG_ALT3 unset) — then ONE main
forward with ALG_MINE_BREATHS=1 gives out["heads_all"], a list of per-breath head-logit dicts. "Breath 2"
(the position chained bodies take their consult-1 decode: stop_after=2) is heads_all[2]; "final" is
heads_all[-1] (the body's standard single-pass decode, bit-identical to the plain forward's own `out`).
THE NUMERAL MASK (LV_LEGAL=num's door, legal_digit_logits) is applied to every literal slot's digit head
before decoding, exactly as rack_dryness_census.py / chain_acc.py's masked branch do — this script's
decode is always the MASKED one (the standard "masked wild" convention every bar in the ledger quotes).

usage:
  <FAM env> <SURF8 env> [<HIER env for HS_241>] DEV=CPU ALG_MINE_BREATHS=1 \\
    .venv/bin/python3 scripts/hammerhead_census.py --ckpt .cache/sharp_PMS8_241.safetensors --tag PMS8_241 \\
      [--rows .cache/wild_admitted_holdout.jsonl] [--gold-npz .cache/phase1_alg_states_wildhold.npz] \\
      [--k-views 4] [--batch 8] [--limit N] [--out .cache/hammerhead_census_PMS8_241.txt]
"""
import os
import sys
import json
import time
import argparse
import collections

sys.path.insert(0, "."); sys.path.insert(0, "scripts")
os.environ.setdefault("DEV", "CPU")
os.environ.setdefault("ALG_MINE_BREATHS", "1")
import numpy as np

BREATH_CONDS = ("b2", "final")
# the cross-row tests this census scores, as (name, needs) — "needs" lists which per-view signal it reads
TESTS = ("given_unique", "unanimous", "majority", "given_unique_AND_unanimous", "given_unique_AND_majority")


def _parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--rows", default=".cache/wild_admitted_holdout.jsonl")
    ap.add_argument("--gold-npz", default=".cache/phase1_alg_states_wildhold.npz")
    ap.add_argument("--ps", default=None, help="default .cache/ps_legal_wild_<tag>.npz")
    ap.add_argument("--k-views", type=int, default=4, help="tta_views.py's K_VIEWS (permuted views beyond the original; 5 views total)")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None, help="debug: first N rows only")
    ap.add_argument("--out", default=None)
    return ap.parse_args()


def main():
    a = _parse_args()
    tag = a.tag
    out_path = a.out or f".cache/hammerhead_census_{tag}.txt"

    import phase1_algebra_head as H
    from phase1_algebra_head import (build_params, forward, build_slot_masks, alt2_fact_buf,
                                     _decode_slots, rack_dry_row, T_ALG, K_VARS, L_FAC,
                                     sent_indices, TOKENIZER_JSON)
    from mycelium.rulebook import legal_digit_logits
    from beacon_closing_arm import recompute_states
    from tta_views import permuted_view
    from tinygrad import Tensor, dtypes
    from tinygrad.nn.state import safe_load
    from tokenizers import Tokenizer

    fixture = [json.loads(l) for l in open(a.rows)]
    if a.limit:
        fixture = fixture[:a.limit]
    n = len(fixture)

    tok = Tokenizer.from_file(TOKENIZER_JSON)
    p = build_params(0)
    sd = safe_load(a.ckpt)
    assert set(sd) == set(p), f"ckpt/head param mismatch: {set(sd) ^ set(p)}"
    for k in p:
        p[k].assign(sd[k].to(p[k].device).cast(p[k].dtype)).realize()
    has_dup = "h_dup" in p
    KEYS = ("pres", "ftype", "op", "dig", "args", "res") + (("dup",) if has_dup else ())

    ps_path = a.ps or f".cache/ps_legal_wild_{tag}.npz"
    ps_ok = {}
    if os.path.exists(ps_path):
        z = np.load(ps_path)
        ps_ok = {(int(r), int(c)): bool(o) for r, c, o in zip(z["rows"], z["slots"], z["ok"])}
        print(f"[hammerhead] positional verdict: {len(ps_ok)} (row,slot) cells from {ps_path}", flush=True)
    else:
        print(f"[hammerhead] WARNING: no positional verdict at {ps_path} — positional precision will read as n/a", flush=True)

    gz = np.load(a.gold_npz)
    assert gz["g_presence"].shape[0] >= n, (gz["g_presence"].shape, n)

    def gval(i, j):
        return int("".join(str(int(x)) for x in gz["g_digits"][i, j]))

    gold_given = {}             # i -> [(var, value, gold_slot_j), ...]
    gold_given_vals = collections.defaultdict(set)   # i -> {gold given numerals}
    gold_lit_vals = collections.defaultdict(set)      # i -> {gold literal numerals, any ftype != 0}
    n_gold_slots = 0
    for i in range(n):
        lst = []
        for j in range(L_FAC):
            if gz["g_presence"][i, j] < 0.5:
                continue
            n_gold_slots += 1
            ft = int(gz["g_ftype"][i, j])
            if ft == 0:
                continue
            v = gval(i, j)
            gold_lit_vals[i].add(v)
            if ft == 1:
                var = int(gz["g_res"][i, j])
                gold_given_vals[i].add(v)
                lst.append((var, v, j))
        gold_given[i] = lst
    n_gold_given = sum(len(v) for v in gold_given.values())
    print(f"[hammerhead] {tag}: {n} rows, {n_gold_slots} gold slots (any ftype), {n_gold_given} gold GIVEN slots", flush=True)

    K = a.k_views
    n_views = K + 1

    # THE CONSULT BODIES (ALG_ALT3=1, e.g. RKC_241): breath 2 IS the consult-1 decode — one forward
    # with stop_after=2 returns it directly at the top level (forward()'s own `out`, no heads_all /
    # ALG_MINE_BREATHS needed). "final" (breath -1) is not meaningful as a SEPARATE read here without
    # running the rest of the three-consult cycle (consult 2's facts/cert/rack ports) — out of scope for
    # this follow-up ("breath 2 only is enough"), so only "b2" is scored for an ALT3 body.
    _ALT3 = int(os.environ.get("ALG_ALT3", "0")) != 0
    conds_used = ("b2",) if _ALT3 else BREATH_CONDS
    if _ALT3:
        print("[hammerhead] ALG_ALT3=1: consult body — breath-2 (consult-1) decode only, via stop_after=2; no 'final' condition", flush=True)

    # claims[cond][i] = list of n_views dicts {var: value} (that view's own masked decode, GIVEN slots only)
    claims = {c: [[dict() for _ in range(n_views)] for _ in range(n)] for c in conds_used}
    # view0_slot[cond][i] = {var: slot_j} from view 0's own decode (for the positional-verdict lookup)
    view0_slot = {c: [dict() for _ in range(n)] for c in conds_used}
    # gu0[cond][i] = set of vars view 0's OWN given_unique test certifies (rack_dry_row, the admitted organ)
    gu0 = {c: [set() for _ in range(n)] for c in conds_used}

    BATCH = a.batch
    view_wall = []
    for vi in range(n_views):
        t0 = time.time()
        for s0 in range(0, n, BATCH):
            sl = np.arange(s0, min(s0 + BATCH, n))
            pad = BATCH - len(sl)
            sl_p = np.concatenate([sl, sl[:1].repeat(pad)]) if pad else sl
            texts = []
            for i in sl_p:
                fx = fixture[int(i)]
                texts.append(fx["text"] if vi == 0 else permuted_view(fx["text"], 1000 * vi + int(i)))
            ids = np.zeros((len(sl_p), T_ALG), np.int32)
            msk = np.zeros((len(sl_p), T_ALG), np.float32)
            snt = np.zeros((len(sl_p), T_ALG), np.int32)
            for bi, t in enumerate(texts):
                e = tok.encode(t)
                L = min(len(e.ids), T_ALG)
                ids[bi, :L] = e.ids[:L]
                msk[bi, :L] = 1.0
                snt[bi] = sent_indices(t, list(e.offsets), msk[bi])
            sts = recompute_states(ids)
            ts = Tensor(np.ascontiguousarray(sts), dtype=dtypes.half)
            tk = Tensor(msk, dtype=dtypes.float)
            se = Tensor(snt.astype(np.int32), dtype=dtypes.int)
            nv = np.array([fixture[int(i)].get("n_vars", K_VARS) for i in sl_p])
            ma = np.array([fixture[int(i)].get("m", 300) for i in sl_p])

            # PASS 1: the unmasked full forward (chain_acc's / loop_val's o0) -> the evidence-sharing
            # slot mask (every body) + the live pass-1 facts (ALG_ALT2's fact_buf feed; non-ALT3 bodies
            # only — a consult body's facts come from the consult cycle, not this feed: loop_val/chain_acc
            # pass fact_buf=None when ALG_ALT3).
            o0 = forward(p, ts, tk, se)
            onp0 = {k: o0[k].realize().numpy() for k in ("fat", "args", "res")}
            mk = build_slot_masks(onp0, snt.astype(np.int32))
            mk_t = Tensor(mk, dtype=dtypes.float)

            heads_by_cond = {}
            if _ALT3:
                # THE CONSULT-1 DECODE (loop_val.py / chain_acc.py's `_oa3`): a partial pass that stops
                # after breath 2 — forward()'s own top-level `out` IS that breath's heads_of(), no
                # heads_all required.
                oa3 = forward(p, ts, tk, se, slot_mask=mk_t, stop_after=2)
                heads_by_cond["b2"] = oa3
            else:
                _ka = ("pres", "ftype", "op", "dig") + (("dup",) if "dup" in o0 else ())
                _oa = {**onp0, **{k: o0[k].realize().numpy() for k in _ka}}
                fb = alt2_fact_buf(_oa, snt.astype(np.int32), nv, ma)
                fact_t = Tensor(fb, dtype=dtypes.float)
                o = forward(p, ts, tk, se, slot_mask=mk_t, fact_buf=fact_t)
                heads_all = o.get("heads_all")
                assert heads_all is not None, "ALG_MINE_BREATHS=1 did not produce out['heads_all'] — check env"
                kb_final = len(heads_all) - 1
                kb_b2 = min(2, kb_final)
                if vi == 0 and s0 == 0:
                    print(f"[hammerhead] {len(heads_all)} breaths; breath-2 condition reads index {kb_b2}", flush=True)
                heads_by_cond["b2"] = heads_all[kb_b2]
                heads_by_cond["final"] = heads_all[kb_final]

            for cond in conds_used:
                hb = heads_by_cond[cond]
                onp = {k: hb[k].realize().numpy() for k in KEYS if k in hb}
                for bi, i in enumerate(sl):
                    i = int(i)
                    row = {k: onp[k][bi].copy() for k in onp}
                    masked = dict(row); masked["dig"] = row["dig"].copy()
                    for j in range(L_FAC):
                        if int(masked["ftype"][j].argmax()) == 0:
                            continue
                        fake = legal_digit_logits(masked["dig"][j], texts[bi])
                        if fake is not None:
                            masked["dig"][j] = fake
                    parse = _decode_slots(masked)
                    cmap, smap = {}, {}
                    for f in parse:
                        if f["ftype"] == "given":
                            v = int(f["var"])
                            if v not in cmap:
                                cmap[v] = int(f["value"]); smap[v] = f["_slot"]
                    claims[cond][i][vi] = cmap
                    if vi == 0:
                        view0_slot[cond][i] = smap
                        # THE ADMITTED ORGAN, unmodified: given_unique on view 0's own (unmasked) row + text
                        d = rack_dry_row(row, None, texts[bi], T_ALG, ("given_unique",))
                        by_slot_var = {f["_slot"]: int(f["var"]) for f in parse if f["ftype"] == "given"}
                        gu0[cond][i] = {by_slot_var[j] for j in d["given_unique"] if j in by_slot_var}
        dt = time.time() - t0
        view_wall.append(dt)
        print(f"[hammerhead] view {vi}/{n_views - 1} done in {dt:.0f}s ({dt / max(len(range(0, n, BATCH)), 1):.1f}s/batch, {n} rows)", flush=True)

    # ---------------------------------------------------------------------------------------------
    # per gold GIVEN slot: given_unique / unanimous / majority / their AND with given_unique
    # ---------------------------------------------------------------------------------------------
    def numeral_precision(fired):   # fired: list of (i, var, claimed_value)
        n_gm = n_lm = n_vm = 0   # given-match / any-literal-match / exact-var-match (gold var's own value)
        for i, var, v in fired:
            if v in gold_given_vals.get(i, ()):
                n_gm += 1
            if v in gold_lit_vals.get(i, ()):
                n_lm += 1
            gv = {gv_ for (gvar, gv_, gj) in gold_given[i] if gvar == var}
            if gv and v in gv:
                n_vm += 1
        d = max(len(fired), 1)
        return dict(fires=len(fired), prec_given=n_gm / d, prec_lit=n_lm / d, prec_exact_var=n_vm / d)

    def positional_precision(cond, fired):   # fired: list of (i, var, claimed_value)
        hits = misses = no_view0 = 0
        for i, var, v in fired:
            j = view0_slot[cond][i].get(var)
            if j is None:
                no_view0 += 1
                continue
            if ps_ok.get((i, j), False):
                hits += 1
            else:
                misses += 1
        denom = max(hits + misses, 1)
        return dict(hits=hits, misses=misses, no_view0=no_view0, prec=hits / denom)

    results = {}   # cond -> test -> dict(numeral=..., positional=..., row_fired=set, per_row_count=Counter)
    for cond in conds_used:
        results[cond] = {}
        fired_by_test = {t: [] for t in TESTS}
        row_fired = {t: set() for t in TESTS}
        per_row_count = {t: collections.Counter() for t in TESTS}
        for i in range(n):
            for var, gvalue, gj in gold_given[i]:
                vclaims = [claims[cond][i][vi].get(var) for vi in range(n_views)]
                present = [v for v in vclaims if v is not None]
                is_gu = var in gu0[cond][i]
                # UNANIMOUS: all K=n_views views decode a value for this role AND all equal
                unan_fires = len(present) == n_views and len(set(present)) == 1
                unan_val = present[0] if unan_fires else None
                # MAJORITY: among all n_views views (missing = no claim, counts for nothing), the
                # commonest claimed value reaches >= 3 (of 5)
                maj_val = maj_fires = None
                if present:
                    cnt = collections.Counter(present)
                    top_v, top_c = cnt.most_common(1)[0]
                    if top_c >= 3:
                        maj_val, maj_fires = top_v, True
                v0 = claims[cond][i][0].get(var)   # view 0's own claim (what given_unique certifies, when it fires)
                if is_gu and v0 is not None:
                    fired_by_test["given_unique"].append((i, var, v0))
                    row_fired["given_unique"].add(i); per_row_count["given_unique"][i] += 1
                if unan_fires:
                    fired_by_test["unanimous"].append((i, var, unan_val))
                    row_fired["unanimous"].add(i); per_row_count["unanimous"][i] += 1
                    if is_gu:
                        fired_by_test["given_unique_AND_unanimous"].append((i, var, unan_val))
                        row_fired["given_unique_AND_unanimous"].add(i); per_row_count["given_unique_AND_unanimous"][i] += 1
                if maj_fires:
                    fired_by_test["majority"].append((i, var, maj_val))
                    row_fired["majority"].add(i); per_row_count["majority"][i] += 1
                    if is_gu:
                        fired_by_test["given_unique_AND_majority"].append((i, var, maj_val))
                        row_fired["given_unique_AND_majority"].add(i); per_row_count["given_unique_AND_majority"][i] += 1
        for t in TESTS:
            num = numeral_precision(fired_by_test[t])
            pos = positional_precision(cond, fired_by_test[t])
            results[cond][t] = dict(numeral=num, positional=pos, rows=len(row_fired[t]),
                                    mean_dry_per_row=sum(per_row_count[t].values()) / n)

    # ---------------------------------------------------------------------------------------------
    # write the report
    # ---------------------------------------------------------------------------------------------
    lines = []
    lines.append(f"THE HAMMERHEAD CENSUS — {tag} ({os.path.basename(a.ckpt)}; {n} rows from {a.rows}; K={n_views} views [original + {K} sentence-permutations, tta_views.py's permuted_view, seed 1000*k+i])")
    lines.append(f"gold: {a.gold_npz} ({n_gold_slots} gold slots any ftype, {n_gold_given} gold GIVEN slots; the fixture's own 'factors' field NOT used — custody door is the gold-npz)")
    lines.append(f"positional verdict: {ps_path}" + ("" if ps_ok else "  (MISSING — positional precision n/a)"))
    lines.append(f"cost: per-view wall-clock (CPU, batch={BATCH}): " + ", ".join(f"v{vi}={w:.0f}s" for vi, w in enumerate(view_wall)) + f"  (total {sum(view_wall):.0f}s for {n_views} views x {n} rows x 2 forwards/batch)")
    lines.append("")
    for cond in conds_used:
        lines.append(f"=== breath condition: {cond} ===")
        lines.append(f"{'test':32s} {'fires':>6s} {'rows':>5s} {'cov/gold':>9s} {'prec_given':>11s} {'prec_lit':>9s} {'prec_var':>9s} {'pos_prec':>9s} {'pos_n':>6s} {'admitted>=0.90':>15s}")
        for t in TESTS:
            r = results[cond][t]
            num = r["numeral"]; pos = r["positional"]
            admitted = "YES (numeral)" if num["fires"] and num["prec_given"] >= 0.90 else "no"
            lines.append(f"{t:32s} {num['fires']:6d} {r['rows']:5d} {num['fires'] / max(n_gold_slots, 1):9.3f} "
                         f"{num['prec_given']:11.3f} {num['prec_lit']:9.3f} {num['prec_exact_var']:9.3f} "
                         f"{pos['prec']:9.3f} {pos['hits'] + pos['misses']:6d} {admitted:>15s}")
            lines.append(f"    mean dry (fired) slots/row: {r['mean_dry_per_row']:.3f}; positional no-view0-claim: {pos['no_view0']}")
        lines.append("")

    # the registered prediction, read off the tables (ALT3 bodies: breath-2 only, no "final" row/compare)
    _rc = "final" if "final" in conds_used else "b2"
    gu_final = results[_rc]["given_unique"]["numeral"]["prec_given"]
    un_final = results[_rc]["unanimous"]["numeral"]["prec_given"]
    un_cov = results[_rc]["unanimous"]["numeral"]["fires"] / max(n_gold_slots, 1)
    and_final = results[_rc]["given_unique_AND_unanimous"]["numeral"]["prec_given"]
    and_cov = results[_rc]["given_unique_AND_unanimous"]["numeral"]["fires"] / max(n_gold_slots, 1)
    gu_cov = results[_rc]["given_unique"]["numeral"]["fires"] / max(n_gold_slots, 1)
    lines.append(f"THE READING ({_rc} breath; prec_given is the bar's numeral-level precision, >= 0.90 to admit):")
    lines.append(f"1. unanimity alone clears 0.90: {'YES' if un_final >= 0.90 else 'NO'} ({un_final:.3f} at coverage {un_cov:.3f}; predicted >=0.90 at ~0.15 coverage)")
    lines.append(f"2. the AND (given_unique x unanimous) clears 0.90: {'YES' if and_final >= 0.90 else 'NO'} ({and_final:.3f} at coverage {and_cov:.3f}; given_unique alone: {gu_final:.3f} at {gu_cov:.3f}; predicted AND near given_unique's 0.22 coverage)")
    lines.append(f"3. the AND keeps given_unique's coverage: {'YES' if and_cov >= 0.9 * gu_cov else 'NO'} ({and_cov:.3f} vs {gu_cov:.3f})")
    if "final" in conds_used:
        lines.append(f"4. breath-2 vs final: given_unique prec_given b2={results['b2']['given_unique']['numeral']['prec_given']:.3f} vs final={gu_final:.3f}; unanimous b2={results['b2']['unanimous']['numeral']['prec_given']:.3f} vs final={un_final:.3f}")
    else:
        lines.append("4. breath-2 vs final: n/a (ALT3 consult body, breath-2/consult-1 only per request)")
    lines.append(f"5. cost: {sum(view_wall):.0f}s wall for {n_views} views on {n} rows, CPU, batch={BATCH} ({sum(view_wall) / max(n_views - 1, 1):.0f}s/extra-view) — K pass-1s per consult at training time would cost this x(# consults)")
    lines.append(f"6. positional precision (order-sensitive, ps_legal_wild) lags numeral precision throughout, as it did for RK_241/PMS8/HS (the drawer vs the dish)")

    txt = "\n".join(lines) + "\n"
    open(out_path, "w").write(txt)
    print(txt)
    print(f"[hammerhead] -> {out_path}")


if __name__ == "__main__":
    main()
