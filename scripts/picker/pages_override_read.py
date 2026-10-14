"""pages_override_read.py -- THE OVERRIDE CURVE (2026-10-14, Fable): can the page reranker be
restricted to cases where its own confidence (margin) is high, instead of the forced-choice mode
pages_read.py measured (b8419d90 / 6d1f6f22 ledger: 0.15-0.22 rescue on wrong-at-first-look slots,
but a 0.20 flip-to-wrong cost on right-at-first-look slots, bar <=0.02 missed by 10x)?

THE RULE: override the first look (the rank-1 page, breath-0's own argmax) ONLY when the
reranker's own margin -- its predicted P(gold) for its forced-choice top page MINUS its predicted
P(gold) for the first-look page -- exceeds a threshold tau. tau=0 degenerates to pages_read.py's
own forced-choice mode (margin>0 whenever the reranker disagrees with the first look at all);
tau=1 (or any tau >= the max observed margin) degenerates to "never override" (pure first look).
Margin is always >= 0 by construction (p_top is a max); when the reranker agrees with the first
look, margin=0 and no tau>=0 can "override" into a different answer anyway (there is nothing to
override to).

tau IS FIXED ON VALID2 ONLY: swept over a grid, for each value computing net corrections
(rescues on valid2's own wrong-at-first-look slots, minus flips on valid2's own right-at-first-
look slots) IN-SAMPLE on the same valid2 slice the reranker is fit on (n=128 rows is too small to
carve a third split; this is named plainly, not hidden -- it is the brief's own instruction: "FIXED
ON VALID2 ... choose tau on valid2"). The selected tau* is the grid point with the highest net
subject to cost (flip rate on valid2's right-at-first-look pages) <= 0.02 there. That FIXED tau*
is then applied, unmodified, to wild -- a true held-out measurement, never touched during tau
selection. The full grid is ALSO run on wild and reported (labeled "informational / ceiling", not
a thing this script re-optimizes against) per the brief's explicit request for both tables.

Wild is MEASURED ONLY: no threshold in this script is ever chosen by looking at wild.

Reuses pages_read.py's own _tokenizer/coverage_table/build_pages/make_model/flatten (import, not
copy) and its already-fit-on-valid2 reranker (identical model family/features/fit call). No new
model class; no new NPZ beyond this script's own report.

Usage: .venv/bin/python3 scripts/picker/pages_override_read.py
Outputs: .cache/pages_override_report.txt / .json (new artifacts; nothing banked touched)
"""
import sys
import json

sys.path.insert(0, "."); sys.path.insert(0, "scripts"); sys.path.insert(0, "scripts/picker")
import numpy as np

import pages_read as PR
import goldmention_read as GM

OUT_TXT = ".cache/pages_override_report.txt"
OUT_JSON = ".cache/pages_override_report.json"

N_GIVEN_WILD_TOTAL = 2051   # THE NUMBER's own denominator (wild fac-exact slots, all ftypes); given
                            # slots (this script's population, n=1009) are a subset of it, so a net
                            # correction count among given slots adds directly to that numerator
                            # over the SAME 2051 denominator -- "the implied masked-wild delta".

TAU_GRID = np.round(np.arange(0.0, 1.0001, 0.02), 4)


def _build_eval_records(model, records):
    """Per GIVEN slot: first-look label, every page's P(gold), the forced-choice top page,
    and the margin (top page's P(gold) minus the first-look page's P(gold))."""
    out = []
    for rec in records:
        X = np.asarray([PR.page_features(p) for p in rec["pages"]], np.float64)
        probs = model.predict_proba(X)[:, 1]
        rank1_idx = next(k for k, p in enumerate(rec["pages"]) if p["rank"] == 1)
        top_idx = int(np.argmax(probs))
        p_fl = float(probs[rank1_idx])
        p_top = float(probs[top_idx])
        out.append(dict(i=rec["i"], j=rec["j"], same_sent=rec["same_sent"],
                         first_look_ok=rec["pages"][rank1_idx]["label"],
                         chosen_ok=rec["pages"][top_idx]["label"],
                         chosen_is_first_look=(top_idx == rank1_idx),
                         margin=max(0.0, p_top - p_fl)))
    return out


def _apply_tau(evalrecs, tau):
    """Decision rule at threshold tau: override (use the forced-choice top page) iff margin>tau;
    else keep the first look. Returns rows with final_ok + overridden flag."""
    rows = []
    for r in evalrecs:
        overridden = r["margin"] > tau and not r["chosen_is_first_look"]
        final_ok = r["chosen_ok"] if overridden else r["first_look_ok"]
        rows.append(dict(**r, overridden=overridden, final_ok=final_ok))
    return rows


def _curve_point(evalrecs, tau, n_total_for_delta=None):
    rows = _apply_tau(evalrecs, tau)
    wrong_fl = [r for r in rows if not r["first_look_ok"]]
    right_fl = [r for r in rows if r["first_look_ok"]]
    rescues = sum(1 for r in wrong_fl if r["overridden"] and r["final_ok"])
    flips = sum(1 for r in right_fl if r["overridden"] and not r["final_ok"])
    n_overridden = sum(1 for r in rows if r["overridden"])
    net = rescues - flips
    cost = flips / max(len(right_fl), 1)
    pt = dict(tau=float(tau), n=len(rows), n_wrong_fl=len(wrong_fl), n_right_fl=len(right_fl),
              override_rate=n_overridden / max(len(rows), 1),
              rescues=rescues, flips=flips, net=net, cost=cost,
              acc=sum(r["final_ok"] for r in rows) / max(len(rows), 1))
    if n_total_for_delta:
        pt["implied_delta_per_%d_slots" % n_total_for_delta] = net / n_total_for_delta
    return pt


def pick_tau_star(evalrecs_valid2, cost_bar=0.02):
    """tau* = grid point maximizing net on valid2 (in-sample), subject to cost<=cost_bar there.
    Tie-break: among tied best net, the SMALLEST tau (more willing to act); if nothing beats
    net=0 under the bar, tau*=1.0 (never override) is the honest answer."""
    pts = [_curve_point(evalrecs_valid2, t) for t in TAU_GRID]
    feasible = [p for p in pts if p["cost"] <= cost_bar]
    if not feasible:
        feasible = [p for p in pts if p["tau"] >= 0.999]  # fallback: never-override is always cost=0
    best_net = max(p["net"] for p in feasible)
    best = min((p for p in feasible if p["net"] == best_net), key=lambda p: p["tau"])
    return best["tau"], pts


def main():
    tok = PR._tokenizer()

    # ---- same slot populations pages_read.py builds (verbatim calls, not copied logic) ----
    cov, wild_cells = PR.coverage_table(tok)
    same_sent_wild = {(c["i"], c["j"]): c["same_sent"] for c in wild_cells}
    raw_wild = np.load(PR.RAW_WILD)
    wild_records = PR.build_pages(raw_wild, tok, same_sent_wild)

    raw_v2 = np.load(PR.RAW_V2)
    ok_v2 = PR._load_ok_generic(PR.OK_V2)
    n_v2 = raw_v2["ids"].shape[0]
    cells_v2, _, _ = GM._collect_cells(
        raw_v2["ids"], raw_v2["tokmask"], raw_v2["sent"], raw_v2["g_presence"], raw_v2["g_ftype"],
        raw_v2["g_digits"], raw_v2["fat_mass"][:, 0, :, :].astype(np.float32), ok_v2, n_v2,
        PR.L_FAC, PR.T_ALG, tok)
    same_sent_v2 = {(c["i"], c["j"]): c["same_sent"] for c in cells_v2}
    v2_records = PR.build_pages(raw_v2, tok, same_sent_v2)

    print(f"[override] wild given slots={len(wild_records)}  valid2 given slots={len(v2_records)}", flush=True)

    # ---- fit the IDENTICAL reranker pages_read.py fits (valid2 only) ----
    Xtr, ytr, _ = PR.flatten(v2_records)
    model = PR.make_model()
    model.fit(Xtr, ytr)

    eval_v2 = _build_eval_records(model, v2_records)
    eval_wild = _build_eval_records(model, wild_records)

    tau_star, v2_curve = pick_tau_star(eval_v2, cost_bar=0.02)
    print(f"[override] tau* fixed on valid2 (net-maximizing, cost<=0.02 there) = {tau_star}", flush=True)

    op_point_wild = _curve_point(eval_wild, tau_star, n_total_for_delta=N_GIVEN_WILD_TOTAL)
    op_point_v2 = _curve_point(eval_v2, tau_star)

    wild_curve = [_curve_point(eval_wild, t, n_total_for_delta=N_GIVEN_WILD_TOTAL) for t in TAU_GRID]
    # best any tau achieves on wild -- a CEILING (chosen by looking at wild; never deployable,
    # reported only because the brief asks for it labeled as such)
    best_wild = max(wild_curve, key=lambda p: p["net"])

    out = dict(
        tau_grid=[float(t) for t in TAU_GRID],
        tau_star_fixed_on_valid2=float(tau_star),
        valid2_curve_at_tau_star=op_point_v2,
        valid2_full_curve=v2_curve,
        wild_operating_point_at_valid2_fixed_tau=op_point_wild,
        wild_full_curve=wild_curve,
        wild_ceiling_best_tau=best_wild,
        n_given_wild_total_denominator=N_GIVEN_WILD_TOTAL,
        registered_prediction="temperature-perp-truth: little rescue expected (confidence tracks "
                               "basin depth, not correctness; the mass gap feature was weak)",
    )

    lines = []
    P = lines.append
    P("=" * 96)
    P("THE OVERRIDE CURVE -- tau fixed on valid2 (in-sample, n=128 rows / %d given-slot pages), "
      "measured on wild (n=311 rows / %d given-slot pages, held out)" % (len(v2_records), len(wild_records)))
    P("=" * 96)
    P(f"tau* = {tau_star}  (valid2 in-sample: net={op_point_v2['net']} rescues={op_point_v2['rescues']} "
      f"flips={op_point_v2['flips']} cost={op_point_v2['cost']:.4f} override_rate={op_point_v2['override_rate']:.4f})")
    P("-" * 96)
    P("WILD operating point at the valid2-fixed tau*:")
    p = op_point_wild
    P(f"  n={p['n']}  override_rate={p['override_rate']:.4f}  rescues(of {p['n_wrong_fl']} wrong-fl)={p['rescues']}  "
      f"flips(of {p['n_right_fl']} right-fl)={p['flips']}  cost={p['cost']:.4f}  net={p['net']:+d}/{p['n']}  "
      f"acc={p['acc']:.3f} (first-look acc={sum(r['first_look_ok'] for r in eval_wild)/len(eval_wild):.3f})")
    P(f"  implied masked-wild delta per {N_GIVEN_WILD_TOTAL} slots: {p[f'implied_delta_per_{N_GIVEN_WILD_TOTAL}_slots']:+.5f}")
    P("-" * 96)
    P(f"WILD ceiling -- best net achieved by ANY tau on the grid (chosen by looking at wild; "
      f"NEVER deployable, reported as a ceiling only):")
    b = best_wild
    P(f"  tau={b['tau']}  override_rate={b['override_rate']:.4f}  rescues={b['rescues']}  flips={b['flips']}  "
      f"cost={b['cost']:.4f}  net={b['net']:+d}/{b['n']}  "
      f"implied delta/{N_GIVEN_WILD_TOTAL}={b[f'implied_delta_per_{N_GIVEN_WILD_TOTAL}_slots']:+.5f}")
    P("-" * 96)
    P("WILD full grid (tau, override_rate, rescues, flips, cost, net/1009, implied_delta/2051):")
    for q in wild_curve:
        P(f"  tau={q['tau']:.2f}  ovr={q['override_rate']:.3f}  rescues={q['rescues']:4d}  "
          f"flips={q['flips']:4d}  cost={q['cost']:.4f}  net={q['net']:+5d}  "
          f"delta={q[f'implied_delta_per_{N_GIVEN_WILD_TOTAL}_slots']:+.5f}")
    P("-" * 96)
    P("VALID2 full grid (the selection surface tau* was chosen from; in-sample, informational):")
    for q in v2_curve:
        P(f"  tau={q['tau']:.2f}  ovr={q['override_rate']:.3f}  rescues={q['rescues']:4d}  "
          f"flips={q['flips']:4d}  cost={q['cost']:.4f}  net={q['net']:+5d}  feasible={'Y' if q['cost']<=0.02 else 'n'}")
    P("=" * 96)
    P("REGISTERED PREDICTION (temperature-perp-truth law): " + out["registered_prediction"])
    # Verdict against THE NUMBER's own wild bar (CLAUDE.md Sec.4): a wild claim needs >= 0.020
    # or a seed twin, because wild fac-exact's own SE is 0.0096 (n=2051 slots) -- so judge the
    # implied delta against THAT bar, not an arbitrary net-count cutoff.
    d_op = op_point_wild[f"implied_delta_per_{N_GIVEN_WILD_TOTAL}_slots"]
    d_ceil = best_wild[f"implied_delta_per_{N_GIVEN_WILD_TOTAL}_slots"]
    WILD_BAR = 0.020
    WILD_SE = 0.0096
    verdict = ("CONFIRMED (little rescue: both the valid2-fixed operating point and the best-"
               "achievable-on-wild ceiling fall under the 0.020 wild bar / ~1 SE of 0.0096 -- "
               "confidence tracked basin depth, not a real correctness signal worth deploying)"
               if d_ceil < WILD_BAR else
               "NOT CONFIRMED (the ceiling clears the wild bar -- a real, if margin-gated, signal)")
    P(f"READING: wild delta at tau* = {d_op:+.5f} ({d_op/WILD_SE:+.2f} SE); ceiling delta = "
      f"{d_ceil:+.5f} ({d_ceil/WILD_SE:+.2f} SE); wild bar = {WILD_BAR:+.3f} -> {verdict}")
    out["verdict"] = verdict
    out["wild_bar"] = WILD_BAR
    out["wild_se"] = WILD_SE

    txt = "\n".join(lines)
    print(txt)
    with open(OUT_TXT, "w") as f:
        f.write(txt + "\n")
    with open(OUT_JSON, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\n[override] wrote {OUT_TXT} / {OUT_JSON}")


if __name__ == "__main__":
    main()
