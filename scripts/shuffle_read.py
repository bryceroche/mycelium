"""shuffle_read.py -- THE GYM's THIRD USE: THE SHUFFLE READ (2026-10-08, worktree
mycelium-wt8, branch replay; Bryce's map-reduce / "the shuffle").

QUESTION: does the slot-to-slot MIXER (phase1_algebra_head.py's sc2/h_slot single-head path
and the FED_MIXER multi-head twin, _mx_sc/_mx_raw) already group slots by OWNER -- the same
"owner" identity THE GOVERNOR CENSUS (scripts/governor_census.py, spaCy en_core_web_sm)
resolves from the dependency parse (numeral -> head noun -> verb -> subject/possessor)?

METHOD: PMS8_241 (SURF8 env, no consults -- a single ALT2 pass is the whole read), 64 wild
rows, DEV=CPU, a direct forward() call with the _CENSUS hook armed (the eyes_autopsy.py
precedent the task names) -- NOT the snapshot/replay path, since _CENSUS already captures
every breath 1..6 in one pass and there is no need to resume from a mid-loop state for a pure
READ. Two new taps were added to breath_step for this (phase1_algebra_head.py, right where
the existing single-head and multi-head mixer softmaxes are computed -- dark unless _CENSUS is
armed, zero cost otherwise, confirmed by the standing unset/role8/hierd gate):
  "mixer_attn1"      (B, L_TOT, L_TOT)            the single-head mixer's real post-mask attn
  "mixer_attn_heads" (B, MX_HEADS, L_TOT, L_TOT)   the FED twin's per-head attn

OWNERS: each gold GIVEN slot's owner = governor_census.governor_chain()'s owner_key off its
bound numeral token (governor_census's own bind_given_occurrences/numeral_candidates reused by
import -- no reimplementation of the spaCy walk). Each gold RELATION slot's owner = the single
owner shared by both its gold args IF both args are themselves GIVEN vars and resolve to the
SAME owner_key ("same-owner"); anything else (an arg that is itself a relation's result, an
unresolved owner, or two different owners) is "mixed" -- the same coarsening governor_census.py
itself uses for relation args (its own PART C skips non-given args as "arg_derived_skipped"
rather than resolving them recursively). "none" is its own owner value (unresolved chain) --
STATED SIMPLIFICATION: two slots that both fail owner resolution count as "same owner" under
plain string equality; not corrected for here.

"Empty" slots = L_TOT positions with no gold fact at all (j >= len(row's factors), and any
scratch rows) -- destinations of attention that are neither same- nor different-owner, just
unoccupied.

right/wrong: .cache/ps_legal_wild_PMS8_241.npz, same convention as resonance_read.py /
replay_gate_check.py.

DEV=CPU asserted; .cache/gpu.lock never touched (no flock in this file).
"""
import os
import sys
import time
import collections

sys.path.insert(0, "."); sys.path.insert(0, "scripts")
os.environ.setdefault("DEV", "CPU")
assert os.environ.get("DEV") == "CPU"

import numpy as np

N_ROWS = 64
CKPT = ".cache/sharp_PMS8_241.safetensors"
PS_LEGAL_PATH = ".cache/ps_legal_wild_PMS8_241.npz"
OUT_TXT = ".cache/shuffle_read_PMS8_241.txt"

_FAM = {
    "DEV": "CPU", "ALG2": "1", "ALG_FTYPES": "9", "ALG_DUP": "1",
    "ALG_HW": "512", "ALG_WIDE": "1", "ALG_BREATH": "7",
    "ALG_NOTEBOOK": "1", "ALG_SIXWAVE": "1", "NB_PERSLOT": "1",
    "ALG_BINDBUS": "7", "ALG_BIND_D": "512",
    "BIND_CODES": ".cache/bindbus_codes512r.npz",
    "ALG_BUSGARAGE": "2", "ALG_SHELF_CIRCLE": "2", "ALG_ALTMASK": "1",
    "ALG_ALT21": "1", "ALG_ALT2": "1", "ALG_MASKHEAD": "1",
    "ALG_FED": "1", "ALG_POLAR": "1", "ALG_POLAR_D": "128",
    "ALG_POLAR_EM": "0.1",
    "ALG_POLAR_D_INIT": ".cache/polar_waist_init_d128u.npz",
    "ALG_PRUNE": "pforms,s4,fednl0,lane2", "ALG_SLOT_ALL": "1",
    "ALG_STELLAR": "2", "ALG_CLOCK_CANON": "1", "SC_EVAL": "0",
    "ALG_TEST": ".cache/wild_admitted_holdout.jsonl", "ALG_TEST_NAME": "wildhold",
    "ALG_ROUTER": "2", "R_GAIN_INIT": "1.0", "ALG_FREEZE": "r_gain", "ALG_ROUTER_PTR": "0.0",
    "ALG_SPAN_ALL": "1", "ALG_SPAN_ARGS": "1", "ALG_SPAN_OP": "1", "ALG_SPAN_RCUE": "1",
    "ALG_SPAN_ARCUE": "1", "ALG_PTR_SURF": "role:add:2.0",
}


def main():
    P = []
    def log(s):
        print(s, flush=True); P.append(s)

    t0 = time.time()
    for k, v in _FAM.items():
        os.environ[k] = v
    import phase1_algebra_head as H
    from tinygrad.nn.state import safe_load
    from tinygrad import Tensor, dtypes
    import spacy
    import governor_census as GC

    p = H.build_params(0)
    sd = safe_load(CKPT)
    assert set(sd.keys()) == set(p.keys()), (sorted(set(sd) - set(p))[:4], sorted(set(p) - set(sd))[:4])
    for k in p:
        p[k].assign(sd[k].to(p[k].device).cast(p[k].dtype)).realize()

    vs, vst_, vtk, vg, vse = H.load_alg("test")
    n = min(N_ROWS, len(vs))
    sl = np.arange(n)
    ts = Tensor(np.ascontiguousarray(vst_[sl]), dtype=dtypes.half)
    tk = Tensor(vtk[sl].astype(np.float32), dtype=dtypes.float)
    se = Tensor(vse[sl].astype(np.int32), dtype=dtypes.int)

    o0 = H.forward(p, ts, tk, se)
    onp0 = {k: o0[k].realize().numpy() for k in ("fat", "args", "res")}
    mk = H.build_slot_masks(onp0, se.numpy().astype(np.int32))
    slot_mask = Tensor(mk, dtype=dtypes.float)
    _ka = ("pres", "ftype", "op", "dig") + (("dup",) if "dup" in o0 else ())
    _oa = {**onp0, **{k: o0[k].realize().numpy() for k in _ka}}
    _nv = np.array([vs[int(i)].get("n_vars", H.K_VARS) for i in sl])
    _ma = np.array([vs[int(i)].get("m", 0) for i in sl])
    fb = H.alt2_fact_buf(_oa, se.numpy().astype(np.int32), _nv, _ma)
    fact_t = Tensor(fb, dtype=dtypes.float)

    # ---- THE CENSUS: one forward pass, _CENSUS armed (eyes_autopsy.py's precedent) ----
    H._CENSUS = []
    o = H.forward(p, ts, tk, se, slot_mask=slot_mask, fact_buf=fact_t)
    census = H._CENSUS
    H._CENSUS = None
    log(f"[shuffle] census forward done ({time.time() - t0:.0f}s so far); {len(census)} tap entries")

    attn1 = {}; attn_heads = {}
    for kb, tag, arr in census:
        if tag == "mixer_attn1":
            attn1[kb] = arr          # (B, L_TOT, L_TOT)
        elif tag == "mixer_attn_heads":
            attn_heads[kb] = arr     # (B, MX_HEADS, L_TOT, L_TOT)
    K_B = int(os.environ.get("ALG_BREATH", "7"))
    breaths = sorted(attn1)
    log(f"[shuffle] mixer_attn1 captured at breaths {breaths}; mixer_attn_heads at {sorted(attn_heads)}")

    # ---- THE GOVERNOR CENSUS, reused by import ----
    NLP = spacy.load("en_core_web_sm")

    ps = np.load(PS_LEGAL_PATH)
    ok_lookup = {(int(r), int(c)): bool(o_) for r, c, o_ in zip(ps["rows"], ps["slots"], ps["ok"])}

    # per-row: owner label for every var that is a GIVEN; per-slot owner label (given or
    # relation, "mixed" / "none" as coarsened above); slot kind (given/rel); gold args (for
    # relation slots, part (d))
    owner_of_slot = {}     # (i, j) -> owner label string
    kind_of_slot = {}      # (i, j) -> "given" / "rel"
    args_of_slot = {}      # (i, j) -> list of arg var indices (relation slots only)
    right_of_slot = {}     # (i, j) -> bool or None

    n_resolved = n_unresolved = 0
    for i in range(n):
        row = vs[i]
        facs = row["factors"]
        text = row["text"]
        doc = NLP(text)
        sent_list = list(doc.sents)
        cands = GC.numeral_candidates(doc)
        given_list = [(f["var"], f["value"]) for f in facs if f["ftype"] == "given"]
        given_vars = {v for v, _ in given_list}
        bound = GC.bind_given_occurrences(doc, given_list, cands)
        owner_of_var = {}
        for var, tok in bound.items():
            if tok is None:
                owner_of_var[var] = "none"; n_unresolved += 1; continue
            ch = GC.governor_chain(doc, sent_list, tok)
            if ch is None:
                owner_of_var[var] = "none"; n_unresolved += 1
            else:
                owner_of_var[var] = ch["owner_key"]; n_resolved += 1

        for j, fac in enumerate(facs):
            if j >= H.L_FAC:
                continue
            right_of_slot[(i, j)] = ok_lookup.get((i, j))
            if fac["ftype"] == "given":
                kind_of_slot[(i, j)] = "given"
                owner_of_slot[(i, j)] = owner_of_var.get(fac["var"], "none")
            elif fac["ftype"] == "rel":
                kind_of_slot[(i, j)] = "rel"
                args = list(fac["args"])
                args_of_slot[(i, j)] = args
                arg_owners = []
                for a in args:
                    if a in given_vars:
                        arg_owners.append(owner_of_var.get(a, "none"))
                    else:
                        arg_owners.append(None)   # derived/non-given arg: unresolved by this script's coarsening
                if len(arg_owners) >= 2 and all(o_ is not None and o_ != "none" for o_ in arg_owners) and len(set(arg_owners)) == 1:
                    owner_of_slot[(i, j)] = arg_owners[0]
                else:
                    owner_of_slot[(i, j)] = "mixed"
            else:
                kind_of_slot[(i, j)] = fac["ftype"]
                owner_of_slot[(i, j)] = "none"
        if i % 16 == 0:
            log(f"[shuffle] governor census row {i}/{n} ({time.time() - t0:.0f}s so far)")

    gold_slots = [(i, j) for (i, j) in owner_of_slot if right_of_slot.get((i, j)) is not None]
    n_right = sum(1 for (i, j) in gold_slots if right_of_slot[(i, j)])
    log(f"[shuffle] {len(gold_slots)} gold slots ({n_right} right, {len(gold_slots)-n_right} wrong); "
        f"owner resolution: {n_resolved} resolved, {n_unresolved} unresolved")
    owner_counts = collections.Counter(owner_of_slot[(i, j)] for (i, j) in gold_slots)
    log(f"[shuffle] owner label distribution (top 8): {owner_counts.most_common(8)}")

    # per-row: which L_FAC positions are "empty" (no fact at all)
    n_facs_of_row = {i: len(vs[i]["factors"]) for i in range(n)}

    # ================================================================================
    # (a)/(b)/(c): the grouping factor and effective k, per breath, right vs wrong
    # ================================================================================
    group_rows = {}   # (kb, right) -> list of (mean_same, mean_diff, mean_empty)
    k_eff_rows = {}    # (kb, right) -> list of k_eff (head-mean attn)
    for kb in breaths:
        A = attn1[kb]   # (B, L_TOT, L_TOT)
        for (i, j) in gold_slots:
            right = right_of_slot[(i, j)]
            row_av = A[i, j]       # (L_TOT,)
            own = owner_of_slot[(i, j)]
            same_m, diff_m, empty_m = [], [], []
            nfacs = n_facs_of_row[i]
            for k in range(H.L_FAC):
                if k == j:
                    continue
                if k < nfacs:
                    ok2 = owner_of_slot.get((i, k))
                    if ok2 is None:
                        empty_m.append(row_av[k]); continue
                    (same_m if ok2 == own else diff_m).append(row_av[k])
                else:
                    empty_m.append(row_av[k])
            # scratch rows (if any), L_FAC..L_TOT-1, are also "empty"
            for k in range(H.L_FAC, A.shape[-1]):
                empty_m.append(row_av[k])
            ms = float(np.mean(same_m)) if same_m else float("nan")
            md = float(np.mean(diff_m)) if diff_m else float("nan")
            me = float(np.mean(empty_m)) if empty_m else float("nan")
            group_rows.setdefault((kb, right), []).append((ms, md, me))
            p_ = row_av[row_av > 1e-12]
            ent = float(-np.sum(p_ * np.log(p_)))
            k_eff_rows.setdefault((kb, right), []).append(float(np.exp(ent)))
    log(f"[shuffle] (a)/(b) aggregated ({time.time() - t0:.0f}s so far)")

    # ================================================================================
    # (d): the binding read -- gold-arg attention vs other-slot attention, relation slots only
    # ================================================================================
    bind_rows = {}   # (kb, right) -> list of (mean_to_args, mean_to_others)
    rel_slots = [(i, j) for (i, j) in gold_slots if kind_of_slot.get((i, j)) == "rel"]
    for kb in breaths:
        A = attn1[kb]
        for (i, j) in rel_slots:
            right = right_of_slot[(i, j)]
            row_av = A[i, j]
            args = args_of_slot.get((i, j), [])
            arg_m, other_m = [], []
            nfacs = n_facs_of_row[i]
            for k in range(nfacs):
                if k == j:
                    continue
                (arg_m if k in args else other_m).append(row_av[k])
            ma = float(np.mean(arg_m)) if arg_m else float("nan")
            mo = float(np.mean(other_m)) if other_m else float("nan")
            bind_rows.setdefault((kb, right), []).append((ma, mo))
    log(f"[shuffle] (d) aggregated ({time.time() - t0:.0f}s so far); {len(rel_slots)} relation gold slots")

    # ================================================================================
    # write the report
    # ================================================================================
    lines = []
    L = lines.append
    L("=" * 92)
    L("THE SHUFFLE READ -- PMS8_241, wild, 64 rows, single-head mixer attention (2026-10-08)")
    L("=" * 92)
    L(f"gold slots: {len(gold_slots)} (right={n_right}, wrong={len(gold_slots)-n_right}); "
      f"owner resolution: {n_resolved} resolved / {n_unresolved} unresolved (both given-slot and "
      f"relation-arg tokens, before the relation same/mixed coarsening)")
    L(f"owner label distribution across gold slots (top 8): {owner_counts.most_common(8)}")
    L("")
    L("-" * 92)
    L("(a) GROUPING: mean mixer attention same-owner / different-owner / empty, by breath x right/wrong")
    L("-" * 92)
    L("  breath  split    n    same      diff      empty    ratio(same/diff)")
    for kb in breaths:
        for right, label in ((True, "right"), (False, "wrong")):
            rows = group_rows.get((kb, right), [])
            if not rows:
                continue
            arr = np.array(rows)
            ms, md, me = np.nanmean(arr[:, 0]), np.nanmean(arr[:, 1]), np.nanmean(arr[:, 2])
            L(f"    b{kb}   {label:5s}  {len(rows):4d}  {ms:.5f}  {md:.5f}  {me:.5f}    {ms/(md+1e-12):7.3f}")
    L("")
    L("-" * 92)
    L("(b) THE MIXER's EFFECTIVE k (exp(attention entropy), head-mean), by breath x right/wrong")
    L("-" * 92)
    L("  breath  split    n     mean-k    p25      p50      p75")
    for kb in breaths:
        for right, label in ((True, "right"), (False, "wrong")):
            rows = k_eff_rows.get((kb, right), [])
            if not rows:
                continue
            arr = np.array(rows)
            L(f"    b{kb}   {label:5s}  {len(rows):4d}  {np.mean(arr):7.3f}  "
              f"{np.percentile(arr,25):7.3f}  {np.percentile(arr,50):7.3f}  {np.percentile(arr,75):7.3f}")
    L("")
    L("-" * 92)
    L("(d) THE BINDING READ (relation slots only): mean attention to gold ARGS vs to OTHER gold slots")
    L("-" * 92)
    L("  breath  split    n    to-args   to-other   ratio")
    for kb in breaths:
        for right, label in ((True, "right"), (False, "wrong")):
            rows = bind_rows.get((kb, right), [])
            if not rows:
                continue
            arr = np.array(rows)
            ma, mo = np.nanmean(arr[:, 0]), np.nanmean(arr[:, 1])
            L(f"    b{kb}   {label:5s}  {len(rows):4d}  {ma:.5f}  {mo:.5f}   {ma/(mo+1e-12):7.3f}")

    L(""); L("-" * 92); L("THE SIX-LINE READING"); L("-" * 92)
    def _ratio_series(right=None):
        out = []
        for kb in breaths:
            if right is None:
                rows = group_rows.get((kb, True), []) + group_rows.get((kb, False), [])
            else:
                rows = group_rows.get((kb, right), [])
            if not rows:
                out.append(float("nan")); continue
            arr = np.array(rows)
            ms, md = np.nanmean(arr[:, 0]), np.nanmean(arr[:, 1])
            out.append(ms / (md + 1e-12))
        return out
    r_all = _ratio_series(None); r_right = _ratio_series(True); r_wrong = _ratio_series(False)
    L(f"1. GROUPING FACTOR (same/diff) by breath, all gold slots: " + " ".join(f"{x:.2f}" for x in r_all))
    L(f"2. RIGHT vs WRONG grouping factor by breath: right " + " ".join(f"{x:.2f}" for x in r_right)
      + "; wrong " + " ".join(f"{x:.2f}" for x in r_wrong))
    L(f"3. TREND: {'rising' if r_all[-1] > r_all[0] + 0.1 else ('falling' if r_all[-1] < r_all[0] - 0.1 else 'flat')} "
      f"across breaths {breaths[0]}..{breaths[-1]} (b{breaths[0]}={r_all[0]:.2f} -> b{breaths[-1]}={r_all[-1]:.2f})")
    k_all = []
    for kb in breaths:
        rows = k_eff_rows.get((kb, True), []) + k_eff_rows.get((kb, False), [])
        k_all.append(float(np.mean(rows)) if rows else float("nan"))
    L(f"4. EFFECTIVE k by breath: " + " ".join(f"{x:.2f}" for x in k_all)
      + f" (L_TOT={H.L_TOT}; leader~1 / shoal~7 / mush~{H.L_TOT})")
    b_all = []
    for kb in breaths:
        rows = bind_rows.get((kb, True), []) + bind_rows.get((kb, False), [])
        if rows:
            arr = np.array(rows)
            b_all.append(float(np.nanmean(arr[:, 0]) / (np.nanmean(arr[:, 1]) + 1e-12)))
        else:
            b_all.append(float("nan"))
    L(f"5. BINDING (args-vs-other ratio) by breath: " + " ".join(f"{x:.2f}" for x in b_all))
    verdict = ("LATENT (grouping factor > 1.5)" if (not np.isnan(r_all[-1]) and r_all[-1] > 1.5)
               else ("ABSENT (~1)" if (not np.isnan(r_all[-1]) and 0.67 < r_all[-1] < 1.5)
                     else "THE WRONG WAY (< 1: attends MORE to different-owner slots)"))
    L(f"6. VERDICT: the shuffle is {verdict} at the last captured breath (b{breaths[-1]}, ratio={r_all[-1]:.2f}); "
      f"right-vs-wrong gap at b{breaths[-1]} = {r_right[-1]-r_wrong[-1]:+.2f}.")

    txt = "\n".join(lines) + "\n"
    open(OUT_TXT, "w").write(txt)
    log(f"[shuffle] wrote {OUT_TXT} ({time.time() - t0:.0f}s total)")


if __name__ == "__main__":
    main()
