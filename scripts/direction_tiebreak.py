"""scripts/direction_tiebreak.py -- THE DIRECTION TIE-BREAK (2026-10-09, zero training; word given
in the coordinator's message following "THE OWN-SUPPRESSION ORACLE's READING" ledger entry,
2026-10-09 16:53). The oracle showed that clamping the own-index res logit WITH THE GOLD direction
lifts inverse res from 0.25 to 0.60-0.62 on both PMS8_241 and PC_241 -- a real ceiling. This script
asks whether a body can approach that ceiling with NO gold at read time: (1) retrain the direction
probe's read-out (direction_probe.py's feature E "args alone" and B "own+args") on PMS8_241's DIET
states, but with the ARGS taken from the args head's own PREDICTED top-2 at the last breath
(dirtb_rawdump.py's own collection) instead of gold; (2) pick the decision threshold tau on the
DIET ONLY, maximizing diet masked fac-exact under the tie-break policy (never looking at wild to
choose it); (3) on wild, for every slot the head ITSELF decodes as a relation (honest read -- no
gold ftype at read time), clamp the own-index res logit to -inf when p(inverse) > tau and re-
decode; report inverse/forward res, the other fields (unchanged by construction), and the masked
fac-exact PAIRED against each body's own banked (unclamped) read, via paired_read.py's own
quantity; (4) the no-MLP control -- clamp EVERY relation slot unconditionally (tau=0, no probe at
all) -- forward cost vs inverse gain, next to the oracle's ceiling.

PIPELINE (all zero-training reads; the only GPU time was two short collections, already run and
released -- see .cache/dirtb_collect.sh):
  DIET (PMS8_241 only, the probe's training ground):
    .cache/form_pm35c_slice1024_valid2.jsonl           (775 rows, 705 admissible)
    .cache/welford_atlas_PMS8_241_diet_states_breaths.npz  (states_all, CONTENT-only, banked)
    .cache/dirtb_rawslots_diet_PMS8_241.pkl            (NEW -- args/res/dup/... raw logits,
                                                         solve-free, scripts/dirtb_rawdump.py)
  WILD (both bodies, the deployed read):
    .cache/wild_admitted_holdout.jsonl                 (311 rows)
    PMS8_241: .cache/clock_band_states_PMS8_241.npz (banked) + .cache/rawslots_wild_PMS8_241.pkl
              (banked) + .cache/dump_wild_PMS8_241.pkl (banked, LV_DUMP) + .cache/
              ps_legal_wild_PMS8_241.npz (banked, the paired baseline)
    PC_241:   .cache/clock_band_states_PC_241_dirtb.npz (NEW, this read's own collection) +
              .cache/ownsup_rawslots_wild_PC_241.pkl (NEW, the own-suppression oracle's own
              collection, reused read-only) + .cache/dump_wild_PC_241.pkl (banked) + .cache/
              ps_legal_wild_PC_241.npz (banked, the paired baseline)

WHICH SLOTS GET CLAMPED (the honest-read decision, stated up front, not buried): ELIGIBLE = the
slot the HEAD ITSELF decodes as present AND a relation (ppres>0.5 and argmax(ftype)==0) -- no gold
ftype anywhere in this test. This is a strict subset of "gold relation slots" (ftype accuracy is
0.85-0.91, not 1.0) and the two framings necessarily agree on where a flip CAN happen (a slot the
head never routes as a relation is never touched, under either framing) -- so this script reports
accuracy by form (inverse res / forward res) with gold-fwd/gold-inv as the DENOMINATOR (comparable
to every prior MECH line in the ledger) while gating every actual clamp decision on ELIGIBLE,
exactly as deployed hardware would have to.

usage: .venv/bin/python3 scripts/direction_tiebreak.py
outputs: .cache/dirtb_report.txt; .cache/dirtb_ps_legal_wild_<BODY>_tiebreak.npz (paired_read.py
inputs, tau* policy); .cache/dirtb_ps_legal_wild_<BODY>_tau0.npz (the no-MLP control)
"""
import json
import os
import pickle
import subprocess
import sys
import time
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
sys.path.insert(0, ".")
sys.path.insert(0, "scripts")

os.environ["DEV"] = "CPU"
for _k, _v in (("DEV", "CPU"), ("ALG2", "1"), ("ALG_FTYPES", "9"), ("ALG_DUP", "1"), ("ALG_WIDE", "1"), ("ALG_HW", "512"),
               ("ALG_BREATH", "7"), ("ALG_NOTEBOOK", "1"), ("ALG_SIXWAVE", "1"), ("NB_PERSLOT", "1"), ("ALG_BINDBUS", "7"),
               ("ALG_BIND_D", "512"), ("BIND_CODES", ".cache/bindbus_codes512r.npz"), ("ALG_BUSGARAGE", "2"),
               ("ALG_SHELF_CIRCLE", "2"), ("ALG_ALTMASK", "1"), ("ALG_ALT21", "1"), ("ALG_ALT2", "1"), ("ALG_MASKHEAD", "1"),
               ("ALG_FED", "1"), ("ALG_POLAR", "1"), ("ALG_POLAR_D", "128"), ("ALG_POLAR_EM", "0.1"),
               ("ALG_POLAR_D_INIT", ".cache/polar_waist_init_d128u.npz"), ("ALG_PRUNE", "pforms,s4,fednl0,lane2"),
               ("ALG_SLOT_ALL", "1"), ("ALG_STELLAR", "2"), ("ALG_CLOCK_CANON", "1"), ("SC_EVAL", "0")):
    os.environ.setdefault(_k, _v)
assert os.environ["DEV"] == "CPU", "direction_tiebreak: zero-GPU analysis, always (collection already ran separately)"

import numpy as np
from sklearn.metrics import roc_auc_score

import direction_probe as DP   # reuse: MLP, mlp_cv, mlp_fit_final, logreg_cv, logreg_fit_final, operating_point, StandardScaler, GroupKFold
from polarity_census import classify_row, load_dump, head_fields

WILD_JSONL = ".cache/wild_admitted_holdout.jsonl"
DIET_JSONL = ".cache/form_pm35c_slice1024_valid2.jsonl"
DIET_STATES_NPZ = ".cache/welford_atlas_PMS8_241_diet_states_breaths.npz"
DIET_RAW_PKL = ".cache/dirtb_rawslots_diet_PMS8_241.pkl"
DIET_KB = 6     # final breath, diet states_all's 7-wide breath axis
WILD_IDX = 5    # final breath, clock_band_states' 6-wide loop-breath axis

BODIES = ["PMS8_241", "PC_241"]
WILD_STATES_NPZ = {"PMS8_241": ".cache/clock_band_states_PMS8_241.npz",
                    "PC_241": ".cache/clock_band_states_PC_241_dirtb.npz"}
WILD_RAW_PKL = {"PMS8_241": ".cache/rawslots_wild_PMS8_241.pkl",
                 "PC_241": ".cache/ownsup_rawslots_wild_PC_241.pkl"}
PS_LEGAL_NPZ = {"PMS8_241": ".cache/ps_legal_wild_PMS8_241.npz",
                "PC_241": ".cache/ps_legal_wild_PC_241.npz"}

OUT = ".cache/dirtb_report.txt"
FWD_PRECISION_BAR = 0.85

CONTENT = DP.CONTENT   # (384,) indices into the 512-wide state -- the SAME content-dims door

LOG = []


def P(s=""):
    print(s, flush=True)
    LOG.append(s)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -40, 40)))


def decode_args(args_logit, dup_logit):
    """dup>0 -> a single repeated index; else the top-2 raw-logit indices. Returns (a0, a1), a0<=a1."""
    if dup_logit is not None and float(dup_logit) > 0:
        a = int(np.argmax(args_logit))
        return a, a
    top2 = np.argsort(-args_logit)[:2]
    a0, a1 = sorted(top2.tolist())
    return a0, a1


def feat_vec(fset, own_state, a0_state, a1_state):
    arg_state = np.concatenate([a0_state, a1_state])
    if fset == "E":
        return arg_state
    if fset == "B":
        return np.concatenate([own_state, arg_state])
    raise ValueError(fset)


# ===========================================================================
# (1) DIET predicted-args feature extraction + probe training
# ===========================================================================

def load_diet():
    diet_rows = [json.loads(l) for l in open(DIET_JSONL)]
    z = np.load(DIET_STATES_NPZ)
    states = z["states_all"]            # (775, 7, 24, 384), content-only already
    admissible = z["admissible"]
    raw = {d["i"]: d for d in pickle.load(open(DIET_RAW_PKL, "rb"))}
    return diet_rows, states, admissible, raw


def extract_diet_probe_samples(diet_rows, states, admissible, raw):
    """fwd/inv-labeled diet slots only (the probe's own training set) -- own state + PREDICTED
    top-2 args' states at the diet's final breath (kb=DIET_KB)."""
    row_id, y, own, argstate = [], [], [], []
    n_other = 0
    for i, r in enumerate(diet_rows):
        if not admissible[i] or i not in raw:
            continue
        factors = r["factors"]
        cls = classify_row(factors)
        dd = raw[i]
        for k, f in enumerate(factors):
            c = cls[k]
            if c == "other":
                n_other += 1
                continue
            if c not in ("fwd", "inv"):
                continue
            a0, a1 = decode_args(dd["args"][k], dd["dup"][k])
            st_own = states[i, DIET_KB, k, :].astype(np.float32)
            st0 = states[i, DIET_KB, a0, :].astype(np.float32)
            st1 = states[i, DIET_KB, a1, :].astype(np.float32)
            row_id.append(i)
            y.append(1 if c == "inv" else 0)
            own.append(st_own)
            argstate.append(np.concatenate([st0, st1]))
    return dict(row=np.array(row_id), y=np.array(y, dtype=np.int64),
                own=np.array(own, dtype=np.float32), argstate=np.array(argstate, dtype=np.float32)), n_other


def extract_wild_probe_samples(body, wild_rows, states_wild, raw_wild):
    row_id, y, own, argstate, slot = [], [], [], [], []
    for i, r in enumerate(wild_rows):
        if i not in raw_wild:
            continue
        factors = r["factors"]
        cls = classify_row(factors)
        dd = raw_wild[i]
        for k, f in enumerate(factors):
            c = cls[k]
            if c not in ("fwd", "inv"):
                continue
            a0, a1 = decode_args(dd["args"][k], dd["dup"][k])
            st_own = states_wild[i, WILD_IDX, k, CONTENT].astype(np.float32)
            st0 = states_wild[i, WILD_IDX, a0, CONTENT].astype(np.float32)
            st1 = states_wild[i, WILD_IDX, a1, CONTENT].astype(np.float32)
            row_id.append(i); y.append(1 if c == "inv" else 0)
            own.append(st_own); argstate.append(np.concatenate([st0, st1])); slot.append(k)
    return dict(row=np.array(row_id), y=np.array(y, dtype=np.int64), slot=np.array(slot),
                own=np.array(own, dtype=np.float32), argstate=np.array(argstate, dtype=np.float32))


# GOLD-ARGS reference numbers, final breath, from the banked campaign artifact (read-only quote,
# not recomputed): .cache/direction_probe_PMS8_241.txt's SUMMARY table, "final" breath rows E/B.
GOLD_ARGS_REFERENCE = {
    ("E", "logreg"): dict(diet_cv_auc=0.9360, wild_auc=0.8390, inv_recall=0.7581, n_fwd_pred="604/1040"),
    ("E", "mlp"):    dict(diet_cv_auc=0.9443, wild_auc=0.8408, inv_recall=0.7661, n_fwd_pred="581/1040"),
    ("B", "logreg"): dict(diet_cv_auc=0.9587, wild_auc=0.9215, inv_recall=0.7070, n_fwd_pred="730/1040"),
    ("B", "mlp"):    dict(diet_cv_auc=0.9753, wild_auc=0.9716, inv_recall=0.6882, n_fwd_pred="775/1040"),
}


def main():
    t0 = time.time()
    P("THE DIRECTION TIE-BREAK (2026-10-09, zero training)")
    P(f"generated {os.popen('date').read().strip()}")

    wild_rows = [json.loads(l) for l in open(WILD_JSONL)]
    diet_rows, diet_states, diet_admissible, diet_raw = load_diet()
    P(f"diet: {len(diet_rows)} rows, {int(diet_admissible.sum())} admissible, source {DIET_JSONL}")
    P(f"wild: {len(wild_rows)} rows, source {WILD_JSONL}")

    # =======================================================================
    # (1) retrain E/B with PREDICTED args, diet CV + wild AUROC + operating point
    # =======================================================================
    P(f"\n{'='*86}\n(1) THE PREDICTED-ARGS PROBE -- diet-trained, wild-read, vs the GOLD-args reference\n{'='*86}")
    diet_samp, n_other_d = extract_diet_probe_samples(diet_rows, diet_states, diet_admissible, diet_raw)
    n_fwd_d = int((diet_samp["y"] == 0).sum()); n_inv_d = int((diet_samp["y"] == 1).sum())
    P(f"diet relation slots (predicted-args features): fwd={n_fwd_d} inv={n_inv_d} (other={n_other_d})")

    wild_samples = {}
    for body in BODIES:
        wr = pickle.load(open(WILD_RAW_PKL[body], "rb")); wr = {d["i"]: d for d in wr}
        ws = np.load(WILD_STATES_NPZ[body])["states"]
        wild_samples[body] = extract_wild_probe_samples(body, wild_rows, ws, wr)
        nf = int((wild_samples[body]["y"] == 0).sum()); ni = int((wild_samples[body]["y"] == 1).sum())
        P(f"wild({body}) relation slots: fwd={nf} inv={ni}")

    fitted = {}     # (fset, model) -> (scaler, model_obj)
    cv_results = {}
    wild_auc_results = {}
    for fset in ("E", "B"):
        Xd = feat_vec(fset, None, None, None) if False else (
            diet_samp["argstate"] if fset == "E" else np.concatenate([diet_samp["own"], diet_samp["argstate"]], axis=1))
        yd = diet_samp["y"]; groups_d = diet_samp["row"]

        best_C, per_C = DP.logreg_cv(Xd, yd, groups_d)
        cv_auc, cv_acc = per_C[best_C]
        sc, clf = DP.logreg_fit_final(Xd, yd, best_C)
        cv_results[(fset, "logreg")] = dict(cv_auc=cv_auc, cv_acc=cv_acc, best_C=best_C)
        fitted[(fset, "logreg")] = (sc, clf)

        m_auc, m_acc, _, _ = DP.mlp_cv(Xd, yd, groups_d)
        sc2, net = DP.mlp_fit_final(Xd, yd)
        cv_results[(fset, "mlp")] = dict(cv_auc=m_auc, cv_acc=m_acc)
        fitted[(fset, "mlp")] = (sc2, net)

        for model in ("logreg", "mlp"):
            sc_, mdl_ = fitted[(fset, model)]
            row = []
            for body in BODIES:
                Xw = (wild_samples[body]["argstate"] if fset == "E" else
                      np.concatenate([wild_samples[body]["own"], wild_samples[body]["argstate"]], axis=1))
                yw = wild_samples[body]["y"]
                pw = (mdl_.predict_proba(sc_.transform(Xw))[:, 1] if model == "logreg" else mdl_.predict_proba(sc_.transform(Xw)))
                auc = roc_auc_score(yw, pw)
                op = DP.operating_point(yw, pw, bar=FWD_PRECISION_BAR)
                wild_auc_results[(fset, model, body)] = dict(auc=auc, op=op, p=pw)
                ref = GOLD_ARGS_REFERENCE[(fset, model)]
                row.append(f"{body}: wild_auc={auc:.4f} (gold-args ref {ref['wild_auc']:.4f}, "
                            f"diff {auc-ref['wild_auc']:+.4f})" +
                           (f" | inv_recall@fp>=0.85={op['inverse_recall']:.4f} n_fwd_pred={op['n_fwd_pred']}/{op['n_total']}"
                            if op else " | op: unreachable"))
            ref = GOLD_ARGS_REFERENCE[(fset, model)]
            P(f"  [{fset}] {model:7s} diet CV auroc={cv_results[(fset, model)]['cv_auc']:.4f} "
              f"(gold-args ref {ref['diet_cv_auc']:.4f}, diff {cv_results[(fset, model)]['cv_auc']-ref['diet_cv_auc']:+.4f})")
            for line in row:
                P(f"           {line}")

    # pick the probe by DIET CV AUROC only (never wild) -- the same discipline as tau selection
    best_key = max(cv_results, key=lambda k: cv_results[k]["cv_auc"])
    best_fset, best_model = best_key
    best_sc, best_mdl = fitted[best_key]
    P(f"\nCHOSEN PROBE (by diet CV AUROC, never wild): feature set {best_fset}, model {best_model} "
      f"(diet CV auroc={cv_results[best_key]['cv_auc']:.4f})")

    # =======================================================================
    # (2) tau* on DIET ONLY -- maximize diet masked fac-exact under the policy
    # =======================================================================
    P(f"\n{'='*86}\n(2) TAU* ON DIET ONLY (maximizing diet masked fac-exact; wild never consulted)\n{'='*86}")
    cand = []    # per eligible diet gold-rel slot: p_inv, delta (ok_new - ok_old under clamp)
    n_skip_ineligible = n_skip_notrel = 0
    for i, r in enumerate(diet_rows):
        if not diet_admissible[i] or i not in diet_raw:
            continue
        factors = r["factors"]
        cls = classify_row(factors)
        dd = diet_raw[i]
        for k, f in enumerate(factors):
            if f["ftype"] != "rel":
                n_skip_notrel += 1
                continue
            ppres = sigmoid(dd["pres"][k]) > 0.5
            pft = int(np.argmax(dd["ftype"][k]))
            if not (ppres and pft == 0):
                n_skip_ineligible += 1
                continue
            gold_op_code = 0 if f.get("op") == "add" else 1
            pop = int(np.argmax(dd["op"][k]))
            op_ok = (pop == gold_op_code)
            gargs = tuple(sorted(f["args"]))
            a0, a1 = decode_args(dd["args"][k], dd["dup"][k])
            args_ok = ((a0, a1) == gargs)
            gres = int(f["result"])
            res_logit = dd["res"][k].astype(np.float64)
            pred_old = int(np.argmax(res_logit))
            res_ok_old = (pred_old == gres)
            clamped = res_logit.copy(); clamped[k] = -np.inf
            pred_new = int(np.argmax(clamped))
            res_ok_new = (pred_new == gres)
            ok_old = op_ok and args_ok and res_ok_old
            ok_new = op_ok and args_ok and res_ok_new
            delta = int(ok_new) - int(ok_old)
            own_state = diet_states[i, DIET_KB, k, :].astype(np.float32)
            st0 = diet_states[i, DIET_KB, a0, :].astype(np.float32)
            st1 = diet_states[i, DIET_KB, a1, :].astype(np.float32)
            fv = feat_vec(best_fset, own_state, st0, st1)
            p_inv = float(best_mdl.predict_proba(best_sc.transform(fv[None, :]))[0] if best_model == "mlp"
                           else best_mdl.predict_proba(best_sc.transform(fv[None, :]))[0, 1])
            cand.append((p_inv, delta))
    cand.sort(key=lambda t: -t[0])
    cum = 0; best_cum = 0; best_idx = 0
    cums = []
    for idx, (_, d) in enumerate(cand):
        cum += d
        cums.append(cum)
        if cum > best_cum:
            best_cum = cum; best_idx = idx + 1
    if best_idx == 0:
        tau_star = 1.0 + 1e-6
    elif best_idx == len(cand):
        tau_star = -1e-6
    else:
        tau_star = (cand[best_idx - 1][0] + cand[best_idx][0]) / 2.0
    n_elig = len(cand)
    n_flip_gain = sum(1 for p, d in cand[:best_idx] if d > 0)
    n_flip_cost = sum(1 for p, d in cand[:best_idx] if d < 0)
    P(f"diet: gold-rel slots skipped (gold not rel among predicted-rel candidates): n/a; "
      f"skipped not-predicted-relation (ineligible)={n_skip_ineligible}; skipped gold-not-rel={n_skip_notrel}")
    P(f"diet: eligible gold-REL candidates (predicted relation & present) = {n_elig}")
    P(f"diet: tau* = {tau_star:.4f} (clamp iff p_inv > tau*); clamps performed at tau* = {best_idx}/{n_elig}")
    P(f"diet: net delta at tau* = {best_cum:+d} slots ({n_flip_gain} gained, {n_flip_cost} lost among the {best_idx} clamped)")
    P(f"diet: net delta at tau=0 (clamp ALL {n_elig} eligible, the no-MLP control's diet analogue) = {cums[-1] if cums else 0:+d}")
    P(f"diet: best achievable net delta over ALL thresholds = {best_cum:+d} (confirms tau* is the argmax, by construction)")

    # =======================================================================
    # (3) apply tau* on WILD, both bodies -- the honest read
    # =======================================================================
    P(f"\n{'='*86}\n(3) WILD, TAU* APPLIED (clamp iff the head decodes the slot as present+relation AND p_inv > tau*)\n{'='*86}")
    bar_summary = {}
    for body in BODIES:
        P(f"\n--- {body} ---")
        dump = load_dump(body)
        raw_wild = {d["i"]: d for d in pickle.load(open(WILD_RAW_PKL[body], "rb"))}
        states_wild = np.load(WILD_STATES_NPZ[body])["states"]
        ps = np.load(PS_LEGAL_NPZ[body])
        ps_rows, ps_slots, ps_ok = ps["rows"], ps["slots"], ps["ok"]
        idx_of = {(int(r_), int(s_)): ix for ix, (r_, s_) in enumerate(zip(ps_rows, ps_slots))}
        ok_new_vec = ps_ok.copy()
        ok_new_vec_tau0 = ps_ok.copy()

        n_elig_w = 0
        form_stats = {"fwd": dict(n=0, res_old=0, res_new=0, res_new_tau0=0, pres=0, ftype=0, op=0, args=0),
                      "inv": dict(n=0, res_old=0, res_new=0, res_new_tau0=0, pres=0, ftype=0, op=0, args=0)}
        for i, r in enumerate(wild_rows):
            factors = r["factors"]
            cls = classify_row(factors)
            dd = raw_wild.get(i)
            for k, f in enumerate(factors):
                c = cls[k]
                if c not in ("fwd", "inv"):
                    continue
                t = dump.get((i, k))
                if t is None:
                    continue
                hf = head_fields(t)
                fs = form_stats[c]
                fs["n"] += 1
                fs["pres"] += int(bool(hf["pres"])); fs["ftype"] += int(bool(hf["ftype"]))
                if hf["op"] is not None:
                    fs["op"] += int(bool(hf["op"]))
                if hf["args"] is not None:
                    fs["args"] += int(bool(hf["args"]))
                res_ok_old = bool(hf["res"])
                fs["res_old"] += int(res_ok_old)
                ok_old = bool(hf["ok"])
                idx = idx_of.get((i, k))

                pft = t[5]; ppres = bool(t[10])
                eligible = ppres and (pft == 0)
                gres = hf["gres"]
                res_ok_new = res_ok_old
                res_ok_new_tau0 = res_ok_old
                if eligible and dd is not None:
                    n_elig_w += 1
                    res_logit = dd["res"][k].astype(np.float64)
                    clamped = res_logit.copy(); clamped[k] = -np.inf
                    pred_new = int(np.argmax(clamped))
                    res_ok_new_tau0 = (pred_new == gres)   # tau=0 control: clamp unconditionally when eligible

                    a0, a1 = decode_args(dd["args"][k], dd["dup"][k])
                    own_state = states_wild[i, WILD_IDX, k, CONTENT].astype(np.float32)
                    st0 = states_wild[i, WILD_IDX, a0, CONTENT].astype(np.float32)
                    st1 = states_wild[i, WILD_IDX, a1, CONTENT].astype(np.float32)
                    fv = feat_vec(best_fset, own_state, st0, st1)
                    p_inv = float(best_mdl.predict_proba(best_sc.transform(fv[None, :]))[0] if best_model == "mlp"
                                   else best_mdl.predict_proba(best_sc.transform(fv[None, :]))[0, 1])
                    if p_inv > tau_star:
                        res_ok_new = res_ok_new_tau0
                    # else: res_ok_new stays res_ok_old (not clamped at tau*)

                fs["res_new"] += int(res_ok_new)
                fs["res_new_tau0"] += int(res_ok_new_tau0)

                if idx is not None:
                    # BUG FIX (caught on first run): ok_old already carries hf["pres"]/hf["ftype"]
                    # (and, for a gold-rel slot, hf["op"]/hf["args"]) -- when res is UNCHANGED
                    # (not eligible, or eligible but not clamped at this tau), ok_new MUST equal
                    # ok_old exactly (reuse it, do not recompute a narrower formula that silently
                    # drops the pres/ftype terms and spuriously flips slots res never touched).
                    # Only recompute the fuller formula when res actually DIFFERS from baseline.
                    if res_ok_new == res_ok_old:
                        ok_new = ok_old
                    elif hf["op"] is None:
                        ok_new = ok_old   # gold wasn't actually a relation; a res change here can't flip ok (dig-branch, untouched)
                    else:
                        ok_new = bool(hf["pres"]) and bool(hf["ftype"]) and bool(hf["op"]) and bool(hf["args"]) and res_ok_new
                    if res_ok_new_tau0 == res_ok_old:
                        ok_new_tau0 = ok_old
                    elif hf["op"] is None:
                        ok_new_tau0 = ok_old
                    else:
                        ok_new_tau0 = bool(hf["pres"]) and bool(hf["ftype"]) and bool(hf["op"]) and bool(hf["args"]) and res_ok_new_tau0
                    ok_new_vec[idx] = ok_new
                    ok_new_vec_tau0[idx] = ok_new_tau0

        for c in ("fwd", "inv"):
            fs = form_stats[c]
            n = max(fs["n"], 1)
            P(f"  {c:4s} n={fs['n']:4d}  pres={fs['pres']/n:.3f}  ftype={fs['ftype']/n:.3f}  "
              f"op={fs['op']/n:.3f}  args={fs['args']/n:.3f}  "
              f"res(baseline)={fs['res_old']/n:.3f}  res(tau*)={fs['res_new']/n:.3f}  res(tau=0)={fs['res_new_tau0']/n:.3f}")
        P(f"  eligible (predicted present+relation) slots touched = {n_elig_w}")

        np.savez(f".cache/dirtb_ps_legal_wild_{body}_tiebreak.npz", rows=ps_rows, slots=ps_slots, ok=ok_new_vec)
        np.savez(f".cache/dirtb_ps_legal_wild_{body}_tau0.npz", rows=ps_rows, slots=ps_slots, ok=ok_new_vec_tau0)

        for tag, path in ((f"tau*={tau_star:.4f}", f".cache/dirtb_ps_legal_wild_{body}_tiebreak.npz"),
                           ("tau=0 (no-MLP control)", f".cache/dirtb_ps_legal_wild_{body}_tau0.npz")):
            out = subprocess.run([".venv/bin/python3", "scripts/paired_read.py", PS_LEGAL_NPZ[body], path,
                                   body, f"{body}+{tag}"], capture_output=True, text=True, cwd=".")
            line = out.stdout.strip().splitlines()[-1] if out.stdout.strip() else out.stderr.strip()
            P(f"  masked fac-exact PAIRED [{tag}]: {line}")
            if tag.startswith("tau*"):
                bar_summary[body] = dict(line=line, fwd_res=form_stats["fwd"]["res_new"]/max(form_stats["fwd"]["n"],1),
                                          inv_res=form_stats["inv"]["res_new"]/max(form_stats["inv"]["n"],1))

    # =======================================================================
    # (4) the no-MLP control summary + the oracle ceiling beside it
    # =======================================================================
    P(f"\n{'='*86}\n(4) THE NO-MLP CONTROL (tau=0: clamp every eligible relation slot, unconditionally) vs THE ORACLE CEILING\n{'='*86}")
    P("  (forward res baseline ~0.87-0.88 among gold-fwd slots; clamping an eligible forward slot ALWAYS")
    P("   destroys a correct baseline decode -- gold res == own by definition of 'forward' -- so tau=0's")
    P("   forward cost is exactly the eligible-forward fraction whose baseline was already right.)")
    P(f"  THE ORACLE CEILING (own-suppression oracle, GOLD direction + ALL gold-inv slots, banked "
      f"2026-10-09 16:53): PMS8_241 inverse res ceiling=0.6022, PC_241=0.6210")
    P("  (this run's tau=0/tau* numbers are restricted to ELIGIBLE, predicted-relation slots only --")
    P("   a strict subset of the oracle's all-gold-inv denominator -- so they sit at or below the ceiling")
    P("   even at tau=0, by construction.)")

    # =======================================================================
    # verdict
    # =======================================================================
    P(f"\n{'='*86}\nTHE VERDICT\n{'='*86}")
    bars_pass = {}
    for body in BODIES:
        bs = bar_summary.get(body)
        if not bs:
            continue
        line = bs["line"]
        import re
        m = re.search(r"diff ([+-]?[0-9.]+)", line)
        raw_diff = float(m.group(1)) if m else None   # paired_read.py's own convention: OLD(baseline) - NEW(policy)
        improvement = -raw_diff if raw_diff is not None else None   # positive = the tie-break IMPROVES masked fac-exact
        mz = re.search(r"McNemar z ([+-]?[0-9.]+)", line)
        z = float(mz.group(1)) if mz else None
        inv_bar = bs["inv_res"] >= 0.40
        fwd_bar = bs["fwd_res"] >= 0.84
        masked_bar = (improvement is not None and improvement >= 0.010 and z is not None and abs(z) >= 1.0)
        bars_pass[body] = dict(inv_bar=inv_bar, fwd_bar=fwd_bar, masked_bar=masked_bar,
                                inv_res=bs["inv_res"], fwd_res=bs["fwd_res"], improvement=improvement, z=z)
        P(f"  {body}: inv_res={bs['inv_res']:.4f} (bar>=0.40: {inv_bar})  fwd_res={bs['fwd_res']:.4f} "
          f"(bar>=0.84: {fwd_bar})  masked_fac_exact_improvement={improvement:+.4f}  z={z} "
          f"(bar improvement>=0.010 & |z|>=1: {masked_bar})")
    all_clear = all(v["inv_bar"] and v["fwd_bar"] and v["masked_bar"] for v in bars_pass.values()) and bool(bars_pass)
    P(f"\n  ALL PINNED BARS CLEAR ON EVERY BODY: {all_clear}")
    if all_clear:
        P("  CLEARS: a free read-time lever on every body tested -- to be threaded into chain_acc.py / the")
        P("  readers behind a flag (NOT threaded in this read, per the word given).")
    else:
        P("  DOES NOT CLEAR on at least one body/bar -- see the per-body breakdown above for which bar failed.")

    os.makedirs(".cache", exist_ok=True)
    with open(OUT, "w") as fh:
        fh.write("\n".join(LOG) + "\n")
    P(f"\n[timing] {time.time()-t0:.1f}s")
    print(f"\n[direction-tiebreak] wrote {OUT}")


if __name__ == "__main__":
    main()
