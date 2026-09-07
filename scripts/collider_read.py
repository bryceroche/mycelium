"""collider_read.py — RUNG 3: THE COLLIDER READ (2026-09-07; ledger
"GUT REGISTERED (the 22nd): THE GODOL COLLIDER" pins the bars).

Zero-training READ. For each fixture item: take the head's PASS-0 (open)
outputs, decode the committed parse exactly as the commit adapter does
(`phase1_algebra_head._alt2_fact_buf_v1`'s decode phase, reproduced in
numpy here — the adapter itself cannot be reused because it returns only
the fact BUFFER, and the collider needs the classification), enumerate
SINGLE-SLOT rival readings at torn slots, collide each candidate with the
rest of the graph through the exact propagator (`alternator_bridge.ping`
— GAC only, contradiction = mass None), and classify what emerges:

  CONTRADICTORY   ping returns mass None (the reading kills itself)
  UNIQUE          the query var is a forced singleton in facts
  UNDERDETERMINED consistent, query not forced

FLIP RULE (pinned, decided WITHOUT the key): flip iff the original is
CONTRADICTORY and EXACTLY ONE alternative is UNIQUE; the flipped reading
is that alternative. The answer key enters ONLY afterwards, to grade.

BARS (pinned before measurement, ledger 2026-09-07):
  (1) COLLIDER VISIBILITY  >= 0.30 of WRONG originals are CONTRADICTORY
      (KILL < 0.10)
  (2) flip precision       >= 0.60
  (3) net recovery         >= 0.05 of wrong originals
  (4) silent-wrong introduced <= 0.01 of items

GOODHART NOTE: every number here is a READ. Nothing printed may enter a
loss or a training-data selection criterion (diagnostic register).

Envs: CL_TEST = wild (default; .cache/wild_admitted_holdout.jsonl,
ALG_TEST_NAME=wildhold) | mint (.cache/algebra_nl_test.jsonl, test23);
CL_CKPT (default .cache/sharp_fedon242.safetensors); CL_N (0 = all rows);
CL_B (batch, default 32); CL_THETA (commit threshold, default 0.9);
CL_MARGIN (ftype-rival margin, default 0.3); CL_MAXALT (default 12).
DEV comes from the caller via setdefault("PCI+AMD"); CPU test:
  DEV=CPU CL_N=4 CL_B=4 CL_TEST=wild .venv/bin/python3 scripts/collider_read.py
`--selftest` runs the pure-python pieces on faked numpy inputs (no GPU,
no ckpt, no fixture).
"""
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "scripts"))

_FIXTURES = {"wild": (".cache/wild_admitted_holdout.jsonl", "wildhold"),
             "mint": (".cache/algebra_nl_test.jsonl", "test23")}
CL_TEST = os.environ.get("CL_TEST", "wild")
_FPATH, _FNAME = _FIXTURES.get(CL_TEST, (CL_TEST, os.path.basename(CL_TEST)))

os.environ.setdefault("DEV", "PCI+AMD")
# Champion env stack — stamp_amplitude_read.py's ENV verbatim MINUS
# ALG_PC_MIX (no pressure-cooker seal on a read), with the fixture
# swapped in. Forced (update, as the amplitude read does): a read must
# carry the TRAINED env or the ckpt keys will not match build_params.
ENV = {"ALG2": "1", "ALG_FTYPES": "9", "ALG_DUP": "1", "ALG_HW": "512",
       "ALG_WIDE": "1", "ALG_BREATH": "7", "ALG_NOTEBOOK": "1",
       "ALG_SIXWAVE": "1", "NB_PERSLOT": "1", "ALG_BINDBUS": "7",
       "ALG_BIND_D": "512", "BIND_CODES": ".cache/bindbus_codes512.npz",
       "ALG_BUSGARAGE": "2", "ALG_SHELF_CIRCLE": "2", "ALG_ALTMASK": "1",
       "ALG_ALT21": "1", "ALG_ALT2": "1", "ALG_MASKHEAD": "1",
       "ALG_FED": "1", "ALG_TEST": _FPATH, "ALG_TEST_NAME": _FNAME}
os.environ.update(ENV)
os.environ.pop("SC_EVAL", None)

import json                                          # noqa: E402
import numpy as np                                   # noqa: E402

from phase1_algebra_head import K_VARS, L_FAC, N_DIG  # CPU-safe import

CKPT = os.environ.get("CL_CKPT", ".cache/sharp_fedon242.safetensors")
THETA = float(os.environ.get("CL_THETA", "0.9"))
MARGIN = float(os.environ.get("CL_MARGIN", "0.3"))
MAXALT = int(os.environ.get("CL_MAXALT", "12"))
CL_N = int(os.environ.get("CL_N", "0"))
CL_B = int(os.environ.get("CL_B", "32"))
OUT = f".cache/collider_read_{_FNAME}.json"

FT_REL, FT_GIVEN = 0, 1                    # bridge-committable ftypes


# ===========================================================================
# DECODE — the commit adapter's decode phase (_alt2_fact_buf_v1), in numpy
# ===========================================================================

def _sig(x):
    return 1.0 / (1.0 + np.exp(-x))


def _smax(x):
    e = np.exp(x - x.max(-1, keepdims=True))
    return e / e.sum(-1, keepdims=True)


def decode_batch(onp, theta):
    """Whole-batch decode, term for term as _alt2_fact_buf_v1 does it.
    Returns a dict of (B, L, ...) arrays + the adapter's `keep` mask (the
    slots that WOULD be committed to the bridge)."""
    D = {}
    D["pres_sig"] = _sig(onp["pres"])
    ftp = _smax(onp["ftype"])
    D["ft_p"] = ftp
    D["ft_am"] = ftp.argmax(-1)
    D["ft_conf"] = np.take_along_axis(ftp, D["ft_am"][..., None], -1)[..., 0]
    rsp = _smax(onp["res"])
    D["res_p"] = rsp
    D["res_am"] = rsp.argmax(-1)
    D["res_conf"] = np.take_along_axis(rsp, D["res_am"][..., None], -1)[..., 0]
    D["agp"] = _sig(onp["args"])                       # BCE 2-hot sigmoids
    D["op_am"] = onp["op"].argmax(-1)
    digs = onp["dig"].argmax(-1)
    D["digs"] = digs
    place = (10 ** np.arange(N_DIG - 1, -1, -1)).astype(np.int64)
    gv = (digs.astype(np.int64) * place).sum(-1)
    if "sgn" in onp:                                   # E1 negative literal
        gv = np.where(onp["sgn"] > 0, -gv, gv)
    D["given_val"] = gv
    has_dup = "dup" in onp
    D["has_dup"] = has_dup
    D["dup_on"] = ((onp["dup"] > 0) if has_dup
                   else np.zeros_like(D["pres_sig"], dtype=bool))
    if "dargs" in onp:                                 # door #12 dup pointer
        dsp = _smax(onp["dargs"])
        D["dup_rank"] = np.argsort(-dsp, axis=-1)      # (B, L, K)
        D["dup_conf"] = dsp
    else:
        D["dup_rank"] = np.argsort(-onp["args"], axis=-1)   # raw-logit order
        D["dup_conf"] = D["agp"]
    a0 = D["dup_rank"][..., 0]
    D["a0"] = a0
    D["a0_conf"] = np.take_along_axis(D["dup_conf"], a0[..., None], -1)[..., 0]
    rank = np.argsort(-onp["args"], axis=-1)           # (B, L, K) raw logits
    D["arg_rank"] = rank
    top2 = rank[..., :2]
    D["top2"] = top2
    D["top2_conf_min"] = np.take_along_axis(D["agp"], top2, axis=-1).min(-1)

    active = ((D["pres_sig"] > theta) & (D["ft_conf"] > theta)
              & (D["res_conf"] > theta))
    is_given = active & (D["ft_am"] == FT_GIVEN)
    is_rel = active & (D["ft_am"] == FT_REL)
    rel_dup_ok = is_rel & D["dup_on"] & (D["a0_conf"] > theta)
    rel_nondup_ok = is_rel & (~D["dup_on"]) & (D["top2_conf_min"] > theta)
    D["keep"] = is_given | rel_dup_ok | rel_nondup_ok
    return D


def fac_at(D, bi, j, ft=None, res=None, args=None):
    """One factor dict in BRIDGE GRAMMAR from slot j's decode, with
    optional single-field overrides (the rival reading). Returns None if
    the (overridden) ftype is not committable (only rel/given ride)."""
    ftv = int(D["ft_am"][bi, j]) if ft is None else int(ft)
    resv = int(D["res_am"][bi, j]) if res is None else int(res)
    if ftv == FT_GIVEN:
        return {"ftype": "given", "var": resv,
                "value": int(D["given_val"][bi, j])}
    if ftv == FT_REL:
        if args is None:
            if D["dup_on"][bi, j]:
                a = int(D["a0"][bi, j])
                ar = [a, a]
            else:
                ar = sorted(int(a) for a in D["top2"][bi, j])
        else:
            ar = [int(args[0]), int(args[1])]
            ar = ar if ar[0] == ar[1] else sorted(ar)
        return {"ftype": "rel",
                "op": "add" if D["op_am"][bi, j] == 0 else "mul",
                "args": ar, "result": resv}
    return None


def nv_for(facs, row_nv):
    """do_eval's / the adapter's n_vars convention: the row's n_vars, or
    one past the highest variable the committed factors reference."""
    return max([int(row_nv)]
               + [v + 1 for f in facs for v in
                  ([f["var"]] if f["ftype"] == "given"
                   else list(f["args"]) + [f["result"]])])


# ===========================================================================
# THE COLLIDER — alternatives, collision, classification
# ===========================================================================

def alternatives(D, bi, theta, margin, maxalt):
    """Single-slot rival readings at TORN slots. Each entry is
      {"slot": j, "mode": "replace"|"add", "kind": str, "gap": float,
       "fac": <bridge-grammar factor>, "ov": {overrides for grading}}
    Torn slots = every slot with presence sigmoid > 0.5: those the adapter
    COMMITS get REPLACE alternatives (second-best args / result / ftype);
    those it does not (below the presence gate, or gated out on ftype/res/
    arg confidence) get ADD alternatives (the slot's own reading, admitted).
    Ranked by `gap` (the probability distance between the committed choice
    and its rival — small gap = torn) and capped at maxalt."""
    alts = []
    keep = D["keep"][bi]
    for j in range(L_FAC):
        if D["pres_sig"][bi, j] <= 0.5:
            continue
        ft = int(D["ft_am"][bi, j])
        rp = D["res_p"][bi, j]
        r_rank = np.argsort(-rp)
        res2 = int(r_rank[1])
        res_gap = float(rp[r_rank[0]] - rp[res2])
        if not keep[j]:
            # ADD-mode: the gate suppressed this slot; the rival reading is
            # "admit it". gap = how far the weakest gate fell short of theta.
            base = fac_at(D, bi, j)
            if base is None:
                continue
            deficit = theta - min(float(D["pres_sig"][bi, j]),
                                  float(D["ft_conf"][bi, j]),
                                  float(D["res_conf"][bi, j]))
            alts.append({"slot": j, "mode": "add", "kind": "add_top",
                         "gap": max(deficit, 0.0), "fac": base, "ov": {}})
            f2 = fac_at(D, bi, j, res=res2)
            if f2 is not None:
                alts.append({"slot": j, "mode": "add", "kind": "add_res2",
                             "gap": max(deficit, 0.0) + res_gap, "fac": f2,
                             "ov": {"res": res2}})
            continue
        # REPLACE-mode: this slot is in the committed parse.
        f2 = fac_at(D, bi, j, res=res2)
        if f2 is not None:
            alts.append({"slot": j, "mode": "replace", "kind": "res2",
                         "gap": res_gap, "fac": f2, "ov": {"res": res2}})
        if ft == FT_REL:
            if D["dup_on"][bi, j]:
                a2 = int(D["dup_rank"][bi, j, 1])
                dc = D["dup_conf"][bi, j]
                g = float(dc[D["a0"][bi, j]] - dc[a2])
                fa = fac_at(D, bi, j, args=(a2, a2))
                alts.append({"slot": j, "mode": "replace", "kind": "dup_a2",
                             "gap": g, "fac": fa,
                             "ov": {"args": (a2, a2)}})
            else:
                t0, t1 = (int(x) for x in D["top2"][bi, j])
                t2 = int(D["arg_rank"][bi, j, 2])
                ag = D["agp"][bi, j]
                # swap the WEAKER of the top-2 for the third-ranked var,
                # then the stronger (both are single-slot arg rivals)
                alts.append({"slot": j, "mode": "replace",
                             "kind": "args_swap_weak",
                             "gap": float(ag[t1] - ag[t2]),
                             "fac": fac_at(D, bi, j, args=(t0, t2)),
                             "ov": {"args": (t0, t2)}})
                alts.append({"slot": j, "mode": "replace",
                             "kind": "args_swap_strong",
                             "gap": float(ag[t0] - ag[t2]),
                             "fac": fac_at(D, bi, j, args=(t1, t2)),
                             "ov": {"args": (t1, t2)}})
        else:                                   # given slot: the ftype rival
            fp = D["ft_p"][bi, j]
            f_rank = np.argsort(-fp)
            ft2 = int(f_rank[1])
            ft_gap = float(fp[f_rank[0]] - fp[ft2])
            if ft2 == FT_REL and ft_gap <= margin:
                fa = fac_at(D, bi, j, ft=FT_REL)
                if fa is not None:
                    alts.append({"slot": j, "mode": "replace",
                                 "kind": "ftype2_rel", "gap": ft_gap,
                                 "fac": fa, "ov": {"ft": FT_REL}})
    alts = [a for a in alts if a["fac"] is not None]
    alts.sort(key=lambda a: (a["gap"], a["slot"], a["kind"]))
    return alts[:maxalt]


def classify(nv, facs, m, qv):
    """Collide one candidate reading with the exact propagator.
    Returns (label, n_facts). Never sees the key."""
    from alternator_bridge import ping
    try:
        facts, mass, _r = ping(nv, facs, int(m))
    except Exception:
        return "ERROR", 0
    if mass is None:
        return "CONTRADICTORY", 0
    if int(qv) in facts:
        return "UNIQUE", len(facts)
    return "UNDERDETERMINED", len(facts)


def collide_item(D, bi, row_nv, m, qv, theta, margin, maxalt):
    """The full per-item collision. Returns a record dict; the key is NOT
    consulted anywhere in here (consistency is checkable without it)."""
    keep_j = [int(j) for j in np.nonzero(D["keep"][bi])[0]]
    facs = [fac_at(D, bi, j) for j in keep_j]
    pos = {j: i for i, j in enumerate(keep_j)}
    facs = [f for f in facs if f is not None]
    o_lab, o_nf = classify(nv_for(facs, row_nv), facs, m, qv)
    alts = alternatives(D, bi, theta, margin, maxalt)
    a_recs = []
    for a in alts:
        cand = list(facs)
        if a["mode"] == "replace" and a["slot"] in pos:
            cand[pos[a["slot"]]] = a["fac"]
        else:
            cand = cand + [a["fac"]]
        lab, nf = classify(nv_for(cand, row_nv), cand, m, qv)
        a_recs.append({"slot": a["slot"], "kind": a["kind"],
                       "mode": a["mode"], "gap": round(float(a["gap"]), 6),
                       "label": lab, "n_facts": nf, "ov": a["ov"]})
    uniq = [r for r in a_recs if r["label"] == "UNIQUE"]
    survivors = len(uniq) + (1 if o_lab == "UNIQUE" else 0)
    flip = None
    if o_lab == "CONTRADICTORY" and len(uniq) == 1:    # THE FLIP RULE
        flip = uniq[0]
    return {"orig_label": o_lab, "orig_facts": o_nf, "n_committed": len(facs),
            "n_alts": len(a_recs), "survivors": survivors,
            "alts": a_recs, "flip": flip}


# ===========================================================================
# GRADING — score_batch's criterion, per slot (the key enters HERE, only)
# ===========================================================================

def slot_rec(onp, bi, j, has_dup):
    """score_batch's own decode of slot j (pres>0, argmaxes, raw-logit
    top-2), as a record the flip's overrides can be applied to."""
    args = onp["args"][bi, j]
    top2 = np.argsort(-args)[:2]
    return {"pres_ok": bool(onp["pres"][bi, j] > 0),
            "ft": int(onp["ftype"][bi, j].argmax()),
            "res": int(onp["res"][bi, j].argmax()),
            "op": int(onp["op"][bi, j].argmax()),
            "dup": bool(onp["dup"][bi, j] > 0) if has_dup else False,
            "a_argmax": int(np.argmax(args)),
            "top2": set(int(x) for x in top2),
            "dig": onp["dig"][bi, j].argmax(-1)}


def apply_ov(rec, ov):
    """The flipped slot's record: the same decode with ONE field replaced."""
    r = dict(rec)
    r["top2"] = set(rec["top2"])
    if "ft" in ov:
        r["ft"] = int(ov["ft"])
    if "res" in ov:
        r["res"] = int(ov["res"])
    if "args" in ov:
        a0, a1 = int(ov["args"][0]), int(ov["args"][1])
        r["top2"] = {a0, a1}
        r["a_argmax"] = a0
        r["dup"] = (a0 == a1)
    return r


def slot_ok(rec, vg, i, j, has_dup):
    """score_batch's per-slot criterion, mirrored line for line (dup branch
    included). Only called on slots where gold presence >= 0.5."""
    ok = rec["pres_ok"]
    ok &= rec["ft"] == vg["ftype"][i, j]
    ok &= rec["res"] == vg["res"][i, j]
    if vg["ftype"][i, j] == 0:
        ok &= rec["op"] == vg["op"][i, j]
        gset = set(np.where(vg["args"][i, j] > .5)[0].tolist())
        if len(gset) == 1 and has_dup:
            ok &= rec["dup"]
            ok &= rec["a_argmax"] in gset
        else:
            ok &= rec["top2"] == gset
    else:
        ok &= bool((rec["dig"] == vg["digits"][i, j]).all())
    return bool(ok)


def grade_flip(per, rec, ov, vg, i, j, has_dup):
    """The flip's slot-level grade: (was_ok, now_ok). was_ok is None when
    the flipped slot carries NO gold factor — such a flip can neither be
    correct nor destroy a correct slot."""
    if j not in per:
        return None, False
    return per[j], slot_ok(apply_ov(rec, ov), vg, i, j, has_dup)


def grade_item(onp, bi, vg, i, has_dup):
    """Per-slot ok over the item's GOLD-present slots (the mirror), plus
    the item-level fac-exact verdict (all gold slots ok)."""
    per = {}
    for j in range(L_FAC):
        if vg["presence"][i, j] < 0.5:
            continue
        per[j] = slot_ok(slot_rec(onp, bi, j, has_dup), vg, i, j, has_dup)
    return per, (len(per) > 0 and all(per.values()))


# ===========================================================================
# MAIN
# ===========================================================================

def main():
    from phase1_algebra_head import build_params, forward, load_alg
    from step_engine_read import score_batch
    from tinygrad import Tensor, dtypes
    from tinygrad.nn.state import safe_load

    vs, vst, vtk, vg, vse = load_alg("test")
    n_all = len(vs)
    n_take = min(CL_N, n_all) if CL_N > 0 else n_all
    # ORDERING ASSERTION: load_alg's samples list is the fixture jsonl in
    # file order and the gold arrays are row-aligned to it — so
    # vs[i]["query_var"] IS the gold query of gold row i. Checked, never
    # assumed (g_query is the states file's own copy).
    for i in range(n_take):
        assert int(vg["query"][i]) == int(vs[i]["query_var"]), (
            f"row {i}: samples query_var {vs[i]['query_var']} != gold "
            f"query {int(vg['query'][i])} — fixture/gold desync")
    print(f"[collider] fixture={_FNAME} ({_FPATH}) rows={n_take}/{n_all} "
          f"ckpt={CKPT} theta={THETA} margin={MARGIN} maxalt={MAXALT}",
          flush=True)

    p = build_params(0)
    sd = safe_load(CKPT)
    assert set(sd.keys()) == set(p.keys()), \
        (sorted(set(sd) - set(p))[:4], sorted(set(p) - set(sd))[:4])
    for k in p:
        p[k].assign(sd[k].to(p[k].device).cast(p[k].dtype)).realize()

    recs = []
    n_items = 0
    wrong = contra_wrong = 0
    hist_lab = {}                  # (label, correct?) -> count
    hist_surv = {}
    flips = flips_ok = silent = flips_item_fixed = 0
    mirror_ok = mirror_tot = sb_ok = sb_tot = 0

    for s0 in range(0, n_take, CL_B):
        sl = np.arange(s0, min(s0 + CL_B, n_take))
        pad = CL_B - len(sl)
        sl_p = np.concatenate([sl, sl[:1].repeat(pad)]) if pad else sl
        ts = Tensor(vst[sl_p].astype(np.float32), dtype=dtypes.float)
        tk = Tensor(vtk[sl_p].astype(np.float32), dtype=dtypes.float)
        se = Tensor(vse[sl_p].astype(np.int32), dtype=dtypes.int)
        o = forward(p, ts, tk, se)                       # PASS 0: open
        kset = ["pres", "ftype", "op", "args", "res", "dig"]
        kset += [k for k in ("dup", "sgn", "dargs") if k in o]
        onp = {k: o[k].realize().numpy() for k in kset}
        has_dup = "dup" in onp
        D = decode_batch(onp, THETA)

        b_ok, b_tot = score_batch(onp, vg, sl)           # the authority
        sb_ok += b_ok
        sb_tot += b_tot
        for bi, i in enumerate(sl):
            i = int(i)
            row = vs[i]
            per, item_ok = grade_item(onp, bi, vg, i, has_dup)
            mirror_ok += sum(1 for v in per.values() if v)
            mirror_tot += len(per)
            r = collide_item(D, bi, int(row.get("n_vars", K_VARS)),
                             int(row.get("m", 0)), int(row["query_var"]),
                             THETA, MARGIN, MAXALT)
            r["idx"] = i
            r["query_var"] = int(row["query_var"])
            r["orig_correct"] = bool(item_ok)
            r["n_gold_slots"] = len(per)
            n_items += 1
            key = (r["orig_label"], "correct" if item_ok else "wrong")
            hist_lab[key] = hist_lab.get(key, 0) + 1
            hist_surv[r["survivors"]] = hist_surv.get(r["survivors"], 0) + 1
            if not item_ok:
                wrong += 1
                if r["orig_label"] == "CONTRADICTORY":
                    contra_wrong += 1
            fl = r["flip"]
            if fl is not None:
                j = fl["slot"]
                flips += 1
                was, now = grade_flip(per, slot_rec(onp, bi, j, has_dup),
                                      fl["ov"], vg, i, j, has_dup)
                r["flip_slot_was_ok"] = None if was is None else bool(was)
                r["flip_slot_now_ok"] = bool(now)
                flips_ok += int(now)
                silent += int(bool(was) and not now)
                if now and not was:
                    # item-level: does the whole reading become exact?
                    flips_item_fixed += int(all(v for jj, v in per.items()
                                                if jj != j))
            recs.append(r)

    assert (mirror_ok, mirror_tot) == (sb_ok, sb_tot), \
        ("per-slot mirror disagrees with score_batch",
         mirror_ok, mirror_tot, sb_ok, sb_tot)

    # ---------------- the report ----------------
    def bar(name, val, cmp_, thr, kill=None):
        if val is None:
            v = "n/a"
            verdict = "N/A"
        else:
            v = f"{val:.4f}"
            good = (val >= thr) if cmp_ == ">=" else (val <= thr)
            verdict = "PASS" if good else "FAIL"
            if kill is not None and val < kill:
                verdict = "KILL"
        print(f"[collider] BAR {name}: {v} (bar {cmp_} {thr}"
              + (f", KILL < {kill}" if kill is not None else "")
              + f") -> {verdict}", flush=True)
        return verdict

    vis = (contra_wrong / wrong) if wrong else None
    prec = (flips_ok / flips) if flips else None
    net = (flips_ok / wrong) if wrong else None
    sw = silent / max(n_items, 1)

    print(f"[collider] items={n_items} fac-exact(items)="
          f"{(n_items - wrong) / max(n_items, 1):.4f} wrong={wrong} "
          f"slot fac-exact={sb_ok}/{sb_tot}={sb_ok / max(sb_tot, 1):.4f} "
          f"(mirror agrees with score_batch)", flush=True)
    print("[collider] original classification (contradictory/unique/"
          "underdetermined x correct/wrong):", flush=True)
    for lab in ("CONTRADICTORY", "UNIQUE", "UNDERDETERMINED", "ERROR"):
        c = hist_lab.get((lab, "correct"), 0)
        w = hist_lab.get((lab, "wrong"), 0)
        if c or w:
            print(f"[collider]   {lab:<16} correct={c:<5} wrong={w:<5} "
                  f"total={c + w}", flush=True)
    print("[collider] survivor-set histogram (candidates incl. original "
          "that are UNIQUE):", flush=True)
    for k in sorted(hist_surv):
        print(f"[collider]   survivors={k:<3} items={hist_surv[k]}",
              flush=True)
    print(f"[collider] flips attempted={flips} correct={flips_ok} "
          f"silent-wrong={silent} item-level fixes={flips_item_fixed}",
          flush=True)

    v1 = bar("visibility (wrong originals contradictory)", vis, ">=", 0.30,
             kill=0.10)
    v2 = bar("flip precision", prec, ">=", 0.60)
    v3 = bar("net recovery (correct flips / wrong originals)", net, ">=", 0.05)
    v4 = bar("silent-wrong introduced / items", sw, "<=", 0.01)

    payload = {"fixture": _FNAME, "path": _FPATH, "ckpt": CKPT,
               "theta": THETA, "margin": MARGIN, "maxalt": MAXALT,
               "n_items": n_items, "wrong": wrong,
               "visibility": vis, "flips": flips, "flip_precision": prec,
               "net_recovery": net, "silent_wrong": sw,
               "verdicts": {"visibility": v1, "precision": v2,
                            "recovery": v3, "silent_wrong": v4},
               "class_hist": {f"{a}|{b}": c for (a, b), c in hist_lab.items()},
               "survivor_hist": {str(k): v for k, v in hist_surv.items()},
               "items": recs}
    with open(OUT, "w") as f:
        json.dump(payload, f, default=lambda x: int(x)
                  if isinstance(x, (np.integer,)) else float(x))
    print(f"[collider] {_FNAME} n={n_items} visibility="
          f"{'n/a' if vis is None else f'{vis:.4f}'} [{v1}] flips={flips} "
          f"precision={'n/a' if prec is None else f'{prec:.4f}'} [{v2}] "
          f"recovery={'n/a' if net is None else f'{net:.4f}'} [{v3}] "
          f"silent-wrong={sw:.4f} [{v4}] -> {OUT}", flush=True)


# ===========================================================================
# SELFTEST — pure python / numpy, no GPU, no ckpt, no fixture
# ===========================================================================

def selftest():
    import py_compile
    py_compile.compile(os.path.abspath(__file__), doraise=True)

    B, L, K = 2, L_FAC, K_VARS
    onp = {"pres": np.full((B, L), -4.0),
           "ftype": np.zeros((B, L, 9)),
           "op": np.zeros((B, L, 2)),
           "args": np.full((B, L, K), -4.0),
           "res": np.zeros((B, L, K)),
           "dig": np.zeros((B, L, N_DIG, 10)),
           "dup": np.full((B, L), -1.0)}
    onp["res"][:] = -4.0
    onp["ftype"][:] = -4.0
    # item 0: given var0 = 5, given var1 = 7, rel add(0,1) -> 0  (impossible)
    def _given(bi, j, var, val):
        onp["pres"][bi, j] = 6.0
        onp["ftype"][bi, j, FT_GIVEN] = 6.0
        onp["res"][bi, j, var] = 6.0
        for d, ch in enumerate(str(val).rjust(N_DIG, "0")):
            onp["dig"][bi, j, d, int(ch)] = 6.0

    def _rel(bi, j, a, b, r, op=0):
        onp["pres"][bi, j] = 6.0
        onp["ftype"][bi, j, FT_REL] = 6.0
        onp["op"][bi, j, op] = 6.0
        onp["args"][bi, j, a] = 6.0
        onp["args"][bi, j, b] = 5.0
        onp["res"][bi, j, r] = 6.0

    _given(0, 0, 0, 5)
    _given(0, 1, 1, 7)
    _rel(0, 2, 0, 1, 0)              # 5 + 7 = 5 -> CONTRADICTORY
    onp["args"][0, 2, 5] = 4.9       # var 5 is the third-ranked arg
    onp["res"][0, 2, 2] = 2.0        # var 2 is the second-best result
    _given(1, 0, 0, 5)
    _given(1, 1, 1, 7)
    _rel(1, 2, 0, 1, 2)              # 5 + 7 = 12 -> UNIQUE for query 2

    D = decode_batch(onp, 0.9)
    assert D["keep"][0].sum() == 3 and D["keep"][1].sum() == 3
    f0 = [fac_at(D, 0, j) for j in np.nonzero(D["keep"][0])[0]]
    assert f0[0] == {"ftype": "given", "var": 0, "value": 5}, f0[0]
    assert f0[2] == {"ftype": "rel", "op": "add", "args": [0, 1],
                     "result": 0}, f0[2]
    assert nv_for(f0, 24) == 24 and nv_for(f0, 2) == 2

    r0 = collide_item(D, 0, 24, 300, 2, 0.9, 0.3, 12)
    assert r0["orig_label"] == "CONTRADICTORY", r0["orig_label"]
    labs = [(a["kind"], a["label"]) for a in r0["alts"]]
    assert ("res2", "UNIQUE") in labs, labs        # 5+7=12 fixes it
    assert r0["flip"] is not None and r0["flip"]["kind"] == "res2", r0["flip"]
    assert r0["survivors"] == sum(1 for a in r0["alts"]
                                  if a["label"] == "UNIQUE")
    r1 = collide_item(D, 1, 24, 300, 2, 0.9, 0.3, 12)
    assert r1["orig_label"] == "UNIQUE" and r1["flip"] is None
    assert r1["survivors"] >= 1

    # cap + ordering
    r2 = collide_item(D, 0, 24, 300, 2, 0.9, 0.3, 2)
    assert r2["n_alts"] == 2
    gaps = [a["gap"] for a in r2["alts"]]
    assert gaps == sorted(gaps)

    # THE FLIP RULE needs EXACTLY ONE unique alternative: make var 2 the
    # third-ranked arg too, so add(0,2)->0 also forces the query (var2=0)
    # -> two survivors -> SILENCE, no flip (the fingerpost rule).
    onp["args"][0, 2, 2] = 4.95
    D2 = decode_batch(onp, 0.9)
    r3 = collide_item(D2, 0, 24, 300, 2, 0.9, 0.3, 12)
    assert r3["orig_label"] == "CONTRADICTORY"
    assert r3["survivors"] == 2 and r3["flip"] is None, r3["alts"]
    onp["args"][0, 2, 2] = -4.0                     # restore for grading

    # ---- grading mirror == score_batch, on the selftest's own inputs ----
    from step_engine_read import score_batch
    vg = {"presence": np.zeros((B, L)), "ftype": np.zeros((B, L), int),
          "res": np.zeros((B, L), int), "op": np.zeros((B, L), int),
          "args": np.zeros((B, L, K)),
          "digits": np.zeros((B, L, N_DIG), int)}
    vg["presence"][0, [0, 1, 2]] = 1
    vg["ftype"][0, 0] = 1; vg["res"][0, 0] = 0
    vg["digits"][0, 0] = [int(c) for c in str(5).rjust(N_DIG, "0")]
    vg["ftype"][0, 1] = 1; vg["res"][0, 1] = 1
    vg["digits"][0, 1] = [int(c) for c in str(7).rjust(N_DIG, "0")]
    vg["ftype"][0, 2] = 0; vg["op"][0, 2] = 0
    vg["args"][0, 2, [0, 1]] = 1; vg["res"][0, 2] = 2       # gold: -> var 2
    vg["presence"][1, [0, 1, 2]] = 1
    vg["ftype"][1, 0] = 1; vg["res"][1, 0] = 0
    vg["digits"][1, 0] = [int(c) for c in str(5).rjust(N_DIG, "0")]
    vg["ftype"][1, 1] = 1; vg["res"][1, 1] = 1
    vg["digits"][1, 1] = [int(c) for c in str(7).rjust(N_DIG, "0")]
    vg["ftype"][1, 2] = 0; vg["args"][1, 2, [0, 1]] = 1; vg["res"][1, 2] = 2
    tot_ok = 0
    for bi in range(B):
        per, item_ok = grade_item(onp, bi, vg, bi, True)
        tot_ok += sum(1 for v in per.values() if v)
        assert item_ok == (bi == 1), (bi, per)
    ok, tot = score_batch(onp, vg, np.array([0, 1]))
    assert (tot_ok, len(vg["presence"].nonzero()[0])) == (ok, tot), \
        (tot_ok, ok, tot)

    # the flip repairs slot 2 of item 0 under the SAME criterion
    base = slot_rec(onp, 0, 2, True)
    assert not slot_ok(base, vg, 0, 2, True)
    fixed = apply_ov(base, r0["flip"]["ov"])
    assert slot_ok(fixed, vg, 0, 2, True)
    # an args override that breaks a correct slot = silent-wrong shape
    b1 = slot_rec(onp, 1, 2, True)
    assert slot_ok(b1, vg, 1, 2, True)
    assert not slot_ok(apply_ov(b1, {"args": (0, 3)}), vg, 1, 2, True)
    # grade_flip bookkeeping: repair, silent-wrong, and no-gold-slot
    per0, _ = grade_item(onp, 0, vg, 0, True)
    per1, _ = grade_item(onp, 1, vg, 1, True)
    assert grade_flip(per0, base, r0["flip"]["ov"], vg, 0, 2, True) == \
        (False, True)                                # a correct flip
    assert grade_flip(per1, b1, {"args": (0, 3)}, vg, 1, 2, True) == \
        (True, False)                                # silent-wrong
    assert grade_flip(per0, base, {"res": 3}, vg, 0, 7, True) == (None, False)

    print("[collider] selftest PASS: adapter decode reproduction, bridge-"
          "grammar factors, nv convention, collision classes (contradictory"
          "/unique), flip rule + cap + gap ordering, grading mirror == "
          "score_batch, grade_flip repair/silent-wrong/no-gold shapes; "
          "zero GPU")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest()
    else:
        main()
