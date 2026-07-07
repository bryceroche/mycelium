"""survivor_suspicion_rank.py — WHERE do the wrong slots sit in the confidence
ranking? (spec registration, 2026-07-08 — the mis-pointed-suspicion test).

THE STORY UNDER TEST: CUT 4 killed omission-blindness (AUC m_add 0.525) but the
(m_add=0, m_corr=1) cell survives at 0.914 — single in-jurisdiction errors the
stack cannot fix. Candidate mechanism: SUSPECT-RANKING BLINDNESS. Withhold-2
removes the two LEAST-confident factors; a confidently-wrong factor escapes,
two correct factors get withheld instead, the flags hand the specialist the
WRONG suspects, and unflagged->copy guarantees the true error propagates
forever. Also explains the front-loaded decay (same wrong flags every round).

REGISTERED PREDICTIONS (before measuring):
  P1. Recovered failures have wrong slots CONCENTRATED at the bottom of the
      confidence ranking; survivors' wrong slots sit HIGH. AUC(min wrong-slot
      rank, normalized -> survival) >= 0.65.
  P2. Withhold-2 COVERAGE (all wrong emitted slots within the bottom-2) is
      several-fold higher among stage-1-recovered than survivors with n_wrong<=2.
  P3. The m<=2 surviving population is dominated by escapes: wrong slot ranked
      outside bottom-2, or query_wrong (no slot at all — structurally
      unflaggable).
  FLAT ranks across populations = the suspicion story dies too and
  decode-degeneracy survives as the standing account.

Zero 4-round replay: survivor/recovered identity comes from
.cache/survivor_profile_bigtest.npz; only the blank parse is re-run.

USAGE: DEV=AMD ALG_TEST=.cache/algebra_nl_bigtest.jsonl ALG_TEST_NAME=bigtest \
           .venv/bin/python3 scripts/survivor_suspicion_rank.py
"""
from __future__ import annotations

import os
import sys
from collections import Counter

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "scripts"))

import numpy as np

from phase1_algebra_head import (  # noqa: E402
    L_FAC, build_params, forward, load_alg, decode, ALG_CKPT,
)
from survivor_multiplicity import midrank_auc  # noqa: E402


def wrong_slots(facs, q_pred, smp):
    """Emitted-slot correctness vs gold. Returns (wrong emitted indices [in
    emission order fi], per-wrong-slot field kinds, query_wrong flag)."""
    grels, ggivs = [], Counter()
    for f in smp["factors"]:
        if f["ftype"] == "rel":
            grels.append((f["op"], tuple(sorted(f["args"])), f["result"]))
        else:
            ggivs[(f["var"], f["value"])] += 1
    grc = Counter(grels)
    gvars = Counter(f["var"] for f in smp["factors"] if f["ftype"] == "given")
    wrong, wrong_keys = [], []
    for fi, f in enumerate(facs):
        if f["ftype"] == "rel":
            key = (f["op"], tuple(sorted(f["args"])), f["result"])
            if grc[key] > 0:
                grc[key] -= 1
            else:
                wrong.append(fi)
                wrong_keys.append(("rel", key))
        else:
            key = (f["var"], f["value"])
            if ggivs[key] > 0:
                ggivs[key] -= 1
            else:
                wrong.append(fi)
                wrong_keys.append(("given", key))
    # field attribution: pair each wrong slot to a leftover gold factor
    leftover = list((grc - Counter()).elements())
    used = [False] * len(leftover)
    kinds = []
    for kt, key in wrong_keys:
        if kt == "given":
            kinds.append("given_value" if gvars[key[0]] > 0 else "phantom")
            continue
        best, best_n = -1, 1
        for gi, g in enumerate(leftover):
            if used[gi]:
                continue
            nm = (g[0] == key[0]) + (g[1] == key[1]) + (g[2] == key[2])
            if nm > best_n:
                best, best_n = gi, nm
        if best < 0:
            kinds.append("phantom")
            continue
        used[best] = True
        g = leftover[best]
        kinds.append("rel_op" if g[0] != key[0]
                     else ("rel_args" if g[1] != key[1] else "rel_result"))
    return wrong, kinds, q_pred != smp["query_var"]


def main():
    from tinygrad import Tensor, dtypes
    from tinygrad.nn.state import safe_load
    from tier0_incumbent import softmax, sig

    prof = np.load(".cache/survivor_profile_bigtest.npz")
    status = {int(i): int(s) for i, s in zip(prof["idx"], prof["status"])}

    samples, states, tokmask, gold, sent = load_alg("test")
    p = build_params(0)
    sd = safe_load(ALG_CKPT)
    for k in p:
        p[k].assign(sd[k].to(p[k].device).cast(p[k].dtype)).realize()

    idx_all = sorted(status)
    outs = {}
    for s0 in range(0, len(idx_all), 8):
        sl = np.array(idx_all[s0:s0 + 8])
        pad = 8 - len(sl)
        sl_p = np.concatenate([sl, sl[:1].repeat(pad)]) if pad else sl
        out = forward(p, Tensor(states[sl_p].astype(np.float32), dtype=dtypes.float),
                      Tensor(tokmask[sl_p].astype(np.float32), dtype=dtypes.float),
                      Tensor(sent[sl_p].astype(np.int32), dtype=dtypes.int))
        o = {k: out[k].realize().numpy() for k in
             ("pres", "ftype", "op", "islit", "dig", "args", "res", "query")}
        for bi, i in enumerate(sl):
            outs[int(i)] = {k: o[k][bi] for k in o}

    def slot_conf(o, j):
        cc = float(sig(o["pres"][j]))
        cc *= float(softmax(o["ftype"][j][None])[0].max())
        cc *= float(softmax(o["res"][j][None])[0].max())
        if sig(o["islit"][j]) > 0.5:
            cc *= float(np.mean(softmax(o["dig"][j]).max(-1)))
        else:
            cc *= float(softmax(o["op"][j][None])[0].max())
            cc *= float(np.sort(softmax(o["args"][j][None])[0])[-2:].sum())
        return cc

    rows = {0: [], 1: [], 2: []}  # status -> per-failure dicts
    for i in idx_all:
        o = outs[i]
        facs, q_pred = decode(o)
        wr, kinds, qw = wrong_slots(facs, q_pred, samples[i])
        confs = []
        fi = 0
        for j in range(L_FAC):
            if o["pres"][j] <= 0:
                continue
            confs.append(slot_conf(o, j))
            fi += 1
        n_slots = len(confs)
        order = np.argsort(confs, kind="mergesort")  # ascending confidence
        rank_of = {int(f): r for r, f in enumerate(order)}
        wranks = [rank_of[f] for f in wr if f in rank_of]
        gv_detail = []
        gold_giv = {f["var"]: f["value"] for f in samples[i]["factors"]
                    if f["ftype"] == "given"}
        for f_i, k in zip(wr, kinds):
            if k != "given_value" or f_i not in rank_of or rank_of[f_i] >= 2:
                continue
            emitted = facs[f_i]["value"]
            same_var_gold = gold_giv.get(facs[f_i]["var"])
            elsewhere = any(v == emitted and vv != facs[f_i]["var"]
                            for vv, v in gold_giv.items())
            e1 = (same_var_gold is not None and
                  len(str(emitted)) == len(str(same_var_gold)) and
                  sum(a != b for a, b in
                      zip(str(emitted), str(same_var_gold))) == 1)
            gv_detail.append((elsewhere, e1))
        rows[status[i]].append({
            "n_wrong": len(wr), "qw": qw, "n_slots": n_slots,
            "gv_detail": gv_detail,
            "flagged_kinds": [k for f, k in zip(wr, kinds)
                              if f in rank_of and rank_of[f] < 2],
            "min_rank": (min(wranks) if wranks else None),
            "min_rank_norm": (min(wranks) / max(n_slots - 1, 1)
                              if wranks else None),
            "frac_bottom2": (np.mean([r < 2 for r in wranks])
                             if wranks else None),
            "covered2": (len(wr) <= 2 and wranks != [] and
                         all(r < 2 for r in wranks)),
        })

    names = {0: "stage1-recovered", 1: "round-recovered", 2: "survivors"}
    print(f"[rank] populations: " +
          " ".join(f"{names[s]}={len(rows[s])}" for s in (0, 1, 2)))
    print(f"\n  population        |  n(w/slot-errs) | min-rank-norm | "
          f"frac-in-bottom2 | withhold2-covers | query_wrong")
    for s in (0, 1, 2):
        rr = [r for r in rows[s] if r["min_rank_norm"] is not None]
        mrn = np.mean([r["min_rank_norm"] for r in rr])
        fb2 = np.mean([r["frac_bottom2"] for r in rr])
        cov = np.mean([r["covered2"] for r in rows[s]])
        qw = np.mean([r["qw"] for r in rows[s]])
        print(f"  {names[s]:17s} |     {len(rr):5d}       |     {mrn:.3f}     |"
              f"      {fb2:.3f}      |      {cov:.3f}       |   {qw:.3f}")

    # P1: AUC of min wrong-slot rank (normalized) -> survival (surv vs round-rec)
    a_s = np.array([r["min_rank_norm"] for r in rows[2]
                    if r["min_rank_norm"] is not None])
    a_r = np.array([r["min_rank_norm"] for r in rows[1]
                    if r["min_rank_norm"] is not None])
    print(f"\n  P1  AUC(min wrong-slot rank -> survival) = "
          f"{midrank_auc(a_s, a_r):.3f}  (bar 0.65)")

    # P3: the low-m surviving population — escapes + unflaggable queries
    lo = [r for r in rows[2] if (r["n_wrong"] + r["qw"]) <= 2]
    esc = np.mean([(r["min_rank"] is not None and r["min_rank"] >= 2) or
                   (r["min_rank"] is None and r["qw"]) for r in lo])
    qonly = np.mean([r["n_wrong"] == 0 and r["qw"] for r in lo])
    print(f"  P3  m<=2 survivors (n={len(lo)}): escape-or-query = {esc:.3f} "
          f"(query-only = {qonly:.3f})")
    print(f"\n  REGISTERED: P1 AUC>=0.65 + coverage gap (P2) -> mis-pointed "
          f"suspicion CONFIRMED (frontier = suspicion quality: transplant / "
          f"better ranker). FLAT -> suspicion story dies; decode-degeneracy "
          f"stands.")

    # ---- CUT 2 (registered 2026-07-08, post-P1-flat): FLAGGED-BUT-UNFIXED
    # field kinds. The suspicion probe found ~73% of m<=2 survivors have their
    # error correctly IN the bottom-2 flags — repair GENERATION is the wall.
    # PREDICTION: flagged survivor errors are ENRICHED for rel_args (binding —
    # the frozen-prefix weakness at its correct jurisdiction) vs flagged
    # recovered errors; bar >=1.5x. Uniform-across-kinds -> head-capacity story;
    # escalate to a trunk-information probe.
    KINDS = ("rel_op", "rel_args", "rel_result", "given_value", "phantom")
    def kind_mix(rr):
        tot = Counter(k for r in rr for k in r["flagged_kinds"])
        s = max(sum(tot.values()), 1)
        return {k: tot[k] / s for k in KINDS}, sum(tot.values())
    mix_s, ns = kind_mix(rows[2])
    mix_r, nr = kind_mix(rows[0] + rows[1])
    print(f"\n  CUT 2: flagged (bottom-2) wrong-slot field kinds")
    print(f"  kind        | survivors (n={ns}) | recovered (n={nr}) | enrich")
    for k in KINDS:
        e = mix_s[k] / max(mix_r[k], 1e-9)
        print(f"  {k:11s} |       {mix_s[k]:.3f}       |       {mix_r[k]:.3f}"
              f"       | {e:.2f}x")
    lo_s = [r for r in rows[2] if (r["n_wrong"] + r["qw"]) <= 2]
    mix_lo, nlo = kind_mix(lo_s)
    print(f"  m<=2 survivors only (n={nlo}): " +
          " ".join(f"{k}:{mix_lo[k]:.2f}" for k in KINDS if mix_lo[k] > 0.01))

    # ---- CUT 3 (registered 2026-07-08, discriminator — no directional
    # prediction): what IS a flagged-but-unfixed given_value error? The pairing
    # rule conflates two stories: VALUE-HALLUCINATION (emitted value in NO gold
    # given — the digit heads cannot reconstruct the literal; a digit-
    # reconstruction wall) vs VALUE-MISBINDING (emitted value belongs to a
    # DIFFERENT gold given — right values shuffled across variables; binding
    # resurrected one level down). edit1 = same length, exactly one digit off
    # from the same-var gold (near-miss reconstruction).
    print(f"\n  CUT 3: flagged given_value anatomy (elsewhere = misbinding;"
          f" not-in-gold = hallucination)")
    for s, nm in ((2, "survivors"), (1, "round-recovered"),
                  (0, "stage1-recovered")):
        det = [d for r in rows[s] for d in r["gv_detail"]]
        if not det:
            print(f"  {nm:17s}: none")
            continue
        elsew = np.mean([d[0] for d in det])
        e1 = np.mean([d[1] for d in det])
        print(f"  {nm:17s}: n={len(det):4d} | value-elsewhere {elsew:.3f} | "
              f"not-in-gold {1 - elsew:.3f} | one-digit-off {e1:.3f}")


if __name__ == "__main__":
    main()
