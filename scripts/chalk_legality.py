"""chalk_legality.py -- THE GYM's FIFTH USE: THE CHALKBOARD'S SECOND FORM (2026-10-09, worktree
mycelium-wt8, branch replay; ledger 2026-10-08 20:2x). The solver's implied value, admitted as a
LEGAL numeral for a relation slot's digit head -- no token, no attention, no bypass; a pure
read-time widening of mycelium.rulebook's numeral mask.

BACKGROUND: mycelium.rulebook.legal_values(text, nd) returns the TEXT's own numerals (+ lexicon
constants + 1) as the only "legal" choices a digit head may commit to (choose_legal picks the
legal value the digit head's own joint log-probability prefers; legal_digit_logits rewrites the
logits so the argmax IS that choice -- the numeral mask CA_MASK=1 reads today). A value the
solver can DERIVE but that never appears as a text numeral can therefore never be decoded,
anywhere this mask applies. THE SECOND FORM: for a slot whose GOLD fact is a RELATION (ftype
"rel"), its own dig[] head (computed for every slot regardless of ftype -- _heads_of's "dig" key
-- but never READ by decode()'s ft==0 branch, which has no digit path of its own) is probed
directly with the SAME choose_legal machinery, with the relation's own SOLVER-IMPLIED value
(read from the consult's facts3/facts5 -- THE CHALKBOARD's own fact source, (B, K_VARS, 4):
col0 known-flag, cols 1-3 MSD digits/9, values capped at 999 per the dialect) ADDED to the legal
set for that one slot. Two uses of the resulting widened choice:
  (per-slot) does the widened choice recover the slot's GOLD value more often than plain?
  (row-level) if the digit head, once offered the option, actually PREFERS the implied value
    (widened choice == implied value), that value is injected as an EXTRA "given" fact for the
    relation's result variable (gv_widened.setdefault(result, value)) alongside the relation
    fact itself -- rulebook.py's own note that a variable may be both a relation's result and a
    given (single_intro was refuted 2026-09-18) -- and the row is re-solved with chain_acc's own
    solver (reused by import). This is the mechanism that could change the ROW-level chain-acc
    verdict; everything else here is read-only.

BODY: RKX_241 (its own env; see pilot_ceiling.py's _FAM, identical here). Snapshot at kb=2
(after consult 1; facts3/cert3/rack3 already folded in), replay UNCHANGED to the end (no
schedule override this time -- condition A only); facts5/cert5/rack5 (consult 2, kb=4) are
threaded automatically during that replay, exactly as every prior gym script's pipeline does.

Diet first (.cache/form_pm35c_slice1024_valid2.jsonl, 128 rows), then the wild holdout in FULL
(.cache/wild_admitted_holdout.jsonl, 311 rows -- a READ-TIME road, no training, one read is
legitimate per the task). DEV=CPU; .cache/gpu.lock never touched (no flock in this file).
"""
import os
import sys
import time
import collections

sys.path.insert(0, "."); sys.path.insert(0, "scripts")
os.environ.setdefault("DEV", "CPU")
assert os.environ.get("DEV") == "CPU"

import numpy as np

CKPT = ".cache/sharp_RKX_241.safetensors"
OUT_TXT = ".cache/chalk_legality_RKX_241.txt"
SNAP_KB = 2
DIET_TEST = ".cache/form_pm35c_slice1024_valid2.jsonl"
DIET_NAME = "pm35cslicevalid2"
WILD_TEST = ".cache/wild_admitted_holdout.jsonl"
WILD_NAME = "wildhold"
N_DIET = 128
N_WILD = 10 ** 9   # "the full 311" -- min(this, len(vs)) below
BATCH = 32

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
    "ALG_ROUTER": "2", "R_GAIN_INIT": "1.0", "ALG_FREEZE": "r_gain", "ALG_ROUTER_PTR": "0.0",
    "ALG_SPAN_ALL": "1", "ALG_SPAN_ARGS": "1", "ALG_SPAN_OP": "1", "ALG_SPAN_RCUE": "1",
    "ALG_SPAN_ARCUE": "1", "ALG_PTR_SURF": "role:add:2.0",
    "ALG_HIER_READ": "1", "ALG_HIER_WAIST": "1", "ALG_HIER_DAMP": "3,5,0", "ALG_HIER_TAU": "0",
    "ALG_ALT3": "1", "ALG_CERT": "2.0", "ALG_RACK": "1",
    "ALG_RACK_TESTS": "given_unique", "ALG_RACK_FREEZE": "leaf",
}


def log(P, s):
    print(s, flush=True); P.append(s)


def load_body(H):
    from tinygrad.nn.state import safe_load
    p = H.build_params(0)
    sd = safe_load(CKPT)
    assert set(sd.keys()) == set(p.keys()), (sorted(set(sd) - set(p))[:6], sorted(set(p) - set(sd))[:6])
    for k in p:
        p[k].assign(sd[k].to(p[k].device).cast(p[k].dtype)).realize()
    return p


def implied_value(facts_np, var, nd=3):
    """facts_np: (K_VARS, 4) float -- col0 known-flag, cols1-3 MSD digits/9 (<=999, the
    dialect's fact-buffer convention). None if not known."""
    if facts_np is None or var is None or not (0 <= var < facts_np.shape[0]):
        return None
    if facts_np[var, 0] <= 0:
        return None
    d0 = int(round(facts_np[var, 1] * 9)); d1 = int(round(facts_np[var, 2] * 9)); d2 = int(round(facts_np[var, 3] * 9))
    return d0 * 100 + d1 * 10 + d2


def build_snapshot(H, p, ts, tk, se, vs, sl):
    from tinygrad import Tensor, dtypes
    o0 = H.forward(p, ts, tk, se)
    onp0 = {k: o0[k].realize().numpy() for k in ("fat", "args", "res")}
    mk = H.build_slot_masks(onp0, se.numpy().astype(np.int32))
    slot_mask = Tensor(mk, dtype=dtypes.float)
    texts = [vs[int(i)]["text"] for i in sl]
    nv = np.array([vs[int(i)].get("n_vars", H.K_VARS) for i in sl])
    ma = np.array([vs[int(i)].get("m", 0) for i in sl])
    ck3 = ("pres", "ftype", "op", "dig", "args", "res", "query") + (("dup",) if "dup" in o0 else ())

    oa3 = H.forward(p, ts, tk, se, slot_mask=slot_mask, stop_after=2)
    onp2 = {k: oa3[k].realize().numpy() for k in ck3}
    fb3 = H.alt2_fact_buf(onp2, se.numpy().astype(np.int32), nv, ma)
    f3_t = Tensor(fb3, dtype=dtypes.float)
    rows3 = [{k: onp2[k][bi] for k in onp2} for bi in range(len(sl))]
    c3_t = Tensor(H.certifier_bias(rows3, fb3, texts, H.T_ALG, H.ALG_CERT, H.ALG_CERT_IMPLIED), dtype=dtypes.float)
    r3_np = H.rack_pack(rows3, fb3, texts, H.T_ALG, H.ALG_RACK_TESTS)
    r3_t = Tensor(r3_np, dtype=dtypes.float)

    ob3 = H.forward(p, ts, tk, se, slot_mask=slot_mask, stop_after=4, facts3=f3_t, cert3=c3_t, rack3=r3_t)
    onp4 = {k: ob3[k].realize().numpy() for k in ck3}
    fb5 = H.alt2_fact_buf(onp4, se.numpy().astype(np.int32), nv, ma)
    f5_t = Tensor(fb5, dtype=dtypes.float)
    rows5 = [{k: onp4[k][bi] for k in onp4} for bi in range(len(sl))]
    c5_t = Tensor(H.certifier_bias(rows5, fb5, texts, H.T_ALG, H.ALG_CERT, H.ALG_CERT_IMPLIED), dtype=dtypes.float)
    r5_np = H.rack_pack(rows5, fb5, texts, H.T_ALG, H.ALG_RACK_TESTS, prev=r3_np)
    r5_t = Tensor(r5_np, dtype=dtypes.float)

    snap_path = ".cache/chalk_legality_snap_tmp.npz"
    os.environ["ALG_SNAP_AT"] = str(SNAP_KB); os.environ["ALG_SNAP_OUT"] = snap_path
    _ = H.forward(p, ts, tk, se, slot_mask=slot_mask, facts3=f3_t, facts5=f5_t,
                 cert3=c3_t, cert5=c5_t, rack3=r3_t, rack5=r5_t)
    os.environ.pop("ALG_SNAP_AT"); os.environ.pop("ALG_SNAP_OUT")
    snap = H.snap_load(snap_path)
    return snap, fb3, fb5


def decode_row_plain(H, out_np, bi, i, vs):
    from mycelium.rulebook import legal_digit_logits
    need = ("pres", "ftype", "op", "dig", "args", "res", "query") + (("dup",) if "dup" in out_np else ())
    row = {k: out_np[k][bi].copy() for k in need}
    for j in range(row["ftype"].shape[0]):
        if int(row["ftype"][j].argmax()) == 0:
            continue
        fake = legal_digit_logits(row["dig"][j], vs[i]["text"])
        if fake is not None:
            row["dig"][j] = fake
    parse = H._decode_slots(row)
    q = int(row["query"].argmax())
    return parse, q, row


def build_task(vs, i, q, parse, gv):
    from mycelium.custody_gold import row_gold
    try:
        key = int(row_gold(vs[i]))
    except Exception:
        key = vs[i].get("key"); key = int(key) if key is not None else None
    if key is None or not parse:
        return None
    used = [f.get("var") for f in parse if f["ftype"] == "given"] + \
           [a for f in parse if f["ftype"] == "rel" for a in list(f["args"]) + [f["result"]]]
    nvv = max([q + 1] + [v + 1 for v in used if v is not None])
    return q, key, parse, gv, nvv


def main():
    P = []
    t0 = time.time()
    for k, v in _FAM.items():
        os.environ[k] = v
    import phase1_algebra_head as H
    from tinygrad import Tensor, dtypes
    from mycelium.rulebook import legal_values, choose_legal
    p = load_body(H)
    log(P, f"[chalk-legal] body loaded ({time.time() - t0:.0f}s so far)")

    def run_split(split_name, test_path, test_name, n_rows):
        os.environ["ALG_TEST"] = test_path; os.environ["ALG_TEST_NAME"] = test_name
        vs, vst_, vtk, vg, vse = H.load_alg("test")
        n = min(n_rows, len(vs))
        log(P, f"[chalk-legal] {split_name}: {n}/{len(vs)} rows")

        task_rows = {}; tasks = []
        slot_rows = []   # (row_i, j, gold_val, gold_in_text, implied, plain_choice, widened_choice)
        uid_ctr = [0]
        def new_uid():
            uid_ctr[0] += 1; return uid_ctr[0]

        for s0 in range(0, n, BATCH):
            sl = np.arange(s0, min(s0 + BATCH, n))
            ts = Tensor(np.ascontiguousarray(vst_[sl]), dtype=dtypes.half)
            tk = Tensor(vtk[sl].astype(np.float32), dtype=dtypes.float)
            se = Tensor(vse[sl].astype(np.int32), dtype=dtypes.int)
            snap, fb3, fb5 = build_snapshot(H, p, ts, tk, se, vs, sl)
            out = H.replay(p, snap, SNAP_KB)
            out_np = {k: out[k].realize().numpy() for k in out if hasattr(out[k], "realize")}

            for bi, i in enumerate(sl):
                i = int(i)
                parse, q, row = decode_row_plain(H, out_np, bi, i, vs)
                gv_plain = {f["var"]: f["value"] for f in parse if f["ftype"] == "given"}
                text = vs[i]["text"]
                text_values = legal_values(text, H.N_DIG)
                rel_facts = [(j, f) for j, f in enumerate(vs[i]["factors"]) if f["ftype"] == "rel"]
                gv_widened = dict(gv_plain)
                for j, f in rel_facts:
                    if j >= H.L_FAC:
                        continue
                    r = f["result"]
                    gold_val = None
                    try:
                        gold_val = int("".join(str(int(x)) for x in vg["digits"][i, j]))
                    except Exception:
                        pass
                    imp5 = implied_value(fb5[bi], r); imp3 = implied_value(fb3[bi], r)
                    imp = imp5 if imp5 is not None else imp3
                    dig_logits = out_np["dig"][bi, j]
                    plain_choice = choose_legal(dig_logits, text_values, H.N_DIG)
                    widened_values = text_values if imp is None else sorted(set(text_values) | {imp})
                    widened_choice = choose_legal(dig_logits, widened_values, H.N_DIG)
                    gold_in_text = (gold_val is not None) and (gold_val in text_values)
                    slot_rows.append((i, j, gold_val, gold_in_text, imp, plain_choice, widened_choice))
                    if imp is not None and widened_choice == imp and r not in gv_widened:
                        gv_widened[r] = imp

                t_plain = build_task(vs, i, q, parse, gv_plain)
                t_wide = build_task(vs, i, q, parse, gv_widened)
                if t_plain is not None:
                    uid = new_uid(); task_rows[uid] = ("plain", i); tasks.append((uid,) + t_plain)
                else:
                    task_rows[new_uid()] = ("plain", i)
                if t_wide is not None:
                    uid = new_uid(); task_rows[uid] = ("widened", i); tasks.append((uid,) + t_wide)
                else:
                    task_rows[new_uid()] = ("widened", i)
            log(P, f"[chalk-legal] {split_name} batch {s0}-{s0+len(sl)} done ({time.time() - t0:.0f}s so far)")

        import multiprocessing as mp
        from chain_acc import _solve_task
        os.environ.setdefault("CA_WALL", "2")
        verdict = {}
        _task_uids = {t[0] for t in tasks}
        for uid in task_rows:
            if uid not in _task_uids:
                verdict[uid] = "refused"
        real_tasks = list(tasks)
        workers = int(os.environ.get("CA_WORKERS", "6"))
        log(P, f"[chalk-legal] {split_name}: solving {len(real_tasks)} tasks with {workers} workers")
        if real_tasks:
            with mp.get_context("spawn").Pool(workers) as pool:
                it = pool.imap_unordered(_solve_task, real_tasks, chunksize=1)
                wall = float(os.environ.get("CA_WALL", "2"))
                got = {}
                try:
                    for _ in range(len(real_tasks)):
                        uid, st, val = it.next(timeout=wall * 3 + 30)
                        got[uid] = (st, val)
                except mp.TimeoutError:
                    log(P, f"[chalk-legal] {split_name}: {len(real_tasks) - len(got)} tasks never returned -- counted as refused")
            for (uid, q, key, parse, gv, nvv) in real_tasks:
                st, val = got.get(uid, ("hung", None))
                if st != "solved":
                    verdict[uid] = "refused"
                elif val == key:
                    verdict[uid] = "correct"
                else:
                    verdict[uid] = "wrong"
        log(P, f"[chalk-legal] {split_name}: solved ({time.time() - t0:.0f}s so far)")

        row_verdict = collections.defaultdict(dict)
        for uid, (tag, i) in task_rows.items():
            row_verdict[i][tag] = verdict.get(uid, "refused")

        return dict(n=n, row_verdict=row_verdict, slot_rows=slot_rows)

    diet = run_split("DIET", DIET_TEST, DIET_NAME, N_DIET)
    wild = run_split("WILD", WILD_TEST, WILD_NAME, N_WILD)

    def render(split_name, res, L):
        n = res["n"]; rv = res["row_verdict"]; sr = res["slot_rows"]
        plain_cnt = collections.Counter(rv[i].get("plain", "refused") for i in rv)
        wide_cnt = collections.Counter(rv[i].get("widened", "refused") for i in rv)
        L(f"n rows = {n}")
        L(f"  rows correct, PLAIN mask:    correct={plain_cnt['correct']:4d}  refused={plain_cnt['refused']:4d}  wrong={plain_cnt['wrong']:4d}")
        L(f"  rows correct, WIDENED mask:  correct={wide_cnt['correct']:4d}  refused={wide_cnt['refused']:4d}  wrong={wide_cnt['wrong']:4d}")
        changed = [i for i in rv if rv[i].get("plain", "refused") != rv[i].get("widened", "refused")]
        L(f"  rows whose verdict changed under widening: {len(changed)}/{n} {changed[:12]}{'...' if len(changed)>12 else ''}")
        L("")
        n_rel = len(sr)
        n_implied = sum(1 for (_, _, _, _, imp, _, _) in sr if imp is not None)
        n_implied_gold_known = [(g, imp) for (_, _, g, _, imp, _, _) in sr if imp is not None and g is not None]
        n_precise = sum(1 for g, imp in n_implied_gold_known if g == imp)
        n_chose_implied = sum(1 for (_, _, _, _, imp, _, wc) in sr if imp is not None and wc == imp)
        L(f"COVERAGE: {n_implied}/{n_rel} gold relation slots have a solver-implied value ({n_implied/max(n_rel,1):.1%})")
        L(f"PRECISION: of those with both an implied value and a known gold value, {n_precise}/{len(n_implied_gold_known)} "
          f"implied values equal gold ({n_precise/max(len(n_implied_gold_known),1):.1%})")
        L(f"CHOSEN: of those with an implied value, the widened digit head PREFERRED it in {n_chose_implied}/{n_implied} "
          f"cases ({n_chose_implied/max(n_implied,1):.1%}) -- the gap (never prefers) = {1 - n_chose_implied/max(n_implied,1):.1%}")
        L("")
        def _acc(rows):
            p_ = sum(1 for (_, _, g, _, _, pc, _) in rows if g is not None and pc == g)
            w_ = sum(1 for (_, _, g, _, _, _, wc) in rows if g is not None and wc == g)
            n_ = sum(1 for (_, _, g, _, _, _, _) in rows if g is not None)
            return p_, w_, n_
        p_all, w_all, n_all = _acc(sr)
        L(f"PER-SLOT DIGIT ACCURACY (all gold relation slots, n={n_all}): plain={p_all}/{n_all} ({p_all/max(n_all,1):.1%})  "
          f"widened={w_all}/{n_all} ({w_all/max(n_all,1):.1%})")
        visible = [r for r in sr if r[3]]
        wall = [r for r in sr if not r[3]]
        pv, wv, nv = _acc(visible); pw, ww, nw = _acc(wall)
        L(f"  gold value IN TEXT   (n={nv}): plain={pv}/{nv} ({pv/max(nv,1):.1%})  widened={wv}/{nv} ({wv/max(nv,1):.1%})")
        L(f"  gold value NOT in text / THE WALL (n={nw}): plain={pw}/{nw} ({pw/max(nw,1):.1%})  widened={ww}/{nw} ({ww/max(nw,1):.1%})")
        return dict(plain_cnt=plain_cnt, wide_cnt=wide_cnt, changed=changed, n_rel=n_rel,
                    n_implied=n_implied, n_precise=n_precise, n_implied_gold_known=len(n_implied_gold_known),
                    n_chose_implied=n_chose_implied, p_all=p_all, w_all=w_all, n_all=n_all,
                    pv=pv, wv=wv, nv=nv, pw=pw, ww=ww, nw=nw)

    lines = []
    L = lines.append
    L("=" * 92)
    L("THE CHALKBOARD'S SECOND FORM -- RKX_241, the numeral mask widened by the solver's implied value (2026-10-09)")
    L("=" * 92)
    L("")
    L("-" * 92); L("DIET (.cache/form_pm35c_slice1024_valid2.jsonl) -- read first"); L("-" * 92)
    d = render("DIET", diet, L)
    L(""); L("-" * 92); L("WILD (.cache/wild_admitted_holdout.jsonl, FULL 311) -- the measurement, read ONCE"); L("-" * 92)
    w = render("WILD", wild, L)

    L(""); L("-" * 92); L("THE SIX-LINE READING"); L("-" * 92)
    L(f"1. ROW-LEVEL EFFECT: diet correct plain={d['plain_cnt']['correct']} -> widened={d['wide_cnt']['correct']} "
      f"({d['wide_cnt']['correct']-d['plain_cnt']['correct']:+d}); wild correct plain={w['plain_cnt']['correct']} -> "
      f"widened={w['wide_cnt']['correct']} ({w['wide_cnt']['correct']-w['plain_cnt']['correct']:+d}); "
      f"{len(d['changed'])}/{diet['n']} diet rows and {len(w['changed'])}/{wild['n']} wild rows change verdict at all.")
    L(f"2. COVERAGE: diet {d['n_implied']}/{d['n_rel']} ({d['n_implied']/max(d['n_rel'],1):.1%}) gold relation slots carry "
      f"a solver-implied value; wild {w['n_implied']}/{w['n_rel']} ({w['n_implied']/max(w['n_rel'],1):.1%}).")
    L(f"3. THE CERTIFICATE'S PRECISION (implied == gold, where both known): diet {d['n_precise']}/{d['n_implied_gold_known']}; "
      f"wild {w['n_precise']}/{w['n_implied_gold_known']}.")
    L(f"4. THE PREFERENCE GAP (widened choice == implied value, among slots with one): diet {d['n_chose_implied']}/{d['n_implied']} "
      f"({d['n_chose_implied']/max(d['n_implied'],1):.1%}); wild {w['n_chose_implied']}/{w['n_implied']} "
      f"({w['n_chose_implied']/max(w['n_implied'],1):.1%}) -- gap = 1 minus this, per the pinned instruction.")
    L(f"5. PER-SLOT DIGIT ACCURACY, plain vs widened: diet {d['p_all']}/{d['n_all']} -> {d['w_all']}/{d['n_all']}; "
      f"wild {w['p_all']}/{w['n_all']} -> {w['w_all']}/{w['n_all']}. THE WALL (gold not in text) alone: diet "
      f"{d['pw']}/{d['nw']} -> {d['ww']}/{d['nw']}; wild {w['pw']}/{w['nw']} -> {w['ww']}/{w['nw']}.")
    L(f"6. VERDICT: see lines 1-5 for the actual numbers (not re-asserted here) -- the road's ceiling is set by "
      f"line 2 (coverage) x line 3 (precision) x line 4 (preference); a small row-level move (line 1) with high "
      f"coverage/precision but a low preference rate points at the digit head, not the certificate, as the bottleneck.")

    txt = "\n".join(lines) + "\n"
    open(OUT_TXT, "w").write(txt)
    log(P, f"[chalk-legal] wrote {OUT_TXT} ({time.time() - t0:.0f}s total)")


if __name__ == "__main__":
    main()
