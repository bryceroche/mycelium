"""pilot_ceiling.py -- THE GYM's FOURTH USE: THE PILOT'S CEILING (2026-10-08, worktree
mycelium-wt8, branch replay; Bryce: "the T6 clock is the conductor, the perceiver the pilot --
how do we integrate them deeply?").

QUESTION: how much is there to gain if a per-row "pilot" could choose THE HIERARCHICAL STATE's
damping schedule per row, instead of one fixed conductor() schedule for every row?

BODY: RKX_241 (.cache/sharp_RKX_241.safetensors; SURF8 + ALG_HIER_READ/WAIST + ALG_HIER_DAMP=
3,5,0 ALG_HIER_TAU=0 + ALG_ALT3=1 ALG_CERT=2.0 + ALG_RACK=1 given_unique leaf -- the exact recipe
in .cache/rack_chain_RKX.sh). Snapshot at kb=2, AFTER consult 1 fires (the chalk dry-run's own
point), on EVERY row (one batch forward() per row-batch, not per condition): THE THREE CONSULTS'
real products (facts3/facts5/cert3/cert5/rack3/rack5) are derived ONCE from the UNMODIFIED
schedule and threaded through every override via the snapshot, exactly as THE SAVE POINT's own
stated scope limit says (replay() does not re-derive a consult live) -- A SCHEDULE OVERRIDE AT
BREATHS 3-4 THEREFORE NEVER CHANGES WHAT CONSULT 2 (kb=4) ITSELF CONCLUDES, only how the state
carries facts3's evidence into it and beyond. Stated, not hidden, exactly as the chalk dry-run's
own note.

THE FIVE OVERRIDES (conductor(kb).hier_damp_shares / .hier_listen, intercepted by directly
reassigning the module globals _HIER_DAMP / ALG_HIER_LISTEN before each replay() and clearing
_CONDUCTOR_CACHE -- the module-level memo keyed only by kb, confirmed by reading conductor()'s
source at scripts/phase1_algebra_head.py:1210-1316; every OTHER field it returns is a pure
function of UNCHANGED globals, so clearing it has no side effect beyond hier_damp_shares/
hier_listen):
  A fixed     ALG_HIER_DAMP=3,5,0  listen=0   (RKX_241's own trained schedule)
  B delay     ALG_HIER_DAMP=4,6,0  listen=0   (settle one breath later)
  C listen    ALG_HIER_DAMP=3,5,0  listen=1   (THE LISTENING BREATH semantics layered on A)
  D none      ALG_HIER_DAMP=0,0,0  listen=0   (every band's settle=0 is falsy -> conductor()
                                               always returns None per band -> no damping ever,
                                               the plain dynamics -- verified against the source,
                                               not assumed)
  E early     ALG_HIER_DAMP=2,4,0  listen=0   (HS_241/RK_241's PRE-fix schedule)

SCORING: chain_acc.py's own machinery, reused by import (_solve_task, the numeral mask via
mycelium.rulebook.legal_digit_logits, the June solver via admit_annotation.solve_walled) --
CORRECT / REFUSED / WRONG per (row, condition), one shared multiprocessing pool over every task
rather than one pool per condition. Per-slot right count: the decoded parse (_decode_slots)
matched against the row's own gold factors -- ftype/var/value for a given, ftype/op/args/result
for a relation (set membership, not an external ps_legal file: no ps_legal exists for this diet
slice under this body).

THE PILOT'S INSTRUMENTS: at kb=2, the dry count (rack_pack's own per-row dry-slot sum, consult
1) and a kb=2 SOLVER STATUS (the SAME chain_acc solve, run on consult 1's own partial decode,
masked) -- correlated post hoc against which override wins each row.

Diet first (.cache/form_pm35c_slice1024_valid2.jsonl, >= 128 rows, CPU-budget capped), wild once
as the measurement (.cache/wild_admitted_holdout.jsonl, 64 rows). DEV=CPU; .cache/gpu.lock never
touched (no flock anywhere in this file).
"""
import os
import sys
import time
import json
import collections

sys.path.insert(0, "."); sys.path.insert(0, "scripts")
os.environ.setdefault("DEV", "CPU")
assert os.environ.get("DEV") == "CPU"

import numpy as np

CKPT = ".cache/sharp_RKX_241.safetensors"
OUT_TXT = ".cache/pilot_ceiling_RKX_241.txt"
SNAP_KB = 2
DIET_TEST = ".cache/form_pm35c_slice1024_valid2.jsonl"
DIET_NAME = "pm35cslicevalid2"
WILD_TEST = ".cache/wild_admitted_holdout.jsonl"
WILD_NAME = "wildhold"
N_DIET = 128
N_WILD = 64
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

OVERRIDES = collections.OrderedDict([
    ("A_fixed",  dict(damp=[3, 5, 0], listen=0)),
    ("B_delay",  dict(damp=[4, 6, 0], listen=0)),
    ("C_listen", dict(damp=[3, 5, 0], listen=1)),
    ("D_none",   dict(damp=[0, 0, 0], listen=0)),
    ("E_early",  dict(damp=[2, 4, 0], listen=0)),
])
RANK = {"correct": 2, "refused": 1, "wrong": 0}   # oracle's preference order


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


def build_snapshot(H, p, ts, tk, se, vs, sl):
    """Pass-1 + the three-consult cycle (membrane_rack.py's own structure), then the SAME
    full final-pass forward() call with ALG_SNAP_AT=2 armed. Returns (snap, onp2, dry_count,
    out_full) -- onp2 is consult 1's own decode (oa3's heads, the kb=2 instrument source),
    dry_count is rack_pack's per-row dry-slot sum at consult 1, out_full is the UNMODIFIED
    (condition-A-equivalent) full forward output for the sanity check against replay(A)."""
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
    dry_count = r3_np[:, :H.L_TOT].sum(1).copy()

    ob3 = H.forward(p, ts, tk, se, slot_mask=slot_mask, stop_after=4, facts3=f3_t, cert3=c3_t, rack3=r3_t)
    onp4 = {k: ob3[k].realize().numpy() for k in ck3}
    fb5 = H.alt2_fact_buf(onp4, se.numpy().astype(np.int32), nv, ma)
    f5_t = Tensor(fb5, dtype=dtypes.float)
    rows5 = [{k: onp4[k][bi] for k in onp4} for bi in range(len(sl))]
    c5_t = Tensor(H.certifier_bias(rows5, fb5, texts, H.T_ALG, H.ALG_CERT, H.ALG_CERT_IMPLIED), dtype=dtypes.float)
    r5_np = H.rack_pack(rows5, fb5, texts, H.T_ALG, H.ALG_RACK_TESTS, prev=r3_np)
    r5_t = Tensor(r5_np, dtype=dtypes.float)

    snap_path = ".cache/pilot_snap_tmp.npz"
    out_full = H.forward(p, ts, tk, se, slot_mask=slot_mask, facts3=f3_t, facts5=f5_t,
                         cert3=c3_t, cert5=c5_t, rack3=r3_t, rack5=r5_t)
    for k in list(out_full):
        if hasattr(out_full[k], "realize"):
            out_full[k] = out_full[k].realize()
    os.environ["ALG_SNAP_AT"] = str(SNAP_KB); os.environ["ALG_SNAP_OUT"] = snap_path
    _ = H.forward(p, ts, tk, se, slot_mask=slot_mask, facts3=f3_t, facts5=f5_t,
                 cert3=c3_t, cert5=c5_t, rack3=r3_t, rack5=r5_t)
    os.environ.pop("ALG_SNAP_AT"); os.environ.pop("ALG_SNAP_OUT")
    snap = H.snap_load(snap_path)
    return snap, onp2, dry_count, out_full


def decode_row(H, out_np, bi, i, vs, mask=True):
    from mycelium.rulebook import legal_digit_logits
    need = ("pres", "ftype", "op", "dig", "args", "res") + (("dup",) if "dup" in out_np else ())
    row = {k: out_np[k][bi].copy() for k in need}
    if mask:
        for j in range(row["ftype"].shape[0]):
            if int(row["ftype"][j].argmax()) == 0:
                continue
            fake = legal_digit_logits(row["dig"][j], vs[i]["text"])
            if fake is not None:
                row["dig"][j] = fake
    parse = H._decode_slots(row)
    q = int(out_np["query"][bi].argmax())
    return parse, q


def per_slot_right(parse, factors):
    """A simple set-membership score (no external ps_legal for this slice/body): a gold slot
    j is right if the decoded parse has a fact at _slot==j matching gold's ftype and payload."""
    by_slot = {f["_slot"]: f for f in parse}
    n_right = 0
    for j, g in enumerate(factors):
        f = by_slot.get(j)
        if f is None:
            continue
        if g["ftype"] == "given" and f.get("ftype") == "given" and f.get("var") == g["var"] and f.get("value") == g["value"]:
            n_right += 1
        elif g["ftype"] == "rel" and f.get("ftype") == "rel" and f.get("op") == g["op"] \
                and f.get("result") == g["result"] and sorted(f.get("args", [])) == sorted(g["args"]):
            n_right += 1
    return n_right, len(factors)


def build_task(vs, i, parse, q):
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
    gv = {f["var"]: f["value"] for f in parse if f["ftype"] == "given"}
    return q, key, parse, gv, nvv


def main():
    P = []
    t0 = time.time()
    for k, v in _FAM.items():
        os.environ[k] = v
    import phase1_algebra_head as H
    from tinygrad import Tensor, dtypes
    p = load_body(H)
    log(P, f"[pilot] body loaded ({time.time() - t0:.0f}s so far)")

    def run_split(split_name, test_path, test_name, n_rows):
        os.environ["ALG_TEST"] = test_path; os.environ["ALG_TEST_NAME"] = test_name
        H._HIER_DAMP = OVERRIDES["A_fixed"]["damp"]; H.ALG_HIER_LISTEN = 0
        H._CONDUCTOR_CACHE.clear()
        vs, vst_, vtk, vg, vse = H.load_alg("test")
        n = min(n_rows, len(vs))
        log(P, f"[pilot] {split_name}: {n}/{len(vs)} rows")

        task_rows = {}    # uid -> (cond, i, is_sane_full)
        tasks = []
        per_slot = {}     # (cond, i) -> (n_right, n_tot)
        kb2_task_uid = {}  # i -> uid for the kb2-status instrument
        dry_count_of = {}
        sane_mismatch = []

        uid_ctr = [0]
        def new_uid():
            uid_ctr[0] += 1; return uid_ctr[0]

        for s0 in range(0, n, BATCH):
            sl = np.arange(s0, min(s0 + BATCH, n))
            ts = Tensor(np.ascontiguousarray(vst_[sl]), dtype=dtypes.half)
            tk = Tensor(vtk[sl].astype(np.float32), dtype=dtypes.float)
            se = Tensor(vse[sl].astype(np.int32), dtype=dtypes.int)
            # BUG FOUND AND FIXED (first run, 2026-10-08): the OVERRIDES loop below leaves
            # H._HIER_DAMP/ALG_HIER_LISTEN at condition E's values after the previous batch
            # (or the previous split, for wild's first batch) -- build_snapshot's own
            # out_full (the un-replayed full forward(), used only by the sanity check) must
            # run under A's schedule explicitly, every batch, not rely on a leftover global.
            # The scored conditions below were NEVER affected (each sets its own schedule
            # right before its own replay() call) -- confirmed by inspection: breaths 1-2
            # (inside build_snapshot) never trigger ANY band's damping under A/B/C/D/E
            # (every schedule's settle breath is >= 2, and conductor()'s own test is kb >
            # settle, strictly, so kb<=2 is never affected regardless of which schedule is
            # stale) -- this reset only cleans up the sanity check's own baseline.
            H._HIER_DAMP = OVERRIDES["A_fixed"]["damp"]; H.ALG_HIER_LISTEN = 0
            H._CONDUCTOR_CACHE.clear()
            snap, onp2, dry_count, out_full = build_snapshot(H, p, ts, tk, se, vs, sl)
            for bi, i in enumerate(sl):
                dry_count_of[int(i)] = float(dry_count[bi])
            # kb=2 instrument: solve consult 1's OWN partial decode (masked)
            for bi, i in enumerate(sl):
                i = int(i)
                parse2, q2 = decode_row(H, onp2, bi, i, vs, mask=True)
                t = build_task(vs, i, parse2, q2)
                if t is not None:
                    uid = new_uid(); task_rows[uid] = ("KB2", i, False); tasks.append((uid,) + t)
                    kb2_task_uid[i] = uid

            # sanity: condition A via replay() must equal the full (unmodified) forward()
            H._HIER_DAMP = OVERRIDES["A_fixed"]["damp"]; H.ALG_HIER_LISTEN = 0
            H._CONDUCTOR_CACHE.clear()
            out_a = H.replay(p, snap, SNAP_KB)
            for bi, i in enumerate(sl):
                i = int(i)
                pa, qa = decode_row(H, {k: out_a[k].realize().numpy() for k in out_a if hasattr(out_a[k], "realize")}, bi, i, vs, mask=True)
                pf, qf = decode_row(H, {k: out_full[k].numpy() for k in out_full if hasattr(out_full[k], "numpy")}, bi, i, vs, mask=True)
                if pa != pf or qa != qf:
                    sane_mismatch.append(i)

            for cond_name, cfg in OVERRIDES.items():
                H._HIER_DAMP = cfg["damp"]; H.ALG_HIER_LISTEN = cfg["listen"]
                H._CONDUCTOR_CACHE.clear()
                out_c = H.replay(p, snap, SNAP_KB)
                out_np = {k: out_c[k].realize().numpy() for k in out_c if hasattr(out_c[k], "realize")}
                for bi, i in enumerate(sl):
                    i = int(i)
                    parse, q = decode_row(H, out_np, bi, i, vs, mask=True)
                    nr, nt = per_slot_right(parse, vs[i]["factors"])
                    per_slot[(cond_name, i)] = (nr, nt)
                    t = build_task(vs, i, parse, q)
                    uid = new_uid(); task_rows[uid] = (cond_name, i, False)
                    if t is not None:
                        tasks.append((uid,) + t)
                    else:
                        task_rows[uid] = (cond_name, i, "refused_no_parse")
            log(P, f"[pilot] {split_name} batch {s0}-{s0+len(sl)} done ({time.time() - t0:.0f}s so far)")

        # ---- the shared solver pool ----
        import multiprocessing as mp
        from chain_acc import _solve_task
        os.environ.setdefault("CA_WALL", "2")
        verdict = {}   # uid -> "correct"/"refused"/"wrong"
        for uid, (cond, i, flag) in task_rows.items():
            if flag == "refused_no_parse":
                verdict[uid] = "refused"
        real_tasks = [t for t in tasks]
        workers = int(os.environ.get("CA_WORKERS", "6"))
        log(P, f"[pilot] {split_name}: solving {len(real_tasks)} tasks ({len(task_rows) - len(real_tasks)} pre-refused) with {workers} workers")
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
                    log(P, f"[pilot] {split_name}: {len(real_tasks) - len(got)} tasks never returned -- counted as refused")
            for (uid, q, key, parse, gv, nvv) in real_tasks:
                st, val = got.get(uid, ("hung", None))
                if st != "solved":
                    verdict[uid] = "refused"
                elif val == key:
                    verdict[uid] = "correct"
                else:
                    verdict[uid] = "wrong"
        log(P, f"[pilot] {split_name}: solved ({time.time() - t0:.0f}s so far)")

        kb2_status = {}
        row_cond_verdict = collections.defaultdict(dict)
        for uid, (cond, i, flag) in task_rows.items():
            v = verdict.get(uid, "refused")
            if cond == "KB2":
                kb2_status[i] = v
            else:
                row_cond_verdict[i][cond] = v

        return dict(n=n, row_cond_verdict=row_cond_verdict, per_slot=per_slot,
                    dry_count_of=dry_count_of, kb2_status=kb2_status,
                    sane_mismatch=sane_mismatch)

    diet = run_split("DIET", DIET_TEST, DIET_NAME, N_DIET)
    wild = run_split("WILD", WILD_TEST, WILD_NAME, N_WILD)

    # ================================================================================
    # report
    # ================================================================================
    def render(split_name, res, L):
        n = res["n"]; rcv = res["row_cond_verdict"]
        L(f"n rows = {n}; sanity (replay(A) == full forward decode) mismatches = {len(res['sane_mismatch'])}"
          + (f" rows={res['sane_mismatch'][:10]}" if res["sane_mismatch"] else ""))
        L("")
        L("per-override row counts:")
        for cond in OVERRIDES:
            cnt = collections.Counter(rcv[i].get(cond, "refused") for i in rcv)
            L(f"    {cond:9s}  correct={cnt['correct']:4d}  refused={cnt['refused']:4d}  wrong={cnt['wrong']:4d}")
        oracle_cnt = collections.Counter()
        winner_hist = collections.Counter()
        reachable = 0
        gain_rows = []
        for i in rcv:
            verdicts = rcv[i]
            a_v = verdicts.get("A_fixed", "refused")
            best_cond, best_rank = "A_fixed", RANK[a_v]
            for cond in OVERRIDES:
                v = verdicts.get(cond, "refused")
                if RANK[v] > best_rank or (RANK[v] == best_rank and cond == "A_fixed"):
                    best_rank, best_cond = RANK[v], cond
            oracle_cnt[verdicts.get(best_cond, "refused")] += 1
            winner_hist[best_cond] += 1
            if best_rank > RANK[a_v]:
                gain_rows.append(i)
            if len({verdicts.get(c, "refused") for c in OVERRIDES}) > 1:
                reachable += 1
        L("")
        L(f"THE ORACLE (best override per row; ties favor A) vs A_fixed:")
        a_cnt = collections.Counter(rcv[i].get("A_fixed", "refused") for i in rcv)
        L(f"    A_fixed  correct={a_cnt['correct']:4d}  refused={a_cnt['refused']:4d}  wrong={a_cnt['wrong']:4d}")
        L(f"    oracle   correct={oracle_cnt['correct']:4d}  refused={oracle_cnt['refused']:4d}  wrong={oracle_cnt['wrong']:4d}")
        L(f"    rows strictly improved by the oracle over A: {len(gain_rows)}/{n}")
        L(f"    per-row winner histogram: {dict(winner_hist)}")
        L(f"    REACHABLE SET (verdict differs across >=2 overrides): {reachable}/{n} rows")
        L("")
        L("per-slot right count, mean by override:")
        for cond in OVERRIDES:
            vals = [res["per_slot"].get((cond, i), (0, 1)) for i in rcv]
            fracs = [nr / nt for nr, nt in vals if nt > 0]
            L(f"    {cond:9s}  mean right-fraction = {np.mean(fracs):.3f}  (n={len(fracs)})")
        L("")
        L("THE PILOT'S INSTRUMENTS (kb=2 dry count / kb=2 solver status) vs who wins the row:")
        dc_a_wins = [res["dry_count_of"].get(i, float("nan")) for i in rcv if winner_hist and _winner(rcv, i) == "A_fixed"]
        dc_other_wins = [res["dry_count_of"].get(i, float("nan")) for i in rcv if _winner(rcv, i) != "A_fixed"]
        L(f"    mean dry-count at kb=2: A-wins rows = {np.nanmean(dc_a_wins) if dc_a_wins else float('nan'):.2f} (n={len(dc_a_wins)}); "
          f"non-A-wins rows = {np.nanmean(dc_other_wins) if dc_other_wins else float('nan'):.2f} (n={len(dc_other_wins)})")
        kb2_by_winner = collections.defaultdict(collections.Counter)
        for i in rcv:
            w = "A_fixed" if _winner(rcv, i) == "A_fixed" else "non-A"
            kb2_by_winner[w][res["kb2_status"].get(i, "refused")] += 1
        for w in ("A_fixed", "non-A"):
            L(f"    kb=2 solver status, {w}-wins rows: {dict(kb2_by_winner[w])}")
        return dict(a_cnt=a_cnt, oracle_cnt=oracle_cnt, gain_rows=gain_rows, reachable=reachable,
                    winner_hist=winner_hist, n=n)

    def _winner(rcv, i):
        verdicts = rcv[i]
        a_v = verdicts.get("A_fixed", "refused")
        best_cond, best_rank = "A_fixed", RANK[a_v]
        for cond in OVERRIDES:
            v = verdicts.get(cond, "refused")
            if RANK[v] > best_rank or (RANK[v] == best_rank and cond == "A_fixed"):
                best_rank, best_cond = RANK[v], cond
        return best_cond

    lines = []
    L = lines.append
    L("=" * 92)
    L("THE PILOT'S CEILING -- RKX_241, schedule overrides via replay()'s _CONDUCTOR_CACHE patch (2026-10-08)")
    L("=" * 92)
    L("")
    L("-" * 92); L("DIET (.cache/form_pm35c_slice1024_valid2.jsonl) -- read first"); L("-" * 92)
    diet_summary = render("DIET", diet, L)
    L(""); L("-" * 92); L("WILD (.cache/wild_admitted_holdout.jsonl) -- the measurement, read ONCE"); L("-" * 92)
    wild_summary = render("WILD", wild, L)

    L(""); L("-" * 92); L("THE SIX-LINE READING"); L("-" * 92)
    L(f"1. ORACLE GAIN (diet): A correct={diet_summary['a_cnt']['correct']}/{diet_summary['n']}, "
      f"oracle correct={diet_summary['oracle_cnt']['correct']}/{diet_summary['n']} "
      f"(+{diet_summary['oracle_cnt']['correct']-diet_summary['a_cnt']['correct']}); "
      f"{len(diet_summary['gain_rows'])}/{diet_summary['n']} rows strictly improved by a perfect per-row pilot.")
    L(f"2. ORACLE GAIN (wild): A correct={wild_summary['a_cnt']['correct']}/{wild_summary['n']}, "
      f"oracle correct={wild_summary['oracle_cnt']['correct']}/{wild_summary['n']} "
      f"(+{wild_summary['oracle_cnt']['correct']-wild_summary['a_cnt']['correct']}); "
      f"{len(wild_summary['gain_rows'])}/{wild_summary['n']} rows strictly improved.")
    dom = diet_summary["winner_hist"].most_common(1)[0] if diet_summary["winner_hist"] else (None, 0)
    L(f"3. DOMINANCE (diet): winner histogram {dict(diet_summary['winner_hist'])} -- "
      f"{'a single override dominates (' + str(dom[0]) + ')' if dom[1] > 0.6*diet_summary['n'] else 'no single override dominates'}.")
    L(f"4. REACHABLE SET: diet {diet_summary['reachable']}/{diet_summary['n']} rows change verdict under some "
      f"override; wild {wild_summary['reachable']}/{wild_summary['n']}.")
    L("5. THE PILOT'S INSTRUMENTS: see the dry-count / kb=2-solver-status breakdown by winner above -- "
      "stated per-number (diet and wild tables), not re-asserted here; a clean separation would show up as "
      "a large gap between the A-wins and non-A-wins rows' mean dry count or status mix.")
    L("6. SCOPE LIMIT (restated): consult 2's own products (facts5/cert5/rack5) are threaded from the ORIGINAL "
      "fixed-schedule run on every override (replay() does not re-derive a consult live) -- B/C/D/E's effect is "
      "on how the state carries evidence, not on what the solver-backed consults themselves concluded.")

    txt = "\n".join(lines) + "\n"
    open(OUT_TXT, "w").write(txt)
    log(P, f"[pilot] wrote {OUT_TXT} ({time.time() - t0:.0f}s total)")


if __name__ == "__main__":
    main()
