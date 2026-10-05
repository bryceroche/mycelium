"""scripts/picker/gen_candidates.py -- THE PANEL PICKER's candidate generator (2026-10-05, zero-GPU,
delegate; CLAUDE.md's "never touch an existing script" rule honored by IMPORTING combined_oracle.py /
beam_oracle.py / sinkhorn_claim.py as libraries, never editing them).

REUSE, NOT REIMPLEMENTATION: this module imports scripts/combined_oracle.py (CO) for its three-axis
candidate machinery (CO.typeop_decisions, CO.apply_typeop_flips, CO.value_variants,
CO.apply_value_variant, CO.candidate_loglik, CO.fact_loglik, CO._digit_loglik) and, through CO, reuses
scripts/beam_oracle.py (BO: mask_row, slot_decisions, apply_flips, parse_sig, build_gv_nvv) and
scripts/sinkhorn_claim.py (SK: given_slots, affinity_row, onehot_digit_logits) EXACTLY as
combined_oracle.py itself does -- CO.BO and CO.SK are the same module objects. The per-row branch
construction loop below (build_row_candidates) is a verbatim copy of combined_oracle.py main()'s loop
body (lines ~426-489 as of 2026-10-05, commit fe4ee74d), refactored into a reusable function so it can
be pointed at a DIFFERENT dump (the training-diet valid2 slice) without touching combined_oracle.py's
own env-var-driven main(). The solve step is EXTENDED in one way only (same convention combined_oracle's
own docstring uses for its _solve_task copy): it also returns the FULL solver assignment (not just the
query variable's value), because the NL certifier's certificates (nl_certifier.py's cue_agree /
derived_stated) need the whole assignment, not only the queried value.

Usage (library):
    from gen_candidates import generate
    generate(dump_path=".cache/rawslots_slicevalid2_PMS8_241.pkl", out_pkl=".cache/picker/cand_diet.pkl",
             k_args=3, m_typeop=2, val_max=4, cand_max=64, wall=3, workers=8, run_uniq=True)

Usage (CLI): env GC_DUMP=... GC_OUT=... [GC_KARGS=3 GC_MTYPEOP=2 GC_VALMAX=4 GC_CANDMAX=64 GC_WALL=3
GC_WORKERS=8 GC_UNIQ=1 GC_LIMIT=0 GC_SELFGATE=<int>] .venv/bin/python3 scripts/picker/gen_candidates.py

Output pickle: a list (one dict per row), each:
  {i, text, key, q, top1_sig, branches: {sig: {parse, rank, tags, status, value, m, assignment,
                                                 unique, loglik}}}
No raw per-slot head arrays are kept in the output (angles.py needs only text/parse/assignment/q) --
this keeps the pickle small even at 775+311 rows x <=64 candidates.
"""
import os
import sys
import json
import time
import pickle

sys.path.insert(0, "."); sys.path.insert(0, "scripts")
import numpy as np

import combined_oracle as CO     # the three-axis machinery (args x type/op x values), reused verbatim
BO = CO.BO                       # beam_oracle: mask_row, slot_decisions, apply_flips, parse_sig, build_gv_nvv
SK = CO.SK                       # sinkhorn_claim: given_slots, affinity_row, onehot_digit_logits

UNIQ_BUDGET = CO.UNIQ_BUDGET      # 5000 -- the mint gate's own uniqueness budget (CLAUDE.md Sec.1)


# ----------------------------------------------------------------------------------------------------------
# THE SOLVE, extended with the full assignment (copied from combined_oracle.py's own _solve_task, which is
# itself copied from chain_acc.py / beam_oracle.py / sinkhorn_claim.py per that module's docstring's stated
# convention -- every one of these scripts carries its own copy for spawn-pool picklability).
# ----------------------------------------------------------------------------------------------------------
def _solve_task(t):
    import os as _os, sys as _sys
    _sys.path.insert(0, "."); _sys.path.insert(0, "scripts")
    from admit_annotation import solve_walled
    from alternator_bridge import problem_from_algebra3
    task_id, q, key, parse, gv, nvv = t
    wall = float(_os.environ.get("GC_WALL", "3"))
    gmax = max([int(v) for v in gv.values()] + [1]); m0 = int(min(10001, max(300, 2 * gmax, 2 * key)))
    res = {"status": "?"}
    for m in ([m0] + ([10000] if m0 < 10000 else [])):
        try:
            res = solve_walled(problem_from_algebra3(nvv, parse, gv, m), budget=5000, wall=wall)
        except Exception:
            return task_id, "unbuildable", None, None, None
        if res.get("status") == "solved":
            return task_id, "solved", int(res["assignment"][q]), m, list(res["assignment"])
    return task_id, res.get("status", "?"), None, None, None


def _uniqueness_task(t):
    """combined_oracle.py's _uniqueness_task, verbatim (the consistency judge's second door)."""
    import os as _os, sys as _sys, signal as _signal
    _sys.path.insert(0, "."); _sys.path.insert(0, "scripts")
    from admit_annotation import _Timeout, _alarm
    from alternator_bridge import problem_from_algebra3
    from mycelium.doors import certify_unique
    task_id, q, parse, gv, nvv, m, val = t
    wall = float(_os.environ.get("GC_WALL", "3"))
    old = _signal.signal(_signal.SIGALRM, _alarm); _signal.setitimer(_signal.ITIMER_REAL, wall)
    try:
        problem2 = problem_from_algebra3(nvv, parse, gv, m)
        uniq = certify_unique(problem2, q, val, budget=UNIQ_BUDGET, seed=0)
    except _Timeout:
        uniq = None
    except Exception:
        uniq = None
    finally:
        _signal.setitimer(_signal.ITIMER_REAL, 0); _signal.signal(_signal.SIGALRM, old)
    return task_id, uniq


# ----------------------------------------------------------------------------------------------------------
# THE CANDIDATE SET PER ROW -- verbatim copy of combined_oracle.py main()'s per-row loop body (the three
# axes' cross product, deduped by BO.parse_sig, truncated to cand_max by rank), refactored as a function.
# ----------------------------------------------------------------------------------------------------------
def build_row_candidates(rec, nd, k_args, m_typeop, val_max, cand_max, decode_fn):
    i, text, key = rec["i"], rec["text"], rec["key"]
    raw_row = {k: np.asarray(rec[k], dtype=np.float64) for k in ("pres", "ftype", "op", "dig", "args", "res") + (("dup",) if "dup" in rec else ())}
    q = int(np.asarray(rec["q"]).argmax())
    base_row = BO.mask_row(raw_row, text)

    args_decs = sorted(BO.slot_decisions(base_row), key=lambda d: d["margin"])
    k_args_row = min(k_args, len(args_decs)); top_args = args_decs[:k_args_row]

    typeop_decs = sorted(CO.typeop_decisions(base_row), key=lambda d: d["margin"])
    k_typeop_row = min(m_typeop, len(typeop_decs)); top_typeop = typeop_decs[:k_typeop_row]

    gslots, val_variants = CO.value_variants(raw_row, text, nd, val_max)

    branches = {}
    for abit in range(1 << k_args_row):
        flip_slots = {top_args[b]["slot"] for b in range(k_args_row) if abit & (1 << b)}
        row_a = BO.apply_flips(base_row, top_args, flip_slots)
        a_k = abit.bit_length()
        for tbit in range(1 << k_typeop_row):
            flip_keys = {(top_typeop[b]["slot"], top_typeop[b]["field"]) for b in range(k_typeop_row) if tbit & (1 << b)}
            row_at = CO.apply_typeop_flips(row_a, top_typeop, flip_keys)
            t_k = tbit.bit_length()
            for vlabel, vvals, vprio in val_variants:
                row_final = CO.apply_value_variant(row_at, gslots, vvals, nd)
                parse = decode_fn(row_final)
                sig = BO.parse_sig(parse)
                rank = a_k + t_k + vprio
                if sig not in branches:
                    branches[sig] = dict(parse=parse, rank=rank, tags={(a_k, t_k, vprio)})
                else:
                    b = branches[sig]
                    b["tags"].add((a_k, t_k, vprio)); b["rank"] = min(b["rank"], rank)

    sigs_sorted = sorted(branches.keys(), key=lambda s: branches[s]["rank"])
    kept_sigs = sigs_sorted[:cand_max]
    row_branches = {}
    for sig in kept_sigs:
        b = branches[sig]
        row_branches[sig] = dict(parse=b["parse"], rank=b["rank"], tags=sorted(b["tags"]))
    return dict(i=i, text=text, key=key, q=q, raw_row=raw_row, branches=row_branches)


# ----------------------------------------------------------------------------------------------------------
def generate(dump_path, out_pkl, k_args=3, m_typeop=2, val_max=4, cand_max=64, wall=3.0, workers=8,
             run_uniq=True, self_gate_expect=None, limit=0, log=print):
    from phase1_algebra_head import _decode_slots, N_DIG
    t0 = time.time()
    recs = pickle.load(open(dump_path, "rb"))
    if limit:
        recs = recs[:limit]
    n = len(recs)
    assert N_DIG == np.asarray(recs[0]["dig"]).shape[1], (
        f"N_DIG={N_DIG} from the head but the dump's digits are {np.asarray(recs[0]['dig']).shape[1]} wide -- "
        f"run under the FAMILY ENV (ALG_WIDE=1 sets N_DIG=7)")
    nd = N_DIG
    log(f"[gen-candidates] {n} rows from {dump_path} | K_ARGS<={k_args} M_TYPEOP<={m_typeop} VAL_MAX<={val_max} "
        f"CAND_MAX<={cand_max} | wall={wall}s workers={workers} uniq={int(run_uniq)}")

    per_row = []
    solve_tasks = {}
    cand_counts = []
    for ri, rec in enumerate(recs):
        row = build_row_candidates(rec, nd, k_args, m_typeop, val_max, cand_max, _decode_slots)
        cand_counts.append(len(row["branches"]))
        for sig, b in row["branches"].items():
            gv, nvv = BO.build_gv_nvv(b["parse"], row["q"])
            task_id = (ri, sig)
            if row["key"] is not None and b["parse"]:
                if task_id not in solve_tasks:
                    solve_tasks[task_id] = (row["q"], row["key"], b["parse"], gv, nvv)
                b["task_id"] = task_id
            else:
                b["task_id"] = None; b["status"] = "refused"; b["value"] = None; b["m"] = None; b["assignment"] = None
        per_row.append(row)
        if (ri + 1) % 50 == 0 or ri + 1 == n:
            log(f"[gen-candidates] decoded {ri+1}/{n} rows, {len(solve_tasks)} distinct solve tasks so far "
                f"({time.time()-t0:.0f}s)")

    log(f"[gen-candidates] {len(solve_tasks)} distinct (row,parse) solves queued "
        f"(candidates/row mean {np.mean(cand_counts):.1f}, max {max(cand_counts)}) -- solving...")
    import multiprocessing as mp
    os.environ["GC_WALL"] = str(wall)
    tasks = [(tid, *args) for tid, args in solve_tasks.items()]
    solved = {}
    with mp.get_context("spawn").Pool(workers) as pool:
        it = pool.imap_unordered(_solve_task, tasks, chunksize=1)
        n_done = 0
        try:
            for _ in range(len(tasks)):
                task_id, st, val, m, asg = it.next(timeout=wall * 3 + 30)
                solved[task_id] = (st, val, m, asg); n_done += 1
                if n_done % 500 == 0 or n_done == len(tasks):
                    log(f"[gen-candidates] solved {n_done}/{len(tasks)} ({time.time()-t0:.0f}s elapsed)")
        except mp.TimeoutError:
            log(f"[gen-candidates] {len(tasks) - len(solved)} tasks never returned -- counted as refused")
    t_solve = time.time() - t0

    for row in per_row:
        for sig, b in row["branches"].items():
            if b.get("task_id") is not None:
                st, val, m, asg = solved.get(b["task_id"], ("hung", None, None, None))
                b["status"] = st; b["value"] = val; b["m"] = m; b["assignment"] = asg
            b["correct"] = (b.get("status") == "solved" and b.get("value") == row["key"])

    uniq_results = {}
    if run_uniq:
        uniq_tasks = {}
        for row in per_row:
            for sig, b in row["branches"].items():
                if b.get("status") == "solved" and b.get("task_id") is not None:
                    if b["task_id"] not in uniq_tasks:
                        gv, nvv = BO.build_gv_nvv(b["parse"], row["q"])
                        uniq_tasks[b["task_id"]] = (row["q"], b["parse"], gv, nvv, b["m"], b["value"])
        u_list = [(tid, *args) for tid, args in uniq_tasks.items()]
        log(f"[gen-candidates] {len(u_list)} distinct solved parses -> uniqueness phase...")
        with mp.get_context("spawn").Pool(workers) as pool:
            it = pool.imap_unordered(_uniqueness_task, u_list, chunksize=1)
            n_done = 0
            try:
                for _ in range(len(u_list)):
                    task_id, uniq = it.next(timeout=wall * 3 + 30)
                    uniq_results[task_id] = uniq; n_done += 1
                    if n_done % 500 == 0 or n_done == len(u_list):
                        log(f"[gen-candidates] uniqueness {n_done}/{len(u_list)} ({time.time()-t0:.0f}s elapsed)")
            except mp.TimeoutError:
                log(f"[gen-candidates] {len(u_list) - len(uniq_results)} uniqueness checks never returned")
        for row in per_row:
            for sig, b in row["branches"].items():
                if b.get("status") == "solved":
                    b["unique"] = uniq_results.get(b["task_id"])
    else:
        for row in per_row:
            for sig, b in row["branches"].items():
                if b.get("status") == "solved":
                    b["unique"] = None

    for row in per_row:
        for sig, b in row["branches"].items():
            b["loglik"] = CO.candidate_loglik(row["raw_row"], b["parse"], nd) if b["parse"] else float("-inf")

    # self-gate (optional)
    top1_correct = 0
    for row in per_row:
        b0 = None
        for sig, b in row["branches"].items():
            if (0, 0, 0) in b["tags"]:
                b0 = b; break
        row["top1_sig"] = None
        if b0 is not None:
            for sig, b in row["branches"].items():
                if b is b0:
                    row["top1_sig"] = sig; break
        if b0 is not None and b0.get("correct"):
            top1_correct += 1
    if self_gate_expect is not None:
        log(f"[gen-candidates] SELF-GATE top1_correct={top1_correct} expected={self_gate_expect} "
            f"-> {'PASS' if top1_correct == self_gate_expect else 'MISMATCH'}")
    else:
        log(f"[gen-candidates] top1_correct={top1_correct}/{n} (no self-gate expectation given)")

    # strip raw_row before saving (angles.py needs only text/parse/assignment/q)
    out = []
    for row in per_row:
        out.append(dict(i=row["i"], text=row["text"], key=row["key"], q=row["q"], top1_sig=row["top1_sig"],
                         branches={sig: {k: v for k, v in b.items() if k != "task_id"} for sig, b in row["branches"].items()}))
    os.makedirs(os.path.dirname(out_pkl) or ".", exist_ok=True)
    pickle.dump(out, open(out_pkl, "wb"))
    log(f"[gen-candidates] wrote {out_pkl} ({len(out)} rows, {time.time()-t0:.0f}s total)")
    return out


def main():
    dump = os.environ.get("GC_DUMP", ".cache/rawslots_wild_PMS8_241.pkl")
    out = os.environ.get("GC_OUT", ".cache/picker/cand_" + os.path.basename(dump))
    k_args = int(os.environ.get("GC_KARGS", "3")); m_typeop = int(os.environ.get("GC_MTYPEOP", "2"))
    val_max = int(os.environ.get("GC_VALMAX", "4")); cand_max = int(os.environ.get("GC_CANDMAX", "64"))
    wall = float(os.environ.get("GC_WALL", "3")); workers = int(os.environ.get("GC_WORKERS", "8"))
    run_uniq = bool(int(os.environ.get("GC_UNIQ", "1"))); limit = int(os.environ.get("GC_LIMIT", "0"))
    sg = os.environ.get("GC_SELFGATE"); sg = int(sg) if sg is not None else None
    generate(dump, out, k_args, m_typeop, val_max, cand_max, wall, workers, run_uniq, sg, limit)


if __name__ == "__main__":
    main()
