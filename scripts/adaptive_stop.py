"""adaptive_stop.py -- ADAPTIVE STOPPING + THE PERCEIVER v0 STREAM (2026-10-05, "WORD GIVEN FOR THE
DISCRETE HALVES", item 1 of 3 delegated builds; docs/phase1_skeleton_spec.md 19:24).

QUESTION: chain_acc.py's headline masked read takes only the LAST breath's decoded graph per row (K_B-1,
the clocked loop's final rung). The continuous/discrete toolkit entry (10-05 19:07) registered adaptive
stopping as the discrete half of "time" (seven fixed breaths / stop on solved+unique) and the perceiver v0
as the organ that WOULD read the stream this script logs -- "nothing trained", the stop rule itself is the
consistency judge, read fresh at every breath.

MECHANISM (THE PER-BREATH DECODE TAP -- phase1_algebra_head.py:6503-6507, read, never edited):
    if int(os.environ.get("ALG_MINE_BREATHS", "0")) and K_B > 1 and slot_mask is not None:
        out["breaths_all"] = [_fed_core(_b9) for _b9 in out_breaths]   # per-breath SLOT CONTENT STATE
        out["heads_all"]   = [heads_of(_b9) for _b9 in out_breaths]    # per-breath DECODE HEADS (pres/
                                                                        # ftype/op/dig/args/res/...)
ALG_MINE_BREATHS=1 is an existing, already-gated env door (scripts/membrane_scale.py's `collect()` is the
precedent: it sets this flag to get out["fat_all"] from the SAME tap). Turning it on buys heads_all AND
breaths_all AND fat_all (slots<-tokens attention) in the SAME two forward calls chain_acc.py already makes
(pass-1 unmasked -> build_slot_masks()+alt2_fact_buf() -> pass-2 masked) -- no extra GPU pass per breath,
no re-running forward() with `stop_after` (which cannot reach breath 0 anyway: `stop_after=0` is falsy in
`_kb_stop = min(K_B, int(stop_after)+1) if stop_after else K_B`, so it silently runs the FULL loop --
a real footgun in that parameter, sidestepped entirely by this tap). out["query"] (the row's query
variable) is read ONCE per batch: qst (phase1_algebra_head.py:6000) is built from the grounding bank
alone, before the breath loop, and is never touched by _fact_inject on this family (no ALG_ALT3) --
breath-invariant by construction, confirmed by the last-breath assertion below.

EVERY FUNCTION THAT TOUCHES THE DECODE/SOLVE/JUDGE PATH IS IMPORTED, NEVER COPIED, from
phase1_algebra_head.py (_decode_slots, build_params, forward, load_alg, build_slot_masks, alt2_fact_buf,
K_VARS, L_FAC, N_DIG, _hier_band_dims), mycelium.rulebook (legal_digit_logits, the CA_MASK numeral mask),
mycelium.custody_gold (row_gold, the custody-gold door), mycelium.doors (certify_unique), admit_annotation
(solve_walled, _Timeout, _alarm) and alternator_bridge (problem_from_algebra3). The ONLY functions below
that duplicate logic from elsewhere are (a) _solve_task / _uniqueness_task, COPIED (not imported) from
chain_acc.py / combined_oracle.py per THIS CODEBASE'S OWN STANDING CONVENTION (every one of chain_acc.py,
beam_oracle.py, sinkhorn_claim.py, combined_oracle.py carries its own copy, each doing its own sys.path
insertion inside the function body, because a spawned multiprocessing worker must import these modules
fresh regardless of which script queued the task -- picklability, not laziness) and (b) _nlc_cert (the
per-row certificate math), which nl_certifier.py never factored into a reusable function (it is inline in
main()'s loop) -- copied here with the SAME cue sets imported directly off the nl_certifier module
(`import nl_certifier as NLC; NLC.ADD_CUES/MUL_CUES`), never redefined.

THE BUILD (per the 19:24 registration):
  for every row of the wild holdout, for EVERY breath kb = 0..K_B-1: decode that breath's heads into a
  factor graph exactly as chain_acc's CA_MASK=1 does (numeral mask on non-relation slots' digits, then
  _decode_slots); run the CONSISTENCY JUDGE (solved + unique, scripts/combined_oracle.py's door: the June
  solver's own status plus mycelium.doors.certify_unique -- the key is NEVER consulted by the judge); stop
  at the FIRST breath that passes -- that breath's solved value is the row's answer. Rows that never pass
  take the last breath (K_B-1), identically to chain_acc's current (non-adaptive) behaviour -- this script
  computes and reports that baseline FROM THE SAME RUN, so "rows >= last-breath's" is a same-run comparison,
  not a cross-run one.

THE PERCEIVER v0 STREAM (nothing trained; the stop rule above already reads this evidence qualitatively --
this sidecar is what a trained perceiver would consume): per row x breath --
  - membrane attention entropy (nats), over GIVEN-predicted present slots and over ALL present slots,
    from the head-mean slots<-tokens attention (out["fat_all"][kb], already a softmax; renormalized over
    real tokens per membrane_scale.py's convention -- padding mass is near-zero but not exactly zero);
  - atlas distances: mean cosine of GIVEN-predicted / REL-predicted present slots' CONTENT state
    (out["breaths_all"][kb], trimmed to atlas_radius_read.py's content dims -- _hier_band_dims()'s band
    union, the 128 clock dims excluded) to a centroid computed ON THE FLY from this run's own states at
    that breath (the mean over the WHOLE read's present slots predicted that kind at that breath -- never
    a banked centroid from a different run/body). Kind is the PREDICTED ftype argmax (class 0 = rel,
    class 1 = given, scripts/combined_oracle.py:353's _FTYPE_IDX), never gold -- the perceiver reads its
    own machine's state, not the key;
  - the nl_certifier score (scripts/nl_certifier.py's "cert", the mean of its 6 certificates) of the row's
    LAST-BREATH decode -- row-level, constant across breaths per the task's own framing (the certificates
    read the TEXT against a graph; broadcasting the one graph's score across the row's breath axis keeps
    every per-breath array the same shape without implying the certifier re-reads anything);
  - solver status per breath (solved / unsat / budget / refused / unbuildable);
  - the rack's dry count per breath, if a rack sidecar for this TAG exists (<LV_DUMP>.rack.npz,
    scripts/loop_val.py's convention, 2026-10-05) -- else NaN throughout. PMS8_241 does not run ALG_RACK
    (THE RACK is RK_241's body only), so this column is NaN for this run by construction, not by omission.

BARS (pinned, ledger 19:24): adaptive rows >= last-breath rows (this run); regressions <= 2 (rows right at
the last breath, wrong under adaptive stop).

THE RAW BREATH DUMP (Bryce's addition, 2026-10-05, "parse-space concentration"): beside everything above,
every row's RAW (pre-numeral-mask) per-breath slot heads are written to .cache/rawslots_breaths_wild_
<TAG>.pkl -- {kb: [row-record, ...] for kb in range(K_B)}, each row-record in chain_acc.py's own CA_RAWDUMP
format ("i","text","q",<KEYS: pres/ftype/op/dig/args/res/optionally dup>,"key"), float16. Pointing
scripts/combined_oracle.py (or beam_oracle.py/sinkhorn_claim.py, which read the identical format) at
rawbreaths[kb] in place of a chain_acc CA_RAWDUMP file lets a zero-GPU meter enumerate candidate graphs
PER BREATH and count how many the solver still accepts -- parse-space concentration as the loop runs,
never a second GPU pass.

usage:
  DEV=CPU .venv/bin/python3 scripts/adaptive_stop.py .cache/sharp_PMS8_241.safetensors --tag PMS8_241 --rows 3
  flock -w 36000 .cache/gpu.lock env DEV=PCI+AMD ... .venv/bin/python3 scripts/adaptive_stop.py \
      .cache/sharp_PMS8_241.safetensors --tag PMS8_241
(the family env is looked up by --tag in FAMILY_ENVS below; pass --family-env "K=V K2=V2 ..." to add to or
override it for a body not yet in the table -- EY_241/RK_241/HS_241-style bodies, when they land, per the
mission brief's own instruction to document which env each body needs.)
"""
import os
import re
import sys
import json
import pickle
import argparse
import collections

sys.path.insert(0, ".")
sys.path.insert(0, "scripts")
import numpy as np

# ===========================================================================================================
# FAMILY ENVS, one per body (documented, per the brief's instruction -- "document which env each body
# needs"). PMS8_241's is $FAM + $SURF8 from .cache/pms8_chain.sh WITHOUT any ALG_HIER_*/ALG_EYES/ALG_RACK
# bits ("the SURF8 recipe without HIER") -- verbatim-identical to scripts/membrane_scale.py's _FAM dict,
# which that script's own docstring confirms matches the banked masked-wild read
# (.cache/read_legal_wild_PMS8_241.log -> .cache/ps_legal_wild_PMS8_241.npz) bit for bit. ALG_MINE_BREATHS=1
# is THIS script's own addition (the per-breath tap; membrane_scale.py adds the same flag for its own,
# narrower, fat_all-only tap).
# ===========================================================================================================
FAMILY_ENVS = {
    "PMS8_241": {
        "DEV": "PCI+AMD", "ALG2": "1", "ALG_FTYPES": "9", "ALG_DUP": "1",
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
        "ALG_ROUTER": "2", "R_GAIN_INIT": "1.0", "ALG_FREEZE": "r_gain",
        "ALG_ROUTER_PTR": "0.0", "ALG_SPAN_ALL": "1", "ALG_SPAN_ARGS": "1",
        "ALG_SPAN_OP": "1", "ALG_SPAN_RCUE": "1", "ALG_SPAN_ARCUE": "1",
        "ALG_PTR_SURF": "role:add:2.0",
        "ALG_TEST": ".cache/wild_admitted_holdout.jsonl", "ALG_TEST_NAME": "wildhold",
        "ALG_MINE_BREATHS": "1",
    },
    # EY_241 (THE EYES): .cache/eyes_chain.sh's $FAM + $SURF8 + the hierarchical-state envs
    # (ALG_HIER_READ=1 ALG_HIER_WAIST=1 ALG_HIER_DAMP=2,4,0 ALG_HIER_TAU=<0 or 1.0>) + ALG_EYES=1 -- not
    # known to be in .cache/ until that chain lands; pass --family-env to extend the PMS8_241 table above
    # (SURF8 is shared) with those four ALG_HIER_*/ALG_EYES keys once EY_241's checkpoint exists.
    # RK_241 (THE RACK): same SURF8 body as PMS8_241 + ALG_RACK=1 ALG_RACK_TESTS=given_unique
    # ALG_RACK_FREEZE=leaf -- .cache/rack_chain.sh, once its checkpoint lands; the dry-count column reads
    # RK_241's own <LV_DUMP>.rack.npz convention ONLY IF this script is pointed at a loop_val-produced dump
    # beside it (not built here: this script runs its own two forwards, mirroring chain_acc, not loop_val).
    # HS_241 / HSd_241: .cache/hierd_chain.sh's FAM + SURF8 + ALG_HIER_READ=1 ALG_HIER_WAIST=1
    # ALG_HIER_DAMP=2,4,0 (+ ALG_HIER_TAU=1.0 for HSd) -- no ALG_EYES.
}


def _build_family_env(tag, extra):
    fam = dict(FAMILY_ENVS.get(tag, {}))
    for kv in extra.split():
        if "=" not in kv:
            continue
        k, v = kv.split("=", 1)
        fam[k] = v
    if not fam:
        raise SystemExit(f"adaptive_stop: no family env known for tag={tag!r} -- pass --family-env 'K=V ...'")
    for k, v in fam.items():
        os.environ.setdefault(k, v)   # setdefault: a caller's `env DEV=CPU ...` wins (the CPU-smoke door)
    return fam


# ===========================================================================================================
# _solve_task / _uniqueness_task -- COPIED (not imported), per chain_acc.py / combined_oracle.py's own
# standing convention (see module docstring): module-level for multiprocessing spawn picklability, each
# doing its own sys.path insertion so a freshly spawned worker can import admit_annotation/alternator_bridge
# regardless of which script queued the task. Identical semantics to combined_oracle.py's copy, which
# itself differs from chain_acc.py's only in returning `m` (the solving rung) for the judge's uniqueness
# door; this copy ALSO returns the full assignment (asg) so the per-row nl_certifier read (derived_stated)
# never needs a second solve of the same (nvv, parse, gv, m).
# ===========================================================================================================
def _solve_task(t):
    import os as _os
    import sys as _sys
    _sys.path.insert(0, ".")
    _sys.path.insert(0, "scripts")
    from admit_annotation import solve_walled
    from alternator_bridge import problem_from_algebra3
    task_id, q, key, parse, gv, nvv = t
    wall = float(_os.environ.get("AS_WALL", "3"))
    gmax = max([int(v) for v in gv.values()] + [1])
    m0 = int(min(10001, max(300, 2 * gmax, 2 * key)))
    res = {"status": "?"}
    for m in ([m0] + ([10000] if m0 < 10000 else [])):
        try:
            res = solve_walled(problem_from_algebra3(nvv, parse, gv, m), budget=5000, wall=wall)
        except Exception:
            return task_id, "unbuildable", None, None, None
        if res.get("status") == "solved":
            asg = res["assignment"]
            return task_id, "solved", int(asg[q]), [int(x) for x in asg], m
    return task_id, res.get("status", "?"), None, None, None


def _uniqueness_task(t):
    """THE CONSISTENCY JUDGE's second door (combined_oracle.py's _uniqueness_task, copied verbatim):
    mycelium.doors.certify_unique on a FRESH problem rebuilt at the same (nvv, parse, gv, m) that solved --
    bans the found value from q's domain, re-solves; True only on a true 'unsat' certificate (budget/
    solved-again both refuse, the door's own twelve-site law)."""
    import os as _os
    import sys as _sys
    import signal as _signal
    _sys.path.insert(0, ".")
    _sys.path.insert(0, "scripts")
    from admit_annotation import _Timeout, _alarm
    from alternator_bridge import problem_from_algebra3
    from mycelium.doors import certify_unique
    task_id, q, parse, gv, nvv, m, val = t
    wall = float(_os.environ.get("AS_WALL", "3"))
    old = _signal.signal(_signal.SIGALRM, _alarm)
    _signal.setitimer(_signal.ITIMER_REAL, wall)
    try:
        problem2 = problem_from_algebra3(nvv, parse, gv, m)
        uniq = certify_unique(problem2, q, val, budget=5000, seed=0)
    except _Timeout:
        uniq = None
    except Exception:
        uniq = None
    finally:
        _signal.setitimer(_signal.ITIMER_REAL, 0)
        _signal.signal(_signal.SIGALRM, old)
    return task_id, uniq


def _cos(a, b):
    return (a * b).sum(-1) / (np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1) + 1e-12)


def _nlc_cert(text, parse, q, asg):
    """THE NL CERTIFIER's per-row score (scripts/nl_certifier.py's "cert" -- the mean of its 6
    certificates). COPIED (not imported): nl_certifier.main()'s inner loop never factors this into a
    function; the cue sets are imported, never redefined (see below). Verbatim formulas."""
    import nl_certifier as NLC
    import stamp_arg_mentions as SAM
    import args_census as AC
    givens = [f for f in parse if f["ftype"] == "given"]
    rels = [f for f in parse if f["ftype"] == "rel"]
    nums = [int(x) for x in re.findall(r"(?<![\d.])\d{1,3}(?![\d])", text)]
    numset = set(nums)
    gvals = [int(f["value"]) for f in givens if f.get("value") is not None]
    value_present = (np.mean([v in numset for v in gvals]) if gvals else 0.0)
    coverage = (np.mean([n in set(gvals) for n in nums]) if nums else 1.0)
    no_double = (1.0 - (len(gvals) - len(set(gvals))) / len(gvals)) if gvals else 1.0
    bounds = AC.sentence_bounds(text)
    sspans = AC.sentence_spans(text, bounds)
    intro = SAM.build_intro_map(parse)
    memo = {}
    sol = list(asg) if asg is not None else []
    agree = []
    for j, f in enumerate(parse):
        if f["ftype"] != "rel":
            continue
        cl = SAM.clause_of(j, parse, text, bounds, sspans, sol, intro, memo)
        words = (set(w.lower() for w in re.findall(r"[a-z']+", " ".join(text[a:b] for a, b in SAM.windows_of(cl, sspans))))
                 if cl is not None else set())
        cues = NLC.ADD_CUES if f.get("op") == "add" else NLC.MUL_CUES
        agree.append(1.0 if (words & cues) else 0.0)
    cue_agree = float(np.mean(agree)) if agree else 1.0
    reach = {f["result"] for f in rels} | {f["var"] for f in givens}
    chain_reaches = 1.0 if q in reach else 0.0
    if asg is not None and rels:
        derived_stated = float(np.mean([int(asg[f["result"]]) in numset for f in rels if f["result"] < len(asg)]))
    else:
        derived_stated = 0.0
    certs = dict(value_present=value_present, coverage=coverage, no_double=no_double,
                 cue_agree=cue_agree, chain_reaches=chain_reaches, derived_stated=derived_stated)
    return float(np.mean(list(certs.values())))


def _rack_sidecar_path(tag):
    """The rack's sidecar convention (scripts/loop_val.py, 2026-10-05): <LV_DUMP>.rack.npz. This script
    runs its own two forwards (never loop_val.py), so no LV_DUMP exists here -- the dry column is NaN
    throughout UNLESS a sidecar matching this tag already sits in .cache/ from a separate loop_val run
    (checked by convention, not produced here). PMS8_241 does not run ALG_RACK (THE RACK is RK_241's body
    only) so this is expected to return None for this run."""
    for cand in (f".cache/dump_wild_{tag}.pkl.rack.npz", f".cache/ps_legal_wild_{tag}.npz.rack.npz"):
        if os.path.exists(cand):
            return cand
    import glob
    hits = glob.glob(f".cache/*{tag}*.rack.npz")
    return hits[0] if hits else None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ckpt")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--rows", type=int, default=0, help="limit to the first N wild rows (0 = all 311; the CPU smoke uses 3)")
    ap.add_argument("--family-env", default="", help="extra K=V pairs, space-separated, merged onto FAMILY_ENVS[--tag]")
    ap.add_argument("--workers", type=int, default=int(os.environ.get("AS_WORKERS", "6")))
    ap.add_argument("--wall", type=float, default=float(os.environ.get("AS_WALL", "3")))
    ap.add_argument("--out", default="", help="sidecar path override (default .cache/perceiver_stream_<tag>.npz)")
    args = ap.parse_args()
    os.environ["AS_WALL"] = str(args.wall)
    _build_family_env(args.tag, args.family_env)

    import time
    import multiprocessing as mp
    from tinygrad import Tensor, dtypes
    from tinygrad.nn.state import safe_load
    import phase1_algebra_head as H
    from phase1_algebra_head import (build_params, forward, load_alg, build_slot_masks,
                                     alt2_fact_buf, _decode_slots, K_VARS, L_FAC, N_DIG)
    from mycelium.rulebook import legal_digit_logits
    from mycelium.custody_gold import row_gold

    t0 = time.time()
    vs, vst, vtk, vg, vse = load_alg("test")
    n_full = len(vs)
    n = min(n_full, args.rows) if args.rows else n_full
    print(f"[adaptive-stop] tag={args.tag} ckpt={args.ckpt} rows={n}/{n_full} DEV={os.environ.get('DEV')}", flush=True)

    p = build_params(0)
    sd = safe_load(args.ckpt)
    assert set(sd) == set(p), (sorted(set(sd) - set(p))[:4], sorted(set(p) - set(sd))[:4])
    for k in p:
        p[k].assign(sd[k].to(p[k].device).cast(p[k].dtype)).realize()

    K_B = int(os.environ.get("ALG_BREATH", "1"))
    assert K_B > 1, "adaptive stopping needs a breathing body (ALG_BREATH > 1)"
    bands, clock_dims = H._hier_band_dims()
    CONTENT = np.sort(np.concatenate(bands))
    assert len(np.intersect1d(CONTENT, clock_dims)) == 0
    C = len(CONTENT)

    ent_given = np.full((n, K_B), np.nan, np.float32)
    ent_all = np.full((n, K_B), np.nan, np.float32)
    state_store = np.zeros((n, K_B, L_FAC, C), np.float32)
    kindpred = np.full((n, K_B, L_FAC), -1, np.int8)   # -1 absent, 0 rel, 1 given, 2 other
    row_info = {}
    tasks_by_id = {}
    # THE RAW BREATH DUMP (Bryce, 2026-10-05, "parse-space concentration"): per breath, every row's RAW
    # (pre-numeral-mask) slot heads in chain_acc's own CA_RAWDUMP record format -- {"i","text","q",
    # "pres","ftype","op","dig","args","res",("dup"),"key"} -- so scripts/combined_oracle.py (or any of
    # beam_oracle.py/sinkhorn_claim.py's branching machinery, which all read that exact format) can be
    # pointed at rawbreaths[kb] and enumerate how many candidate graphs the solver still accepts AT THAT
    # BREATH, zero GPU. float16 throughout (Bryce: "float16 logits are fine; cap at what combined_oracle
    # actually reads") -- combined_oracle.py only ever does np.asarray(rec[k]) on these fields, which
    # upcasts transparently.
    rawbreaths = {kb: [] for kb in range(K_B)}

    for s0 in range(0, n, 8):
        sl = np.arange(s0, min(s0 + 8, n))
        pad = 8 - len(sl)
        sl_p = np.concatenate([sl, sl[:1].repeat(pad)]) if pad else sl
        nv = np.array([vs[int(i)].get("n_vars", K_VARS) for i in sl_p])
        ma = np.array([vs[int(i)].get("m", 0) for i in sl_p])
        se_np = vse[sl_p].astype(np.int32)
        ts = Tensor(np.ascontiguousarray(vst[sl_p]), dtype=dtypes.half)
        tk = Tensor(vtk[sl_p].astype(np.float32))
        se = Tensor(se_np, dtype=dtypes.int)
        # pass 1: unmasked parse -> the slot mask + live facts (chain_acc's own cycle, verbatim)
        o0 = forward(p, ts, tk, se)
        onp0 = {k: o0[k].numpy() for k in ("fat", "args", "res")}
        mk = build_slot_masks(onp0, se_np)
        _ka = ("pres", "ftype", "op", "dig") + (("dup",) if "dup" in o0 else ())
        _oa = {**onp0, **{k: o0[k].numpy() for k in _ka}}
        fb = alt2_fact_buf(_oa, se_np, nv, ma)
        fact_t = Tensor(fb, dtype=dtypes.float)
        # pass 2: masked walk, with THE PER-BREATH TAP (ALG_MINE_BREATHS=1, set in the family env)
        o = forward(p, ts, tk, se, slot_mask=Tensor(mk, dtype=dtypes.float), fact_buf=fact_t)
        qlog = o["query"].numpy()
        qv = qlog.argmax(-1)
        heads_all = o["heads_all"]
        fat_all = [t.numpy() for t in o["fat_all"]]
        breaths_all = [t.numpy() for t in o["breaths_all"]]
        assert len(heads_all) == K_B and len(fat_all) == K_B and len(breaths_all) == K_B
        KEYS = ("pres", "ftype", "op", "dig", "args", "res") + (("dup",) if "dup" in heads_all[0] else ())
        tkm_np = (vtk[sl_p].astype(np.float32) > 0.5)
        for kb in range(K_B):
            if kb == K_B - 1:
                # IDENTITY FIX (bug audit 2026-10-06): heads_all[K_B-1] is the bare pre-injection
                # decode (phase1_algebra_head.py's heads_of() on the raw final-breath state), but
                # `o` -- this SAME forward() call's return dict -- has already received post-hoc
                # head injections that apply only to the final breath (e.g. ALG_PTR_SURF's trained
                # role-pointer addition into out["args"], gain 2.0, on PMS8_241). The tap therefore
                # silently disagrees with chain_acc.py's read of `o` directly on the one breath that
                # matters most -- the root cause of the banked 11-vs-14 discrepancy (ledger
                # 2026-10-06 07:09; row-level repro: rows 48/81/292 flip correct -> wrong/refused
                # under the tap's pre-injection args, net 14-3=11). Read the last breath from `o`
                # (falling back to the tap only for a key `o` doesn't carry) so "last breath" means
                # the same thing here as it does in chain_acc.py, by construction.
                hk = {k: (o[k].numpy() if k in o else heads_all[kb][k].numpy()) for k in KEYS}
            else:
                hk = {k: heads_all[kb][k].numpy() for k in KEYS}
            fa_kb = fat_all[kb]
            st_kb = breaths_all[kb][:, :, CONTENT]
            for bi, i in enumerate(sl):
                i = int(i)
                row = {k: hk[k][bi].copy() for k in KEYS}
                try:
                    key_dump = int(row_gold(vs[i]))
                except Exception:
                    key_dump = None
                rawbreaths[kb].append({"i": i, "text": vs[i]["text"], "q": qlog[bi].astype(np.float16),
                                       **{k: row[k].astype(np.float16) for k in KEYS}, "key": key_dump})
                pres_mask = row["pres"] > 0
                ftype_am = row["ftype"].argmax(-1)
                kindpred[i, kb] = np.where(~pres_mask, -1, np.where(ftype_am == 0, 0, np.where(ftype_am == 1, 1, 2)))
                state_store[i, kb] = st_kb[bi]
                p_tok = fa_kb[bi] * tkm_np[bi][None, :]
                p_tok = p_tok / (p_tok.sum(-1, keepdims=True) + 1e-12)
                ent = -(p_tok * np.log(p_tok + 1e-12)).sum(-1)
                if pres_mask.any():
                    ent_all[i, kb] = float(ent[pres_mask].mean())
                    giv = pres_mask & (ftype_am == 1)
                    if giv.any():
                        ent_given[i, kb] = float(ent[giv].mean())
                # THE NUMERAL MASK (chain_acc's CA_MASK=1), applied to THIS breath's own digits, then decode
                masked_row = dict(row)
                masked_row["dig"] = row["dig"].copy()
                for j in range(row["ftype"].shape[0]):
                    if int(row["ftype"][j].argmax()) == 0:
                        continue
                    fake = legal_digit_logits(masked_row["dig"][j], vs[i]["text"])
                    if fake is not None:
                        masked_row["dig"][j] = fake
                parse = _decode_slots(masked_row)
                q = int(qv[bi])
                ri = row_info.setdefault(i, {"text": vs[i]["text"], "q": q, "parses": {}})
                ri["parses"][kb] = parse
                if "key" not in ri:
                    try:
                        ri["key"] = int(row_gold(vs[i]))
                    except Exception:
                        ri["key"] = None
                if ri["key"] is None or not parse:
                    ri.setdefault("refused_kb", set()).add(kb)
                    continue
                used = ([f.get("var") for f in parse if f["ftype"] == "given"]
                        + [a for f in parse if f["ftype"] == "rel" for a in list(f["args"]) + [f["result"]]])
                nvv = max([q + 1] + [v + 1 for v in used if v is not None])
                gvd = {f["var"]: f["value"] for f in parse if f["ftype"] == "given"}
                tasks_by_id[(i, kb)] = (q, ri["key"], parse, gvd, nvv)
        if (s0 // 8) % 10 == 0:
            print(f"[adaptive-stop] GPU pass {s0}/{n} ({time.time()-t0:.0f}s)", flush=True)

    print(f"[adaptive-stop] GPU passes done ({time.time()-t0:.0f}s); {len(tasks_by_id)} solver tasks "
          f"over {n} rows x {K_B} breaths", flush=True)

    # THE RAW-DUMP PATH GUARD (bug audit 2026-10-06): the banked dump is the FULL read (all n_full rows);
    # a `--rows N` smoke must never overwrite it (one did, 21:05 — a 24-row file replaced the 311-row
    # artifact the ledger's 07:09 entry banks). A partial read writes beside it, suffixed by its n.
    raw_path = (f".cache/rawslots_breaths_wild_{args.tag}.pkl" if n == n_full
                else f".cache/rawslots_breaths_wild_{args.tag}_rows{n}.pkl")
    pickle.dump(rawbreaths, open(raw_path, "wb"))
    print(f"[adaptive-stop] wrote {raw_path}: dict breath(0..{K_B-1}) -> list of {n} row-records "
          f"(chain_acc's CA_RAWDUMP format -- i/text/q/{'/'.join(KEYS)}/key, float16) "
          f"-- point combined_oracle.py/beam_oracle.py/sinkhorn_claim.py's DUMP at rawbreaths[kb] for a "
          f"per-breath zero-GPU read (~{os.path.getsize(raw_path)/1e6:.1f}MB on disk)", flush=True)

    # ---------------- solve phase ----------------
    solve_out = {}
    tasks = [(tid, q, key, parse, gvd, nvv) for tid, (q, key, parse, gvd, nvv) in tasks_by_id.items()]
    if tasks:
        with mp.get_context("spawn").Pool(args.workers) as pool:
            it = pool.imap_unordered(_solve_task, tasks, chunksize=1)
            for _ in range(len(tasks)):
                try:
                    tid, st, val, asg, m = it.next(timeout=args.wall * 3 + 30)
                except mp.TimeoutError:
                    break
                solve_out[tid] = (st, val, asg, m)
    print(f"[adaptive-stop] solve phase done ({time.time()-t0:.0f}s); "
          f"{sum(1 for v in solve_out.values() if v[0] == 'solved')}/{len(tasks)} solved", flush=True)

    # ---------------- uniqueness phase (the judge's second door) ----------------
    uniq_tasks = []
    for tid, (q, key, parse, gvd, nvv) in tasks_by_id.items():
        st, val, asg, m = solve_out.get(tid, ("hung", None, None, None))
        if st == "solved":
            uniq_tasks.append((tid, q, parse, gvd, nvv, m, val))
    uniq_out = {}
    if uniq_tasks:
        with mp.get_context("spawn").Pool(args.workers) as pool:
            it = pool.imap_unordered(_uniqueness_task, uniq_tasks, chunksize=1)
            for _ in range(len(uniq_tasks)):
                try:
                    tid, uniq = it.next(timeout=args.wall * 3 + 30)
                except mp.TimeoutError:
                    break
                uniq_out[tid] = uniq
    print(f"[adaptive-stop] uniqueness phase done ({time.time()-t0:.0f}s); "
          f"{sum(1 for v in uniq_out.values() if v is True)}/{len(uniq_tasks)} unique", flush=True)

    # ---------------- per-row, per-breath status + the stop rule ----------------
    solver_status = np.full((n, K_B), "hung", dtype="<U11")
    last_kb = K_B - 1
    stop_kb_arr = np.full((n,), -1, np.int32)
    last_label = [None] * n
    adaptive_label = [None] * n
    last_asg = [None] * n

    def _solved(tid):
        """(status, value, assignment) for a task id -- 'refused' with no solve attempt at all
        (no parse / no key, chain_acc's own convention) when the id was never queued."""
        if tid not in tasks_by_id:
            return "refused", None, None
        st, val, asg, m = solve_out.get(tid, ("hung", None, None, None))
        return st, val, asg

    def _label(tid, key):
        st, val, asg = _solved(tid)
        if st != "solved":
            return "refused", asg
        return ("correct" if val == key else "wrong"), asg

    for i in range(n):
        ri = row_info[i]
        key = ri["key"]
        for kb in range(K_B):
            solver_status[i, kb] = _solved((i, kb))[0]
        stop_kb = None
        for kb in range(K_B):
            st, val, asg = _solved((i, kb))
            if st == "solved" and uniq_out.get((i, kb)) is True:
                stop_kb = kb
                break
        stop_kb_arr[i] = stop_kb if stop_kb is not None else -1
        lab_last, asg_last = _label((i, last_kb), key)
        last_label[i] = lab_last
        last_asg[i] = asg_last
        adaptive_label[i] = _label((i, stop_kb), key)[0] if stop_kb is not None else lab_last

    last_correct = sum(1 for l in last_label if l == "correct")
    last_refused = sum(1 for l in last_label if l == "refused")
    last_wrong = sum(1 for l in last_label if l == "wrong")
    adaptive_correct = sum(1 for l in adaptive_label if l == "correct")
    adaptive_refused = sum(1 for l in adaptive_label if l == "refused")
    adaptive_wrong = sum(1 for l in adaptive_label if l == "wrong")
    regressions = [i for i in range(n) if last_label[i] == "correct" and adaptive_label[i] != "correct"]
    fixes = [i for i in range(n) if last_label[i] != "correct" and adaptive_label[i] == "correct"]
    stop_hist = collections.Counter(int(stop_kb_arr[i]) for i in range(n))

    print("=" * 100)
    print(f"[adaptive-stop] {args.tag} ({os.path.basename(args.ckpt)}) on wildhold, n={n}, K_B={K_B}")
    print(f"  LAST-BREATH BASELINE (breath {last_kb}, == chain_acc CA_MASK=1): "
          f"correct {last_correct} ({last_correct/n:.3f}) | refused {last_refused} | wrong {last_wrong}")
    print(f"  ADAPTIVE STOP (judge: solved+unique, key never consulted): "
          f"correct {adaptive_correct} ({adaptive_correct/n:.3f}) | refused {adaptive_refused} | wrong {adaptive_wrong}")
    print(f"  STOP HISTOGRAM (breath index -> rows stopped there; -1 = never passed, used last breath): "
          + ", ".join(f"b{k}:{stop_hist[k]}" for k in sorted(stop_hist)))
    print(f"  REGRESSIONS (last-breath correct -> adaptive not correct) [bar <= 2]: {len(regressions)}  rows={regressions}")
    print(f"  FIXES       (last-breath not correct -> adaptive correct): {len(fixes)}  rows={fixes}")
    bar_rows = adaptive_correct >= last_correct
    bar_regr = len(regressions) <= 2
    print(f"  BARS: rows >= last-breath's ({adaptive_correct} >= {last_correct}): {'PASS' if bar_rows else 'MISS'} | "
          f"regressions <= 2: {'PASS' if bar_regr else 'MISS'}  ==> {'PASS' if (bar_rows and bar_regr) else 'MISS'}")
    if args.tag == "PMS8_241" and n == 311:
        if last_correct == 14:
            print("  ASSERT OK: last-breath baseline == chain_acc's banked PMS8_241 mask=1 read (14/311).")
        else:
            print(f"  ASSERT MISMATCH: last-breath baseline {last_correct} != banked 14 -- explain before trusting the "
                  f"adaptive comparison (candidates: AS_WALL={args.wall} vs chain_acc's CA_WALL default, a solver timeout "
                  f"under load, or a non-determinism in the m-ladder/budget walk; the per-breath tap itself is read-only "
                  f"and should not change the final breath's heads -- see the module docstring's bit-identity claim).")

    # ---------------- THE PERCEIVER v0 STREAM ----------------
    mu_given = np.full((K_B, C), np.nan, np.float32)
    mu_rel = np.full((K_B, C), np.nan, np.float32)
    for kb in range(K_B):
        gm = kindpred[:, kb, :] == 1
        rm = kindpred[:, kb, :] == 0
        if gm.any():
            mu_given[kb] = state_store[:, kb, :, :][gm].mean(0)
        if rm.any():
            mu_rel[kb] = state_store[:, kb, :, :][rm].mean(0)
    atlas_given = np.full((n, K_B), np.nan, np.float32)
    atlas_rel = np.full((n, K_B), np.nan, np.float32)
    for kb in range(K_B):
        for i in range(n):
            gi = np.where(kindpred[i, kb] == 1)[0]
            rj = np.where(kindpred[i, kb] == 0)[0]
            if len(gi) and not np.isnan(mu_given[kb]).any():
                atlas_given[i, kb] = float(_cos(state_store[i, kb, gi, :], mu_given[kb][None]).mean())
            if len(rj) and not np.isnan(mu_rel[kb]).any():
                atlas_rel[i, kb] = float(_cos(state_store[i, kb, rj, :], mu_rel[kb][None]).mean())

    nlc = np.full((n,), np.nan, np.float32)
    for i in range(n):
        ri = row_info[i]
        parse_last = ri["parses"].get(last_kb, [])
        try:
            nlc[i] = _nlc_cert(ri["text"], parse_last, ri["q"], last_asg[i])
        except Exception as e:
            print(f"[adaptive-stop] nl_certifier row {i} failed: {e}", flush=True)
    nlc_bykb = np.repeat(nlc[:, None], K_B, axis=1)   # row-level, constant across breaths (per the brief)

    rack_path = _rack_sidecar_path(args.tag)
    dry = np.full((n, K_B), np.nan, np.float32)
    if rack_path is not None:
        rz = np.load(rack_path, allow_pickle=True)
        rrows = {int(r): k for k, r in enumerate(rz["rows"])}
        for i in range(n):
            if i in rrows:
                d = float(rz["dry"][rrows[i]].sum())
                dry[i, last_kb] = d   # the sidecar holds only the LAST consult's committed flags (not per-breath)
        print(f"[adaptive-stop] rack sidecar found at {rack_path} -- dry counts broadcast to breath {last_kb} only "
              f"(the rack has no per-breath record; everything earlier is NaN).", flush=True)
    else:
        print(f"[adaptive-stop] no rack sidecar for tag={args.tag} -- dry column is NaN throughout "
              f"(expected unless this body runs ALG_RACK).", flush=True)

    out_path = args.out or f".cache/perceiver_stream_{args.tag}.npz"
    np.savez(out_path,
             rows=np.arange(n, dtype=np.int32),
             key=np.array([row_info[i]["key"] if row_info[i]["key"] is not None else -1 for i in range(n)], np.int64),
             q=np.array([row_info[i]["q"] for i in range(n)], np.int32),
             texts=np.array([row_info[i]["text"] for i in range(n)], dtype=object),
             solver_status=solver_status, stop_kb=stop_kb_arr,
             last_label=np.array(last_label, dtype="<U8"), adaptive_label=np.array(adaptive_label, dtype="<U8"),
             ent_given=ent_given, ent_all=ent_all,
             atlas_given=atlas_given, atlas_rel=atlas_rel,
             nl_certifier=nlc_bykb, dry=dry)
    print(f"[adaptive-stop] wrote {out_path}", flush=True)

    # ---------------- the compact table: means per breath, split by stopped-early vs never ----------------
    early = stop_kb_arr >= 0
    never = ~early
    print("-" * 100)
    print(f"THE PERCEIVER v0 STREAM -- means per breath (stopped early: {int(early.sum())} rows; never: {int(never.sum())} rows)")
    hdr = f"  {'breath':>6} {'grp':>6} {'ent_given':>10} {'ent_all':>9} {'atlas_giv':>10} {'atlas_rel':>10} {'nl_cert':>8} {'dry':>7} {'n':>5}"
    print(hdr)
    for kb in range(K_B):
        for name, sel in (("early", early), ("never", never)):
            if sel.sum() == 0:
                continue

            def mn(a):
                v = a[sel, kb]
                v = v[np.isfinite(v)]
                return float(v.mean()) if len(v) else float("nan")
            print(f"  {kb:6d} {name:>6} {mn(ent_given):10.3f} {mn(ent_all):9.3f} {mn(atlas_given):10.3f} "
                  f"{mn(atlas_rel):10.3f} {mn(nlc_bykb):8.3f} {mn(dry):7.2f} {int(sel.sum()):5d}")
    print(f"[adaptive-stop] done ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
