"""rtb_lib.py -- THE REPLAY-TRAINED BREATH (RTB), shared plumbing (2026-10-11, worktree
mycelium-wt28, branch rtb; ledger 2026-10-11 17:21 registration + docs/replay_gym.md's three
gaps). Used by scripts/rtb_buffer.py (gap a, the buffer writer), scripts/rtb_identity_check.py
and scripts/rtb_grad_probe.py (the gate), and scripts/rtb_exposure_read.py (gap c's read, part C
of THE BUILD). Not imported by phase1_algebra_head.py itself -- this is read/write-side tooling
around the two small core-file edits (the multi-point ALG_SNAP_AT extension and replay()'s new
kb_stop param), never a third place those mechanisms are reimplemented.

THE FAMILY SCOPING NOTE (load-bearing for the whole RTB build, stated once here): the chassis
RTB_241 continues (PMS8_241 -> K3_241 -> PK_241's own lineage, SURF8 + the K3/PK FAM string,
ledger 2026-10-11 16:48) sets ALG_ALT2=1 and leaves ALG_ALT3 UNSET. ALG_ALT2's fact_buf
(alt2_fact_buf) is a SELF-CONSULT derived from the body's OWN pass-0 decode (read time: see
scripts/loop_val.py's read(), lines ~325-375) or from a one-time GOLD-keyed precompute at
mask-prep (train time: do_train's FACTS array) -- either way it is captured generically inside
ctx['fact_buf'] by _snap_dump (THE REPLAY GYM's own §4: "this is a feature, not an oversight").
So for THIS family, THE REPLAY-TRAINED BREATH's gap (c) -- "the offline feedback pass" -- is
MOOT: there is no live, checkpoint-dependent, per-step consult to move out of the training step,
because this family never runs one. Gap (c) stays real and unbuilt for a future RTB arm on an
ALG_ALT3 family (RK_241 and kin), where facts3/facts5 DO depend on the current checkpoint's own
decode and WOULD need a refresh-tied offline map-reduce pass; `_build_consult_kwargs` below
builds ONLY the ALT2 (consults=False) form for exactly this reason -- extending it to ALT3 is
the natural next step for that future arm, not built here.
"""
import hashlib
import json
import os

import numpy as np


def build_consult_kwargs(H, p, ts, tk, se, vs, sl):
    """THE PASS-0 / SLOT-MASK / FACT_BUF TRIO, verbatim in spirit from replay_gate_check.py's
    `_final_pass_kwargs` (consults=False branch) -- the exact host-side recipe scripts/
    loop_val.py's read() and do_train's own _quick_val use for an ALT2 (no ALT3) family: an
    open pass-0 forward to get a slot_mask, then alt2_fact_buf from that SAME pass's own decode.
    Returns a dict of kwargs (slot_mask, fact_buf) to pass straight into forward()/the first
    breath of a fresh (on-policy) rollout. Not used for a SNAPSHOT-resumed call -- a snapshot's
    own ctx['fact_buf'] already carries this; see docs/replay_gym.md §4.
    """
    from tinygrad import Tensor, dtypes
    o0 = H.forward(p, ts, tk, se)
    onp0 = {k: o0[k].realize().numpy() for k in ("fat", "args", "res")}
    mk = H.build_slot_masks(onp0, se.numpy().astype(np.int32))
    slot_mask = Tensor(mk, dtype=dtypes.float)
    _ka = ("pres", "ftype", "op", "dig") + (("dup",) if "dup" in o0 else ())
    _oa = {**onp0, **{k: o0[k].realize().numpy() for k in _ka}}
    _nv = np.array([vs[int(i)].get("n_vars", H.K_VARS) for i in sl])
    _ma = np.array([vs[int(i)].get("m", 0) for i in sl])
    fb = H.alt2_fact_buf(_oa, se.numpy().astype(np.int32), _nv, _ma)
    return dict(slot_mask=slot_mask, fact_buf=Tensor(fb, dtype=dtypes.float))


def gold_feed(H, gold, idx):
    """A direct port of do_train's own `feed = {...}` gold-dict construction (scripts/
    phase1_algebra_head.py, do_train, ~line 12382 at the time of this port) for a batch of row
    indices `idx` into the ALREADY-LOADED `gold` array dict (load_alg's own return) -- used
    where a snapshot-resumed replay step needs the SAME bg feed the normal fixed-buffer path
    builds, without constructing do_train's whole buffer apparatus (`bg`/`_fd`/the fixed device
    tensors) for an eager (non-JIT) training step. Conditions are read DIRECTLY off os.environ
    (the same flags the bg-buffer tuple and this feed dict both gate on), not off a `bg` dict's
    membership, since no `bg` object exists in this calling context.

    SCOPE (stated, mirrors rtb_lib's module docstring): covers every key the K3_241/PK_241/
    RTB_241 lineage's own FAM string actually feeds (confirmed against that FAM: ALG_FTYPES=9,
    ALG_WIDE=1, ALG_BINDBUS=7 -- bind_ids only, bindvec's "0<x<5" window excludes 7; ALG_CARIC/
    ALG_REF/ALG_DIAL/ALG_VALATT/ALG_POSCH/ALG_OPCOUNT/ALG_CONSUME/OPATT all unset on this
    family). A future RTB arm on a DIFFERENT family that arms one of those doors would need this
    function extended the same way do_train's own feed dict would need a new clause -- not a
    silent gap today (every key this specific lineage trains against is covered), but not a
    universal port of the entire conditional block either.
    """
    ftypes = int(os.environ.get("ALG_FTYPES", "4"))
    feed = {"presence": gold["presence"][idx], "is_lit_f": gold["is_lit"][idx],
            "args": gold["args"][idx], "fspan": gold["fspan"][idx],
            "vspan": gold["vspan"][idx], "ftype": gold["ftype"][idx],
            "op": gold["op"][idx], "res": gold["res"][idx],
            "digits": gold["digits"][idx], "query": gold["query"][idx],
            "sel": gold["sel"][idx], "is_rel": gold["is_rel"][idx],
            "is_mod": gold["is_mod"][idx], "is_sel": gold["is_sel"][idx],
            "is_pct": gold["is_pct"][idx], "is_fdiv": gold["is_fdiv"][idx],
            "arg_dup": (gold["arg_dup"][idx] if "arg_dup" in gold
                        else np.zeros_like(gold["is_rel"][idx])),
            "arg_dir": (gold["arg_dir"][idx] if "arg_dir" in gold
                        else np.zeros_like(gold["is_rel"][idx])),
            "dir_mask": (gold["dir_mask"][idx] if "dir_mask" in gold
                         else np.zeros_like(gold["is_rel"][idx]))}
    if ftypes >= 9 and "is_chain" in gold:
        feed["is_chain"] = gold["is_chain"][idx]
    if ftypes >= 7 and "is_macro" in gold:
        feed["is_macro"] = gold["is_macro"][idx]
        feed["digits2"] = gold["digits2"][idx]
        feed["y"] = gold["y"][idx]
    if ftypes >= 8 and "is_frac" in gold:
        feed["is_frac"] = gold["is_frac"][idx]
    if int(os.environ.get("ALG_WIDE", "0")) and "sign" in gold:
        feed["sign"] = gold["sign"][idx]
    if int(os.environ.get("ALG_BINDBUS", "0")) >= 3 and "bind_ids" in gold:
        feed["bind_ids"] = gold["bind_ids"][idx]
    return feed


def feed_to_tensors(H, feed):
    """feed (plain numpy, from gold_feed) -> {key: Tensor} with each key's dtype taken from
    H's own per-key int/float convention (int for ids/classes, float for everything else) --
    the SAME split do_train's bg-buffer tuple encodes (dtypes.int for ftype/op/res/digits/
    query/sel/args/is_macro/digits2/y/bind_ids; dtypes.float for the rest)."""
    from tinygrad import Tensor, dtypes
    # dtypes taken VERBATIM from the bg-buffer tuple in do_train (scripts/phase1_algebra_head.py,
    # ~line 11213 at the time of this port) -- "args"/"is_macro"/"is_frac"/"is_chain"/"sign" are
    # dtypes.float (multi-hot / indicator buffers), not int, despite the name "is_macro" etc.
    int_keys = {"ftype", "op", "res", "digits", "query", "sel", "digits2", "y", "bind_ids"}
    out = {}
    for k, v in feed.items():
        dt = dtypes.int if k in int_keys else dtypes.float
        npdt = np.int32 if dt == dtypes.int else np.float32
        out[k] = Tensor(np.ascontiguousarray(v, dtype=npdt), dtype=dt)
    return out


def ckpt_hash(path):
    """sha256 of a checkpoint file, truncated to 16 hex chars -- THE SHA-FENCE idiom (load_alg's
    mix_sha16), so a buffer manifest can assert staleness honestly (the ckpt that WROTE a
    snapshot vs the ckpt a training step is currently running under)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def manifest_path(buf_dir):
    return os.path.join(buf_dir, "index.json")


def load_manifest(buf_dir):
    with open(manifest_path(buf_dir)) as f:
        return json.load(f)


def save_manifest(buf_dir, manifest):
    os.makedirs(buf_dir, exist_ok=True)
    tmp = manifest_path(buf_dir) + ".tmp"
    with open(tmp, "w") as f:
        json.dump(manifest, f)
    os.replace(tmp, manifest_path(buf_dir))
