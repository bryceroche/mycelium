"""climb_read.py -- THE GOVERNOR CLIMB's read (2026-10-14; registered 2026-10-13 14:50, bars amended
2026-10-14 ~17:40). Zero-GPU-capable (DEV=CPU works; it runs only the token side).

WHAT IT READS: a checkpoint's sorting room (ALG_SORT=4) on a fixture, through the SAME token side the
trainer runs (phase1_algebra_head._waist_prefix: trunk -> waist (+HUD/FED/T1) -> the sorting room), with
the climb's tap on head 0 of blocks 0,1,2. The model's OWN links (argmax of head-0 attention; no gold
anywhere in the forward path):
  PER HOP: argmax == the gold parent, over the deduplicated supervised queries --
      block 0: verb -> owner; block 1: noun -> verb; block 2: numeral -> noun.
  CHAINED OWNER: per numeral with a gold owner, follow numeral -> noun' -> verb' -> owner' through the
      model's own argmaxes (block 2, then block 1 at noun', then block 0 at verb'); right iff owner' is
      the owner's head piece (and, a lenient column, iff owner' lies inside the owner's full piece span);
      reported separately for NAMED and PRONOUN owners (coreference is out of scope: a pronoun owner is
      graded against the pronoun token itself).
  THE FLOOR: run the same read on an untrained body (a step-0 / ALG_CLIMB=0 checkpoint, or the warm body
      with the sorting room at its seeded fresh init: CR_INIT_MISSING=1) -- the depth probe's untrained
      head link at the tap read 0.57.
BARS it serves (the ledger, 2026-10-14 ~17:40): >= 0.85 per hop on wild from the model's own predictions;
floor reference 0.57.

THE MEASURE-ONLY FENCE: the wild targets (.cache/climb_targets_wild_DIAGNOSTIC_ONLY.npz) may be opened
by THIS read alone (climb_build_targets(..., allow_diagnostic=True)); every trainer refuses them.

ENV: the body's family env (ALG_SORT=4, ALG_HW, ... exactly as trained; ALG_CLIMB itself is not needed);
  ALG_TEST / ALG_TEST_NAME  the fixture (wild: .cache/wild_admitted_holdout.jsonl / wildhold);
  CR_CKPTS      comma list of label=path (e.g. floor=.cache/x_s0.safetensors,climb=.cache/x.safetensors);
                every param must be in the checkpoint (hard error) unless CR_INIT_MISSING=1, which keeps
                missing params at build_params(SEED)'s fresh init and NAMES them;
  CR_TARGETS    the fixture's climb targets (default: the wild DIAGNOSTIC file when ALG_TEST_NAME starts
                with 'wildhold', else required);
  CR_ROWS       optional 'lo:hi' slice (smoke); CR_BATCH (default 16); CR_OUT optional json path.
"""
import json
import os
import sys

sys.path.insert(0, "."); sys.path.insert(0, "scripts")
import numpy as np  # noqa: E402

import phase1_algebra_head as H  # noqa: E402
from tinygrad import Tensor, dtypes  # noqa: E402
from tinygrad.nn.state import safe_load  # noqa: E402

assert H.ALG_SORT == 4, "climb_read reads the sorting room's blocks 0..2: the body's env must carry ALG_SORT=4"
assert not H.ALG_SORT_HG, "the hourglass's blocks 1,2 are not token->token: no climb read"


def _targets_path():
    tp = os.environ.get("CR_TARGETS", "")
    if not tp:
        assert H.TEST_NAME.startswith("wildhold"), "CR_TARGETS is required for a non-wild fixture"
        tp = ".cache/climb_targets_wild_DIAGNOSTIC_ONLY.npz"
    return tp


def _load_body(path, seed):
    p = H.build_params(seed)
    sd = safe_load(path)
    miss = [k for k in p if k not in sd]
    if miss and not int(os.environ.get("CR_INIT_MISSING", "0")):
        raise KeyError(f"{path}: {len(miss)} params missing (eval loads hard-error; CR_INIT_MISSING=1 keeps "
                       f"them at fresh init, named): {miss[:8]}")
    for k in p:
        if k in sd:
            p[k].assign(sd[k].to(p[k].device).cast(p[k].dtype)).realize()
    return p, miss


def read_body(p, samples, states, tokmask, sent, rows, batch):
    """-> dict h -> (N, T) int argmax of head-0 attention per query position (rows outside `rows` = -1)."""
    N, T = tokmask.shape
    am = {h: np.full((N, T), -1, np.int32) for h in range(len(H.CLIMB_HOPS))}
    Tensor.training = False
    for b0 in range(0, len(rows), batch):
        sl = rows[b0:b0 + batch]
        ts = Tensor(np.ascontiguousarray(states[sl]).astype(np.float32), dtype=dtypes.float)
        tk = Tensor(tokmask[sl].astype(np.float32), dtype=dtypes.float)
        se = Tensor(sent[sl].astype(np.int32), dtype=dtypes.int)
        hud = None
        if H.ALG_HUD:
            hud = Tensor(H.hud_build_array([samples[int(i)] for i in sl], tokmask[sl], sent[sl], T), dtype=dtypes.int)
        taps = []
        H._waist_prefix(p, ts, tk, se, hud=hud, climb_taps=taps)
        assert len(taps) == len(H.CLIMB_HOPS), f"{len(taps)} taps: the sorting room did not run"
        for h, sc in enumerate(taps):
            am[h][sl] = sc.argmax(-1).numpy().astype(np.int32)
    return am


def grade(am, tgt, z, rows):
    out = {}
    for h, (qk, tk) in enumerate(H.CLIMB_HOPS):
        t = tgt[rows, h, :]
        a = am[h][rows]
        m = t >= 0
        n = int(m.sum())
        # the SELF-LINK split: climb_targets.py's noun link falls back to tok.head when no NOUN/PROPN
        # ancestor exists, which lands on the verb ("x is 5") -- noun == verb, the hop's target is the
        # query itself (37 % of block 1's pool queries); graded both ways
        qpos = np.broadcast_to(np.arange(t.shape[1]), t.shape)
        ms = m & (t == qpos); mn = m & (t != qpos)
        out[f"hop{h}_{qk}->{tk}"] = {"n": n, "acc": float((a[m] == t[m]).mean()) if n else float("nan"),
                                     "n_self": int(ms.sum()),
                                     "acc_nonself": float((a[mn] == t[mn]).mean()) if mn.any() else float("nan")}
    # the chained owner, through the model's own argmaxes
    res = {"named": [0, 0, 0], "pronoun": [0, 0, 0]}   # [n, head-piece right, in-span right]
    for i in rows:
        for k in range(int(z["n_num"][i])):
            if z["numeral_pos"][i, k] < 0 or not z["owner_valid"][i, k]:
                continue
            q = int(z["numeral_pos"][i, k])
            n_ = int(am[2][i, q]); v_ = int(am[1][i, n_]); o_ = int(am[0][i, v_])
            key = "pronoun" if z["owner_pronoun"][i, k] else "named"
            lo, hi = (int(x) for x in z["owner_span"][i, k])
            res[key][0] += 1
            res[key][1] += int(o_ == int(z["owner_pos"][i, k]))
            res[key][2] += int(lo <= o_ < hi)
    for key, (n, r, rs) in res.items():
        out[f"chain_owner_{key}"] = {"n": n, "acc": r / n if n else float("nan"),
                                     "acc_in_span": rs / n if n else float("nan")}
    n_all = res["named"][0] + res["pronoun"][0]
    out["chain_owner_all"] = {"n": n_all, "acc": (res["named"][1] + res["pronoun"][1]) / n_all if n_all else float("nan")}
    return out


def main():
    samples, states, tokmask, gold, sent = H.load_alg("test")
    tp = _targets_path()
    tgt = H.climb_build_targets(tp, samples, tokmask, H.ALG_TEST, placebo=False, verbose=True,
                                allow_diagnostic=True)
    z = np.load(tp)
    N = len(samples)
    rows = np.arange(N)
    if os.environ.get("CR_ROWS"):
        lo, hi = (int(x) for x in os.environ["CR_ROWS"].split(":"))
        rows = rows[lo:hi]
    batch = int(os.environ.get("CR_BATCH", "16"))
    seed = int(os.environ.get("SEED", "0"))
    report = {"fixture": H.ALG_TEST, "split": H.TEST_NAME, "targets": tp, "rows": [int(rows[0]), int(rows[-1]) + 1],
              "bars": {"per_hop_wild": 0.85, "floor_ref_untrained_head_link": 0.57}, "bodies": {}}
    for spec in os.environ["CR_CKPTS"].split(","):
        label, path = spec.split("=", 1)
        p, miss = _load_body(path, seed)
        if miss:
            print(f"[climb-read] {label}: {len(miss)} params at FRESH INIT (CR_INIT_MISSING=1, seed {seed}): "
                  f"{sorted(set(k.split('_')[0] for k in miss))}", flush=True)
        am = read_body(p, samples, states, tokmask, sent, rows, batch)
        g = grade(am, tgt, z, rows)
        report["bodies"][label] = {"ckpt": path, "fresh_init": miss, **g}
        line = " ".join(f"{k}={v['acc']:.4f}(n={v['n']})" for k, v in g.items())
        print(f"[climb-read] {label} ({path}) on {H.TEST_NAME} rows {rows[0]}:{rows[-1] + 1}: {line}", flush=True)
        for k, v in g.items():
            if k.startswith("hop"):
                print(f"[climb-read]   {label} {k}: {v['acc']:.4f} (n={v['n']}; non-self {v['acc_nonself']:.4f}, "
                      f"self-links {v['n_self']}) bar 0.85 {'MET' if v['acc'] >= 0.85 else 'not met'}; floor ref 0.57",
                      flush=True)
            elif k.startswith("chain_owner_") and "acc_in_span" in v:
                print(f"[climb-read]   {label} {k}: head-piece {v['acc']:.4f} / in-span {v['acc_in_span']:.4f} "
                      f"(n={v['n']})", flush=True)
    if os.environ.get("CR_OUT"):
        assert not os.path.exists(os.environ["CR_OUT"]), f"{os.environ['CR_OUT']} exists: never overwrite a banked read"
        with open(os.environ["CR_OUT"], "w") as f:
            json.dump(report, f, indent=1)
        print(f"[climb-read] report -> {os.environ['CR_OUT']}", flush=True)


if __name__ == "__main__":
    main()
