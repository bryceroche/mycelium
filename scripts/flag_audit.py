"""flag_audit.py — THE SHREDDED-FLAG AUDIT (2026-09-08). A read-only,
zero-training measurement of the slot mask's OPENNESS and what it costs.

THE QUESTION. The slot mask (`build_slot_masks`) is built from the model's
OWN breath-0 outputs: two slots share a lane if they land in the same
sentence OR share a variable. On MINT text the sentences are short and the
argument sets are sharp, so the mask is a few tight cliques. On WILD text
— longer sentences, blurrier pointers — the same rule opens far more
lanes. A mask that opens everything is a flag shredded to threads: the
slot mixer's attention has nothing left to exclude, and every slot reads
every other slot's evidence. This script measures BOTH halves of that
claim on banked artifacts, and prints its two PINNED BARS.

WHAT IS MEASURED, per item, on the SAME pass-0 (open) forward:
  * DEGREE      the mean out-degree of the 24x24 baseline slot mask
                (mean row sum) and the fraction of the mask that is open
                (degree / 24 — the same number in the other unit);
  * CORRECTNESS the slot-level fac-exact of that item, through
                step_engine_read.score_batch — the AUTHORITY, called per
                item on its own row (the meter-divergence law: a check
                calls its organ, it does not re-implement it).
Then, on WILD, items are split into DEGREE TERCILES and the slot-level
fac-exact (ok/tot, pooled within the tercile) is read per tercile.

BARS (PINNED BEFORE MEASUREMENT — read them as written, never bent after):
  BAR 1  wild mean degree exceeds mint mean degree by >= 1.50 slots
  BAR 2  the TOP-degree tercile's wild slot fac-exact trails the BOTTOM
         tercile's by >= 0.03
Bar 1 without bar 2 says the mask opens on wild but openness is not what
costs; bar 2 without bar 1 says openness costs but wild is not where it
happens. Both together is the shredded flag.

GOODHART NOTE: every number here is a READ. Nothing printed may enter a
loss or a training-data selection criterion (the diagnostic register).
It is also a CORRELATIONAL read on the deployed checkpoint: degree is
built from the model's own pass-0 outputs, so a hard item can raise its
own degree. The tercile split is a description, not a causal claim.

Envs (collider_read.py's idiom): FA_CKPT (default
.cache/sharp_fedon242.safetensors); FA_N (0 = all rows, per fixture);
FA_B (batch, default 32); FA_FIX (default "wild,mint").
DEV comes from the caller via setdefault("PCI+AMD"). CPU test:
  DEV=CPU FA_N=8 FA_B=8 .venv/bin/python3 scripts/flag_audit.py
Outputs: .cache/flag_audit_<fixture>.npz per fixture (per-item arrays).
"""
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "scripts"))
os.chdir(_ROOT)

_FIXTURES = {"wild": (".cache/wild_admitted_holdout.jsonl", "wildhold"),
             "mint": (".cache/algebra_nl_test.jsonl", "test23")}
FIX = [f.strip() for f in os.environ.get("FA_FIX", "wild,mint").split(",")
       if f.strip()]
assert all(f in _FIXTURES for f in FIX), f"FA_FIX must name {sorted(_FIXTURES)}"

os.environ.setdefault("DEV", "PCI+AMD")
# Champion env stack — collider_read.py's ENV verbatim (the trained env, or
# the ckpt keys will not match build_params), MINUS the fixture keys: this
# audit walks BOTH fixtures in ONE process (the AM driver is SINGLE-PROCESS)
# by rebinding the head's ALG_TEST / TEST_NAME globals between passes.
ENV = {"ALG2": "1", "ALG_FTYPES": "9", "ALG_DUP": "1", "ALG_HW": "512",
       "ALG_WIDE": "1", "ALG_BREATH": "7", "ALG_NOTEBOOK": "1",
       "ALG_SIXWAVE": "1", "NB_PERSLOT": "1", "ALG_BINDBUS": "7",
       "ALG_BIND_D": "512", "BIND_CODES": ".cache/bindbus_codes512.npz",
       "ALG_BUSGARAGE": "2", "ALG_SHELF_CIRCLE": "2", "ALG_ALTMASK": "1",
       "ALG_ALT21": "1", "ALG_ALT2": "1", "ALG_MASKHEAD": "1",
       "ALG_FED": "1"}
os.environ.update(ENV)
os.environ.pop("SC_EVAL", None)
for _k in ("ALG_PC_MIX", "ALG_PC_LIVE", "ALG_POLAR",
           "ALG_POLAR_D", "ALG_POLAR_EM"):
    os.environ.pop(_k, None)          # a READ carries no research door

import numpy as np                                        # noqa: E402

CKPT = os.environ.get("FA_CKPT", ".cache/sharp_fedon242.safetensors")
FA_N = int(os.environ.get("FA_N", "0"))
FA_B = int(os.environ.get("FA_B", "32"))

BAR1_DEG = 1.50          # wild mean degree - mint mean degree
BAR2_GAP = 0.03          # bottom-tercile fac-exact - top-tercile fac-exact


def terciles(x):
    """Rank-based tercile labels 0/1/2 (ties broken by index, so the three
    groups are as equal as n allows). Returns (labels, cut_lo, cut_hi)."""
    n = len(x)
    order = np.argsort(x, kind="stable")
    lab = np.empty(n, np.int64)
    b = n // 3
    lab[order[:b]] = 0
    lab[order[b:n - b]] = 1
    lab[order[n - b:]] = 2
    lo = float(x[order[b - 1]]) if b else float("nan")
    hi = float(x[order[n - b]]) if b else float("nan")
    return lab, lo, hi


def run_fixture(name, H, p, Tensor, dtypes, score_batch):
    """One fixture, one pass-0 forward per batch. Rebinds the head's
    fixture globals (single process, the AM driver's law) and asserts the
    rebind took before reading a single row."""
    path, sname = _FIXTURES[name]
    H.ALG_TEST = path
    H.TEST_NAME = sname
    vs, vst, vtk, vg, vse = H.load_alg("test")
    assert len(vs) == vst.shape[0] == vg["presence"].shape[0], \
        f"{name}: samples/states/gold desync"
    n_all = len(vs)
    n = min(FA_N, n_all) if FA_N > 0 else n_all
    # ORDERING ASSERTION (collider_read's): the samples list is the jsonl in
    # file order and the gold arrays are row-aligned to it.
    for i in range(n):
        assert int(vg["query"][i]) == int(vs[i]["query_var"]), \
            f"{name} row {i}: fixture/gold desync"
    print(f"[flag] fixture={sname} ({path}) rows={n}/{n_all} ckpt={CKPT}",
          flush=True)

    deg = np.zeros(n, np.float64)      # mean out-degree over the 24 slots
    opn = np.zeros(n, np.float64)      # fraction of the 24x24 mask open
    ok = np.zeros(n, np.int64)         # slot-level fac-exact numerator
    tot = np.zeros(n, np.int64)        # gold-present slots
    for s0 in range(0, n, FA_B):
        sl = np.arange(s0, min(s0 + FA_B, n))
        pad = FA_B - len(sl)
        sl_p = np.concatenate([sl, sl[:1].repeat(pad)]) if pad else sl
        ts = Tensor(vst[sl_p].astype(np.float32), dtype=dtypes.float)
        tk = Tensor(vtk[sl_p].astype(np.float32), dtype=dtypes.float)
        se = Tensor(vse[sl_p].astype(np.int32), dtype=dtypes.int)
        o = H.forward(p, ts, tk, se)                     # PASS 0: open
        kset = ["pres", "ftype", "op", "args", "res", "dig", "fat"]
        kset += [k for k in ("dup", "sgn", "dargs") if k in o]
        onp = {k: o[k].realize().numpy() for k in kset}
        # THE BASELINE MASK — the head's OWN organ on its OWN pass-0 read
        mk = H.build_slot_masks(onp, vse[sl_p].astype(np.int32))
        assert mk.shape[1:] == (H.L_FAC, H.L_FAC), mk.shape
        for bi, i in enumerate(sl):
            i = int(i)
            deg[i] = float(mk[bi].sum(-1).mean())
            opn[i] = float(mk[bi].mean())
            row = {k: v[bi:bi + 1] for k, v in onp.items()}
            a, b = score_batch(row, vg, np.array([i]))   # THE AUTHORITY
            ok[i] = int(a)
            tot[i] = int(b)
    assert np.allclose(opn, deg / H.L_FAC), "degree/open-fraction disagree"
    out = f".cache/flag_audit_{sname}.npz"
    np.savez(out, degree=deg, open_frac=opn, ok=ok, tot=tot,
             idx=np.arange(n), ckpt=np.array(CKPT), fixture=np.array(sname),
             path=np.array(path), l_fac=np.int64(H.L_FAC))
    fx = float(ok.sum()) / max(int(tot.sum()), 1)
    print(f"[flag]   {sname}: mean degree {deg.mean():.3f}/{H.L_FAC} "
          f"(sd {deg.std():.3f}, min {deg.min():.2f}, max {deg.max():.2f}), "
          f"mask open {opn.mean() * 100:.1f}%; slot fac-exact "
          f"{int(ok.sum())}/{int(tot.sum())}={fx:.4f} -> {out}", flush=True)
    return {"name": sname, "deg": deg, "open": opn, "ok": ok, "tot": tot,
            "facx": fx, "out": out}


def main():
    from tinygrad import Tensor, dtypes
    from tinygrad.nn.state import safe_load
    import phase1_algebra_head as H
    from step_engine_read import score_batch

    p = H.build_params(0)
    sd = safe_load(CKPT)
    assert set(sd.keys()) == set(p.keys()), \
        (sorted(set(sd) - set(p))[:4], sorted(set(p) - set(sd))[:4])
    for k in p:
        p[k].assign(sd[k].to(p[k].device).cast(p[k].dtype)).realize()

    R = {f: run_fixture(f, H, p, Tensor, dtypes, score_batch) for f in FIX}

    # ---- BAR 1: wild mean degree exceeds mint mean degree -----------------
    v1 = "N/A"
    d_gap = None
    if "wild" in R and "mint" in R:
        d_gap = float(R["wild"]["deg"].mean() - R["mint"]["deg"].mean())
        v1 = "PASS" if d_gap >= BAR1_DEG else "FAIL"
        print(f"[flag] BAR 1 wild-minus-mint mean degree: {d_gap:+.3f} slots "
              f"(bar >= {BAR1_DEG:.2f}) -> {v1}", flush=True)
    else:
        print(f"[flag] BAR 1: N/A (needs both fixtures; FA_FIX={FIX})",
              flush=True)

    # ---- BAR 2: wild fac-exact by DEGREE TERCILE --------------------------
    v2 = "N/A"
    t_gap = None
    if "wild" in R:
        w = R["wild"]
        lab, lo, hi = terciles(w["deg"])
        print("[flag] wild slot fac-exact by degree tercile "
              f"(cuts {lo:.2f} / {hi:.2f} of {len(lab)} items):", flush=True)
        fx = {}
        for k, nm in ((0, "bottom (tightest mask)"), (1, "middle"),
                      (2, "top (most open)")):
            m = lab == k
            a, b = int(w["ok"][m].sum()), int(w["tot"][m].sum())
            fx[k] = a / max(b, 1)
            print(f"[flag]   {nm:<24} items={int(m.sum()):<4} "
                  f"degree={w['deg'][m].mean():6.3f}  fac-exact={a}/{b}="
                  f"{fx[k]:.4f}", flush=True)
        t_gap = fx[0] - fx[2]
        v2 = "PASS" if t_gap >= BAR2_GAP else "FAIL"
        print(f"[flag] BAR 2 bottom-minus-top tercile fac-exact: "
              f"{t_gap:+.4f} (bar >= {BAR2_GAP:.2f}) -> {v2}", flush=True)
    else:
        print("[flag] BAR 2: N/A (needs the wild fixture)", flush=True)

    parts = [f"{R[f]['name']} degree={R[f]['deg'].mean():.3f} "
             f"open={R[f]['open'].mean() * 100:.1f}% "
             f"fac-exact={R[f]['facx']:.4f}" for f in FIX]
    print(f"[flag] {' | '.join(parts)} | BAR1 "
          f"{'n/a' if d_gap is None else f'{d_gap:+.3f}'} [{v1}] | BAR2 "
          f"{'n/a' if t_gap is None else f'{t_gap:+.4f}'} [{v2}] -> "
          + ", ".join(R[f]["out"] for f in FIX), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
