"""perceiver.py — THE PERCEIVER (2026-09-14, word given): the organ that
WATCHES the two loops and pulls levers between cycles. Standing law: the
perceiver is retired as core and sanctioned as MONITOR / SEGMENTER
(CLAUDE.md s6) — this is that role, with levers.

THE READ SIDE — LoopHealth: one record per row per breath, joined from
meters the loops already expose (nothing new is computed in the graph):
  settle        ||s_k - s_{k-1}|| / ||s_{k-1}||, mean over slots
  z_slot/z_tok  the atlas z-radius to the nearest kind on the SLOT chart
                (the commitment) and the TOKEN chart (the reading)
  gap           z_tok - z_slot: THE CHART GAP — lost on the token chart
                but placed on the slot chart = a READING failure; the
                reverse = a MATH failure (which loop to pull levers on)
  margin_*      the least-confident present slot's margin per field
                (res / op / args; scripts' _slot_margins)
  claim_conflict fraction of real tokens claimed (> tau) by > 1 present slot
  census        each organ's injection RMS over its band's state, per row
  wheel_status / wheel_core   the solver's verdict on the final parse
  slot_ok / row_ok            gold-scored (RESEARCH reads only; absent at
                              inference by law)

THE FENCES (mycelium/diagnostic_register.py): every meter is a METER,
never a TARGET — no gradient flows through a read, the head is never
trained through the perceiver; if a policy is ever learned its reward is
the ANSWER KEY at the cycle level and the meters are inputs only.
"never"-tier meters (settle, self-loops, fat overlap) may NOT drive data
selection (the annotation queue reads unanimity / vote entropy, a
"loss-never" meter with declared selection use).

THE LEVERS — Perceiver.decide (v0 = a RULE TABLE, ledgered, deterministic,
no training): every cross-medium lever is a Certificate the bridge
projects (mycelium/loop_bridge); the scalars are the cycle count, the
views to spend, and ABSTAIN. Thresholds live in RULES and are read from
the ledger's table, never tuned on the meters they read.
"""
import numpy as np

METERS = ("settle", "z_slot", "z_tok", "gap", "margin_res", "margin_op",
          "margin_args", "claim_conflict")


def auroc(score, label):
    """Rank AUROC of label==1 having the higher score (ties averaged);
    NaN when a class is empty. scripts/happy_family_read.py's formula."""
    score = np.asarray(score, np.float64); label = np.asarray(label).astype(int)
    order = np.argsort(score); ranks = np.empty(len(score)); s = score[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1; i = j + 1
    pos = label == 1; n1 = pos.sum(); n0 = (~pos).sum()
    return float((ranks[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)) if n1 and n0 else float("nan")


class LoopHealth:
    """The record: meters (K, N) per breath per row; census {organ: (K, N)};
    wheel (N,); gold (N,) / (N, L). Built batch by batch with add()."""

    def __init__(self, K):
        self.K = int(K)
        self.m = {k: [] for k in METERS}
        self.census = {}
        self.cls_slot, self.cls_tok = [], []
        self.wheel_status, self.wheel_core = [], []
        self.row_ok, self.slot_ok, self.present = [], [], []
        self.ids = []

    def add(self, ids, meters, census=None, cls_slot=None, cls_tok=None,
            wheel=None, row_ok=None, slot_ok=None, present=None):
        """meters: {name: (K, b)}; census: {organ: (K, b)}; wheel: [(status, core)]."""
        self.ids.extend(int(i) for i in ids)
        for k in METERS:
            self.m[k].append(np.asarray(meters[k], np.float64))
        for o, arr in (census or {}).items():
            self.census.setdefault(o, []).append(np.asarray(arr, np.float64))
        if cls_slot is not None: self.cls_slot.append(np.asarray(cls_slot))
        if cls_tok is not None: self.cls_tok.append(np.asarray(cls_tok))
        if wheel is not None:
            self.wheel_status.extend(st for st, _ in wheel)
            self.wheel_core.extend(len(c) for _, c in wheel)
        if row_ok is not None: self.row_ok.append(np.asarray(row_ok, bool))
        if slot_ok is not None: self.slot_ok.append(np.asarray(slot_ok, bool))
        if present is not None: self.present.append(np.asarray(present, bool))

    def meter(self, name):
        return np.concatenate(self.m[name], axis=1)          # (K, N)

    def organ(self, name):
        return np.concatenate(self.census[name], axis=1)     # (K, N)

    @property
    def n(self): return len(self.ids)

    def rows_ok(self):
        return np.concatenate(self.row_ok) if self.row_ok else None

    def report(self, bar=0.65):
        """Per breath: the medians and, when gold is present, the AUROC of
        row-correct vs each meter (lower = healthier, so the score is
        -meter). A meter is a COMPASS at AUROC >= bar (the happy-family
        bar). Returns {(meter, k): auroc} and prints the table."""
        ok = self.rows_ok()
        out = {}
        print(f"[perceiver] N={self.n} rows, K={self.K} breaths" + (f", rows correct {ok.mean():.3f}" if ok is not None else ""))
        hdr = "meter        " + " ".join(f"   b{k}   " for k in range(self.K))
        print(hdr)
        for name in METERS:
            M = self.meter(name)
            med = " ".join(f"{np.nanmedian(M[k]):8.3f}" for k in range(self.K))
            print(f"  {name:12s} median {med}")
            if ok is not None:
                aus = []
                for k in range(self.K):
                    v = M[k]; msk = ~np.isnan(v)
                    a = auroc(-v[msk], ok[msk]) if msk.sum() >= 10 else float("nan")
                    out[(name, k)] = a; aus.append(a)
                flag = " <- COMPASS" if np.nanmax(aus) >= bar else ""
                print(f"  {'':12s} AUROC  " + " ".join(f"{a:8.3f}" for a in aus) + flag)
        if self.wheel_status and ok is not None:
            st = np.array(self.wheel_status); un = st == "unsat"
            if un.any() and (~un).any():
                print(f"  wheel: refused {un.mean():.3f} of rows; P(wrong|unsat) {(~ok[un]).mean():.3f} vs P(wrong|sat) {(~ok[~un]).mean():.3f}")
        for o in sorted(self.census):
            C = self.organ(o)
            print(f"  census {o:14s} median ratio " + " ".join(f"{np.nanmedian(C[k]):8.4f}" for k in range(self.K)))
        return out

    def save(self, path):
        d = {f"m_{k}": self.meter(k) for k in METERS}
        d.update({f"c_{o}": self.organ(o) for o in self.census})
        d["ids"] = np.array(self.ids)
        if self.row_ok: d["row_ok"] = self.rows_ok()
        if self.slot_ok: d["slot_ok"] = np.concatenate(self.slot_ok)
        if self.present: d["present"] = np.concatenate(self.present)
        if self.wheel_status:
            d["wheel_status"] = np.array(self.wheel_status); d["wheel_core"] = np.array(self.wheel_core)
        if self.cls_slot: d["cls_slot"] = np.concatenate(self.cls_slot, axis=1); d["cls_tok"] = np.concatenate(self.cls_tok, axis=1)
        np.savez(path, **d); return path


# THE RULE TABLE v0 (ledgered 2026-09-14; thresholds are PINNED here, never
# tuned on the meters they read). Reads declared per rule; the register's
# "never" tier (settle) drives inference control only — no selection.
RULES = {
    "settled": {"settle_max": 0.02, "margin_min": 1.0},   # stop early: the state stopped moving and every field is confident
    "unstable": {"gap_min": 1.0},                         # the reading is lost, the math is placed: spend views on the reading
    "conflict": {"claim_conflict_min": 0.10},             # tokens claimed twice: the claims certificate
    "views": 4,                                           # permutations to spend when unstable
    "abstain": {"margin_min": 0.0, "wheel": "unsat"},     # a refusal or a zero-margin field at the cycle's end
}


class Perceiver:
    """observe -> decide. decide(health_row) -> levers: {"stop", "extra_cycle",
    "views", "abstain", "certificates": [Certificate]} from one row's final-
    breath meters (a dict name -> float, plus optional "wheel_status" and
    "unstable_slots" (L,) / "conflict_slots" (L,) masks)."""

    def __init__(self, rules=RULES):
        self.rules = dict(rules)

    def decide(self, h):
        from mycelium.loop_bridge import Certificate
        r = self.rules; lev = {"stop": False, "extra_cycle": False, "views": 0, "abstain": False, "certificates": []}
        mmin = min(h.get("margin_res", np.inf), h.get("margin_op", np.inf), h.get("margin_args", np.inf))
        # declared read of "settle" (diagnostic_register tier "never"): inference control only, never selection
        if h.get("settle", np.inf) <= r["settled"]["settle_max"] and mmin >= r["settled"]["margin_min"]:
            lev["stop"] = True
            return lev
        if h.get("gap", 0.0) >= r["unstable"]["gap_min"]:
            lev["views"] = r["views"]; lev["extra_cycle"] = True
            if h.get("unstable_slots") is not None and np.any(h["unstable_slots"]):
                lev["certificates"].append(Certificate("locus", slots=h["unstable_slots"][None], breath=h.get("breath")))
        if h.get("claim_conflict", 0.0) >= r["conflict"]["claim_conflict_min"] and h.get("conflict_slots") is not None:
            lev["certificates"].append(Certificate("claims", slots=h["conflict_slots"][None], breath=h.get("breath")))
        if h.get("wheel_status") == r["abstain"]["wheel"] or mmin <= r["abstain"]["margin_min"]:
            lev["abstain"] = True
        return lev


if __name__ == "__main__":
    import os, sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # the repo root (mycelium.loop_bridge)
    rng = np.random.default_rng(5)
    K, N = 7, 200
    hl = LoopHealth(K)
    ok = rng.random(N) < 0.4
    # a designed compass: the gap is larger on wrong rows at the late breaths
    meters = {k: rng.random((K, N)) for k in METERS}
    meters["gap"][4:] += (~ok)[None] * 1.5
    hl.add(np.arange(N), meters, census={"tokloop": rng.random((K, N)) * 0.05}, wheel=[("unsat", [1]) if (not o and rng.random() < 0.6) else ("sat", []) for o in ok], row_ok=ok)
    assert hl.n == N and hl.meter("gap").shape == (K, N)
    tab = hl.report()
    assert tab[("gap", 6)] >= 0.65 and tab[("gap", 0)] < 0.65, tab[("gap", 6)]
    assert abs(auroc(np.arange(10), np.arange(10) >= 5) - 1.0) < 1e-9 and abs(auroc(np.zeros(10), np.arange(10) >= 5) - 0.5) < 1e-9
    import tempfile, os
    pth = hl.save(tempfile.mktemp(suffix=".npz")); z = np.load(pth); assert z["m_gap"].shape == (K, N) and z["row_ok"].sum() == ok.sum(); os.remove(pth)
    pv = Perceiver()
    assert pv.decide({"settle": 0.01, "margin_res": 2, "margin_op": 2, "margin_args": 2})["stop"]
    lv = pv.decide({"settle": 0.5, "gap": 2.0, "unstable_slots": np.array([0, 1, 0, 1]), "margin_res": 1, "margin_op": 1, "margin_args": 1, "breath": 6})
    assert lv["views"] == 4 and lv["extra_cycle"] and lv["certificates"][0].kind == "locus" and lv["certificates"][0].born == "slot"
    assert pv.decide({"wheel_status": "unsat", "margin_res": 3, "margin_op": 3, "margin_args": 3, "settle": 1})["abstain"]
    assert not pv.decide({"settle": 1, "margin_res": 3, "margin_op": 3, "margin_args": 3})["abstain"]
    print("[perceiver] self-test PASS: LoopHealth joins/reports/saves, the designed compass reads >= 0.65 late and < 0.65 early, "
          "auroc sane, the rule table stops/spends views + locus certificate/abstains as pinned")
