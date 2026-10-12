"""scripts/direction_probe.py -- THE DIRECTION PROBE (2026-10-09, zero-GPU, CPU-only; word given
in the ledger's DIRROLE_241 entry, "THE DIRECTION LINE CLOSES ON FOUR FORMS": four forms (two
masks, a dose, a cue-as-role) left the inverse-form `res` pointer pinned at 0.25 on wild while
forward `res` sits at 0.85-0.87. THE QUESTION this answers (ledger, verbatim): is the DIRECTION of
a relation ("the result is one of my arguments" = inverse vs "the result is the new variable" =
forward) PRESENT in the body's frozen state and merely unexpressible by the res pointer's
bilinear, or ABSENT from the state entirely?

METHOD: train small read-outs (logistic regression + a 2-layer numpy MLP) from PMS8_241's own
FROZEN per-slot content states (no new forward pass, no gradient into the head -- the Goodhart
fence: these read-outs never feed back into training) to the gold direction label (classify_row's
fwd/inv, polarity_census.py, THE POSITIONAL LAW), on the DIET (.cache/welford_atlas_PMS8_241_
diet_states_breaths.npz, the 705 custody-passing rows of .cache/form_pm35c_slice1024_valid2.jsonl,
GroupKFold by row), then read ONCE on wild (.cache/clock_band_states_PMS8_241.npz, 311 rows) with
the diet-frozen read-out.

FEATURE SETS per relation slot (own state = CONTENT dims only, 384 of 512 -- _hier_band_dims(),
the same door every atlas read uses):
  A: the slot's own state (384)
  B: A's own state concatenated with its two gold args' states, ordered by slot index (384*3=1152)
  C: A's own state concatenated with the dirrole cue one-hot over its clause (384+N_DIRROLE_CUES)
  D: the cue one-hot alone, no state at all (N_DIRROLE_CUES) -- THE LEXICAL CEILING
  E: the two gold args' states alone, no own-slot state (384*2=768) -- does the DIRECTION live in
     the OPERANDS' roles rather than the relation slot's own state?
Two breaths: the FINAL breath (diet kb=6 of K_B=7; wild loop-breath 6 = states index 5) and breath
2 (diet kb=2; wild loop-breath 2 = states index 1) as a second condition.

Cues: phase1_algebra_head.dirrole_cue_spans(text) -- THE DIRECTION ROLE's own matcher, the
SAME function the trained organ reads (ledger 2026-10-09 "THE DIRECTION ROLE"), not a
reimplementation that could drift. Clause = polarity_census.clause_of (mentions[result] span if
present, else the sentence containing an arg's gold numeral) -- the SAME clause the cue census and
the polarity census both use.

BARS (pinned, from the ledger's DIRROLE_241 closing entry):
  PRESENT        if (B) or (C) reads wild AUROC >= 0.85
  ABSENT         if the best state-bearing readout is within 0.05 of (D)'s AUROC, OR < 0.70
  (anything else is reported as the honest middle, not forced into either bucket)
Also reported: whether (E) -- the args' states alone -- carries the direction (a property of the
OPERANDS' roles, not the relation slot's own state); and, at the operating point that holds the
probe's FORWARD-class precision >= 0.85 (the deployed res pointer's own forward-precision
register), the achieved INVERSE recall -- the res bar's own shape, for direct comparison against
the deployed system's wild inverse-res = 0.250.

OPERATING-POINT CAVEAT (stated, not hidden): forward precision and inverse recall move TOGETHER as
the decision threshold t (predict inverse iff p(inverse) >= t) is lowered -- lowering t shrinks the
predicted-FORWARD set to its most confident tail, which raises precision on that shrinking set
while simultaneously growing the predicted-INVERSE set, raising its recall. A literal
"maximize recall subject to precision>=0.85" search is therefore degenerate (it is satisfied,
vacuously, by carving out a handful of the single most extreme low-score points as "forward" and
calling nearly everything else "inverse"). This script instead reports the threshold t* = the
LARGEST t with forward-precision(t) >= 0.85 (the most GENEROUS forward bucket the 0.85 floor still
allows, i.e. the natural reading of "the operating point that keeps forward precision >= 0.85") and
states the size of the predicted-forward set there (n_fwd_pred) alongside the recall number, so a
reader can see when t* sits in the thin, unstable tail (n_fwd_pred small) rather than the bulk of
the distribution.

usage: .venv/bin/python3 scripts/direction_probe.py [BODY]
  BODY defaults to PMS8_241 (unchanged paths/output, below). Any other BODY (e.g. SM_241, the
  self-match-term autopsy, 2026-10-09) reads .cache/sm_autopsy_diet_states_<BODY>.npz (THE
  REPRESENTABILITY RE-READ's diet states, scripts/sm_autopsy_diet_states.py) and
  .cache/clock_band_states_<BODY>.npz (clock_band_probe.py CB_MODE=collect, same as PMS8_241's)
  by default -- override either with DP_DIET_STATES / DP_WILD_STATES / DP_OUT env vars (the rule,
  2026-10-06..08: "read scripts take the body as an argument").
outputs: .cache/direction_probe_<BODY>.txt (PMS8_241) or .cache/sm_autopsy_direction_probe_<BODY>.txt (other bodies)
"""
import collections
import json
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

os.environ["DEV"] = "CPU"
# needed ONLY for _hier_band_dims() (the content/clock dim split) -- no checkpoint load, no
# forward pass, no GPU; the exact env block atlas_radius_read.py uses for the same reason.
for _k, _v in (("DEV", "CPU"), ("ALG2", "1"), ("ALG_FTYPES", "9"), ("ALG_DUP", "1"), ("ALG_WIDE", "1"), ("ALG_HW", "512"),
               ("ALG_BREATH", "7"), ("ALG_NOTEBOOK", "1"), ("ALG_SIXWAVE", "1"), ("NB_PERSLOT", "1"), ("ALG_BINDBUS", "7"),
               ("ALG_BIND_D", "512"), ("BIND_CODES", ".cache/bindbus_codes512r.npz"), ("ALG_BUSGARAGE", "2"),
               ("ALG_SHELF_CIRCLE", "2"), ("ALG_ALTMASK", "1"), ("ALG_ALT21", "1"), ("ALG_ALT2", "1"), ("ALG_MASKHEAD", "1"),
               ("ALG_FED", "1"), ("ALG_POLAR", "1"), ("ALG_POLAR_D", "128"), ("ALG_POLAR_EM", "0.1"),
               ("ALG_POLAR_D_INIT", ".cache/polar_waist_init_d128u.npz"), ("ALG_PRUNE", "pforms,s4,fednl0,lane2"),
               ("ALG_SLOT_ALL", "1"), ("ALG_STELLAR", "2"), ("ALG_CLOCK_CANON", "1"), ("SC_EVAL", "0")):
    os.environ.setdefault(_k, _v)
assert os.environ["DEV"] == "CPU", "direction_probe: zero-GPU, always"

sys.path.insert(0, ".")
sys.path.insert(0, "scripts")

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.preprocessing import StandardScaler

import phase1_algebra_head as H                     # _hier_band_dims, dirrole_cue_spans, DIRROLE_LEXICON
from polarity_census import classify_row, clause_of  # THE POSITIONAL LAW's own fwd/inv classifier + clause locator
from args_census import sentence_bounds, sentence_spans

SEED = 0
np.random.seed(SEED)

BODY = sys.argv[1] if len(sys.argv) > 1 else "PMS8_241"
DIET_JSONL = ".cache/form_pm35c_slice1024_valid2.jsonl"
WILD_JSONL = ".cache/wild_admitted_holdout.jsonl"
if BODY == "PMS8_241":   # unchanged defaults -- no regression for the banked campaign artifact
    DIET_STATES_NPZ = os.environ.get("DP_DIET_STATES", ".cache/welford_atlas_PMS8_241_diet_states_breaths.npz")
    WILD_STATES_NPZ = os.environ.get("DP_WILD_STATES", ".cache/clock_band_states_PMS8_241.npz")
    OUT_TXT = os.environ.get("DP_OUT", ".cache/direction_probe_PMS8_241.txt")
else:   # other bodies default into the autopsy namespace (never collide with a banked PMS8_241 path)
    DIET_STATES_NPZ = os.environ.get("DP_DIET_STATES", f".cache/sm_autopsy_diet_states_{BODY}.npz")
    WILD_STATES_NPZ = os.environ.get("DP_WILD_STATES", f".cache/clock_band_states_{BODY}.npz")
    OUT_TXT = os.environ.get("DP_OUT", f".cache/sm_autopsy_direction_probe_{BODY}.txt")

# (label, diet kb index into states_all's K_B=7 axis, wild index into clock_band_states' 6-breath axis)
# diet kb: 0..6 = breath "outside time" .. loop breath 6 (final); wild index: 0..5 = loop breath 1..6
BREATH_SPECS = [("final", 6, 5), ("breath2", 2, 1)]
FEATURE_SETS = ["A", "B", "C", "D", "E"]
PRESENT_BAR = 0.85
ABSENT_DELTA = 0.05
ABSENT_FLOOR = 0.70
FWD_PRECISION_BAR = 0.85
LOGREG_CS = (0.01, 0.1, 1.0, 10.0, 100.0)
MLP_HIDDEN = 64
MLP_EPOCHS = 300
MLP_LR = 1e-3
MLP_L2 = 1e-4
MLP_BATCH = 256
N_SPLITS = 5

bands, clock_dims = H._hier_band_dims()
CONTENT = np.sort(np.concatenate(bands))
assert len(np.intersect1d(CONTENT, clock_dims)) == 0
C_DIM = len(CONTENT)
N_CUES = H.N_DIRROLE_CUES

LOG = []


def P(s=""):
    print(s, flush=True)
    LOG.append(s)


# ===========================================================================
# (1) dataset extraction -- per (row, relation-slot) sample: own state, two
# gold args' states (sorted by slot index), cue one-hot over the slot's own
# clause (polarity_census.clause_of, THE SAME clause the cue census reads)
# ===========================================================================

def extract_samples(rows, states_content, admissible, kb_idx):
    """states_content: (n_rows, n_breaths, 24, C_DIM) float, CONTENT dims only, already sliced.
    kb_idx: index into the breath axis. admissible: (n_rows,) bool or None (wild: all admitted)."""
    row_id, y, state, args_state, cue = [], [], [], [], []
    n_other = n_unresolved = 0
    for i, r in enumerate(rows):
        if admissible is not None and not admissible[i]:
            continue
        factors = r["factors"]
        cls = classify_row(factors)
        if not any(c in ("fwd", "inv") for c in cls):
            continue
        text = r["text"]
        bounds = sentence_bounds(text)
        spans = sentence_spans(text, bounds)
        cue_spans = H.dirrole_cue_spans(text)
        for k, f in enumerate(factors):
            c = cls[k]
            if c == "other":
                n_other += 1
                continue
            if c not in ("fwd", "inv"):
                continue
            a0, a1 = sorted(f["args"])
            st_own = states_content[i, kb_idx, k, :].astype(np.float32)
            st_a0 = states_content[i, kb_idx, a0, :].astype(np.float32)
            st_a1 = states_content[i, kb_idx, a1, :].astype(np.float32)
            a_start, a_end, _src = clause_of(r, k, f, bounds, spans)
            cue_vec = np.zeros(N_CUES, dtype=np.float32)
            if a_start is None:
                n_unresolved += 1
            else:
                for (cs, ce, cid) in cue_spans:
                    if cs < a_end and ce > a_start:
                        cue_vec[cid - 1] = 1.0
            row_id.append(i)
            y.append(1 if c == "inv" else 0)
            state.append(st_own)
            args_state.append(np.concatenate([st_a0, st_a1]))
            cue.append(cue_vec)
    out = dict(row=np.array(row_id), y=np.array(y, dtype=np.int64),
               state=np.array(state, dtype=np.float32),
               args_state=np.array(args_state, dtype=np.float32),
               cue=np.array(cue, dtype=np.float32))
    return out, n_other, n_unresolved


def self_reference_census(rows, admissible=None):
    """STRUCTURAL CHECK (not a model output -- the data's own construction): for every classified
    relation slot, is the slot's OWN index k among its two gold args? THE POSITIONAL LAW's
    identity convention (slot k == the variable newly introduced at k) plus reencode_ops's sub/div
    -> add/mul rewrite together imply this should be TRUE for every inverse relation (the rewritten
    factor's args are [the other operand, the newly-introduced one == k]) and FALSE for every
    forward one (both args are strictly earlier than k) -- i.e. "k in my own args" is a candidate
    DETERMINISTIC alternate label for fwd/inv, with no state or cue needed at all. Reported because
    it bears directly on how to read feature set (B)'s AUROC: if true everywhere, (B)'s own-state
    vs args-state comparison has access to an EXACT, bit-identical self-match on every inverse
    slot (one of the two concatenated arg-state blocks IS the own-state block, same slot, same
    breath) -- a much cheaper signal than "the model learned a semantic direction geometry"."""
    n_inv = n_inv_self = n_fwd = n_fwd_self = 0
    for i, r in enumerate(rows):
        if admissible is not None and not admissible[i]:
            continue
        factors = r["factors"]
        cls = classify_row(factors)
        for k, f in enumerate(factors):
            c = cls[k]
            if c == "inv":
                n_inv += 1
                n_inv_self += int(k in f["args"])
            elif c == "fwd":
                n_fwd += 1
                n_fwd_self += int(k in f["args"])
    return dict(n_inv=n_inv, n_inv_self=n_inv_self, n_fwd=n_fwd, n_fwd_self=n_fwd_self)


def feat(samples, which):
    if which == "A":
        return samples["state"]
    if which == "B":
        return np.concatenate([samples["state"], samples["args_state"]], axis=1)
    if which == "C":
        return np.concatenate([samples["state"], samples["cue"]], axis=1)
    if which == "D":
        return samples["cue"]
    if which == "E":
        return samples["args_state"]
    raise ValueError(which)


# ===========================================================================
# (2) the 2-layer MLP -- plain numpy, Adam, seeded (no sklearn MLP: the
# word asked for "numpy or tinygrad CPU", a controlled, auditable trainer)
# ===========================================================================

class MLP:
    def __init__(self, n_in, n_hidden=MLP_HIDDEN, seed=SEED, l2=MLP_L2):
        rng = np.random.RandomState(seed)
        self.W1 = (rng.randn(n_in, n_hidden) * np.sqrt(2.0 / n_in)).astype(np.float64)
        self.b1 = np.zeros(n_hidden)
        self.W2 = (rng.randn(n_hidden, 1) * np.sqrt(2.0 / n_hidden)).astype(np.float64)
        self.b2 = np.zeros(1)
        self.l2 = l2

    def _forward(self, X):
        Z1 = X @ self.W1 + self.b1
        A1 = np.maximum(Z1, 0.0)
        Z2 = A1 @ self.W2 + self.b2
        P_ = 1.0 / (1.0 + np.exp(-np.clip(Z2, -30, 30)))
        return Z1, A1, Z2, P_

    def predict_proba(self, X):
        return self._forward(X)[3].ravel()

    def fit(self, X, y, epochs=MLP_EPOCHS, lr=MLP_LR, batch_size=MLP_BATCH, seed=SEED):
        rng = np.random.RandomState(seed)
        n = len(X)
        yb_full = y.reshape(-1, 1).astype(np.float64)
        mW1 = np.zeros_like(self.W1); vW1 = np.zeros_like(self.W1)
        mb1 = np.zeros_like(self.b1); vb1 = np.zeros_like(self.b1)
        mW2 = np.zeros_like(self.W2); vW2 = np.zeros_like(self.W2)
        mb2 = np.zeros_like(self.b2); vb2 = np.zeros_like(self.b2)
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        t = 0
        for ep in range(epochs):
            idx = rng.permutation(n)
            for s0 in range(0, n, batch_size):
                bi = idx[s0:s0 + batch_size]
                Xb, yb = X[bi], yb_full[bi]
                Z1, A1, Z2, Pp = self._forward(Xb)
                m = len(bi)
                dZ2 = (Pp - yb) / m
                dW2 = A1.T @ dZ2 + self.l2 * self.W2
                db2 = dZ2.sum(0)
                dA1 = dZ2 @ self.W2.T
                dZ1 = dA1 * (Z1 > 0)
                dW1 = Xb.T @ dZ1 + self.l2 * self.W1
                db1 = dZ1.sum(0)
                t += 1
                for (p_, g_, m_, v_) in ((self.W1, dW1, mW1, vW1), (self.b1, db1, mb1, vb1),
                                         (self.W2, dW2, mW2, vW2), (self.b2, db2, mb2, vb2)):
                    m_ *= beta1; m_ += (1 - beta1) * g_
                    v_ *= beta2; v_ += (1 - beta2) * (g_ ** 2)
                    mhat = m_ / (1 - beta1 ** t)
                    vhat = v_ / (1 - beta2 ** t)
                    p_ -= lr * mhat / (np.sqrt(vhat) + eps)
        return self


def mlp_cv(X, y, groups, n_splits=N_SPLITS, seed=SEED):
    gkf = GroupKFold(n_splits=n_splits)
    aucs, accs = [], []
    for tr, te in gkf.split(X, y, groups):
        sc = StandardScaler().fit(X[tr])
        Xtr, Xte = sc.transform(X[tr]), sc.transform(X[te])
        net = MLP(Xtr.shape[1], seed=seed).fit(Xtr, y[tr], seed=seed)
        p = net.predict_proba(Xte)
        aucs.append(roc_auc_score(y[te], p))
        accs.append(accuracy_score(y[te], p >= 0.5))
    return float(np.mean(aucs)), float(np.mean(accs)), aucs, accs


def mlp_fit_final(X, y, seed=SEED):
    sc = StandardScaler().fit(X)
    Xs = sc.transform(X)
    net = MLP(Xs.shape[1], seed=seed).fit(Xs, y, seed=seed)
    return sc, net


def logreg_cv(X, y, groups, Cs=LOGREG_CS, n_splits=N_SPLITS, seed=SEED):
    gkf = GroupKFold(n_splits=n_splits)
    best_C, best_auc = None, -1.0
    per_C = {}
    for C in Cs:
        aucs, accs = [], []
        for tr, te in gkf.split(X, y, groups):
            sc = StandardScaler().fit(X[tr])
            Xtr, Xte = sc.transform(X[tr]), sc.transform(X[te])
            clf = LogisticRegression(penalty="l2", C=C, max_iter=3000, random_state=seed)
            clf.fit(Xtr, y[tr])
            p = clf.predict_proba(Xte)[:, 1]
            aucs.append(roc_auc_score(y[te], p))
            accs.append(accuracy_score(y[te], p >= 0.5))
        m_auc, m_acc = float(np.mean(aucs)), float(np.mean(accs))
        per_C[C] = (m_auc, m_acc)
        if m_auc > best_auc:
            best_auc, best_C = m_auc, C
    return best_C, per_C


def logreg_fit_final(X, y, C, seed=SEED):
    sc = StandardScaler().fit(X)
    Xs = sc.transform(X)
    clf = LogisticRegression(penalty="l2", C=C, max_iter=3000, random_state=seed)
    clf.fit(Xs, y)
    return sc, clf


# ===========================================================================
# (3) the operating-point read: t* = largest threshold with forward
# precision (predicted p(inverse) < t) >= FWD_PRECISION_BAR; report inverse
# recall AND n_fwd_pred there (stated caveat: degenerate at the thin tail)
# ===========================================================================

def operating_point(y, p, bar=FWD_PRECISION_BAR):
    order = np.argsort(p)              # ascending p; t sweeps candidate thresholds = unique p values
    ts = np.unique(p)
    best_t, best_recall, best_n, best_prec = None, None, None, None
    n_inv_total = int((y == 1).sum())
    for t in ts:
        fwd_pred = p < t
        n_fwd_pred = int(fwd_pred.sum())
        if n_fwd_pred == 0:
            continue
        prec = float((y[fwd_pred] == 0).mean())
        if prec >= bar:
            inv_pred = ~fwd_pred
            recall = float(((y == 1) & inv_pred).sum()) / max(n_inv_total, 1)
            if best_t is None or t > best_t:
                best_t, best_recall, best_n, best_prec = float(t), recall, n_fwd_pred, prec
    if best_t is None:
        return None
    return dict(t=best_t, inverse_recall=best_recall, n_fwd_pred=best_n, fwd_precision=best_prec,
                n_total=len(y))


# ===========================================================================
# main
# ===========================================================================

def main():
    t0 = time.time()
    P(f"THE DIRECTION PROBE -- {BODY} (2026-10-09, zero-GPU)")
    P(f"generated {os.popen('date').read().strip()}")
    P(f"content dims {C_DIM}/512 (bands {[len(b) for b in bands]}; {len(clock_dims)} clock dims excluded); "
      f"N_DIRROLE_CUES={N_CUES}")

    diet_rows = [json.loads(l) for l in open(DIET_JSONL)]
    wild_rows = [json.loads(l) for l in open(WILD_JSONL)]
    z_diet = np.load(DIET_STATES_NPZ)
    states_diet_full = z_diet["states_all"]          # (n, 7, 24, 384) -- already content-only (welford_atlas's own build slices CONTENT before saving)
    admissible = z_diet["admissible"]
    assert states_diet_full.shape[0] == len(diet_rows), (states_diet_full.shape, len(diet_rows))
    assert states_diet_full.shape[-1] == C_DIM, (states_diet_full.shape, C_DIM)
    P(f"diet: {len(diet_rows)} rows, {int(admissible.sum())} custody-admissible, source {DIET_JSONL}")

    z_wild = np.load(WILD_STATES_NPZ)
    states_wild_raw = z_wild["states"]                # (311, 6, 24, 512) -- full width, slice CONTENT below
    assert states_wild_raw.shape[0] == len(wild_rows), (states_wild_raw.shape, len(wild_rows))
    states_wild_full = states_wild_raw[:, :, :, CONTENT].astype(np.float32)
    P(f"wild: {len(wild_rows)} rows, source {WILD_STATES_NPZ}")

    P(f"\n{'='*78}\nTHE SELF-REFERENCE STRUCTURAL CHECK (data construction, not a model output)\n{'='*78}")
    sr_d = self_reference_census(diet_rows, admissible)
    sr_w = self_reference_census(wild_rows, None)
    P(f"  diet: inv slots with own-index k in its own args: {sr_d['n_inv_self']}/{sr_d['n_inv']} "
      f"({sr_d['n_inv_self']/max(sr_d['n_inv'],1):.4f}); fwd slots: {sr_d['n_fwd_self']}/{sr_d['n_fwd']}")
    P(f"  wild: inv slots with own-index k in its own args: {sr_w['n_inv_self']}/{sr_w['n_inv']} "
      f"({sr_w['n_inv_self']/max(sr_w['n_inv'],1):.4f}); fwd slots: {sr_w['n_fwd_self']}/{sr_w['n_fwd']}")
    sr_clean = (sr_d['n_inv_self'] == sr_d['n_inv'] and sr_d['n_fwd_self'] == 0 and
                sr_w['n_inv_self'] == sr_w['n_inv'] and sr_w['n_fwd_self'] == 0)
    P(f"  READING: {'EXACT and UNIVERSAL' if sr_clean else 'not universal (see counts)'} -- "
      "every inverse relation's own slot index is literally one of its own two gold args (the "
      "reencode_ops sub/div->add/mul rewrite's signature); every forward relation's never is. "
      "Feature set (B) therefore has access to a cheap, bit-identical SELF-MATCH signal on every "
      "inverse slot (one of its two concatenated arg-state blocks IS the own-state block, same "
      "slot, same breath) that (E) cannot see directly -- (B)'s AUROC above should be read with "
      "this in mind: it may substantially reflect an identity/self-match detector, not a learned "
      "semantic direction geometry. (E)'s own AUROC (args alone, no own-state to self-match "
      "against) is the cleaner read of whether the STATE carries a genuine direction signature.")

    results = {}       # (breath, fset, model) -> dict
    wild_results = {}  # (breath, fset, model) -> dict

    for breath_label, kb_diet, idx_wild in BREATH_SPECS:
        P(f"\n{'='*78}\nBREATH: {breath_label} (diet kb={kb_diet}, wild idx={idx_wild})\n{'='*78}")
        diet_samples, n_other_d, n_unres_d = extract_samples(diet_rows, states_diet_full, admissible, kb_diet)
        wild_samples, n_other_w, n_unres_w = extract_samples(wild_rows, states_wild_full, None, idx_wild)
        n_fwd_d = int((diet_samples["y"] == 0).sum()); n_inv_d = int((diet_samples["y"] == 1).sum())
        n_fwd_w = int((wild_samples["y"] == 0).sum()); n_inv_w = int((wild_samples["y"] == 1).sum())
        P(f"  diet relation slots: fwd={n_fwd_d} inv={n_inv_d} (other={n_other_d}, clause-unresolved={n_unres_d})")
        P(f"  wild relation slots: fwd={n_fwd_w} inv={n_inv_w} (other={n_other_w}, clause-unresolved={n_unres_w})")

        for fset in FEATURE_SETS:
            Xd = feat(diet_samples, fset)
            yd = diet_samples["y"]
            groups_d = diet_samples["row"]
            Xw = feat(wild_samples, fset)
            yw = wild_samples["y"]

            # --- logistic regression ---
            best_C, per_C = logreg_cv(Xd, yd, groups_d)
            cv_auc, cv_acc = per_C[best_C]
            sc, clf = logreg_fit_final(Xd, yd, best_C)
            pw = clf.predict_proba(sc.transform(Xw))[:, 1]
            w_auc = roc_auc_score(yw, pw); w_acc = accuracy_score(yw, pw >= 0.5)
            op = operating_point(yw, pw)
            results[(breath_label, fset, "logreg")] = dict(cv_auc=cv_auc, cv_acc=cv_acc, best_C=best_C, per_C=per_C)
            wild_results[(breath_label, fset, "logreg")] = dict(auc=w_auc, acc=w_acc, op=op)
            P(f"  [{fset}] logreg  diet CV auroc={cv_auc:.4f} acc={cv_acc:.4f} (C={best_C})  "
              f"| wild auroc={w_auc:.4f} acc={w_acc:.4f}" +
              (f"  | op t*={op['t']:.3f} fwd_prec={op['fwd_precision']:.3f} n_fwd_pred={op['n_fwd_pred']}/{op['n_total']} "
               f"INV_RECALL={op['inverse_recall']:.4f}" if op else "  | op: NO threshold reaches fwd precision>=0.85"))

            # --- 2-layer MLP ---
            m_auc, m_acc, _, _ = mlp_cv(Xd, yd, groups_d)
            sc2, net = mlp_fit_final(Xd, yd)
            pw2 = net.predict_proba(sc2.transform(Xw))
            w_auc2 = roc_auc_score(yw, pw2); w_acc2 = accuracy_score(yw, pw2 >= 0.5)
            op2 = operating_point(yw, pw2)
            results[(breath_label, fset, "mlp")] = dict(cv_auc=m_auc, cv_acc=m_acc)
            wild_results[(breath_label, fset, "mlp")] = dict(auc=w_auc2, acc=w_acc2, op=op2)
            P(f"  [{fset}] mlp     diet CV auroc={m_auc:.4f} acc={m_acc:.4f}            "
              f"| wild auroc={w_auc2:.4f} acc={w_acc2:.4f}" +
              (f"  | op t*={op2['t']:.3f} fwd_prec={op2['fwd_precision']:.3f} n_fwd_pred={op2['n_fwd_pred']}/{op2['n_total']} "
               f"INV_RECALL={op2['inverse_recall']:.4f}" if op2 else "  | op: NO threshold reaches fwd precision>=0.85"))

    # ================= summary tables =================
    P(f"\n{'='*78}\nSUMMARY -- diet 5-fold GroupKFold CV AUROC / wild (once) AUROC, per breath x feature set\n{'='*78}")
    for breath_label, _, _ in BREATH_SPECS:
        P(f"\n--- {breath_label} ---")
        P(f"  {'set':4s} {'model':7s} {'diet_cv_auc':>11s} {'diet_cv_acc':>11s} {'wild_auc':>9s} {'wild_acc':>9s} {'inv_recall@fp>=0.85':>20s} {'n_fwd_pred':>11s}")
        for fset in FEATURE_SETS:
            for model in ("logreg", "mlp"):
                r = results[(breath_label, fset, model)]
                wr = wild_results[(breath_label, fset, model)]
                op = wr["op"]
                rec_s = f"{op['inverse_recall']:.4f}" if op else "n/a"
                nfp_s = f"{op['n_fwd_pred']}/{op['n_total']}" if op else "-"
                P(f"  {fset:4s} {model:7s} {r['cv_auc']:11.4f} {r['cv_acc']:11.4f} {wr['auc']:9.4f} {wr['acc']:9.4f} {rec_s:>20s} {nfp_s:>11s}")

    # ================= the verdict =================
    P(f"\n{'='*78}\nTHE READING\n{'='*78}")
    # best-of over (B,C) x (logreg,mlp) x breath, on WILD auroc -- the bar is a wild number
    best_bc = max(((wild_results[(bl, fs, m)]["auc"], bl, fs, m)
                    for bl, _, _ in BREATH_SPECS for fs in ("B", "C") for m in ("logreg", "mlp")))
    best_bc_auc, best_bc_breath, best_bc_fset, best_bc_model = best_bc
    best_d = max(((wild_results[(bl, "D", m)]["auc"], bl, m) for bl, _, _ in BREATH_SPECS for m in ("logreg", "mlp")))
    best_d_auc, best_d_breath, best_d_model = best_d
    best_state_only = max(((wild_results[(bl, "A", m)]["auc"], bl, m) for bl, _, _ in BREATH_SPECS for m in ("logreg", "mlp")))
    best_state_auc, best_state_breath, best_state_model = best_state_only
    best_e = max(((wild_results[(bl, "E", m)]["auc"], bl, m) for bl, _, _ in BREATH_SPECS for m in ("logreg", "mlp")))
    best_e_auc, best_e_breath, best_e_model = best_e
    best_any_state = max(best_bc_auc, best_state_auc, best_e_auc)

    if best_bc_auc >= PRESENT_BAR:
        verdict = "PRESENT-BUT-UNEXPRESSIBLE"
    elif (best_any_state - best_d_auc) <= ABSENT_DELTA or best_any_state < ABSENT_FLOOR:
        verdict = "ABSENT"
    else:
        verdict = "HONEST MIDDLE (neither bar met: some state signal above the lexical ceiling, but below 0.85)"

    L1 = (f"1. BEST (B)/(C) WILD AUROC = {best_bc_auc:.4f} ({best_bc_fset} x {best_bc_model} @ {best_bc_breath}) "
          f"vs PRESENT bar {PRESENT_BAR:.2f}.")
    L2 = (f"2. LEXICAL CEILING (D, cue ids alone) best WILD AUROC = {best_d_auc:.4f} ({best_d_model} @ {best_d_breath}); "
          f"best state-bearing set (A/B/C/E) WILD AUROC = {best_any_state:.4f} -- gap over the lexical ceiling = "
          f"{best_any_state - best_d_auc:+.4f} (ABSENT needs this <= {ABSENT_DELTA:.2f} or the state-bearing best < {ABSENT_FLOOR:.2f}).")
    L3 = (f"3. OWN-STATE ALONE (A) best WILD AUROC = {best_state_auc:.4f} ({best_state_model} @ {best_state_breath}); "
          f"ARGS-STATES ALONE (E, no self-match possible -- see THE SELF-REFERENCE STRUCTURAL CHECK above) "
          f"best WILD AUROC = {best_e_auc:.4f} ({best_e_model} @ {best_e_breath}) -- "
          f"{'the operands carry MORE than the relation slot itself, and (E) alone nearly clears the PRESENT bar without any self-match shortcut' if best_e_auc > best_state_auc + 0.02 else 'no clear operand-vs-own-slot asymmetry'}; "
          f"(B)'s higher score likely mixes this real signal with the self-reference artifact.")
    op_best = wild_results[(best_bc_breath, best_bc_fset, best_bc_model)]["op"]
    L4 = (f"4. AT THE OPERATING POINT holding forward precision >= {FWD_PRECISION_BAR:.2f} (the deployed res pointer's own "
          f"forward register, 0.85-0.87), the best (B)/(C) read-out's INVERSE RECALL = " +
          (f"{op_best['inverse_recall']:.4f} (n_fwd_pred={op_best['n_fwd_pred']}/{op_best['n_total']}, fwd_prec={op_best['fwd_precision']:.3f})"
           if op_best else "UNREACHABLE (no threshold clears the 0.85 forward-precision floor)") +
          f" vs the deployed system's wild inverse res = 0.250.")
    L5 = (f"5. VERDICT: {verdict}.")
    L6 = ("6. CAVEAT: forward precision and inverse recall move TOGETHER as the threshold is lowered (see the "
          "module docstring) -- a small n_fwd_pred at the operating point means that number sits in the thin, "
          "unstable tail of the score distribution, not the bulk; read it alongside fwd_precision and n_fwd_pred, "
          "not alone.")
    for l in (L1, L2, L3, L4, L5, L6):
        P(l)

    P(f"\n[timing] {time.time()-t0:.1f}s")
    os.makedirs(".cache", exist_ok=True)
    with open(OUT_TXT, "w") as fh:
        fh.write("\n".join(LOG) + "\n")
    print(f"\n[direction_probe] wrote {OUT_TXT}")


if __name__ == "__main__":
    main()
