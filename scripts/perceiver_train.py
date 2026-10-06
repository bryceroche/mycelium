"""perceiver_train.py -- THE LEARNED PERCEIVER v1, PART 2 (2026-10-06, "WORD GIVEN: THE LEARNED
PERCEIVER v1", docs/phase1_skeleton_spec.md 06:45, extended same day by Bryce's "change of diet"
message). Trains the tiny temporal net on scripts/perceiver_collect.py's telemetry.

THE GOODHART FENCE (mycelium/diagnostic_register.py; CLAUDE.md's own law, restated in the mission
brief): the perceiver is trained ON diagnostics (entropy, atlas distance, the certifier score, solver
status, parse-space count) and its output NEVER enters the head's loss -- this script trains a
STANDALONE model that reads the SAME telemetry a perceiver_collect.py run banked to disk; nothing
here touches phase1_algebra_head.py's parameters, loss, or training loop. It emits read-time
decisions only (COMMIT / STOP), the telegraph's operator, never a gradient into the body.

TWO HEADS, TWO DIETS (per the 2026-10-06 "change of diet" message):
  COMMIT (per slot: P(right at the final breath)) trains on the ANNOTATED diet slice ONLY
    (.cache/perceiver_telemetry_PMS8_241_pm35cslicevalid2.npz) -- per-slot labels need a gold factor
    graph, which raw GSM8K rows do not carry.
  STOP (per row, per breath: P(this breath's decode passes the judge AND is right)) trains on the
    diet slice PLUS every GSM8K-train telemetry chunk handed to --gsm8k (globbed/concatenated) --
    STOP labels need only the row's own key (the GSM8K "#### answer" or, for the diet slice, the
    custody-gold key), which both sources carry.

MODEL: tinygrad on CPU (DEV=CPU; this script never touches the GPU or .cache/gpu.lock), chosen over
hand-rolled numpy for autodiff correctness -- "simplest and deterministic" per the mission brief, and
every other model in this codebase already uses tinygrad, so there is nothing new to debug. A 1-D
conv over the breath axis (kernel 3, 'same' padding, built as a stack-the-window-then-Linear trick
rather than a dedicated conv op -- K_B is fixed and tiny (7), so this is exactly a kernel-3 conv1d
with no extra machinery) + a 2-layer MLP. < 20k params for EACH head (counted at the bottom of
build_model()). Fixed seed (SEED, default 0) everywhere numpy/tinygrad touch randomness.

CV: GroupKFold(5) BY ROW on the diet slice (both heads share the SAME row->fold assignment, so a
slot's fold is its row's fold) -- never splits a row's slots across train/val. Baselines: (1) the
final-breath entropy threshold (AUROC of -entropy as a right/wrong score); (2) the breath-1 (index 0)
entropy, same scoring. The commit operating point: the probability threshold whose coverage (fraction
of ALL gold-present slots fired on) is >= 20%, read off the val fold; its precision is reported
against the bar (>= 0.90). The stop head is compared against "the judge alone" (stop_kb_judge, i.e.
first solved+unique breath with NO perceiver and NO key) by ROW ACCURACY under each rule, plus
regressions (last-breath-correct rows the rule gets wrong).

BARS (ledger, pinned before this script ran): slice CV commit AUROC >= entropy baseline + 0.05; on
ONE wild read (never trained on, measured once): stop rows >= last-breath's (PMS8_241 mask=1 = 14/311
rows, the banked read chain_acc.py quotes); regressions <= 2; commit precision at >= 20% coverage >=
0.90; AUROC >= entropy baseline + 0.05. KILL: wild AUROC below the entropy baseline.

usage:
  .venv/bin/python3 scripts/perceiver_train.py .cache/perceiver_telemetry_PMS8_241_pm35cslicevalid2.npz \\
      --gsm8k ".cache/perceiver_telemetry_PMS8_241_gsm8kTRAIN*.npz" \\
      --eval .cache/perceiver_telemetry_PMS8_241_wildhold.npz
"""
import os
os.environ.setdefault("DEV", "CPU")
import sys
import glob
import json
import argparse
import numpy as np

sys.path.insert(0, ".")
sys.path.insert(0, "scripts")

SEED = 0
K_B_EXPECT = None  # filled from the first npz read; every telemetry file must agree

STATUS_CLASSES = ["solved", "unsat", "budget", "refused", "unbuildable", "hung"]
BAND_CLASSES = [-1, 0, 1, 2, 3, 4]  # -1 = no_ref/no gold match


def _status_onehot(status_arr):
    """(..., len(STATUS_CLASSES)) one-hot off an array of status strings; anything not in the
    table (should not happen -- solver_status is always one of the six) lands all-zero, never a
    crash."""
    out = np.zeros(status_arr.shape + (len(STATUS_CLASSES),), np.float32)
    for ci, c in enumerate(STATUS_CLASSES):
        out[..., ci] = (status_arr == c).astype(np.float32)
    return out


def _band_onehot(band_arr):
    out = np.zeros(band_arr.shape + (len(BAND_CLASSES),), np.float32)
    for ci, c in enumerate(BAND_CLASSES):
        out[..., ci] = (band_arr == c).astype(np.float32)
    return out


def _nan0(x):
    return np.nan_to_num(x, nan=0.0)


def load_npz(path):
    z = np.load(path, allow_pickle=True)
    meta = json.loads(str(z["meta"]))
    return z, meta


# ===========================================================================================================
# FEATURE BUILDERS
# ===========================================================================================================

def commit_features(z):
    """(N_rows, K_B, L_FAC, F) slot-level feature tensor + (N_rows, L_FAC) right_final labels
    (-1 N/A). Only telemetry with has_gold=True carries a meaningful right_final/band -- this is
    asserted by the caller, not silently tolerated here."""
    ent, numarg, nummass, band = z["ent"], z["numarg"], z["nummass"], z["band"]
    cos_given, cos_rel = z["cos_given"], z["cos_rel"]
    leaf_chg, root_chg, branch_chg = _nan0(z["leaf_chg"]), _nan0(z["root_chg"]), _nan0(z["branch_chg"])
    nlc, ps, status = z["nl_certifier"], z["parse_space"], z["solver_status"]
    n, K_B, L = ent.shape
    ps_valid = (ps >= 0).astype(np.float32)
    ps_f = np.where(ps >= 0, ps, 0).astype(np.float32)
    band_oh = _band_onehot(band)                       # (n, K_B, L, 6)
    status_oh = _status_onehot(status)                  # (n, K_B, 6)
    status_oh_b = np.repeat(status_oh[:, :, None, :], L, axis=2)
    ps_b = np.repeat(ps_f[:, :, None], L, axis=2)
    psv_b = np.repeat(ps_valid[:, :, None], L, axis=2)
    nlc_b = np.repeat(nlc[:, :, None], L, axis=2)
    feats = [ent[..., None], numarg[..., None].astype(np.float32), nummass[..., None],
             band_oh, cos_given[..., None], cos_rel[..., None],
             leaf_chg[..., None], root_chg[..., None], branch_chg[..., None],
             nlc_b[..., None], ps_b[..., None], psv_b[..., None], status_oh_b]
    X = np.concatenate([_nan0(f) for f in feats], axis=-1).astype(np.float32)  # (n, K_B, L, F)
    return X, z["right_final"]


def stop_features(z):
    """(N_rows, K_B, F2) row-level feature tensor + (N_rows, K_B) binary stop targets
    ((judge_pass==1)&(key_match==1)) + (N_rows,) stop_kb_judge (the key-blind judge-alone rule, the
    comparison point) + (N_rows,) key + (N_rows,) q + rows' own text (for last-label bookkeeping)."""
    ent, numarg, nummass = z["ent"], z["numarg"], z["nummass"]
    cos_given, cos_rel = z["cos_given"], z["cos_rel"]
    leaf_chg, root_chg, branch_chg = _nan0(z["leaf_chg"]), _nan0(z["root_chg"]), _nan0(z["branch_chg"])
    nlc, ps, status, dry = z["nl_certifier"], z["parse_space"], z["solver_status"], _nan0(z["dry"])
    n, K_B, L = ent.shape
    mean_ax = lambda a: a.mean(-1)
    max_ax = lambda a: a.max(-1)
    ps_valid = (ps >= 0).astype(np.float32)
    ps_f = np.where(ps >= 0, ps, 0).astype(np.float32)
    status_oh = _status_onehot(status)   # (n, K_B, 6)
    kb_idx = np.repeat((np.arange(K_B, dtype=np.float32) / max(K_B - 1, 1))[None, :], n, axis=0)
    feats = [mean_ax(ent), max_ax(ent), mean_ax(numarg.astype(np.float32)), mean_ax(nummass),
             mean_ax(cos_given), mean_ax(cos_rel), max_ax(cos_given),
             mean_ax(leaf_chg), mean_ax(root_chg), mean_ax(branch_chg),
             nlc, ps_f, ps_valid, dry, kb_idx]
    X = np.stack([_nan0(f) for f in feats], axis=-1).astype(np.float32)  # (n, K_B, F2)
    X = np.concatenate([X, status_oh.astype(np.float32)], axis=-1)
    jp, km = z["judge_pass"], z["key_match"]
    y = ((jp == 1) & (km == 1)).astype(np.float32)
    return X, y, z["stop_kb_judge"], z["key"], z["q"]


# ===========================================================================================================
# THE MODEL (tinygrad; a kernel-3 "conv1d over breaths" built as a window+Linear, + a 2-layer MLP)
# ===========================================================================================================

def _window3(X):
    """(N, K_B, F) -> (N, K_B, 3F): zero-padded kernel-3 window per breath position (host-side numpy,
    no tinygrad op needed -- K_B is fixed/tiny)."""
    N, K_B, F = X.shape
    Xp = np.zeros((N, K_B + 2, F), X.dtype)
    Xp[:, 1:K_B + 1, :] = X
    return np.concatenate([Xp[:, 0:K_B, :], Xp[:, 1:K_B + 1, :], Xp[:, 2:K_B + 2, :]], axis=-1)


class TinyTemporalNet:
    """Shared architecture for both heads: window3 -> Linear(3F, H) -> relu -> [COMMIT: flatten K_B*H
    -> Linear(H2) -> relu -> Linear(1) -> sigmoid, ONE output per sample] or [STOP: per-breath shared
    Linear(H,1) -> sigmoid, K_B outputs per sample]. H/H2 kept small enough that EITHER head is well
    under 20k params (printed at construction)."""

    def __init__(self, F_in, K_B, head, H=16, H2=16, seed=SEED):
        from tinygrad import Tensor, dtypes
        self.Tensor = Tensor
        self.dtypes = dtypes
        self.K_B = K_B
        self.head = head
        rng = np.random.RandomState(seed)

        def lin(fan_in, fan_out):
            w = (rng.randn(fan_in, fan_out) * (1.0 / np.sqrt(fan_in))).astype(np.float32)
            b = np.zeros(fan_out, np.float32)
            return Tensor(w, requires_grad=True), Tensor(b, requires_grad=True)

        self.W1, self.b1 = lin(3 * F_in, H)
        if head == "commit":
            self.W2, self.b2 = lin(K_B * H, H2)
            self.W3, self.b3 = lin(H2, 1)
            params = [self.W1, self.b1, self.W2, self.b2, self.W3, self.b3]
        else:
            self.W2, self.b2 = lin(H, 1)
            params = [self.W1, self.b1, self.W2, self.b2]
        self.params = params
        n_params = sum(int(np.prod(p.shape)) for p in params)
        print(f"[perceiver-train] {head} head: F_in={F_in} H={H} H2={H2 if head == 'commit' else '-'} "
              f"params={n_params} (bar < 20000: {'OK' if n_params < 20000 else 'OVER'})", flush=True)

    def forward(self, X_np):
        """X_np: (N, K_B, F_in) -> commit: (N,) sigmoid logits; stop: (N, K_B) sigmoid logits."""
        Tensor, dtypes = self.Tensor, self.dtypes
        Xw = _window3(X_np)                                      # (N, K_B, 3F)
        N, K_B, F3 = Xw.shape
        h = (Tensor(Xw.reshape(N * K_B, F3), dtype=dtypes.float) @ self.W1 + self.b1).relu()  # (N*K_B, H)
        if self.head == "commit":
            h = h.reshape(N, K_B * h.shape[-1])
            h2 = (h @ self.W2 + self.b2).relu()
            out = (h2 @ self.W3 + self.b3).reshape(N)
        else:
            out = (h @ self.W2 + self.b2).reshape(N, K_B)
        return out.sigmoid()

    def bce_fit(self, X_np, y_np, mask_np=None, epochs=200, lr=0.05, verbose=False):
        """Full-batch Adam on binary cross-entropy (small data -- no minibatching needed). mask_np
        (same shape as y_np): 1 = include in the loss, 0 = exclude (COMMIT's -1 N/A labels)."""
        from tinygrad.nn.optim import Adam
        from tinygrad import Tensor
        opt = Adam(self.params, lr=lr)
        y = self.Tensor(y_np.astype(np.float32), dtype=self.dtypes.float)
        m = self.Tensor((mask_np if mask_np is not None else np.ones_like(y_np)).astype(np.float32),
                        dtype=self.dtypes.float)
        Tensor.training = True
        try:
            for ep in range(epochs):
                p = self.forward(X_np).clip(1e-6, 1 - 1e-6)
                bce = -(y * p.log() + (1 - y) * (1 - p).log())
                loss = (bce * m).sum() / m.sum().clip(1, None)
                opt.zero_grad()
                loss.backward()
                opt.step()
                if verbose and ep % 50 == 0:
                    print(f"    epoch {ep}: loss {loss.numpy():.4f}", flush=True)
        finally:
            Tensor.training = False
        return self

    def predict(self, X_np):
        from tinygrad import Tensor
        Tensor.training = False
        return self.forward(X_np).numpy()


# ===========================================================================================================
# METRICS
# ===========================================================================================================

def auroc(y_true, score, mask=None):
    from sklearn.metrics import roc_auc_score
    y_true = np.asarray(y_true).ravel()
    score = np.asarray(score).ravel()
    if mask is not None:
        mask = np.asarray(mask).ravel().astype(bool)
        y_true, score = y_true[mask], score[mask]
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, score))


def precision_at_coverage(y_true, score, mask, coverage=0.20):
    """threshold chosen so that >= `coverage` fraction of mask==1 entries fire; returns
    (threshold, precision, actual_coverage)."""
    y_true = np.asarray(y_true).ravel()[mask]
    score = np.asarray(score).ravel()[mask]
    order = np.argsort(-score)
    n_fire = max(1, int(np.ceil(coverage * len(score))))
    thr = score[order[n_fire - 1]]
    fired = score >= thr
    prec = float(y_true[fired].mean()) if fired.any() else float("nan")
    return float(thr), prec, float(fired.mean())


# ===========================================================================================================
# CV (diet slice only; GroupKFold BY ROW)
# ===========================================================================================================

def run_cv(diet_npz, n_folds=5, epochs=200):
    from sklearn.model_selection import GroupKFold
    z, meta = load_npz(diet_npz)
    assert meta["has_gold"], f"{diet_npz}: not an annotated fixture (has_gold=False) -- CV needs gold"
    Xc_full, right_final = commit_features(z)   # (n, K_B, L, Fc), (n, L)
    Xs_full, stop_y, stop_judge, key, q = stop_features(z)   # (n, K_B, Fs), (n,K_B)
    n, K_B, L, Fc = Xc_full.shape
    Fs = Xs_full.shape[-1]
    ent_final = z["ent"][:, -1, :]
    ent_b0 = z["ent"][:, 0, :]
    last_label = z["last_label"]

    rows_idx = np.arange(n)
    gkf = GroupKFold(n_splits=n_folds)
    fold_of = np.full(n, -1, np.int32)
    for f, (_, val_idx) in enumerate(gkf.split(rows_idx, groups=rows_idx)):
        fold_of[val_idx] = f

    print(f"\n{'='*100}\nSLICE CV ({diet_npz}): n={n} rows, {n_folds}-fold GroupKFold BY ROW, epochs={epochs}\n{'='*100}")

    commit_aurocs, commit_aurocs_ent, commit_aurocs_b0, commit_precisions, commit_coverages = [], [], [], [], []
    stop_rows_model, stop_rows_judge, stop_regr = [], [], []
    for f in range(n_folds):
        val_mask_rows = fold_of == f
        tr_mask_rows = ~val_mask_rows
        # ---- COMMIT ----
        gp_tr = right_final[tr_mask_rows] >= 0
        gp_val = right_final[val_mask_rows] >= 0
        Xc_tr, yc_tr = Xc_full[tr_mask_rows], np.clip(right_final[tr_mask_rows], 0, 1)
        Xc_val, yc_val = Xc_full[val_mask_rows], np.clip(right_final[val_mask_rows], 0, 1)
        # slot-level flatten: (n_tr, K_B, L, Fc) -> (n_tr*L, K_B, Fc), keep only gold-present slots'
        # labels in the loss mask (trained on ALL slots' features -- the mask zeroes the N/A ones'
        # contribution to the loss, never drops them from the batch, so shapes stay rectangular).
        def flat(X):
            return X.transpose(0, 2, 1, 3).reshape(-1, K_B, Fc)
        Xc_tr_f, Xc_val_f = flat(Xc_tr), flat(Xc_val)
        yc_tr_f = yc_tr.reshape(-1)
        mask_tr_f = gp_tr.reshape(-1)
        net_c = TinyTemporalNet(Fc, K_B, "commit", seed=SEED + f)
        net_c.bce_fit(Xc_tr_f, yc_tr_f, mask_tr_f, epochs=epochs)
        p_val = net_c.predict(Xc_val_f)
        yc_val_f = yc_val.reshape(-1)
        mask_val_f = gp_val.reshape(-1)
        a = auroc(yc_val_f, p_val, mask_val_f)
        a_ent = auroc(yc_val_f, -ent_final[val_mask_rows].reshape(-1), mask_val_f)
        a_b0 = auroc(yc_val_f, -ent_b0[val_mask_rows].reshape(-1), mask_val_f)
        thr, prec, cov = precision_at_coverage(yc_val_f, p_val, mask_val_f, 0.20)
        commit_aurocs.append(a); commit_aurocs_ent.append(a_ent); commit_aurocs_b0.append(a_b0)
        commit_precisions.append(prec); commit_coverages.append(cov)
        print(f"  fold {f}: COMMIT auroc(model)={a:.4f} auroc(ent_final)={a_ent:.4f} "
              f"auroc(ent_b0)={a_b0:.4f}  precision@{cov*100:.0f}%cov={prec:.4f}")

        # ---- STOP ----
        Xs_tr, ys_tr = Xs_full[tr_mask_rows], stop_y[tr_mask_rows]
        Xs_val, ys_val = Xs_full[val_mask_rows], stop_y[val_mask_rows]
        net_s = TinyTemporalNet(Fs, K_B, "stop", seed=SEED + 100 + f)
        net_s.bce_fit(Xs_tr, ys_tr, epochs=epochs)
        p_stop_val = net_s.predict(Xs_val)   # (n_val, K_B)
        key_val, q_val = key[val_mask_rows], q[val_mask_rows]
        judge_val = stop_judge[val_mask_rows]
        last_val = last_label[val_mask_rows]
        learned_kb = np.array([int(np.argmax(p_stop_val[r] >= 0.5)) if (p_stop_val[r] >= 0.5).any()
                               else K_B - 1 for r in range(len(key_val))])
        # the learned rule's "correct": at the breath it stops on, judge_pass&key_match was the TRAIN
        # target -- but at EVAL time we don't get to re-ask the solver, so correctness is read off
        # the SAME per-breath stop_y (ground truth) at the learned stop breath, matching "did the
        # perceiver stop where the key actually agreed" (never re-consults the key for the DECISION,
        # only for scoring, exactly like the judge-alone rule's own scoring convention).
        ys_val_at_learned = ys_val[np.arange(len(key_val)), learned_kb]
        rows_model = int(ys_val_at_learned.sum())
        rows_judge = int(sum(1 for r in range(len(key_val))
                             if judge_val[r] >= 0 and ys_val[r, judge_val[r]] == 1))
        rows_last = int(sum(1 for l in last_val if l == "correct"))
        regr = int(sum(1 for r in range(len(key_val))
                       if last_val[r] == "correct" and ys_val_at_learned[r] != 1))
        stop_rows_model.append(rows_model); stop_rows_judge.append(rows_judge); stop_regr.append(regr)
        print(f"  fold {f}: STOP rows(model)={rows_model}/{len(key_val)} rows(judge-alone)={rows_judge} "
              f"rows(last-breath)={rows_last} regressions={regr}")

    print(f"\nSLICE CV SUMMARY (mean over {n_folds} folds):")
    print(f"  COMMIT: auroc(model)={np.mean(commit_aurocs):.4f}  auroc(ent_final)={np.mean(commit_aurocs_ent):.4f}  "
          f"auroc(ent_b0)={np.mean(commit_aurocs_b0):.4f}  "
          f"BAR (model >= ent_final + 0.05): {'PASS' if np.mean(commit_aurocs) >= np.mean(commit_aurocs_ent) + 0.05 else 'MISS'}")
    print(f"  COMMIT: precision@20%cov mean={np.nanmean(commit_precisions):.4f}  "
          f"BAR (>= 0.90): {'PASS' if np.nanmean(commit_precisions) >= 0.90 else 'MISS'}")
    print(f"  STOP:   rows(model) sum={sum(stop_rows_model)}  rows(judge-alone) sum={sum(stop_rows_judge)}  "
          f"regressions sum={sum(stop_regr)}")
    return dict(commit_auroc=float(np.mean(commit_aurocs)), commit_auroc_ent=float(np.mean(commit_aurocs_ent)),
               commit_precision20=float(np.nanmean(commit_precisions)),
               stop_rows_model=sum(stop_rows_model), stop_rows_judge=sum(stop_rows_judge),
               stop_regressions=sum(stop_regr))


# ===========================================================================================================
# FINAL FIT (diet slice, full -- for COMMIT; diet + gsm8k concatenated -- for STOP) + ONE wild read
# ===========================================================================================================

def run_final_and_eval(diet_npz, gsm8k_globs, eval_npz, epochs, out_txt):
    z_diet, meta_diet = load_npz(diet_npz)
    Xc, right_final = commit_features(z_diet)
    n, K_B, L, Fc = Xc.shape
    Xc_f = Xc.transpose(0, 2, 1, 3).reshape(-1, K_B, Fc)
    yc_f = np.clip(right_final, 0, 1).reshape(-1)
    mask_f = (right_final >= 0).reshape(-1)
    net_c = TinyTemporalNet(Fc, K_B, "commit", seed=SEED + 1000)
    net_c.bce_fit(Xc_f, yc_f, mask_f, epochs=epochs)

    Xs_diet, ys_diet, _, _, _ = stop_features(z_diet)
    Xs_list, ys_list = [Xs_diet], [ys_diet]
    n_gsm8k = 0
    gsm8k_files = []
    for g in gsm8k_globs:
        gsm8k_files += sorted(glob.glob(g))
    for gf in gsm8k_files:
        zg, metag = load_npz(gf)
        assert not metag["has_gold"], f"{gf}: expected a gsm8k (has_gold=False) telemetry file"
        Xg, yg, _, _, _ = stop_features(zg)
        Xs_list.append(Xg); ys_list.append(yg)
        n_gsm8k += Xg.shape[0]
        print(f"[perceiver-train] folded in gsm8k STOP telemetry {gf}: {Xg.shape[0]} rows", flush=True)
    Fs = Xs_diet.shape[-1]
    Xs_all = np.concatenate(Xs_list, axis=0)
    ys_all = np.concatenate(ys_list, axis=0)
    print(f"[perceiver-train] STOP head final fit: {Xs_diet.shape[0]} diet rows + {n_gsm8k} gsm8k rows "
          f"= {Xs_all.shape[0]} total", flush=True)
    net_s = TinyTemporalNet(Fs, K_B, "stop", seed=SEED + 1001)
    net_s.bce_fit(Xs_all, ys_all, epochs=epochs)

    lines = [f"THE LEARNED PERCEIVER v1 -- final fit (diet slice={diet_npz}, gsm8k files={len(gsm8k_files)}, "
            f"gsm8k rows={n_gsm8k}) + ONE wild read ({eval_npz})"]

    def P(s):
        print(s); lines.append(s)

    if eval_npz:
        z_wild, meta_wild = load_npz(eval_npz)
        assert meta_wild["has_gold"], f"{eval_npz}: wild telemetry should carry gold (correctness bookkeeping) -- MEASURED, never trained on (this script never calls .bce_fit on it)"
        Xc_w, right_w = commit_features(z_wild)
        nw, K_B_w, L_w, _ = Xc_w.shape
        Xc_w_f = Xc_w.transpose(0, 2, 1, 3).reshape(-1, K_B_w, Fc)
        yc_w_f = np.clip(right_w, 0, 1).reshape(-1)
        mask_w_f = (right_w >= 0).reshape(-1)
        p_w = net_c.predict(Xc_w_f)
        ent_final_w = z_wild["ent"][:, -1, :].reshape(-1)
        a_w = auroc(yc_w_f, p_w, mask_w_f)
        a_ent_w = auroc(yc_w_f, -ent_final_w, mask_w_f)
        thr_w, prec_w, cov_w = precision_at_coverage(yc_w_f, p_w, mask_w_f, 0.20)
        P(f"\nWILD COMMIT: n_slots={int(mask_w_f.sum())} auroc(model)={a_w:.4f} auroc(ent_final)={a_ent_w:.4f} "
          f"precision@{cov_w*100:.0f}%cov={prec_w:.4f}")
        P(f"  BAR commit AUROC >= entropy+0.05: {'PASS' if a_w >= a_ent_w + 0.05 else 'MISS'} "
          f"({a_w:.4f} vs {a_ent_w + 0.05:.4f})")
        P(f"  BAR commit precision@20%cov >= 0.90: {'PASS' if prec_w >= 0.90 else 'MISS'} ({prec_w:.4f})")
        P(f"  KILL commit AUROC < entropy baseline: {'KILL TRIPPED' if a_w < a_ent_w else 'clear'}")

        Xs_w, ys_w, stop_judge_w, key_w, q_w = stop_features(z_wild)
        p_stop_w = net_s.predict(Xs_w)
        learned_kb_w = np.array([int(np.argmax(p_stop_w[r] >= 0.5)) if (p_stop_w[r] >= 0.5).any()
                                 else K_B_w - 1 for r in range(nw)])
        ys_w_at_learned = ys_w[np.arange(nw), learned_kb_w]
        rows_model_w = int(ys_w_at_learned.sum())
        last_label_w = z_wild["last_label"]
        rows_last_w = int(sum(1 for l in last_label_w if l == "correct"))
        regr_w = int(sum(1 for r in range(nw) if last_label_w[r] == "correct" and ys_w_at_learned[r] != 1))
        P(f"\nWILD STOP: n_rows={nw} rows(model)={rows_model_w} rows(last-breath)={rows_last_w} regressions={regr_w}")
        P(f"  BAR stop rows >= last-breath's: {'PASS' if rows_model_w >= rows_last_w else 'MISS'} "
          f"({rows_model_w} >= {rows_last_w})")
        P(f"  BAR regressions <= 2: {'PASS' if regr_w <= 2 else 'MISS'} ({regr_w})")
        if rows_last_w == 14 and nw == 311:
            P("  ASSERT OK: wild last-breath baseline == the banked PMS8_241 mask=1 read (14/311).")
        else:
            P(f"  NOTE: wild last-breath baseline {rows_last_w}/{nw} -- compare against the banked 14/311 "
              f"read before trusting the bars above (a mismatch means this telemetry run's solver phase "
              f"disagreed with the banked chain_acc read; explain, do not silently trust).")
    with open(out_txt, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[perceiver-train] wrote {out_txt}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("diet_npz")
    ap.add_argument("--gsm8k", nargs="*", default=[], help="glob(s) for gsm8k STOP-only telemetry npz files")
    ap.add_argument("--eval", default="", help="wild telemetry npz -- ONE measurement read, never trained on")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--out", default=".cache/perceiver_v1_PMS8_241.txt")
    args = ap.parse_args()
    assert os.environ.get("DEV") == "CPU"
    np.random.seed(SEED)

    cv = run_cv(args.diet_npz, n_folds=args.folds, epochs=args.epochs)
    run_final_and_eval(args.diet_npz, args.gsm8k, args.eval, args.epochs, args.out)


if __name__ == "__main__":
    main()
