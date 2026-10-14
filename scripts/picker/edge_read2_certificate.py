"""edge_read2_certificate.py -- READ 2: THE THREE-VIEW EDGE CERTIFICATE (JSD).

Delegated task (2026-10-12, zero-GPU parser-head reads). For every (relation slot i, candidate
variable slot j) pair on the same 64 wild rows (.cache/wild_admitted_holdout.jsonl) used throughout
this read's sibling (edge_read1_transfer.py) and the whole gym lineage, three independent views of
"does the args pointer's attention on candidate j agree with the reading / the mixer":

  (a) THE POINTER'S OWN VIEW: softmax(args_logits[i, :]) (the bilinear pointer's own distribution
      over the 24-wide variable space; from a plain forward() pass -- no replay needed, this is
      the ordinary final decode).
  (b) THE MIXER'S VIEW: relation slot i's row of the single-head mixer's post-mask/post-bias
      self-attention (_at2, the "mixer_attn1" tap) over slots -- captured on branch `shuffletap`
      (worktree /home/bryce/mycelium-wt19, forked off gen-weights 2026-10-10; merged forward with
      gen-weights at fd6c34a2 so the gym/replay() machinery is present there, per the delegated
      task's own recipe) via scripts/picker/edge_mixer_tap.py (a one-off, NOT committed -- "code
      for the taps only on that branch"), written to .cache/edge_mixer_rows_PMS8_241.npz (breaths
      1..6, (6, 64, 24, 24)). TWO BREATHS REPORTED, as the task asks: the LAST breath (6) and the
      MEAN over breaths 3-6.
  (c) THE READING OVERLAP: for slots i and j, 1 - JSD(token_dist(i), token_dist(j)) where
      token_dist(s) is slot s's breath-0 slot->token cross-attention row (the banked membrane
      census .cache/membrane_raw_wild_PMS8_241.npz's `fat_mass[:, 0, :, :]`, real tokens only via
      `tokmask`, renormalized) -- "two slots that read the same words" as a proxy for "the pointer
      SHOULD connect them."

All three are restricted to the SAME support (the row's present slots other than i) and renormalized
there, so the three pairwise JSDs (scipy's jensenshannon distance, base=2, bounded [0,1]; "JSD"
throughout this script means this bounded distance, not the raw divergence) are computed on
comparable distributions. The per-decision CERTIFICATE = the mean of the three pairwise JSDs (low =
the three views agree). AUROC of the certificate (and of each pairwise JSD alone, and of the
pointer's own entropy, the baseline "hydrometer") against whether the FINAL decode's args pointer is
CORRECT on that decision (custody gold via dn_common.score_row_slots).

BAR (pinned, from the delegated task): certificate AUROC >= 0.75. Also reports precision at 20%
coverage (sorted by certificate ascending = most agreement = most confident).

Outputs: .cache/edge_read2_certificate_PMS8_241.json / .txt. DEV=CPU; .cache/gpu.lock never touched;
reads only pre-existing banked artifacts plus the one new mixer-tap npz.
"""
import os
import sys
import json
import time

os.environ.setdefault("DEV", "CPU")
assert os.environ.get("DEV") == "CPU"

sys.path.insert(0, "."); sys.path.insert(0, "scripts"); sys.path.insert(0, "scripts/picker")
import numpy as np
from scipy.spatial.distance import jensenshannon
from sklearn.metrics import roc_auc_score

from adaptive_stop import FAMILY_ENVS
FAM = dict(FAMILY_ENVS["PMS8_241"])
FAM["DEV"] = "CPU"
for k, v in FAM.items():
    os.environ[k] = v

from dn_common import load_gold, score_row_slots
from welford_atlas import gold_kind

N_ROWS = int(os.environ.get("EDGE_NROWS", "64"))
CKPT = ".cache/sharp_PMS8_241.safetensors"
MIXER_NPZ = ".cache/edge_mixer_rows_PMS8_241.npz"
MEMBRANE_NPZ = ".cache/membrane_raw_wild_PMS8_241.npz"
OUT_JSON = ".cache/edge_read2_certificate_PMS8_241.json"
OUT_TXT = ".cache/edge_read2_certificate_PMS8_241.txt"


def jsd(p, q):
    p = np.clip(p, 1e-12, None); p = p / p.sum()
    q = np.clip(q, 1e-12, None); q = q / q.sum()
    return float(jensenshannon(p, q, base=2))


def renorm(v, idx):
    sub = np.clip(v[idx], 0.0, None)
    s = sub.sum()
    return sub / s if s > 1e-12 else np.full_like(sub, 1.0 / len(sub))


def entropy(v):
    v = np.clip(v, 1e-12, None); v = v / v.sum()
    return float(-np.sum(v * np.log(v)))


def main():
    t0 = time.time()
    print(f"[edge-read2] PMS8_241, {N_ROWS} wild rows, DEV={os.environ['DEV']}", flush=True)
    from tinygrad import Tensor, dtypes
    from tinygrad.nn.state import safe_load
    import phase1_algebra_head as H
    from phase1_algebra_head import (build_params, forward, load_alg, build_slot_masks,
                                     alt2_fact_buf, K_VARS, L_FAC)

    p = build_params(0)
    sd = safe_load(CKPT)
    assert set(sd) == set(p), (sorted(set(sd) - set(p))[:4], sorted(set(p) - set(sd))[:4])
    for k in p:
        p[k].assign(sd[k].to(p[k].device).cast(p[k].dtype)).realize()

    vg = load_gold(".cache/phase1_alg_states_wildhold.npz")
    vs, vst_, vtk, _vg2, vse = load_alg("test")
    n = min(N_ROWS, len(vs))
    sl = np.arange(n)
    ts = Tensor(np.ascontiguousarray(vst_[sl]), dtype=dtypes.half)
    tk = Tensor(vtk[sl].astype(np.float32), dtype=dtypes.float)
    se = Tensor(vse[sl].astype(np.int32), dtype=dtypes.int)

    o0 = forward(p, ts, tk, se)
    onp0 = {k: o0[k].realize().numpy() for k in ("fat", "args", "res")}
    mk = build_slot_masks(onp0, se.numpy().astype(np.int32))
    slot_mask = Tensor(mk, dtype=dtypes.float)
    _ka = ("pres", "ftype", "op", "dig") + (("dup",) if "dup" in o0 else ())
    _oa = {**onp0, **{k: o0[k].realize().numpy() for k in _ka}}
    _nv = np.array([vs[int(i)].get("n_vars", K_VARS) for i in sl])
    _ma = np.array([vs[int(i)].get("m", 0) for i in sl])
    fb = alt2_fact_buf(_oa, se.numpy().astype(np.int32), _nv, _ma)
    fact_t = Tensor(fb, dtype=dtypes.float)

    o = forward(p, ts, tk, se, slot_mask=slot_mask, fact_buf=fact_t)
    KEYS = ("pres", "ftype", "op", "dig", "args", "res") + (("dup",) if "dup" in o else ())
    final = {k: o[k].realize().numpy() for k in KEYS}
    args_logits = final["args"]   # (n, L_FAC, K_VARS)
    print(f"[edge-read2] final decode computed ({time.time()-t0:.0f}s so far)", flush=True)

    # ---- (b) mixer tap (shuffletap worktree artifact) ----
    mz = np.load(MIXER_NPZ)
    mix_breaths = [int(b) for b in mz["breaths"]]
    mix_attn = mz["attn1"]   # (nb, n_rows_capture, L_FAC, L_FAC)
    assert int(mz["n_rows"]) >= n, (int(mz["n_rows"]), n)
    mix_attn = mix_attn[:, :n]
    b_last_idx = mix_breaths.index(max(mix_breaths))
    mean36_idx = [mix_breaths.index(b) for b in (3, 4, 5, 6) if b in mix_breaths]
    mix_last = mix_attn[b_last_idx]                       # (n, L_FAC, L_FAC)
    mix_mean36 = mix_attn[mean36_idx].mean(axis=0)         # (n, L_FAC, L_FAC)
    print(f"[edge-read2] mixer tap loaded, breaths={mix_breaths} ({time.time()-t0:.0f}s so far)", flush=True)

    # ---- (c) reading overlap: breath-0 slot->token attention ----
    bz = np.load(MEMBRANE_NPZ)
    assert bz["fat_mass"].shape[0] >= n
    fat0 = bz["fat_mass"][:n, 0, :, :].astype(np.float64)   # (n, L_FAC, T)
    tokmask = bz["tokmask"][:n].astype(bool)                 # (n, T)
    tokdist = np.zeros_like(fat0)
    for i in range(n):
        m = tokmask[i]
        for j in range(L_FAC):
            v = fat0[i, j] * m
            s = v.sum()
            tokdist[i, j] = v / s if s > 1e-12 else np.where(m, 1.0 / max(m.sum(), 1), 0.0)
    print(f"[edge-read2] reading-overlap token distributions built ({time.time()-t0:.0f}s so far)", flush=True)

    # ---- per-row gold facts: present slots, kind ----
    present_slots = {}
    slot_kind = {}
    for i in range(n):
        facs = vs[i]["factors"]
        pj = []
        for j, fac in enumerate(facs):
            if j >= L_FAC:
                continue
            slot_kind[(i, j)] = gold_kind(fac, j)
            pj.append(j)
        present_slots[i] = sorted(pj)

    # ---- build per-decision records ----
    recs = []   # dict(row,slot,y_correct, cert_last, cert_mean36, jsd_ab_*, jsd_ac, jsd_bc_*, entropy)
    for i in range(n):
        s_base = score_row_slots({k: final[k][i] for k in KEYS}, vg, i)
        for j in present_slots[i]:
            if not slot_kind[(i, j)].startswith("rel:"):
                continue
            rec = s_base.get(j)
            if rec is None or not rec["is_rel"]:
                continue
            cands = [c for c in present_slots[i] if c != j]
            if len(cands) < 2:
                continue   # JSD needs >=2-wide support to be meaningful
            idx = np.array(cands)
            a_dist = renorm(_softmax(args_logits[i, j]), idx)
            b_last = renorm(mix_last[i, j], idx)
            b_mean = renorm(mix_mean36[i, j], idx)
            c_raw = np.array([1.0 - jsd(tokdist[i, j], tokdist[i, c]) for c in cands])
            c_dist = c_raw / c_raw.sum() if c_raw.sum() > 1e-12 else np.full(len(cands), 1.0 / len(cands))
            ent = entropy(a_dist)
            for tag, b_dist in (("last", b_last), ("mean36", b_mean)):
                jab = jsd(a_dist, b_dist); jac = jsd(a_dist, c_dist); jbc = jsd(b_dist, c_dist)
                cert = (jab + jac + jbc) / 3.0
                recs.append(dict(row=i, slot=j, breath_view=tag, y=bool(rec["f_args"]),
                                 cert=cert, jab=jab, jac=jac, jbc=jbc, entropy=ent))
    print(f"[edge-read2] {len(recs)} decision x breath-view records built ({time.time()-t0:.0f}s so far)", flush=True)

    def auroc_table(tag):
        rr = [r for r in recs if r["breath_view"] == tag]
        y = np.array([r["y"] for r in rr], dtype=np.int32)
        if len(set(y.tolist())) < 2:
            return dict(n=len(rr), note="degenerate y")
        out = dict(n=len(rr), n_correct=int(y.sum()))
        for metric in ("cert", "jab", "jac", "jbc", "entropy"):
            x = np.array([r[metric] for r in rr])
            # low metric -> predicted correct, so AUROC of (-x) vs y
            out[f"auroc_{metric}"] = float(roc_auc_score(y, -x))
        order = np.argsort(np.array([r["cert"] for r in rr]))
        k20 = max(1, int(0.2 * len(rr)))
        top20 = order[:k20]
        out["precision_at_20pct"] = float(y[top20].mean())
        out["base_rate"] = float(y.mean())
        return out

    table_last = auroc_table("last")
    table_mean36 = auroc_table("mean36")

    report = dict(tag="PMS8_241", fixture="wildhold", n_rows=n,
                  mixer_breath_last=max(mix_breaths), mixer_breaths_mean="3-6",
                  table_last_breath=table_last, table_mean_breaths_3_6=table_mean36,
                  bar=dict(auroc_cert_bar=0.75))
    def _jd(o):
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        return str(o)
    json.dump(report, open(OUT_JSON, "w"), indent=1, default=_jd)

    lines = []
    L = lines.append
    L("READ 2 -- THE THREE-VIEW EDGE CERTIFICATE (PMS8_241, wildhold, n_rows=%d)" % n)
    L("")
    for name, t in (("mixer view = LAST breath (%d)" % max(mix_breaths), table_last),
                   ("mixer view = MEAN of breaths 3-6", table_mean36)):
        L(f"-- {name} --")
        if "note" in t:
            L(f"  {t['note']}")
            continue
        L(f"  n_decisions={t['n']}  n_correct={t['n_correct']}  base_rate={t['base_rate']:.3f}")
        L(f"  AUROC  certificate={t['auroc_cert']:.3f}  (a,b)JSD={t['auroc_jab']:.3f}  "
          f"(a,c)JSD={t['auroc_jac']:.3f}  (b,c)JSD={t['auroc_jbc']:.3f}  entropy={t['auroc_entropy']:.3f}")
        L(f"  precision@20%% coverage (lowest-certificate 20%%) = {t['precision_at_20pct']:.3f}")
        L("")
    bar_hit = ("auroc_cert" in table_last and table_last["auroc_cert"] >= 0.75) or \
              ("auroc_cert" in table_mean36 and table_mean36["auroc_cert"] >= 0.75)
    L(f"VERDICT: BAR (certificate AUROC >= 0.75) {'MET' if bar_hit else 'NOT MET'}.")
    txt = "\n".join(lines)
    open(OUT_TXT, "w").write(txt + "\n")
    print(txt)
    print(f"\n[edge-read2] wrote {OUT_JSON} and {OUT_TXT} ({time.time()-t0:.0f}s total)", flush=True)


def _softmax(x):
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()


if __name__ == "__main__":
    main()
