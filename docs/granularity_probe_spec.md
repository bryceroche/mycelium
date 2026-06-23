# KenKen Granularity Probe — Design Spec

**Author:** Bryce + Claude · **Status:** spec + skeleton (not yet run on GPU)
**Script:** `scripts/diag_kenken_granularity_probe.py` (skeleton — import-clean, function
stubs, full arg parsing + probe/eval scaffolding; the GPU forward loop is stubbed).

---

## 0. One-paragraph summary

The deducer runs K breaths through 4 shared Pythia L0–L3 layers, accumulating a 1024d
residual that is the persistent factor-graph belief state. The open question is whether
that residual **stratifies by scale across breaths** — i.e. whether *coarse* (puzzle-level)
properties become decodable early and *fine* (per-cell) properties sharpen late (or vice
versa, or oscillate). If a clean coarse→fine wave exists, an **expand–collapse waist**
(compress coarse early, refine fine late) is architecturally justified. If the per-breath
decodability is flat at chance for every scale, the waist is **not** justified and the
expand–collapse story is refuted. This probe captures per-breath readout-LN reps on a
KenKen instance bank, then runs a **leak-free, by-instance cross-validated logistic AUC
probe** at three granularity scales per breath, and prints a thresholded verdict.

---

## 1. Hypothesis

The breathing residual carries information at three orthogonal **granularities**, and that
information appears at different **breaths** k = 0..K−1. Specifically, one of these holds:

1. **COARSE-EARLY + FINE-LATE wave** — global/puzzle-level structure is decodable in the
   first few breaths and per-cell structure sharpens in the last few breaths. This is the
   signature that **supports an expand–collapse waist** (the deducer first commits a coarse
   plan, then refines local detail).
2. **OSCILLATING / V-cycle** — coarse decodability peaks at k=0, dips at k≈K/2, peaks again
   at k=K−1 (with fine/medium showing the inverse). This is the signature that supports an
   **oscillating waist** (multigrid V-cycle: restrict to coarse, relax, prolong to fine).
3. **FLAT (null)** — no scale is decodable above chance at any breath; the residual does not
   stratify by scale. This **refutes** coarse-early, fine-late, and V-cycle and says a waist
   is not justified by the dynamics.

The probe measures decodability per breath per scale and routes to exactly one of these
verdicts (plus a WEAK/INCONCLUSIVE fall-through) using stated thresholds (§5).

The probe also runs a **fresh radial-depth control** (§5.5): it correlates the per-instance
coarse-decodability slope (AUC_final − AUC_0) with each instance's `deduction_depth`. If the
slope is uncorrelated with depth, the radial-depth axis is orthogonal to the granularity
axis — a distinct claim from the refuted radial-depth deep-prize, re-tested here from scratch.

---

## 2. The three granularity scales (features)

At each breath k we decode three label families from the captured residual. COARSE is decoded
from the **pooled** (B, H) per-breath rep (mean over valid cells); MEDIUM and FINE are decoded
from the **un-pooled** (B, S, H) per-cell rep, masked by `cell_valid`.

| Scale | Label (predicted) | Feature source | Encoder |
|---|---|---|---|
| **COARSE / GLOBAL** | puzzle size N (categorical 3..7); difficulty band (g40/g30/g20/g10); total #cages; #givens | pooled readout-LN rep `(x_ln * cell_valid).sum(1) / cell_valid.sum(1)` → (B, H) | logistic / ridge on (K·B, H) |
| **MEDIUM / REGIONAL** | cage ID per cell (multiclass: which of ~10–20 cages); constraint type | per-cell readout-LN rep masked by `cell_valid` → one (S, H) per instance | logistic on (K·B·S_valid, H), grouped by instance |
| **FINE / LOCAL** | per-cell gold value (categorical 1..N, or `is_filled` vs unknown); per-cage arithmetic-satisfaction (binary) | same per-cell reps, optionally cage-aggregated (mean per cage) | logistic on (K·B·S_valid, H) |

**Decodability metric** per breath k and per scale: held-out AUC (logistic, by-instance CV;
ridge for continuous targets). The **trend across k** is what drives the verdict, not the
absolute level.

For multiclass labels (cage ID, value, N), AUC is computed via a one-vs-rest reduction or a
balanced binary recoding (e.g. value > median, "is this the modal cage", or the dominant
binary contrast) so the same Mann-Whitney `auc_mann_whitney` machinery applies unchanged. The
binary recoding used is printed in the table header so it is auditable.

---

## 3. Capture points

Per-breath capture requires intercepting the **final readout layernorm** inside
`factor_breathing_forward` (`mycelium/factor_graph_engine.py`). Each breath calls
`_layernorm(x, model.ln_f_g, model.ln_f_b, eps)` exactly once (engine line ~478,
`x_ln = _layernorm(...)` before the value-codebook projection), so a hook that records on
`gamma is model.ln_f_g` and **appends** (rather than overwriting) collects exactly K reps.

### 3.1 Readout-LN hook (mirrors `_DartCapture`)
- Installed at the final readout LN, per breath k, **before** the value-codebook projection.
- Captures the per-breath (B, S, H) hidden state.
- Monkeypatch `mycelium.breathing._layernorm` (non-invasive, eager-only — the engine imports
  `_layernorm` locally inside the forward, so patching the module symbol intercepts the
  readout call). The engine + oracle stay git-clean.
- **Difference from `_DartCapture`:** `_DartCapture` keeps only the LAST breath (overwrites a
  single slot). This probe **appends every breath** into a list → K × (B, S, H), so the
  per-breath axis is preserved.

### 3.2 Per-cell residual capture
- COARSE pools the captured (B, S, H) over valid cells → (B, H).
- MEDIUM/FINE use the **un-pooled** (B, S, H) directly, masked by `cell_valid`.
- Both come from the same `x_ln` readout rep — no second forward needed. The un-pooled slot
  is stored per instance so per-cell / per-cage aggregation happens offline.

### 3.3 Waist d-rep capture (optional, FG_WAIST)
- When a waist is attached (`model.fg_waist_down is not None`), the per-breath d-rep from the
  bottleneck is available in the forward loop (engine line ~463,
  `waist_drep_history.append(d_rep)`), and is also exposed via the `model.fg_waist_capture`
  list on the eager re-eval path. Reuse that mechanism to probe the bottleneck rep directly.
- Optional; the default probe runs on the 1024d readout rep (no waist required).

### 3.4 Storage contract
- Per-breath: K × (B, S, H) tensors stacked into (K, B, S, H); for the pooled probe, reduce
  to (K, B, H). Flatten to (K·B, …) for the probe, tag by breath index in metadata.
- All captures realized to numpy fp32 (or fp16 for the per-cell bottleneck to fit the size
  budget) eagerly inside the hook.

---

## 4. Leak-free methodology

Reuses `dart_cluster_probe.py` helpers **unchanged** so the protocol is identical to the
validated cluster-probe:

1. **By-instance CV split (FIX 1).** Whole instances are assigned to folds via
   `assign_instance_folds` (the exact by-instance round-robin from `learned_waist_gate.py` /
   `dart_cluster_probe._cv_logistic_auc`). No cell or breath of a train-instance ever appears
   in the held-out fold. Prevents residual per-instance structure from leaking into test.
   `folds=5` default (3 if n_inst < 15).

2. **Per-breath probe loop.**
   ```
   FOR k in 0..K-1:
     FOR fold in 0..folds-1:
       train = reps[fold_id != fold, k, :] ; test = reps[fold_id == fold, k, :]
       mu, sd = train.mean(0), train.std(0).clip(1e-8)        # standardize on TRAIN stats
       train, test = (train - mu)/sd, (test - mu)/sd
       w, b = _logreg_fit(train, train_labels, l2=1.0)         # reuse DCP._logreg_fit
       scores = test @ w + b
       auc = auc_mann_whitney(scores, test_labels)             # reuse DCP.auc_mann_whitney
     auc_per_breath[k] = mean(auc over folds)
   ```
   The per-cell scales (MEDIUM/FINE) flatten (B, S, H) → (B·S, H), keep only valid cells, and
   propagate a per-cell instance id (`np.repeat(inst_id, S)[valid_mask]`) so the by-instance
   fold split still holds at the cell level.

3. **Shuffle null.** The same probe is run on **shuffled labels** (independent of reps);
   expect AUC ≈ 0.5 at every k. Detects an over-loose CV split or an artifact of the rep
   space. The null curve is printed alongside the real curve.

4. **Outputs.**
   - Table: `breath k | scale (COARSE/MEDIUM/FINE) | mean_auc | p25 | p75 | trend`.
   - Optional plot: decodability (AUC) vs breath k, one line per scale (+ null).

Reused unchanged: `auc_mann_whitney`, `_logreg_fit`, `_center_by_instance`,
`_cv_logistic_auc` (DCP); `assign_instance_folds` (the by-instance split). New code: the
per-breath capture loop, per-breath pooling, the 3-scale table builder, the
decodability-vs-breath verdict, and the radial-depth correlation check.

---

## 5. Decision rule (thresholds stated, not hidden)

Let `auc[scale][k]` be the per-breath held-out AUC. The verdict is one of:

### 5.1 COARSE-EARLY + FINE-LATE (CONFIRMED)
- COARSE peaks early: `mean(auc[COARSE][:K/2]) > 0.65` **and** `mean(auc[COARSE][K/2:]) < 0.60`.
- FINE peaks late: `mean(auc[FINE][K/2:]) > 0.65` **and** `mean(auc[FINE][:K/2]) < 0.60`.
- **VERDICT:** coarse-first / fine-late wave → **expand–collapse waist JUSTIFIED** (compress
  coarse early, refine fine late).

### 5.2 V-CYCLE / OSCILLATING (CONFIRMED)
- `max(auc[COARSE][:K/3]) > 0.65` **and** `min(auc[COARSE][K/3:2K/3]) < 0.55` **and**
  `max(auc[COARSE][2K/3:]) > 0.65` (peak–valley–peak in coarse, with medium/fine inverse).
- **VERDICT:** V-cycle / multigrid → **oscillating waist JUSTIFIED**.

### 5.3 FLAT (REFUTED)
- `|auc[scale] − 0.5|.max() < 0.05` for **all** scales **and** all breaths, and the shuffle
  null is indistinguishable.
- **VERDICT:** no scale stratification → coarse-early, fine-late, and V-cycle are all
  **REFUTED** → do **not** build the waist.

### 5.4 WEAK / INCONCLUSIVE (fall-through)
- A signal above chance somewhere but not clearing the bars above → mixed/weak; suggestive,
  not decisive.

### 5.5 RADIAL-DEPTH CONTROL (a distinct claim, re-tested fresh)
- Compute per-instance COARSE decodability **slope** = (AUC at k=K−1) − (AUC at k=0).
- Correlate that slope against `deduction_depth` per instance (Pearson ρ; clean shuffle-null
  on the depth labels for the bar).
- If `|ρ| < 0.30` (real spread, clean null) → the depth axis is **orthogonal** to the
  granularity axis; the breath allocation is not depth-ordered (consistent with the refuted
  radial-depth deep-prize — re-confirmed independently here).
- If the coarse slope is flat or negative AND depth shows no correlation, the depth axis is
  **not** the explanatory variable.

**Output format:** a printable verdict line + the data table (k, scale, auc, std, p25, p75) +
the radial-depth ρ with its shuffle-null bar.

---

## 6. Reuse map

**Reused unchanged**

- `scripts/dart_cluster_probe.py`: `auc_mann_whitney`, `_logreg_fit`, `_center_by_instance`,
  `_cv_logistic_auc` (the by-instance CV held-out AUC).
- `scripts/learned_waist_gate.py`: `assign_instance_folds` (the whole-instance fold split).
- `scripts/amortized_frontier_measure.py`: the `_DartCapture` readout-LN monkeypatch pattern
  (`install`/`uninstall`/`arm`, gamma-IS-`model.ln_f_g`); reused as a *pattern* (the probe's
  hook **appends per breath** rather than overwriting).
- `mycelium/kenken_data.py`: `KenKenLoader`, `encode_puzzle`, `KenKenBatch`, `N_MAX`,
  `N_CELLS` (instance builder + the per-puzzle metadata: `N`, `band`, `deduction_depth`,
  `cell_cage_id`, `gold`, `cell_valid`).
- `mycelium/factor_graph_engine.py`: `FactorGraphSpec`, `make_kenken_factor_batch`,
  `factor_breathing_forward`, `attach_factor_graph_params`.
- `mycelium/kenken.py`: `build_verification_inlet` (KenKen sets `has_factor_inlet=True`).

**New (this script)**

- Per-breath capture loop (append every breath, not K independent forwards).
- Per-breath pooling (mean over valid cells per breath, stored separately).
- The COARSE/MEDIUM/FINE 3-scale table builder.
- The decodability-vs-breath verdict + optional plot.
- The fresh radial-depth correlation check (depth ⟂ granularity).

**Caveat (honest):** `_build_deducer_model` in `amortized_frontier_measure.py` is
**coloring-specific** (it imports `scripts.search_coloring` + `GraphColoringLoader`, and the
`--domain kenken` hook raises `NotImplementedError`). KenKen also requires `has_factor_inlet=
True` (the verification inlet). The skeleton therefore provides a **KenKen model-builder
stub** (`build_kenken_deducer_model`) that mirrors the validated build path
(`load_breathing` → `cast_layers_fp32` → `attach_factor_graph_params(spec)` →
`load_ckpt`) but for the KenKen spec. **No KenKen factor-graph deducer checkpoint exists yet**
(per CLAUDE.md §5) — `FG_TASK=kenken` on `scripts/factor_graph_train.py` would produce one
with no new engine code. The probe runs against that ckpt once trained.

---

## 7. Invocation (when a KenKen ckpt exists; Bryce/main-thread fires GPU runs)

```
DEV=AMD \
FG_CKPT=.cache/fg_ckpts/fg_kenken_k16/fg_kenken_k16_final.safetensors \
.venv/bin/python3 scripts/diag_kenken_granularity_probe.py \
    --bands g40,g30,g20,g10 --per-band 50 --K 16 --eval-batch 8 --folds 5 --seed 0
```

Args: `--bands` (default `g40,g30,g20,g10`), `--per-band` (50), `--ckpt`
(`$FG_CKPT` or `.cache/fg_ckpts/fg_kenken_k16/...`), `--K` (16), `--eval-batch` (8),
`--folds` (5), `--seed` (0).

**Output:** the (k, COARSE AUC, MEDIUM AUC, FINE AUC) table, the verdict text, the
radial-depth ρ, and (optionally) a matplotlib plot.

**GPU note:** AMD, **eager forward** (no JIT) — the readout-LN monkeypatch requires the eager
path; the JIT-unrolled K-graph would not surface per-breath `_layernorm` calls to the hook.
The probe is read-only on a trained ckpt; it fires no training. Agents do **not** run the GPU
job — the skeleton is import-clean only (`python -c "import ast; ast.parse(open(...).read())"`
passes) and ships the structure for Bryce to run.
