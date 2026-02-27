# C4/C5/C6 Training Todo

All models: Qwen-0.5B-Instruct, frozen backbone, classification head only. Training data in S3 at s3://mycelium-data/. Scripts in train/ directory.

---

## C4 — Bridging Detector

**What it does:** Predicts whether an implicit operation (unit conversion, algebraic rearrangement) is needed between steps.

**Data:** 117 files from S3. ~583 samples per file. Two classes: BRIDGING (~2.6%) and NOT_BRIDGING (~97.4%).

**Critical:** Extreme class imbalance. Use weighted cross-entropy with BRIDGING upweighted ~19x. Without this, model predicts NOT_BRIDGING for everything.

**Config:** Frozen backbone, classification head only, fp16 if driver supports it, num_workers=4, lr=1e-3, 3-5 epochs.

---

## C5 — Dependency Resolver

**What it does:** For each pair of operation steps, predicts whether step_i's result feeds into step_j. Binary: DEPENDS or INDEPENDENT.

**Data:** 117 files from S3. Built using structural detection — if step_j uses a number equal to step_i's computed result, that's DEPENDS. Mathematically exact, not heuristic.

**Watch for:** Likely imbalanced (most pairs are INDEPENDENT). Use weighted cross-entropy. Check class distribution before training and set weights accordingly.

**Config:** Frozen backbone, pairwise classification head, fp16, num_workers=4, lr=1e-3, 3-5 epochs. Input format: concatenated (step_i text [SEP] step_j text).

---

## C6 — Answer Type Classifier

**What it does:** Predicts what format the final answer should be (INTEGER, FRACTION, RADICAL, SET, EXPRESSION, etc.).

**Data:** 117 files from S3. Multiple answer type classes.

**Critical:** This model has failed twice already.
- First attempt: predicted INTEGER for everything (majority class)
- Second attempt: aggressive weights overcorrected to RADICAL only
- **Fix: clip class weights at max 5x.** No class gets more than 5x the base weight.

**Config:** Unlike C4/C5, C6 needs more capacity. **Unfreeze last 4 transformer layers** + classification head. lr=2e-5 for unfrozen layers, lr=1e-3 for head. fp16, num_workers=4, 3-5 epochs.

**If it still collapses to one class:** Try sqrt(inverse_freq) weights instead of clipped inverse_freq. Or focal loss (gamma=2).

---

## General Notes

- All trained models save to local disk then upload to s3://mycelium-data/models/
- Save all epoch checkpoints (not just final) — we pick best by validation metric
- C4/C5: pick checkpoint by F1 on minority class, not accuracy
- C6: pick checkpoint by macro F1 across all answer types
- If frozen backbone gets <80% on any model, unfreeze last 4 layers with lr=2e-5 as fallback
