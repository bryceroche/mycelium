# Verification Head — Design Spec

**Context**: today's probes proved verification doesn't emerge naturally from π-rotation
(`project_v5_probes_findings.md`). The model treats correct and wrong candidate answers
identically in representation space (ratio 1.00 across all breaths on v4/v5/v6). To
activate the closed loop, we need to TRAIN verification explicitly.

This document specifies that training.

## Two-phase design

### Phase 1 — Frozen-model probe (cheap, 30 min)

Before building any training infrastructure, test the **prerequisite question**: does
the v6 integrated rep already CONTAIN verification information, or is it absent?

**Procedure**:
1. Take `arith_mixed_v6_step300.safetensors`, freeze.
2. Generate training set: 1000 (problem, candidate, is_correct) triples, half with
   gold answer (correct=1), half with one-digit-corrupted gold (correct=0).
3. For each triple: tokenize "problem = candidate", forward through `breathe_controlled`
   with `return_per_breath_reps=True`, extract `integrated_per_breath[-1]` at the last
   answer-token position → 1024d rep.
4. Train a 2-layer MLP (`1024 → 512 → 1`) on this dataset with BCE loss. ~520K params.
   Hold out 200 examples for eval.
5. Measure: validation accuracy distinguishing correct vs wrong.

**Interpretation**:
- **acc > 80%**: the signal IS in the rep; v6 representations encode verification
  information. Phase 2 makes sense — just need to integrate the head into the loop.
- **acc 55-70%**: signal is weak but present. Need a bigger head, or jointly
  fine-tune.
- **acc ~50%**: the rep doesn't encode verification at all. Need to train the
  TRANSFORMER to produce verification-encoding reps, which is a much bigger change.

**Cost**: ~10 min generation + 20 min MLP training on CPU/GPU. Decides whether to
invest in Phase 2.

### Phase 2 — Online verification head (only if Phase 1 succeeds)

If the rep encodes verification, integrate the head into the closed loop.

**Architecture**:
- Verification head: 1024 → 512 → 1 (BCE) — same shape as Phase 1.
- Read at the integrated rep's last-answer-position per breath.
- Output per breath: `verify_logit_l` (scalar per example).

**Training mode**: jointly trained alongside main + controller during ARITH_MIXED
training, with these losses:

1. **Main CE** (existing) — transformer learns answer prediction.
2. **Verification BCE** — verification head learns correct/wrong on labelled examples.
3. **Stop-calib (revised)** — `stop_logit_l` is supervised to PREDICT `verify_logit_l`.
   The verification head provides the "ground truth" energy signal; the controller's
   stop_logit learns to predict it. This gives the controller the closed-loop signal
   that's been missing.

**Data generation**: half of each minibatch is "verification" pairs (problem +
deliberately-wrong answer). The other half is normal training (problem alone). The
verification head only loss-applies on verification pairs. The transformer's main
training is unaffected by the wrong-answer examples (we mask them out of the main CE).

**Critical**: verification head has its own optimizer, like the controller. Gradient
separation: verification head and main transformer don't share gradient channels (we
don't want the transformer to learn to "fool" the verification head).

**Integration with controller's stop decision**:
- At training time: stop_logit calibrated against verify_logit per breath.
- At inference time: model generates candidate via existing greedy path; verification
  head scores the rep; if score < threshold, optionally re-generate with more breaths
  or different temperature. (Inference-time use is optional polish; the main win is
  using the verification signal during training to teach the controller.)

## Training data generation

Reuse existing ARITH_MIXED generators. For each problem `ex`:

```python
correct_input = tok.encode(f"{ex.problem} {ex.answer}.").ids
wrong_ans = corrupt_answer_digit(ex.answer, rng)
wrong_input = tok.encode(f"{ex.problem} {wrong_ans}.").ids
```

Labels: 1 for correct, 0 for wrong. Use `corrupt_answer_digit` from
`scripts/probe_verification.py` (one-digit corruption — close enough to feel like a
plausible mistake but unambiguously wrong).

50/50 split per minibatch keeps the head from learning class imbalance.

## Loss formulation (Phase 2)

```python
# Existing
main_loss = transformer_loss(tokens, labels)     # only on "normal" half
ctrl_aux_loss = op_classification_ce             # existing aux

# NEW
verify_logits = verify_head(integrated_per_breath[-1][:, last_pos, :])  # (B,)
verify_loss = BCE(verify_logits, is_correct_labels)                      # only on "verify" half

# Stop calibration: now supervises against verify_head's output
stop_calib_loss = 0
for l, decision in enumerate(decisions_per_breath[1:]):  # skip init
    # The verify_head's view at the FINAL breath is our "ground truth" energy.
    # stop_logit at breath l should predict it.
    target = verify_logits.detach()  # (B,)
    diff = decision["stop_logit"] - target
    stop_calib_loss += diff.square().mean()
```

**Why per-breath stop_logit targets ALL converge to the final verify_logit**: that's
the controller learning to ANTICIPATE what verification will say. Early breaths predict
poorly (high error → strong gradient on early decision_heads). Late breaths predict
well (low error → small gradient). The controller develops a "convergence trajectory"
where stop_logit climbs toward verify_logit over breaths, naturally giving us the
adaptive-stopping signal.

## Open questions

1. **Should the verification head see the rep at breath N (last) or all breaths?** Start with last; if accuracy is low, try aggregating across breaths.

2. **Should we corrupt the answer in different ways?** Single-digit corruption is the
   easiest distinction. Could also do digit-swap, magnitude-shift, off-by-1. More
   diverse negatives → more robust head.

3. **Does the verification head need its own breathing?** Probably not — it operates
   on the existing rep. But we could test a "verifier breathes once on the rep" variant
   if the simple linear-head version doesn't work.

4. **Inference-time use**: do we add a re-breathing loop if verify_logit is low? That's
   real adaptive computation but introduces variable-time inference. Defer.

## Estimated cost

- **Phase 1**: ~30 min (10 min generate data, 20 min train + eval the MLP probe)
- **Phase 2 implementation**: ~2-3 hours
- **Phase 2 training run**: ~1 hour for 500 steps on ARITH_MIXED with verification
- **Phase 2 follow-up on L4**: ~1 hour for L4 training with verification active

Total: ~5-6 hours if Phase 1 passes.

## Success criteria

- Phase 1 acc > 70%: signal exists, proceed.
- Phase 2: stop_logit within-batch std on ARITH_MIXED jumps from current 0.45 to >1.5
  (the threshold we set as "closed loop active" back during v2/v3).
- Phase 2 on L4: multi-step accuracy at A=8 exceeds the projected 0.78² = 61% — would
  confirm that explicit verification activates the closed loop and pushes performance
  past the naive multiplicative bound.

## Files this will touch

- `mycelium/breathing.py` — add `VerificationHead` class
- `mycelium/l3_training.py` — add `verify_head_train_step` + verification loss in
  `controller_train_step`
- `scripts/l3_train.py` — wire env vars (VERIFY_TRAIN, VERIFY_LR, VERIFY_WEIGHT)
- `scripts/probe_verification_mlp.py` — new — Phase 1 MLP probe
- `mycelium/l3_data.py` — add verification-pair generator
