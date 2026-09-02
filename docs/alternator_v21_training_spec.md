# Alternator v2.1 Training Spec — Per-Step Ping-Pong Under JIT
*(2026-09-02; word given "we'll build this shortly." Companion to the
v2.1 ledger entries of 2026-09-02 and rotational_bus.md §6.)*

## 0. The unlock

**The dual-terminal contract makes solver pings gradient-transparent.**
Facts enter the forward pass as detached constants (dL/dp through the
solver ≡ 0 by law), so no gradient ever crosses a ping. The seams that
normally make interleaved-CPU BPTT painful are free passes for
autodiff: the only tensor crossing a seam backward is dL/dstate_k.

## 1. The execution graph (training, 7 pings/cycle)

A ladder of seven identical fused segments, walked twice.

**Forward — 7 dispatches of ONE capture:**
```
jit_step_fwd(state_k, fact_buf_k, breath_idx_k) -> (state_{k+1}, commits_k)
```
- The four-station step (gather / relate / integrate / compress-commit),
  weight-tied; breath index as an input, never a re-capture.
- Between dispatches, the CPU stub at each seam: realize commits ->
  decode confident slots (theta = 0.9; the Blackbird panel gap — early
  breaths SHOULD leak few facts) -> gac ping (alternator_bridge, the
  organ) -> pack facts into a fixed (B, 24, 4) buffer -> assign-in-place
  as the next segment's leaf constant.
- Bank per step: entry state s_k (GPU-resident) + fact_buf_k. Seven thin
  tensors, not seven activation sets.

**Backward — 7 dispatches of a second capture, k = 6 .. 0:**
```
jit_step_fwdbwd(state_k, fact_buf_k, breath_idx_k, grad_out) -> grad_in
```
- Checkpointed BPTT: recompute segment k's forward inside the JIT,
  backprop it immediately, ACCUMULATE param grads into fixed buffers
  (assign-add idiom), hand dL/dstate_k down the ladder.
- fact_buf_k re-enters as the same constant it was — bit-identical
  recomputation, no gradient into it (the contract, enforced by
  construction).

**Then one optimizer capture.** Three captures total for the whole
regime — step-fwd, step-fwdbwd, optimizer — independent of depth.

## 2. Prices (stated before build, per protocol)

- Compute: ~2x forward (the checkpointing tax) -> ~3x today's step all-in
  with the 7 sync barriers and 7 CPU decodes. Wall-clock measured at the
  smoke rung before any verdict rides on it.
- Memory: O(1) in depth (states + fact bufs only) -> the mega-batch
  grows to fill the card; target B = 64-256.
- Overlap: pings for batch N pipeline into batch N+1's GPU segment (the
  banked prefetch-gain pattern). At 0.13 ms/ping the solver hides
  entirely inside the barriers.

## 3. Losses and terminals

- All existing losses unchanged, applied at the final step's emissions
  (and any mid-step emission losses ONLY if separately registered).
- No solver-imitation loss, ever. No supervision of any diagnostic
  (Goodhart fence; the per-step matryoshka radius is diagnostic
  register only — free to READ at every seam, never in a loss).
- Commit adapter runs the emission heads on intermediate states each
  step: 7x emission-head compute, included in the ~3x price.

## 4. Bring-up ladder (the law; rungs pinned)

1. **Equivalence:** step-partitioned forward with pings DISABLED must
   reproduce the fused-graph forward bit-identically (same ckpt, same
   inputs, np.array_equal on all emission outputs). Backward rung:
   step-partitioned fwdbwd grads must match fused-graph grads within
   float tolerance on a fixed micro-batch (tolerance pinned at
   measurement time, before comparison).
2. **Smoke:** 300 steps, step-time + NaN guard + fact-rate log
   (facts/step/breath — the leak gauge; expect the Blackbird profile:
   near-zero at breath 1, rising by breath 5-6).
3. **Twin:** per-step-ping training vs cycle-seam training (single-bit
   delta: the ping schedule), warm-continued from alt21cand241's basin
   (the right-basin economics: continuation, not rebirth — 6-10k steps).
4. **Fleet:** wild holdout fac-exact + artery reads, matched envs,
   2-seed law for any promotion claim.

READ-FIRST REMAINS THE GATE: the step-partitioned READ engine (no
backward) runs on the banked alt21cand241 first; the trainer above is
built ONLY if the seven-ping read pays on a cycle-trained head.

## 5. The horizon (v3, registered not designed)

Tensorized GAC: domain masks as GPU bitvectors, propagation as a fixed
number of tensor-op iterations INSIDE the fused graph, stop-gradient
where the contract demands. One seamless capture, zero CPU stubs — the
"internal fused op" (Modular correspondence). The June engine's ghost
with a law degree: real GAC in tensor clothing, exact by construction,
not a network imitating deduction. Fires only if the ping count or
batch scale makes the CPU stub the measured bottleneck.

## 6. The picture

The execution graph is the blog's braid rendered in silicon: seven
golden segments chained in a ring, a solver knot at every seam standing
outside the differentiable world, gradient flowing back around the
braid — use earned, never enforced.
