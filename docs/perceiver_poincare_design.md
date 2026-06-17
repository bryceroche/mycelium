# Perceiver-Poincaré — Generalizable Reasoning Engine (design memo)

**Branch:** `perceiver-poincare` (off `main` 2b7566b). **Status:** design — nothing fired.
The PERCEIVER revival, with the one ingredient it never had: the **Poincaré ball + the
Tier-2 `g_φ` anchor machinery** — the documented fix for the bootstrap wall that killed it.

This branch is the speculative bet. **`main` (the v98 KenKen executor) is the permanent
safety net** and stays validated/working regardless of what happens here.

---

## 0. Thesis, and why this is not refutation #6

The perceiver (latents-as-primary-state, Perceiver-IO style) is the right *shape* for a
generalizable engine: a fixed pool of latent observer/executor nodes that route into **any**
problem's variables — "a different region of the manifold per problem." v98 (per-cell
residual) is welded to the grid; the perceiver is problem-agnostic.

It was refuted **5×** (v118–v121) and v300 failed flat at chance, all for ONE cause: the
**bootstrap wall on the latent→token routing** — random-init attention to ~N positions never
learns content-dependent routing on diverse data → garbage context → chance. The fix we
validated this whole arc: **anchor the routing to a known structure, relax from there.** The
perceiver never had an anchor. The Poincaré ball gives it one. *That* is the new ingredient.

---

## 1. The factor-graph anchor (NOT hardcoded roles)

The universal language of every logic problem is the **factor graph** (variables +
constraints + topology). Latents anchor to the problem's **constraints**, never to
KenKen-specific row/col/cage:

```
z_latent = closed_form_base(constraint) + g_φ(cells_in_constraint)     # g_φ zero-init
```

`g_φ` is the **Stage-2 DeepSets encoder** (permutation-invariant over a constraint's
cell-set) — already built + validated. For KenKen the constraints *are* row/col/cage → this
**organically recreates the partitioned brain at t=0**. For a GSM8K DAG, latents land at
sub-computation nodes. For an alien puzzle, at its factors. **Zero grid-logic in the code —
the factor graph is the input.** The DAG pivot is a *parameter* (feed a different graph), not
a rewrite. (This is the paradigm shift: anchor to constraints, not roles.)

---

## 2. Co-embedded geodesic engine

Cells AND latents are points in **one** Poincaré ball:
- **Cells** = the raw problem nodes (placed by their factor-graph role).
- **Latents** = floating observer/executor nodes (placed by their constraint, §1).

`READ = d_hyp(z_latent, z_cell)` — **spatial attention and hierarchical depth become one
mechanism.** Global latents at the origin (widest horizon, see everything); constraint
latents out near the rim (local). Climb inward → abstract; descend → project. The geodesic
engine is only possible co-embedded.

---

## 3. KEY CLARIFICATION — the anchor is NOT a bit-exact v98 reproduction

The Tier-2 foothold matched v98 **bit-for-bit** because both were *cell→cell* masks. The
perceiver is a **different mechanism** (latent-mediated: latents read cells → think → write
back). Therefore:
- The anchor gives **sensible latent→cell routing at init** (each latent reads its
  constraint's cells) → coherent context from step 0 → the **bootstrap wall is cured**.
- But **matching v98's accuracy requires TRAINING** — the latents must *learn to deduce*,
  even with perfect routing. Brick-1 is a real training run, **not** a frozen replication.
- **Kill criterion (brick-1):** trains *off chance* + latents stay **engaged** (the prior
  perceivers' tell was `select_norm`/engagement stuck ~0 → anchored, they start engaged →
  verify they *stay*) + approaches v98 cell_acc. NOT "t=0 == v98."

The anchor removes the *routing-bootstrap* excuse. If it still flatlines, the perceiver has a
**deeper** problem and we'll have learned exactly what (an honest 6th refutation, not a
mystery). **Anchor is necessary; maybe not sufficient.**

---

## 4. The breath cycle

`READ` (d_hyp latent←cell, anchored) → `THINK` (latent self-attn) → `COMMIT` (multi-res
waist) → `WRITE` (notebook — deferred, §6).

---

## 5. Multi-resolution waist (the v99 receipt)

A single 512-d squeeze every breath tanked GSM8K (`v99_optimal_prefill`: "decode read from a
compressed 512-d waist with **no expansion**"). So:
- **Dancer (1024d):** bypasses the waist, stays in the `THINK` loop — keeps the high-freq
  arithmetic (exact digits) for the next deductive step.
- **Silhouette (512d, RMSNorm seam):** the compressed common-mode → the notebook.

Both, every breath. Never just the squeeze. (RMSNorm-before-the-waist is the substrate law
already coded in `kenken_llama.py` — reuse `_rms_norm_detached`.)

---

## 6. Notebook = hyperbolic branch-tree (DEFERRED — brick 4)

MCTS search is a **tree** → hyperbolic's native object. Slots as a hyperbolic branch-tree →
`d_hyp(branch_i, branch_j)` = branch similarity → "does branch B match a trap I mapped in A"
becomes a **geodesic query** (strictly better than Euclidean + π-RoPE, a phase-locking patch
for a problem the tree geometry doesn't have). **Build LAST** — don't build the roof before
the foundation. Note the target design; defer the build.

---

## 7. Build ladder (each anchored, each a kill criterion, each earns the next)

1. **ANCHOR TEST** — co-embed latents+cells, constraint-anchored routing (zero-init `g_φ`),
   **train**. Kill: off chance + latents engaged + approaches v98 cell_acc. *Make-or-break:
   does an anchored perceiver hold the engine at all.*
2. **RELAXATION** — unfreeze `g_φ` (dynamic constraint placement). Kill: holds/improves vs (1).
3. **MULTI-RES WAIST** — dancer + silhouette. Kill: holds vs (2) (the waist doesn't lose the
   arithmetic).
4. **HYPERBOLIC NOTEBOOK + MCTS** — last.

KenKen is the **proving substrate** (match v98 → the architecture works on a flat CSP), NOT
the goal. Then DAGs, where pooling + hierarchy actually pay.

---

## 8. Discipline (inherited, hard-won)

- **Anchor everything** (routing, waist init, notebook) to a known structure; never bootstrap
  from random. That single lesson, applied everywhere.
- **Fidelity↔trainability:** relaxation lives in the soft-mask regime (`RELAX_BLOCK_ARG≈6`) —
  a faithful mask gives vanishing gradient. Anchor sharp, relax soft.
- **Substrate laws (Tier-2):** `d_hyp` boundary clamps (|z|²≤1−1e-5, arccosh-arg≥1+1e-7),
  where()-gated NaN guard, tangent-space params (standard Adam, no Riemannian optimizer),
  coord-grad clip + tangent-norm bound (now covering `g_φ` outputs too), no `dtypes.float32`
  literal in the JIT step, finite −1e4 block.
- **Pre-registered kill criteria + baselines** (vs v98) at every brick. No instrument/metric
  shopping after the fact (the Property-2 lesson).
- **HONEST framing:** the perceiver is the project's most-refuted bet. The anchor is the new,
  evidence-based fix, but it may fail for *other* reasons (latent capacity; the waist
  info-loss — mitigated by multi-res; alt-fixed-point sharpness). This is a long-haul research
  bet *with* kill criteria, not a sure thing. A clean failure at brick-1 is a real finding,
  not a setback.

---

## Reuse map (what's already built, on `main`, that this inherits)

- `mycelium/kenken.py`: the hyperbolic mask generator (`_d_hyp_pairwise`, `_exp0_map`,
  `_relation_bias_from_z`, the simplex anchors, the **Stage-2 `g_φ` DeepSets encoder** +
  segment-mean perm-invariance + zero-init), the relaxation guards, the convergence instrument.
- `scripts/kenken_train.py`: the coord-only-optimizer + freeze + warmup + grad-clip +
  tangent-clamp + grad-norm logging harness (the validated relaxation rig).
- `mycelium/kenken_llama.py`: the 512-waist + `_rms_norm_detached` seam (for §5).
- `docs/hyperbolic_mask_generator_spec.md`: the Tier-2 spec + §8 relaxation harness + findings.
