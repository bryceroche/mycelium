# Hyperbolic Mask Generator — design spec (Poincaré, per-relation)

**Status:** design note (nothing fired). Drop-in alternative to
`build_kenken_attn_bias` (`mycelium/kenken.py:455`). Author: Bryce + Claude.

Goal: replace the *hardwired* boolean attention masks with masks **generated
from continuous coordinates** in a Poincaré ball, anchored at `t=0` to reproduce
the v98 hard mask to ~1e-3, then allowed to relax. This is the foothold for the
"virtual factor graph" / radial-traversal program. The foothold proves two
things only: (1) a coordinate field can *hold* the mask, (2) **one shared field
generalizes across N=5/6/7**. Hyperbolic geometry does NO work at `t=0` (the
mask is a flat partition); the ball is used so coordinates already live in the
space where Phase-2/3 relaxation can discover the radial hierarchy.

---

## 0. The non-negotiable structural fact: one field PER RELATION

A single metric space CANNOT hold row ∪ col ∪ cage simultaneously. Cell A=(0,0)
must be close to B=(0,1) (same row) and to C=(1,0) (same col), but B and C must
be far — and the triangle inequality forbids that (`d(B,C) ≤ d(A,B)+d(A,C)`).
v98 already resolves this by using **separate head-groups** for row/col/cage;
the geometry mirrors it exactly:

| head-group (n_heads=16, split 5/5/5/1) | coordinate field | partition |
|---|---|---|
| heads 0–4   ROW    | `z_row`  `(49, dim)`        | by row (7 groups, FIXED) |
| heads 5–9   COL    | `z_col`  `(49, dim)`        | by col (7 groups, FIXED) |
| heads 10–14 CAGE   | `z_cage` `(B, 49, dim)`     | by cage (per-instance) |
| head  15    GLOBAL | none (zero bias = all attend) | — |

`z_row`/`z_col` are FIXED (depend only on the 7×7 grid) → precompute once.
`z_cage` is per-batch (cages vary) → built on the fly, exactly like the current
`build_kenken_attn_bias` builds its per-instance cage clique.

Every v98 mask is a **partition** (each cell is in exactly one row / col / cage),
and partitions are trivially metric-realizable as disjoint clusters → **closed
form, no offline optimizer.**

---

## 1. Closed-form coordinate init (the `t=0` anchor)

For a relation with `G` groups, place `G` max-separated **anchors** on a shell of
the ball at Euclidean radius `ρ` (0<ρ<1), and put each cell on its group's anchor.

```
exp_0(v) in the Poincaré ball = tanh(|v|) · v/|v|   (origin exponential map)
=> a tangent direction `dir_g` (unit) at norm atanh(ρ) lands at μ_g = ρ · dir_g
```

So anchors are simply `μ_g = ρ · dir_g`, with `dir_g` unit vectors chosen
max-separated:
- `G ≤ dim+1`: regular **simplex** vertices (exactly equiangular, cosθ = −1/(G−1)).
- `G > dim+1`: spherical code / orthonormal-ish rows (pick `dim ≥ G−1`; for
  KenKen `G ≤ ~? cages`, `dim = 8–16` is ample).

Cell coordinate: `z_i = μ_{g(i)}`  (optionally `+ ε·u` tangent jitter, ε~1e-3, to
break exact degeneracy before relaxation; ε=0 for the pure frozen-replication
sanity).

```python
def _poincare_anchors(G, dim, rho):
    # unit directions, max-separated. simplex for G<=dim+1; else normalized rows.
    dirs = _simplex_dirs(G, dim) if G <= dim + 1 else _spherical_code(G, dim)
    return rho * dirs                       # (G, dim), |mu_g| = rho

def init_relation_field(group_of_cell, G, dim, rho, jitter=0.0):
    mu = _poincare_anchors(G, dim, rho)     # (G, dim)
    z = mu[group_of_cell]                   # (49, dim)
    if jitter: z = mobius_add(z, jitter * randn_like(z))   # stay in ball
    return z                                 # (49, dim)
```

---

## 2. The bias function + closed-form calibration of `r`, `α`

Poincaré distance (norm-clamped for the AM driver — see §4):
```
d_hyp(u,v) = arccosh( 1 + 2|u−v|² / ((1−|u|²)(1−|v|²)) )
bias(i,j)  = −softplus( α · (d_hyp(z_i,z_j) − r) )      # 0 = attend, −big = block
```

At `t=0` only two distances exist per relation:
- within-group `d_in ≈ 0` (cells coincide on their anchor),
- between-group `d_out = arccosh(1 + 4ρ²G/((G−1)(1−ρ²)²))`  (simplex, cosθ=−1/(G−1)).

To reproduce the v98 `{0, −1e4}` mask **by construction** (dial it, don't fit it):
```
r = d_out / 2
α = 2 · 1e4 / d_out
=> bias_within  = −softplus(α(d_in − r)) = −softplus(−1e4) ≈ 0
   bias_between = −softplus(α(d_out − r)) = −softplus(+1e4) = −1e4
```
Match to ~1e-3 is exact (softplus saturates). `r`, `α` start as these constants;
later phases make `r` dynamic (§5).

---

## 3. Drop-in assembly (same signature/shape as `build_kenken_attn_bias`)

```python
def build_kenken_hyperbolic_attn_bias(model, cage_ids, cell_valid):
    # returns (B, n_heads, 49, 49) additive bias, 0 allow / −1e4 block.
    n_row, n_col, n_cage, n_global = model.kenken_head_split
    Brow = _relation_bias(model.z_row, model.r_row, model.alpha_row)   # (49,49) const
    Bcol = _relation_bias(model.z_col, model.r_col, model.alpha_col)   # (49,49) const
    Bcage = _relation_bias_per_instance(                              # (B,49,49)
        init_cage_field(cage_ids, ...), model.r_cage, model.alpha_cage)
    # broadcast each relation's bias across its head slots; global head = 0.
    return _stack_heads([Brow]*n_row + [Bcol]*n_col + [Bcage]*n_cage + [0]*n_global)
```

- `z_row`, `z_col`, `r_*`, `alpha_*` are model attributes (params when learnable).
- `z_row`/`z_col` biases are CONSTANT → compute once, cache. Only the cage bias is
  per-batch (same cost profile as today).
- Call site unchanged (`kenken.py:553`). **Env-gated:** `KENKEN_HYP_MASK=1` swaps
  the generator in; off → byte-identical to `build_kenken_attn_bias`.
- Frozen + calibrated (§2) → reproduces the boolean mask → executor sees ~the same
  masks → cell_acc unchanged (the replication sanity).

---

## 4. Substrate / JIT caveats (the AM-driver landmines)

- **Boundary NaN:** `d_hyp` blows up as `|z|→1`. Clamp `|z|² ≤ 1 − 1e-5` before the
  metric; clamp the arccosh argument `≥ 1 + 1e-7`. `arccosh(x) = log(x + sqrt(x²−1))`.
- Use the established guards: **no `dtypes.float32` literal inside the JIT step**;
  `scores.clip(-1e4, 1e4)`; **where()-gated** NaN guard (not multiply-gate, NaN×0=NaN);
  scalar `isfinite`. (Per `reference_tinygrad_am_quirks` + the kenken trainer.)
- Mirror the v98 block magnitude (−1e4), not −inf, so the softmax stays finite.
- Frozen-replication path: precompute `z_row`/`z_col` biases as constants outside
  the JIT graph; only the cage field enters per-batch.

---

## 5. Phased roadmap for `r` (do NOT bundle these)

1. **Foothold — static global `r` (per relation).** Prove the geometry holds the
   mask AND one shared `z_row`/`z_col` field serves N=5/6/7. One variable.
2. **Geodesic engine — monotonic `r_k` per breath** (continuous form of the v100
   topological-staging mask): `r_k = r_0 + softplus(Σ_{j≤k} Δ_j)`, `Δ_j ≥ 0`,
   `r_0` = the hard-mask radius. Diagnostic: does learned `Δ_j` accelerate on
   deeper puzzles (radial-depth ↔ breath-count, read off the schedule)?
3. **Ultimate — `r = f(|z|)`** (horizon a function of radial position): the waist's
   inward climb auto-widens the horizon; the breath cycle becomes literal radial
   traversal. Earn it; don't start here (that's the perceiver bootstrap mistake).

**Testbed honesty:** the radial-depth bloom (Phase 3) is richest on *hierarchical*
problems (GSM8K DAGs). KenKen is flat (lateral cliques), so the N=5/6/7 foothold
cleanly proves §1–§3-static, but a muted KenKen radial signal is the geometry
faithfully reflecting a flat problem, NOT a manifold failure. Reserve the Phase-3
verdict for a DAG task.

---

## 6. Foothold test protocol (when the GPU is free)

1. **Replication sanity:** init per §1–§2 (frozen, calibrated), `KENKEN_HYP_MASK=1`,
   eval the existing v98 ckpt → per-band/per-N cell_acc must match the boolean-mask
   baseline to noise. (Confirms the generator reproduces the engine.)
2. **N-generalization (the real foothold):** does ONE shared `z_row`/`z_col` field
   serve N=5/6/7 — i.e. eval all three under a single coordinate field. (The closed
   form hands you the match for free; the *shared-across-N* question is the new bit.)
3. **Relaxation drift:** unfreeze the coordinates (small separate LR, per the
   attention-bootstrap principle — new structure needs gentle, possibly supervised
   bootstrapping), train briefly, watch: do coords stay near the anchor clusters, or
   drift toward placements that *improve* solve-accuracy? The drift is the first
   diagnostic of how the manifold wants to self-organize.

Anchor discipline throughout: the hard mask is the `t=0` slice of a continuous
family. Learning *relaxes a known geometry*, never *discovers* one from random —
that's what neutralizes the gradient void that killed the perceiver.

---

## 7. Training WITH the ball (the relaxation run)

After §6 (frozen replication + N-generalization) AND after the K=16 powered
Property-2 verdict is banked, co-train the coordinates with the executor.

**Parameterization — tangent-space, NOT Riemannian-Adam.** Storing `z` directly
in the ball requires a Riemannian optimizer (RSGD / Riemannian-Adam: Riemannian
gradient via the inverse metric, exp-map retraction) — extra machinery on
tinygrad/AM. Pragmatic alternative: keep the **learnable params in the tangent
space** (Euclidean, unconstrained, standard Adam) and map into the ball only when
computing `d_hyp`:
```
z_i = exp_0(v_i) = tanh(|v_i|) · v_i/|v_i|        # v_i is the learned param
```
The manifold constraint is enforced by the exp-map, not the optimizer → the
existing optimizer just works, no custom RSGD on the AM driver. Init `v` so
`exp_0(v)` = the §1 closed-form anchors.

**LR — small + separate on the coordinates.** The geometry is sensitive (a big
step pushes `z` toward the boundary → instability). Give the coordinate params a
**lower/separate LR** than the executor, and consider a frozen→unfreeze
graduation (two-knob pattern, v119): train the executor on frozen masks first,
then unfreeze coordinates gently. The anchor IS the bootstrap (the
attention-bootstrap principle is satisfied — coords start at the validated mask),
so a small relaxation LR is all that's needed.

**The real stability landmine — boundary gradients.** `d_hyp`'s gradient carries
the conformal factor `1/(1−|z|²)` (and arccosh's `1/√(x²−1)`), which **explode as
|z|→1**. Forward clamps (§4) protect the value; the BACKWARD needs: moderate `ρ`
(anchors not near the boundary), coordinate-gradient clipping, bounded tangent
norms, and the where()-gated NaN guard around the metric in the JIT step. This is
the thing most likely to destabilize a relaxation run — watch grad norms.

**Controlled experiment vs the frozen-hard-mask baseline.** Frozen masks =
current K=16 behavior = the safety net. Unfreeze and watch cell_acc against that
baseline:
- holds/improves → the manifold is earning its keep (continuous masks ≥ hard) →
  proceed to Phase 2 (`r_k`) / Phase 3 (`r=f(|z|)`), both of which are
  *training-only* mechanisms;
- regresses → the hard mask wins, **we keep v98 unchanged** (zero loss — the
  engine is never at risk because frozen-off is byte-identical).

So train-with-it is a *strictly additive* experiment gated behind a banked result:
the working engine stays the fallback, and the geometry only ships if it beats it.

---

## 8. Relaxation harness — the cage-structure-generalization experiment

Status (2026-06-16): the frozen foothold (§6) is DONE + CONFIRMED — CPU tensor-match
0.000, GPU eval HYP=1==HYP=0 identical, `d_hyp` JIT-compiles on the AM driver,
cross-N falls out for free. This §8 is the next experiment: unfreeze the geometry
and test whether it generalizes to UNSEEN cage configurations.

### 8.0 THE CRUX (read first — it changes what the experiment is)

**Cross-N is a unit test, not science** (a row is a row; `cell_valid` chops the 7×7
to N×N — no geometry learned). The real variance is the **cages**, and there is a
trap: **the cage field's slot-anchor parameterization generalizes PERFECTLY when
frozen but CANNOT generalize under relaxation.** At `t=0`, `cage_id → anchor[id]`
gives a correct membership clique for *any* partition (distinctness is all the mask
needs — that's why the foothold hit 0.0 on held-out cages). But `cage_id` is an
**arbitrary per-puzzle slot** (creation order), so *training* the slot-anchor table
learns slot-specific positions that **do not transfer** — a held-out puzzle maps a
different cell-set to the same slot. **Relaxing slot-anchors ≠ structural interpolation.**

Two consequences:
1. **For relaxation to generalize, cage coordinates must be a function of cage
   STRUCTURE (its cell-set), via a shared learned encoder** — not a slot lookup.
2. **The honest bar is HIGH:** the frozen clique already generalizes perfectly, so
   relaxation must learn useful *soft* structure (e.g. graded cross-cage coupling)
   that BEATS the hard clique AND transfers to held-out cages. "Ties the hard mask"
   is the null; "beats it on held-out" is the only win.

### 8.1 Three arms + capacity matching (your ask #1)

All three load the SAME banked executor (`kenken_curric_k16_cont_final`) and differ
ONLY in the mask generator:

| arm | mask source |
|---|---|
| **FROZEN (baseline)** | hyperbolic generator, coords frozen at anchors = the v98 hard mask |
| **RELAXED-HYPERBOLIC** | same coords, unfrozen, `z = exp_0(v)`, `d_hyp` |
| **RELAXED-EUCLIDEAN (control)** | identical params/shapes, unfrozen, **Euclidean** `‖u−v‖` instead of `d_hyp` (no `exp_0`) |

**Capacity match is exact by construction:** the Euclidean arm uses the *same tensors*
(`v_row`,`v_col`, the cage coord-source — same shapes, same count of trainable
scalars, same per-relation `r`/`α`); the ONLY differences are (a) drop the `exp_0`
tanh map and (b) swap `arccosh(1 + 2‖u−v‖²/((1−|u|²)(1−|v|²)))` → `‖u−v‖`. Recalibrate
`r`/`α` to the Euclidean anchor distances so the Euclidean arm ALSO reproduces the
hard mask at `t=0` (same anchor discipline). Identical DOF → any hyperbolic>Euclidean
gap is the *geometry*, not capacity. (This is the v112b attribution control: is it the
structure or just learnability?)

### 8.2 Optimizer isolation (your ask #2)

- **FREEZE the v98 backbone.** Load `kenken_curric_k16_cont_final`, set the executor
  weights non-trainable; train **only the coordinate params**. Rationale: isolates the
  geometry's contribution — a frozen competent executor means any Δ in accuracy is
  attributable to the mask, not to weights co-adapting. (Co-training muddies exactly
  the attribution we're after.)
- **Separate, small LR + warmup on coords:** a dedicated param group, LR ~1e-4 to 1e-3
  (≪ a from-scratch run), **linear warmup ~100–300 steps** from 0 (the anchor IS the
  bootstrap; ease off it gently). Two-knob graduation optional (frozen → unfreeze).
- **Boundary-gradient guards (the §7 landmine, now live):** coordinate-gradient
  CLIPPING (this is where `1/(1−|z|²)` explodes in the backward), bounded tangent
  norms (keep `|z|` away from the rim — moderate ρ already helps), the where()-gated
  NaN guard around `d_hyp`. **Lower `KENKEN_HYP_ALPHA_MARGIN` toward ~1.5–2** before
  training (the foothold's margin=4 makes the softplus near-saturated → steep boundary
  gradients; margin≥1.5 still reproduces the mask <1e-3 per the review). Watch grad norms.

### 8.3 Metric hooks — leak-free held-out cage split (your ask #3)

- **Held-out = the curriculum TEST set** (`kenken_test_curriculum.jsonl`) — already
  leak-free by D4-canonical cage STRUCTURE (overlap=0 vs train, verified). So
  "held-out cell_acc" *is* held-out-cage-structure cell_acc by construction.
- **Log, per eval, per (band × N):** `cell_acc`, `puzzle_acc`, settled-count, for ALL
  THREE ARMS on the SAME held-out puzzles/seed (apples-to-apples). Reuse the existing
  per-band eval (`evaluate()` / `_print_eval_table`); add the arm label.
- **Pre-registered binding comparison (set BEFORE looking):**
  - Win-vs-frozen: `relaxed-hyperbolic cell_acc ≥ frozen` on held-out (lower-CI of the
    paired Δ > 0), per band, leaning N=6/N=7.
  - Geometry-is-load-bearing: `relaxed-hyperbolic > relaxed-Euclidean` on held-out
    (paired Δ, lower-CI > 0). **Without this, no "hyperbolic interpolation" claim.**
  - Co-report the **deconfounded** view (the Property-2 lesson): split the held-out Δ
    by band (givens-density) and by depth so a gain isn't just "easier held-out cells."

### 8.4 Drift dump (mechanism corroboration, NOT the metric)

Checkpoint the coordinate tensors (`v_row`,`v_col`, cage coord-source) every N steps.
Inspect: do same-cage cells pull together; does any radial/hierarchical organization
emerge; is drift *structured* or noise. Drift explains a *win*; it never substitutes
for one (a beautifully-drifting manifold that doesn't beat the baseline on held-out is
a null).

### 8.5 Pre-registration + honest expectation

- Pre-register §8.3's bar and the Euclidean control BEFORE the relaxation data exists
  (no post-hoc instrument/metric selection — the Property-2 lesson).
- **Expect hyperbolic ≈ Euclidean on KenKen** (it's flat; the radial-depth payoff
  needs hierarchy). That's NOT a failure: a win over frozen = "continuous
  mask-generation generalizes" (real, useful); hyperbolic ≈ Euclidean = "geometry not
  load-bearing *here* — reserve the hyperbolic claim for a DAG testbed (§5)."

### 8.6 THE FORK needing sign-off — cage parameterization

Per §8.0, the cage coordinate source is the crux. Two stages:
- **Stage 1 (cheap, ~1 run): unfreeze the EXISTING coords** (`v_row`,`v_col` + the
  cage slot-anchor table). EXPECTED per §8.0: row/col relax (little to learn,
  near-symmetric); cage slot-anchors **fail to generalize** (in-dist maybe up, held-out
  flat/down). This empirically *demonstrates* the slot-anchor limit rather than just
  arguing it — and it's the literal "unfreeze + measure" harness. Low cost, low info,
  but it's the honest first data point.
- **Stage 2 (the real experiment): structure-based cage coordinates** — a shared
  encoder `g_φ(cage-cell-set) → tangent coord`, distinct-per-instance (cell-sets
  differ) AND generalizable (φ shared). To preserve the `t=0` anchor, run it as a
  **zero-init correction** on the slot base: `coord = anchor[id] + g_φ(cellset)`,
  `g_φ` zero-init → `t=0` = exact hard mask, then `g_φ` learns transferable structure.
  The Euclidean control mirrors `g_φ` with Euclidean output.

**Recommendation:** Stage 1 first (it's the "unfreeze" harness you asked for + proves
the slot limit cheaply), then Stage 2 (the encoder) as the actual cage-generalization
test. Sign-off needed on whether to build Stage 2's encoder now or gate it on Stage 1's
(expected) slot-anchor null.

### 8.7 RESULTS — Stage 1 (2026-06-16): clean empirical null + two findings

**FINDING A — fidelity↔trainability tradeoff (matters for ALL relaxation, incl. Stage 2).**
A *faithful* (sharp) mask gives **VANISHING** coord gradient, not exploding — a blocked
entry has attention ≈ exp(−30) ≈ 1e-13, so the loss is insensitive to its bias → ~no
gradient (raw ~1e-8). The boundary explosion we feared (1/(1−|z|²)) never materialized.
Relaxation can only happen in a **soft-mask regime**: the new `KENKEN_HYP_RELAX_BLOCK_ARG`
knob softens the block (between-group softplus arg = `margin*ARG`); `ARG≈6` opens BOTH
softplus tails (two-sided gradient) while keeping the t=0 leak <1e-3 (faithful). So:
anchor at the sharp mask (frozen, proven), but **relax soft**.

**FINDING B — slot-anchor relaxation does not generalize (the floor).** With `ARG=6`,
LR=3e-4, 600 steps, frozen backbone (bit-identical verified): coords drifted
substantially (|v−v_init| row=1.10, col=0.84, cage=0.26) and gradient flowed two-sided
(0.04–0.22, never hit the clip — GREEN throughout, guards held trivially), but
**held-out cell_acc was FLAT: 0.827 → 0.827** (puzzle 0.431 → 0.438, within noise; per-N
N5 0.95 / N6 0.87 / N7 0.74 all flat). The coords moved (fitting in-dist slot-specific
noise) but did not transfer — an arbitrary creation-order `cage_id` carries no structure
to interpolate. **This is the floor; any Stage-2 gain above 0.827 held-out is rigorous.**

Stage 1 also VALIDATED the relaxation harness under live dynamics (real two-sided
gradient + real drift, boundary guards + freeze + warmup all GREEN) — Stage 2 inherits a
proven-safe rig. NEXT = Stage 2 (`g_φ(cell-set)` encoder, zero-init correction on the
anchor base, soft `ARG≈6`) + the capacity-matched Euclidean control; bar = beat 0.827
held-out AND hyperbolic > Euclidean.
