# Organ Capacity Audit (2026-09-03)

**Scope:** read-only code audit of every trained organ named in
`docs/param_ledger.md`, plus spec-level assessment of the three newly
registered/proposed organs (mask head, step atlas, alt21 integrate
stations). Triggered by the mask-head finding (`docs/mask_head_spec.md`
§0): dynamic masking (`ALG_MASKRE`) turned out to be a REFLEX — 0
dedicated heads, 0 learned params, a hard threshold input where graded
signal existed. This audit asks the same three questions of every
other organ: COMPUTE (heads/dims), DATA (real gradient path? graded or
boolean input?), STORAGE (dedicated buffer or forced through a
bottleneck?).

**Method:** direct inspection of `scripts/phase1_algebra_head.py`
(`build_params()` = inventory, `forward()` = usage, `loss_fn`/
`_loss_single` = gradient attachment, the `TERMINALS` registry =
two-terminal enforcement), cross-checked against `docs/param_ledger.md`
and the audit trail already banked in `docs/phase1_skeleton_spec.md`
(search "THE STARVATION AUDIT", "HEAD-COUNT SCALING", "CAPACITY
BRAINSTORM", entries dated 2026-09-01 through 2026-09-03).

**Load-bearing prior finding, confirmed independently by this audit:**
the project already ran essentially this exact audit on 2026-09-01
("THE STARVATION AUDIT" entry in the ledger) and reached the same
conclusions on the compute axis for mixer/waist/pointers/notebook/
router, proposing a ~+5M (~2.5x) capacity package. That package has
NOT fired yet (no GENERATION.json change, no new ckpt referenced) —
it is registered and awaiting the word, consistent with the
hold-for-the-word discipline. This report corroborates that reading
from the code directly, adds the DATA and STORAGE axes the compute-
only brainstorm didn't cover, adds one organ the existing package
missed (macro value system), and flags one look-alike failure mode
that is NOT capacity (alt21 cold-birth).

**Deployment-status caveat:** the currently deployed parser (gen-41,
`.cache/GENERATION.json`: `ALG2=1 ALG_FTYPES=8 ALG_DUP=1 ALG_HW=512
ALG_WIDE=1`) does NOT set `ALG_BREATH>1`, so the breath loop never
runs in production — the slot mixer, notebook, router, bind MLP, and
alt21 stations are all currently INACTIVE (zero forward passes, zero
gradient) in the deployed stack. They are real, trained organs in the
research lineage (param_ledger's "genesis" config) and the audit below
treats them as such, but a reader should not infer any of these
organs are starving the shipped system today — only that they would
be starved *when active*, which matters for every future generation
that turns the breath loop back on.

---

## Ranked table

| organ | compute verdict | data verdict | storage verdict | overall | cheapest fix | confirming measurement |
|---|---|---|---|---|---|---|
| **Slot mixer** (`W_bq/W_bk/W_bv/W_bo`, `forward()` ~L1330-1360) | single QK^T over full H_W=512, no head split (`sc2 = bq@bk.T`) — the ledger's own words: "the most head-starved organ" | real gradient when `ALG_BREATH>1` (breath output feeds every downstream head via `cur`); currently OFF in deployed | reads/writes full 512-d slot state, no bottleneck | **STARVED** | reshape into N_HEADS (zero new params — head-count is a free reshape per the 2026-09-01 entry) | reshape to 8 heads, warm-continue on the mixer's own incumbent, compare wild-holdout fac-exact/artery; interference-matrix (pairwise head cosine) probe before/after |
| **Pointer heads** (`W_args`/`W_res`/`W_query`, `forward()` ~L1517-1522) | one 512×512 bilinear form expresses every arg/res/query relation ("the artery organ — wild args 0.22") | strongly supervised: BCE (2-hot args), CE (res, query) against gold every step — well-fed | reads final slot state directly, no bottleneck | **STARVED** | multi-form pointer (ensemble/low-rank sum of forms, +0.5-1M) | expand to k-form, measure wild-args accuracy delta; check whether residual errors cluster by relation type (confirms single-form ceiling vs. a data problem) |
| **Macro value system** (`W_y`, `h_dig2`, ~L815, L1523-1526) | same single-bilinear (`W_y`) / single-linear (`h_dig2`) pattern as the pointer heads, for OP_APPLY's second operand — **not named in the existing capacity package** | well supervised (`y` CE at 2x weight, `dig2` CE) — L1710-1718 | fine | **STARVED** (same class as pointer heads, previously unflagged) | fold into the same multi-form-pointer fix — shares the architecture change, near-zero marginal cost | same wild-args-style probe restricted to macro (OP_APPLY) rows |
| **Waist encoder** (`waist_w`, one linear + gelu, ~L754, L1039) | ONE linear map, 2048→512, 4:1 squeeze, single nonlinearity — the sole channel from the frozen trunk into everything downstream | receives the full rich trunk hidden state (not a bottleneck on the input side) | is the bottleneck itself by construction (a single matrix) | **STARVED** | gated 2-layer encoder (+1M, linear cost) | does the register-wall / wild-generalization gap shrink with a deeper waist, holding everything else fixed? |
| **Router** (`W_rs/W_ra/W_rb`, ~L945-948, L1310-1320) | single 64-d low-rank bilinear bias; ledger proposes 4×32-d role-dedicated patterns | span loss (`fspan` CE) is directly, gradedly supervised (bootstrap law) — good; **but** the `_snaps` features it conditions on inside the breath loop are hard one-hot ARGMAX codes from the garage's canonical shelf (L1483-1500), not the graded pre-snap logits — the same reflex pattern (boolean where graded signal exists) that motivated the mask head, not yet recognized as such here | fine | **STARVED** (compute) + **a data-axis flag worth its own line item** | (a) widen to multi-form router bias; (b) feed the router the pre-snap graded logits alongside/instead of the one-hot `_snap_*` codes | A/B the router's snap-input (graded logits vs. current one-hot) at fixed compute; separately, head-split the router bias and re-measure `fspan` CE |
| **Notebook ink** (`W_sil`, one linear "stamp", ~L963, L1195/1215) | single linear projection into the 8-row phase-coded shelf, single query (`W_nq`) — "the notebook's ink is one linear stamp (feed the loud one)" | implicit end-to-end supervision only (no direct target) — standard for memory nets, acceptable but weak | **hard 8-row ceiling**, asserted at import time (`NB_STAMPS holds 8 rows`) and coupled 1:1 to `K_B` (breath count) — a real storage bottleneck if breath count is ever scaled past 8, independent of the compute fix | **STARVED** (mild, compute) + **storage watch-item** | richer ink projection (+0.5M, per ledger); decouple shelf capacity from `K_B` before any future breath-count increase | read-attention entropy over the 8 stamps across breaths — concentrated (expressivity-limited) vs. flat (shelf-count-limited, not ink-limited) |
| **Bank attention + FFN** (`bank()`, ~L1041-1061) | attention itself is properly 8-headed (`N_HEADS=8`, `hd=64`) — **not** head-starved; FFN is 2× expansion vs. the 4× transformer convention (ledger's own correction: "1x" first claimed, verified 2x) | the best-fed organ in the stack: dozens of direct CE/BCE terms per step | shared K/V/FFN weights serve three roles (var-bank, factor-bank, query) via three different learned query vectors (`vq`/`fq`/`qq`) — a deliberate weight-sharing choice, not obviously a bottleneck | **LOW-SUSPICION / mostly RIGHT-SIZED** | FFN → 4× (+1.6M, linear cost) — already scoped in the existing package | none urgent; if pursued, compare per-role probe accuracy (var-span vs. factor-span vs. query) before/after the FFN widen to see whether shared K/V is actually a limiting factor |
| **Classifier + digit heads** (`h_pres/h_ftype/h_op/h_islit/h_dig/h_dup/h_sgn`) | single-linear read-outs over a rich 512-d representation for 2-8-way decisions — standard, appropriate | heavily, directly supervised (CE/BCE every term) | reads final slot state directly | **RIGHT-SIZED** | none | none needed |
| **Bind MLP** (`W_bind1`→gelu→`W_bind2`) | proper 2-layer MLP (512→512→128), not a single bilinear form | strong direct supervision when active (role-factored CE over 4 roles at BINDBUS≥3) | fine | **RIGHT-SIZED** (currently inactive: `ALG_BINDBUS=0` deployed) | none | none needed |
| **step_atlas.py** (Welford + cosine-KNN, `mycelium/step_atlas.py`) | zero learned parameters by design — non-parametric centroid store | **not gradient-trained at all**, by explicit Goodhart-fence design ("CONDITIONING ONLY — never a loss target") | fixed 7×C×D centroid bank, adequate for its role | **STRUCTURALLY EXCLUDED** (correctly — this is not a learner, don't apply the starvation frame to it) | n/a | n/a |
| **Mask head** (spec-stage, `docs/mask_head_spec.md`, not yet built) | 4 dedicated attention heads proposed, explicitly sized UP as the fix for the reflex it replaces | graded metadata by design (confidences, atlas consult, domain-mass radius, current beliefs, step id) — the anti-pattern the router flag above should imitate | dedicated mask-context buffer, per spec | **TEMPLATE — build to spec, don't shrink it** | n/a (already correctly sized on paper) | bring-up ladder in the spec (equivalence → smoke → twin → fleet) is itself the confirming measurement plan |
| **alt21 integrate stations** (stations 3-4, `forward()` ~L1375-1418) | station 3 (second bank-attention) is full 8-head; station 4 (second slot-mixer) inherits the primary mixer's single-head limitation | **the measured failure mode is NOT capacity** — cold-from-noise training kills these stations outright (val flatlines at 0.003-0.0085 per the 2026-09-02 "ISOLATION VERDICT"); warm-continuation from a parse-capable incumbent fixes it (in progress, gentle LR 1e-4 per the same-day "GENTLE-CONTINUATION LAW") | fine | **LOOK-ALIKE, NOT STARVED — a training-regime/cold-birth exclusion, already correctly diagnosed and separated from capacity by the project itself** | n/a for capacity; the real fix (warm continuation) is already underway | already running: the repair3 warm-LR arms |

---

## Per-organ notes (anything not RIGHT-SIZED)

**Slot mixer.** The single clearest case. `forward()`'s breath loop
computes `sc2 = (bq @ bk.T) / sqrt(H_W)` with no head reshape at all,
while the adjacent `bank()` closure two lines away *does* reshape into
`N_HEADS`. It is also structurally the organ doing the most relational
work per the project's own framing (slot↔slot wiring, the alternator's
committed-structure channel, the garage's role-bound reads all rendezvous
here). The fix is free in the strict sense the ledger already noted:
head-count is a reshape, not new parameters, and barely moves FLOPs.
Only currently inert because the deployed config runs `K_B=1`.

**Pointer heads / macro value system.** Both are literally
`(s @ W) @ vst.T` — a single learned bilinear form standing in for what
is functionally a small relation-classification problem (which variable
fills this factor's argument/result/query slot). The macro system's
`W_y` is the same shape doing the same job for the second operand of
`OP_APPLY` macros and was not named in the existing capacity brainstorm
— worth folding into the same fix rather than treating as a separate
initiative.

**Waist encoder.** Not head-starved (it's not attention), but it is the
single linear bottleneck between the frozen, information-rich trunk and
every downstream organ. The project's own suspicion ("the register wall
may partly live at the front door") is worth taking seriously precisely
because every other organ's DATA axis traces back through this one
matrix.

**Router.** This is the organ closest in shape to the original masking
reflex, and it hasn't been named as such yet. The `fspan` supervision
itself is fine (graded, direct). But when the router consumes `_snaps`
(the garage's committed-fact features) inside the breath loop, those
features are hard one-hot argmax reads off the canonical codebook
(`_oh4 = (_lg4 == _lg4.max(...)).float()`), not the graded logits that
exist one line earlier. That is exactly the "threshold where graded
signal exists" pattern the mask-head ruling was written to fix — it
just hasn't been pointed at the router yet. Recommend treating it as a
second, cheaper instance of the same ruling rather than a separate
compute-only fix.

**Notebook.** The ink projection is a plausible target for widening
(per the existing package), but the more structural issue is the hard
`assert ALG_BREATH <= 8` coupling shelf capacity 1:1 to breath count —
a storage ceiling that isn't visible from the parameter count alone and
that the capacity-curve's own proposed +16 scratch slots (20M-mind tier)
would run straight into if breathing and the notebook are ever combined
at a higher `K_B`.

**Bank attention + FFN.** Genuinely close to right-sized; flagged low
here mainly for completeness. The FFN's 2×-not-4× gap is real but small
and already scoped as a cheap linear-cost item. The one open question —
three roles (var-bank/factor-bank/query) sharing one K/V/FFN — is
plausibly a deliberate and fine design (query-vector differentiation is
a standard cross-attention pattern) rather than starvation; listed as
"worth measuring," not "confirmed starved."

**alt21 integrate stations.** Included because the task named it, but
the honest reading is that this is NOT a capacity story — the project's
own isolation experiment (2026-09-02) already separated "starved" from
"born cold" for these exact stations and got a clean answer: full-width,
multi-head-adjacent compute, killed outright by cold initialization, cured
by warm continuation. Worth stating plainly so it doesn't get
mis-filed alongside the genuinely starved organs above.

**step_atlas.py.** Included because the task named it. It has no
parameters and by explicit design is never a gradient target (the
Goodhart fence: "CONDITIONING ONLY — never a loss target"). Applying a
starvation frame to it is a category error — it is correctly excluded,
the same way the mouth and monitor centroids are correctly excluded
elsewhere in the stack.

**Mask head.** Spec-only; nothing to audit in running code yet. Its
value here is as the calibration yardstick the task asked for: 4
dedicated heads, graded (not boolean) metadata inputs, a dedicated
buffer — sized up on purpose, explicitly breaking from "the
single-headed frugality the other organs carry" (the spec's own words).
The one risk worth flagging in advance: cost-cutting pressure at
build time to make it "consistent" with the other organs' frugality
would reproduce the exact failure this organ exists to fix.
