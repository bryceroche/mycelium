# The Mask Head — the fourth trained organ (2026-09-03, word given)

*The learned precision channel. Companion to
docs/alternator_v21_training_spec.md and the ledger's v2.1 entries.*

## 0. Why it exists (the gut's ruling)

Today's dynamic masking (ALG_MASKRE) is a REFLEX, not a skill: zero
dedicated attention heads (it rides the shared bank/mixer), zero
learned parameters (a hard threshold on the snap adjacency `_A5 >
0.5`, additive-open), near-zero metadata (only committed edges, no
confidences). The ping-pong's whole value — solver facts reshaping
the next breath's attention — flows through a channel with no
capacity to learn from what flows through it. We built the pipe and
forgot the pump.

**THE RULING**: dynamic masking (and climbing the atlas ladder 7
rungs) is the FRUIT of the neural/symbolic ping-pong. It gets a
dedicated trained organ — the MASK HEAD — sized with ENOUGH compute
(multiple attention heads), storage, and metadata to succeed. This is
the subsumption law's other half made real: the breath loop covers
the propagation-CONTENT channel; masking is the PRECISION channel it
explicitly does not cover (scope narrowed 2026-08-30). The precision
channel now gets its organ.

## 1. What it is

A dedicated multi-head attention organ (env ALG_MASKHEAD; propose
M_HEADS = 4, its own head bank, dim H_W=512, ~1.5-2M params — sized
UP on purpose per the ruling, not the single-headed frugality the
other organs were audit-flagged for). It runs at each breath_step's
seam, AFTER the solver ping and atlas consult, and BEFORE the next
step's gather station. Its job: decide the next step's attention
GEOMETRY — a soft, learned mask bias over the slot<->slot and
slot<->token scores.

## 2. Inputs (the metadata the reflex throws away)

The mask head reads the rich state the threshold ignores:
- **Snap adjacency WITH confidences** — not `>0.5` booleans but the
  graded `_A5` producer->consumer matrix (which committed edges, how
  strongly).
- **The per-step atlas consult** (docs: step_atlas.py) — the k nearest
  centroid pages for this breath_step (the 7-rung climb): which
  operation-KIND the parse currently resembles conditions which
  attention pattern that kind wants.
- **The domain-mass matryoshka radius** — per-variable domain size
  from the solver ping (how SEALED each var is; sealed vars need less
  attention, open vars need more).
- **Current beliefs** — presence/pointer logits at this step.
- **Breath-step id** — the mask head is step-CONDITIONED (early steps
  survey wide, late steps focus; the tightening is learnable).

## 3. Output + the open-only constraint

A soft mask bias `mb (B, L, L)` added to scores BEFORE the -1e4 close:
`sc = sc + mb`, where mb is bounded so it may OPEN (raise) attention
on committed/atlas-favored edges but MAY NOT tighten below the
first-pass heuristic mask — A0's grave stays honored (the killed
tightening road; MASKRE's open-only law generalizes from hard to
soft). Implementation: `mb = softplus(head_output) * (first_pass_open)`
— nonnegative, gated to the already-open region; the hard floor
survives underneath.

## 4. Training (dual-terminal, Goodhart-fenced)

- Trained ONLY by the downstream parse loss (fac-exact CE through the
  re-masked pass). NO mask-imitation target, ever — a supervised mask
  teaches concealment, not precision (the fence; assert_not_supervised
  at the training door).
- Two-terminal: emission gradient flows back through the mask into the
  head's weights (earned), while the atlas page and solver facts enter
  DETACHED (the contract — dL/dp through solver/atlas = 0).
- COLD-BIRTH LAW: the mask head is a new attention organ — it enters
  WARM (continuation from a parse-capable incumbent) or with an
  attention-bootstrap supervised warmup, NEVER from noise.
- GENTLE-CONTINUATION: warm starts at LR<=1e-4.
- Gate AJAR (0.02) over its own learnable path (gate-deadlock
  corollary), zero-init OUTPUT projection so birth is bit-identical
  (the ResNet law); it earns its amplitude.

## 5. Compute / storage / metadata (the three the gut named)

- COMPUTE: M_HEADS=4 dedicated attention heads (own Wq/Wk/Wv/Wo);
  scale axis registered — 4 -> 8 heads is a capacity-curve dial, not
  a rebuild (head count is free reshape).
- STORAGE: reads/writes the slot bank + a small dedicated mask-context
  buffer (the graded adjacency + atlas pages staged per step).
- METADATA: the full input list in Section 2 — the organ's edge over
  the reflex is precisely that it SEES more and can WEIGH it.

## 6. Bring-up ladder (law; rungs pinned)

1. Equivalence: ALG_MASKHEAD off = bit-identical baseline; on with
   zero-init output = bit-identical at birth.
2. Smoke: 300 steps, step-time (priced: +1 attention block/step ~
   the alt21 integrate-station cost), NaN guard, mask-open-rate log
   (should rise breath 1->6, the Blackbird profile).
3. Twin: mask-head ON vs OFF, warm-continued from the v2.1 winner,
   single-bit delta, wild holdout.
4. Fleet: wild holdout fac-exact + artery, 2-seed law for any claim.

READ-FIRST caveat: the soft mask needs the re-masked pass to train,
so unlike the atlas consult it cannot be a pure read-time add — it
enters at the warm-continuation training stage on the v2.1 winner,
gated behind the step-partitioned engine (its natural home: the seam
where ping + atlas + mask all live).

## 7. The reframe

v2.1 was "the four-layer step." With this organ it becomes "the
four-layer step + a masking LEARNER" — the ping-pong's actual
fruit-bearing branch. The alternation was always meant to change what
the model LOOKS AT; until now the change was a reflex. Now it learns.
