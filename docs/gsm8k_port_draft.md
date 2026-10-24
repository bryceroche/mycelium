# The GSM8K port — DRAFT registration (2026-10-24; not registered, nothing fires)

**Status:** FORK DECIDED 2026-10-24 by Bryce: **B, the lean new tree** — but HELD until the custom breathing transformer has
proved itself ("let's be patient"). Nothing here is registered; it enters the ledger only after that gate.
**Sources:** ledger 2026-10-22..24 (T2h–T2m, the previous-subject check, CF fixture v2) and Bryce's 17 architecture
notes (2026-10-24).

## 1. What the toys hand the port

| Organ | Toy verdict | Form the port carries |
|---|---|---|
| Write join: mention → per-problem entity slot | T2i: the blind query closes only with BOTH joins; a global codebook scores 0.000 on unseen names | slots are per-EPISODE, never a global vocabulary |
| Resolver | T2k port rule: HYBRID_ROLE (overlap + kind + type + rank + learned + role flag) | role flag = the DISCOURSE TOPIC (protagonist ~0.75 in GSM8K, recency ~0.20), not the previous subject (0.327) |
| Type | T2l: from the permanent tree node, not the name's embedding (0.703 → 0.894) | the entity DB carries the type; unknown names enter via the stored procedure (note 4) |
| Time | T2m: geared six-wave stamps in the slot's volatile key = HAND in range; period-6 wheel aliases | geared stamps; no extrapolating readout needed (GSM8K is in range under train = test) |
| Composite key | T2l L2 / T2m M3: bind–unbind lossless (recovery 1.0000) | (slot, time, tree level) by phasor binding, snapped on per problem |
| Residual 0.11 of held-out role | T2l term split (running 10-24) | decided by that verdict |

## 2. The fork (Bryce's call)

**A. Port into the main line** — the tinygrad parser head on the frozen LLaMA trunk, K3 chassis, BP states. The insertion
point is known (the slot QUERY, _bus7/_Q4; ledger 52196), and baselines exist (CLK3_241 wild 0.3793).
Against it: note 3 says replace the trunk; every arm is GPU time; the main line has no gold owner labels (only spaCy's,
ledger 52286).

**B. A lean new tree (recommended)** — notes 2, 3, 9–12 taken literally. A small MHA reader trained from scratch on
GSM8K-train text, with the toys' organs at the joins. CPU/PyTorch first, as the toys were; tinygrad only once the form holds.
Against it: there is no reader yet, and the supervision road has to be built (§3).

## 3. Arm B, staged (each stage gets its own bars when registered)

- **B0 — the data road (zero training).** From GSM8K-train: numerals, the `<<>>` trace as the gold graph (op + argument
  pointers to text numerals or earlier intermediates), mention spans (spaCy NPs = structure, as the toys assumed),
  and the entity DB's first fill (names + types via the stored procedure). Output = the toys' episode format on real
  text. Bar: a hand-check of 40 episodes; ≥ 0.90 correct spans and pointers, else VOID.
- **B0' — the toy honesty step (already queued, ledger 52818).** T2k's generator with the protagonist-centric rule
  at GSM8K's measured rates and a topic flag. CPU, ~30 min. Tells us whether the role organ survives the real register
  before we build on it.
- **B1 — reader + joins on GSM8K-train, scored on CF fixture v2.** Train = test on the TEXT (note 11). Scored on
  counterfactual rows with executor-recomputed intermediates. Floors: value memorizer 0.058; plain-MHA reader (same
  size, no joins) as the control. Claim bar: joins − control ≥ 0.10 on CF intermediate exact, sign on 3/3 seeds.
- **B2 — the capacity read (note 11's own test).** Shrink the reader until the plain control fails CF2; check
  whether the DB-backed arm still passes. This reconciles notes 5 and 16: the DB overfits, the weights cannot.

## 4. Known limits, banked up front

- CF2 cannot separate a GRAPH memorizer from a reader (it scores 1.0). Name swaps carry that separation, and they
  are the T2l/T2m organs' home ground. GSM8K-test stays the honest generalization read.
- Mention spans and owners come from spaCy, which is NOT gold. Claims are made only on CF2's executor gold, never on
  agreement with spaCy (the 52286 rule).
- Model size alone does not guard against memorization (CF fixture memo); the counterfactuals do.
