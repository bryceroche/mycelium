# THE ANSWER-SPACE SPEC — E1+E3, the integer-preserving pair (2026-08-03)

*(Drafted on the word; the retroactive read ran first and priced the
menu — .cache/answer_space_retro_read.json. This generation: E1 NEG
(sign) + E3 BIG (width to 10⁶). E2 RAT and E4 RAD are LATER
generations, each with its own spec. E5 LEN is a SEPARATE TRACK
(§7) — a platform lever, never bundled with the dialect. Ceiling
claim for this pair: 197 → ~265/500 attemptable. **CEILING ≠
COMPETENCE** — nothing here promises conversion; the never-trained
0.02 governs what converts.)*

## 1. The design principle (why this pair)

**The pharyngeal jaw stays untouched.** Sign and width are
representational changes upstream of the crush; the CSP core remains
an integer solver with unchanged arithmetic. Changing what the jaws
GRIP is a different class of change from changing how they CRUSH —
the solver is the one component whose correctness the whole chain
inherits, and this pair never touches it. (Rationals rewrite
arithmetic in the core; that is why E2 is its own generation.)

## 2. Solver (domains only; arithmetic unchanged)

- Domains extend to integers in **[−10⁶, 10⁶]** (`signed=True` on the
  algebra2/3 bridges — built, default-off, callers unchanged).
- **SMOKE FINDING (2026-08-03, part 1): "domains only" was optimistic
  for E1.** Wide (E3) is FULL GREEN: m=10⁶ solves + uniqueness
  certifies in 0.29s — the propagator path handles wide unsigned
  natively. Signed (E1): SOLVING works (search+check computes
  −7+2=−5 natively) but **GAC does not prune signed domains — the
  registry/propagator layer enumerates [0,m] only**, so ban-and-
  resolve dies at budget and the uniqueness gate correctly REFUSES
  to certify. E1's true price RELOCATED TWICE and
  landed (2026-08-03, same session): not tables, not pruning logic —
  **THE SENTINEL COLLISION**: UNASSIGNED was the integer −1, a
  legitimate value under signed domains; arith3_pred read it as a
  hole, so phantom support kept −1 alive through every GAC pass and
  ban-and-resolve churned to budget. THE FIX: the sentinel moved out
  of band (−2³¹ — outside every domain in every regime); numeric
  identity changed, nothing else. THE BOUNDARY NOTE, as ordered:
  this touched csp_core — THE JAW — and the change is to the data
  model's sentinel identity, never to crush semantics; the one
  `v < 0` site audited (a var-index guard, csp_core:584);
  equivalence verified — 200 sampled gold rows behave IDENTICALLY
  under both sentinels (189/200 both, same 11 harness-reconstruction
  artifacts). BATTERY GREEN: signed uniqueness certifies in 0.01s;
  wide+signed (987654−999999=−12345 @ m=10⁶) certifies in 0.14s;
  **the collision value itself certifies (1−2=−1, unique)**. THE
  GATE IS OPEN: signed rows can mint.
- **mod/fdiv semantics fence:** constructions keep operands
  NON-NEGATIVE at mint (generator constraint), and predicates GUARD
  (reject negative operands into mod/fdiv — the k≤0-guard precedent
  extends). Negative-operand floor/mod semantics are a decision this
  spec deliberately does NOT make; if a future corpus needs them,
  they arrive with their own pinned convention.
- Uniqueness budget unchanged (5000; exhaustion rejects).

## 3. The head (the one architecture change; new generation)

- Digits MSD-first: **N_DIG 3 → 7** + a **sign channel** per number
  slot. New env flag `ALG_WIDE=1` (manifest-carried, deployed-env
  door). This is gen-23-era: **pad-warm from g22** (never discard a
  trained router); new digit positions and the sign head get direct
  mint supervision (the attention-bootstrap principle — new pathways
  need direct supervision, and the mint supplies it).
- Two-terminal check at build: sign/wide-digit emission AND gold feed
  both wired, or the grad is None (the representability audit's law).
- **THE SIGN-TERMINAL SMOKE (Bryce's rider — its own smoke, never
  riding the build check):** the sign head is Fire-0's inert-patch
  shape — a new pathway whose gradient can be structurally absent
  while everything compiles and prints. Before ANY corpus mints: one
  deliberately-signed row through the training step, **gradient
  asserted nonzero at the sign terminal.** The check is named; the
  smoke exercises it.

## 4. Mint generators

- Value ranges configurable per construction; solution-first;
  consecutive letters; knot dedup unchanged (canon carries values).
- Sign enters STRUCTURALLY (the pointer law's remedy): via subadd
  with a−b < 0 now legal, and explicit negative givens.
- **Surface forms for negatives, pinned at mint:** symbolic "-7" and
  lexical "negative 7" both licensed; "minus 7" REFUSED (collides
  with subtraction phrasing — the unique-reading audit's
  alphabet-exclusion clause applies at authoring).
- Tokenizer pin extends: every rendered value's digits AND the minus
  sign must tokenize present in text (trace-layer value-consistency
  inherits sign+7-digit literals).
- Dose law as always: share-of-mix AND reps declared before fire.

## 5. Custody + entourage

- **Nothing banked moves.** New dialect = new generation; every
  banked gold stays in its vintage; old fixtures read under old
  rules (relative bars across the vintage change).
- Full entourage duty at promotion: specialist remine, centroids
  re-anchor (rotation law), mouth bank rebuild for new families,
  panel re-audition, manifest hashes.
- **THE CAP-AUDIT RIDER (Bryce's, banked here):** the ≤300 cap
  SHAPED constructions four measured times — [525] (cap-blown),
  [691], [776], [1013] — and those rulings were lawful precisely
  because the cap was an external constraint the pen didn't choose.
  Under a 10⁶ cap those construction pressures relax: **the bench's
  cap-forced rulings become HISTORICAL, not governing.** A future
  reader must not apply cap-forced reasoning under a cap that no
  longer binds; this line ships with the spec so the ledger says it
  before anyone needs it.

## 6. Bars (pre-registered at spec; pinned numerically before fire)

- **B1 parity:** new-range mint fixtures (wide-digit and negative
  givens/rel/subadd) parse at ≥ old-range parity minus a pinned
  margin. **Margin pins WHEN THE MINT SMOKE PASSES, at the latest**
  (Bryce's rider: registration and ignition drift apart by hours;
  the margin's whole protection is that it predates the number).
- **B2 no-regression:** bigtest and the standing fixtures hold
  within band under relative bars (own-vintage comparisons only).
- **B3 honesty:** NO M500 conversion bar — the ceiling moved from
  197 to ~265 attemptable; competence is measured after training,
  never promised by the spec.

## 7. E5 — the length track (separate; possibly the cheapest +35)

Length 300→600 chars is a tokenizer-and-window question (T_ALG
256 → ~512), orthogonal to the dialect: +34/35 at EVERY rung of the
ladder. **THE DISK ARITHMETIC, RUN 2026-08-03 (the law's receipt —
the dup fire's ENOSPC — honored before assembly):** states memmap
86GB (T_ALG 256) → **173GB per resident copy at T_ALG 512**; disk
now 1.4T free — ONE resident copy lawful under
assemble→train→delete; two resident copies (e.g., an A/B arm pair)
= 346GB, still lawful but named; the six-copy shape that killed the
dup fire would be 1.04TB and is FORBIDDEN by the standing rule.
Remaining prices to measure: trunk recompute time, step time at
T_ALG 512. **It does not wait behind the dialect generation** — it
can ride any generation or none; its read is banked and its word is
its own.

## 8. Rollout order

spec (this page) → mint smoke (generator + solver roundtrip on
wide/negative values, zero-GPU) → head change behind `ALG_WIDE` →
corpus mint under dose law → **gen-23 fire ON THE WORD** (hold for
the word; nothing in this spec authorizes a training run).
