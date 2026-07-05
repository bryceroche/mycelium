# Session 2026-07-04 — Alternator docs adopted; Brick-0 entry point (handoff)

**What happened:** fresh CLAUDE.md + README.md (the two-phase **Alternator** spec) were
reviewed and adopted, with one restoration patch. This doc is the handoff: what changed,
the review findings, and the concrete entry point for the next session.

---

## 1. What the fresh docs are

- **README.md** — purely additive: existing content untouched; a new "two-phase
  Alternator" section (with ASCII diagram) + matching perceiver-retirement wording
  updates. Taken as-is.
- **CLAUDE.md** — §0–§7 essentially intact; **§8 completely rewritten** from the June-26
  "Phase 1 construction" direction into the concrete **Alternator spec**: six
  parse/deduce cycles, a TCP-style SYN/ACK/NACK handshake, the perceiver reborn narrowly
  as session monitor + spectral segmenter, the zero-LoRA null hypothesis, a Matryoshka
  waist schedule (512→128), and a brick ladder (Brick-0 → C) with kill criteria stated
  before anything fires.

**Commits:** `93db7e8` (adopt fresh docs) + `6956424` (restoration patch, below).

## 2. The one problem found (and fixed — commit `6956424`)

The new §8 was authored from the **June-20 doc state** (its own §8.9 cites "the prior
framing (2026-06-20)"; zero mentions of Sudoku/QCP/two-death-mode). A wholesale replace
would have silently dropped the **June-26 settlement** — exactly the kind of hard-won
dead-end guard CLAUDE.md exists to preserve:

- **Neural-guided clean-CSP search is CLOSED** — the two-death-mode law; Sudoku
  5000/5000 at median 0 decisions; five negatives.
- The deducer's narrowed role: **critic / format-definer / soft-graph solver** — NOT a
  better solver.
- The cheap **oracle-upper-bound kill-gate** (`csp_core.policy_valorder`, pure CPU).
- The `docs/phase1_construction_brief.md` pointer + June-26 memory keys.

Restored as **CLAUDE.md §8.0** + memory-key entries; also fixed the now-stale
"neural-ordering prior is an UNVERIFIED hint" parenthetical in §4 (the QCP kill-gate,
commit `316897a`, settled it). Separate commit → trivially revertable if the doc should
stay verbatim.

## 3. Review of the Alternator design itself

**Strengths (the discipline is the best part):** names its three unvalidated
load-bearing assumptions out loud (§8.7); keeps the validated deducer as an untouched
regression anchor; §8.3 honestly admits alternation only earns its cost if the NACK path
works, with staged parse-then-solve as the fallback.

**Three critiques on the table before building:**

1. **The parser trunk may re-hit the reading-comprehension wall.** The June-26 brief
   concluded the constructor needs an "LLM-grade comprehension base" — the GSM8K wall
   was hit by a 4-layer breathing trunk. The Alternator bets 4 Llama layers (2048d
   L0–L3) clear it. Want a cheap comprehension gate early in the ladder: can this trunk
   do *single-shot* NL→graph on templated data before any alternation machinery exists?
   Arguably a **Brick-A′ upstream of even the zero-LoRA question**.
2. **Brick-0's ~0.85 pass bar looks stale.** The spec cites "the ~0.85 linear
   valid/invalid probe" as the bar, but the `gen-weights` branch work
   (`scripts/learned_waist_gate.py` header, citing `dart_cluster_probe.py`) records that
   the Anna-Karenina common-mode signal, once instance identity is removed (the honest,
   transferable read), is **AUC 0.582 raw-1024d / 0.658 under PCA-256** — the 0.85
   figure is presumably the non-centered version. Brick-0's pass bar must be grounded
   against the centered numbers or it will pass/fail against the wrong baseline.
3. **The Poincaré ball is one of only three interface objects while §7's relaxation is
   still blocked.** The spec flags it as the hard risk — the fallback should be stated
   operationally: parser emits discrete membership → hard masks, so the Alternator never
   gates on unblocked hyperbolic research.

## 4. Entry point for next session — Brick-0 baseline reconciliation

Brick-0 is the designed entry (one session, zero new architecture, **eval-only** — no
training run needed: the multitask general-weights ckpt covers coloring + circuit +
kenken). Given critique #2, step one is reconciling the pass bar:

1. Re-run/read `scripts/dart_cluster_probe.py` on the KenKen arm of the multitask ckpt
   to fix the real linear valid/invalid baseline (centered, by-instance CV).
2. Wire frozen Brick-1-style latents (`docs/perceiver_poincare_design.md` §9) to read
   the same reps; pass bar = beat the honest linear baseline from step 1.
3. Fail → rework the matched-filter story (§8.5) before wiring anything else.

**Loose ends:**
- `docs/NEXT_SESSION.md` (June-26 cold-start handoff) is now partially superseded by the
  new §8 — refresh once oriented.
- Three untracked scripts from the prior `gen-weights` session are related groundwork,
  not yet committed: `scripts/learned_waist_gate.py` (learned-vs-PCA waist gate),
  `scripts/read_at_settle_eval.py` (cheap control vs the cathedral build),
  `scripts/probe_svd_collapse_multitask.py` (collapse probe on the multitask ckpt).
