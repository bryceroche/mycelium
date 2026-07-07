# Mycelium: A General Factor-Graph Reasoning Engine — Agent Brief

**Author:** Bryce + Claude · **Deadline:** Dec 25, 2026 · **Target:** MATH-500
**Platform:** Shadow Glass (AMD 7900 XTX, 24GB) · tinygrad + AM driver · no ROCm

**What Mycelium is (2026-06-20): ONE general engine for ANY factor graph.** A problem
becomes (variables + typed factor nodes + membership); the engine reasons over it. Two
halves are BUILT + VALIDATED: a general **DEDUCER** (the v98-lineage breathing
transformer, now domain-general) and a general symbolic **SEARCH tier** ("the deducer
proposes, complete search disposes"). **Generality is the holy grail — resist all
domain-specific code in the engine/core.** A would-be geometric front-end (the Poincaré
tiers) remains **spec-stage**, as does the current forward design — the two-phase
**ALTERNATOR** (2026-07-04: iterative NL→factor-graph parsing interleaved with deduction,
a TCP-style handshake, and a perceiver reborn as session monitor / spectral segmenter,
§8). Read §0 first. Conceptual writeup: `README.md`; search-tier
design: `docs/general_factor_graph_search.md`; pre-v98 vision/empirics: `docs/archive/`.

---

## 0. The frame (read first)

Mycelium decomposes any problem into a **factor graph** and reasons over it with two
validated, general components, plus a spec-stage geometric front-end:

- **THE DEDUCER (built, validated, general).** A small iterative transformer (Pythia-410M
  L0–L3 SHARED across K=16 breaths) performs factor-graph inference by K passes through
  the same weights, with per-head attention masks built at runtime from `membership`. It
  is **byte-identical to the validated v98 KenKen executor** when driven with KenKen
  inputs, and **generalizes to graph coloring and hierarchical Boolean circuits** — one
  engine, three structurally different factor graphs (`mycelium/factor_graph_engine.py`).
  Its real superpower: **distributed/PARALLEL deduction that SCALES** (§4).
- **THE SEARCH TIER (built, validated, general).** A predicate-driven systematic search
  that wraps the deducer: "deducer proposes the ordering, complete symbolic search
  disposes." The **only** domain-specific code is a per-factor-type predicate + a thin
  bridge; the verifier / GAC propagation / variable-ordering (MRV) / value-ordering (LCV)
  / backtracking are ALL derived generically. Proven on coloring, SAT, and KenKen with
  **zero general-core edits each** (`mycelium/csp_core.py`, `docs/general_factor_graph_search.md`).
- **THE GEOMETRIC FRONT-END (spec-stage, NOT built).** A learned Poincaré (hyperbolic)
  ball that would *generate* the attention masks (topology) instead of hardwiring them.
  Reproduces hard masks frozen (byte-exact) but its relaxation is **blocked** for
  non-partition graphs, and its headline "radial-depth" payoff is **REFUTED** (§4, §7).

**THE TWO CHANNELS (the conceptual spine — keeps each piece honest).** A factor graph
splits into two ORTHOGONAL channels: **TOPOLOGY** (who connects to whom — `membership` →
attention masks) and **SEMANTICS** (what relation must hold — the predicate). The
Poincaré ball is the topology channel (it *generates masks*, carries no relation content).
The search tier's **predicate registry** is the semantics channel — and it is the
symbolic twin of the deducer's **verification inlet** (op+target features). Folding
semantics into the topology/mask channel is the **refuted move** (v100's "C2 death":
arithmetic as VERIFICATION, never an op-type mask channel).

**Discipline (the over-claim guard — getting this wrong is the main failure mode):**
- **Built/validated = the DEDUCER + the SEARCH TIER.** The Poincaré embedding + hyperbolic
  mask generator are **spec-stage** — NEVER state or imply they are built/working.
- **The engine is the v98-lineage deducer, NOT a perceiver.** Perceiver-as-CORE IS RETIRED
  (5× refuted v118–v121; v300 failed flat at chance). Brick-1 (2026-06-16) later validated
  a small latent bank breathing against Poincaré anchors; the ONLY sanctioned perceiver
  role is the spec-stage monitor/segmenter of §8.5 — never the core engine.
- **On clean verifiable CSPs, SYMBOLIC search dominates** (free, smaller trees); learned
  *probabilistic* propagation was net-negative. The deducer's demonstrated value is
  **generality + parallel deduction**, NOT search value/ordering (§4).
- **The radial-depth "deep prize" is REFUTED.** Do not present it as the goal.

---

## 1. The deducer in one paragraph (the validated executor)

A small iterative transformer (4 Pythia-410M L0–L3 layers SHARED across all K breaths,
h=1024, 16 heads, ~32M trainable + ~52M token-embeddings) performs factor-graph inference
by K passes through the same weights. Each breath: add a per-breath additive marker → 4-layer
transformer with a structured per-head attention mask encoding the factor topology → a
learnable per-breath `delta_gate` convex residual blend → per-breath layernorm +
value-codebook readout → per-breath calibration head. K breaths are JIT-unrolled into one
graph. Training: per-breath weighted CE (`loss = Σ_k (1 + k/(K−1))·CE(logits_k, target)`),
the "ladder" that makes K matter. The general engine (`mycelium/factor_graph_engine.py`,
`factor_breathing_forward` + `FactorGraphSpec`) is byte-identical to the v98 KenKen path
(`mycelium/kenken.py`, the regression oracle, never touched) and parameterizes it for any
typed factor graph: variable nodes, typed factor nodes, `membership`, an optional
verification inlet (`has_factor_inlet` — KenKen feeds op+target; coloring/circuits leave it off).

---

## 2. Deducer components, as-built (the breathing recipe)

Entry: `factor_breathing_forward` (`mycelium/factor_graph_engine.py`); KenKen oracle:
`kenken_breathing_forward` (`mycelium/kenken.py`).

| Component | What it does | Where |
|---|---|---|
| **Iterative shared-weight prefill** | K passes through Pythia L0–L3, SAME weights every breath; 1024d residual is the persistent state | `factor_graph_engine.py` |
| **Per-breath additive marker** | Orthogonal per-breath embedding added to the residual | `breath_embed` |
| **Per-head masks from membership** | `adj = membershipᵀ@membership > 0` → per-relation per-head masks (v98's 5/5/5/1 falls out generically via `_cell_mp_head_allocation`). Hard `{0,−1e4}` bias. | `mycelium/factor_masks.py`, `factor_graph_engine.py` |
| **Verification inlet (semantics channel)** | Per-factor op-type + log-bucketed target features (arithmetic as VERIFICATION, never an op-type mask channel — v100's C2 death). The neural twin of the search predicate. | `build_verification_inlet`, `kenken.py` |
| **Per-breath `delta_gate`** | Learnable convex residual blend `x = x_pre + gate_k·(h − x_pre)` | `model.*_delta_gate` |
| **Per-breath calibration head** | Scalar confidence per breath (Dopri5-style error-estimator hook) | `*_calib_head_*` |
| **Value codebook readout** | N-value codebook; aligned to `state_embed` at init | `value_codebook` |
| **Convergence instrument** | Min-based gold-free `breath_count_min` (argmin consecutive-belief JSD) + settled=correct-at-settle | `convergence_instrument` |
| **Codebook-orthogonality penalty** | Penalizes off-diagonal cos of row-normalized codebook gram; *rotates* collinear rows apart | trainer |
| **Per-breath weighted CE (the ladder)** | `1 + k/(K−1)` weighting; the reason K matters | trainer |

Trainer: `scripts/factor_graph_train.py` (`FG_TASK={kenken,coloring,circuit}`, kenken default).

---

## 3. The general search tier (Path B — `docs/general_factor_graph_search.md`)

Two layers over the factor-graph abstraction. **The only domain code is a per-factor-type
predicate + a thin bridge; everything else is general.**

- **Layer 1 — the general CSP interface** (`mycelium/csp_core.py`, ZERO domain identifiers):
  a three-valued predicate `predicate(ftype, params, member_values) → {SAT|UNVIOLATED|VIOLATED}`
  is the sole seam. The verifier (all factors SAT), GAC propagation (prune values with no
  supporting tuple; **commit only FORCED singletons**), MRV variable-ordering, and LCV
  value-ordering all derive from it. **DSATUR = MRV-on-coloring; AC-3 = GAC-on-not-equal**,
  proven by construction + parity-pinned. An `arity_cap` guards generic GAC; large-arity
  factors get an opt-in `specialized_propagator`.
- **Layer 2 — pluggable search strategies** (`backtrack_search` + `solve_symbolic`): systematic
  DFS now; MCTS / local-search later slot in with zero domain code. The neural deducer enters
  ONLY as **ordering priors** (bias MRV/LCV), never as commits.
- **The registry + per-domain bridges** live in `mycelium/csp_domains.py` (the ONLY file with
  domain knowledge). Adding a domain = a few `register()` calls + one bridge.

Drivers: `scripts/search_coloring.py` (+ the neural ordering arms), `scripts/search_kenken.py`;
parity gates: `scripts/test_csp_parity.py`, `scripts/test_kenken_parity.py`; legacy reference:
`mycelium/csp_coloring_legacy.py`.

---

## 4. Empirical status (the findings that fix the architecture)

**Generality PROVEN (Phase 0 + Phase 2 done, branch `mycelium-factor-graph`).**
- Phase 0 (commit `6af771c`): the predicate-driven core reproduces coloring **byte-identical**
  (symbolic `solve_symbolic` new==old on 22+ fixtures; GPU anchors B0 0.025 / B1 0.85 / B2b
  0.825 / B3 0.95 exact, VERIFY_PARITY max|Δlogit|=0). Zero domain identifiers in the core; a
  full **SAT** domain was added via predicate+bridge alone (solved + certified UNSAT), touching
  neither core nor registry.
- Phase 2 (commit `afc4a2f`): the SAME core solves **KenKen** (param-carrying arithmetic cages +
  7-ary all-different) with `git diff` showing only `csp_domains.py`. Symbolic search solves
  **100% across all givens bands**; B3 (GAC + all-different) collapses the tree (g10: 305
  decisions → 0.9). Three structurally different domains, one search tier, zero core edits each.

**Distributed deduction is PARALLEL and SCALES (the real superpower).** The deducer resolves
~4 deduction LEVELS per breath (a 4-layer transformer ≈ 4 attention hops), so it solves
depth-16 Boolean circuits in ~4 breaths (K_min ≈ D/4, sub-linear in depth; per-D flat to D16).
It is depth-PARALLEL, not depth-sequential.

**Symbolic dominates clean CSPs (the honest negative).** Neural-propagation-inside-backtracking
on hard 3-coloring LOST to symbolic search for free (AC-3 ceiling 0.95; neural probabilistic
propagation net-negative). A CONF_THRESH sweep confirmed the mechanism: at threshold→1.0 the
neural arm becomes byte-identical to no-propagation — sub-100% commits are pure losing bets.
**A propagation commit must be logically FORCED, not a confident guess.** The neural signal's
role is ordering, not committing. (Neural-ordering-as-PRIOR was subsequently CLOSED too:
the QCP kill-gate + two-death-mode law, §8.0 — no clean exact-propagatable CSP rewards
neural ordering. The real test is the non-symbolic frontier.)

**Radial-depth deep-prize REFUTED.** On the circuit DAG testbed, ρ(per-node settle-breath,
topological depth) = 0.13 (bar 0.30, real spread, clean shuffle-null) — the breath allocation
is NOT depth-ordered. The engine solves hierarchy (~0.97) via distributed constraint-satisfaction,
not a depth-ordered geodesic traversal. Consistent with depth-PARALLEL above.

Detailed empirics live in `memory/` + git history.

---

## 5. Specifications

- **Init:** Pythia-410M L0–3 (attn + FFN + token embeddings 50304×1024), all 4 layers SHARED
  across K breaths.
- **Dimensions:** h=1024, 16 heads × 64, FFN 4096, vocab 50304. KenKen: 49-cell grid (N_max=7),
  7-value codebook, K=16, BATCH=8. Coloring/circuit set `n_values`/`s_max` per task.
- **Params:** ~32M trainable + ~52M token embeddings. AM K-graph limit: K=28 hangs; K=16 known-good.
- **Checkpoints:** `.cache/fg_ckpts/` — coloring (`fg_coloring_k16`), circuit (`fg_circuit_*`),
  **KenKen (`fg_kenken_k16_reg` — the HEALTHY one**: curriculum + v45 reg stack; test cell 0.80 /
  puzzle 0.35; per-band g40 ≈0.65 → g10 ≈0.06), multi-task (`fg_multi_fair`,
  coloring+circuit+kenken in one weight set). **Footguns (2026-07-04, both produced
  plausible-looking WRONG evals):** `fg_kenken_k16` (no `_reg`) is the OVERFIT base (test cell
  ~0.50); `fg_kk_2k*` are hidden=2048 experiments, incompatible with the default h=1024 build —
  the loader silently keeps init on shape mismatch → chance-level output. Sudoku is NOT an FG
  task: it lives in the v98 ancestor (`.cache/sudoku_ckpts/v98_prod_step4000.safetensors`,
  K_MAX=20, `scripts/eval_v98_sudoku.py`; 24/band: easy 96/79 cell/puzzle, med 84/4, hard 80/0).
- **Phase-1 residency budget (2026-07-05, `scripts/phase1_residency_smoke.py`):** frozen
  Llama-3.2-1B L0–L3 + 128k×2048 embed (fp32) 2.03GB + trunk JIT buffers 0.39GB + deducer
  0.49GB ≈ **2.9GB total → ~21GB headroom** on the 7900 XTX. Trunk: real tokenized
  KenKen-in-words, finite activations (norm ~12.9), TinyJit assign-in-place replay 0.34s
  per (B=8, T=512) forward, compile 0.4s — NO AM-driver hazards on the new graph shape.
  Co-resident deducer eval unchanged (cell 0.745).
- **Platform:** AMD 7900 XTX, tinygrad, AM driver (Secure Boot off + `vm.compact_unevictable_allowed=0`).
  Ubuntu 24.04. No ROCm/CUDA/PyTorch.

---

## 6. Editing rules (durable, hard-won)

- **No mid-breath token generation.** Reasoning stays in the 1024d residual; tokens (if any)
  generated once at the end. ("had had had" if violated.)
- **Diversity must be structural, not learned.** Masks are geometric/structural. Every learned
  diversity mechanism (scales, soft tokens, fingerprints) collapsed to a constant within one epoch.
- **Digit/value-spaced for arithmetic.** Single-cell values; whole-number BPE tokens force memorization.
- **Positional/referential structure must enter AS STRUCTURE, not as prose/computation
  (2 sightings, 2026-07-06/07 — the parse-side counterpart of "diversity must be structural").**
  One attention hop can't COUNT sentences (fix: the sentence-index embedding, given as a
  feature); a frozen 4-layer prefix can't BIND references ("statement 7" -> sentence 7 —
  the text-NACK arm was perfectly content-blind: fix(true)==fix(shuffled) exactly).
  Conditioning and positional identity enter as position-aligned features, never as text
  the shallow reader must decode.
- **Factor per-NODE, not per-EDGE (v112b).** Prefer per-position gating over pairwise structures
  (learned attention biases, edge tensors). Edges are already captured by the binary masks.
- **Trained structure is RELATIONAL; post-hoc recombination of its parts inherits none of
  it** (promoted 2026-07-07, third sighting). The perceiver add-on failures (new pathways
  can't join a trained circuit), the codebook's pairwise geometry (value lives BETWEEN
  codewords), and the Matryoshka prefix (85-91% dim overlap, swapped 10-15% costs the whole
  cliff; incumbent dims have ZERO standalone value below their trained composition widths).
  Circuits don't survive member substitution, even by near-neighbors.
- **Attention bootstrap.** New attention/pointer pathways (~30+ positions) don't bootstrap from
  task gradient on diverse data — they need an anchor or direct supervision. Codebook selection
  (≤32-way) bootstraps from task gradient alone. *Why the perceiver failed 5×.*
- **A propagation commit must be logically FORCED, not a confident guess** (the search-tier law).
  The neural deducer enters search ONLY as ordering priors, never as commits.
- **Soundness tests must cover the GENERAL regime, not just the deployed one.** Phase 2's
  all-different propagator was sound as deployed but unsound off the permutation regime, and the
  test only covered the deployed regime → false confidence. Test the general case to protect generality.
- **Eval-only checkpoint loads must be COMPLETE — hard-error on missing/mismatched keys.** The
  loader keeps init on shape mismatch (fine for warm-start); in eval it is a FALSE-RESULT
  generator (2026-07-04: a silent 43-key fallback scored chance-level as if it were the model).
- **No silent fallbacks anywhere in the chain** (generalized 2026-07-06 — the same bug class
  hit twice: the loader's keep-init AND a pipeline grep masking a stage failure from `set -e`).
  Permissive defaults that swallow a failure signal are false-result generators: hard-error at
  boundaries (`set -eo pipefail` in chains; truncation/budget gates at data boundaries — the
  token-budget guard caught a 1-in-39,996 corrupted-gold case a smoke test can never see).
- **Per-breath CE direction is a free eval sanity gate.** Flat at ln(n_values) = broken load;
  RISING across breaths = overfit/wrong regime; DESCENDING = healthy ladder. It diagnosed both
  bad-checkpoint evals above before any deeper digging.
- **Every error-behavior prediction must state its assumed ERROR-DENSITY REGIME**
  (promoted 2026-07-07, third sighting: delete-one blame recall 0.034; the chain>>coupled
  silent ordering inverting; the withholding peak displacing k=1-2 -> 3). Isolated-error
  predictions are VOID above measured multi-error density (~5 errors/failure at parser
  plateau). Pre-registration checklist item, not just a named trap: state the regime at
  writing time.
- **A registered metric must match the DECISION STRUCTURE of the mechanism** (promoted
  2026-07-06 on the third sighting). Scalar convergence summaries degenerate (argmin-JSD,
  delta-settle — 2×), and continuous aggregates over sub-threshold jitter degenerate the same
  way in reverse (L1 prob-mass "concentration" read 0.5 while decision-level flips ran 200:1).
  For discrete commitments, count FLIPS; for converging fields, read the FIELD. The
  localization instrument going forward: flagged-vs-unflagged FLIP-RATE RATIO (pre-registered
  ahead of the text-rendered arm).
- **Substrate laws (tinygrad + AM driver):** no `dtypes.float32` literal inside the JIT step;
  `scores.clip(-1e4,1e4)`; where()-gated NaN guard (NOT multiply — NaN×0=NaN); single-kernel
  `isfinite`; knobs in the JIT cache key; assign-in-place fixed buffers for repeated JIT'd forwards
  (compile once, replay). Hyperbolic-specific: clamp `|z|²≤1−1e-5`, arccosh arg `≥1+1e-7`. See
  `memory/reference_tinygrad_am_quirks.md`.
- **Bryce wants root-cause perf fixes**, not workarounds, when perf is the bottleneck.
- **Process discipline:** **commit freely (no need to ask)** — local + reversible; still **ask before
  push** (outward-facing) and **hold for the word before firing training runs** (GPU cost); **offer
  engineering critique before rubber-stamping** (esp. enthusiastic/gut-feel relays).

---

## 7. The geometric front-end (Tiers 1–2 — spec-stage, NOT built)

The would-be Poincaré embedding (Tier 1) + hyperbolic mask generator (Tier 2):
`docs/hyperbolic_mask_generator_spec.md`. **Status: spec-stage.**
- **Reproduces frozen, byte-exact** (the geometry can regenerate any factor graph's hard masks at
  `t=0`, machine-precision, partition + non-partition).
- **Relaxation BLOCKED for non-partition graphs.** The clique-union + both-members gate is
  gradient-dead (anchor grad = 0 under real CE); partition relations (KenKen rows/cols) relax but
  are flat. A relaxable non-partition construction is unbuilt research.
- **The radial-depth deep-prize is REFUTED** (§4) — radial position ≠ abstraction depth in the
  executor's dynamics. The geometry's transfer/interpolation payoff is UNPROVEN (gated on the
  blocked relaxation). Strictly additive, the hard mask is the permanent fallback if ever built.

---

## 8. Current direction (2026-07-04) — the ALTERNATOR (spec-stage)

The frontier from the two threads below is now instantiated: **NL-specified problems**,
where symbolic propagation is unavailable until the factor graph exists. The design (the
"Alternator") interleaves parsing and solving so the graph is built *iteratively under
deductive feedback*, not in one shot. **STATUS (2026-07-06): the Phase-1 SKELETON IS
BUILT** — steps 1–3 of `docs/phase1_skeleton_spec.md` (NL generator with
round-trip-gated gold; residency; the delta head + parse-side Matryoshka waist at
**factor-exact 0.748**, errors **100% symbolically detectable / zero silent**, blame
delete-one = honest negative recall 0.034 → add-back sweep + neural tiers MANDATED).
The ALTERNATION LOOP itself (notebook/NACK cycles, Bricks A/C) is NOT yet built. The
validated deducer is untouched and remains the regression anchor.

### 8.0 The ground under this design (the 2026-06-26 settlement — still binding)

**The two jaws:** CONSTRUCTION (NL → factor graph) + SOLVING (factor graph → answer).
SOLVING is **DONE + VALIDATED** — symbolic search solves clean hard-constraint graphs exactly
and fast (Sudoku 5000/5000 at *median 0 decisions*). **Neural-guided clean-CSP search is
CLOSED** — the two-death-mode law: neural value-ordering needs DEEP tree AND value-sensitivity
AND no symbolic incumbent *simultaneously*, and no clean exact-propagatable CSP has all three
(Sudoku shallow; QCP value-symmetric; SAT/TSP/coloring have CDCL/LKH/DSATUR). Proven across 5
negatives — stop hunting clean-CSP solving wins. The deducer's role inside the Alternator
follows from this settlement: a **differentiable, general approximate-inference backend** —
(a) *critic* (solve the proposed graph → end-to-end training signal; can't backprop through
symbolic), (b) *format-definer* (its membership + inlet vocabulary IS the parser's output
target), (c) *soft-graph solver* (for the uncertain graphs NL parsing produces) — NOT a better
solver. The cheap oracle-upper-bound kill-gate (one-hot policy via `csp_core.policy_valorder`,
pure CPU) settles "can any neural ordering help here?" before building anything — reuse it.
Construction groundwork: `docs/phase1_construction_brief.md`.

### 8.1 The loop

Six breath cycles; each cycle = Phase 1 (PARSER) then Phase 2 (DEDUCER); a 7th breath
decodes the iteratively-built KV cache.

- **Phase 1 — PARSER (NL → factor-graph delta).** Llama-base 2048d L0–L3,
  weight-invariant across cycles. Consumes tokens + notebook state + NACK; emits a graph
  *delta* expressed in the two channels (registry + ball). Never commits values.
- **Phase 2 — DEDUCER.** The validated v98-lineage Pythia 1024d L0–L3 engine, exactly as
  in §1–§2. Consumes only (variables + factors + membership + inlet). **Never sees NL.**
- **TWO TRUNKS, NOT ONE.** Each trunk is weight-stable across all cycles; they are NOT a
  single shared weight set. (A "one shared trunk" phrasing is a known drift error — do
  not reproduce it.)
- **Progressive resizing across cycles:** coarse (global scaffold, rough connectivity) →
  fine (exact predicates, arithmetic inlet detail). All per-cycle variation is
  input-conditioned (§8.4) — the trunks run the identical program at different resolution,
  which is the anti-gradient-tug-of-war design.

### 8.2 The interface — exactly THREE objects (two channels + memory)

- **ONE canonical predicate registry** (SEMANTICS channel): the relation menu + learned
  centroids. One version, fixed meaning across all cycles. (Six per-cycle registry
  versions were considered and rejected: the interface language must not change meaning
  mid-conversation, or the notebook accumulates state written in six dialects.)
- **ONE Poincaré ball** (TOPOLOGY channel): parser-emitted, differentiable mask
  generation, hierarchical by construction. One version. Carries no relation content.
  §7's blocked-relaxation caveat applies unchanged — this is the hard research risk.
  **v0 (2026-07-05): the parser emits membership DIRECTLY; the ball sits behind a flag
  as a strictly-additive upgrade — alternation does not wait on unsolved geometry.**
- **ONE notebook** (TEMPORAL MEMORY): fed from the deducer's silhouette common mode —
  see §8.6 for WHICH physical read-point ("the waist" vs "the silhouette TAP" are two
  different objects; all existing evidence lives at the tap, and the tap — post-deduction
  state — is likely the right notebook source). Two write disciplines: an append-only **accumulate ledger** (committed
  facts — deduced assignments, verified factors; remove at READ, never from state) and a
  **replace scratch** (provisional state, active hypotheses; overwritten each cycle).
  Deductions are monotone; hypotheses are not.

**Do NOT conflate the ball with the notebook.** Topology ≠ memory. Three interface
objects, not two. ("One Ball = the notebook waist" is a known drift error.)

### 8.3 The TCP handshake (what justifies the alternation)

- **SYN** — parser emits the cycle's graph delta.
- **ACK** — deducer returns settled state via the notebook.
- **NACK** — a deducer contradiction is routed *backward* as a localized signal; the next
  parse cycle **re-transmits** (re-parses) the offending region of the graph. This is the
  factor-graph error-localization role reborn (the 97.8%-localization heritage). (v0 needs
  no learned back-route: symbolic per-factor VIOLATED flags as parse-cycle input — §8.7 #3.)
- Sequence number = breath index. **Alternation earns its cost only if the NACK path
  works** — without back-pressure, staged parse-then-solve is strictly simpler and should
  be the fallback.

### 8.4 The zero-LoRA null hypothesis

**No LoRAs anywhere is the null.** Same weights, different conditioning (notebook + NACK)
→ different per-cycle behavior (portable principle: explicit self-feedback is the escape
valve). The parser's input is genuinely different every cycle — blank notebook + raw
tokens at cycle 1; rich ledger + scratch + localized NACK at cycle 3 — so weight mutation
may be unnecessary. Fallback ladder if the null fails:

1. Small-rank (≤16) LoRA on the **PARSER only**, absorbing band differences.
2. If the deducer ever needs per-cycle flavor: LoRA on **waist/readout projections
   only** — the four deducer layers stay naked-shared, always (v108 lesson: breath
   variation works through gating and masks, never trunk weights).

### 8.5 The perceiver, reborn NARROW (monitor, not engine)

Perceiver-as-CORE remains retired (5× refuted; the §6 attention-bootstrap law explains
why). Brick-1 (2026-06-16, `docs/perceiver_poincare_design.md` §9) validated that a small
latent bank *breathes* against Poincaré anchors (membership 0.785/0.883 vs 0.008
baseline; select_norm ≫ uniform floor). The revived role is deliberately small — a
component too tiny to compute the answer, so it cannot become a third gradient faction:

- **Session monitor** (the TCP connection-state machine): persistent latents
  cross-attending the parse output and the waist; tracks SYN/ACK/NACK, measures session
  convergence, decides retransmission. Neither phase can be this observer — each sees
  only its own side.
- **Spectral segmenter**: the waist silhouette superposes ~4–5 step signatures; the
  latents act as **learned matched filters** — a latent's attention over the silhouette
  IS the segmentation; its cosine to registry centroids IS the classification; the
  unmatched residual IS the NACK payload. This replaces the speculative analytic
  machinery (impulse "ringing", band masking, Laplace/Fourier decomposition) with one
  learned component that has a validated ancestor. **SCOPE (2026-07-05):**
  step-segmentation-vs-registry is inherently a PARSE-side task (steps live in the NL;
  no parse-side silhouette exists yet). The DEDUCE-side silhouette's measured role is
  session health + NACK localization: late-breath belief-JSD flags wrong cells at AUC
  0.687, gold-free, per-cell (§8.8 Brick-B results).
- **Hosts the global-broadcast latents** (the spatial channel — v300's latents-28–31
  role): within-pass broadcast between distant graph regions + toxic-noise lightning rod,
  keeping non-local chaos out of the invariant deducer trunk. **Notebook = TEMPORAL
  memory; global latents = SPATIAL broadcast.** Distinct jobs; neither subsumes the other.

### 8.6 The waist schedule (Matryoshka, 512→128) — and the waist-vs-tap split

**"The waist" is TWO different objects — do not conflate them (2026-07-04):**
- **THE WAIST (built, DORMANT):** an in-loop bottleneck in the deducer forward
  (`factor_graph_engine.py`, `FG_WAIST`): 1024→**256**→1024 after **L1** (the v38 B-field
  site), runs INSIDE every breath, convex gate init-closed (sigmoid(−8)≈0 = exact
  pass-through; byte-identical off). **No checkpoint has ever trained it** — parked NOT
  for lack of signal but because both adversarial reviewers caught the objective as a dud
  before firing (classify = discriminative trap: the rep learns to REPORT validity,
  argmax unchanged; attract = wrong geometry: pulls toward graph-identity in raw space).
  See `memory/project_waist_build_objective_finding.md`. It shapes COMPUTATION — the
  Matryoshka handicap site if revived.
- **THE SILHOUETTE TAP (probe point — where ALL existing silhouette evidence lives):**
  the final-breath readout-LN read (the dart captures). A passive tap on post-deduction
  state — likely the right NOTEBOOK source (§8.2), NOT a bottleneck.

The valid/invalid separation numbers, labeled correctly (coloring darts; instance-identity
DECONFOUNDED unless noted): raw-uncentered **0.755** (CONFOUNDED — not the target);
PCA-linear floor **0.658** (d=256); learned-nonlinear **0.85** (the Jun 22
`learned_waist_gate` GREEN light).

**Parse-side Matryoshka answer (2026-07-06, measured):** the Phase-1 delta head at
width 128 ≈ width 512 on EVERY head (factor exact 0.724 vs 0.748; op/type flat; the
~2pt cost sits in the fine-detail target/member heads). The structure a parser extracts
is intrinsically LOW-DIMENSIONAL, uniformly — aggressive waist scheduling needs no
head-aware carve-outs. Two silhouettes, both measured to compress well, for different
reasons (deduce-side: the 0.85 nonlinear read sharpens under compression; parse-side:
uniform 128d survival).

The SPEC (unbuilt): one 512d waist, importance-ordered dims, a scheduled mask exposing a
prefix (128 → 512). Two schedule axes, each a separate brick: over **training time**
(deliberate handicap — train hard, race easy) and over **breath cycles** (coarse cycles
narrow, fine cycles wide — the polarized-band idea applied to waist dims). No projection
re-instantiation; evaluation at any prefix width is free. Companion instrument: the 0.85
verifier-substitute read **as a function of prefix width** — decides whether the
Anna-Karenina signature is intrinsically low-dimensional. (The spec's 512d readout waist
vs the built 256d mid-stack mechanism must be reconciled at build time.)

### 8.7 Unvalidated load-bearing assumptions (say them out loud)

1. **Input-conditioning suffices** — the zero-LoRA null may fail; the parser may not
   switch bands from conditioning alone.
2. **Matched-filter segmentation works** — learned latents factoring a superposed
   silhouette into step components is plausible, not demonstrated.
3. **The fully-DIFFERENTIABLE NACK is unbuilt — but a v0 NACK needs no new learning
   machinery (downgraded 2026-07-04).** Most of a NACK already exists validated: the
   calibration head is the ACK/session-health scalar (tracks difficulty honestly —
   sudoku 0.76→0.42→0.29), and factor-level LOCALIZATION is free + symbolic (the search
   tier's predicate returns VIOLATED per factor). v0 NACK = verifier flags violated
   factors → encoded as input features to the next parse cycle; the parser learns to
   RESPOND (ordinary supervised conditioning — the sanctioned division of labor: symbolic
   asserts facts, neural adjusts behavior). The differentiable back-route remains
   research, but the alternation's existence proof no longer depends on it.

### 8.8 The brick ladder (each earns the next; kill criteria BEFORE firing)

- **Brick-0** (one session, ZERO new architecture): frozen Brick-1-style latents read an
  *existing* trained silhouette-tap common mode (captures on disk are COLORING darts —
  `.cache/dart_silhouettes_fg_coloring_k16.npz`; KenKen = re-run the capture harness on
  `fg_kenken_k16_reg`). Pass bar: beat the 0.658 PCA-linear floor AND the stronger
  analytic null — FIXED matched filters (projections onto single-factor prototype
  subspaces); target the 0.85 learned-nonlinear read (§8.6 labels). Fail → rework the
  matched-filter story before wiring anything.
- **Brick-A**: the zero-LoRA parser ablation (upstream of everything; decides the
  parameter budget).
- **Brick-B** (UNGATED from the alternation, 2026-07-04 — the BirdNET move): spectral
  segmentation does NOT need the alternation running. Synthesize supervision by running
  COMPOSED problems (known constituent factors) through the trained engine — ground truth
  is free because we control the generator. Protocol order: (1) capture UNPOOLED K×dim
  trajectories (solo-factor + composed; the existing dart capture pools away both axes),
  (2) LINEARITY CHECK — is a composed silhouette ≈ the sum of its constituents'? Do NOT
  assume it (audio superposes linearly by physics; residual streams don't have to — if
  grossly nonlinear, linear matched filters need rework), (3) train the segmenter to
  recover constituents. Center per-instance or mix across instances, else the segmenter
  cheats on graph identity (the 0.755-vs-0.658/0.85 confound).
  **RESULTS (2026-07-05, deduce-side — `scripts/capture_silhouette_trajectories.py`,
  data `.cache/silhouette_traj_kenken_reg.npz`, capture-once schema):** steps (1)+(2)
  DONE on the deducer silhouette. LINEARITY REFUTED (residual ratio 0.72 / cos 0.76,
  stable across breaths → compose PROBLEMS, not silhouettes). NO temporal banding in
  RESIDUAL space (the carrier never stops moving; delta-settle degenerates). Temporal
  structure LIVES IN BELIEF SPACE: early ~4-breath transient, givens settle first, hard
  cages deliberate to the last breath, and late-breath belief-JSD → wrong-cell AUC 0.687
  (gold-free, per-cell — the deduce-side NACK-localization signal). Variant accs also a
  finding: rowcol-only 0.52, cage-only 0.13 ≈ base 0.13 — KenKen's constraint value is
  the row/col∩cage INTERSECTION (the nonlinearity, stated as a feature). **These are
  DEDUCE-side verdicts, not parse-side conclusions** — the parser's axis is token
  position, word problems are banded by narrative order, and sentences are loosely
  coupled, so parse-side banding/linearity priors point the OTHER way; untestable until
  a Phase-1 waist exists (Brick-A era).
- **Brick-C** (REFRAMED smaller, 2026-07-04): v0 = does the parser USE symbolic NACK
  features (§8.7 #3)? The differentiable back-route is the stretch goal, not the
  alternation's existence proof.

Budget lines: 7900 XTX / 24GB; K=16 known-good, K=28 hangs the AM JIT; fp32 THINK path.

**Resist ALL domain-specific code in the engine/core.** New domains enter through the
predicate + bridge (search) and the membership + inlet (deducer) — never the core.

### 8.9 The prior framing (2026-06-20) this instantiates

1. **The frontier — a problem where symbolic propagation isn't enough:** soft /
   probabilistic / learned / **NL-specified** constraints, keeping the factor-graph
   abstraction. The Alternator is this thread's chosen instantiation.
2. **Minimize retraining when switching tasks (weight-side generality):** the zero-LoRA
   null + one-registry/one-ball decision is this thread's strongest form — a single
   invariant weight set per trunk, constraint semantics fed as INPUT.

**Key memory notes:**
- `docs/phase1_skeleton_spec.md` — **the Phase-1 build plan (2026-07-05)**: frozen
  Llama-3.2-1B L0–L3 + slot-based delta head + parse-side 512→128 waist tap;
  KenKen-in-words templates with span↔factor gold; Brick-A/Brick-C-v0 measurables;
  the corrected two-jaws framing; NO external API calls.
- `docs/perceiver_poincare_design.md` — Brick-1 results (the latents breathe) + the
  Cathedral spec; ancestor of the §8.5 monitor role.
- `docs/state_of_mycelium.md` — the brainstorm-ready stock-take (solid ground / spec /
  refuted / open questions).
- `memory/project_neural_guided_search_clean_csp_closed.md` — the two-death-mode law; clean-CSP solving closed.
- `memory/project_sudoku_search_tier_solve.md` — search tier 100% on Sudoku (median 0 decisions).
- `memory/project_multitask_generality_works.md` — the weight-side generality grail (won at parity).
- `docs/phase1_construction_brief.md` + `docs/session_2026_06_26_solving_closed_phase1_pivot.md` — the Phase-1 pivot this design descends from.
- `memory/project_phase2_kenken_generality_proven.md` — generality proven on KenKen, zero core edits.
- `memory/project_phase0_general_search_core.md` — the predicate-driven core + the SAT-via-bridge proof.
- `memory/project_pathb_search_coloring_result_jun19.md` — symbolic dominates clean CSPs (the honest negative).
- `memory/project_factor_graph_two_channels.md` — topology vs semantics; registry ≠ Poincaré ball.
- `memory/project_distributed_deduction_scales_parallel.md` — the parallel-deduction superpower.
- `memory/project_radial_depth_thesis_refuted.md` — the refuted deep-prize.
- `memory/project_waist_build_objective_finding.md` — the waist mechanism (built, dormant) +
  why it was parked (objective dud, not signal absence); the 0.755/0.658/0.85 AUC labels.
- `memory/project_kenken_generalization_fixed.md` — curriculum + v45 reg unlock; `fg_kenken_k16_reg`.
- `memory/feedback_offer_engineering_critique.md` — push back before rubber-stamping.
- `memory/reference_tinygrad_am_quirks.md` — substrate laws.
