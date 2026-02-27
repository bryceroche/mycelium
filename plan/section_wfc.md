# Section X: Inference as Progressive Wave Function Collapse

## X.1 The Superposition Problem

A decomposed reasoning pipeline faces a fundamental tension: each component must pass a decision to the next, but premature commitment at any stage discards information that downstream components need. If the segmenter commits to a single span boundary, the classifier never sees the alternative segmentation that would have led to the correct template. If the classifier commits to a single operation type, the extractor never considers the operand binding that only makes sense under an alternative classification.

In a monolithic model, this problem does not arise — all decisions are made jointly in a single forward pass, with implicit cross-talk between what would be our C1–C6 functions. Decomposition purchases diagnosability and efficiency at the cost of this joint inference. The question is how to recover it.

We propose framing pipeline inference as **progressive wave function collapse**: the input problem begins in superposition over all possible (segmentation, classification, extraction, dependency) tuples, and each specialist performs a partial measurement that narrows the distribution without forcing premature collapse. Full collapse — commitment to a single deterministic computation — is deferred to the symbolic executor, which is the only stage that requires it.

## X.2 Low-Resolution Pointers

The IAF attention data that supervises our pipeline is inherently soft. When the 7B teacher generates a computation step and attends to input tokens, the attention distribution is not binary. A typical pattern shows primary mass on 2–3 tokens (the core operands), secondary mass on 5–10 tokens (contextual modifiers, variable names, units), and a long tail over the rest of the input. Thresholding this distribution into a hard span discards the secondary mass — which often contains exactly the information needed to disambiguate between competing operation templates.

We introduce the concept of **low-resolution pointers**: instead of producing a crisp span boundary, C1 outputs a relevance score for each input token. High-scoring regions are likely operational clauses. But the boundaries are fuzzy, and tokens outside the primary region retain non-zero relevance. This mirrors the teacher's own attention pattern — the 7B model does not attend to a clean span either. It attends to a distribution. The student should preserve that distributional character.

In practice, we implement low-resolution pointers not as continuous distributions (which would require architectural changes to every downstream model) but as **multiple discrete hypotheses**. C1 proposes 2–3 candidate segmentations with associated confidence scores, each representing a different "measurement outcome." This discrete sampling of the underlying continuous uncertainty preserves the benefits of soft representation — downstream models see multiple possibilities — while maintaining the clean input interface that each specialist was trained on.

## X.3 Progressive Collapse Through the Pipeline

Each specialist narrows the space of possible interpretations without eliminating alternatives:

**C1 (Segmentation) — Position measurement.** The uniform prior over input tokens collapses into 2–3 candidate segmentations. The superposition over "where are the operational clauses" has been narrowed but not resolved. Analogous to measuring a particle's position with finite detector resolution: the probability density concentrates in a region but retains spread.

**C2 (Classification) — Type measurement.** For each candidate segmentation, C2 produces a distribution over operation templates via its softmax output. We retain the top-k templates (typically k=3) rather than taking the argmax. The joint superposition is now over (segmentation × template) pairs: if C1 proposed 3 segmentations and C2 proposes 3 templates each, the system maintains up to 9 hypotheses. In practice, confidence-weighted pruning reduces this — high-confidence C2 predictions collapse to a single template, while ambiguous clauses maintain multiplicity.

**C3 (Extraction) — Value measurement.** For each surviving (segmentation, template) pair, C3's generative head produces candidate expressions via beam search. Each beam represents a different operand binding — a different answer to "which numbers from the problem text belong in this expression." The extraction beam width adapts to confidence: a clause containing explicit operands ("5 apples at $2 each") collapses to a single expression, while a clause with implicit operands ("sold half as many") maintains 2–3 candidates.

**C5 (Dependencies) — Entanglement measurement.** The dependency resolver constrains which step combinations are mutually compatible. If step 3 depends on step 1's output, then only (step 1, step 3) pairs where step 1 actually produces a result that step 3 can consume are valid. This is analogous to measuring entangled particles: the measurement of one instantly constrains the other. Dependency resolution propagates constraints between hypotheses, pruning incompatible combinations without requiring explicit search.

**Sympy (Execution) — Final collapse.** The symbolic executor is the decoherence event. It accepts a fully specified expression graph — concrete spans, concrete templates, concrete operands, concrete dependencies — and either produces a valid result or fails. There is no superposition at this stage. Every surviving hypothesis is independently executed, and the system selects among valid results.

## X.4 MCTS as Quantum Branching

Monte Carlo Tree Search provides the mechanism for exploring the hypothesis space efficiently. Rather than exhaustively enumerating all (segmentation × template × extraction × dependency) combinations — which grows combinatorially — MCTS samples paths through the hypothesis tree guided by component confidence scores.

The tree structure mirrors the collapse sequence.

Each path from root to leaf represents one complete collapse — one fully classical interpretation of the problem. Sympy evaluation provides a binary reward signal: valid execution (the measurements were mutually consistent) or failure (the measurements were incompatible). No learned value function is required.

**Adaptive search depth.** The branching factor at each level is determined by the component's confidence distribution. When C2 assigns >90% probability to a single template, that level contributes branching factor 1 — it has already collapsed. When C2 is ambiguous (60/30/10), branching factor is 3. The total number of paths explored thus adapts automatically to problem difficulty: easy problems with high-confidence components explore 1–3 paths; hard problems with pervasive ambiguity explore 20–50.

We define a **decoherence threshold** θ that controls how aggressively the pipeline forces collapse. At θ = 0.95, any component with >95% confidence on its top prediction collapses immediately (single hypothesis); components below this threshold maintain top-k hypotheses. Lowering θ increases exploration at the cost of more sympy evaluations. In practice, θ = 0.85 provides a good balance: easy problems (MATH Level 1–2) typically trigger no branching at all, while hard problems (Level 4–5) explore 15–40 candidate paths.

## X.5 Constraint Propagation as Entanglement

A crucial advantage of the multiple-hypothesis framework is that constraints propagate between components, pruning the search space without explicit enumeration.

**Forward constraints (measurement narrows future possibilities):**
- C2 classifying a clause as QUADRATIC_SOLVE constrains C3 to extract polynomial coefficients, not arbitrary operands
- C5 determining that step 3 depends on steps 1 and 2 constrains C3's extraction for step 3 to include outputs from prior steps
- C6 determining the answer must be an integer prunes any MCTS path whose sympy output is non-integral

**Backward constraints (downstream evidence revises upstream beliefs):**
- If C3 can only find operands consistent with SUBSTITUTE (not SOLVE), this revises C2's posterior — SUBSTITUTE becomes more likely regardless of C2's original softmax
- If all MCTS paths through segmentation A fail at sympy while paths through segmentation B succeed, segmentation B's posterior increases

In the factor graph formulation, each component is a factor node, the hypothesis variables are connected by compatibility edges, and belief propagation iteratively updates marginal probabilities until convergence. A single forward pass provides the initial beliefs; constraint propagation refines them. In practice, one forward pass plus MCTS exploration approximates this inference without requiring explicit message-passing implementation — the "messages" are implicit in which paths survive sympy validation.

## X.6 Computational Cost Analysis

The multiple-hypothesis approach appears expensive but is remarkably efficient due to three properties of the architecture:

**Shared prefix computation.** When two hypotheses share the same segmentation but differ in template assignment, C1's computation is shared. The incremental cost of exploring a second template is one additional C2 + C3 + C5 forward pass through 0.5B models — approximately 3ms on modern hardware.

**Early termination via sympy.** Most incorrect hypotheses fail at sympy execution — they produce malformed expressions, division by zero, or type errors. These failures are detected in microseconds, pruning entire subtrees instantly. In our experiments, >70% of candidate paths are eliminated by sympy before requiring full evaluation.

**Agreement as confidence.** When multiple surviving paths produce the same final answer via different reasoning routes, confidence is very high. This is structurally analogous to self-consistency sampling but over structured reasoning paths rather than free-form chain-of-thought generations. The critical difference is cost: self-consistency with a 70B model requires N complete autoregressive generations (thousands of tokens each); our approach requires N sympy evaluations (microseconds each) after a single set of 0.5B forward passes.

For a typical MATH Level 3 problem with moderate ambiguity (3 segmentations × 2 templates × 2 extractions = 12 paths), total inference cost is approximately: 6 forward passes through 0.5B models (~18ms) + 12 sympy evaluations (~0.1ms) = ~18ms. This is roughly 1000× cheaper than a single 70B chain-of-thought generation for the same problem.

## X.7 Connection to the Spectral Framework

The wave function collapse framework extends the spectral decomposition analogy introduced in Section 1.1. If IAF extraction is the prism that separates white light into component frequencies, then multiple-hypothesis inference is **observing the spectrum through a finite-resolution spectrometer**.

A perfect spectrometer (infinite resolution, zero noise) would identify each spectral line exactly — one template, one operand binding, one dependency structure. A real spectrometer sees broadened lines that overlap at the edges. Two nearby spectral lines (e.g., SOLVE vs SUBSTITUTE for the same clause) appear as a single broadened peak until higher resolution (more evidence, more context, sympy verification) separates them.

MCTS search provides that higher resolution. Each sympy evaluation is like increasing the spectrometer's resolving power — it distinguishes between hypotheses that looked identical at coarser resolution. Easy problems have well-separated spectral lines that require no additional resolution. Hard problems have crowded spectra where many operations are plausible, requiring search to resolve the ambiguity.

The decoherence threshold θ is, in spectral terms, the minimum peak separation below which we decline to resolve and instead explore both possibilities. High θ means we accept blurred peaks (faster, risking misidentification). Low θ means we always resolve (slower, more accurate). The optimal θ depends on the spectral density of the problem domain — GSM8K's sparse spectrum permits high θ, MATH's dense spectrum requires lower θ.
