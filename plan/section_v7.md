# Section Y: From Moves to Openings — Strategy-Level Search in Mycelium v7

## Y.1 The Limits of Move-Level Search

A chess engine that evaluates individual moves — knight to f3, bishop to c4 — can play competent chess through brute-force search. But a grandmaster thinks differently. Before considering any individual move, the grandmaster recognizes the *position type* and recalls a *strategy*: "this is a closed Sicilian — I should aim for a kingside pawn storm." The strategy constrains which moves are worth considering. The grandmaster searches fewer positions than the engine but finds better ones, because the search is structured by strategic understanding.

Mycelium v6 is the chess engine. MCTS explores individual operations — which template? which operands? which dependency wiring? — evaluating each path by symbolic execution. This works, but the search is combinatorial: 3 template options × 3 operand bindings × 2 dependency wirings = 18 paths, most of them incoherent combinations that crash at sympy. The search finds correct solutions by exhaustion, not by understanding.

Mycelium v7 is the grandmaster. Before exploring any operation, a strategy selector identifies the *problem type* and retrieves a *strategy template*: "this is a complete-the-square problem — I should normalize the leading coefficient, add the square term, factor, and solve." The strategy constrains which operations to consider, in what order, with what dependency structure. The search space shrinks from dozens of random operation combinations to a handful of coherent plans.

## Y.2 Strategies as DAG Motifs

A strategy, in our formulation, is a recurring directed acyclic graph pattern — a subgraph topology that appears across many problems with the same dependency structure but different operands.

**Integration by parts** is not a single operation. It is a DAG motif:

```
ASSIGN(u) ──→ DIFFERENTIATE(u→du) ──→ SUBSTITUTE(uv - ∫v·du)
                                          ↑
ASSIGN(dv) ──→ INTEGRATE(dv→v) ─────────┘
```

**Complete the square** is a different motif:

```
NORMALIZE(coefficient) → ADD(square term) → FACTOR(perfect square) → SOLVE(variable)
```

**System of equations by substitution** is yet another:

```
SOLVE(eq₁ for x) → SUBSTITUTE(x into eq₂) → SOLVE(eq₂ for y) → BACK-SUBSTITUTE(y into x)
```

Each motif specifies a topology (which operations connect to which), slot types (what kinds of operands fill each position), and a characteristic dependency structure. The motif is a template of templates — a meta-pattern one level above the operation taxonomy.

These motifs are not designed by us. They are discovered by mining the DAG corpus that v6 produces — thousands of solved problems, each decomposed into an operation DAG, searched for recurring subgraph patterns. The same graph motif mining techniques used in bioinformatics (to find network motifs in protein interaction graphs) and social network analysis (to find community structures) apply directly. Mathematical reasoning strategies are network motifs in the space of computational DAGs.

## Y.3 Two-Level MCTS

The MCTS tree in v7 has two distinct branching regimes, each with different characteristics.

**Level 1: Strategy branching.** The root of the tree is the raw problem text. The strategy selector (C0) proposes 2–4 candidate strategies, each with a confidence score. Each strategy branch expands into a complete DAG skeleton — a pre-specified topology with empty operand slots. The branches at this level are *structurally independent*: completing the square and applying the quadratic formula are entirely different computational plans, sharing no intermediate results.

**Level 2: Operation branching (within strategy).** Within each strategy branch, the operation-level search from v6 proceeds as before: C2 fills template slots (usually constrained to 1–2 options by the strategy), C3 extracts operands via beam search, C5 confirms dependency wiring. But the search space is dramatically reduced because the strategy has already determined which operations to perform and in what order. C2's job narrows from "which of 30 templates?" to "fill this specific slot in the strategy DAG."

The tree structure:

```
Root: "Solve x² + 6x + 5 = 0"
│
├── Strategy: FACTOR (confidence 0.55)
│   ├── C2: {FIND_FACTORS, APPLY_ZERO_PRODUCT}
│   │   ├── C3: (x+1)(x+5) → x = -1, -5    ✓ sympy validates
│   │   └── C3: (x+2)(x+3) → x = -2, -3    ✗ sympy rejects
│   └── Result: {-1, -5}
│
├── Strategy: COMPLETE_THE_SQUARE (confidence 0.25)
│   ├── C2: {ADD_TERM, FACTOR_SQUARE, SOLVE}
│   │   └── C3: (x+3)² = 4 → x = -1, -5    ✓ sympy validates
│   └── Result: {-1, -5}
│
└── Strategy: QUADRATIC_FORMULA (confidence 0.20)
    ├── C2: {IDENTIFY_COEFFICIENTS, SUBSTITUTE_FORMULA}
    │   └── C3: (-6 ± √16)/2 → x = -1, -5  ✓ sympy validates
    └── Result: {-1, -5}
```

Three strategies. All three produce {-1, -5}. Maximum confidence.

## Y.4 Strategy-Level Agreement

When multiple operation-level paths within the *same* strategy converge on the same answer, the evidence is moderate — different operand bindings led to the same result, but the reasoning structure was identical.

When multiple *strategies* converge on the same answer, the evidence is strong. Factoring, completing the square, and the quadratic formula are structurally independent computational plans. They share no intermediate computations, no common templates, no dependency edges. Agreement between them is the mathematical equivalent of independent experimental replication.

In self-consistency sampling with monolithic models, N chain-of-thought samples may appear different on the surface (different wording, different intermediate steps) while following essentially the same reasoning path. The model has one way of thinking about the problem, expressed N times with variations. True independence is rare because the same weights produce the same computational strategy.

Strategy-level agreement in v7 is genuinely independent. Factoring performs trial-and-error on factor pairs. The quadratic formula substitutes coefficients into an algebraic identity. Completing the square applies a geometric transformation. These are not surface variations — they are fundamentally different algorithms that happen to solve the same class of problems. When they agree, Bayesian confidence updates multiplicatively rather than additively.

We define a **strategy agreement score**: for a problem where k out of n strategy branches produce the same answer, confidence scales as 1 - (1-p)^k, where p is the base accuracy of each strategy. Two independent strategies at 70% individual accuracy yield 91% joint confidence. Three yield 97.3%. The combinatorial power of independent verification makes strategy-level MCTS far more powerful than operation-level MCTS at the same computational budget.

## Y.5 Adaptive Search at Two Scales

Problem difficulty determines how deeply the tree is explored at each level.

**Easy problems** (Level 1–2) trigger no search at either level. C0 assigns a single strategy with >90% confidence. C2 fills slots with >90% confidence. C3 produces a single beam. Total: one strategy, one path, one sympy evaluation. Latency: ~15ms.

**Medium problems** (Level 3) trigger strategy-level search but minimal operation-level search. C0 proposes 2–3 strategies. Within each, the operations are mostly determined by the strategy template. Total: 2–3 strategies × 1–2 operation variants = 3–6 paths. Latency: ~40ms.

**Hard problems** (Level 4–5) trigger search at both levels. C0 proposes 3–4 strategies, none with high confidence. Within each strategy, C2 and C3 maintain multiple hypotheses. Total: 3–4 strategies × 3–5 operation variants = 12–20 paths. Latency: ~150ms.

The decoherence threshold θ from the wave function collapse framework operates at both levels independently. Strategy-level θ controls when C0 collapses to a single strategy. Operation-level θ controls when C2/C3 collapse within a strategy. A problem can be strategically ambiguous (explore multiple strategies) but operationally clear (each strategy has obvious operand choices), or vice versa.

This two-scale adaptivity mirrors expert mathematical problem-solving. A difficult integral may be strategically ambiguous ("should I use u-substitution or integration by parts?") but once the strategy is chosen, the operand choices are determined. A routine algebra problem may be strategically obvious ("clearly factor") but operationally tricky ("which factor pair?"). The search focuses where the uncertainty actually lives.

## Y.6 The Grandmaster's Advantage

The grandmaster does not see more moves than the engine. The grandmaster sees fewer moves, better organized. Strategic understanding acts as a compression of the search space — from the combinatorial explosion of all possible move sequences to the structured exploration of a few coherent plans.

Mycelium v7 achieves the same compression. The raw operation-level search space for a 5-step MATH problem is approximately 30⁵ ≈ 24 million possible operation sequences (30 templates, 5 steps). Strategy-level search reduces this to perhaps 4 strategies × 3² operand variants ≈ 36 paths — a reduction factor of nearly one million.

This compression is possible because mathematical reasoning strategies are highly structured and relatively few in number. Just as the space of chess openings (a few hundred well-known lines) is vastly smaller than the space of 10-move sequences (~10²⁹ possibilities), the space of mathematical strategies (~20–50 distinct motifs) is vastly smaller than the space of operation sequences.

The compression comes not from discarding possibilities but from recognizing structure. The grandmaster and the engine explore the same game tree. The grandmaster simply knows where in the tree to look.
