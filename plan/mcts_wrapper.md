# Mycelium v6: MCTS as a Search Layer Over the Pipeline

## The Core Idea

The six specialist models (C1–C6) each produce a distribution over possible outputs, not a single deterministic answer. Today we take the argmax at every step — the most likely segmentation, the most likely template, the most likely expression — and chain them together. If any step picks wrong, the whole pipeline fails.

MCTS (Monte Carlo Tree Search) wraps the existing pipeline as a search layer. Instead of committing to one path, it explores the tree of alternatives, using sympy execution as a free verifier to prune bad branches. It's not a 7th model — it's search infrastructure that uses the six models as move generators and sympy as the judge.

---

## Why Math is Uniquely Suited for MCTS

Most domains lack a reliable reward signal at the leaf. In code generation, you need test cases. In text generation, quality is subjective. In math, you get something extraordinary: **deterministic binary feedback for free.**

Sympy either executes and produces a valid answer, or it crashes. There's no gray area. This means:

- No learned value function needed (unlike AlphaGo)
- No human evaluation in the loop
- No reward model to train and calibrate
- Just: did the expression graph execute cleanly and match a plausible answer format?

This makes the MCTS rollout trivially cheap. Generate a candidate path through the pipeline, hand it to sympy, get pass/fail. The expensive part is running the models, not evaluating the result.

---

## Tree Structure

Each level of the tree corresponds to a pipeline stage where the model's confidence is below a threshold. High-confidence decisions don't branch — they pass through as-is.


**Key insight:** Most branches die at sympy execution. Wrong template + wrong expression = sympy crash. This natural pruning keeps the effective branching factor small even when theoretical branching factor is large.

---

## Where Branching Matters Most

Not all pipeline stages contribute equal uncertainty. Based on our error attribution work:

| Stage | Branching Factor | Why |
|-------|-----------------|-----|
| C1 (Segmenter) | Low (1–2) | Clause boundaries are usually unambiguous. IO tagging at 86.5% F1 means few alternatives worth exploring. |
| C2 (Classifier) | **High (3–8)** | "What operation is this?" is genuinely ambiguous. A clause like "find when they're equal" could be SOLVE, SUBSTITUTE, or SET_EQUAL. This is the primary branching point. |
| C3 (Extractor) | **High (2–5)** | "What are the operands?" has multiple valid interpretations, especially with implicit operands. The merged-span approach reduces this, but doesn't eliminate it. |
| C4 (Bridging) | Medium (1–3) | Implicit operations are either needed or not. Limited options when they are. |
| C5 (DAG Wiring) | Low (1–2) | Dependency structure is usually clear from the problem. Rarely ambiguous. |
| C6 (Goal Resolver) | Low (1–2) | Answer format is almost always unambiguous from the question. |

**Practical implication:** MCTS search concentrates on the C2 × C3 subspace. A tree exploring 5 C2 options × 3 C3 options × 1 everything else = 15 candidate paths. At ~3ms per sympy execution, that's 45ms of verification. Negligible.

---

## Adaptive Search Depth

Not every problem needs search. The pipeline's own confidence scores determine how much exploration to do.

**Level 1–2 problems (easy):**
- C2 confidence typically > 90% on a single template
- C3 extraction is usually unambiguous
- Search depth: 0 (just take argmax, single pipeline pass)
- Latency: ~15ms (six 0.5B forward passes)

**Level 3 problems (medium):**
- Some C2 ambiguity (top-2 templates within 20% of each other)
- Search depth: 5–10 candidate paths
- Latency: ~50ms

**Level 4–5 problems (hard):**
- Significant C2/C3 ambiguity
- Multi-step DAGs where errors compound
- Search depth: 20–50 candidate paths
- Latency: ~200ms
- This is where MCTS delivers the most accuracy gain

**The confidence threshold is tunable.** Set it high (0.95) for speed, low (0.5) for accuracy. Can even be set per-problem based on detected difficulty level.

---

## Self-Consistency Through Execution

A powerful property emerges when multiple valid paths exist: **if different template/expression combinations all execute to the same answer, confidence is very high.**

This is essentially self-consistency (Wang et al., 2022) but over structured reasoning paths rather than free-form chain-of-thought. The key difference:

| | Standard Self-Consistency | Mycelium MCTS |
|---|---|---|
| Sampling | Regenerate entire CoT | Explore pipeline alternatives |
| Verification | Compare final answers | Sympy validates each path |
| Cost | N full LLM generations | N sympy executions (near-free) |
| Diversity | Temperature sampling (random) | Structured branching (systematic) |

Standard self-consistency with a 70B model: sample 40 CoTs × 2000 tokens each = 80K tokens generated. Mycelium MCTS: 40 sympy executions × 0 tokens generated = 0 tokens. The "generation" was already done by the six tiny forward passes.

---

## Failure Modes and Mitigations

**Problem: Combinatorial explosion on complex problems**
Some problems have 6+ clauses × 5+ templates each = millions of paths. 
*Mitigation:* Beam search, not exhaustive. Keep top-k paths by joint probability at each level. Prune paths where sympy crashes early (partial execution). Budget of 100 candidate paths max.

**Problem: Multiple valid answers**
Some MATH problems have multiple correct forms (e.g., equivalent fractions).
*Mitigation:* Sympy normalization — simplify all candidate answers to canonical form before comparison. Use sympy.simplify() and sympy.nsimplify() to detect equivalence.

**Problem: Sympy can't execute everything**
Some operations (geometric reasoning, combinatorial arguments) don't have clean sympy representations.
*Mitigation:* This is a pipeline limitation, not an MCTS limitation. For problems where sympy can't execute, MCTS gracefully degrades to confidence-weighted argmax (same as no-search baseline).

**Problem: Slow for hard problems**
50 candidate paths × 6 model forward passes each = 300 forward passes.
*Mitigation:* Shared computation. C1 and C6 run once (low branching). C5 DAG wiring is cheap. The real cost is C2 × C3 alternatives, and these are tiny 0.5B models. 300 forward passes through 0.5B ≈ 5 forward passes through 30B. Still an order of magnitude cheaper than a frontier model doing CoT.

---

## Implementation Phases

**Phase 1: Greedy baseline (current)**
Take argmax at every stage. No search. This is what we're evaluating now against the 17.2% Track 1 baseline.

**Phase 2: Top-k with majority vote**
Generate top-3 from C2, top-2 from C3. Execute all 6 combinations. Majority vote on valid results. Minimal code change — just run the pipeline k times with different inputs.

**Phase 3: Full MCTS with adaptive depth**
Confidence-based branching. Beam search with pruning. Per-problem search budget based on difficulty. This is the production system.

**Phase 4: Learning from search**
Use MCTS results to improve the base models. When MCTS finds that the argmax path was wrong but the 3rd-ranked path was right, that's a training signal. Retrain C2/C3 on the corrected labels. Over time, the models get better and MCTS has less work to do — the search "teaches" the models.

---

## The Big Picture

The six specialists provide the moves. Sympy provides the evaluation. MCTS provides the search. Together: a system that can explore structured reasoning paths at negligible cost compared to having a large model explore by generating thousands of tokens.
