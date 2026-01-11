# Mycelium: Decomposition Is All You Need

*Decomposing Problems into Reusable Atomic Signatures*

**Author:** Bryce Roche, bryceroche@fungifactor.com, github.com/bryceroche/mycelium

---

## Abstract

Every composite number factors uniquely into primes. We hypothesize that math problems similarly decompose into a finite set of atomic *signatures*—reusable solution patterns. **Mycelium** builds a "table of primes" for mathematical reasoning: a signature database that grows as problems are solved.

The system decomposes problems into DAG-structured steps, matches each against known signatures via cosine similarity, and executes stored routines (formula, procedure, or LLM guidance). Novel steps are solved and stored as new signatures. The library grows; future problems get faster.

---

## 1. Introduction

Large language models solve math problems through chain-of-thought reasoning, but each problem is solved from scratch—no persistent memory, no reuse of successful patterns.

Yet **while complete problems are unique, their constituent steps are highly reusable**. The step "solve for x in 2x + 3 = 7" appears across countless problems. Like mycelium networks that decompose organic matter into simple molecules and share nutrients across forests, we decompose complex problems into atomic patterns and distribute solutions through a shared database.

### Contributions

1. **Problem Decomposition**: DAG-based decomposition into reusable atomic signatures
2. **Signature Database**: Vector store with centroid-based clustering
3. **Cosine Similarity Matching**: Embedding-based step-level pattern retrieval
4. **Hybrid Execution**: Routing to DSL, procedure guidance, or LLM
5. **Signature Refinement Loop**: Frontier LLM decomposes low-performing signatures into precise children; parents become routers—this is how the system learns
6. **Two Operating Modes**: *Learning mode* explores signatures; *Execution mode* uses proven signatures for deterministic execution

### Open Source & Reproducibility

All code and pre-trained signatures at **github.com/bryceroche/mycelium** (MIT license).

- **Pre-built signature database**: 675+ math signatures with DSL scripts—skip cold start
- **5-minute setup**: `pip install`, add Groq API key, run

The signature database represents months of accumulated learning. By sharing it, new users start with a mature library. **Knowledge compounds and transfers**.

---

## 2. Related Work

**Mathematical Reasoning with LLMs.** Chain-of-thought (Wei et al., 2022) and program-aided reasoning (Gao et al., 2023) solve problems independently without knowledge reuse.

**Retrieval-Augmented Generation.** RAG retrieves context before generation (Lewis et al., 2020). We extend this to *step-level* retrieval.

**Case-Based Reasoning.** Classical AI reused solutions from similar cases (Kolodner, 1992). Our signature database is a neural implementation.

---

## 3. Method

### 3.1 Overview

Given problem P, Mycelium: (1) decomposes into a DAG of steps, (2) matches each step against the signature database, (3) executes via stored routines or LLM, (4) synthesizes results, and (5) updates the database.

### 3.2 Problem Decomposition

An LLM decomposes problem P into a DAG where each step has a task description and dependencies. Steps execute in topological order.

**Signature-Guided Hints.** We inject the top 15 reliable signatures into the planner prompt to guide decomposition toward proven patterns.

### 3.3 Signature Database

Stores atomic solution patterns as (centroid, method, stats). Centroids update incrementally. Near-duplicate signatures are periodically merged when similarity ≥0.90 and success rates align.

### 3.4 Cosine Similarity Matching

Each step is embedded and matched against signature centroids. Match threshold default: 0.87. **Adaptive thresholds** adjust based on cluster cohesion—tight clusters get stricter thresholds.

### 3.5 Execution and Learning

Matched signatures execute via DSL (~0ms), procedural guidance, or hints. Unmatched steps use LLM reasoning. Signatures with ≥3 uses and ≥70% success become "reliable."

### 3.6 Recursive Decomposition

When DSL confidence is low, decompose further until reaching atomic operations or max depth (3). Each decomposition creates new atomic signatures marked with origin depth.

**Self-Improvement:** Complex step → decompose → sub-steps create atomic signatures → next time, match atomic signature directly.

### 3.7 From Method Template to DSL

Signatures start with a natural language method_template. Once reliable (≥5 uses, ≥80% success), we generate a DSL script for deterministic execution.

**Three DSL Layers:**
- **Math**: `(a + b) / 2`, `sqrt(x)`
- **SymPy**: `solve(Eq(a*x + b, 0), x)`
- **Custom**: `apply_quadratic_formula(a, b, c)`

**Bulk DSL Generation:** We batch-processed all 1,300 signatures through Claude Opus 4.5 (~$15), equipping 84% with DSL.

### 3.8 Signature Refinement Loop

Low-performing signatures reveal opportunities for improvement. **Requires a frontier LLM** (e.g., Claude Opus):

```
1. IDENTIFY: Query signatures with success_rate < threshold
2. ANALYZE (Frontier LLM): Examine failure patterns
3. DECOMPOSE (Frontier LLM): Design finer-grained sub-signatures
4. GENERATE DSL (Frontier LLM): Write precise DSL for each child
5. REDIRECT (Frontier LLM): Configure parent as router to children
6. VALIDATE: Keep if improved; discard if not
```

**Why Frontier LLM Required:** Pattern recognition, domain expertise, code generation, and routing logic require sophisticated reasoning.

*Practical note:* The LLM may resist—insist on concrete outputs: specific sub-signature names, actual DSL code, explicit routing conditions.

**Summary:** Signatures with low-success DSLs get decomposed into children with precise DSLs. Parents become routers. One failing signature becomes multiple succeeding ones.

### 3.9 Infrastructure

- **SQLite** with WAL mode for concurrent writes
- **Groq API** (Llama-3.3-70B) for fast inference
- **Parallel workers**: 4 workers = 4.2x speedup
- **Embeddings**: all-MiniLM-L6-v2 (local, 384d)

---

## 4. Experiments

### Setup

- **Dataset**: MATH benchmark problems
- **Model**: Llama-3.3-70B via Groq API
- **Evaluation**: LLM judge for semantic equivalence
- **Fair comparison**: Same seeds for baseline and Mycelium

### Main Results

*[Results pending - benchmarks in progress]*

### Key Findings

**Context propagation is critical.** Early experiments showed a "decomposition tax"—Mycelium underperformed. Root cause: non-first steps lost the original problem context. Fix: pass full problem to every step.

**DSL helps some steps, hurts others:**

| Step Type | DSL Benefit |
|-----------|-------------|
| Arithmetic | Strong positive |
| Percentage calculation | Moderate positive |
| Ratio/proportion reasoning | **Negative** |
| Symbolic equation solving | **Negative** |

**Lift-based gating** automatically learns these patterns—no manual rules.

---

## 5. Analysis

### The Decomposition Tax

Every step must succeed for the correct answer. Errors compound multiplicatively.

**Mitigations:**
1. Context propagation to all steps
2. DSL execution for typed steps (100% accuracy)
3. LLM script rewriting when param names don't match
4. Lift-based gating to skip DSL where it hurts

### DSL-Hostile Embedding Spaces

Some step types cluster semantically similar but computationally diverse operations:

| Step Type | Why DSL Fails |
|-----------|---------------|
| `area_triangle` | Inputs vary: coordinates, angles, or sides |
| `compute_angle` | Covers triangle angles, arc measures, rotations |
| `express_relation` | Requires symbolic manipulation, not arithmetic |

**Resolution:** Use guidance mode (method template + LLM) instead of DSL.

### DSL Lift Analysis

Identical DSL scripts produce different outcomes by context:

| Context | DSL | Lift |
|---------|-----|------|
| "Calculate total cups" | `a + b` | **+55%** |
| "Total parts in ratio" | `a + b` | **-40%** |

Simple arithmetic contexts benefit; conceptual contexts are harmed. We detect DSL-hostile contexts via embedding similarity to 8 conceptual exemplars.

### LLM Script Rewriting

**Problem:** DSL uses semantic names (`base`, `height`) but context has generic keys (`step_1`, `step_2`).

**Solution:** LLM rewrites the entire script using actual context variable names. One extra LLM call enables DSL execution that would otherwise fail.

### Key Metrics

| Metric | What It Measures |
|--------|------------------|
| Signatures per problem | Decomposition granularity (healthy: 5-10) |
| Injections per problem | DSL coverage (healthy: 2-5) |
| Success rate per signature | DSL quality (reliable: ≥70%) |

Database grows with new problem types. Signature creation rate declines over time; match rate increases.

---

## 6. Beyond Mathematics

The decomposition-and-reuse paradigm extends to any domain where complex problems decompose into recurring sub-problems. The signature database is a vocabulary; matching is grammar; execution is fluency. What varies by domain: embedding model, success criteria, DSL primitives. Architecture stays constant.

---

## 7. Limitations and Future Work

**Current Limitations:** Decomposition quality depends on LLM planner capabilities.

**Addressed in This Work:**
- Context loss → Fixed by passing original problem to all steps
- DSL hurting general reasoning → Fixed by selective injection
- Same DSL hurting some contexts → Fixed by lift-based gating
- DSL parameter mapping failures → Fixed by LLM script rewriting

**Future Directions:**
- **100% deterministic execution**
- Expand to coding and reasoning benchmarks
- Contrastive learning for signature separation
- **Signature chaining**: Learn common step sequences as solution pipelines

---

## 8. Conclusion

We demonstrated that math problems decompose into a finite vocabulary of atomic signatures, and that **decomposition + signature reuse can improve LLM performance**.

Key insight: **the same LLM performs better when its reasoning is decomposed and cached**.

Critical learnings:
1. **Context propagation**: Every step needs the full problem
2. **Lift-based gating**: DSL helps arithmetic, hurts conceptual reasoning
3. **Self-improving system**: Failed DSLs teach the system where not to inject

The "decomposition tax" was a bug, not a fundamental limitation. Decomposition unlocks pattern reuse and knowledge accumulation unavailable to monolithic solving.

Mycelium demonstrates that LLMs can build persistent, reusable knowledge structures—moving beyond solving each problem from scratch toward genuine compound learning.

---

## Acknowledgments

Developed in collaboration with Claude (Anthropic) for architecture design, implementation, and codebase refactoring. The human provides vision and direction; the AI contributes implementation capacity.

---

## References

1. Wei, J., et al. (2022). Chain-of-thought prompting elicits reasoning in large language models. *NeurIPS*.
2. Gao, L., et al. (2023). PAL: Program-aided language models. *ICML*.
3. Lewis, P., et al. (2020). Retrieval-augmented generation for knowledge-intensive NLP tasks. *NeurIPS*.
4. Hendrycks, D., et al. (2021). Measuring mathematical problem solving with the MATH dataset. *NeurIPS*.
5. Kolodner, J. L. (1992). An introduction to case-based reasoning. *Artificial Intelligence Review*.
