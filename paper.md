# Mycelium: Decomposing Mathematical Problems into Reusable Atomic Signatures

**Bryce Roche**

bryceroche@fungifactor.com

github.com/bryceroche/mycelium

---

## Abstract

Large language models solve mathematical problems through chain-of-thought reasoning, but each problem is solved independently with no persistent memory or reuse of successful solution patterns. We introduce Mycelium, a system that decomposes problems into atomic signatures—reusable solution patterns stored in a vector database. Given a problem, Mycelium constructs a directed acyclic graph (DAG) of steps, matches each step against known signatures via cosine similarity, and executes matched patterns through deterministic domain-specific language (DSL) scripts or LLM-guided reasoning. Novel steps are solved and stored as new signatures. On the MATH benchmark, Mycelium achieves 82% accuracy on Level 3 problems (vs. 80% for direct LLM) and 65% on Level 5 problems (vs. 60% for direct LLM). We demonstrate that decomposition enables systematic knowledge accumulation: 88.6% of steps match existing signatures, achieving a 4.4x reuse ratio. The signature library grows with each problem solved, enabling the system to improve over time.

---

## 1. Introduction

Mathematical reasoning with large language models has advanced significantly through chain-of-thought prompting (Wei et al., 2022) and program-aided reasoning (Gao et al., 2023). However, these approaches treat each problem independently—no knowledge persists between problems, and successful solution strategies are not reused.

We observe that while complete mathematical problems are unique, their constituent steps are highly reusable. The operation "solve for x in 2x + 3 = 7" appears across countless problems in algebra, physics, and economics. This suggests an analogy to prime factorization: just as composite numbers factor uniquely into primes, complex problems may decompose into a finite set of atomic solution patterns.

We introduce Mycelium, a system that builds a "table of primes" for mathematical reasoning—a signature database that grows as problems are solved. The system decomposes problems into DAG-structured steps, matches each step against known signatures via embedding similarity, and executes stored routines (deterministic formulas or procedural guidance). Novel steps are solved via LLM reasoning and stored as new signatures for future reuse.

### Contributions

1. **DAG-based problem decomposition** into reusable atomic signatures with signature-guided hints to improve decomposition quality
2. **Signature database** with centroid-based clustering and periodic consolidation of near-duplicate signatures
3. **Cosine similarity matching** for step-level pattern retrieval with adaptive thresholds
4. **Hybrid execution** routing steps to DSL evaluation, procedural guidance, or LLM reasoning
5. **Self-improving system** with lift-based gating that learns which signatures benefit from DSL execution

---

## 2. Related Work

**Mathematical Reasoning with LLMs.** Chain-of-thought prompting (Wei et al., 2022) enables step-by-step reasoning but solves each problem from scratch. Program-aided language models (Gao et al., 2023) execute code but don't accumulate reusable patterns.

**Retrieval-Augmented Generation.** RAG systems retrieve context before generation (Lewis et al., 2020). Mycelium extends this paradigm to step-level retrieval, matching individual problem steps rather than whole documents.

**Case-Based Reasoning.** Classical AI systems reused solutions from similar cases (Kolodner, 1992). Our signature database is a neural implementation using learned embeddings for similarity computation.

---

## 3. Method

### 3.1 Overview

Given problem P, Mycelium: (1) decomposes P into a DAG of steps, (2) matches each step against the signature database via embedding similarity, (3) executes via stored routines or LLM reasoning, (4) synthesizes results, and (5) updates the database with new patterns.

### 3.2 Problem Decomposition

An LLM decomposes problem P into a DAG where each node represents a step with a task description and dependencies on previous steps. Steps execute in topological order with independent steps parallelized.

**Signature-Guided Decomposition.** To improve alignment between decomposition and existing signatures, we inject the top 15 reliable signatures into the planner prompt as available atomic operations. This guides the LLM to decompose into steps that match proven patterns without constraining novel decompositions.

### 3.3 Signature Database

The database stores atomic solution patterns as tuples (centroid, method_template, DSL_script, statistics). Each signature represents a cluster of semantically similar steps.

**Centroid Updates.** When a new step joins a cluster, the centroid is updated incrementally as a weighted average of the existing centroid and the new embedding.

**Cluster Consolidation.** Near-duplicate signatures may emerge from steps phrased differently but semantically equivalent. We periodically merge signatures with high cosine similarity (≥0.90) between centroids and similar success rates (within 15%), combining their statistics and computing a weighted-average centroid.

### 3.4 Cosine Similarity Matching

Each step is embedded using a sentence transformer (all-MiniLM-L6-v2, 384 dimensions) and matched against signature centroids. A match occurs when similarity exceeds a threshold (default 0.87).

**Adaptive Thresholds.** Fixed thresholds fail when cluster tightness varies. We adjust based on cohesion:

$$\text{threshold} = \text{base} + (\text{cohesion} - 0.5) \times 0.2$$

Tight clusters receive stricter thresholds; loose clusters receive lenient ones.

### 3.5 Execution

Matched signatures execute via one of three methods:
1. **DSL execution**: Deterministic computation (~0ms latency)
2. **Method template injection**: LLM reasoning with procedural guidance (~500ms)
3. **Pure LLM**: Fallback for unmatched or novel steps

**DSL Layers.** We support three DSL execution layers:
- Math: Safe arithmetic operations
- SymPy: Symbolic algebra
- Custom: Registered domain-specific operators

### 3.6 Learning and Reliability

New signatures accumulate uses via LLM execution. After proving reliable (≥5 uses, ≥70% success rate), signatures receive DSL scripts generated by analyzing successful execution patterns.

**Lift-Based Gating.** We track success rates for DSL-injected vs. non-injected executions per signature:

$$\text{lift} = \text{injected\_success\_rate} - \text{baseline\_success\_rate}$$

After a cold-start period (10 uses), signatures with negative lift automatically fall back to LLM reasoning.

### 3.7 Recursive Decomposition

When a DSL has low confidence for a step (below threshold 0.5), we recursively decompose that step into sub-steps until reaching atomic operations or maximum depth. Signatures created during decomposition are marked with their origin depth and flagged as atomic patterns.

### 3.8 LLM Script Rewriting

A fundamental challenge with DSL execution is the semantic gap between script variable names and runtime context keys. When heuristic parameter mapping fails, we invoke the LLM to rewrite the script using actual context variable names:

Original: `base * height / 2` with context `{"step_1": 10, "step_2": 5}`

Rewritten: `step_1 * step_2 / 2`

This enables DSL execution for scripts where heuristic matching would fail.

---

## 4. Experiments

### 4.1 Setup

- **Dataset**: MATH benchmark (Hendrycks et al., 2021), Levels 3-5
- **Model**: Llama-3.3-70B via Groq API for all LLM tasks
- **Embeddings**: all-MiniLM-L6-v2 (384 dimensions)
- **Evaluation**: LLM judge for semantic answer equivalence
- **Reproducibility**: Fixed random seeds; code available under MIT license

### 4.2 Results

**MATH Level 3 (50 problems, seed 123):**

| Method | Accuracy |
|--------|----------|
| Direct Llama 3.3 70B | 80% |
| Mycelium | **82%** |

**MATH Level 5 (100 problems, seed 5678):**

| Method | Accuracy |
|--------|----------|
| Direct Llama 3.3 70B | 60% |
| Mycelium | **65%** |

**Signature Statistics:**
- Match rate: 88.6% of steps match existing signatures
- Reuse ratio: 4.4x (299 step instances matched 68 unique signatures)
- Average steps per problem: 5.1 (range 4-6)

### 4.3 The Decomposition Tax

Decomposition introduces error compounding: each step must succeed for the final answer to be correct. With N sequential steps and per-step accuracy p, problem accuracy is approximately p^N.

| Per-Step Accuracy | 5 Steps | 9 Steps |
|-------------------|---------|---------|
| 95% | 77% | 63% |
| 90% | 59% | 39% |
| 85% | 44% | 23% |

Our mitigations:
1. **Context propagation**: Passing full problem context to every step prevents information loss
2. **DSL execution**: Deterministic execution achieves 100% per-step accuracy for matched signatures
3. **Signature hints**: Guiding decomposition toward proven patterns improves average per-step accuracy
4. **Embedding-based detection**: Avoiding DSL in conceptual reasoning contexts prevents accuracy degradation

### 4.4 DSL-Hostile Step Types

Some step types cluster semantically similar but computationally diverse operations. We identified "DSL-hostile" types where a single formula cannot capture the variance:

| Step Type | Failure Mode |
|-----------|--------------|
| area_triangle | Input format varies (coordinates, angles, side lengths) |
| compute_angle | Covers diverse operations (triangle angles, arc measures, rotations) |
| express_relation | Requires symbolic manipulation, not arithmetic |

For these types, we use guidance mode (method template injection without DSL execution).

---

## 5. Analysis

### 5.1 Signature Convergence

The discovery rate of new signatures decreases as the library matures. In a 100-problem run, we created 109 new signatures while achieving 86.4% match rate. This supports the hypothesis that mathematical steps form a finite vocabulary.

### 5.2 Lift Analysis

Identical DSL scripts produce different outcomes depending on semantic context:

| Context | DSL | Lift |
|---------|-----|------|
| "Calculate total cups" | `a + b` | +55.6% |
| "Total parts in ratio" | `a + b` | -40.0% |

Simple arithmetic contexts benefit from DSL; conceptual/abstract contexts are harmed. We address this via embedding-based detection of DSL-hostile contexts.

### 5.3 Key Metrics

| Metric | Value | Interpretation |
|--------|-------|----------------|
| Signatures per problem | 5.1 | Decomposition granularity |
| Signature matches per problem | 3.5 | Library coverage |
| Injections per problem | 1.1 | DSL utilization |
| Signature success rate | 54% | Current bottleneck |

The 54% signature success rate indicates room for improvement in DSL quality, particularly for geometry and linear algebra domains.

---

## 6. Discussion

**Generalization.** The decomposition-and-reuse paradigm extends beyond mathematics to any domain where complex problems decompose into recurring sub-problems. The signature database represents a learned vocabulary; the matching system provides grammar; the execution layer enables fluency.

**Limitations.** Decomposition quality depends on LLM planner capabilities. The 8.7s average solve time (vs. 2.1s for direct LLM) reflects decomposition overhead.

**Future Work.** Promising directions include: expanding to other domains (coding, reasoning benchmarks), contrastive learning for signature separation, and signature chaining to learn common step sequences.

---

## 7. Conclusion

We demonstrated that mathematical problems decompose into a finite vocabulary of atomic signatures, and that decomposition with signature reuse can match or exceed direct LLM performance. On MATH Level 3, Mycelium achieves 82% vs. 80% for direct LLM (+2 points). On MATH Level 5, Mycelium achieves 65% vs. 60% (+5 points).

Key findings:
1. Context propagation to all steps is critical for decomposition to succeed
2. DSL execution benefits typed operations but harms conceptual reasoning
3. Lift-based gating automatically learns which signatures benefit from DSL
4. The signature library converges toward a finite vocabulary, supporting knowledge accumulation

Mycelium demonstrates that LLMs can build persistent, reusable knowledge structures—moving beyond solving each problem independently toward genuine learning.

---

## Acknowledgments

This project was developed in collaboration with Claude (Anthropic), which contributed to architecture design, implementation, and systematic analysis. The development demonstrated a productive human-AI collaboration model where the human provides vision and direction while the AI contributes implementation capacity.

---

## References

1. Wei, J., et al. (2022). Chain-of-thought prompting elicits reasoning in large language models. *NeurIPS*.
2. Gao, L., et al. (2023). PAL: Program-aided language models. *ICML*.
3. Lewis, P., et al. (2020). Retrieval-augmented generation for knowledge-intensive NLP tasks. *NeurIPS*.
4. Hendrycks, D., et al. (2021). Measuring mathematical problem solving with the MATH dataset. *NeurIPS*.
5. Kolodner, J. L. (1992). An introduction to case-based reasoning. *Artificial Intelligence Review*.

---

## Reproducibility

Code is available at github.com/bryceroche/mycelium under MIT license. The stack requires only:
- Python with standard ML libraries
- Groq API key (free tier available)
- SQLite (included with Python)

Setup time: approximately 5 minutes.
