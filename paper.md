# Guided by Primes

*Decomposing Problems into Reusable Atomic Signatures*

**Author:** Bryce Roche, bryceroche@fungifactor.com, github.com/bryceroche/mycelium

---

*Mycelium is the underground root network of fungi. It decomposes organic matter into simple molecules and distributes nutrients across entire forests. This project does the same for math problems—decomposing them into atomic patterns and sharing solutions through a growing signature database.*

---

## Abstract

Every composite number factors uniquely into primes. We hypothesize that math problems similarly decompose into a finite set of atomic *signatures*—reusable solution patterns. **Mycelium** builds a "table of primes" for mathematical reasoning: a signature database that grows as problems are solved.

An LLM decomposes problems into DAG-structured steps, matches each against known signatures via cosine similarity, and executes DSLs. Novel steps are solved and stored as new signatures. The library grows; future problems get faster.

---

## 1. Introduction

Large language models solve math problems through chain-of-thought reasoning, but each problem is solved from scratch—no persistent memory, no reuse of successful patterns.

Yet **while complete problems are unique, their constituent steps are highly reusable**. The step "solve for x in 2x + 3 = 7" appears across countless problems. Like mycelium networks that decompose organic matter into simple molecules and share nutrients across forests, we decompose complex problems into atomic patterns and distribute solutions through a shared database.

### Contributions

1. **Problem Decomposition**: LLM DAG-based decomposition into reusable atomic signatures
2. **Signature Database**: Vector store of signatures with centroid-based clustering
3. **Cosine Similarity Matching**: Embedding-based retrieval for step-level pattern matching
4. **DSL Execution**: LLM pass parameters to DSLs for execution
5. **Signature Refinement Loop**: Frontier LLM decomposes low-performing signatures into precise child signatures; parents become routers—this is how the system learns

### Open Source & Reproducibility

All code, data, and pre-trained signatures are available at **github.com/bryceroche/mycelium** (MIT license).

**What we're sharing:**
- Complete source code with documented architecture
- **Pre-built signature database** with 2.1k+ math signatures and DSL scripts — skip cold start entirely for Math500 problems
- Benchmark scripts to reproduce our results
- SQLite database file ready to use (no setup required)
- Groq API key required (this is the only external dependency)

**5-minute replication:**

```bash
git clone https://github.com/bryceroche/mycelium
pip install -r requirements.txt
export GROQ_API_KEY=your_key

# Solve a single problem
python -m mycelium "What is 15% of 80?"

# Run MATH benchmark (Level 5, 20 problems)
python scripts/pipeline_runner.py --dataset math --levels 5 --problems 20 --workers 4

# Run in benchmark mode (conservative, max accuracy)
python scripts/pipeline_runner.py --dataset math --levels 5 --problems 20 --mode benchmark

# Run in training mode (explore signatures, collect data)
python scripts/pipeline_runner.py --dataset math --levels 5 --problems 20 --mode training
```

---

## 2. Related Work

**Mathematical Reasoning with LLMs.** Chain-of-thought prompting (Wei et al., 2022) and program-aided reasoning (Gao et al., 2023) solve problems independently without knowledge reuse.

**Retrieval-Augmented Generation.** RAG systems retrieve context before generation (Lewis et al., 2020). We extend this to *step-level* retrieval.

**Case-Based Reasoning.** Classical AI reused solutions from similar cases (Kolodner, 1992). Our signature database is a neural implementation with learned embeddings.

---

## 3. Method

### 3.1 Overview

Given problem LLM decomposes into a DAG of steps, matches each step against the signature database, executes DSLs, and updates the database with new patterns.

### 3.2 Problem Decomposition

An LLM decomposes problems into a DAG where each step has a task description and dependencies. 

### 3.3 Signature Database

The database stores atomic solution patterns as tuples (centroid, method, stats). Centroids update incrementally as new examples join clusters.

**Convergence.** After training on MATH500, the library contains ~2.2k signatures with matching rates converging on 100%. MATH500 L5 problems have ~5 DAG steps with ~3.5 injectable steps per problem. Non-injected steps with negative lift are solved from scratch by LLM and tracked for lift data. 94 distinct step types account for nearly all decomposed steps.

**Distribution.** Signature usage follows a right-skewed power law: mean usage is 8.7 but median is only 4. A small number of signatures handle most of the work—the top 10 signatures account for 16% of all step executions, top 50 account for 31%. Meanwhile, 23% of signatures are used only once (rare edge cases). This mirrors natural language word frequency: a few common patterns (`count_items`, `solve_equation`, `compute_sum`) appear everywhere, while the long tail captures domain-specific variations. New problems increasingly match existing signatures rather than creating new ones.


### 3.4 Cosine Similarity Matching

Each step is embedded and matched against signature centroids using cosine similarity. A match occurs when similarity exceeds a threshold (default 0.92, see `config.py`). The best-matching signature's method template is injected to guide the LLM's solution.

**Why Cosine Similarity?** We evaluated several matching strategies:

Cosine similarity emerged as the best default: it's robust to embedding magnitude variations, computationally cheap (single dot product), and produces interpretable 0-1 scores. 

**Adaptive Thresholds.** Fixed thresholds fail when cluster tightness varies. We adjust based on cohesion:

```
threshold = base + (cohesion - 0.5) × 0.2
```

Tight clusters (cohesion > 0.5) get stricter thresholds; loose clusters get lenient ones. This prevents false matches in well-defined clusters while allowing exploration in sparse regions.

### 3.5 Execution and Learning: Signal Over Accuracy

We are moving towards higher rates of injected DSLs per problem. In **learning mode**, we mandate DSL injection on every signature hit, even when we expect it might fail.

**The Philosophy:** Both DSL successes AND failures provide valuable signal:
- **Success** → positive lift recorded → inject more in future
- **Failure** → negative lift recorded → implement signature refinement loop see 3.9

A failed DSL execution provides good signal that updates the lift statistics that guide future routing decisions. Short-term accuracy loss is acceptable for long-term learning.

**Injection Rate vs Accuracy Trade-off:**

In training mode, we deliberately inject DSLs that the lift data suggests will fail. This "exploration" fills gaps in our knowledge—maybe the negative lift was from a bug we've since fixed, or from a different problem context. Only by trying again do we update our beliefs.


This creates a system that improves over time: aggressive exploration in early runs builds data about which DSLs work, and lift-based gating automatically optimizes routing in later runs.

### 3.6 Recursive Decomposition

When a DSL has low confidence for a step, that's a signal the step is too complex. Rather than falling back to pure LLM reasoning, we **decompose further** until reaching truly atomic operations.  See 3.9 for Refinement loop

**The Self-Improvement Loop: Secondard processing**

1. Signature has low DSL confidence
2. System decomposes signature into sub-signatures
3. Sub-signatures have new DSL created
4. Parent signature DSL now routes to children signatures

Each problem that triggers deep decomposition *teaches* the system new atomic patterns. Over time, decomposition becomes rarer as the atomic vocabulary grows.

**Configuration:**
- `MAX_DECOMPOSITION_DEPTH = 3` — prevent infinite recursion
- `DECOMPOSITION_CONFIDENCE_THRESHOLD = 0.5` When the signature DSL has confidence below 50%, we decompose it.

**Why This Works:**
1. **Self-adapting**: The system finds the right granularity automatically
2. **No hardcoding**: DSL-hostile signatures get decomposed; DSL-friendly execute directly
3. **Builds vocabulary**: Atomic signatures accumulate, improving future coverage
4. **Depth tracking**: Signatures know if they're atomic (origin_depth > 0)

### 3.7 Cold Start

With an empty database, no signatures exist to match against so every step is novel and solved from scratch by the LLM. The system bootstraps by storing successful solutions as new signatures. Initially, we need to boost new signature injection to sample their success rates. As signatures accumulate and prove reliable, injection rates climb. We observe a characteristic warm-up period of ~50-100 problems before meaningful reuse emerges. 

### 3.8 Parameter Matching
**Parameter Matching:** The LLM generates parameter aliases during DSL creation (e.g., `percentage` → `pct`, `percent`). At runtime, alias matching maps context values to DSL parameters without additional LLM calls.

**Example Evolution:**

*Example DSL:*
```json
{"type": "math", "script": "(percentage / 100) * base",
 "params": ["percentage", "base"],
 "aliases": {"percentage": ["pct", "percent"], "base": ["value", "total"]}}
```

This is the "smart work once, execute forever" principle: invest LLM reasoning to generate the DSL once, then execute deterministically for all future matches.

**Bulk DSL Generation with Claude Opus 4.5:** Rather than waiting for signatures to prove reliable organically, we batch-processed all ~1,300 signatures in the database through Claude Opus 4.5 to generate custom DSL scripts. For each signature, Claude analyzed the step type, example problems, and success patterns to write precise executable code. This one-time investment (~$15 in API costs) equipped 84% of typed signatures with deterministic DSL—turning months of organic learning into a single afternoon of batch processing. The remaining 16% are guidance-only signatures where LLM flexibility outperforms rigid formulas.

### 3.9 DSL Parameter Passing: Structured Output

A DSL is only useful if we can pass the right parameters. Early versions used regex to extract numbers from LLM responses—brittle and error-prone.

**The Problem with Regex Parsing:**

```
LLM output: "Let me calculate... the area is 42 square units."

Regex attempts:
  r"RESULT:\s*(.+)" → no match (no RESULT: prefix)
  r"area is (\d+)" → matches, but fragile
  r"(\d+)\s*$" → matches "42", but also matches page numbers, step counts...
```

Every new phrasing required a new regex pattern. Edge cases multiplied. The parsing layer became a liability.

**The Solution: JSON Response Format**

Modern LLM APIs support `response_format: {"type": "json_object"}`. Instead of parsing free-form text, we instruct the LLM to output structured JSON:

```python
# Old approach (fragile)
response = await client.generate(messages)
result = extract_result(response)  # regex parsing

# New approach (clean)
response = await client.generate_json(messages)
result = response["result"]  # direct extraction
```

The LLM outputs:
```json
{"reasoning": "base=7, height=12, area=0.5*7*12", "result": 42}
```

**Implementation:**

```python
# client.py - Added JSON generation method
async def generate_json(self, messages, temperature=0.3):
    content = await self.generate(
        messages,
        response_format={"type": "json_object"},
    )
    return json.loads(content)

# solver.py - Step execution with JSON output
json_response = await self.solver_client.generate_json(messages)
result = str(json_response.get("result", ""))
```

**Prompt Template for JSON Output:**

```
Solve this step. Output your response as JSON:
{"reasoning": "your step-by-step reasoning", "result": <numeric/symbolic result>}

IMPORTANT:
- "result" should be the direct value (number, expression, or equation)
- Use a number for numeric results: {"reasoning": "...", "result": 42}
- Use a string for expressions: {"reasoning": "...", "result": "x^2 - 4"}
```

**Why This Eliminates Regex:**

| Aspect | Regex Parsing | JSON Output |
|--------|--------------|-------------|
| Extraction | Pattern matching | `response["result"]` |
| Numeric values | Parse from text | Already typed as number |
| Edge cases | Endless patterns | None - structure enforced |
| Failure mode | Silent wrong parse | JSON decode error (catchable) |
| Maintainability | Growing regex library | Single format |

**Results:**

| Metric | Before (Text) | After (JSON) |
|--------|---------------|--------------|
| GSM8K Accuracy | 90% | 95% |
| MATH L5 Injection | 60% | 69% |
| Parsing failures | ~5% | <1% |

The accuracy improvement comes from eliminating silent parsing errors—cases where regex extracted the wrong number from a response.

**Graceful Fallback:**

JSON mode isn't universally supported (some API configurations return 400 errors). We implement fallback:

```python
try:
    json_response = await client.generate_json(messages)
    result = str(json_response.get("result", ""))
except Exception:
    # Fallback to text mode with regex
    response = await client.generate(messages)
    result = extract_result(response)
```

This ensures robustness while preferring the cleaner JSON path.

**The Insight:** The LLM already understands what each value represents—it computed the answer. Asking it to output structured data is trivial. Asking regex to parse natural language is impossible. Let the LLM do what LLMs do best: understanding context and extracting meaning.

**Parameter Mapping for DSL Execution:**

When a step matches a signature with a DSL like `(base * height) / 2`, we need to map prior step results to DSL parameters. The flow:

```
Step 1 executes → {"result": 7}   → stored as step_1
Step 2 executes → {"result": 12}  → stored as step_2
Step 3 matches DSL: (base * height) / 2
```

The DSL executor builds a context dict from prior results:
```python
context = {"step_1": 7, "step_2": 12}
```

Then the LLM rewrites the DSL script to use positional references:
```python
# Original DSL script (semantic names)
"(base * height) / 2"

# LLM-rewritten script (positional refs)
"(step_1 * step_2) / 2"

# Executes with context → 42
```

This "LLM script rewriting" approach works because the LLM sees both the step descriptions and the context keys—it knows `step_1` is the base and `step_2` is the height from the problem context. No brittle parameter name matching required.

### 3.10 Infrastructure

**Storage:** SQLite database stores signatures, embeddings (as packed binary), examples, and statistics. Single-file deployment with no external database dependencies.

**SQLite Tuning for Parallel Execution:** The default SQLite journal mode blocks concurrent writes—problematic when running multiple Groq workers in parallel. We enable Write-Ahead Logging (WAL) mode:

```python
conn.execute("PRAGMA journal_mode = WAL")
conn.execute("PRAGMA busy_timeout = 30000")  # 30s lock timeout
```

WAL allows concurrent readers and writers, eliminating "database is locked" errors. Combined with a 30-second busy timeout, this enables parallel benchmark execution without database contention.

**LLM Inference:** Groq API with Llama-3.3-70B for fast inference (~500ms per call). Used for problem decomposition, step execution, and DSL generation.

**Parallel Groq Workers:** With WAL mode enabled, we can run multiple problems concurrently:

| Workers | 100 Problems | Speedup |
|---------|--------------|---------|
| 1 | 941s | 1.0x |
| 4 | 226s | **4.2x** |

The 4.2x speedup comes from overlapping API latency across workers. Each worker maintains its own database connection; WAL ensures writes don't block each other. Benchmark time drops from ~16 minutes to ~4 minutes.

**Embeddings:** all-MiniLM-L6-v2 (384-dimensional) via sentence-transformers. Local inference, no API calls.

**LRU Caching (future work):** Two-layer caching eliminates redundant computation:
- *Embedding cache* (1000 entries): Identical step text returns cached vector instantly
- *Classification cache* (1024 entries): Step type lookups skip pattern matching on cache hit

Cache hit rates exceed 60% in typical runs—steps like "solve for x" appear repeatedly across problems. Combined with DSL execution, a fully-cached step resolves in <1ms (vs ~500ms for LLM).

**Development:**
- **Claude Code** (Anthropic): AI pair programming for architecture design and implementation
- **Beads**: Git-native issue tracking for task management (`.beads/` directory)
- **tmux**: Parallel development sessions—multiple Claude instances working on different components
- **Git**: Version control with hooks for automated beads sync

This lightweight stack enables rapid iteration: SQLite for portability, Groq for speed, and Claude + tmux for parallelized AI-assisted development.

### 3.11 Signature Refinement Loop

Low-performing signatures reveal opportunities for improvement. We propose an automated refinement loop that **requires a frontier LLM** (e.g., Claude Opus) to perform the sophisticated analysis and code generation:

**The Loop:**

```
1. IDENTIFY: Query signatures with success_rate < threshold
   → "area_triangle" at 15% success, 200 uses

2. ANALYZE (Frontier LLM): Examine failure cases and identify patterns
   → "Failures occur when inputs are coordinates vs. side lengths vs. angles"

3. DECOMPOSE (Frontier LLM): Design finer-grained sub-signatures
   → area_triangle_coordinates (Shoelace formula)
   → area_triangle_sides (Heron's formula)
   → area_triangle_angle (½ab·sin(C))

4. GENERATE DSL (Frontier LLM): Write precise DSL for each child
   → Each sub-signature gets a single-purpose, tested DSL script

5. REDIRECT (Frontier LLM): Configure parent as router to children
   → Parent signature stores pointers to sub-signatures
   → LLM writes routing logic based on input type detection

6. VALIDATE: Test on held-out examples
   → Keep if success_rate improves; discard if not
```

**Why a Frontier LLM is Required:**

Steps 2-5 require sophisticated reasoning that only frontier models can reliably perform:
- **Pattern recognition** across failure cases to identify root causes
- **Domain expertise** to know Heron's formula vs. Shoelace vs. trigonometric approaches
- **Code generation** to write correct, tested DSL scripts
- **Routing logic** to classify input types and direct to appropriate children

A weaker model would hallucinate formulas or mis-classify input patterns. The refinement loop is where frontier LLM capability pays dividends—each refinement improves thousands of future executions.

*Practical note:* The LLM may initially resist this task ("I can help you think through approaches...") or produce overly cautious responses. Insist on concrete outputs: specific sub-signature names, actual DSL code, explicit routing conditions. The model is capable; it just needs clear direction that you want executable artifacts, not suggestions.

**The Compound Effect:**

Each refinement cycle:
- Converts low-performing signatures into high-performing sub-signatures
- Increases overall DSL injection rate
- Moves the system toward fully deterministic execution

The parent becomes a router; the atomic children get precise DSLs. This is the "learning" in self-improving: not just accumulating signatures, but actively refining them based on observed performance.

**Summary:** Signatures with low-success DSLs get decomposed into child signatures with new, precise DSLs. The parent signatures become routers that direct incoming traffic to the appropriate child. The result: what was one failing signature becomes multiple succeeding ones.

---

## 4. Experiments

### Setup

- **Dataset**: MATH Level 3 problems (algebra, precalculus, geometry, number theory)
- **Model**: Llama-3.3-70B via Groq API
- **Embeddings**: all-MiniLM-L6-v2 (384d)
- **Evaluation**: LLM judge for semantic answer equivalence
- **Reproducibility**: Fixed random seeds for problem selection
- **Fair comparison**: Baseline (direct LLM) and Mycelium run on identical problem sets using the same seed

### Reproducibility

Results are trivial to verify. The entire stack is accessible:

- **Code**: Open source under MIT license at github.com/bryceroche/mycelium
- **LLM Inference**: Groq API (free tier available)
- **Database**: SQLite (single file, no setup)
- **Embeddings**: Local sentence-transformers (no API keys)

Total setup time: ~5 minutes. Run `pip install -r requirements.txt`, add a Groq API key, and execute the benchmark. No cloud infrastructure, no GPU cluster, no waiting for API quotas.

### Main Results

*[Results pending - benchmarks in progress]*

### How We Got Here

Early experiments showed a "decomposition tax"—Mycelium initially *underperformed* direct LLM prompting. Investigation revealed two bugs:

1. **Context loss:** Non-first steps only received dependency results, losing the original problem's constraints and meaning. Fix: pass full problem context to every step.

2. **DSL parameter mapping:** DSL scripts couldn't find inputs when parameter names (`base`, `height`) didn't match context keys (`step_1`, `task_num_0`). Fix: LLM-based script rewriting.

### The Self-Improving Loop

The key insight: **failures are learning signals**. When a DSL execution fails:
1. Negative lift is recorded for that signature
2. Future runs skip DSL for signatures with negative lift
3. System automatically routes to LLM reasoning where DSL hurts

Over time, this feedback loop:
- Identifies which DSLs work (positive lift → keep injecting)
- Identifies which DSLs hurt (negative lift → skip to LLM)
- Builds a library of new signatures for future reuse

### DSL Execution: When It Helps vs. Hurts

Not all steps benefit from DSL. Analysis revealed:

| Step Type | DSL Benefit |
|-----------|-------------|
| Arithmetic (`a + b * c`) | Strong positive |
| Unit conversion | Strong positive |
| Percentage calculation | Moderate positive |
| Ratio/proportion reasoning | **Negative** (skip DSL) |
| Symbolic equation solving | **Negative** (skip DSL) |

Lift-based gating automatically learns these patterns. No manual rules needed.

---

## 5. Analysis

### The Decomposition Tax: Error Compounding in DAG Execution

Decomposition introduces a fundamental challenge: **every step must succeed for the final answer to be correct**. In a DAG with N sequential steps, errors compound multiplicatively. This explains why naive decomposition can hurt performance—the "decomposition tax."

**Our Mitigations:**

1. **Context Propagation:** Pass the full problem to every step, preventing information loss that caused cascading failures.

2. **DSL Execution:** Deterministic execution eliminates LLM error on typed steps.

3. **LLM Script Rewriting:** When DSL parameter names don't match context keys, LLM rewrites the script using actual variable names.

4. **Lift-Based Gating:** Automatically skip DSL for signatures where it hurts accuracy.

### Step-Level Reusability

Decomposition unlocks reuse that monolithic approaches cannot access. The signature library converges toward a stable vocabulary where new problems reuse existing patterns.

### Signature Convergence

The signature library converges toward a finite vocabulary—supporting the hypothesis that mathematical reasoning decomposes into a bounded set of atomic operations.

### Signature Deduplication: A Lesson in Database Hygiene

Analysis of our signature database revealed significant redundancy: **43% of stored signatures were duplicates** with identical centroids. The raw count showed 1,189 signatures, but only 675 were unique.

| Step Type | Duplicates | Root Cause |
|-----------|------------|------------|
| `solve_equation` | 357 | High-volume type, race conditions |
| `setup_equation` | 19 | Similar phrasing variations |
| `count_items` | 18 | Parallel worker collisions |

**Root cause:** When multiple workers process problems simultaneously, they may each create a new signature for the same step pattern before the first write commits. The `find_or_create` logic races against itself.

**Impact:** Low apparent reuse rates. With 357 duplicates of `solve_equation`, each copy averaged only 1.1 uses instead of the combined ~400 uses going to one signature. This fragmentation:
- Obscures true reuse metrics
- Dilutes success rate statistics
- Wastes storage and lookup time

**Fix:** Periodic consolidation merges signatures with identical (step_type, centroid) pairs, combining their usage statistics. After deduplication, the 675 unique signatures show healthier reuse patterns with 6.7 uses per signature on average.

### The DSL Selective Injection Principle

A key insight: **DSL helps typed signatures but hurts general reasoning steps**.

When we added DSL to *every* signature (including `general_step`), accuracy dropped:

```
DSL on typed signatures only:  63.3%
DSL on all signatures:         36.7%
```

**Root cause:** General steps (`general_step`) require flexible LLM thinking—problem interpretation, setup, multi-step reasoning. Adding DSL interferes with this flexibility.

**The solution:** Only typed signatures (compute_sum, solve_quadratic, etc.) receive DSL. General steps use pure LLM reasoning. Our step type classifier makes this distinction automatically via linguistic pattern matching, categorizing steps into 40+ specific types.

### DSL Lift Analysis: Same Script, Different Outcomes

A deeper analysis revealed a surprising pattern: **identical DSL scripts produce wildly different outcomes depending on semantic context**.

| Step Type | DSL Script | Context | Lift |
|-----------|-----------|---------|------|
| compute_sum | `a + b` | "Calculate total cups" | **+55.6%** |
| compute_sum | `a + b` | "Total parts in ratio" | **-40.0%** |
| compute_difference | `a - b` | "Common difference in sequence" | **-27.4%** |
| simplify_expression | `simplify(expr)` | "Simplify the equation" | **+54.5%** |

**The Pattern:** Simple arithmetic contexts benefit from DSL (+55% lift), while conceptual/abstract contexts are harmed (-40% lift). The step type `compute_sum` is too coarse—it covers both "add two numbers" and "analyze ratio components."

**Lift-Based Gating.** We track success rates for injected vs non-injected executions per signature:

```
lift = injected_success_rate - baseline_success_rate
```

After a cold-start period (10 uses), signatures with negative lift automatically fall back to LLM reasoning. This self-correcting mechanism ensures DSL injection only occurs when it demonstrably helps.

**Implications for Step Type Design:** Finer-grained step types (e.g., `simple_addition` vs `ratio_analysis`) would enable more precise DSL matching. Our current 40+ step types represent a first approximation; the lift data suggests further subdivision would improve accuracy.

### Current System Health

Key metrics to monitor:

| Metric | What It Measures |
|--------|------------------|
| Steps/problem | Decomposition granularity |
| Signature match rate | Library coverage |
| DSL injections/problem | Deterministic execution rate |
| Problem accuracy | End-to-end performance |

The lift-based gating automatically routes each step to its optimal execution path.

### DSL-Hostile Embedding Spaces

Some step types cluster steps that *sound* similar but require fundamentally different computation. These "DSL-hostile" embedding spaces look uniform to the classifier but contain high variance in actual solution methods.

| Step Type | Failed DSL | Why It Fails |
|-----------|------------|--------------|
| `area_triangle` | `0.5 * base * height` | Problems provide coordinates, angles, or side lengths—rarely base and height directly. Requires Heron's formula, coordinate geometry, or trigonometry depending on input format. |
| `compute_magnitude` | `sqrt(sum(c**2))` | Generator expressions unsupported by AST evaluator. Even with fix, vectors come as strings like "[3, 4]" requiring parsing. |
| `vector_operation` | `Matrix(v1).dot(v2)` | "Vector operation" covers dot product, cross product, addition, scaling, projection—no single formula fits. Input format varies wildly. |
| `compute_angle` | `180 - a - b` | Only works for "find third angle in triangle." Step type matches angle bisectors, arc measures, rotation angles, phase shifts—completely different operations. |
| `matrix_operation` | `Matrix(m).det()` | Covers determinant, inverse, multiplication, eigenvalues, row reduction. The step type is a category, not an operation. |
| `express_relation` | `a / b` | Semantic step: "express X in terms of Y" requires symbolic manipulation, not arithmetic. The relation *is* the answer, not a computation. |
| `apply_amgm` | `sqrt(a * b)` | AM-GM is an *inequality* technique for finding extrema. Computing GM is a tiny part; the reasoning about when equality holds is what matters. |

**The Core Problem:** These step types are *semantic categories* (geometry, linear algebra, optimization) rather than *computational operations* (add, multiply, solve quadratic). DSL excels at the latter but fails at the former.

**Resolution:** Switch to `guidance` mode—LLM reasoning with method template injection, no DSL execution. The method template provides strategy ("use Heron's formula when given three sides") while the LLM handles the actual computation.

**DSL-Friendly vs DSL-Hostile:**

| DSL-Friendly | DSL-Hostile |
|--------------|-------------|
| `compute_percentage` | `area_triangle` |
| `solve_quadratic` | `compute_angle` |
| `compute_sum` | `express_relation` |
| `evaluate_expression` | `apply_amgm` |

The distinction: DSL-friendly types have **one canonical formula** with **predictable input format**. DSL-hostile types are **semantic umbrellas** covering diverse operations.

### DSL Input Mapping: From 0% to 64% Confidence

Early versions had a subtle bug where DSL execution failed silently on most signatures. Analysis revealed:

**The Problem:**
- DSL params expect names like `base`, `exponent`
- Step context only contains prior step results: `{"step_1": "1024"}`
- No `io_schema` to map between them → `numeric_inputs = {}`
- Result: 0% confidence, DSL never executes

**The Fix (three parts):**

1. **Extract from step task**: Parse numbers from the task itself. "Calculate 2^10" → `{task_num_0: 2, task_num_1: 10}`

2. **Positional fallback**: When param names don't match context keys, map by position. `["base", "exponent"]` + `{task_num_0: 2, task_num_1: 10}` → `{base: 2, exponent: 10}`

3. **Adjusted threshold**: Positional matching incurs a 20% penalty per param (0.8^n). Two params = 0.64 confidence. Lowered threshold from 0.7 to 0.5.

**Result:** DSL injections per problem increased from ~1 to ~5. The "Calculate 2^10" step now executes via DSL in <1ms instead of requiring an LLM call.

### LLM Script Rewriter: Bridging the Semantic Gap

The fundamental challenge with DSL execution is the **semantic gap** between script variable names and runtime context keys. A DSL script written as `base * height / 2` contains meaningful semantic names, but at runtime the context only contains generic identifiers like `{"step_1": 10, "step_2": 5}`. Heuristic matching (substring, alias lookup, positional) fails when there's no linguistic overlap.

**The Problem in Practice:**

```
DSL Script: (percentage / 100) * base
Context: {"step_1": 15, "step_2": 240}
Heuristic confidence: 0.0 (no matches)
Result: DSL cannot execute
```

The LLM that generated this DSL knows `percentage` means "the percentage value" and `base` means "the value to calculate percentage of." But at runtime, we've lost that semantic information—we only have `step_1` and `step_2`.

**The Solution: LLM Script Rewriting**

Instead of trying to map parameter names to context keys (which requires understanding semantic equivalence), we ask the LLM to **rewrite the entire script** using the actual context variable names:

```
Prompt:
  Original script: base * height / 2
  Available context: {"step_1": 10, "step_2": 5}
  Task: Rewrite using ONLY context variable names.

LLM Response: step_1 * step_2 / 2
```

The rewritten script can be executed directly with the context dictionary—no mapping required.

**Implementation:**

```python
async def llm_rewrite_script(dsl_spec, context, client):
    prompt = f"""Rewrite this DSL script to use ONLY the available context variable names.

    Original script: {dsl_spec.script}
    Available context variables and their values:
    {json.dumps(context, indent=2)}

    Return ONLY the rewritten script, nothing else:"""

    response = await client.generate([{"role": "user", "content": prompt}],
                                      max_tokens=200, temperature=0.0)
    return response.strip()
```

**The Execution Flow:**

`execute_dsl_with_llm_matching(dsl_json, inputs, client)`:
1. Parse DSL spec
2. Compute heuristic confidence
3. If confidence < llm_threshold: call `llm_rewrite_script()`, create new DSLSpec with rewritten script, execute with full context, return result with confidence=1.0
4. Else: execute with heuristic param mapping

**Why Rewriting Beats Mapping:**

| Approach | Pros | Cons |
|----------|------|------|
| **Param Mapping** | Preserves original script | Fails when names don't overlap |
| **Script Rewriting** | Works regardless of naming | Requires LLM call; may mismap |

Script rewriting is more robust because it:
1. Handles complex expressions with multiple variable references
2. Works even when param names have zero semantic overlap with context keys
3. Produces executable code rather than a mapping that might leave params unresolved
4. Leverages LLM's semantic understanding of what each variable represents

**Real Examples from Benchmark Runs:**

| Original Script | Context | Rewritten Script | Result |
|-----------------|---------|------------------|--------|
| `speed * 3` | `{"step_1": 45}` | `step_1 * 3` | 135.0 ✓ |
| `total_hours - regular_hours` | `{"step_1": 400, "step_2": 360}` | `step_1 - step_2` | 40.0 ✓ |
| `sqrt((x2-x1)**2 + (y2-y1)**2)` | `{"step_1": 3, "step_2": 4}` | `sqrt((step_2-step_1)**2 + ...)` | 5.0 ✓ |
| `base * height / 2` | `{"step_1": 10, "step_2": 5}` | `step_1 * step_2 / 2` | 25.0 ✓ |

**Failure Modes:**

The rewriter can fail when:
1. **Ambiguous context**: `{"step_1": 10, "step_2": 10}` — both values identical, LLM guesses wrong mapping
2. **Missing values**: Context doesn't contain values the script needs
3. **Type mismatches**: Context has strings where script expects numbers

In these cases, the system falls back to full LLM reasoning (no DSL execution).

**Performance Characteristics:**

| Metric | Heuristic Only | With LLM Rewriter |
|--------|----------------|-------------------|
| DSL injection rate | ~15% of matches | ~60% of matches |
| Latency per injection | ~1ms | ~500ms |
| Overall accuracy | Lower (more fallbacks) | Higher (more DSL executions) |

The trade-off: one extra LLM call (~500ms) to enable DSL execution that would otherwise fail. This is still faster than full LLM reasoning (~1-2s) and produces deterministic results.

**Two Operating Modes:**

The system supports two modes, configured in `config.py`:

```python
# config.py
ACTIVE_MODE = Mode.TRAINING   # or Mode.BENCHMARK
```

| Setting | Training | Benchmark |
|---------|----------|-----------|
| `MIN_MATCH_THRESHOLD` | 0.92 | 0.95 |
| `DSL_MIN_CONFIDENCE` | 0.0 | 0.3 |
| `EXPLORATION_RATE` | 1.0 | 0.5 |
| `RECURSIVE_DECOMPOSITION_ENABLED` | False | True |
| `DSL_PROBATION_ENABLED` | False | True |

**Training mode:**
- Lower match threshold (0.92) to explore more signatures
- Try DSL on every signature hit (`DSL_MIN_CONFIDENCE = 0.0`)
- Collect success/failure data for lift-based learning
- Decomposition disabled to maximize DSL attempts

**Benchmark mode:**
- Higher match threshold (0.95) for conservative matching
- Only execute high-confidence DSLs (`DSL_MIN_CONFIDENCE = 0.3`)
- Enable recursive decomposition for complex steps
- Use proven patterns via probation gating

All thresholds are centralized in `config.py` for easy tuning.

**Injection Rate Ceiling:** The ~50% injection rate is the practical maximum. The remaining steps have truly empty context (first steps with no numbers in task text). DSL cannot execute without inputs—these steps require LLM reasoning.

**The Learning Loop:** Problem arrives → step matches signature with DSL → heuristic confidence low → LLM rewrites script → rewritten script executes → success/failure recorded. Over time, signatures with consistent rewrite failures get negative lift and fall back to pure LLM. System learns which DSLs benefit from rewriting vs which should skip DSL entirely.

This creates a self-improving system: aggressive exploration in early runs builds data about which DSLs work with rewriting, and lift-based gating automatically disables problematic DSLs in later runs.

### LLM Script Rewriting: A Cautionary Tale

**The Benchmark Regression:**

In a benchmark run with LLM script rewriting enabled, we observed a severe accuracy drop:

| Configuration | L5 Accuracy | Notes |
|---------------|-------------|-------|
| Previous (guidance mode) | 65% | Baseline |
| LLM script rewriting | **45%** | -20 points! |

**Root Cause:** The LLM was producing "successful" DSL executions with **wrong parameter mappings**. The scripts executed without error, but the answers were garbage:

```
DSL: area_ENG / area_ABC
Context: {"step_1": 25, "step_2": 16, "step_3": 9}

LLM rewrites to: step_1 / step_1  → 1.0 ❌
Should have been: step_2 / step_1 → 0.64 ✓
```

The LLM sees generic names like `step_1`, `step_2` and has no semantic context to determine which step computed which value. It guesses—and guesses wrong.

**The Core Problem:**

```
DSL params:    area_ABC, area_DEF  (semantic meaning)
Context keys:  step_1, step_2      (generic identifiers)
```

Without knowing that `step_1` computed "area of triangle ABC" and `step_2` computed "area of triangle DEF", the LLM cannot reliably map parameters.

### Semantic Parameter Mapping

**The Solution:** Instead of LLM guessing, use **semantic matching** based on step task descriptions.

When each step executes, we track:
- **Value:** The numeric result (e.g., `25.0`)
- **Meaning:** What it represents (e.g., "area of triangle ABC")
- **Type:** Category (e.g., "area", "length", "count")

This enables deterministic parameter mapping:

```
DSL param: area_ABC
Step 1 meaning: "area of triangle ABC"
Match by string similarity → area_ABC = step_1.value ✓
```

**Implementation:**

```python
@dataclass
class StepResult:
    step_id: str
    result: str
    # Semantic context
    semantic_meaning: str  # "area of triangle ABC"
    semantic_type: str     # "area"
    numeric_value: float   # 25.0

def semantic_rewrite_script(dsl_spec, context, step_descriptions):
    """Map DSL params to context by semantic similarity."""
    param_mapping = {}
    for param in dsl_spec.params:
        for ctx_key, desc in step_descriptions.items():
            if param_matches_description(param, desc):
                param_mapping[param] = ctx_key
                break
    return rewrite_script(dsl_spec.script, param_mapping)
```

**Matching Rules:**

1. **Exact match:** `area_ABC` in "Calculate the area of triangle ABC" → 0.95 confidence
2. **Token overlap:** `base` shares tokens with "Find the base of the rectangle" → 0.7 confidence
3. **Suffix match:** `_ABC` matches description containing "ABC" → 0.8 confidence

**Results:**

| Approach | Accuracy | Deterministic | Latency |
|----------|----------|---------------|---------|
| LLM rewriting | 45% | No | +500ms |
| **Semantic matching** | TBD | **Yes** | +0ms |

Semantic matching is:
- **Deterministic:** Same inputs always produce same mapping
- **Fast:** No LLM call needed
- **Explainable:** Can log exactly why each param was mapped
- **Testable:** Unit test each mapping rule

### Embedding-Based Conceptual Detection

The lift analysis revealed that keywords like "ratio" and "proportion" predict poor DSL performance. But keyword matching is brittle—it catches "calculate the ratio" but misses semantically equivalent phrasings like "find the proportion" or "determine the relative amounts."

**The Solution:** Replace keyword matching with embedding similarity. We define 8 *conceptual exemplars*—representative phrases where DSL historically hurts:

```python
CONCEPTUAL_EXEMPLARS = [
    "Calculate the total number of parts in a ratio of 3:5",
    "Find the proportion of red to blue marbles",
    "Determine the ratio between the two quantities",
    "Express the relationship as a ratio",
    "Find the rate of change between the values",
    "Calculate how much faster one is than the other",
    "Express the answer in terms of the original variable",
    "Interpret the result in the context of the problem",
]
```

At runtime, we compute cosine similarity between the step embedding and each exemplar. If max similarity exceeds 0.7, DSL is skipped:

```python
is_conceptual, max_sim = is_conceptual_context_embedding(step_embedding, embedder)
if is_conceptual:  # max_sim >= 0.7
    skip_dsl = True  # Use LLM reasoning instead
```

**Results:** Testing shows clean semantic separation:

| Step | Similarity | Action |
|------|------------|--------|
| "Calculate the total cups of flour" | 0.42 | **USE DSL** |
| "Find the sum of these two numbers" | 0.34 | **USE DSL** |
| "Calculate total parts in ratio 3:5" | 0.95 | **SKIP DSL** |
| "Determine the proportion of red to blue" | 0.79 | **SKIP DSL** |
| "Express the answer in terms of x" | 0.73 | **SKIP DSL** |
| "Apply the quadratic formula" | 0.39 | **USE DSL** |

The embedding approach captures semantic similarity that keyword matching cannot. "Parts in a ratio" matches our exemplars even though it doesn't contain the exact word "proportion."

**Why This Works:** The all-MiniLM-L6-v2 embedding model encodes semantic meaning, not just lexical overlap. Steps requiring conceptual reasoning cluster together in embedding space, making similarity-based detection robust to paraphrasing.

This is another example of *learning once, applying forever*: we identified problematic contexts through lift analysis, encoded them as exemplar embeddings, and now automatically detect semantically similar contexts without maintaining a fragile keyword list.

### Rich Typed Step Outputs: Unlocking Symbolic DSL

**The Problem:** Complex DSL scripts like `solve(equation, x)` need **structured symbolic inputs**, not just numbers from prior steps.

Consider this problem flow:

```
Problem: "Find k where x² + kx + 4 = 0 has exactly one solution"

Step 1: "Write discriminant condition" → "k² - 16 = 0"
Step 2: "Solve for k" → DSL: solve(discriminant, k)
                              ↑
                              We have "k² - 16 = 0" as TEXT
                              DSL needs sympy.Eq(k**2 - 16, 0)
```

Traditional step outputs are strings: `"42"` or `"k² - 16 = 0"`. DSL scripts operating on symbolic expressions need the *structure*, not just the string representation.

**The Solution: StepOutput with Type Information**

```python
@dataclass
class StepOutput:
    raw: str                    # Original: "k² - 16 = 0"
    value_type: str             # "number" | "equation" | "expression" | "list" | "text"
    numeric: Optional[float]    # 42.0 if it's a number
    sympy_expr: Optional[str]   # "Eq(k**2 - 16, 0)" for equations
    variables: list[str]        # ["k"] - symbols in expression
```

**Type Detection:**

```
"42"              → number    (numeric=42.0)
"x^2 - 16 = 0"    → equation  (sympy_expr="Eq(x**2 - 16, 0)")
"3x + 2y - 5"     → expression (sympy_expr="3*x + 2*y - 5")
"[1, 2, 3]"       → list
"The answer is"   → text
```

**DSL Execution with Types:**

```python
# Old (broken):
context = {"step_1": "k² - 16 = 0"}  # Just a string
solve(context["step_1"], k)  # Fails - can't solve a string

# New (works):
context = {"step_1": StepOutput(
    raw="k² - 16 = 0",
    value_type="equation",
    sympy_expr="Eq(k**2 - 16, 0)"
)}
solve(parse_expr(context["step_1"].sympy_expr), k)  # Works!
```

**Why This Matters:**

The 5-7 wrong answers on MATH L5 problems with 0-1 injections share a pattern: they require symbolic manipulation (solving equations, simplifying expressions) but the DSL received strings instead of parseable symbolic forms.

| Problem Type | Old Context | New Context | DSL Works? |
|-------------|-------------|-------------|------------|
| Arithmetic | `"42"` | `numeric=42.0` | ✓ Both work |
| Equation solving | `"x² - 4 = 0"` | `sympy_expr="Eq(x**2-4,0)"` | ✓ New works |
| Expression simplify | `"3x + 2x"` | `sympy_expr="3*x + 2*x"` | ✓ New works |
| System of equations | Multiple strings | List of Eq() | ✓ New works |

**Implementation Path:**

1. **detect_output_type()** - Parse step result strings into StepOutput
2. **DSL executor** - Use `output.for_dsl(prefer_type="symbolic")` for sympy operations
3. **Context passing** - Replace string context with StepOutput objects

This converts the ~35% of DSL failures caused by type mismatches into successes, potentially increasing injection rate from 34% to 50%+ on symbolic problems.

### Cold Start Bootstrap

New signatures face a chicken-and-egg problem: can't prove effectiveness without being used, but the system won't use unproven signatures. We guarantee injection for the first **10 uses** to sample success rate. After bootstrap:
- High-performing signatures (≥80% success) continue injection
- Low-performing signatures fall back to LLM

### Exploration Phase: Let the System Breathe

When entering a new domain, the system needs room to explore and fail. **We deliberately accept lower accuracy in exchange for signal collection.**

**The Core Philosophy: Signal Over Accuracy**

In training mode (`ACTIVE_MODE = Mode.TRAINING`), we mandate DSL injection on every signature hit, regardless of historical lift data. Both outcomes provide value:

- **DSL succeeds** → positive lift recorded → reinforces this DSL
- **DSL fails** → negative lift recorded → teaches system to refine or avoid

A failed DSL execution is not wasted work—it's a data point. The system learns from mistakes. Short-term accuracy loss is the price of long-term improvement.

**Why We Don't Avoid Low-Success DSLs**

Traditional ML would gate on success rate: "this DSL only works 30% of the time, skip it." We reject this approach because:

1. **30% success is still signal** - those successes executed in ~1ms instead of ~500ms LLM calls
2. **Failures reveal patterns** - maybe the DSL fails on ratio problems but succeeds on simple arithmetic
3. **Refinement needs data** - to decompose a bad DSL into better children, we need to know *when* and *why* it fails
4. **Context matters** - the same DSL might have -40% lift in one semantic context and +55% in another

Instead of avoidance, we use the **refinement loop**: low-performing DSLs get decomposed into finer-grained child signatures with more precise DSLs.

**The Breathing Room Principle**

When entering a new problem domain:

```
Week 1-2: Exploration
  - Run aggressive injection mode
  - Accept 40-50% accuracy (vs 70%+ baseline)
  - Collect 100+ problem runs worth of lift data
  - Create hundreds of new signatures

Week 3-4: Refinement
  - Analyze low-lift signatures
  - Run refinement loop to decompose into children
  - Generate precise DSLs for child signatures
  - Parent signatures become routers

Week 5+: Harvest
  - Switch to benchmark mode
  - Lift-based gating routes traffic optimally
  - Accuracy exceeds baseline
  - Execution time drops (more DSL, less LLM)
```

**Current State (after 200+ L5 MATH problems):**

| Metric | Value |
|--------|-------|
| Total signatures | 2,150+ |
| Signature match rate | 100% |
| Injection rate | ~50% per run |
| DSL types | math (71%), sympy (19%), custom (7%), guidance (3%) |
| New signatures per 30 problems | ~15-20 |

**Key Insight:** A run with 45% accuracy that collects lift data on 70 DSL executions is more valuable than a run with 60% accuracy that avoids DSLs entirely. We're building the knowledge base now; we harvest the benefits when we flip to benchmark mode.

**The Payoff**

After sufficient exploration, switching to benchmark mode (`ACTIVE_MODE = Mode.BENCHMARK`) enables:
- Lift-based routing skips known-bad DSLs
- Proven DSLs execute deterministically
- Accuracy exceeds baseline LLM
- Latency drops significantly

The breathing room we give the system during exploration directly translates to performance gains in production.

### One Model Architecture: Quality Over Cost

We use Llama-3.3-70B for all LLM tasks: decomposition, step solving, and DSL generation. Why not use smaller models where possible?

**The DSL argument:** DSLs are cached forever and reused across thousands of problems. A bad DSL—one that mishandles edge cases or encodes incorrect logic—produces systematic errors that compound over time. The cost difference between 8B and 70B for DSL generation is negligible when amortized across thousands of future executions.

| Task | Frequency | Impact of Mistake | Model Choice |
|------|-----------|-------------------|--------------|
| Decomposition | Every problem | Medium (wrong steps) | 70B |
| Step solving | Every step | Medium (wrong answer) | 70B |
| DSL generation | Rare (new sigs only) | **HIGH** (cached forever) | 70B |

**The simplicity argument:** A single model simplifies the codebase, deployment, and debugging. No routing logic to maintain, no model-specific prompt tuning, no version mismatches. The complexity cost of a multi-model architecture outweighs the marginal cost savings.

**The quality-everywhere argument:** If 70B produces better decompositions (1.4 steps vs 7-10) and better DSLs, the accuracy gains compound. Each component benefits from the larger model's better reasoning. Trying to save costs on one component can degrade the entire pipeline.

**When multi-model makes sense:** If DSL generation were frequent (every problem), the cost argument would change. But with a maturing signature library, DSL generation becomes increasingly rare—most steps match existing signatures. The "smart model for everything" approach optimizes for the steady state where generation is rare but execution is frequent.

### Why Mycelium Wins

With the context fix in place, decomposition becomes an advantage:

1. **Structured reasoning**: Complex problems benefit from explicit step breakdown
2. **Pattern reuse**: Most steps match known signatures
3. **DSL acceleration**: Typed operations execute in ~0ms vs ~500ms
4. **Compound learning**: Each problem improves the library for future problems

The advantage will grow as the signature library matures across more problem types.

### Key Metrics to Watch

As the system runs in production, three metrics indicate health and growth:

| Metric | What It Measures | Healthy Range |
|--------|------------------|---------------|
| **Signatures per problem** | Decomposition granularity | 5-10 steps |
| **Signature hits per problem** | How many steps matched a signature | ~3 for L5 |
| **Injections per problem** | DSL coverage (sig had good DSL) | 2-5 injections |
| **Success rate per signature** | DSL quality | ≥70% for reliable sigs |

**Signatures per problem** reflects decomposition behavior. Too few (1-2) means problems aren't being broken down; too many (15+) means over-decomposition creating noise.

**Signature hits per problem** measures how well the signature library covers new problems. For Level 5 problems with a moderately mature database, expect ~3 signature hits per problem—roughly 5 DAG steps where 3 match existing signatures with high-confidence DSL. This ratio improves as the library matures; early runs may see 1-2 hits, while a mature library approaches 4-5. *Current benchmark: ~675 unique signatures across 56 step types yields 3.5 signature hits per problem on L5.*

**Injections per problem** shows how much the signature library is actually being used. Low injection rates indicate either (a) signatures don't match new problems, or (b) DSL quality is poor so lift-gating blocks injection.

**Success rate per signature** is the ground truth. Signatures with <70% success remain in probation; those with ≥70% become reliable and inject their DSL. Monitoring the distribution of success rates reveals whether the library is maturing.

**Expect the database to grow** as new problem types are encountered. Early runs create many signatures; later runs increasingly match existing ones. A healthy system shows signature creation rate declining over time while match rate increases.

---

## 6. Beyond Mathematics

The decomposition-and-reuse paradigm extends far beyond mathematical reasoning. Any domain where complex problems decompose into recurring sub-problems can benefit from signature-based learning.

**The Core Insight.** Mycelium's contribution isn't math-specific—it's a framework for *learning the atoms of any reasoning domain*. The signature database is a vocabulary; the matching system is grammar; the execution layer is fluency. What varies by domain is the embedding model, the success criteria, and the DSL primitives. The architecture remains constant.

Just as mycelium networks in nature decompose organic matter across ecosystems—from forest floors to grasslands—this computational mycelium can decompose problems across domains. The signature library is not a static knowledge base but a living network that grows, consolidates, and adapts as it encounters new problem types.

---

## 7. Limitations and Future Work

**Current Limitations:**
- Decomposition quality depends on LLM planner capabilities

**Addressed in This Work:**
- ~~Context loss in step isolation~~ → Fixed by passing original problem to all steps
- ~~DSL hurting general reasoning~~ → Fixed by selective injection (typed only)
- ~~Same DSL hurting some contexts~~ → Fixed by lift-based gating (auto-disable negative-lift signatures)
- ~~DSL parameter mapping failures~~ → Fixed by extracting numbers from step task + positional fallback
- ~~Poor DSL quality from Llama~~ → Fixed by having Claude generate all DSL scripts

**Future Directions:**
- **100% deterministic execution**: Improving signature coverage and DSL quality to achieve fully deterministic DAG execution without LLM calls
- Expand to other problem domains (coding, reasoning benchmarks)
- Contrastive learning for better signature separation
- Cross-problem dependency tracking
- Distributed signature sharing across deployments
- **Signature chaining**: Learn common sequences of signatures that co-occur. If "isolate variable" frequently precedes "substitute value" which precedes "simplify expression," the system could recognize and execute the entire chain as a unit. This transforms atomic signatures into reusable *solution pipelines*—multi-step recipes that reduce LLM calls and enable higher-level pattern matching across problem types.
- **Cluster consolidation**: Over time, near-duplicate signatures may emerge—steps phrased differently but semantically equivalent. Periodically merging similar signatures (high cosine similarity, similar success rates) could produce a cleaner library with stronger per-signature statistics.

---

## 8. Conclusion

We demonstrated that math problems decompose into a finite vocabulary of atomic signatures, and that **decomposition + signature reuse can improve LLM performance**.

The key insight: **the same LLM performs better when its reasoning is decomposed and cached**. The signature library acts as external memory that compounds knowledge across problems.

Critical learnings:

1. **Context propagation**: Every step needs the full problem, not just dependency results.

2. **LLM Script Rewriting**: DSL scripts use semantic names (`base`, `height`) but runtime context has generic keys (`step_1`, `step_2`). LLM rewrites scripts using actual variable names—bridging the semantic gap.

3. **Lift-based gating**: DSL helps arithmetic but hurts conceptual reasoning. Tracking success rates per-signature enables automatic routing—DSL where it helps, LLM where it doesn't.

4. **Self-improving system**: Every problem generates lift data. Failed DSL executions teach the system where not to inject. The signature library matures over time.

The "decomposition tax" was a bug, not a fundamental limitation. With proper context propagation and adaptive DSL routing, decomposition unlocks pattern reuse and knowledge accumulation unavailable to monolithic solving.

Mycelium demonstrates that LLMs can build persistent, reusable knowledge structures—moving beyond solving each problem from scratch toward genuine compound learning.

---

## Acknowledgments

I could not have built Mycelium without Claude. What would have taken months took weeks. The velocity of iteration—from half-formed idea to working code to refined architecture—was unlike anything I've experienced in two decades of software development. Claude didn't just write code; it challenged assumptions, proposed alternatives, caught edge cases I'd have discovered only in production, and maintained coherence across a growing codebase. The contribution was invaluable.

This project was developed through extensive human-AI pair programming: I provided vision, direction, and domain intuition; Claude contributed implementation capacity, systematic analysis, and an inexhaustible willingness to refactor when we found a better way. The collaboration model worked remarkably well—neither of us could have built this alone.

**A note on abstraction:** Delegating implementation to Claude freed me to think at a higher level of abstraction. Instead of getting lost in debugging SQLite queries or regex parsing, I could focus on *"should signatures boost their neighbors?"* and *"what makes a step DSL-hostile?"* The cognitive load shifted from syntax to semantics, from code to architecture. This is perhaps the real unlock of AI pair programming—not just faster coding, but thinking at a higher altitude.

**On the primes analogy:** Claude and I would go back and forth on implementation details, questions would come up, and my answers were largely guided by primes. The analogy was so strong that it made many decisions easy. Leaning into failing DSLs is scary when you're trying for SOTA results on a benchmark, but knowing that failing DSLs either need to be rewritten or decomposed further gave resolution about direction. During training, DSL failures provide good signal. A negative-lift signature is telling you it's not atomic yet and you need to decompose further.

---

## References

1. Wei, J., et al. (2022). Chain-of-thought prompting elicits reasoning in large language models. *NeurIPS*.
2. Gao, L., et al. (2023). PAL: Program-aided language models. *ICML*.
3. Lewis, P., et al. (2020). Retrieval-augmented generation for knowledge-intensive NLP tasks. *NeurIPS*.
4. Hendrycks, D., et al. (2021). Measuring mathematical problem solving with the MATH dataset. *NeurIPS*.
5. Kolodner, J. L. (1992). An introduction to case-based reasoning. *Artificial Intelligence Review*.

---
