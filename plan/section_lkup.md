# Section X: Mathematical Reasoning as Learned Table Lookup

## X.1 The Lookup Table Perspective

Strip away the terminology — attention distillation, wave function collapse, spectral decomposition — and what Mycelium actually builds is a lookup table. A very fancy one, but a lookup table nonetheless.

Consider what a traditional lookup table does: given a structured key, return a precomputed value. A multiplication table maps (7, 8) → 56. The key is the input pair. The value is the result. No reasoning occurs at query time — all the "intelligence" was front-loaded into constructing the table.

Mycelium does the same thing, except the keys are natural language math problems and the table is too large to enumerate. So instead of storing every possible entry, we learn a function that constructs the key at query time and a function that retrieves the value.

The pipeline is the key constructor:

- **C1** extracts the relevant fragments from the problem (which parts matter?)
- **C2** identifies what type of operation each fragment represents (what kind of entry is this?)
- **C3** extracts the specific operands (what are the coordinates in the table?)
- **C5** determines the order of operations (in what sequence do we chain lookups?)
- **C6** determines the output format (what column of the table do we read?)

The constructed key is a structured representation: a DAG of (template, operands) pairs with dependency edges. This key is precise enough that sympy — the table itself — can look up the answer deterministically. There is no ambiguity left. The key fully specifies the computation.

## X.2 Why Key Construction is the Hard Problem

A multiplication table is easy to use but was hard to construct — someone had to compute 56 = 7 × 8 and record it. Similarly, sympy evaluation is trivial, but knowing that "Natalia sold half as many clips in May" should be looked up under (DIVIDE, [48, 2]) is the entire challenge.

This is the fundamental insight: **mathematical reasoning is not about doing math. It is about converting natural language into a structured query against a symbolic computation engine.** The "reasoning" is the translation, not the execution.

Large language models conflate these two tasks. When GPT-4 solves a math problem, it simultaneously figures out what to compute and computes it, using the same autoregressive generation for both. This is like asking someone to simultaneously design a database schema and query it, in a single stream of consciousness, using only natural language. It works — impressively often — but it's architecturally incoherent. The model uses billions of parameters' worth of floating-point arithmetic to compute 48 / 2 = 24, a calculation that takes nanoseconds in sympy.

Mycelium separates key construction from table lookup. Six small models construct the key. Sympy performs the lookup. Each part does what it's good at.

## X.3 The Table Has Structure

The IB template discovery reveals that mathematical operations are not uniformly distributed across some vast continuous space. They cluster into discrete types with clean boundaries between them. The "table" has natural rows and columns.

This is why the lookup metaphor works so well. A continuous, unstructured space would resist tabulation — you'd need a model that can interpolate between entries. But discrete operation types with well-defined operand schemas are exactly the kind of structure that tables excel at. The C2 classifier selects the row (which template). The C3 extractor fills in the column values (which operands). The lookup is the sympy call.

The hierarchical structure discovered by β-annealing makes this even more table-like. At the coarsest level, there are ~10 operation families (arithmetic, algebraic manipulation, solving, evaluation, etc.). Within each family, 5–15 specific templates. The table has sections, subsections, and entries — exactly the structure you'd design by hand if you were building a mathematical operation reference.

But nobody designed it. The table structure emerged from attention patterns. The 7B teacher model, through the structure of its computation, implicitly encodes a taxonomy of mathematical operations. IB merely makes that implicit taxonomy explicit. The table was always there — hidden in the attention weights, distributed across 28 layers of a transformer. Mycelium extracts it and pins it to the wall.

## X.4 What the Teacher Knows That the Table Captures

When the 7B teacher solves "If m = r + 2 and m × r = 98, find m," it doesn't look up a table. It generates tokens autoregressively, attending to different parts of the input at each step. But the *structure* of that attention — which tokens it reads, in what order, grouped how — encodes a decomposition that is equivalent to a table lookup:

1. Recognize "m = r + 2" as EQUATION type (table row)
2. Recognize "m × r = 98" as EQUATION type (table row)
3. Recognize "find m" as SOLVE objective (output format)
4. Determine that step 3 depends on steps 1 and 2 (lookup order)
5. Extract operands: m, r, 2, 98 (column values)
6. Execute: sympy.solve([m - r - 2, m*r - 98], [m, r]) (table lookup)

The teacher "knows" this decomposition in the same way that a fluent speaker "knows" grammar — implicitly, encoded in weights, never articulated. IAF extraction articulates it. The six specialists learn to replicate each step of the articulated decomposition. The result is a system that performs the same lookup as the teacher but without the teacher's overhead of generating hundreds of tokens of chain-of-thought to arrive at the query.

## X.5 Generalization as Interpolation in Key Space

A traditional lookup table cannot generalize — if the key isn't in the table, there's no answer. Mycelium's learned key constructor can, because it operates in a continuous embedding space that supports interpolation.

A problem the model has never seen before still gets decomposed into templates it has seen before, with operands it can extract, in dependency structures it recognizes. The key is novel; the components of the key are familiar. Just as a multiplication table doesn't need an entry for 7.3 × 8.1 if you understand the underlying operation, Mycelium doesn't need to have seen the exact problem before — it needs to have seen the operation types and the operand patterns.

This is the sense in which the six specialists generalize: they've learned the structure of the key space from 7B attention patterns, not a fixed set of entries. New problems that fall within the span of known templates and operand patterns are handled automatically. Problems that require genuinely novel operation types — operations not present in the training data's attention patterns — will fail. But this failure mode is well-defined and diagnosable: C2 will produce a flat, low-confidence distribution (no matching row in the table), which MCTS will detect and report as an explicit "I don't know how to decompose this."

This is better than a monolithic model's failure mode, which is to confidently generate plausible-looking but wrong chain-of-thought with no indication that it has encountered an unfamiliar pattern.

## X.6 The Cost of Building the Table

The dominant cost of Mycelium is not inference — six 0.5B forward passes plus sympy is negligible. The dominant cost is **table construction**: running the 7B teacher on thousands of problems with full attention capture to generate the IAF data from which everything else is derived.

This cost is paid once. The 7B runs at training time, generates attention patterns, and is then discarded. The resulting table — six specialist models plus the IB template taxonomy — is compact, fast, and reusable. Every subsequent inference is a cheap lookup.

This amortization is the economic argument for Mycelium. A frontier model pays the full cost of reasoning on every query. Mycelium pays the full cost once, at table construction time, and amortizes it across unlimited cheap lookups. The breakeven point — where total Mycelium cost (construction + N lookups) equals total frontier cost (N full inferences) — is reached quickly, because the per-lookup cost ratio is approximately 1000:1.

We are, in the end, building a fancy lookup table. But it's a table whose structure was discovered from a teacher's attention patterns, whose keys are constructed by learned specialists, whose entries are verified by symbolic execution, and whose coverage extends across the full spectrum of mathematical operations. The lookup is cheap. The table is the product.
