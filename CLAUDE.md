# Mycelium

Attention distillation for math word problem decomposition. Extract span structure from large models, use LLM with specialized templates at inference.

## Terminology

- **Attention Entropy** — Low entropy = important token (focused attention). High entropy = diffuse attention.
- **Attention Received** — Which tokens get looked back to. High received attention = structurally important (entities, operators).
- **Attention Span Connectivity** — How strongly tokens within a span attend to each other. High connectivity = cohesive semantic unit.
- **Centroid Embedding** — Average embedding of a span's tokens. Used for template matching.
- **Span** — Contiguous tokens forming a semantic unit (e.g., "half the eggs").
- **SET** — Initial value assignment operation.
- **Attention Sink** — Token that receives attention from many others (usually the subject/entity).
- **SubGraphDSL** — Dataclass (`subgraph_dsl.py`) defining a composable sub-graph: params (from span text), inputs (from upstream sub-graphs), steps (ordered computation), single output. 1:1 with each deduplicated template.
- **DAG** — Directed Acyclic Graph. Sub-graphs compose into a DAG, not a tree. Supports convergence (multiple inputs) and fan-out (one output to many downstream).

## Panama Hats Problem & Span Boundary Detection

- "panama" = country
- "panama hats" = a type of hat (completely different meaning)

We want the longest continuous sequence that retains attention connectivity. Naive tokenization breaks these into separate words and loses the semantic unit. The Panama Hats problem guides our span creation: we need the **longest span** that forms a cohesive operation.

**The algorithm:** greedily extends span boundaries while attention connectivity stays high. For each candidate span:
1. Compute average mutual attention between all token pairs in the span
2. If connectivity > threshold, the tokens form a cohesive unit — try extending
3. When adding a token drops connectivity below threshold, the boundary is found

One span = one operation = one SubGraphDSL = one output.

## The Pipeline

1. Training and generalization (see below)
2. Compute atomic span centroids for cosine clustering
3. Collapse span count at cosine similarity threshold
4. Create sub-graph DSLs → Python dataclass
5. Fine-tune MiniLM on atomic spans
6. Inference with MiniLM → vectorized centroid matrix

## Training and Generalization

The first pass through GSM8K with Qwen creates ~25k spans with our Panama Hats algorithm. The second pass summarizes each span with a limited vocabulary. The third pass uses the vocabulary-constrained summarized spans to create finer-grained atomic spans.

## Attention Signals

Three signals extracted from attention matrices:

**Attention Entropy (per token)**
- Low entropy → token attends to specific targets → important structural role
- High entropy → token attends broadly → less discriminative

**Attention Received (per token)**
- Sum of attention each token receives from all other tokens
- High received → many tokens look back to this one → entity or anchor

**Span Connectivity (per span)**
- Average mutual attention between tokens in a candidate span
- High connectivity → tokens form cohesive unit → valid span
- Low connectivity → tokens don't belong together → split or reject

## Cross-Attention Between Spans

1. Sequence awareness → position in the problem
2. Previous span tracking → what operation came before
3. Entity tracking → which entities have been introduced and referenced

Cross-attention between spans captures dependencies: "she sold half" depends on knowing what "she" refers to from a previous span. This guides sub-graph composition.

## Why MiniLM is Perfect for Distillation

MiniLM was originally trained with: `loss = MSE(student_attention, teacher_attention)`

This means MiniLM already learned to mimic attention patterns from a larger teacher. When we fine-tune it on Qwen 7B attention patterns, it's doing exactly what it was designed for — just with a new teacher.

## SubGraphDSL

Every atomic span has one `SubGraphDSL` — a composable computation graph:

1. **params** — values extracted from span text at inference
2. **inputs** — values wired from upstream sub-graphs
3. **steps** — ordered computation. Operators: SET, ADD, SUB, MUL, DIV, MOD, NEG
4. **output** — single value exposed to downstream sub-graphs

Sub-graphs compose into a DAG via their typed ports. An output from one span wires into an input of another. Cross-attention between spans determines the wiring. Topological sort, execute in order, done.

## Specialized Templates with Generic Entities

Each span maps to a specialized template, and the LLM executes the DSL. We use generic entity placeholders instead of specific nouns from GSM8K. "Half the apples" and "half the cookies" both map to the same template. By replacing specific nouns with `[ENTITY]` placeholders, we get specialized structure with generic applicability.

## Building the Graph

- Match MiniLM spans to span templates
- Sub-graph composition guided by cross-attention

Topologically sort the final graph and execute each node in order. Each template's DSL runs with its extracted values. The final node's output is the answer.

## Overfitting

The system is a lookup table — and that's the point. We're cataloging a finite set of operations, not predicting an unbounded distribution.

Traditional ML guards against memorization because training data is a sample from a larger space. But mathematical operations aren't sampled from infinity. There are only so many ways to add, subtract, multiply, divide, and compose them. Once you've seen "sold half," "gave away a third," and "lost a quarter," you've covered fractional reduction. New phrasings map to existing operations.

Templates aren't answers, they're operation types. Memorizing "Sally has 5 apples" doesn't help you solve "Bob has 7 oranges" — but recognizing both as SET operations does. The lookup table maps surface variation to operational invariants.

We believe this extends beyond GSM8K to higher math and other domains with finite operational vocabularies under infinite lexical variation.

## Results

End-to-end test on held-out samples shows 96.8% average correlation — better than training (94.5%).

## Conclusion

We demonstrated that:
1. Transformer attention patterns reveal semantic spans in math problems
2. These patterns can be distilled from Qwen 7B to MiniLM → 318x smaller
3. Vocabulary reduction strips lexical noise while preserving operational integrity
4. MiniLM's training makes it ideal for this task
5. Specialized templates with generic entities enable execution
6. Span templates compose into computation graphs via cross-attention
7. Fine-tuning MiniLM closes the train/inference gap

Large models learn span boundaries implicitly. We make this explicit through attention distillation, enabling fast inference with an LLM executing structured templates.

## Future Work

**MCTS for Chaining:** We currently assume uniform chaining probability across atomic spans — that any span is equally likely to compose with any other. In practice, chaining probability is conditioned on operation type, entity binding, and position in the problem. This is a known simplification and a clear axis for improvement. MCTS rollouts over chaining configurations could learn this transition distribution.

**Layers of Abstraction:** There are layers of abstraction we could apply to our lookup table. Humans are good at recognizing patterns across scales and frequencies. We suspect these abstraction layers correspond to matching over compressed representations of composed graphs.

## New Favorite Pattern

Consolidate methods. All database connections go through a data layer. All span detection through one interface. All embedding lookups through cache. Reduces bugs, simplifies codebase.

# How to use Beads

```bash
bd prime        # Load context from beads
bd ready        # See available work
```

**When you encounter a bug or feature idea, create a beads issue to track it.**

```bash
bd create --title="Bug: description" --type=bug
bd create --title="Feature: description" --type=feature
```

Don't fix and forget - always track issues in beads.

## Workflow

1. Check `bd ready` for available issues
2. `bd update <id> --status=in_progress` to claim work
3. Make changes
4. `bd close <id> --reason="..."` when done
5. `bd sync` to sync changes

See `AGENTS.md` for detailed guidance.
