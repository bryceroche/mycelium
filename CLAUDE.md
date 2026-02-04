# The Big 6
1. Our "Panama Hats" Problem (guides span creation)
2. Attention Signals (Entropy, Received, Connectivity)
3. Why MiniLM is Perfect (trained with MSE attention loss)
4. Trained Signal Mapping (17k spans dataset)
5. Cross-Attention Between Spans
6. Building the Graph

## Terminology
- **Attention Entropy** — Low entropy = important token (focused attention). High entropy = diffuse attention.
- **Attention Received** — Which tokens get looked back to. High received attention = structurally important (entities, operators).
- **Span Connectivity** — How strongly tokens within a span attend to each other. High connectivity = cohesive semantic unit.
- **Centroid Embedding** — Average embedding of a span's tokens. Used for template matching.
- **Welford's** — Online algorithm for calculating running mean/variance without storing all data.
- **Span** — Contiguous tokens forming a semantic unit (e.g., "half the eggs").
- **SET** — Initial value assignment operation.
- **Attention Sink** — Token that receives attention from many others (usually the subject/entity).

## Our Panama Hats Problem
How to put sub-graphs togther into one graph?  Span detection guides sub-graph composition
- "panama" = country
- "panama hats" = a type of hat (completely different meaning)

We're lookinig for the longest continuous sequence that retains attention connectivity.  Naive tokenization breaks these into separate words and loses the semantic unit. The Panama Hats problem guides our span creation: we need the **longest span** that forms a cohesive operation.

## Core Principle: Failures Are Valuable Data Points
**Let the system fail.** This is how it learns.
- Record every failure — it feeds the learning loop
- Accumulated failure patterns (not individual failures) refine thresholds
- Success/failure stats drive classification decisions

The goal is NOT 100% accuracy on every run. The goal is collecting data that makes the system smarter over time.

## Attention Signals

Three signals extracted from attention matrices:

**1. Attention Entropy (per token)**
- Low entropy → token attends to specific targets → important structural role
- High entropy → token attends broadly → less discriminative
- Use case: Identify operators and key nouns

**2. Attention Received (per token)**
- Sum of attention each token receives from all other tokens
- High received → many tokens look back to this one → entity or anchor
- Use case: Find subjects ("Janet"), referenced quantities

**3. Span Connectivity (per span)**
- Average mutual attention between tokens in a candidate span
- High connectivity → tokens form cohesive unit → valid span
- Low connectivity → tokens don't belong together → split or reject
- Use case: Validate span boundaries, detect multi-token operations

## Why MiniLM is Perfect for Distillation

MiniLM was originally trained with: `loss = MSE(student_attention, teacher_attention)`

This means MiniLM already learned to mimic attention patterns from a larger teacher. When we fine-tune it on Qwen 7B attention patterns, it's doing exactly what it was designed for — just with a new teacher.

## Trained Signal Mapping (17k Spans)
**The dataset:**
We have 17k spans with BOTH MiniLM embeddings AND Qwen attention signals. This lets us train a mapping:

`MiniLM features → predicted Qwen signals`

**Fine-tuning process:**
- Extract Qwen 7B attention on 17k spans
- Deduplicate and embed 17k spans -> 200 specialized span templates with custom DSL (sub graph)
- Extract MiniLM embeddings on same 17k spans
- Train mapping: predict Qwen signals from MiniLM features ~95% correlation

## Cross-Attention Between Spans

Spans don't exist in isolation. We track:
1. **Sequence awareness** — Position in the problem (first span usually SET, later spans usually operations)
2. **Previous span tracking** — What operation came before? (context for current span)
3. **Entity tracking** — Which entities have been introduced? Which are being referenced?

Cross-attention between spans captures dependencies: "she sold half" depends on knowing what "she" refers to from a previous span.

## Inference Pipeline

1. Run MiniLM (fast, 22M params)
2. Apply learned mapping → approximate Qwen signals
3. Use signals for span detection + template matching
4. LLM executes specialized template

No Qwen 7B needed at inference — just the trained mapping + LLM for execution.

## Specialized Templates with Generic Entities

Each span maps to a specialized template.  LLM matches problem text to our span templates which are sub-graphs that are composed via attention span connectivity.

**Examples:** Circle geometry, Ratio, Percentage, Half of

**Generic entities:**
GSM8K problems mention many entities (apples, cookies, cheese). We use `{entity}` placeholders:

## Building the graph
 - match span templates to subgraphs
 - granularity – guided by our “panama hats” problem
 - Spans -  guide subgraph boundaries
 - Subgraph composition - guided by attention span connections 

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
