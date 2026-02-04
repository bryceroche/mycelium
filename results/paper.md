# Mycelium: Attention Distillation for Math Word Problem Decomposition

**Author:** Bryce Roche, bryceroche@fungifactor.com, github.com/bryceroche/mycelium

## Abstract

We present Mycelium, a system for solving math word problems by distilling attention patterns from large language models into lightweight classifiers. The key insight is that transformer attention reveals semantic spans — contiguous tokens that form cohesive operations. We extract these patterns from Qwen 7B (7 billion parameters), train a mapping to predict them from MiniLM features (22 million parameters), and use an LLM with specialized templates at inference. Our approach achieves 94.5% correlation with the teacher model's attention patterns while being 318x smaller.

## 1. Introduction

Math word problems require decomposing natural language into executable operations. The challenge is identifying semantic spans — multi-token units that represent single operations. "Half the price of the cheese" is one operation (multiply by 0.5), not four separate words.

We call this the **Panama Hats Problem**: just as "panama" means country but "panama hats" means something entirely different, mathematical phrases have compositional meaning that naive tokenization destroys.

Large language models implicitly learn these span boundaries through attention patterns. Tokens within a semantic span attend strongly to each other. We extract this signal and distill it into a fast, lightweight system.

## 2. The Panama Hats Problem in Math

Why do we need span detection? Because meaning is compositional.

- "panama" = country
- "panama hats" = a type of hat (completely different meaning)

In math word problems, this is critical:
- "half" = 0.5
- "half the price of the cheese" = ONE operation (cheese_price × 0.5)
- "twice as many apples as oranges" = ONE comparison operation

Naive tokenization breaks these into separate words and loses the semantic unit. The Panama Hats problem guides our span creation: we need the **longest span** that forms a cohesive operation.

Attention patterns solve this: tokens within a semantic span attend strongly to each other (high connectivity). "half," "price," and "cheese" form an attention cluster — that's the model recognizing them as a single operation.

## 3. Attention Signals

We extract three signals from attention matrices:

**Attention Entropy (per token)**
- Low entropy → token attends to specific targets → important structural role
- High entropy → token attends broadly → less discriminative
- Use case: Identify operators and key nouns

**Attention Received (per token)**
- Sum of attention each token receives from all other tokens
- High received → many tokens look back to this one → entity or anchor
- Use case: Find subjects ("Janet"), referenced quantities

**Span Connectivity (per span)**
- Average mutual attention between tokens in a candidate span
- High connectivity → tokens form cohesive unit → valid span
- Low connectivity → tokens don't belong together → split or reject
- Use case: Validate span boundaries, detect multi-token operations

## 4. Why MiniLM is Perfect for Distillation

MiniLM was originally trained with: `loss = MSE(student_attention, teacher_attention)`

This means MiniLM already learned to mimic attention patterns from a larger teacher. When we fine-tune it on Qwen 7B attention patterns, it's doing exactly what it was designed for — just with a new teacher.

Key advantages:
- Bidirectional encoder (sees full context, unlike causal Qwen)
- Prior distillation training made it a good student
- Sentence-transformer architecture already optimized for semantic similarity
- The training objective aligns perfectly with our goal

## 5. Trained Signal Mapping

We have 17k spans with BOTH MiniLM embeddings AND Qwen attention signals. This lets us train a mapping:

`MiniLM features → predicted Qwen signals`

**Fine-tuning process:**
1. Extract Qwen 7B attention on 17k spans
2. Extract MiniLM embeddings on same 17k spans
3. Train mapping: predict Qwen signals from MiniLM features
4. Learn optimal head weights (heads 5 & 8 most important)
5. Learn optimal layer weights (layers 4 & 5 most important)
6. Result: 0.58 → 0.945 correlation

**Distillation results:**

| Model | Params | Correlation with Qwen 7B |
|-------|--------|--------------------------|
| MiniLM-L6 (fine-tuned) | 22M | 0.945 |
| MiniLM-L6 (baseline) | 22M | 0.58 |
| Qwen2-0.5B | 500M | 0.31 |
| BERT-base | 110M | 0.30 |

The key insight: architecture matters more than size. Same-family Qwen2-0.5B only achieves 0.31 correlation, while the much smaller MiniLM achieves 0.945 after fine-tuning.

## 6. Cross-Attention Between Spans

Spans don't exist in isolation. We track:

1. **Sequence awareness** — Position in the problem (first span usually SET, later spans usually operations)
2. **Previous span tracking** — What operation came before? (context for current span)
3. **Entity tracking** — Which entities have been introduced? Which are being referenced?

Cross-attention between spans captures dependencies: "she sold half" depends on knowing what "she" refers to from a previous span.

## 7. Inference Pipeline

At inference (no KNN — we use LLM with templates):

1. Run MiniLM (fast, 22M params)
2. Apply learned mapping → approximate Qwen signals
3. Use signals for span detection + template matching
4. LLM executes specialized template

No Qwen 7B needed at inference — just the trained mapping + LLM for execution.

## 8. Specialized Templates with Generic Entities

Each span maps to a specialized template. The LLM uses the detected span type directly.

**Examples:**
- Circle geometry: `area = π × {radius}²`
- Ratio: `{entity_a} = {ratio} × {entity_b}`
- Percentage: `{result} = {entity} × ({percent}/100)`
- Half of: `{result} = {entity} × 0.5`

**Generic entities:**
GSM8K problems mention many entities (apples, cookies, cheese). We use `{entity}` placeholders:
- "half the apples" → template: `{entity} × 0.5`
- "half the cookies" → same template: `{entity} × 0.5`

This gives us **specialized structure** (each operation type has its own template) with **generic applicability** (works for any entity).

Span detection identifies WHICH template. Entity extraction fills placeholders. LLM executes.

## 9. Results

End-to-end test on held-out samples shows **96.8% average correlation** — even better than training (94.5%).

| Sample Problem | Correlation with Qwen 7B |
|----------------|--------------------------|
| Natalia sold clips... | 0.965 |
| Weng earns $12... | 0.966 |
| Betty is saving money... | 0.972 |
| Julie is reading... | 0.972 |
| James writes a letter... | 0.964 |
| **Average** | **0.968** |

The model correctly identifies semantic spans:
- "Sally has 12 apples" → captured as single span
- "Tom found 8 coins" → captured as single span
- "A farmer has 20 eggs" → captured as single span

## 10. Core Principle: Failures Are Valuable

**Let the system fail.** This is how it learns.
- Record every failure — it feeds the learning loop
- Accumulated failure patterns (not individual failures) refine thresholds
- Success/failure stats drive classification decisions via Welford's algorithm

The goal is NOT 100% accuracy on every run. The goal is collecting data that makes the system smarter over time.

## 11. Related Work

**Attention Distillation.** MiniLM (Wang et al., 2020) demonstrated that attention patterns can be distilled from large to small models using MSE loss. We extend this to domain-specific attention patterns for math reasoning.

**Math Word Problems.** GSM8K (Cobbe et al., 2021) provides a benchmark of grade school math problems. Our approach complements chain-of-thought prompting by providing structured span detection.

**Compositional Semantics.** The Panama Hats problem relates to classical work on compositionality in linguistics — meaning is built from parts but isn't simply the sum of parts.

## 12. Conclusion

We demonstrated that:
1. Transformer attention patterns reveal semantic spans in math problems
2. These patterns can be distilled from Qwen 7B to MiniLM (318x smaller)
3. MiniLM's original training objective (MSE on attention) makes it ideal for this task
4. Cross-attention between spans captures sequence and entity dependencies
5. Specialized templates with generic entities enable execution

The key insight: large models learn span boundaries implicitly. We make this explicit through attention distillation, enabling fast inference with an LLM executing structured templates.

## Acknowledgments

Built with Claude Code. The velocity of iteration was extraordinary — Claude enabled thinking at a higher level of abstraction, focusing on system design rather than implementation details.

## References

- Cobbe, K., et al. (2021). Training Verifiers to Solve Math Word Problems. arXiv:2110.14168
- Wang, W., et al. (2020). MiniLM: Deep Self-Attention Distillation for Task-Agnostic Compression of Pre-Trained Transformers. NeurIPS 2020

---

**Open Source:** github.com/bryceroche/mycelium (MIT License)
