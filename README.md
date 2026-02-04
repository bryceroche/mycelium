# Mycelium

Attention distillation for math word problem decomposition. Extract span structure from large models, run inference on small models.

## The Insight

Transformer attention patterns reveal semantic spans. When processing "she sold half her eggs," attention weights show "half," "eggs," and "sold" attending to each other — the model recognizes this as a single operation (multiply by 0.5).

We extract these patterns from Qwen 7B and distill them into MiniLM (22M params, 318x smaller).

## The Panama Hats Problem

Why do we need span detection? Because meaning is compositional.

- "panama" = country
- "panama hats" = a type of hat (completely different meaning)

In math word problems:
- "half" = 0.5
- "half the price of the cheese" = ONE operation (cheese_price × 0.5)

Naive tokenization breaks "half the price of the cheese" into separate words and loses the semantic unit. We need the **longest span** that forms a cohesive operation.

Attention patterns solve this: tokens within a semantic span attend strongly to each other (high connectivity). "half," "price," and "cheese" form an attention cluster — that's the model recognizing them as a single operation.

## Attention Signals

Three signals extracted from attention matrices:

| Signal | What it measures | Use case |
|--------|------------------|----------|
| **Entropy** | Low = focused attention = important token | Find operators, key nouns |
| **Received** | High = many tokens look back here | Find entities, anchors |
| **Connectivity** | High = tokens form cohesive unit | Validate span boundaries |

## Attention Distillation

| Model | Params | Correlation with Qwen 7B |
|-------|--------|--------------------------|
| **MiniLM-L6 (fine-tuned)** | **22M** | **0.945** |
| MiniLM-L6 (baseline) | 22M | 0.58 |
| Qwen2-0.5B | 500M | 0.31 |
| BERT-base | 110M | 0.30 |

**Key insight**: MiniLM used attention distillation from a larger teacher. The training objective was literally: loss = MSE(student_attention, teacher_attention)

Fine-tuning improved correlation from 0.58 → 0.945 (+63%).

![Distillation Results](screen_shots/distillation_results.png)

## Dual-Signal Architecture

Two orthogonal signals for robust matching:

1. **Attention** — Structural relationships (which tokens attend to each other)
2. **Embeddings** — Semantic similarity (centroid distance)

**Pipeline:**
- **Training**: Qwen 7B → attention patterns + centroid embeddings → span templates
- **Inference**: MiniLM 22M → match patterns + embeddings → span template → custom DSL

Quality of 7B model at cost of 22M model.

## Specialized Templates

Each span type has a specialized template with generic entity placeholders.

**Examples:**
- Circle geometry: `area = π × {radius}²`
- Ratio: `{entity_a} = {ratio} × {entity_b}`
- Percentage: `{result} = {entity} × ({percent}/100)`
- Half of: `{result} = {entity} × 0.5`

**Why generic entities?**

Instead of hardcoding "apples" or "cookies", we use `{entity}` placeholders:
- "half the apples" → `apples × 0.5`
- "half the cookies" → `cookies × 0.5`
- Both match the same template: `{entity} × 0.5`

This gives us **specialized structure** (each operation type has its own template) with **generic applicability** (works for any entity).

The span detection identifies WHICH template to use. The entity extraction fills in the placeholders.

## Results

End-to-end test on held-out samples shows **96.8% average correlation** — even better than training (94.5%).

![End to End Test](screen_shots/end_to_end_test.png)

## Core Principle

**Let the system fail.** Failures feed the learning loop. No LLM fallback. Accumulated failure patterns refine thresholds via Welford's algorithm.

## License

MIT — Bryce Roche ([github.com/bryceroche/mycelium](https://github.com/bryceroche/mycelium))

Built with [Claude Code](https://claude.ai/claude-code)
