# Mycelium

Attention distillation for math word problem decomposition. Extract span structure from large models, use LLM with specialized templates at inference.

## The Insight

Transformer attention patterns reveal semantic spans. When processing "she sold half her eggs," attention weights show "half," "eggs," and "sold" attending to each other — the model recognizes this as a single operation (multiply by 0.5).

We extract these patterns from Qwen 7B, train a mapping to predict them from MiniLM features, then use an LLM with specialized templates at inference.

## The Panama Hats Problem

Why do we need span detection? Because meaning is compositional.

- "panama" = country
- "panama hats" = a type of hat (completely different meaning)

**In math word problems, this is critical:**
- "half" = 0.5
- "half the price of the cheese" = ONE operation (cheese_price × 0.5)
- "twice as many apples as oranges" = ONE comparison operation

Naive tokenization breaks these into separate words and loses the semantic unit. The Panama Hats problem guides our span creation: we need the **longest span** that forms a cohesive operation.

**How attention solves this:**
Tokens within a semantic span attend strongly to each other (high connectivity). "half," "price," and "cheese" form an attention cluster — that's the model recognizing them as a single operation. This guides where to draw span boundaries.

## Attention Signals

Three signals extracted from attention matrices:

| Signal | What it measures | Use case |
|--------|------------------|----------|
| **Entropy** | Low = focused attention = important token | Find operators, key nouns |
| **Received** | High = many tokens look back here | Find entities, anchors |
| **Connectivity** | High = tokens form cohesive unit | Validate span boundaries |

## Why MiniLM is Perfect for Distillation

MiniLM was originally trained with: `loss = MSE(student_attention, teacher_attention)`

This means MiniLM already learned to mimic attention patterns from a larger teacher. When we fine-tune it on Qwen 7B attention patterns, it's doing exactly what it was designed for — just with a new teacher.

**Fine-tuning process:**
1. Extract Qwen 7B attention on 17k spans
2. Extract MiniLM embeddings on same 17k spans
3. Train mapping: predict Qwen signals from MiniLM features
4. Learn optimal head weights (heads 5 & 8 most important)
5. Learn optimal layer weights (layers 4 & 5 most important)
6. Result: 0.58 → 0.945 correlation

| Model | Params | Correlation with Qwen 7B |
|-------|--------|--------------------------|
| **MiniLM-L6 (fine-tuned)** | **22M** | **0.945** |
| MiniLM-L6 (baseline) | 22M | 0.58 |
| Qwen2-0.5B | 500M | 0.31 |
| BERT-base | 110M | 0.30 |

![Distillation Results](screen_shots/distillation_results.png)

## Cross-Attention Between Spans

Spans don't exist in isolation. We track:

1. **Sequence awareness** — Position in the problem (first span usually SET, later spans usually operations)
2. **Previous span tracking** — What operation came before? (context for current span)
3. **Entity tracking** — Which entities have been introduced? Which are being referenced?

Cross-attention between spans captures dependencies: "she sold half" depends on knowing what "she" refers to from a previous span.

## Trained Signal Mapping

**The fast approximation:**

We have 17k spans with BOTH MiniLM embeddings AND Qwen attention signals. This lets us train a mapping:

`MiniLM features → predicted Qwen signals`

At inference:
1. Run MiniLM (fast, 22M params)
2. Apply learned mapping → approximate Qwen signals
3. Use signals for span detection + template matching
4. LLM executes specialized template

No Qwen 7B needed at inference — just the trained mapping.

## Specialized Templates

Each span maps to a specialized template. No KNN lookup — the LLM uses the detected span type directly.

**Examples:**
- Circle geometry: `area = π × {radius}²`
- Ratio: `{entity_a} = {ratio} × {entity_b}`
- Percentage: `{result} = {entity} × ({percent}/100)`
- Half of: `{result} = {entity} × 0.5`

**Generic entities:**
GSM8K problems mention many entities (apples, cookies, cheese). We use `{entity}` placeholders:
- "half the apples" → template: `{entity} × 0.5`
- "half the cookies" → same template: `{entity} × 0.5`

Span detection identifies WHICH template. Entity extraction fills placeholders. LLM executes.

## Results

End-to-end test on held-out samples shows **96.8% average correlation** — even better than training (94.5%).

![End to End Test](screen_shots/end_to_end_test.png)

## Core Principle

**Let the system fail.** Failures feed the learning loop. Accumulated failure patterns refine thresholds via Welford's algorithm.

## License

MIT — Bryce Roche ([github.com/bryceroche/mycelium](https://github.com/bryceroche/mycelium))

Built with [Claude Code](https://claude.ai/claude-code)
