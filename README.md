# Mycelium

Attention distillation for math word problem decomposition. Extract span structure from large models, run inference on small models.

## The Insight

Transformer attention patterns reveal semantic spans. When processing "she sold half her eggs," attention weights show "half," "eggs," and "sold" attending to each other — the model recognizes this as a single operation (multiply by 0.5).

We extract these patterns from Qwen 7B and distill them into MiniLM (22M params, 318x smaller).

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

## Dual-Signal Architecture

Two orthogonal signals for robust matching:

1. **Attention** — Structural relationships (which tokens attend to each other)
2. **Embeddings** — Semantic similarity (centroid distance)

**Pipeline:**
- **Training**: Qwen 7B → attention patterns + centroid embeddings → span templates
- **Inference**: MiniLM 22M → match patterns + embeddings → classify spans

Quality of 7B model at cost of 22M model.

## Core Principle

**Let the system fail.** Failures feed the learning loop. No LLM fallback. Accumulated failure patterns refine thresholds via Welford's algorithm.

## License

MIT — Bryce Roche ([github.com/bryceroche/mycelium](https://github.com/bryceroche/mycelium))

Built with [Claude Code](https://claude.ai/claude-code)
