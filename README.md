# Mycelium

## Attention-Based Decomposition
**Decomposition is the crux.** Everything downstream of step-level intermediate representation (IR) is solved.

Transformer attention reveals semantic spans. The "Panama Hats" problem: "panama" = country, but "panama hats" = completely different meaning. We need the **longest span** that matches a semantic unit.

Standard embedding models learn lexical similarity, not operational similarity. They think "x + y" and "x * y" are similar because the tokens overlap. This is useless for math — we need to group by what computations do, not what they look like.

Our solution extracts implicit structure from LLMs and distills it into a lightweight classifier. The key insight: transformer attention patterns reveal semantic spans. When processing "she sold half her eggs," the attention weights show "half," "eggs," and "sold" attending to each other. That cluster is the model recognizing a single operation: multiply eggs by 0.5.

We extract attention matrices from a 7B model on 10K math problems and discovered two orthogonal signals. First, attention magnitude from numbers to verbs: state verbs like "has" produce ~0.05, action verbs like "sold" produce ~0.07. This distinguishes SET from transformations. Second, verb embeddings cluster semantically — "sold/spent/lost" cluster together (subtraction), "bought/received/earned" cluster together (addition).

We use Welford's algorithm to learn these thresholds online rather than hardcoding. Classification becomes a z-score: how many standard deviations from each operation's learned mean?

After classifying spans, we build a computation graph. Entity binding comes from attention sinks — all spans attend to the problem's subject, telling us they chain together. "Lay 16 eggs" → SET(16), "eats 3" → SUB(result, 3), execute in order.
## License
MIT — Bryce Roche ([github.com/bryceroche/mycelium](https://github.com/bryceroche/mycelium))
Built with [Claude Code](https://claude.ai/claude-code)
