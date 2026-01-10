# Mycelium

**Decomposing Problems into Reusable Atomic Signatures**

Every composite number factors uniquely into primes. Math problems similarly decompose into a finite set of atomic *signatures*—reusable solution patterns. Mycelium builds a "table of primes" for mathematical reasoning: a signature database that grows as problems are solved.

## How It Works

1. **Decompose** problems into DAG-structured steps
2. **Match** each step against known signatures (cosine similarity)
3. **Execute** via stored routines (DSL, formula, or LLM guidance)
4. **Learn** by storing novel solutions as new signatures

The library grows; future problems get faster.

## Results

On MATH benchmark problems:

| Level | Direct LLM | Mycelium |
|-------|-----------|----------|
| Level 3 (Medium) | 80% | **82%** |
| Level 5 (Hardest) | 60% | **60%** |

Key metrics:
- **88.6%** signature match rate
- **4.4x** step reuse ratio (299 step instances → 68 unique signatures)
- **~0ms** DSL execution vs ~500ms LLM calls

## Key Insights

1. **Context matters**: Every step receives the original problem, not just dependency outputs
2. **Selective DSL**: Only typed signatures (compute_sum, solve_quadratic) get deterministic execution—general reasoning needs LLM flexibility
3. **Lift-based gating**: Same DSL can help (+55% lift) or hurt (-40% lift) depending on semantic context; tracked per-signature with automatic fallback

## Quick Start

```bash
pip install -r requirements.txt
export GROQ_API_KEY=your_key
python -m mycelium.solver "What is 15% of 80?"
```

## Stack

- **LLM**: Llama-3.3-70B via Groq API (free tier available)
- **Database**: SQLite (single file, no setup)
- **Embeddings**: all-MiniLM-L6-v2 (local, no API keys)

Total setup time: ~5 minutes.

## Architecture

```
src/mycelium/
├── solver.py            # Main solver
├── planner.py           # DAG decomposition
├── step_signatures/     # Signature DB + DSL executor
├── embedder.py          # Sentence embeddings
└── client.py            # Groq LLM client
```

## The DSL System

Signatures evolve from natural language templates to deterministic scripts:

1. New signature created with `method_template` (natural language)
2. Accumulates uses via LLM execution
3. Once reliable (≥5 uses, ≥80% success), generates DSL script
4. DSL executes in ~0ms; falls back to LLM if execution fails

Three DSL layers:
- **Math**: Safe arithmetic (`(a + b) / 2`, `sqrt(x)`)
- **SymPy**: Symbolic algebra (`solve(Eq(a*x + b, 0), x)`)
- **Custom**: Registered operators (`apply_quadratic_formula(a, b, c)`)

## Development

```bash
# Run tests
uv run pytest

# Run benchmark
uv run python scripts/run_benchmark.py --level 3 --n 50
```

## License

MIT

## Author

Bryce Roche — [github.com/bryceroche/mycelium](https://github.com/bryceroche/mycelium)

Built with [Claude Code](https://claude.ai/claude-code) (Anthropic)

---

*Like mycelium networks that decompose organic matter and share nutrients across forests, this system decomposes complex problems into atomic patterns and distributes solutions through a shared database.*
