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
| Level 3 (Medium) | 80% | **82%** (+2) |
| Level 5 (Hardest) | 60% | **65%** (+5) |

Key metrics:
- **100%** signature match rate
- **4.4x** step reuse ratio (299 step instances → 68 unique signatures)
- **<1ms** DSL execution vs ~500ms LLM calls

## Quick Start

```bash
# Clone and setup
git clone https://github.com/bryceroche/mycelium.git
cd mycelium
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Set API key (get free key at console.groq.com)
export GROQ_API_KEY=your_key

# Test it works
python -m mycelium.solver "What is 15% of 80?"
```

## Reproduce Our Results

```bash
# Run Level 3 benchmark (50 problems, ~5 min)
python scripts/pipeline_runner.py --dataset math --level 3 --problems 50

# Run Level 5 benchmark (100 problems, ~15 min)
python scripts/pipeline_runner.py --dataset math --level 5 --problems 100

# Results saved to results/pipeline_results.json
```

**Expected output:**
```
Level 3: ~82% accuracy (41/50)
Level 5: ~65% accuracy (65/100)
```

**Requirements:**
- Python 3.11+
- ~2GB disk (for sentence-transformers model)
- Groq API key (free tier: 30 req/min, sufficient for benchmarks)

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
pytest

# Run benchmark with options
python scripts/pipeline_runner.py --dataset math --level 5 --problems 20 --workers 4

# Check signature database stats
python -c "from mycelium.step_signatures import StepSignatureDB; print(StepSignatureDB().get_stats())"
```

## License

MIT

## Author

Bryce Roche — [github.com/bryceroche/mycelium](https://github.com/bryceroche/mycelium)

Built with [Claude Code](https://claude.ai/claude-code) (Anthropic)

---

*Like mycelium networks that decompose organic matter and share nutrients across forests, this system decomposes complex problems into atomic patterns and distributes solutions through a shared database.*
