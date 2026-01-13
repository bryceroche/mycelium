# Mycelium

**Decompose math problems into reusable atomic signatures.**

Problems decompose into DAG-structured steps. Each step matches against known signatures (cosine similarity) and executes via DSL (~0ms) instead of LLM (~500ms). Novel solutions become new signatures. The library grows; future problems get faster.

## Quick Start (~5 min)

```bash
git clone https://github.com/bryceroche/mycelium.git && cd mycelium
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Pre-trained signatures (skip cold start)
gh release download v1.0-db
mv dsl_*.json src/mycelium/

export GROQ_API_KEY=your_key  # free at console.groq.com

python scripts/pipeline_runner.py --dataset math --levels 5 --problems 20 --workers 4
```

**Requirements:** Python 3.11+, ~2GB disk (MathBERT), Groq API key (free tier)

## Stack

- **LLM:** Llama-3.3-70B via Groq (free tier)
- **DB:** SQLite + WAL mode
- **Embeddings:** MathBERT 768-dim

## Stats

~4k signatures, ~5 DAG steps per L5 problem, ~3.5 injectable steps per problem.

## License

MIT — Bryce Roche ([github.com/bryceroche/mycelium](https://github.com/bryceroche/mycelium))

Built with [Claude Code](https://claude.ai/claude-code)
