# Mycelium

**Decompose math problems into reusable atomic signatures.**
**Please read the [CLAUDE.md](CLAUDE.md) for the latest thinking.**
**The Paper is in the github repo**

First solve of a problem type is hard. Second solve is easier. 100th solve is trivial.

Problems decompose into DAG-structured steps. MCTS routes each step through a tree of signatures, recursively decomposing until a matching DSL executes. Novel solutions become new signatures. The tree grows; future problems route faster.


## Quick Start (~5 min)

```bash
git clone https://github.com/bryceroche/mycelium
cd mycelium
pip install -r requirements.txt
export OPENAI_API_KEY=your_key
python scripts/pipeline_runner.py --dataset math --levels 1 2 --problems 20 --workers 4

```

**Requirements:** Python 3.11+, ~2GB disk MathBERT

## Stack

- **LLM:** gpt-4.1-nano with OpenAI API
- **DB:** SQLite + WAL mode
- **Embeddings:** MathBERT 768-dim

## License

MIT — Bryce Roche ([github.com/bryceroche/mycelium](https://github.com/bryceroche/mycelium))

Built with [Claude Code](https://claude.ai/claude-code)
