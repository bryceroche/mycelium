# Mycelium

**Decompose math problems into reusable atomic signatures.**

**[Read the Paper](https://drive.google.com/file/d/1Gn8Efk4F2GW1bT3qGlHmKV-V_C6hIaLk/view)**

First solve of a problem type is hard. Second solve is easier. 100th solve is trivial.

Problems decompose into DAG-structured steps. MCTS routes each step through a tree of signatures, recursively decomposing until a matching DSL executes. Novel solutions become new signatures. The tree grows; future problems route faster.


## Quick Start (~5 min)

```bash
git clone https://github.com/bryceroche/mycelium
cd mycelium
pip install -r requirements.txt

# Download pre-trained DB (13K signatures, 342MB compressed)
gh release download v1.4.0
gunzip mycelium.db.gz embedding_cache.db.gz

export OPENAI_API_KEY=your_key
python scripts/pipeline_runner.py --dataset math --levels 1 2 --problems 20 --workers 4
```

**Requirements:** Python 3.11+, ~2GB disk for MathBERT

## Pre-trained Database

Download from [v1.4.0 release](https://github.com/bryceroche/mycelium/releases/tag/v1.4.0):

| File | Size | Contents |
|------|------|----------|
| `mycelium.db.gz` | 342 MB | 13,016 signatures (6,776 umbrellas, 6,240 leaves) |
| `embedding_cache.db.gz` | 25 MB | MathBERT embedding cache |

```bash
# Alternative: curl
curl -L https://github.com/bryceroche/mycelium/releases/download/v1.4.0/mycelium.db.gz | gunzip > mycelium.db
curl -L https://github.com/bryceroche/mycelium/releases/download/v1.4.0/embedding_cache.db.gz | gunzip > embedding_cache.db
```


## Stack

- **LLM:** gpt-4.1-nano with OpenAI API
- **DB:** SQLite + WAL mode
- **Embeddings:** MathBERT 768-dim

**Please read the [CLAUDE.md](CLAUDE.md) for the latest thinking.**
## License
MIT — Bryce Roche ([github.com/bryceroche/mycelium](https://github.com/bryceroche/mycelium))
Built with [Claude Code](https://claude.ai/claude-code)
