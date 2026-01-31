# Mycelium

| Before (v2.x) | After (v3.0.0) |
|---------------|----------------|
| Hierarchical tree traversal | Flat k-NN classification |
| DSL code execution | Function pointer registry |
| Fixed recursion depth | Similarity-trend recursion |
| Separate signatures | Merged prototypes with variants |
| No LLM guidance | Signature menu "phrasebook" |

### The Three Components

```
Problem Text
     │
     ▼
┌─────────────────────────────────┐
│  LLM Decomposer                 │
│  (with signature menu)          │
└─────────────────────────────────┘
     │
     ▼ [atomic steps with func names]
┌─────────────────────────────────┐
│  Signature Store (k-NN)         │
│  ~1000 prototypes               │
└─────────────────────────────────┘
     │
     ▼ [best matching signature]
┌─────────────────────────────────┐
│  Function Registry              │
│  call_function(name, *args)     │
└─────────────────────────────────┘
     │
     ▼
   Result
```

- **LLM**: Recursive decomposition with signature examples as guidance
- **Tree**: Semantic → function mapping via k-NN classification
- **Python**: Deterministic execution via function registry (57-200 functions)


## Quick Start (~5 min)

```bash
git clone https://github.com/bryceroche/mycelium
cd mycelium
pip install -r requirements.txt

# Download pre-trained DB (13K signatures, 342MB compressed)
gh release download v1.4.0
gunzip mycelium.db.gz embedding_cache.db.gz

# Inference (just needs API key from Google AI Studio)
export GOOGLE_API_KEY=your_key
python scripts/pipeline_runner.py --dataset math --levels 1 2 --problems 20 --workers 4
```


## License
MIT — Bryce Roche ([github.com/bryceroche/mycelium](https://github.com/bryceroche/mycelium))
Built with [Claude Code](https://claude.ai/claude-code)
