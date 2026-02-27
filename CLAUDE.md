# Mycelium v6

## avoid these like the plague!  if you spot these patterns in our repo pls purge immediately!
avoid keyword heuristics
avoid hardcoded patterns
avoid regex

## What This Is

A 72B teacher model solves math problems via chain-of-thought. We extract computation structure from its attention patterns and distill it into three 0.5B student models that reproduce the reasoning without generating any text at inference.

```
TRAINING:  Teacher solves problems → extract spans via JSD → train 3 specialists
INFERENCE: Problem text → 3 specialists → candidate search → symbolic executor → answer
```


## Core Scripts

**Inference** (`inference/`):
- `e2e_pipeline.py` — Main E2E pipeline
- `executor.py` — Symbolic executor
- `candidate_generator.py` — Span groupings
- `eval_quick.py` — Evaluation
- `diagnostic.py` — Error attribution

**Training** (`train/`):
- `train_*.py` — Train models
- `generate_*.py` — Generate data


## Current Architecture (3 Models)

| Model | Task | Approach |
|-------|------|----------|
| C1: Relevance Scorer | Per-token relevance (0-1) | Regression head (MSE loss) |
| C2: Classifier | Span group → operation (ADD/SUB/MUL/DIV) | Sequence classification |
| C3: Extractor | Operation → typed arguments | Generative (Instruct) |

All models are Qwen-0.5B (~500M params each).

**C1 Note:** Outputs continuous relevance scores, NOT binary IO tags. Teacher attention is scattered (non-contiguous clusters), not ribbons. Binary tagging = premature collapse. See `plan/six_model.md`.


## The Inference Pipeline

1. **Segment** problem text into spans (C1)
2. **Generate candidate groupings** (search, not learned — 5-15 candidates)
3. **Classify + extract** each group (C2, C3)
4. **Execute + score + pick** best answer

**Key trick:** Batch all candidates together. Cuts latency from ~20s to ~2s per problem.


## Core Principles

**Search where the space is small, learn where the space is large.**
Groupings (5-15 candidates): searched. Classification (language understanding): learned.

**The executor is the validator.**
Models don't need to be perfect. Bad predictions fail to produce valid answers → eliminated.

**Error attribution drives development.**
Every improvement came from diagnosing failures.


## Aspirational: Full 6-Model Architecture

The current 3-model architecture achieves 99.95%. Planned additions for MATH dataset:

| Model | Task | Why Needed |
|-------|------|------------|
| C4: Implicit Ops | Bridging pattern detection | Handle "total", "average", "remaining" |
| C5: Dep Resolver | Wire execution DAG | Connect operations that share values |
| C6: Goal Resolver | Answer type + hint | "How much change?" → try SUB |

**Domain constants** — only genuinely orphaned values (>80% orphan rate):
```python
DOMAIN_CONSTANTS = {
    60: "min/hr", 24: "hr/day", 7: "day/wk",
    52: "wk/yr", 12: "mo/yr", 365: "day/yr",
    100: "%", 1000: "g/kg",
}
```
Small numbers (2-10) excluded — they're in problem text or derived, not world knowledge.

**Bridging templates** (IB-discovered):
- ACC_ADD: sum all results ("total", "altogether")
- ACC_SUB: max minus others ("remaining", "change")
- AVG: mean of results ("average")
- DERIVED_MUL: product of recent results (unit conversion)


## Training Data Prep (Lambda MapReduce)

**Bring compute to data, not data to compute.**

For large-scale training data extraction (C1 relevance, C2 templates, etc.):
- Chunk large files into ~200MB pieces ("golden_medusa" chunks)
- Process chunks via AWS Lambda in parallel
- Lambda reads/writes S3 directly, no data movement

```
S3 chunks (200MB) → 74 parallel Lambdas → S3 (extracted data)
```

**Why not EC2?** Moving 16GB+ to/from EC2 is slow and costs egress. Lambda runs adjacent to S3.

**Golden Medusa chunks:** `s3://mycelium-data/iaf_extraction/chunked/`


## Beads Workflow

```bash
bd prime        # Load context
bd ready        # See available work
bd create --title="..." --type=bug|task|feature
bd close <id>   # Complete work
```


## Shell Command Delegation

**Always delegate long-running bash/SSH commands to Sonnet or Haiku subagents.** This keeps the main conversation responsive while remote GPU/VM work runs in parallel.

Use `haiku` for quick status checks, `sonnet` for complex setup/debugging.

```python
# Good - user can keep talking while this runs
Task(
    prompt="SSH to VM and run extraction...",
    subagent_type="Bash",
    model="sonnet"
)

# Bad - blocks conversation for 30+ seconds
Bash(command="ssh ubuntu@... long command")
```

Bozeman MT

