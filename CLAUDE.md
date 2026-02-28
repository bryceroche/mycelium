# Mycelium v6

## Large Files will FREEZE you
Check the size of the file before opening it
Do not open any file over 5MB 
Many of our files in S3 are over 5MB DO NOT OPEN THEM
never cat or print large JSON files. 
Use head -c 1000 or python -c "import json; d=json.load(open(f)); print(len(d))" for inspection

**Safe preview method:**
```bash
aws s3 cp s3://mycelium-data/path/to/file.json - | head -c 3000
```

## Use Lambda Map Reduce to process S3 files
Do not copy them to EC2 VM
Process directly in S3 with Lambda Map Reduce 
Set Lambda memory to 3GB NOT 1GB


## GSM8K Data Quarantine

All GSM8K data has been moved to `s3://mycelium-data/archive/gsm8k/`.
**NEVER use GSM8K for training C2/C3** - it only has 6 basic operations.
**ALWAYS use MATH data** - it has 40 IB templates.

## avoid these patterns like the plague  
if you spot these patterns in our repo pls purge immediately!
avoid keyword heuristics
avoid hardcoded patterns
avoid regex

## What we are building

A 72B teacher model solves math problems via chain-of-thought. We extract computation structure from its attention patterns and distill it into three 0.5B student models that reproduce the reasoning without generating any text at inference.

```
TRAINING:  Teacher solves problems → extract spans via JSD → train 3 specialists
INFERENCE: Problem text → 3 specialists → candidate search → symbolic executor → answer
```

## Current Architecture 

| Model | Task | Approach |
|-------|------|----------|
| C2: Classifier | Span group → operation (ADD/SUB/MUL/DIV) | Sequence classification |
| C3: Extractor | Operation → typed arguments | Generative (Instruct) |

All models are Qwen-0.5B (~500M params each).

## The Inference Pipeline

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

