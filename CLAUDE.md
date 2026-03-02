# Mycelium v6

Distill math reasoning from attention patterns

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


## A10G doesn't support bfloat16 Use float32 for compatibility
torch_dtype=torch.float32  

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
C2: What operations?         MiniLM-22M (multi-label)
C3: Extract operand spans    Qwen-0.5B (SQuAD-style start/end)
C4: Assemble expression      Deterministic (template + resolved operands)
Sympy: Evaluate
```

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

