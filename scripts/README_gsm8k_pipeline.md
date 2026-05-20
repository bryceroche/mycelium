# GSM8K Step-Target Pipeline (Haiku-distilled)

Bridges the v54+ rep-space-thinking paradigm (which needs per-step gen_targets) with GSM8K (which only ships final-answer solutions).

## Pipeline

1. **Generate per-step targets** with Haiku → JSONL
2. **Load into curriculum** via `load_gsm8k_steps`
3. **Train** with per-breath supervision (bucket by step count for uniform-K batches)

## Step 1: Generate step targets

Install the anthropic SDK (one-time):
```bash
/home/bryce/mycelium/.venv/bin/pip install anthropic
```

Run a small test first to validate format + acceptance rate:
```bash
ANTHROPIC_API_KEY=sk-ant-... \
  /home/bryce/mycelium/.venv/bin/python scripts/generate_gsm8k_step_targets.py \
    --num 20 --split train --output .cache/gsm8k_steps_v1_test.jsonl
```

Expected output: ~80-90% acceptance rate. Output JSONL has one record per line:
```json
{"problem": "Sam had 8 5 cookies. ...", "gen_targets": ["Sam had 8 5 cookies and doubled them. 8 5 * 2 = 1 7 0 cookies now.", "Then Sam gave 1 3 2 away. 1 7 0 - 1 3 2 = 3 8 cookies remaining."], "answer": 38, "n_steps": 2}
```

If acceptance rate is below ~70%, the prompt or parser needs tuning. Inspect rejected examples (the script prints first 5).

When happy with quality, scale up to full train set (~7.5k problems):
```bash
ANTHROPIC_API_KEY=sk-ant-... \
  /home/bryce/mycelium/.venv/bin/python scripts/generate_gsm8k_step_targets.py \
    --num 7500 --split train --output .cache/gsm8k_steps_v1_train.jsonl
```

Cost estimate: Haiku 4.5 at ~$0.80/MTok in + $4/MTok out, average problem ~200 tokens in + ~150 tokens out = ~$0.0007/problem → ~$5 for full 7.5k train set.

## Step 2: Loader

```python
from mycelium.l3_data import load_gsm8k_steps

# Flat list (for inspection)
examples = load_gsm8k_steps(".cache/gsm8k_steps_v1_train.jsonl")
print(f"{len(examples)} examples")

# Bucketed by step count (for uniform-K training)
buckets = load_gsm8k_steps(".cache/gsm8k_steps_v1_train.jsonl", bucket_by_k=True)
for k in sorted(buckets):
    print(f"K={k}: {len(buckets[k])} examples")
```

## Step 3: Training (TODO)

Per-breath supervision in `l3_training.py:per_breath_train_step` REQUIRES uniform K across the batch (asserts on line 1175). For GSM8K's variable K (2-8 steps), we need bucketing.

Design: extend the train loop in `scripts/l3_train.py` to:
1. Build buckets at startup if level == `GSM8K_STEPS`
2. Each step: sample a bucket (uniformly across step counts, or proportional to bucket size)
3. Sample a batch from that bucket
4. Train with that batch's K (the JIT cache keys K, so each bucket compiles its own kernel)

Implementation: ~50 lines in `l3_train.py`. Defer until we have the actual generated data and know the bucket distribution.

## Files

- `scripts/generate_gsm8k_step_targets.py` — Haiku generation script
- `mycelium/l3_data.py:load_gsm8k_steps` — JSONL loader
- TODO: `scripts/l3_train.py` GSM8K_STEPS dispatch + bucketing
- TODO: `scripts/v60_gsm8k_kbreath.sh` — training launcher
