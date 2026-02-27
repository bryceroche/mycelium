# Raw Attention Data Inventory

## Summary

We have **JSD step counts** but **NOT full raw attention weights** in S3. Here's what exists:

## Available Data

### 1. JSD Traces (GSM8K 7B model)
**Location:** `s3://mycelium-data/gsm8k_7b_jsd/`

**Structure:**
- 200 problems total (100 in shard_2, 100 in shard_3)
- Per-problem JSON files with:
  - `jsd_trace`: List[float] - JSD between consecutive attention distributions
  - `n_tokens`: int - Number of tokens generated
  - `tokens`: List[str] - Generated tokens
  - Problem text, gold answer, predicted answer, etc.

**JSD Trace Properties:**
- Length = n_tokens - 1 (one JSD value per generation step)
- Values range: ~0.04 to ~0.28
- Mean: ~0.14

**Example file:** `s3://mycelium-data/gsm8k_7b_jsd/shard_2/shard_2/problem_03736.json`

```json
{
  "idx": "...",
  "question": "...",
  "jsd_trace": [0.173, 0.074, 0.180, ...],  // n_tokens - 1 values
  "tokens": ["To", "determine", ...],         // n_tokens values
  "n_tokens": 386
}
```

**What JSD represents:** Jensen-Shannon divergence between attention distribution at step `t` and step `t-1`:
```python
# From train/compare_7b_72b_segmentation.py
def compute_jsd(attn_row_i, attn_row_j):
    p = attn_row_i / (attn_row_i.sum() + 1e-10)
    q = attn_row_j / (attn_row_j.sum() + 1e-10)
    m = 0.5 * (p + q)
    jsd = 0.5 * (KL(p||m) + KL(q||m))
    return jsd
```

### 2. IAF Traces (72B model - MATH dataset)
**Location:** `s3://mycelium-data/iaf_extraction/chunked/`

**Structure:**
- 117 chunk files (~200MB each)
- Per-problem structure:
  - `iaf_traces`: List[Dict[str, float]] - Top-10 head attention weights per step
  - `num_tokens`: int - Total generation steps
  - `top_positions`: Top-k attended positions per step
  - Problem text, solution, generated CoT, etc.

**IAF Trace Properties:**
- Length = num_tokens (one dict per generation step)
- Each step tracks **top 10 attention heads only** (not full attention)
- Format: `{"L22H4": 0.9997, "L22H3": 0.9869, ...}` - 10 heads per step

**Example:** `s3://mycelium-data/iaf_extraction/chunked/instance1_iaf_v3_gpu0_valid_chunk_000.json`

```json
{
  "iaf_traces": [
    {"L22H4": 1.0, "L22H3": 1.0, "L23H23": 1.0, ...},  // Step 0: 10 heads
    {"L22H4": 0.9997, "L22H3": 0.9869, ...},           // Step 1: 10 heads
    ...
  ],
  "num_tokens": 907
}
```

### 3. V6 Attention Metadata (Reference, not uploaded)
**Location:** `s3://mycelium-data/v6/v6_attention/results_partial.json`

**Structure:**
- Metadata file referencing **local** .npz files (not in S3)
- Contains JSD traces + paths to attention files
- Attention files stored locally at: `data/v6_attention/attention_*.npz`

```json
{
  "jsd_trace": [{"step": 1, "jsd": 0.287, ...}, ...],
  "attention_file": "data/v6_attention/attention_00000.npz"  // LOCAL PATH
}
```

## What We DON'T Have

**Full raw attention weights:**
- No full layer×head×seq×seq attention tensors in S3
- IAF only tracks **top-10 heads** per step (not all 28 layers × 32 heads)
- JSD is a **scalar summary** of attention shift (not raw weights)

**To get full attention:**
- Would need to re-run inference with `output_attentions=True`
- Store full tensors: shape `(num_layers, num_heads, seq_len, seq_len)`
- For 72B model with 28 layers, 32 heads, 1000 tokens: ~7GB per problem (float16)

## Why This Matters

**Current C1 training** uses:
1. IAF traces (top-10 head weights) → relevance labels
2. Heuristics to convert top-10 → binary IO tags

**With full attention**, we could:
1. Compute exact JSD per head (not just top-10)
2. Find which heads/layers best predict computation boundaries
3. Train C1 on continuous relevance (not binary tags)

**But:** Full attention storage is expensive. Current approach (IAF top-10 + JSD) is a practical compromise.

## Recommendation

**For now:** Use JSD traces as training signal:
- High JSD = attention shift = likely computation boundary
- Train C1 to predict JSD (regression), not binary tags
- Eliminates heuristic threshold for "what is a segment"

**Future:** If needed, re-run inference on small sample (50-100 problems) with full attention to validate assumptions.
