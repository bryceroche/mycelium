# Task: Qwen-7B Attention Head Analysis via Lambda MapReduce

## Context

We're building the Mycelium pipeline where C1 (Qwen-0.5B) distills structural primitives from Qwen-7B's attention patterns. We need to analyze the attention heads in our IAF data to determine which heads exhibit clean telegraph behavior (alternating between "reading input" and "internal computing" states during decode).

**Important discovery:** The IAF extraction already pre-selected 10 specific heads (out of Qwen-7B's 784 total). The selection criteria are unknown — could have been IAF contrast scoring, manual inspection, or literature references. This analysis will validate whether those 10 are actually good telegraph candidates, or if some are noise.

The heads stored in the data:

```
L5H19   (layer 5)   — early layer outlier, may do global problem-type detection vs step tracking
L14H0   (layer 14)
L22H3, L22H4        (layer 22)
L23H11, L23H23      (layer 23)
L24H4, L24H6, L24H16 (layer 24) — most heads from this layer
L25H1   (layer 25)
```

The IAF data is in S3 (~28GB, 117 chunks). It stores per-head attention to input token positions during decode. It does NOT store attention to generated tokens. We can recover the telegraph signal: for each head at each decode step, sum attention across all input positions. High sum = head is reading input. Low sum = head is attending to its own generated CoT (computing). This scalar time series per head is the "reading signal."

## Existing Code

Look at the existing Lambda MapReduce infrastructure in this repo:
- `lambda_c1_datagen.py` — existing Lambda function for C1 data generation
- `orchestrate_c1_datagen.py` — existing orchestrator that handles chunking, S3 reads, Lambda invocation

Reuse the same chunking/orchestration pattern. The infra works well: 3GB Lambda memory, 200MB chunks, full dataset completes in ~30 seconds.

**IMPORTANT:** Before writing any code, inspect the existing Lambda code and a sample of the raw IAF data in S3 to understand the exact data schema, S3 paths, and file format. Look at what fields are available per problem, per head, per decode step. Do NOT assume the schema — read it from the data. When inspecting data files, only read the first few KB or first record — these files can be very large and will freeze the session if opened fully.

## What to Compute

For each of Qwen-7B's 10 pre-selected attention heads, across ALL problems in the dataset, compute:

### 1. Reading Signal Time Series
For each problem, for each head, for each decode step:
```
reading_signal[head, step] = sum(attention_to_input_tokens[head, step, :])
```
This is the raw telegraph signal.

### 2. Per-Head Metrics (aggregated across all problems)

**Telegraph quality (bimodality):**
- Compute the reading_signal distribution per head across all steps and all problems
- Fit a two-component Gaussian mixture (or simply compute Hartigan's dip statistic) to test for bimodality
- Report: dip statistic or bimodality coefficient, means of the two modes, separation between modes

**Contrast ratio:**
- For each head: mean(reading_signal when above median) / mean(reading_signal when below median)
- Higher = cleaner separation between reading and computing states

**Transition statistics:**
- Mean transition frequency per problem (how often the head switches between above/below median per decode sequence)
- Mean dwell time in each state (consecutive steps in reading vs computing)
- Regularity: variance of dwell times (low variance = more regular telegraph)

**Cross-problem consistency:**
- Per head, compute the coefficient of variation of the contrast ratio across problems
- Low CV = head behaves consistently regardless of problem type
- Also track: what fraction of problems show bimodal behavior for this head?

### 3. Cross-Head Analysis

**Synchronization matrix (10x10):**
- For each pair of heads, compute the fraction of transitions that co-occur within a ±1 step window
- High synchronization between heads = they transition together at boundaries

**Correlation matrix (10x10):**
- Pearson correlation of reading_signal time series between head pairs, averaged across problems

### 4. Layer Distribution
- Group heads by layer and report whether telegraph behavior clusters in early/mid/late layers

## Output

Write results to `s3://mycelium-data/head_analysis/`:

```
head_analysis/
├── per_head_stats.json          # 10-entry array with all per-head metrics
├── sync_matrix.json             # 10x10 synchronization matrix
├── correlation_matrix.json      # 10x10 correlation matrix
├── layer_summary.json           # per-layer aggregated stats
├── global_summary.json          # top-level summary, ranked head list
├── example_waveforms.json       # raw time series for top/bottom heads
└── per_chunk_stats/*.json       # raw per-chunk intermediate stats
```

`global_summary.json` should include:
- All 10 heads ranked by telegraph quality (composite of bimodality + contrast + consistency)
- Recommended top-k heads for C1 with justification
- Clusters of synchronized heads (potential boundary-detection groups)
- Any heads that are clearly noise (unimodal, low contrast, inconsistent)

`example_waveforms.json` should include:
- For the top 3 and bottom 3 heads by telegraph quality
- Raw reading_signal time series for 5 representative problems each
- This allows visual sanity-checking — if the "best" head looks noisy, metrics are wrong

## Implementation Notes

- Reuse the existing orchestration pattern from `orchestrate_c1_datagen.py`
- Lambda function: map phase computes per-head stats for each problem in the chunk, emits per-chunk aggregates
- Orchestrator: reduce phase combines chunk stats into global stats, computes cross-head matrices
- Be careful with memory: process one problem at a time within a chunk, don't hold all IAF tensors simultaneously
- Use numpy in the Lambda layer for the signal processing

## What This Enables

The analysis output tells us:
1. Which of the 10 heads show clean telegraph behavior (for C1 phase detection)
2. Which heads synchronize at boundaries (for knot crossing detection)
3. Which heads are noise and can be ignored
4. Whether L5H19 behaves differently from the layer 22-25 cluster (global vs local signal)
