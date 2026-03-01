# Mycelium: Open Source Release Plan

## Philosophy

Someone should be able to clone the repo, read 00 through 06 in order, and understand the entire pipeline. No hidden utilities, no deep folder hierarchies, no "see utils/attention/extractors/base.py". Each script is self-contained with clear input→output. The paper's impact doubles if someone can reproduce it in a weekend.

---

## Repository Structure

```
mycelium/
│
├── README.md                          # The λανθάνω epigraph + quickstart
│
├── 00_extract_attention.py            # Run Qwen-7B with output_attentions=True
│                                      # Input:  MATH dataset (auto-downloads)
│                                      # Output: data/iaf_traces.json
│                                      # Time:   ~2hrs on 1x A10G
│                                      # Notes:  S3 daemon pattern for memory management
│                                      #         Streams chunks to disk, never holds >500MB
│
├── 01_detect_heartbeats.py            # Threshold IAF traces → count computation pulses
│                                      # Input:  data/iaf_traces.json
│                                      # Output: data/heartbeats.json
│                                      # Time:   ~5 min on CPU
│                                      # Notes:  threshold=0.5, min_run_length=5
│                                      #         Expected: Level 1→3.8, Level 5→14.8
│
├── 02_discover_templates.py           # Information Bottleneck with Y labels → templates
│                                      # Input:  data/iaf_traces.json (CoT steps)
│                                      # Output: data/templates.json (~30 operation types)
│                                      # Time:   ~30 min on 1x GPU (embedding) + 10 min CPU (IB)
│                                      # Notes:  TRUE IB with min I(X;T) - β·I(T;Y)
│                                      #         Y = sympy AST operator (with Pow splitting)
│                                      #         Elbow detection at gain threshold 0.15
│                                      #         Without Y this is just k-means. Y is critical.
│
├── 03_build_training_data.py          # Assemble C2 multi-label + C3 pointer training pairs
│                                      # Input:  data/iaf_traces.json, data/templates.json,
│                                      #         data/heartbeats.json
│                                      # Output: data/c2_train.json (5,591 multi-label examples)
│                                      #         data/c3_train.jsonl (26K+ pointer examples)
│                                      # Time:   ~10 min on CPU
│                                      # Notes:  C2 format: problem_text → {label_set} + heartbeat_count
│                                      #         C3 format: [TEMPLATE] text + priors → provenance labels
│                                      #         Hard filter: only sympy.sympify() parseable expressions
│
├── 04_train_c2.py                     # Multi-label operation classifier + heartbeat auxiliary
│                                      # Input:  data/c2_train.json
│                                      # Output: models/c2/
│                                      # Time:   ~30 min on 1x GPU
│                                      # Notes:  MiniLM-22M backbone
│                                      #         Head 1: sigmoid over ~15 operation types (BCE)
│                                      #         Head 2: heartbeat count regression (MSE)
│                                      #         Joint loss: BCE + 0.1 * MSE
│                                      #         Freeze backbone 2 epochs, then unfreeze
│                                      #         Threshold 0.3 at inference (recall > precision)
│                                      #         Target: >99% any-correct, 100% all-correct @ 0.3
│
├── 05_train_c3.py                     # Operand pointer model
│                                      # Input:  data/c3_train.jsonl
│                                      # Output: models/c3/
│                                      # Time:   ~1hr on 1x GPU (or ~15min on 8x with DDP)
│                                      # Notes:  Qwen-0.5B backbone
│                                      #         Pointer head: for each operand slot, predict
│                                      #           source type (TEXT/PRIOR/IMPLICIT/CONSTANT)
│                                      #           + location (token position or step number)
│                                      #         C3 can ONLY select from what exists — no hallucination
│                                      #         Include EOS token in training data (critical!)
│
├── 06_eval_e2e.py                     # Full pipeline evaluation
│                                      # Input:  models/c2/, models/c3/, MATH test set
│                                      # Output: results/eval_report.json
│                                      # Time:   ~5 min on 1x GPU
│                                      # Notes:  C2 (threshold 0.3) → C3 (pointer) → C4 (assemble)
│                                      #         → sympy → MCTS with beam k=10
│                                      #         Greedy execution order + permutation fallback
│                                      #         Majority voting on valid sympy results
│                                      #         Reports: accuracy, correct-in-beam, per-level breakdown
│
├── c4_assembler.py                    # Deterministic expression builder (~50 lines)
│                                      # Not a model. Lookup table: template + operands → expression
│                                      # ADD + [a, b] → "a + b"
│                                      # DIV + [a, b] → "a / b"
│                                      # SQRT + [a]   → "sqrt(a)"
│                                      # Includes WORD_TO_NUMBER: "half"→2, "double"→2, etc.
│                                      # Includes DOMAIN_CONSTANTS: 60 min/hr, 365 days/yr, etc.
│
├── config.yaml                        # Every hyperparameter in one place (see below)
├── requirements.txt                   # Pinned versions (see below)
│
├── data/                              # Created by scripts or downloaded from release
│   ├── .gitkeep
│   └── sample/                        # Checked into repo — 10 problems for smoke tests
│       ├── sample_iaf_10.json
│       └── sample_math_50.json
│
├── models/                            # Created by training scripts
│   └── .gitkeep
│
├── results/                           # Created by eval script
│   └── .gitkeep
│
└── paper/                             # Paper assets
    ├── figures/
    │   ├── heartbeat_trace.png        # The IAF oscillation visualization
    │   ├── ib_elbow.png               # β-annealing gain vs cluster count
    │   ├── heartbeat_by_level.png     # Heartbeat count vs MATH difficulty
    │   └── pipeline_diagram.png       # C2→C3→C4→sympy flow
    └── mycelium_paper.pdf
```

---

## config.yaml

```yaml
# =============================================================================
# Mycelium Configuration — Every magic number in one place
# =============================================================================

# --- Teacher Model ---
teacher:
  model: Qwen/Qwen2-7B
  max_new_tokens: 1024
  temperature: 0.0                     # Deterministic CoT generation
  output_attentions: true

# --- Attention Extraction ---
extraction:
  computing_heads: [L22H4]             # Identified via head analysis
  chunk_size: 50                       # Problems per output chunk
  max_memory_mb: 500                   # Streaming daemon memory limit

# --- Heartbeat Detection ---
heartbeat:
  iaf_threshold: 0.5                   # Above = reading, below = computing
  min_run_length: 5                    # Minimum tokens for a real computation pulse
  # Expected: Level 1→3.8, Level 2→5.1, Level 3→7.0, Level 4→9.7, Level 5→14.8

# --- Information Bottleneck ---
ib:
  embedding_model: Qwen/Qwen2-0.5B
  embedding_dim: 896
  beta_start: 0.1
  beta_end: 50.0
  min_size: 50                         # Surface tension — reabsorb tiny clusters
  coherence_threshold: 0.72
  elbow_gain_threshold: 0.15           # → ~30 clusters
  max_depth: 3
  # Y extraction: sympy AST with Pow splitting
  pow_subtypes: [sqrt, square, cube, inverse, high_pow, neg_pow, frac_pow, nth_root, pow_general]

# --- C2: Operation Classifier ---
c2:
  backbone: microsoft/MiniLM-L12-H384-uncased
  num_labels: 15                       # From IB template discovery
  threshold: 0.3                       # Low threshold = high recall, MCTS handles false positives
  freeze_epochs: 2                     # Freeze backbone, then unfreeze
  learning_rate: 2.0e-5
  batch_size: 32
  max_epochs: 15
  early_stopping_patience: 3
  heartbeat_loss_weight: 0.1           # Joint loss: BCE + 0.1 * MSE(heartbeat)
  labels:
    - ADD
    - SUB
    - MUL
    - DIV
    - SQRT
    - SQUARE
    - CUBE
    - HIGH_POW
    - SIN
    - COS
    - TAN
    - LOG
    - FACTORIAL
    - MOD
    - OTHER

# --- C3: Operand Pointer ---
c3:
  backbone: Qwen/Qwen2-0.5B
  learning_rate: 2.0e-5
  batch_size: 16
  max_epochs: 15
  early_stopping_patience: 3
  provenance_types:
    - TEXT                             # Operand from problem text at token position
    - PRIOR                            # Result from a prior computation step
    - IMPLICIT                         # Implied value ("half"→2, "double"→2)
    - CONSTANT                         # Domain constant (60 min/hr, etc.)

# --- C4: Expression Assembler ---
c4:
  # Not a model. Deterministic lookup table.
  word_to_number:
    half: 2
    double: 2
    twice: 2
    triple: 3
    quarter: 4
    third: 3
    percent: 100
  domain_constants:
    minutes_per_hour: 60
    hours_per_day: 24
    days_per_week: 7
    weeks_per_year: 52
    months_per_year: 12
    days_per_year: 365
    percentage: 100

# --- MCTS / Evaluation ---
eval:
  beam_width: 10                       # C3 candidates per operation
  mcts_max_paths: 100                  # Maximum paths to explore
  voting: weighted                     # weighted > majority > top-1
  numerical_consistency_boost: 0.2     # Boost candidates using problem numbers
  simplicity_penalty: 0.1              # Prefer simpler expressions
  n_problems: 50                       # Default eval size

# --- Infrastructure ---
infrastructure:
  ddp_gpus: 8                          # For g5.48xlarge training
  lambda_memory_mb: 3072               # For MapReduce jobs
  lambda_timeout_sec: 300
```

---

## requirements.txt

```
torch>=2.1.0
transformers>=4.36.0
sympy>=1.12
numpy>=1.24.0
scipy>=1.11.0
scikit-learn>=1.3.0
datasets>=2.14.0                       # HuggingFace datasets for MATH
pyyaml>=6.0
tqdm>=4.65.0
```

---

## README.md

```markdown
# Mycelium

**λ — λανθάνω (lanthánō) — to escape notice; to be unseen**

> JSD reveals latent boundaries in the flow of attention
> IAF separates reading from reasoning — the dual phases hidden in every forward pass
> The heartbeat of an attention head pulses between thought and ground —
>     each silence a computation the model never knew it performed
> IB discovers the taxonomy of operations — spectral lines emerging from continuous signal
> The prism decomposes. The specialists distill. Sympy collapses the wave function.
>
> Mycelium — the unseen network of computation, made visible

---

Mycelium distills mathematical reasoning from transformer attention patterns into
lightweight specialist models. Instead of autoregressive chain-of-thought generation,
Mycelium decomposes problems into typed operations, extracts operands, and executes
symbolic computation — replacing a 7B model's reasoning with a 22M classifier,
a 0.5B pointer model, and a deterministic symbolic engine.

## Quickstart — Evaluate Only (5 minutes)

```bash
pip install -r requirements.txt

# Download pre-trained models and sample data
python 06_eval_e2e.py --download-pretrained

# Run evaluation on 50 MATH problems
python 06_eval_e2e.py --use-pretrained
```

## Reproduce From Scratch (4-6 hours, 1x A10G GPU)

```bash
pip install -r requirements.txt

# Step 0: Extract attention patterns from teacher model (~2hrs)
python 00_extract_attention.py

# Step 1: Detect heartbeat signal in attention traces (~5min)
python 01_detect_heartbeats.py

# Step 2: Discover operation templates via Information Bottleneck (~40min)
python 02_discover_templates.py

# Step 3: Build training data for C2 and C3 (~10min)
python 03_build_training_data.py

# Step 4: Train C2 operation classifier (~30min)
python 04_train_c2.py

# Step 5: Train C3 operand pointer (~1hr)
python 05_train_c3.py

# Step 6: Evaluate end-to-end pipeline (~5min)
python 06_eval_e2e.py
```

Each script reads from `data/` and writes to `data/` or `models/`.
All hyperparameters live in `config.yaml`.

## Smoke Test (2 minutes, CPU only)

```bash
# Run on 10 sample problems with tiny models
python 06_eval_e2e.py --smoke-test
```

## Results

| Metric | Value |
|--------|-------|
| MATH accuracy (50 problems) | 18.0% |
| Baseline (17.2%) | ✓ Exceeded |
| Correct-in-beam (k=10) | 44.0% |
| Valid sympy rate | 98.0% |
| Total parameters | ~522M |
| Inference time (easy) | ~15ms |
| Inference time (hard) | ~150ms |

## Pipeline

```
Problem text
    → C2: What operations?     {DIV, ADD}           MiniLM-22M
    → C3: Which operands?      [TEXT_48, PRIOR_1]   Qwen-0.5B
    → C4: Build expression     "48 / 2"             Lookup table
    → Sympy: Evaluate          24
    → (loop until all operations executed)
    → Final answer
```

The DAG builds itself: every time C3 outputs PRIOR_N, that's a directed edge.

## Key Discoveries

1. **The Heartbeat**: Attention heads pulse between reading (high IAF) and
   computing (low IAF). Pulse count encodes problem complexity.

2. **Measurement Order Matters**: Measuring operation TYPE before token POSITION
   yields clean results. The reverse (three failed C1 attempts) yields overlapping
   blobs. Wave function collapse: measure the highest-information observable first.

3. **IB Needs Y**: Information Bottleneck without a target variable is just
   k-means. Five failed clustering runs confirmed this. The target (sympy AST
   operator) is what forces clusters to align with computational semantics.

## Configuration

All hyperparameters in `config.yaml`. No magic numbers in code.

## Citation

[paper citation here]

## License

MIT
```

---

## GitHub Release Assets

Three tiers so people can engage at different depths:

### Tier 1: `mycelium-quickstart.tar.gz` (~50MB)
For: "I want to see it work in 5 minutes"
```
models/c2/                    # Pre-trained C2 classifier
models/c3/                    # Pre-trained C3 pointer
data/sample_math_50.json      # 50 evaluation problems
```
Just download, run `06_eval_e2e.py --use-pretrained`, see the 18% result.

### Tier 2: `mycelium-training-data.tar.gz` (~2GB)
For: "I want to train the models myself"
```
data/c2_train.json            # 5,591 multi-label examples with heartbeats
data/c3_train.jsonl           # 26K+ pointer training pairs
data/templates.json           # 30 IB-discovered templates
data/heartbeats.json          # Heartbeat counts per problem
```
Run scripts 03-06. Skip attention extraction, use our pre-computed features.

### Tier 3: `mycelium-iaf-full.tar.gz` (~16GB)
For: "I want to reproduce everything from raw attention patterns"
```
data/iaf_traces/              # Full IAF traces for 7K+ MATH problems
data/generated_cot/           # Teacher CoT solutions
```
Run scripts 00-06. Full reproduction from scratch.

---

## Design Principles for Code

1. **One file per step.** No `utils/` folder. No `common/` package. If two scripts share code, inline it or put it in a clearly named shared file.

2. **Top-of-file docstring explains everything.** What does this script do? What does it read? What does it write? How long does it take? What should I see when it runs?

3. **Progress bars on everything.** Every script uses tqdm. Nobody should stare at a silent terminal wondering if it's working.

4. **Fail fast with clear errors.** Check inputs exist before processing. Print what went wrong, not a stack trace.

5. **Print key metrics as it runs.** Each training script prints epoch, loss, accuracy per epoch. The eval script prints per-problem results.

6. **config.yaml is the single source of truth.** Scripts read from config. No argparse defaults that differ from config values.

7. **Sample data in repo for smoke tests.** 10 problems with IAF traces, enough to verify every script runs without downloading 16GB.

8. **Pin versions in requirements.txt.** Reproducibility means the same library versions.

---

## Pre-Release Checklist

- [ ] All scripts run end-to-end on a fresh environment
- [ ] Smoke test passes on CPU in <2 minutes
- [ ] Config.yaml matches all values used in paper
- [ ] README quickstart works for someone who has never seen the project
- [ ] Release assets download and work with `--use-pretrained`
- [ ] No hardcoded S3 paths (all local `data/` references)
- [ ] No API keys or AWS credentials in code
- [ ] License file present
- [ ] Paper figures generated by scripts (reproducible plots)
- [ ] Every magic number traceable to config.yaml
