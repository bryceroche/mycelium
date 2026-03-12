# Mycelium v7

Distill mathematical reasoning from a 7B teacher's attention patterns. A 0.5B canonicalizer transcodes compressed problem text into explicit telegraphic instructions. A learned energy landscape refines rough to precise. SymPy certifies truth. They co-evolve.

---

## Seven Principles

Seven ways of distributing. Each one prevents concentration into a single point of failure.

```
1. EXPAND      distribute the PROBLEM across steps
               1 ambiguous blob → N explicit instructions

2. DECOMPOSE   distribute the WORK across workers  
               each step is independent, forms a DAG

3. SMOOTH      distribute the EFFORT evenly
               each step is ~4 tokens, uniform difficulty

4. VERIFY      distribute the CHECKING per step
               SymPy executes each instruction, oracle certifies

5. LOW → HIGH  distribute the PRECISION between model and ODE
               rough telegrams → ODE refines → precise SymPy

6. COMPRESS    distribute the LOAD away from the model
               minimal tokens, zero syntax noise, lossless meaning

7. EVOLVE      distribute the LEARNING across cycles
               verified solutions train both models, oracle grounds evolution
```

Every design decision passes this test: does it help expand, decompose, smooth, verify, refine, compress, or evolve? If not, we don't need it.

---

## The Core Insight: Transcoding, Not Reasoning

Mathematical problem text is BADLY compressed — ambiguous, implicit, everything jammed together. The canonicalizer TRANSCODES it into WELL compressed telegraphic instructions — explicit, unambiguous, minimal.

```
Bad compression (input):
    "If x² + y² = 90 and xy = 27, what is (x+y)²?"
    One sentence. Two equations, one target, an implicit expansion,
    substitution, and computation all hidden inside.

Good compression (output):
    GIVEN x^2+y^2=90
    GIVEN xy=27
    EXPAND (x+y)^2
    SUBS _prev x^2+y^2 90
    EVAL _prev
    Five telegrams. Everything explicit. Nothing hidden. ~20 tokens total.
```

Structure EXPANDS (1 blob → 5 steps). Content COMPRESSES (each step is ~4 tokens). The diamond shape:

```
         Raw problem (1 sentence, everything hidden)
              │
         EXPAND structure
              │
    ┌────┬────┼────┬────┐
  GIVEN GIVEN EXPAND SUBS EVAL    ← widest: 5 explicit steps
    │    │     │     │    │
    └────┴─────┴─────┴────┘
              │
         COMPRESS content (each step = VERB + ARGUMENTS)
              │
         EXECUTE (SymPy evaluates each line)
              │
         COLLAPSE to answer
```

---

## Critical Rules

```
 1. Every model outputs ROUGH approximations, never precise
 2. ODE + energy landscape refines rough to precise
 3. SymPy oracle verifies — incorruptible, no reward hacking
 4. Fresh LoRA from base EVERY cycle — never continue previous
 5. Combine ALL data each cycle — original + all accumulated traces
 6. Problem-level data splits — never step-level
 7. parse_latex for answer comparison — never string matching
 8. SymPy timeout 5 seconds — every execution call
 9. float32 on A10G — never bfloat16
10. vLLM 0.4.3 — never 0.17.x
11. Deep Learning AMI — never bare Ubuntu
12. No regex in inference path — model generates, SymPy parses
13. Delete landmines — don't quarantine, delete
14. Validate on 10 problems before scaling to 50 or 500
15. Stage-based model loading — 3 loads total, not per-problem
16. Training targets are ROUGH — right neighborhood, not exact syntax
```

---
## THE MYCELIUM BUG LIST
```
 1. Generate, never extract (sixteen twenty-one → 1621)
 2. Perfect boundaries don't exist (but boundary COUNT matters)
 3. Train/inference distribution must match
 4. Train/inference preprocessing must match
 5. Only SymPy-verified targets in training data
 6. Rare classes get ignored (BIO B-tag, SOLVE with 3 examples)
 7. Ordering mismatch between teacher and student
 8. One model, one job (split beats generalist every time)
 9. Check batch size and vectorize before long runs
10. Checkpoint to S3 (ephemeral storage will betray you)
11. Never open large files (head -c 3000)
12. Model learns what you show it, not what you intend
13. LoRA scale = alpha/rank. Scale > 1.0 destroys base model behavior. Use scale ≤ 1.0 for generation tasks.
```

## Large Files Will FREEZE You

Check file size before opening. Never open files over 5MB. Never cat or print large JSON files.

```bash
aws s3 cp s3://mycelium-data-v7/path/to/file.json - | head -c 3000
```

---

## Architecture — The Assembly Line

### The Telegraphic Instruction Language

Every math solution is a sequence of VERB + ARGUMENTS instructions:

```
VERB        MEANING                     ARGUMENTS
─────────────────────────────────────────────────────
GIVEN       state a fact/equation       equation or assignment
EVAL        compute a value             expression
SOLVE       find a variable             equation variable
EXPAND      expand an expression        expression
SIMPLIFY    simplify an expression      expression
SUBS        substitute a value          expression old new
APPLY       use a theorem/formula       theorem_name args
ANSWER      final result                _prev
```

`_prev` references the previous step's result. Each instruction is ~4 tokens. The sequence forms a complete program that SymPy can execute line by line.

### C1-A: The Structural Guide (frozen, F1=0.741)

Tells the canonicalizer HOW MANY instructions to write and WHAT TYPE each one is.

```
Input:  raw problem text
Output: boundary_count (N boundaries → N+1 steps)
        scaffold_types per step (7 classes, from scaffold MLP on hidden states)
```

C1-A provides the TEMPLATE. The canonicalizer fills in the ARGUMENTS.

```
C1-A says:          4 steps: [GIVEN, EXPAND, SUBS, EVAL]
Canonicalizer adds: GIVEN x^2+y^2=90; EXPAND (x+y)^2; SUBS _prev x^2+y^2 90; EVAL _prev
```

### The Canonicalizer (Qwen-0.5B + LoRA — the ONLY text generator)

Transcodes problem text into telegraphic instructions. ONE model, ONE job, ONE output format.

```
Input:
    Problem: [full problem text]
    Structure: N steps [SCAFFOLD_TYPE, SCAFFOLD_TYPE, ...]
    Rewrite as instructions:

Output:
    GIVEN x^2+y^2=90
    GIVEN xy=27
    EXPAND (x+y)^2
    SUBS _prev x^2+y^2 90
    EVAL _prev
```

The output is ROUGH. Carets not double-stars. No Eq() wrappers. No Symbol declarations. Right neighborhood, not precise syntax. The ODE handles precision.

### Energy Landscape (learned MLPs, alternating pair terms)

The mountain topography. Correct instruction sequences sit in energy basins. The ODE walks downhill toward them.

```
Node energy:    MLP(instruction_embedding) → scalar
                "Is this instruction reasonable on its own?"

Pair energy:    (MLP(h_i, h_j) - MLP(h_j, h_i)) / 2 → scalar
                Antisymmetric by construction — ORDER MATTERS
                Swap two instructions → energy changes sign

Total energy:   E = Σ node_energy(h_i) + λ Σ_{i<j} pair_energy(h_i, h_j)
                π-normalized across nodes for scale invariance

Training:       contrastive on correct vs incorrect instruction sequences
                Shuffle correct sequence → high energy (wrong order)
                Correct sequence → low energy (right basin)
```

The alternating structure encodes the mathematical truth that operation order matters: EXPAND before SUBS works, SUBS before EXPAND doesn't.

### ODE Solver (refines rough → precise)

The hiker navigating downhill on the energy landscape.

```
Dynamics:   dh/dt = -∇E(h)  (gradient descent on learned energy)
Solver:     dopri5 (adaptive Runge-Kutta)
Bounds:     tanh * 0.1 (prevents explosion)
π-norm:     at input, state, and energy levels

What it refines:
    Rough telegrams → precise SymPy function calls
    x^2 → x**2
    sin30 → sin(pi/6)
    SUBS _prev x^2+y^2 90 → subs(prev_result, x**2+y**2, 90)

Integration time from C1-A boundary count:
    1-2 steps  → bp_depth=1.0 (simple, fast)
    3-4 steps  → bp_depth=2.0 (medium)
    5+ steps   → bp_depth=3.0 (complex)
```

### SymPy Oracle (incorruptible)

```
Every refined instruction executed with 5-second timeout
parse_latex for LaTeX → SymPy conversion
Answer comparison via sympy.simplify(a - b) == 0
No regex for mathematical content

The oracle CAN'T be fooled — math is right or wrong
Grounds the entire feedback loop
```

### Factor Graph (verification + error localization)

```
After ODE converges:
    Check each instruction: does it execute?
    Check the chain: do results flow via _prev correctly?
    Check the answer: does final EVAL/ANSWER match gold?

Error localization: 97.8% accuracy
Correction convergence: 90.9%
```

### Scaffold Perturbation Recovery (when C1-A is wrong)

The structural guide can be wrong and the building still stands.

```
If ODE can't converge (energy stays high):
    1. Identify highest-energy instruction
    2. Try nearby verbs: EXPAND → SIMPLIFY, SUBS → SOLVE
    3. Re-run ODE with perturbed scaffold
    4. Pick lowest-energy configuration

If STILL can't converge:
    Split one instruction into two (add a step)
    Merge two instructions into one (remove a step)

If NOTHING works:
    Flag "beyond current capability"
    Corrections feed back to C1-A training in Phase 5
```

---

## Inference Pipeline (stage-based)

Three model loads total. Clean integer problem_id keys throughout.

```
Stage 1: Load C1-A (frozen)
         Run ALL problems → boundaries + scaffold types
         Save per problem_id
         Unload C1-A

Stage 2: Switch to canonicalizer LoRA adapter (same Qwen-0.5B base)
         For each problem:
             Input: problem text + structure hint from C1-A
             Output: telegraphic instruction sequence (~4 tokens per step)
         Wave batching across problems
         Unload canonicalizer

Stage 3: Load energy landscape + ODE
         For each problem:
             Refine rough telegrams → precise SymPy
             Execute each instruction
             Verify energy is low
             If high: perturb scaffold → retry
         Unload
```

C1-A and canonicalizer share the same Qwen-0.5B base. Different LoRA adapters, instant switching via PEFT.

---

## The Three-Body System

```
Canonicalizer (creative):     rough telegrams (direction)
Energy Landscape (critic):     order-aware evaluation (structure)
SymPy (oracle):               incorruptible truth (precision)

Creator → Critic → Oracle → verified traces → both retrain → cycle
```

### Self-Improvement Loop (Principle 7: EVOLVE)

```
For each cycle:
    1. Train canonicalizer (fresh LoRA from base on ALL accumulated data)
    2. Train energy landscape (contrastive on ALL accumulated pairs)
    3. Stage-based inference on all problems
    4. SymPy oracle verifies answers
    5. Harvest from correct solutions:
       - Verified (rough telegram → precise SymPy) pairs
       - Correct/incorrect sequence pairs for energy landscape
       - Scaffold corrections for C1-A (Phase 5)
       - Per-instruction error rates for SMOOTH monitoring
    6. Convergence check (patience=2 cycles)
    7. Save everything to S3 with full provenance

Fresh LoRA from base each cycle. Never continue previous.
Combined data: original + ALL accumulated verified traces.
Oracle-grounded: only execution-verified traces enter training.
```

### Error Attribution (maintaining SMOOTH)

```
Per-verb error rates:
    GIVEN:    ??%
    EVAL:     ??%
    SOLVE:    ??%
    EXPAND:   ??%
    SUBS:     ??%
    APPLY:    ??%

Coefficient of variation = std(rates) / mean(rates)
Target: CV < 0.3 (difficulty evenly distributed)

If APPLY at 30% while others at 5%:
    Split: APPLY → APPLY_IDENTIFY + APPLY_EXECUTE
    Keep splitting until balanced
```

### Data Provenance

```
Every harvested trace carries:
    problem_id, step_idx, verb
    rough_telegram (canonicalizer output)
    refined_sympy (ODE output)
    executed_result (SymPy output)
    cycle_harvested, model_version, energy_score
    was_corrected, was_scaffold_perturbed
```

### Versioned Rollback

```
s3://mycelium-data-v7/cycles/
├── cycle_0/
│   ├── canonicalizer_lora/
│   ├── energy_model.pt
│   ├── results.json
│   ├── harvested_traces.jsonl
│   ├── energy_pairs.jsonl
│   └── metrics.json
├── cycle_1/ ...
└── loop_history.json
```

---

## Canonicalizer Training

### Target Format: ROUGH Telegrams

```
Targets are ROUGH — right neighborhood, not exact syntax.
The ODE handles precision. The model handles direction.

DO train on:     SOLVE x^2-9 x           (rough, ~4 tokens)
DON'T train on:  solve(Eq(x**2-9,0), x)  (precise, teaches fragile exactness)

DO train on:     EVAL 1/2*8*10*sin30      (rough, student math)
DON'T train on:  evaluate(Rational(1,2)*8*10*sin(pi/6))  (precise, compiler code)

DO train on:     SUBS _prev x^2+y^2 90   (rough, telegraphic)
DON'T train on:  subs(prev, {x**2+y**2: 90})  (precise, dict syntax)
```

### Training Config

```
Base:       Qwen/Qwen2.5-0.5B (always fresh, never continue previous)
LoRA:       r=16, alpha=32, dropout=0.05, Q/K/V/O
Epochs:     3
Batch:      4, grad_accum 8 (effective 32)
LR:         2e-4, cosine schedule
dtype:      float32 (A10G)
Split:      problem-level (never step-level)
Validation: targets must contain a valid VERB and at least one argument
```

### Building Training Data

Source: Sonnet batch API converts 50K CoT steps into rough telegrams.

```
Sonnet prompt:
    "Convert this math step into a telegraphic instruction.
     Format: VERB argument1 argument2
     Use rough math notation (^ not **, fractions as a/b).
     Output ONE line only. 3-6 tokens maximum."

Validate: each target starts with a valid VERB
          each target is < 10 tokens
          reject any target containing English words (except VERB)
```

Save to s3://mycelium-data-v7/training_data/canonicalizer/

---

## S3 Data Map

### v7 Bucket (all new work)
```
s3://mycelium-data-v7/
├── training_data/
│   └── canonicalizer/        # rough telegram targets
├── models/
│   └── canonicalizer/        # Qwen-0.5B + LoRA
├── cycles/                   # feedback loop versioned rollback
├── evaluation/               # per-verb error attribution
├── feedback_loop/            # harvested traces
└── checkpoints/              # crash recovery
```

### Frozen Models (read-only from old bucket)
```
s3://mycelium-data/models/c1a_coarse_v6_aux_telegraph/  # F1=0.741
s3://mycelium-data/models/c1b_sequence_v5/               # 62%, 1.37
```

### Reference Data (read-only from old bucket)
```
s3://mycelium-data/c2c3_training_data_v2/parsed_steps.jsonl  # 50K Sonnet steps
s3://mycelium-data/ib_results_math/                           # 25 IB clusters
```

### NEVER USE
```
s3://mycelium-data/sympy_training_v1/auto_converted.jsonl     # 45% invalid
s3://mycelium-data/self_improvement_v1/                        # poisoned
```

---

## Infrastructure

### VM Setup
```bash
# ALWAYS Deep Learning AMI
ssh -i ~/.ssh/mycelium-key.pem ubuntu@<IP>

aws ec2 describe-instances \
    --filters "Name=instance-state-name,Values=running" \
    --query "Reservations[].Instances[].{ID:InstanceId,Type:InstanceType,IP:PublicIpAddress}"
```

### Systemd (crash-resilient)
```bash
sudo systemctl start mycelium-loop   # main loop
sudo systemctl start mycelium-sync   # S3 backup every 5 min
sudo journalctl -u mycelium-loop -f  # monitor
```

### vLLM
```bash
pip install vllm==0.4.3 --break-system-packages
# Merge LoRA before serving: base + LoRA → merged → vLLM
```

### Lambda MapReduce for S3 Processing
```
Memory: 3GB NOT 1GB
Chunks: 200MB
Do not copy large files to EC2
```

---

## Codebase

```
scripts/
├── eval.py                           # DEFAULT: batched stage inference eval
├── staged_inference.py               # Batched stage pipeline (Stage 1-2-3)
├── stage1_c1a_inference.py           # Stage 1: C1-A batch segmentation
├── train_canonicalizer.py            # Qwen-0.5B + LoRA training
├── build_canonicalizer_data.py       # Sonnet → rough telegram targets
├── test_pipeline_e2e.py              # End-to-end pipeline test
└── deprecated/                       # Phased out: two-model split (LoRA C/D)

plan/
├── oracle.py                         # SymPy execution + timeout + parse_latex
├── ode_solver.py                     # dopri5 + tanh*0.1 + π-norm
└── build_notation_data.py            # Notation data builder

src/
├── templates.py                      # EXPAND, SIMPLIFY, SOLVE, COMPUTE, ANSWER
├── oracle.py                         # SymPy execution + timeout + parse_latex
├── energy_landscape.py               # node + antisymmetric pair MLPs
└── ode_solver.py                     # dopri5 + tanh*0.1 + π-norm
```

### Default Evaluation

```bash
# Run 50-problem eval with batched stage inference
python scripts/eval.py --problems data/math_50_test.jsonl --n 50

# Custom canonicalizer model
python scripts/eval.py --canonicalizer /path/to/model --n 20
```

---

## MVP Build Order

```
Day 1:  Build canonicalizer training data (Sonnet → rough telegrams)
        Train canonicalizer. Test on 10 problems.

Day 2:  Wire stage-based pipeline: C1-A → canonicalizer → ODE → SymPy
        Test on 10 problems: do telegrams execute?

Day 3:  Add ODE with node energy. Does refinement help?

Day 4:  Add alternating pair energy. 50-problem eval.

Day 5:  Scaffold perturbation + Parts Lookup. 50-problem eval.

Day 6:  Feedback loop cycle 0. Harvest traces.

Day 7:  Cycles 1-3. Convergence curve.

Day 8-12: Polish from error attribution. 200-problem stress test.

Day 13: March 22, 2026 — MATH500 benchmark.
```

---

## Future Directions

- C1-A joins feedback loop (scaffold corrections as training data)
- C1-B for adaptive ODE integration time
- Laplace spectral fingerprint (telegraph poles/residues conditioning ODE)
- Teacher trajectory distillation (full path supervision from IAF data)
- MCTS for solution path discovery (search over SymPy transformations)
- Scale to Qwen-1.8B if 0.5B ceiling is too low
- CoT distillation baseline for paper comparison
