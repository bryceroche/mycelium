# Mycelium v7

Distill mathematical reasoning from a 7B teacher's attention patterns. An assembly line of 0.5B specialists transcodes compressed problem text into explicit telegraphic instructions through gradual descent — each worker crosses one representational boundary, each step is small, difficulty is distributed evenly. A learned energy landscape refines rough to precise. SymPy certifies truth. The pipeline breathes — expand, collapse, verify, correct, repeat. They co-evolve.

---

## Eight Principles

Eight ways of distributing. Each one prevents concentration into a single point of failure.

```
1. EXPAND      distribute the PROBLEM across steps
               1 ambiguous blob → N explicit instructions

2. DECOMPOSE   distribute the WORK across workers
               each worker crosses ONE representational boundary

3. SMOOTH      distribute the EFFORT evenly
               no worker is dramatically harder than any other
               coefficient of variation across workers < 0.3

4. BREATHE     distribute the RECOVERY across cycles
               expand → collapse → verify → expand again if wrong
               C1-B's bp_depth sets the breath budget

5. VERIFY      distribute the CHECKING per step
               SymPy executes each instruction, oracle certifies

6. LOW → HIGH  distribute the PRECISION between model and ODE
               rough telegrams → ODE refines → precise SymPy

7. COMPRESS    distribute the LOAD away from the model
               minimal tokens, zero syntax noise, lossless meaning

8. EVOLVE      distribute the LEARNING across cycles
               verified solutions train all components, oracle grounds evolution
```

Every design decision passes this test: does it help expand, decompose, smooth, breathe, verify, refine, compress, or evolve? If not, we don't need it.

When a worker struggles, SPLIT it (DECOMPOSE + SMOOTH). When the pipeline fails on a step, BREATHE (expand back, localize error, try correction, collapse again). When accuracy plateaus, EVOLVE (harvest verified traces, retrain from fresh LoRA).

---

## The Core Insight: Gradual Descent, Not Reasoning

Mathematical problem text is BADLY compressed — ambiguous, implicit, everything jammed together. The assembly line TRANSCODES it into WELL compressed telegraphic instructions through five representational layers. Each layer boundary is crossed by exactly one specialist. No large leaps.

```
Layer 0:  RAW PROBLEM TEXT
          "If x² + y² = 90 and xy = 27, what is (x+y)²?"
              │
          ── C1-A: compressed natural language → explicit structure ──
              │
Layer 1:  SCAFFOLD
          5 steps: [GIVEN, GIVEN, EXPAND, SUBS, EVAL]
              │
          ── Narrator (LoRA C): structure → natural language description ──
              │
Layer 2:  NARRATION
          "State the equation x² + y² = 90"
          "Substitute x² + y² = 90 into the expansion"
              │
          ── Slot Tagger (LoRA E) + Resolver (LoRA F): description → bound values ──
              │
Layer 2.5: BOUND PARAMETERS
          SLOT_1 = 90 (from step_1), SLOT_2 = expansion (from step_3)
              │
          ── Translator (LoRA D): narration + values → rough expression ──
              │
Layer 3:  ROUGH EXPRESSION
          SUBS _prev x^2+y^2 90
              │
          ── Energy landscape + ODE: rough → precise ──
              │
Layer 4:  SYMPY EXECUTION
          subs(prev_result, x**2+y**2, 90) → 144 ✓
```

The diamond shape: structure EXPANDS (1 blob → 5 steps). Content COMPRESSES (each step → ~4 tokens). Both happen simultaneously across the worker chain.

---

## The Breathing Model

The pipeline breathes. Each step inhales (expands compressed reference into explicit meaning) then exhales (collapses into verified result). C1-B's bp_depth prediction sets the breath budget.

```
Breath cycle per step:
    INHALE:  Narrator expands scaffold into description
             Slot Tagger identifies parameter references
             Resolver binds references to concrete values
    EXHALE:  Translator formats expression
             ODE refines rough → precise
             SymPy executes and verifies

    PASS → result enters state table for next step's breathing
    FAIL → factor graph localizes error → next breath corrects
```

Breath 1 might get it right. Breath 2 corrects what breath 1 missed. The whole solution emerges from accumulated verified breaths, not from a single large generation.

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
15. Stage-based model loading — minimize loads, not per-problem
16. Training targets are ROUGH — right neighborhood, not exact syntax
17. Each normalizer matches its model's training distribution (see C1-A preprocessor below)
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
13. LoRA scale = alpha/rank. Scale > 1.0 destroys base model behavior
14. Compare SymPy execution results to gold, never raw model output
15. Never inject data into prompts at inference that wasn't present during training
16. Eval pipeline must use the exact same prompt formatting function as training
17. Each component's preprocessor must match what it was trained with
    (C1-A uses whitespace-only because antlr4 wasn't installed during training)
18. Normalize BOTH input and target in training data
    (Slot Tagger v1 failed: narrations had LaTeX, targets were normalized)
```

---

## C1-A Preprocessor (FROZEN — DO NOT MODIFY)

```
C1-A was trained WITHOUT antlr4. Its preprocessor is whitespace-only.
Do not install antlr4 on C1-A's inference path.
Downstream components use their own normalizers.
```

C1-A's `preprocess_latex` falls back to `' '.join(text.split())` because antlr4 wasn't installed during training. This is now a **fact about C1-A's world**, not a bug to fix. The model learned on whitespace-cleaned text. Changing this breaks train/inference distribution.

```python
# C1-A's preprocessor — DO NOT MODIFY
def preprocess_latex_c1a(text):
    """Whitespace cleaning only. Matches C1-A training distribution."""
    return ' '.join(text.split())

# Downstream components (Slot Tagger, Translator) — SEPARATE normalizers
def preprocess_latex_slot_tagger(text):
    """Full LaTeX normalization for slot tagger and downstream."""
    # \frac{A}{x-5} → A/(x-5)
    # \sqrt{x} → sqrt(x)
    # \left( \right) → ( )
    # Applied to BOTH narration input AND operand targets during extraction
    ...
```

Two functions, two code paths, clearly named. Never swap them.

---

## Large Files Will FREEZE You

Check file size before opening. Never open files over 5MB. Never cat or print large JSON files.

```bash
aws s3 cp s3://mycelium-data-v7/path/to/file.json - | head -c 3000
```

---

## Architecture — The Assembly Line

### The Gradual Descent Workers

```
Worker          Model              Job (ONE boundary)              Current Status
──────────────────────────────────────────────────────────────────────────────────
C1-A            Qwen-0.5B+LoRA     text → structure scaffold       frozen, F1=0.741
C1-B            Qwen-0.5B+LoRA     → bp_depth + co-transitions     frozen, BP 62%
Narrator        Qwen-0.5B+LoRA C   scaffold → step description     trained
Slot Tagger     Qwen-0.5B+LoRA E   narration → tagged slots        retraining (norm fix)
Resolver        Qwen-0.5B+LoRA F   slot desc → step_id binding     90% accuracy
Translator      Qwen-0.5B+LoRA D   narration+values → expression   100% parseable
Energy+ODE      learned MLPs        rough → precise                 trained
SymPy           symbolic engine     execute + verify                deterministic
Factor Graph    energy-based        error localize + correct        97.8% / 90.9%
```

All learned components share the same Qwen-0.5B base. Different LoRA adapters, instant switching via PEFT (~3-4MB per adapter).

### The Binding Split (Slot Tagger → Resolver → Translator)

The old monolithic Translator was doing three jobs at once (Bug #8). Now:

```
=== SLOT TAGGER (LoRA E) ===
Input:
    Narration: Subtract the area of the circle from the area of the square
    Extract slots:
Output:
    SLOT_1 REF the area of the circle
    SLOT_2 REF the area of the square
    -- or for literals --
    SLOT_1 REF the radius
    SLOT_2 LITERAL 2

=== RESOLVER (LoRA F) ===
Input:
    Slots:
      SLOT_1: the area of the circle
      SLOT_2: the area of the square
    State:
      step_1: Find the side length of the square → 10
      step_2: Compute the area of the square → 100
      step_3: Find the area of the circle with radius 5 → 25*pi
    Resolve:
Output:
    SLOT_1 step_3
    SLOT_2 step_2

=== TRANSLATOR (LoRA D) ===
Input:
    Narration: Subtract the area of the circle from the area of the square
    Values:
      SLOT_1 = 25*pi
      SLOT_2 = 100
    Expression:
Output:
    100 - 25*pi
```

Key design decisions:
- Line-based format, not JSON (Qwen-0.5B can't reliably generate balanced braces)
- Resolver outputs step_ids, not values (pipeline code dereferences)
- LITERALs bypass the Resolver entirely
- Slot Tagger expands anaphora ("itself" → "the radius") so Resolver never sees coreference
- State table carries narrator descriptions from every prior step (language-to-language matching)
- GIVEN/SETUP steps surface problem values into state table — no special "problem text" section
- Stop sequences at inference: ["\n\n", "Human:"] for all models

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

Tells the assembly line HOW MANY instructions to write and WHAT TYPE each one is.

```
Input:  raw problem text (whitespace-cleaned only — no antlr4)
Output: boundary_count (N boundaries → N+1 steps)
        scaffold_types per step (7 classes, from scaffold MLP on hidden states)
        cached hidden states (896-dim) shared with downstream
```

C1-A provides the TEMPLATE. The assembly line fills in the CONTENT.

### C1-B: The Breath Budget (frozen, BP 62%, MAE 1.37)

Predicts belief propagation depth and co-transition statistics. Sets how many expand-collapse cycles the factor graph allows before giving up.

```
1-2 steps  → bp_depth=1.0 (simple, fast)
3-4 steps  → bp_depth=2.0 (medium)
5+ steps   → bp_depth=3.0 (complex)
```

### Energy Landscape (learned MLPs, alternating pair terms)

```
Node energy:    MLP(instruction_embedding) → scalar
                "Is this instruction reasonable on its own?"

Pair energy:    (MLP(h_i, h_j) - MLP(h_j, h_i)) / 2 → scalar
                Antisymmetric by construction — ORDER MATTERS
                Swap two instructions → energy changes sign

Total energy:   E = Σ node_energy(h_i) + λ Σ_{i<j} pair_energy(h_i, h_j)
                π-normalized across nodes for scale invariance

Training:       contrastive on correct vs incorrect instruction sequences
```

### ODE Solver (refines rough → precise)

```
Dynamics:   dh/dt = -∇E(h)  (gradient descent on learned energy)
Solver:     dopri5 (adaptive Runge-Kutta)
Bounds:     tanh * 0.1 (prevents explosion)
π-norm:     at input, state, and energy levels

What it refines:
    x^2 → x**2
    sin30 → sin(pi/6)
    SUBS _prev x^2+y^2 90 → subs(prev_result, x**2+y**2, 90)
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

### Factor Graph (verification + error localization + breathing)

```
After ODE converges:
    Check each instruction: does it execute?
    Check the chain: do results flow via _prev correctly?
    Check the answer: does final EVAL/ANSWER match gold?

Error localization: 97.8% accuracy
Correction convergence: 90.9%

Controls breathing: bp_depth from C1-B sets max correction cycles
```

### Scaffold Perturbation Recovery (when C1-A is wrong)

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

Minimize model loads. Clean integer problem_id keys throughout.

```
Stage 1: Load C1-A (frozen, whitespace-only preprocessor)
         Run ALL problems → boundaries + scaffold types + cached hidden states
         Save per problem_id
         Unload C1-A

Stage 2: Load assembly line LoRAs sequentially (same Qwen-0.5B base)
         For each problem, for each step:
             Narrator (LoRA C):     scaffold → step description
             Slot Tagger (LoRA E):  description → tagged slots
             Resolver (LoRA F):     slots + state table → bindings
             Translator (LoRA D):   description + values → rough expression
         LoRA hot-swap between workers (~3-4MB, instant via PEFT)
         Wave batching across problems where possible

Stage 3: Load energy landscape + ODE
         For each problem:
             Refine rough telegrams → precise SymPy
             Execute each instruction
             Verify energy is low
             If high: perturb scaffold → retry (breathing)
         Unload
```

---

## The Three-Body System

```
Assembly Line (creative):      rough telegrams through gradual descent
Energy Landscape (critic):     order-aware evaluation
SymPy (oracle):               incorruptible truth

Creator → Critic → Oracle → verified traces → all retrain → cycle
```

### Self-Improvement Loop (Principle 8: EVOLVE)

```
For each cycle:
    1. Train all LoRAs (fresh from base on ALL accumulated data)
    2. Train energy landscape (contrastive on ALL accumulated pairs)
    3. Stage-based inference on all problems
    4. SymPy oracle verifies answers
    5. Harvest from correct solutions:
       - Verified trace pairs for each worker
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
Per-worker accuracy:
    C1-A:         F1=0.741 (frozen)
    Narrator:     ??%
    Slot Tagger:  fixing (LaTeX normalization)
    Resolver:     90%
    Translator:   100% parseable
    ODE+Energy:   ??%

Per-verb error rates:
    GIVEN:    ??%
    EVAL:     ??%
    SOLVE:    ??%
    EXPAND:   ??%
    SUBS:     ??%
    APPLY:    ??%

Coefficient of variation = std(rates) / mean(rates)
Target: CV < 0.3 (difficulty evenly distributed)

If any worker or verb is dramatically harder:
    Split it. Keep splitting until balanced.
    (Example: monolithic Translator → Slot Tagger + Resolver + Translator)
```

### Data Provenance

```
Every harvested trace carries:
    problem_id, step_idx, verb
    narrator_text (LoRA C output)
    slot_tags (LoRA E output)
    resolved_bindings (LoRA F output)
    rough_expression (LoRA D output)
    refined_sympy (ODE output)
    executed_result (SymPy output)
    cycle_harvested, model_version, energy_score
    was_corrected, was_scaffold_perturbed, breath_number
```

### Versioned Rollback

```
s3://mycelium-data-v7/cycles/
├── cycle_0/
│   ├── narrator_lora/
│   ├── slot_tagger_lora/
│   ├── resolver_lora/
│   ├── translator_lora/
│   ├── energy_model.pt
│   ├── results.json
│   ├── harvested_traces.jsonl
│   ├── energy_pairs.jsonl
│   └── metrics.json
├── cycle_1/ ...
└── loop_history.json
```

---

## Training Data Formats

### Slot Tagger Training

```
Input:
    Narration: Subtract the area of the circle from the area of the square
    Extract slots:
Target:
    SLOT_1 REF the area of the circle
    SLOT_2 REF the area of the square

Extraction: from existing ~10K CoT parses
    - Identify operand references in narrator text
    - Expand anaphora ("itself" → repeated description)
    - Normalize BOTH narration AND target with slot_tagger normalizer
    - Mark literal constants as LITERAL, references as REF
```

### Resolver Training

```
Input:
    Slots:
      SLOT_1: the area of the circle
    State:
      step_1: Find the side length → 10
      step_2: Compute the area of the square → 100
      step_3: Find the area of the circle with radius 5 → 25*pi
    Resolve:
Target:
    SLOT_1 step_3

Extraction: from same CoT parses
    - State table built from prior steps' narrator outputs + values
    - REF slots only (LITERALs bypass Resolver)
    - Output is step_id, not value
```

### Translator Training

```
Input:
    Narration: Subtract the area of the circle from the area of the square
    Values:
      SLOT_1 = 25*pi
      SLOT_2 = 100
    Expression:
Target:
    100 - 25*pi

Stop sequences: ["\n\n", "Human:", "="] (unless Eq() expressions expected)
```

### Canonicalizer Training (legacy, may be superseded by assembly line)

```
Base:       Qwen/Qwen2.5-0.5B (always fresh, never continue previous)
LoRA:       r=16, alpha=32, dropout=0.05, Q/K/V/O
Epochs:     3
Batch:      4, grad_accum 8 (effective 32)
LR:         2e-4, cosine schedule
dtype:      float32 (A10G)
Split:      problem-level (never step-level)

Targets are ROUGH — right neighborhood, not exact syntax.
DO train on:     SOLVE x^2-9 x
DON'T train on:  solve(Eq(x**2-9,0), x)
```

---

## S3 Data Map

### v7 Bucket (all new work)
```
s3://mycelium-data-v7/
├── training_data/
│   ├── canonicalizer/            # rough telegram targets
│   ├── slot_tagger_train.jsonl   # 49,643 examples
│   ├── resolver_train.jsonl      # 17,328 examples
│   └── translator_v3_train.jsonl # 36,268 examples
├── models/
│   ├── canonicalizer/            # Qwen-0.5B + LoRA
│   ├── slot_tagger/              # LoRA E
│   ├── resolver/                 # LoRA F
│   └── translator_v3/            # LoRA D
├── scripts/
│   ├── extract_slot_resolver_data.py
│   ├── train_slot_resolver_translator.py
│   └── full_pipeline.py
├── cycles/                       # feedback loop versioned rollback
├── evaluation/                   # per-verb + per-worker error attribution
├── feedback_loop/                # harvested traces
└── checkpoints/                  # crash recovery
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
├── train_slot_resolver_translator.py # Binding split training
├── extract_slot_resolver_data.py     # Training data extraction from CoT
├── full_pipeline.py                  # Full assembly line pipeline
├── build_canonicalizer_data.py       # Sonnet → rough telegram targets
├── test_pipeline_e2e.py              # End-to-end pipeline test
└── deprecated/                       # Phased out: pre-binding-split monolithic translator

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

# Custom model paths
python scripts/eval.py --narrator /path/to/lora_c --slot-tagger /path/to/lora_e --n 20
```

---

## MVP Build Order

```
Day 1-4:  ✅ Build canonicalizer training data + train + test
Day 5-6:  ✅ Wire stage-based pipeline: C1-A → canonicalizer → ODE → SymPy
Day 7:    ✅ Diagnose Translator bottleneck → binding split design
Day 8:    ✅ Implement Slot Tagger + Resolver + Translator v3
Day 9:    ✅ Train binding split. Resolver 90%, Translator 100% parseable
Day 10:   🔄 Fix Slot Tagger LaTeX normalization. Retrain.
Day 11:   Full cascade eval. Error attribution across all workers.
Day 12:   Self-improvement loop cycle 0. Harvest traces.
Day 13:   March 22, 2026 — MATH500 benchmark.
```

---

## Future Directions

- C1-A joins feedback loop (scaffold corrections as training data)
- C1-B adaptive ODE integration time based on problem complexity
- Laplace spectral fingerprint (telegraph poles/residues conditioning ODE)
- Teacher trajectory distillation (full path supervision from IAF data)
- MCTS for solution path discovery (search over SymPy transformations)
- Scale to Qwen-1.8B if 0.5B ceiling is too low
- CoT distillation baseline for paper comparison
- Parts Lookup retrieval for theorem/identity database
