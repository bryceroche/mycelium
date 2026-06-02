# Phase 1: NL → Factor Graph Parser Spec

**Goal:** Build a small, edge-deployable model that maps GSM8K-style natural-language math problems to factor graphs (the input format for our v100+ breathing transformer).

**Status:** Spec only. Implementation deferred until v100+ Phase 2 architecture is locked.

## Architecture decision

**T5-small (60M params, ~250MB; ~70MB INT8 quantized)** fine-tuned via Haiku distillation.

Why T5-small:
- Seq2seq matches the task shape (text → structured output)
- Pre-trained on text-to-structure tasks (T5's original training mix included translation, classification, summarization)
- Small enough for phone deployment with quantization
- HuggingFace ecosystem makes training cheap

Alternative: DistilBERT (66M, encoder + heads). More engineering, smaller after quantization (~40MB), but multi-head architecture is more complex to ship.

## Factor graph schema (v1)

```json
{
  "n_vars": 7,
  "n_factors": 3,
  "domain": [0, 10000],
  "factor_types": ["sub", "sub", "mul"],
  "factor_args": [
    [0, 1, 4],
    [4, 2, 5],
    [5, 3, 6]
  ],
  "observed_mask": [1, 1, 1, 1, 0, 0, 0],
  "observed_values": [16, 3, 4, 2, null, null, null],
  "query_idx": 6,
  "var_descriptions": [
    "janet eggs total",
    "eggs eaten for breakfast",
    "eggs used for baking",
    "price per egg in dollars",
    "eggs remaining after breakfast",
    "eggs remaining after baking",
    "total revenue from selling remaining eggs"
  ]
}
```

Differences from v100's synthetic schema:
- `domain: [0, 10000]` — GSM8K has bigger numbers than synthetic [0,99]
- `var_descriptions` — for debugging / human inspection (not used by Phase 2)
- Otherwise identical structure → drops into existing v100/v101/v103 evaluators

## Data pipeline

### Stage 1: Prompt design (1 day)

Iterate the Haiku prompt on ~50 GSM8K problems until output quality is consistent:
- Required format compliance (valid JSON matching schema)
- Correct variable identification (every quantity → one variable)
- Correct operation mapping (verbs → ops, modifiers → composition)
- Correct operand binding (pronouns resolved, "remaining" referenced properly)

Iterate prompt until ~95%+ of sampled outputs are correct on manual check.

### Stage 2: Full-scale labeling (1 day, ~$100 in Haiku)

Run Haiku on 5K GSM8K train problems + 1K GSM8K test problems.

Cost estimate:
- Average problem ~150 tokens input
- Average output ~250 tokens (JSON)
- Haiku rate: ~$0.005/1K input, ~$0.025/1K output
- Per problem: ~0.15*0.005 + 0.25*0.025 ≈ $0.007
- 6000 problems × $0.007 = **~$50**

Filter outputs:
- Must be valid JSON
- Must match schema (all required fields present)
- All factor_args indices must be valid
- All observed_values must be numeric where mask=1
- Reject + relabel ~5% that fail filters

Save to:
- `.cache/gsm8k_factor_graphs_train.jsonl` (5K records)
- `.cache/gsm8k_factor_graphs_val.jsonl` (1K records)

### Stage 3: Schema validation (4 hours)

Sample 200 records, manually check:
- Does the factor graph correctly represent the problem?
- Would executing the DAG produce the gold answer?

Compute "computability rate" — fraction where:
1. Parsed factor graph is structurally valid
2. Topological eval matches gold answer

Target: >90% computability rate. If below, iterate prompt.

### Stage 4: Parser training (1 day)

Fine-tune T5-small on (NL_problem, factor_graph_JSON) pairs:
- 5K training examples
- 1K validation examples
- 3-5 epochs (T5-small overfits quickly with too many)
- AdamW, LR 3e-5, batch 8
- Loss: standard seq2seq CE on output tokens

Eval metrics:
- **Exact schema match**: parser output matches Haiku label exactly
- **Structural match**: same n_vars, n_factors, factor topology (op types + indices)
- **Computability**: parser output → topological eval → matches gold answer
- **End-to-end accuracy**: parser output → v100+ → matches gold answer

Target: >70% structural match, >50% computability.

### Stage 5: End-to-end evaluation (4 hours)

Pipeline: GSM8K problem → T5-small parser → v100+ breathing transformer → answer

Compare to baselines:
- Direct Haiku solve (cloud, expensive)
- Direct Pythia-1B solve (no parsing — uses model knowledge directly)
- Direct breathing transformer (Phase 2 only, no parsing)

Expected: our pipeline matches or exceeds Pythia-1B with much smaller deployable footprint.

## Implementation roadmap

```
Week 1: Prompt design + small-scale validation
  - Day 1-2: Design + iterate Haiku prompt on 50 problems
  - Day 3: Computability check, iterate if <85%
  - Day 4-5: Lock schema + prompt

Week 2: Full labeling + parser training
  - Day 1: Full Haiku labeling (5K + 1K problems)
  - Day 2: Schema validation, filter, save
  - Day 3-4: T5-small fine-tuning
  - Day 5: Eval + ablations

Week 3: End-to-end testing
  - Day 1-2: Pipeline integration with v100+
  - Day 3-4: End-to-end accuracy measurements
  - Day 5: Paper section writeup
```

Total: ~3 weeks of focused work. Most of the work is data + training; the architectural choices are settled.

## Prompt template (draft for Stage 1)

```
You are a math problem parser. Given a natural-language math problem,
output a factor graph that represents the problem's computational structure.

The factor graph format:
{
  "n_vars": int (total number of variables)
  "n_factors": int (number of arithmetic operations)
  "domain": [min, max] (value range for all variables)
  "factor_types": list of "add"|"sub"|"mul"|"div"
  "factor_args": list of [arg1_idx, arg2_idx, result_idx]
  "observed_mask": list of 0/1 (1 if value is given in the problem)
  "observed_values": list (int or null for unknowns)
  "query_idx": int (which variable is the answer)
  "var_descriptions": list of strings (human-readable for each variable)
}

Rules:
1. Every numerical quantity in the problem becomes a variable.
2. Every arithmetic operation becomes a factor.
3. The QUERY is the variable whose value the problem asks for.
4. Indices are 0-based.
5. Observed variables come first; computed variables after.
6. If the problem requires non-arithmetic reasoning (comparison, conditional),
   indicate this with operation type "compare" or "if" (these will be
   handled later — for now we focus on pure arithmetic problems).

Example:

Problem: "Janet has 16 eggs. She eats 3 for breakfast. She bakes muffins
         with 4. She sells the remaining eggs at $2 each. How much money?"

Output:
{
  "n_vars": 7,
  "n_factors": 3,
  "domain": [0, 10000],
  "factor_types": ["sub", "sub", "mul"],
  "factor_args": [[0, 1, 4], [4, 2, 5], [5, 3, 6]],
  "observed_mask": [1, 1, 1, 1, 0, 0, 0],
  "observed_values": [16, 3, 4, 2, null, null, null],
  "query_idx": 6,
  "var_descriptions": [
    "janet eggs total",
    "eggs eaten for breakfast",
    "eggs used for baking",
    "price per egg in dollars",
    "eggs remaining after breakfast",
    "eggs remaining after baking",
    "total revenue from selling remaining eggs"
  ]
}

Now parse this problem:
{PROBLEM}
```

## Key design questions to resolve in Stage 1

1. **Division: integer or fractional?** GSM8K has division problems that
   produce non-integer intermediates. Schema needs to support this.

2. **Comparison / conditional operations** — GSM8K has "if X else Y" type
   problems. Out of scope for v1; tag and skip.

3. **Multi-line / multi-paragraph problems** — most GSM8K is 1-2 sentences
   but some are longer. Test prompt on the long-tail.

4. **Units / unit conversion** — "5 dollars = 500 cents" type. Probably
   represent as multiplication by conversion constant.

5. **Implicit variables** — "X total cost" might not appear as a number
   in the text but is the query. Handle via var_descriptions.

## Connection to Phase 2

The factor graph schema MUST match what v100+ consumes:
- Same JSON keys
- Same indexing scheme (observed first, computed after)
- Same op type vocabulary (add/sub/mul/div)
- Compatible variable domain (we may need to extend v100's [0,99] to [0,10000])

**Action item for Phase 2 work:** when training v100+ for GSM8K, regenerate synthetic factor graphs with domain [0,10000] to match Phase 1 output. Or alternatively, train v100+ on Phase 1's Haiku-labeled data directly (use both synthetic + real factor graphs in the training mix).

## Cost summary

```
Haiku labeling:          ~$50  (5K + 1K problems, one-time)
Prompt iteration:        ~$10  (50 problems × 10 iterations during design)
T5-small training:        ~$0  (runs on existing GPU, ~1 day)
End-to-end eval:          ~$0  (parser locally + v100+ locally)
                         ─────
Total:                    ~$60 + 3 weeks of work
```

## Why the gradient-conflict avoidance matters

Phase 1 and Phase 2 use OPPOSITE inductive biases:
- Phase 1: encoder + decoder, attention over text, learning language patterns
- Phase 2: shared-weight loop over fixed-structure attention masks, learning constraint propagation

Combining into one model means optimization tries to be good at both with the same parameters. The v85 family's structural ceiling proved this fails — within ONE model, tasks at different breath positions still gradient-conflicted. Across DIFFERENT tasks (parse vs solve), the conflict is even stronger.

The two-system architecture eliminates this by construction. Each set of weights learns one task. The interface between them (the factor graph schema) is the abstraction boundary.
