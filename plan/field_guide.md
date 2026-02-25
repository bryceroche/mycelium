# Mycelium v6: Field Guide & Lessons Learned

## Updated: February 24, 2026

---

## The Killer Bug: Token-to-Character Index Alignment

**Impact:** Poisoned every downstream component for the entire day.
**Symptom:** BIO F1 stuck at 36-54%, classifier at 35%, E2E at 0.5-2%.
**Root cause:** JSD boundary extraction saved token indices. Downstream code treated them as character indices. Every span was garbled mid-word fragments.

```
BEFORE (character indices — WRONG):
  [0] chars 0-8:   " To dete"
  [1] chars 8-20:  "rmine the va"
  [2] chars 20-39: "lues of \(a\) and\"

AFTER (token-aligned — CORRECT):
  [0] "To determine the values of \(a\) and"
  [1] "n = 2m^2 + ..."
  [2] "Combine the \(x^4\) terms:"
```

**Fix:** Convert token indices to character positions using tokenizer offsets.

**Lesson:** Always visualize your spans early. We built an entire pipeline — IO tagger, clause expansion, classifier, E2E eval — on top of garbage before anyone printed what the spans actually looked like. Five minutes of eyeballing would have caught this immediately.

**Data impact:**
- Training pairs: 852 → 3,547 (4.2x increase from same data)
- Classifier accuracy: 35% → 71.9%
- Everything before the fix was measuring noise, not architecture quality

---

## Bug Hall of Fame

| Bug | Symptom | Impact | Time to Find |
|---|---|---|---|
| Token→char index misalignment | Mid-word spans, garbled text | Poisoned ALL downstream models | ~8 hours |
| GSM8K segmenter on MATH | Scattered B-OP/I-OP tags | 5% E2E (vs 78% with JSD) | ~1 hour |
| BIO label imbalance | Model predicts I-OP without B-OP | 0% usable regions at inference | ~30 min |
| `\boxed{}` extraction masking real eval | 91% accuracy (all from boxed) | Pipeline never actually tested | ~1 hour |
| `max_new_tokens=512` truncation | 43% solve rate on GSM8K | Many "wrong" answers were truncated | ~2 hours |
| `\boxed{}` regex not matching | 43% solve rate (extraction bug) | 95.5% actually correct after fix | ~1 hour |
| Classifier on CoT text, inference on problem text | Nonsense predictions at inference | 0.5% E2E | ~30 min |
| `attn_implementation="eager"` missing | No attention capture | Silent failure, no attention data | Early catch |

---

## Tricks That Work

### 1. JSD + Savitzky-Golay (w=5) for Span Boundaries
- Compute Jensen-Shannon Divergence between consecutive token attention distributions
- Smooth with Savitzky-Golay filter, window=5, polyorder=2
- Peaks = computation step boundaries
- Works on both 7B and 72B (7B validation in progress)
- ~1 boundary per 10-15 tokens on MATH CoT
- **Critical:** Boundaries are TOKEN indices — convert to char positions downstream

### 2. IAF (Input Attention Fraction) for CoT→Problem Text Mapping
- For each CoT span, measure how much generation-phase attention flows to each input token
- IAF(j, span) = attention from span's gen tokens to input token j / total attention to all input tokens
- Captures semantic context: "Natalia", "altogether", "clips"
- Orthogonal to operand matching (0.5% token overlap)
- Operand matching finds numbers, IAF finds meaning
- Combined = complete operation specification

### 3. Clause Expansion (IAF Tokens → Regions)
- Raw IAF gives scattered high-attention tokens
- Use spacy dependency parse to find enclosing clause for each token
- Merge overlapping clauses
- Result: contiguous, linguistically coherent operation regions
- BIO F1: 36% (raw tokens) → 54% (clause expanded) — and that was with the index bug!

### 4. IO Tagging Instead of BIO
- BIO creates severe label imbalance (1 B-OP per ~9 I-OP)
- Model learns to predict I-OP everywhere, never predicts B-OP
- IO tagging: just OP vs O, any contiguous OP run = region
- 100% problem coverage (vs 36% with BIO)
- Loses ability to detect adjacent regions, but math problems rarely have them

### 5. Hybrid CoT Parser (LaTeX + Classifier Fallback)
- Tier 1: LaTeX pattern matching + sympy parsing (~80% of MATH CoT)
  - Handles \frac, ^, \sqrt, \binom, \min, \max, \log, \mod, etc.
- Tier 2: Existing 60% classifier for semantic operations parser misses (~20%)
- Tier 3: Execution validation filters everything — wrong labels produce wrong answers and get discarded

### 6. Execution Validation as Free Supervision
- If the pipeline produces the correct numerical answer, all intermediate labels are validated
- No human annotation needed at any stage
- Analogous to rejection sampling: generate candidates, keep correct ones, train on survivors
- This is what makes the self-improving loop work

### 7. Error Attribution Framework
- Hierarchical: teacher errors first, then pipeline component errors
- ANSWER_NOT_IN_COT → TEACHER_ERROR → SEGMENTATION → CLASSIFICATION → EXTRACTION → DEPENDENCY → SELECTION → EXECUTION
- Tells you exactly where to invest optimization effort
- Key finding: 77% of failures were teacher-side, not pipeline-side

### 8. Hybrid Teacher Strategy (7B + 72B)
- Run 7B on everything (cheap, 69% solve rate on MATH)
- Failed indices → 72B CoT mop-up (expensive, targeted)
- Reduces 72B compute by ~70-85%
- 78% → 85% on MATH500

### 9. `\boxed{}` Extraction as Highest-Priority Answer Source
- MATH format always puts final answer in \boxed{}
- Extract before attempting any pipeline computation
- Catches 69% of correct answers trivially
- **But also masks real pipeline performance** — must disable for honest eval

### 10. Generation-Phase vs Reading-Phase Attention
- Same attention heads do completely different things depending on phase
- Reading phase: linguistic structure (syntax, entities, sentences)
- Generation phase: computational structure (operations, operands, results)
- Mycelium v3-v5 failed on reading-phase attention → v6 succeeded on generation-phase
- JSD on generation phase = operation boundaries
- JSD on reading phase = linguistic boundaries (useless for math)

---

## Things That Don't Work

| Approach | Why It Failed |
|---|---|
| Trained segmenter across domains | GSM8K segmenter fails on MATH — domain mismatch |
| Confidence thresholding for segmentation | No bimodality in classifier confidence — can't find spans |
| Raw IAF tokens as BIO labels | Scattered tokens, not contiguous regions |
| BIO tagging with imbalanced labels | Model never predicts B-OP, only I-OP |
| Regex-only CoT parsing for MATH | Misses LaTeX, semantic operations — 45% data loss |
| Classifier trained on CoT, eval on problem text | Completely different distributions |
| Sliding window classification (v3-v5) | Operations need global context, not local windows |
| T5 seq2seq for operation prediction | 8% accuracy — wrong tokenizer, wrong output format |
| Single 0.5B doing segmentation + classification | Confidence thresholding doesn't work as implicit segmentation |

---

## Current Pipeline State (Post-Bug-Fix)

```
Component               Status          Performance
─────────────────────────────────────────────────────
7B CoT generation       Complete        12,500 problems, 68.8% correct
JSD extraction          Complete        7,842 files (NOW WITH CORRECT SPANS)
IAF extraction          Complete        1,516 stratified sample
LaTeX parser            Complete        4.2x more training pairs
IO tagger               Trained         ~54% F1 (needs retrain on clean data)
Classifier v3           Trained         71.9% accuracy (up from 35%)
E2E (problem text only) RUNNING         ???
```

---

## Architecture (Current)

```
TRAINING:                                   INFERENCE:

  Problem Text                                Problem Text
      │                                           │
      ▼                                           ▼
  7B generates CoT ◄── attention ──┐        ┌──────────────────┐
      │                            │        │ 0.5B IO Tagger   │
      ▼                            │        │ (finds OP regions)│
  JSD segments CoT ◄──────────────┤        └────────┬─────────┘
  (token→char aligned!)            │                 │
      │                            │                 ▼
      ▼                            │        ┌──────────────────┐
  IAF maps CoT spans ─────────────┘        │ 0.5B Classifier  │
  to problem text tokens                   │ (region → op)    │
      │                                    └────────┬─────────┘
      ▼                                             │
  Clause expansion (spacy)                          ▼
      │                                       Execute DAG
      ▼                                             │
  LaTeX parser + classifier fallback                ▼
      │                                          Answer
      ▼
  Execution validation
      │
      ▼
  Train 0.5B models on problem text
```

**Teacher:** Qwen2.5-Math-7B-Instruct
**Student:** 2x Qwen2-0.5B (IO tagger + classifier) = 1B total
**No CoT at inference. No 72B required.**

---

## Key Numbers Timeline

| Metric | Before Bug Fix | After Bug Fix |
|---|---|---|
| Training pairs | 852 | 3,547 |
| Classifier accuracy | 35% | 71.9% |
| E2E (with CoT/boxed) | 91% (fake — all boxed) | N/A |
| E2E (problem text only) | 0.5% - 2% | TBD |
| IO tagger F1 | 54% (on garbled data) | Needs retrain |
| BIO tagger F1 | 36% (on garbled data) | Abandoned for IO |

---

## Self-Improving Loop (Next Phase)

```
Round 1:
  - Train on ~3,547 execution-validated pairs
  - Eval on full MATH → collect correct traces
  - 7B mop-up on failures → more training data

Round 2+:
  - Expand IAF to all 7,842 (currently only 1,516)
  - With better parser: ~18,000+ training pairs
  - Retrain → eval → collect → repeat
  - Stop when delta < 1% between rounds
```

---

## Mantras

1. **Always visualize your spans.** Print them. Read them. If they look like "rmine the va" you have a bug.
2. **Execution is a free verifier.** If the answer is right, the labels are right.
3. **Generation-phase attention, not reading-phase.** The computation structure is only visible during generation.
4. **IAF is the bridge.** It maps what the model computed (CoT) to where it was looking (problem text).
5. **Clause expansion makes IAF learnable.** Scattered tokens → contiguous phrases.
6. **Disable `\boxed{}` for honest eval.** Otherwise you're measuring the 7B, not the pipeline.
7. **Error attribution before guessing.** Run the diagnostic, don't assume where the bug is.
8. **The self-improving loop is the endgame.** Execution-validated traces → better models → more correct traces.
