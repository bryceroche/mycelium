# Mycelium v6: Field Guide & Lessons Learned

## Updated: February 24, 2026

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

### 4. Error Attribution Framework
- Hierarchical: teacher errors first, then pipeline component errors
- ANSWER_NOT_IN_COT → TEACHER_ERROR → SEGMENTATION → CLASSIFICATION → EXTRACTION → DEPENDENCY → SELECTION → EXECUTION
- Tells you exactly where to invest optimization effort
- Key finding: 77% of failures were teacher-side, not pipeline-side

### 5. `\boxed{}` Extraction as Highest-Priority Answer Source
- MATH format always puts final answer in \boxed{}
- Extract before attempting any pipeline computation
- Catches 69% of correct answers trivially
- **But also masks real pipeline performance** — must disable for honest eval

### 6. Generation-Phase vs Reading-Phase Attention -- Prefill vs Decode
- Same attention heads do completely different things depending on phase
- Reading phase: linguistic structure (syntax, entities, sentences)
- Generation phase: computational structure (operations, operands, results)
- Mycelium v3-v5 failed on reading-phase attention → v6 succeeded on generation-phase
- JSD on generation phase = operation boundaries
- JSD on reading phase = linguistic boundaries (useless for math)

---


## Mantras

1. **Always visualize your spans.** Print them. Read them. If they look like "rmine the va" you have a bug.
2. **Execution is a free verifier.** If the answer is right, the labels are right.
3. **Generation-phase attention, not reading-phase.** The computation structure is only visible during generation.
4. **IAF is the bridge.** It maps what the model computed (CoT) to where it was looking (problem text).
5. **Clause expansion makes IAF learnable.** Scattered tokens → contiguous phrases.
6. **Disable `\boxed{}` for honest eval.** Otherwise you're measuring the 7B, not the pipeline.
7. **Error attribution before guessing.** Run the diagnostic, don't assume where the bug is.


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