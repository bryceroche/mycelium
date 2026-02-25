# Lambda Handoff: Fix Label Transfer Pipeline

## Problem

The classifier has only 852 training examples because the regex CoT parser drops 45% of spans. MATH CoT uses LaTeX notation and semantic operations that simple digit-operator-digit regex can't catch.

## Solution: Hybrid Parser (LaTeX + Classifier Fallback)

### Tier 1: LaTeX Parser (primary, should catch ~80%)

Use sympy.parsing.latex to parse LaTeX math expressions from CoT spans:

```python
from sympy.parsing.latex import parse_latex
import re

LATEX_PATTERNS = {
    # Fractions
    r'\\frac\{(.+?)\}\{(.+?)\}': 'DIV',
    # Powers
    r'(.+?)\^(\{.+?\}|\d)': 'POW',
    # Square roots
    r'\\sqrt(\[.+?\])?\{(.+?)\}': 'ROOT',
    # Binomials
    r'\\binom\{(.+?)\}\{(.+?)\}': 'COMB',
    # Summation
    r'\\sum': 'ADD',
    # Product
    r'\\prod': 'MUL',
    # Min/Max
    r'\\min': 'MIN',
    r'\\max': 'MAX',
    # Trig
    r'\\(sin|cos|tan|cot|sec|csc)': 'TRIG',
    # Logarithm
    r'\\(log|ln)': 'LOG',
    # Modular arithmetic
    r'\\pmod|\\mod|\\equiv': 'MOD',
}

# Standard arithmetic (enhanced from current regex)
ARITHMETIC_PATTERNS = {
    r'(\d+\.?\d*)\s*[\+]\s*(\d+\.?\d*)': 'ADD',
    r'(\d+\.?\d*)\s*[\-]\s*(\d+\.?\d*)': 'SUB',
    r'(\d+\.?\d*)\s*[\*×·]\s*(\d+\.?\d*)': 'MUL',
    r'(\d+\.?\d*)\s*[/÷]\s*(\d+\.?\d*)': 'DIV',
    r'(\d+\.?\d*)\s*=\s*(\d+\.?\d*)': 'ASSIGN',
}

def parse_cot_span_latex(span_text):
    """Parse a CoT span for operation using LaTeX patterns."""
    
    # Try LaTeX patterns first
    for pattern, op_label in LATEX_PATTERNS.items():
        if re.search(pattern, span_text):
            # Extract arguments using sympy if possible
            try:
                expr = parse_latex(span_text)
                args = extract_args_from_sympy(expr)
                return op_label, args
            except:
                return op_label, extract_args_regex(span_text)
    
    # Fall back to arithmetic patterns
    for pattern, op_label in ARITHMETIC_PATTERNS.items():
        match = re.search(pattern, span_text)
        if match:
            return op_label, list(match.groups())
    
    # No pattern matched
    return None, None
```

### Tier 2: Existing 60% Classifier (fallback for ~20% that parser misses)

For spans where the LaTeX parser returns None:

```python
def parse_cot_span_hybrid(span_text, classifier_model):
    """Hybrid parser: LaTeX first, classifier fallback."""
    
    # Tier 1: LaTeX parser
    label, args = parse_cot_span_latex(span_text)
    if label is not None:
        return label, args, "parser"
    
    # Tier 2: Classifier fallback
    pred_label, confidence = classifier_model.predict(span_text)
    if confidence > 0.5:
        args = extract_args_regex(span_text)
        return pred_label, args, "classifier"
    
    # Neither worked
    return None, None, "failed"
```

### Tier 3: Execution Validation (filters everything)

After labeling all spans for a problem:

```python
def validate_by_execution(problem, labeled_spans, gold_answer):
    """Only keep traces where execution produces correct answer."""
    dag = build_dag(labeled_spans)
    result = execute(dag)
    
    if result == gold_answer:
        # All labels validated — save as training data
        return [(span.clause, span.label, span.args) for span in labeled_spans]
    else:
        return []  # Discard entire trace
```

Wrong labels from either source get filtered out because they produce wrong answers.

## Execution Plan

### Step 1: Implement hybrid parser
- Write `parse_cot_span_latex()` with the LaTeX patterns above
- Test on 100 CoT spans, report: how many does Tier 1 catch? Tier 2? Failed?
- Target: >90% spans get a label (vs 55% with current regex)

### Step 2: Re-run label transfer on existing 1,516 IAF problems
- Use hybrid parser instead of regex
- Report: how many training pairs now? (target: >1,200 vs current 852)

### Step 3: Extract IAF for remaining ~6,300 problems
- Same forward pass approach as the 1,516 batch
- This is the big volume step — 7,842 total IAF files
- With better parser, should yield ~5,000+ training pairs

### Step 4: Execution validation
- Build DAGs from labeled spans
- Execute and compare to gold answers
- Keep only correct traces as training data

### Step 5: Retrain problem text classifier
- On the execution-validated (clause, operation_label) pairs
- Should have 3,000-5,000 examples (vs 852 currently)
- Report: validation accuracy (target: >50%, up from 35%)

### Step 6: E2E eval (problem text only, no CoT)
- IO tagger → clause classifier → execute → compare to gold
- This is the honest eval — no \boxed{}, no CoT

## What to Track
- Parser Tier 1 vs Tier 2 vs Failed breakdown
- Training pair count at each stage of the funnel
- Classifier accuracy on clause-level problem text
- E2E accuracy (the real number)

## Priority
Steps 1-2 first (quick, validates the parser improvement on existing data).
Then Steps 3-6 (scaling up).
