# Signatures as Examples

## Current Flow

```
LLM outputs: func="add", semantic="total cost"
     ↓
Tree routes via embedding similarity → finds signature_42
     ↓
Execute: call_function("add", 3, 2) → 5
```

---

## What Signatures Store

```python
signature_42:
    func_name: "add"
    description: "combine two prices"
    centroid: [0.12, -0.34, ...]  # learned embedding
    successes: 47
    uses: 52
```

---

## Insight: Signatures as Examples

Right now the LLM doesn't see the signatures. But we could flip this:

```
PROMPT: "Here are known patterns for 'add':
  - combine two prices
  - sum the quantities
  - total distance traveled

Here are known patterns for 'mul':
  - calculate total cost (price × quantity)
  - compute area (length × width)
  ..."
```

---

## The Feedback Loop

1. Tree learns which descriptions succeed for which functions
2. High-success signatures become few-shot examples for LLM
3. LLM produces better decompositions
4. Tree gets even better signal

**The signatures become a learned vocabulary** - not just for routing, but for teaching the LLM what operations look like in natural language.

The tree essentially **curates examples from experience**.
