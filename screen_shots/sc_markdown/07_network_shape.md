# Network Shape

## The Architecture

```
┌─────────────────────────────┐
│      INPUT (narrow)         │
│                             │
│      [step embedding]       │
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│      MIDDLE (wide)          │
│                             │
│  sig_1  sig_2  sig_3        │  ← 300-1000 signature centroids
│  sig_4  sig_5  sig_6        │    (many ways to describe each op)
│  ...    ...    ...          │
└──────────────┬──────────────┘
               │
               ▼ (collapse: many-to-one)
┌─────────────────────────────┐
│      OUTPUT (narrow)        │
│                             │
│      1 of 200 functions     │
└─────────────────────────────┘
```

---

## The Shape Makes Sense

| Layer | Size | What it represents |
|-------|------|-------------------|
| Input | 1 | One step description |
| Middle | 300-1000 | All the ways humans describe operations |
| Output | 200 | The actual functions |

---

## The Middle Layer is Semantic Diversity

```
"add" function has signatures:
  - "combine two prices"      ─┐
  - "sum the quantities"       ├─→ ALL map to add()
  - "total distance traveled" ─┘

"multiply" function has signatures:
  - "calculate total cost"    ─┐
  - "compute area"             ├─→ ALL map to multiply()
  - "find the product"        ─┘
```

The wide middle captures the **many-to-one** mapping from natural language → functions.

---

## This is Like...

**Prototype Networks / Exemplar-based classification:**
- Store many examples (signatures) per class (function)
- Classify by nearest prototype
- Collapse to class label

The tree is learning **prototypes** for each function class, curated by success rate.

---

## The Implication

- **Early on:** Few signatures, sparse middle → more misclassification
- **Mature:** Rich middle layer → robust classification from many angles

The system literally grows its "hidden layer" through experience.
