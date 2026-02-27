# Mycelium

---

> λ — λανθάνω (lanthánō) — to escape notice; to be unseen
> 
> JSD reveals latent boundaries in the flow of attention  
> IAF separates reading from reasoning — the dual phases hidden in every forward pass  
> FFT extracts the frequency of thought — each oscillation a computation the model never knew it performed  
> IB discovers the taxonomy of operations — spectral lines emerging from continuous signal  
> The prism decomposes. The specialists distill. Sympy collapses the wave function  
> 
> Mycelium — the unseen network of computation, made visible

---

## Documentation

| Document | Description |
|----------|-------------|
| [What we are building](plan/section_lkup.md) | Lookup table approach for inference |
| [Six Model Training Guide](plan/six_model.md) | Step-by-step guide for training all six specialist models |
| [MCTS Wrapper](plan/mcts_wrapper.md) | Monte Carlo Tree Search layer design |
| [Wave Function Collapse](plan/section_wfc.md) | Beam inference via wave function collapse |
| [Observations: Abacus](plan/obs_abacus.md) | Abacus analogy and observations |
| [Observations: Prism](plan/obs_prism.md) | Prism analogy and observations |
| [Observations: Lambda](plan/obs_lambda.md) | Lambda (λανθάνω) etymology and observations |
| [Field Guide](plan/field_guide.md) | Lessons learned, bugs encountered, tricks that work |
---

## The Big Picture

A 7B teacher model solves math problems via chain-of-thought. We extract **three signals** from its attention patterns — span boundaries (JSD), problem text mapping (IAF), and CoT operation structure — and distill them into small student models that reproduce the reasoning without generating any text.

```
TRAINING:  7B solves problems → JSD segments CoT → IAF maps to problem text → train specialists
INFERENCE: Problem text → specialists parse structure → assemble sympy program → execute → answer
```

No chain-of-thought at inference. The answer comes from understanding problem structure and executing symbolically, not from language modeling.

---

## Training Pipeline

### Three Signals from One Forward Pass

The 7B teacher provides everything from a single forward pass with attention hooks:

| Signal | What It Captures | How It's Extracted |
|--------|-----------------|-------------------|
| **CoT text** | The reasoning trace | Standard generation |
| **JSD boundaries** | Where computation steps are | Jensen-Shannon Divergence between consecutive attention distributions, Savitzky-Golay smoothed (w=5) |
| **IAF mapping** | Which problem text each step attended to | Input Attention Fraction — generation→input attention flow |


## Reading vs generation attention (Prefill vs Decode)

Same model, same heads, two phases, fundamentally different information:

**Reading-phase attention** organizes by linguistic structure — syntax, entities, sentence boundaries. **Generation-phase attention** organizes by computational structure — operation steps, operand routing, result production.


**IAF and operand matching are orthogonal** (0.5% token overlap):
- IAF captures semantic context: "Natalia", "altogether", "clips", "May"
- Operand matching finds numbers: "48", "half"
- Combined = complete operation specification

---

## Error Attribution and Benefits of inference model specialization

Splitting span segmentation, classification, and extraction into separate components allows for precise error attribution.  It also allows the qwen .5b inference model to specialize for each task.

---

### Tricks That Work

1. **JSD + Savitzky-Golay (w=5)** — peaks in generation-phase attention = computation step boundaries
2. **IAF** — maps CoT computation back to problem text regions (orthogonal to operand matching)
3. **Clause expansion** — converts scattered IAF tokens to linguistically coherent regions via spacy
4. **IO tagging over BIO** — avoids label imbalance, 100% problem coverage
5. **Hybrid parser** — LaTeX patterns primary, classifier fallback, execution validation filters all
6. **Execution validation** — correct answer = valid labels, free supervision at scale
7. **Error attribution** — always diagnose before guessing, run the diagnostic script
8. **Role classification over operation classification** — EQUATION/CONSTRAINT/OBJECTIVE captures problem structure, not just arithmetic

### Mantras

1. **Always visualize your spans.** If they look like "rmine the va" you have a bug.
2. **Execution is a free verifier.** If the answer is right, the labels are right.
3. **Generation-phase attention, not reading-phase.** Computation structure is only visible during generation.
4. **Disable `\boxed{}` for honest eval.** Otherwise you're measuring the teacher, not the pipeline.
5. **Error attribution before guessing.** Run the diagnostic, don't assume where the bug is.
6. **The self-improving loop is the endgame.** Execution-validated traces → better models → more correct traces.

---
**The Scaling Story**
Inference cost: A frontier model doing chain-of-thought on a hard MATH problem might generate 2000+ tokens autoregressively. That's 2000 forward passes through hundreds of billions of parameters. Mycelium does one forward pass through each 0.5B specialist — six passes total, each through a tiny model. Orders of magnitude cheaper.
Knowledge leverage: Those 3B parameters could contain distilled reasoning structure from a model 1,000x their size. They're not trying to know everything — they just know how to decompose, classify, extract, and wire up math problems. Sympy handles the actual math for free.  This could apply to any domain where you have a small set of primitives that compose in predictable ways.
The real unlock: The teacher only runs at training time. At inference, it's gone. You pay the big model cost once to generate attention patterns, then amortize that across unlimited cheap inference. It's like a master mathematician teaching six apprentices each one specific skill, then retiring.

