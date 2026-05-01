# Mycelium — Breathing Models

Differentiable recurrent reasoning for small language models. A frozen 1B LLM learns to decompose math problems through a tree-structured breathing loop — each node rewires attention, computes one step, and records understanding in a hierarchical notebook.

**The thesis:** decomposition is everything. The model doesn't become smarter — it learns to break hard problems into pieces it can already solve.

**Lead:** Bryce Roche · **Target:** MATH-500 · **Deadline:** September 1, 2026

---

## Results

| Task | Base Llama | Breathing | Notes |
|------|-----------|-----------|-------|
| L3 (1-step arithmetic) | 18.8% | **100%** | 3 epochs from scratch |
| L4 (2-step word problems) | 6.0% | **99.5%** | Genuine 2-cycle decomposition |
| L4.5 (3-step word problems) | — | **99.5%** | All 3 cycles contribute |
| GSM8K | 2.2% | **14%** (v1) | v2 rebuild in progress |

**Key:** Llama 3.2 1B (frozen) + ~248M trainable params. The breathing loop chains reasoning that the base model cannot do alone.

**The breakthroughs:**
- **Equal-reward decomposition:** Each target worth 1/N, consumed once — the ONLY way to maximize reward is to decompose
- **Baked L1 math-mode:** L4.5 LoRA atoms permanently absorbed into Llama's weights — zero runtime cost
- **Direct controller gradient:** Straight-through estimator bypasses 1.2B frozen Llama — controller gets full-strength signal

```
Without equal-reward:  Cycle 1=100%  Cycle 2=6%    Final=0%   (one-shot, dead loop)
With equal-reward:     Cycle 1=100%  Cycle 2=99.5%  Final=99.5% (genuine chaining!)
```

---

## How It Works

### The Breathing Loop (v2 — Tree-Structured)

```
COMPREHEND: Llama reads problem → Controller observes

BUILD TREE:
  DECOMPOSE → split into independent subproblems
  SOLVE     → inner loop refines atom blend → Llama generates answer
  MERGE     → combine child results

Each node earns 1/N reward per matched target.
```

### Architecture

```
Component                        Params      Role
──────────────────────────────────────────────────────────────
Llama 3.2 1B (frozen + baked L1) 1,230M      Thinks (math-mode attention)
BreathingController              ~166M       Observes + plans + decides
64 L2 LoRA Atoms                   82M       Per-node attention steering
Energy Head                      incl.       Adaptive inner loop stopping
Tree Notebook                    incl.       Hierarchical memory
──────────────────────────────────────────────────────────────
Trainable:                       ~248M
Frozen:                          1.23B
```

### Key Discoveries

- **V,O projections reprogram Llama** from text-completion to math-computation (1.9%→13.8%)
- **Gentle atoms [-0.46, 0.46]** don't corrupt Llama's arithmetic circuits
- **Equal-reward incentive** is what makes decomposition happen — not architecture, not loss functions
- **Controller gradient must bypass Llama** — routing through 1.2B frozen params attenuates 500x
- **0.46 * tanh(x)** for scales — clamp+tanh creates dead gradient zones that collapse the controller

---

## Current Status: v2 Rebuild

v1's controller was a **constant function** — identical outputs for every problem. All results came from L2 atoms learning a universal blend without steering. v2 rebuilds with direct gradient as the cardinal rule.

```
Phase 0: Controller smoke test     ✓ PASSED (scale_cos=-0.02, orthogonal!)
Phase 1: Linear breathing (L3-L4.5) ← NEXT
Phase 2: Tree structure curriculum
Phase 3: GSM8K with trees (>25%)
Phase 4: MATH-500 (>15%)
```

---

See `CLAUDE.md` for full technical context and `plan/mycelium_v2_master_rebuild_handoff.md` for the rebuild design.
