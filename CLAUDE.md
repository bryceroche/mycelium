# CLAUDE.md — Mycelium v2

## Project Overview

Mycelium builds differentiable recurrent reasoning for small language models. A frozen Llama 3.2 1B learns to decompose math problems through a tree-structured breathing loop — each node rewires attention via LoRA atoms, observes the result, and plans the next step. The core thesis: **decomposition is everything.**

**Lead:** Bryce (Manhattan Beach, CA) · **Target:** MATH-500 · **Deadline:** September 1, 2026
**Infrastructure:** AWS EC2 g5.xlarge (A10G 24GB)

---

## Architecture (v2 — Tree-Structured Breathing)

```
BREATHING LOOP (tree-structured):

  STEP 1: COMPREHEND
    Llama (with baked L1 math-mode) reads problem
    Controller reads hidden states → initial tree plan

  STEP 2: BUILD TREE (recursive)
    DECOMPOSE: split into child subproblems
    SOLVE: inner loop refines L2 scales → Llama generates "equation #### answer"
    MERGE: combine child results into final answer

  STOP when: all targets claimed OR confidence high
```

**Components:**
- **Llama 3.2 1B (frozen + baked L1):** Base model with L4.5 math-mode permanently absorbed into weights. Template-invariant, 99.5% on 3-step procedural.
- **BreathingController (~166M, scaling to ~400M):** Reads Llama hidden states → produces scales + page + branch embedding + branch action + energy + confidence. Gets DIRECT gradient via straight-through estimator (never through Llama).
- **64 L2 LoRA Atoms (~82M):** Per-node attention steering at [-0.46, 0.46] volume via `0.46 * tanh(x)`. V,O do the heavy lifting.
- **Tree Notebook:** Hierarchical memory with parent/child/sibling attention. Pages on unit hypersphere (256d), branch embeddings in Euclidean with L2 clipping (64d).
- **Energy Head:** Learned stopping for adaptive inner loop. Contrastive training (low energy = correct, high = wrong). Lyapunov regularization (energy must decrease across passes).

**Cardinal rule:** Controller gradient NEVER flows through Llama. Separate backward passes always.

---

## Results

### v1 (completed)
```
L3 (1-step):   100.0%   ✓ (from scratch)
L4 (2-step):    99.5%   ✓ (genuine 2-step decomposition)
L4.5 (3-step):  99.5%   ✓ (genuine 3-step decomposition)
GSM8K:          ~14%     ✗ (controller was constant function — atoms2 only)
```

### v2 (in progress)
```
Phase 0: Controller smoke test  ✓ PASSED
  scale_xproblem_cos = -0.02 (orthogonal — different outputs per input)
  scale_mid_frac = 0.96 (scales in linear tanh regime)
  dead_dims = 0/64 (all dimensions active)
  Gradient flows to all components

Phase 1: Linear breathing on curriculum — NEXT
Phase 2: Tree structure on curriculum
Phase 3: GSM8K with trees (>25%)
Phase 4: MATH-500 (>15%)
```

---

## Key Design Decisions

- **Equal-reward (1/N per target):** The ONLY way to maximize reward is to decompose. Proven on L4/L4.5.
- **Baked L1 atoms:** L4.5 math-mode LoRA permanently added to Llama weights. No runtime cost.
- **0.46 * tanh(x) scales:** Smooth gradient everywhere. No clamp — v1's clamp+tanh killed gradients at boundaries, causing controller collapse.
- **Normalized ST gradient:** Per-sample unit-length direction signal. Bypasses 500x Llama attenuation.
- **Separate backward passes:** gen_loss→atoms2, ST loss→controller. Never combined.
- **Fresh controller init:** Never load collapsed controller weights. The v1 controller was a constant function its entire life.
- **Tree structure from day one:** Linear chains are degenerate trees. DECOMPOSE/SOLVE/MERGE via Gumbel-softmax.
- **Energy-based adaptive stopping:** Replaces fixed K inner passes. Contrastive + Lyapunov regularization.
- **Generation-only:** No answer head. Each cycle generates "equation #### number". Extraction via regex.
- **Number augmentation (0.8-1.2x):** Gentle range lets number patterns settle. Was 0.5-2.0x, too aggressive.
- **3x weight on number tokens:** Language tokens dominate gen_loss otherwise.

---

## File Structure

```
scripts/
  controller.py              # v2 BreathingController (166M, clean impl)
  smoke_test_controller.py   # Phase 0: verify controller differentiates
  atom_lora.py               # v1 model (AtomLoRAModel, LoRAAtoms, baking)
  train_per_cycle.py          # v1 training loop (equal-reward, augmentation)
  generate_per_cycle_data.py  # Data gen: L3-L4.9 procedural
  parse_gsm8k.py              # Parse GSM8K step annotations
  annotate_gsm8k_cycles.py    # Claude API: decompose problems
  diag_*.py                   # Diagnostics (8 scripts)
plan/
  mycelium_v2_master_rebuild_handoff.md  # v2 architecture + phases
  energy_based_inner_loop_design.md      # Energy head design
  + 10 more design documents
src/
  contrastive_page_loss.py    # Cross-problem page diversity
data/
  per_cycle/                  # L3-L4.9 + GSM8K JSONL
```

---

## AWS Setup

```bash
aws ec2 start-instances --instance-ids i-08c1c295a4113a908
ssh -i ~/.ssh/mycelium-key.pem ubuntu@<IP>
cd ~/mycelium && tmux new -s train
```

---

## Diagnostics

| Diagnostic | What it reveals |
|-----------|----------------|
| `smoke_test_controller.py` | Phase 0: controller alive? Different outputs per input? |
| `diag_debug_cycles.py` | Full chain per problem: scales, pages, generation |
| `diag_deep_single_step.py` | Failure categories: wrong_arithmetic, copying, format |
| `diag_laplace.py` | Page trajectory: convergence, oscillation, DC ratio |
| `diag_atom_inspect.py` | What each atom does to attention |

**v2 health metrics (run every epoch):**
| Metric | Healthy | Meaning |
|--------|---------|---------|
| `scale_xproblem_cos` | < 0.9 | Controller differentiates between problems |
| `scale_mid_frac` | > 0.3 | Scales not saturated |
| `dead_dims` | < 10/64 | Dimensions active |
| `controller_grad_norm` | 0.01-1.0 | Controller is learning |

---

## Failed Experiments (Don't Repeat)

| What | Result | Lesson |
|------|--------|--------|
| Answer head (5 versions) | 4% peak | Generation IS the output |
| Separate perceiver+hypernetwork | Constant function | Unified controller reads hidden states |
| Atoms at [-3,3] | 14.6% | Too loud, corrupts arithmetic. Use [-0.5,0.5] |
| Q,K-only atoms | 0.4% | WORSE than vanilla. V,O does the work |
| Per-layer dance (loud early, quiet late) | 10.7% | Uniform 0.4 beats any per-layer scheme |
| Cycle multiplier (2x correct, 0.2x wrong) | 11.5% drop | Too aggressive at low accuracy |
| Split generation (atoms off for arithmetic) | 15.3% drop | Disrupts generation flow |
| No decomposition incentive | Cycle 2=6%, final=0% | Without 1/N reward, model one-shots |
| Full LR on controller | Destabilizes | Controller needs conservative LR |
| Routing gradient through Llama | 500x attenuation | Use straight-through estimator |
| clamp + tanh on scales | Dead gradient | Use 0.46 * tanh(x) only |
| Loading collapsed controller weights | Still saturated | Always fresh init or reinit scale head |
| Aggressive augmentation (0.5-2.0x) | Prevents settling | Use gentle 0.8-1.2x |
