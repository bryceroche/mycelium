# Session memo — 2026-06-25/26: capacity scaling + the dual-view/channeling experiment

**Authors:** Bryce + Claude · **Branch:** `gen-weights`

## TL;DR
Two threads, one conclusion. **(1)** Scaling *width* (Pythia-410M h=1024 → Pythia-1.4B
h=2048) **does not lift the KenKen ceiling** — it's a *speed* lever (reaches the same ~0.79
in ~60% of the steps), not a *ceiling* lever. **(2)** Bryce's **dual-view / channeling** idea
(a generality mechanism: primal "what number goes here?" + dual "which cell does this number
belong to?") was built, and once a proper **anchor** made the channeling actually engage, it
*also* came back **speed-not-ceiling**. Together with the prior transfer result, that's the
**fourth "faster climb, same ceiling" sighting** — the KenKen ceiling is robustly
**capacity-bound**, and accelerators (width, transfer, multi-view) don't raise the wall.

## The arc (one line)
width-capacity probe → speed-not-ceiling → strategic pivot to generality → dual-view build →
hardwired channeling didn't bootstrap (confounded null) → explicit-message anchor → channeling
engages but still speed-not-ceiling → bank it.

## Thread 1 — width capacity is a speed lever, not a ceiling lever
Single-domain KenKen, identical recipe (K=16, B=8, v45 reg, curriculum) at h=2048 (Pythia-1.4B
base) vs the h=1024 baseline (0.796 @ 3000):

| global step | h=1024 | h=2048 |
|---|---|---|
| 1000 | 0.720 | 0.737 |
| 1500 | 0.751 | ~0.776 |
| 2400 | ~0.78 | 0.795 |
| 3000 | **0.796** | (plateaued ~0.793) |

h=2048 sat a steady ~+0.02 above h=1024 at every matched step, then asymptoted to the *same*
~0.79 — the shifted-left-same-curve signature. **Verdict: width = ~1.5× training speed-up, same
ceiling.** Keep the grail backbone at h=1024; width is a faster-to-train base, not a higher one.
Bryce's "it's plateauing" call was right; I twice over-read mid-trajectory head-fakes before the
plateau settled it. (memory: `project_kenken_width_capacity_plateau`)

## Interlude — strategic next-move panel
A 5-lens design workflow (adversarial-critiqued) on "if width is tapped, what's the lever?"
converged: **stop chasing the KenKen cell-acc ceiling** (symbolic search owns clean CSPs for
free; a clean +0.02 lands in eval noise) → **play to generality**. Top moves: a CPU-gated
"predicate-by-demonstration" inlet (read an unseen relation from demo tuples) and `s_max`-cap
removal (durable plumbing). Bryce chose to pursue his **dual-view** idea, reframed (correctly)
as a generality mechanism.

## Thread 2 — the dual-view / channeling experiment
**Build:** s_max=98 = 49 primal cells + 49 dual variables `D[v,r]` (= the column of value `v`
in row `r`) + channeling factors (per-row cell↔dual). Needed a new **S-agnostic general
layer-forward** (the oracle `kenken_layer_forward` hard-asserts S==49; nobody had ever run the
general engine at S≠49) — parity-gated byte-identical at S=49 (`max|Δ|=0.000e+00`). Primal half
byte-identical to the single-view encoding → primal solve directly comparable to 0.796 (no
control train needed). Inlet built at 49 (oracle untouched) + zero-padded to 98.

**Run 1 — hardwired channeling mask (no anchor): the channeling was a dead wire.**
`dual_cell_acc` flat at chance (0.16) through 1500 steps; primal = baseline − 0.02. The
attention-bootstrap law: the mask (topology) was correct/hardwired, but the *dynamic,
content-dependent selection within it* (`D[v,r]` must attend to whichever cell currently holds
value `v`) never self-organized. A **confounded** null — can't tell "dual doesn't help" from
"couldn't learn channeling."

**Conceptual clarification (Bryce: does the Poincaré ball handle this?).** Attention has two
layers: the **mask** (static gate, *who may attend* = topology = the ball's job) and the
**selection** (dynamic content-dependent weights = semantics). The channeling failure is the
*selection* layer; the ball generates the *gate* and can't do per-instance routing (and is
blocked on non-partition graphs anyway, and folding semantics into the mask channel is the
refuted C2-death move). Ball and anchor are complementary, not substitutes.

**Run 2 — explicit bidirectional channeling messages (the anchor): it engaged.**
I first proposed a consistency *loss*, then caught (offer-critique) that it's readout-level →
same cold-softmax gradient path → same bootstrap weakness as the dual-gold CE that already
failed. Pivoted to **explicit messages**: compute the cross-view BP messages *deterministically*
each breath from current beliefs, via learned **codebook-like** `E_col`/`E_val` (which bootstrap
from task gradient, unlike pointer attention):
- primal→dual: `msg_d[v,r] = Σ_c P(cell(r,c)=v)·E_col[c]`
- dual→primal: `msg_c[r,c] = Σ_v P(D[v,r]=c)·E_val[v]`

Result: **the channeling engaged — `dual_cell_acc` 0.16 → 0.777.** And the dual view gave a *big
early acceleration* (step 500: **0.631 vs baseline 0.516, +0.115**). **But it converged to the
same ceiling: primal 0.774 vs single-view 0.796 (≈ −0.02), identical to the no-anchor run.** So
whether the channeling is dead (dual 0.16) or thriving (dual 0.78), the converged primal is the
same ~baseline−overhead. The early lead fully eroded.

**Why (the transferable lesson):** the dual is a *lossy re-encoding* of the primal's own soft
beliefs (informationally a subset — same all-different constraints, cages primal-only). In
*symbolic* CP the dual still helps because it enables **different exact propagations**; a *soft
learned* deducer gets no new exact deduction from a re-encoding → no information added. **The
classical dual-view win doesn't transfer to learned-soft BP.** Bryce's "parachute" (30% wrong
dual feedback drags the primal) is the right *erosion* mechanism — but the ceiling is
*capacity*-bound, not noise-bound. (memory: `project_dual_view_channeling_result`)

## The cross-cutting finding: "faster climb, same ceiling" (×4)
Width capacity, the transfer test (prior session), and the dual-view (both dead and thriving
channeling) **all** accelerate the climb and converge to the same ~0.79 KenKen ceiling. The
ceiling is set by **per-round capacity**, not by any of these accelerators. Decision rule going
forward: **don't chase the KenKen single-domain ceiling with accelerators** — it's a fixed wall;
spend effort on generality (where the engine's value actually lives).

## Process notes (durable)
- **analyze→predict held:** the bootstrap confound was flagged *before* building Run 1; it bound
  exactly as predicted. A confounded null is not a finding — the anchored run is.
- **offer-critique paid off twice:** killed the consistency-loss dead-end *before* GPU spend
  (it would have rebuilt the failure); and surfaced the cost/confound honestly at each fork.
- **Bryce's persistence was correct:** "don't give up too easily" pushed past the confounded
  null to the *clean* result. The right amount of stubbornness.

## Artifacts
- Code (`gen-weights`, committed): S-agnostic `factor_graph_layer_forward` + parity test;
  `FactorGraphSpec.{primal_s_max, channel_messages}`; `_channeling_messages` + `E_col`/`E_val`;
  `mycelium/kenken_dual_data.py`; `_build_dual_kenken_task` + `FG_CHANNEL_MSG` + inlet
  zero-pad + primal/dual eval split; `scripts/{test_general_layer_parity, smoke_dual_kenken_encoding}.py`;
  env-overridable `kenken_volume_eval.build_model`.
- Ckpts: `.cache/fg_ckpts/{fg_kk_2k, fg_kk_2k_cont, fg_dual_kenken_chan}/`.
- Memory: `project_{kenken_width_capacity_plateau, dual_view_channeling_result}`.

## Next (open)
- The generality frontier the panel pointed at: **predicate-by-demonstration** (CPU kill-gate
  first) and **`s_max`-cap removal** (the plumbing the dual-view build already half-laid via the
  S-agnostic layer). These are where the engine's *proven* strength (generality) compounds —
  unlike the capacity ceiling, which is a wall.
- Multi-view is parked; if ever revived, the place it could *actually* bind is a
  **non-capacity-bound / sparser-graph** regime where reach genuinely limits (the zero-init
  trust-gate is the ready first move there).
