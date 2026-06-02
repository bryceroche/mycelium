# The Closed Feedback Loop: Seven Components (Archived)

This is the canonical agent-facing description of the v1-v95 era
breathing transformer, lifted from the pre-2026-05-29 `CLAUDE.md` §2.
Preserved here for historical reference.

The machinery described below — Controller, Notebook, LookupTable,
controller-emitted gate / temperature / step_mult, π-cycled per-breath
RoPE — survives in `mycelium/breathing.py` and `mycelium/controller.py`
but is NOT called by current training paths
(`scripts/v98_sudoku_*.sh`, `scripts/v100_factor_graph_*.sh`,
`scripts/v105_1_2_*.sh`, `scripts/v107_factor_graph_*.sh`).

The current architecture is documented in `CLAUDE.md` §2 and `README.md`
§2. This archive captures the design vision that motivated the codebase
through May 2026.

Archived 2026-06-01.

---

## The Seven Components of the Closed Feedback Loop

The breathing transformer's reasoning is an **irreducible closed feedback loop** of seven components. Each amplifies the others; remove any one and the system degrades. As of 2026-05-10, all seven are implemented and wired together — this is the architecture's first complete realization.

| # | Component | What it does | Why it's necessary | Implementation |
|---|---|---|---|---|
| 1 | **Rotation** (π-cycled RoPE) | Per-head phase offsets rotate the attention geometry each breath, so each loop sees the problem from a different angle | Provides **independent observations**. Without it, every breath sees the same view and integration accumulates redundant info. Geometric, not learned — gradient descent cannot erase it. | `breathing.py: RoPE` |
| 2 | **Integration** | Gated running integral across breaths; controller-emitted gate weights novel observations high, redundant ones low | Makes observations **cumulative**. Rotation without integration is amnesia — each breath's insight is forgotten when the next begins. Bayesian evidence combination over independent angles. | `breathing.py: BreathingBlock.breathe` |
| 3 | **Notebook** | 512d pages written after each breath, persisting across both inner loops and outer execution cycles; tree-structured attention over ancestors/siblings/children | Provides **memory across breaths and cycles**. Without it the controller has no basis for comparing "now" against "three breaths ago" — cannot detect convergence or track factorization evolution. | `controller.py: Notebook` |
| 4 | **Lookup table** | 16×1024 cosine matcher storing prime operations (add/sub/mul/div/fraction/compare/combine/sequential...) each with pattern, resonant angle, subtraction mask, confidence threshold; plus a 16×16 coupling matrix | Provides the **reference library and target map**. Without it the controller adapts on energy signals alone ("something is changing") without knowing **what** to look for. Transforms blind search into guided search. Empirically validated 2026-05-10: trained 16×1024 table hits 100% op classification. | `lookup_table.py: LookupTable` |
| 5 | **Controller** | ~40M conductor thinking in 512d. State reader (Perceiver, 1024d→512d) + notebook attention + decision heads emitting `{temperature, gate, stop_logit, step_mult}`. Trained by REINFORCE + lookup-CE + stop calibration on a **separate optimizer**. | Provides **adaptive feedback**. Without it rotation is uniform, temperature is fixed, stopping is arbitrary. The intelligence of the loop. | `controller.py: Controller` |
| 6 | **Temperature modulation** | Controller scalar × sine baseline. Warm = broad/coarse attention; cool = sharp/fine attention. | Controls **resolution at each angle**. Without it every breath has the same precision — wastes early breaths on unnecessary precision and starves late breaths of needed precision. | `breathing.py: BreathingLayer (temp_mult)` |
| 7 | **Step size / rotation rate** | Controller emits `step_mult` adjusting the π/max_loops baseline step | Determines **spectral coverage efficiency**. Nyquist: to resolve two primes separated by Δθ, the rotation step must be ≤ Δθ/2. Too large misses closely-spaced modes; too small wastes breaths. | `breathing.py: breathe_controlled` |

### The Loop, End-to-End

The closed loop is invoked via `model.breathe_controlled(tokens, max_loops, notebook)`. Per breath:

1. Controller reads the running integrated rep
2. → writes a 512d page into the notebook
3. → notebook attention refines the page over all prior pages
4. → lookup table matches the page against its 16 prime entries, returning match weights + confidence
5. → decision heads emit `{temperature, gate, stop_logit, step_mult}` for the next breath
6. → next breath rotates at the controller's adaptive phase, runs at the controller's temperature, integrates weighted by the controller's gate

The loop **terminates** when both: (a) the integral has stabilized (Lyapunov criterion — new breaths add negligible information), and (b) the spectral residual is noise (all significant primes identified and subtracted).

### Gradient Separation Is Enforced by Construction

**The controller's gradient NEVER flows through the transformer.** Three days of v1-v3 evidence proved any such path collapses to one basin.

- `model.parameters()` returns transformer + lookup_table params only. Trained on main CE + small joint lookup-aux CE.
- `controller_train_step` uses a separate optimizer over `model.controller_parameters()`. Loss is per-breath lookup-CE + stop calibration. Gradients that reach transformer params are discarded by the next `main_opt.zero_grad()`.
- Verified on a 5-step joint smoke: 0/39 transformer params changed in controller training; 61/62 controller params changed.

**Do not introduce any code path that lets controller gradients reach transformer weights.** This was the single most important architectural rule of the v1-v95 era. It is currently moot — no Controller is in the active code path — but if a Controller is ever reintroduced the rule applies.

---

## What this design solved (and what it left unsolved)

The seven-component framing was the architecture's design vision through
May 2026. Each component was empirically validated to function in isolation
(see `feedback_attention_bootstrap_principle.md`, `project_lookup_table_validated.md`,
`project_controller_instrumentation.md`).

What ablations established by 2026-05-27 (see also `empirical_v45_to_v95.md`):

| Component | Status | Load-bearing? |
|---|---|---|
| Rotation (per-head π-cycled) | Validated | YES (−73 pts if ablated; only clearly load-bearing piece) |
| Integration (gated running integral) | Validated | Decorative at converged ARITH_HARD; untested in multi-step |
| Notebook | Validated | Decorative on single-cycle data |
| Lookup table | 100% op classification | Useful as supervised target, decorative as live signal |
| Controller (temperature/gate/step_mult/stop) | Validated | Learned `f(breath_idx)` not `f(rep)` — open-loop schedule, problem-blind |
| Temperature modulation (sine baseline 2.0→0.7) | Validated | Load-bearing for warm-start stability |
| Step size (controller-emitted) | Validated | Decorative (no problem-dependent signal) |

The empirical reality was that **only rotation + sine temperature** were
load-bearing. The controller and the rest of the closed-loop machinery
were correctly wired but didn't earn their gradient — they remained
scaffolding through the GSM8K push (v45-v95).

The v98 Sudoku pivot replaced this design with a simpler, working one:
iterative shared-weight prefill + structured per-head attention masks +
per-breath weighted CE supervision + delta_gate + calibration head +
variant codebook readout. The current architecture is documented in the
top-level `CLAUDE.md` §2 and `README.md` §2.
