# Session memo — 2026-06-24: the generality grail, confirmed at parity

**Authors:** Bryce + Claude · **Branch:** `gen-weights`

## TL;DR
The ECC/neural-BP frontier attempt **failed** (a structural limit, cleanly diagnosed),
which forced a strategic reframe: **stop hunting for a domain where the generic deducer
*beats a specialist* — that search is structurally rigged — and play to the engine's
*proven* strength, generality.** We did, and it paid off immediately: the **weight-side
generality grail (§8.2) is CONFIRMED at parity** — one weight set solves coloring +
circuit + KenKen at **~96–99% of each single-domain level**, with healthy breaths and no
domain domination. Separately, the "multigrid / propagation-reach" thread was probed and
**resolved as *not* the lever** (the ceiling is capacity, not reach) — and an
adversarial-verify pass caught a verdict-logic bug that would have greenlit a multi-week
build on a false premise.

## The arc (one line)
ECC frontier → structural limit → reframe to generality → grail confirmed at parity →
multigrid resolved as capacity-not-reach (verify caught the bug) → transfer test is next.

## ECC / neural-BP — a clean structural negative
Pivoted to ECC as the §8.1 "soft constraint" frontier (deducer = learned BP, K breaths =
K message rounds, `membership` = parity-check H). Built it on BCH(31,16) (fits the 49-cell
grid; the kill-gate confirmed a real convergent BP gap). **It lost 3–13× to classical BP**,
and the diagnosed fixes — per-breath **channel re-injection** (Lucy's-notebook for the
*evidence*) + per-breath **LoRA** (un-tie iterations) — **changed nothing**: the run came
back byte-identical to the un-fixed one, the per-breath ladder collapsed to one-shot.
**Verdict: a generic transformer can't learn BP's hand-crafted iterative box-plus rule;
"fixing" it = becoming Nachmani neural-BP = no generality gain.** (memory:
`project_ecc_first_attempt_loses`)

**The broader pattern (the strategic finding):** every frontier with a *specialized
near-optimal algorithm* — symbolic CSP, soft-MRF, MaxSAT, BP — beats the generic engine.
Near-tautological: a domain *has* a specialist because its structure rewards hand-crafting,
exactly where a generic learner is disadvantaged. The deducer's value was never
"beats a specialist" — it's **generality + parallel deduction**.

## The grail — weight-side generality, CONFIRMED at parity
One shared Pythia L0–L3 backbone, multi-task co-trained (`FG_TASK=multi`,
mix=coloring/circuit/kenken). Final per-domain test cell-acc (after ~2000 steps/domain):

| domain | single-domain | multi-task (ONE weight set) | parity |
|---|---|---|---|
| coloring | ~0.62 | 0.60–0.63 | ~96–100% |
| circuit | ~0.97 | 0.94 | ~97% |
| kenken | ~0.80 | 0.78–0.79 | ~98–99% |

**Design = the two-channels framing realized:** one *unified* spec (s_max=49, n_values=7,
n_factor_types=8) + a **generic inlet** (8 global factor types = the *semantics* channel) +
per-domain `membership` (the *topology* channel) + value-domain masking on the shared
codebook. The backbone is **domain-agnostic**; each domain is carried entirely by membership
+ the generic inlet fed as **input** — exactly §8.2 (constraint semantics as input, not
per-domain weights). Three checks all pass: **(1)** one weight set learns all three;
**(2)** no collapse — ladders form across domains (refine-then-converge, unlike ECC's
one-shot flat); **(3)** no domination — all three climb together. This is the **AlphaFold
wager** (general non-equivariant backbone + the right training > specialized/equivariant).
(memory: `project_multitask_generality_works`)

## Multigrid / propagation-reach — probed, NOT the lever (and a verify catch)
A design relay (the Bombe diagonal board → channeling/multigrid) argued the ceiling might be
a *propagation-reach* limit (K caps deduction-chain length) that multigrid would fix.
Probed it: per-cell error *does* rise with deduction-chain depth and survives the hardness
control (77% persists). The analyze-agent auto-verdicted **"REACH → build multigrid"** — but
the **adversarial-verify pass caught a decision-logic bug** (it ignored the reach-budget for
any positive slope) and the disambiguating evidence: the reach budget is **provably unused**
(max depth 13 ≪ 64-level budget, rook graph diameter 2, 0 cells over budget, no error-cliff
at depth≈K), and the curve **decays geometrically + saturates ~0.6** — the error-*compounding*
(capacity) signature. **Corrected verdict: CAPACITY (per-round fidelity), not reach.**
So multigrid is **not** the lever for the current 7×7 ceiling (cleaner lever = raise
per-round fidelity, e.g. a bigger model). Caveats: multigrid *is* the lever for **>49-cell
scaling** (reach exhausts there); and 6.6% local-prop-insufficient cells need the **search
tier**. (memory: `project_kenken_ceiling_is_capacity`)

## Other findings
- **Equivariance ↔ AlphaFold "equivariance myth":** symmetry lived in *TTA*, not progressive
  resizing; perm-aug (training for equivariance) was refuted — the deducer's non-equivariance
  is the feature. The KenKen volume-collapse re-reads as "deducer is already ~equivariant =
  reasoning." (memory note appended to `project_perm_aug_refuted`)
- **Perf audit (Bryce's "map-reduce gain" gut):** the train step is ~90% GPU-compute, but
  ~10% GPU-idle on serial CPU batch-prep → a prefetch/double-buffer reclaims ~10% (banked as
  a follow-up, not a detour). (memory: `project_trainstep_prefetch_gain`)

## The durable lesson
**Play to the engine's proven strength.** After every specialist domain beat the generic
deducer, the moment we asked it to do the thing it's *built* for — one engine, many factor
graphs — it delivered at parity. And **analyze→adversarial-verify earns its keep**: it
killed the ECC over-claim *and* caught the multigrid verdict-bug, both cheaply.

## Next
- **The transfer test (the §8.2 payoff):** does the shared backbone + generic inlet handle a
  domain it *never trained on* with near-zero retraining? Held-out co-train (zero a domain's
  sampling weight, keep its slots) → few-shot the held-out → does it reach parity far faster
  than from-scratch? *This* is what makes the grail matter.
- Raise the per-domain ceiling (capacity lever: bigger model) — secondary.
- Multigrid — deferred to the >49-cell scaling frontier.

## Artifacts
- Ckpts: `fg_multi_k16` / `fg_multi_k16_cont` (the grail), `fg_kenken_k16_reg`,
  `fg_ecc_*` (the ECC negative).
- Code (`gen-weights`): the ECC port (`ecc_data.py`, engine continuous-embed, `eval_ecc_vs_bp.py`,
  `frontier_ecc_bp_gate.py`), re-injection + per-breath LoRA (engine), `FG_PROFILE` instrumentation,
  `diag_kenken_reach_vs_capacity.py`.
- Memory: `project_{ecc_first_attempt_loses, multitask_generality_works, kenken_ceiling_is_capacity,
  frontier_pivot_ecc, trainstep_prefetch_gain}`.
