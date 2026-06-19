# Cathedral-Port Spec — deep cross-breath memory for the Mycelium engine

**Status:** SPEC, ready to delegate. **DO NOT BUILD until the trigger fires.** Bryce delegated the
timing to the agent (see `memory/project_cathedral_b3_candidate.md`). This ports the perceiver's
"cathedral" (notebook + 512-waist silhouette + π-cycled memory wave) onto the *cells-primary*
Mycelium factor-graph engine, as **deep cross-breath memory** for hierarchical (DAG) deduction.

---

## 0. The trigger (when to build this — not before)

Build ONLY when **B3 Rung-1 shows a deduction-DEPTH cap with a MEMORY signature**:

- **The cap:** per-node accuracy by topological level (the Rung-1 read) FALLS OFF at deep levels
  (e.g. holds at lvl 1–3, collapses at 4–5) — and it is NOT breath-starvation (K=16 ≫ 5 levels,
  so the engine has enough breaths to propagate; the failure is capacity, not time).
- **The MEMORY signature (the discriminator):** lower levels first RESOLVE then DEGRADE as breaths
  continue — the "forgetting" pattern (the perceiver's CE-down/acc-down). I.e. the single
  accumulating cell residual cannot HOLD resolved lower-level deductions while it processes higher
  levels; resolved values get overwritten/washed out. The notebook PINS them.
- **If instead** the cap is "deep levels NEVER resolve at all" (a propagation/structural cap, not
  forgetting) → the cathedral is the WRONG fix; the right move is more breaths (carefully, the AM
  K-graph limit is <28 — see §6) or direct cell↔cell-by-depth, NOT memory. Diagnose the signature
  before building.

This is the measured-cap discipline: relax/add only against evidence, never preemptively.

---

## 1. What the cathedral is (ported, cells-primary)

In the perceiver it stored the *latent* state; here it stores the *cell* (variable-node) state —
a stable memory the deduction can pin resolved intermediates into, beyond the single residual.

1. **Notebook (K-slot deep memory).** A per-graph tensor bank of `NB_SLOTS` slots
   (×`NB_DIM`). Each breath k: the cells WRITE a summary into the slots (accumulate or slot-k), and
   READ from the slots (multi-head cross-attention, cells = query, slots = key/value) → a read
   context added to the persistent cell residual. This gives the engine memory that does NOT decay
   with the residual — deep levels can reference resolved lower-level deductions held stably.
2. **512-dim silhouette waist.** A compressed common-mode of the post-update cell state
   (`W_sil: H→SIL_DIM`, the "silhouette of the dancer") — used as the notebook WRITE source
   (compressed memory is cleaner than the full state; the perceiver finding). Side-channel: it is
   the write source ONLY, never fed back into the deduction directly.
3. **π-cycled memory wave.** An aperiodic per-breath phase stamp `angle = k·π/K_max` applied to the
   notebook WRITE source and the slot-READ QUERY (NEVER the deduction/cell-MP attention — that
   transfer failed 3× on the perceiver). Phase-indexes the slots so memory is ordered across
   breaths/levels (aperiodic memory wave).

---

## 2. Integration (the seam)

In `mycelium/factor_graph_engine.py`, the per-breath cell-side, AFTER the cell residual update
(persist/cell-MP) and BEFORE/around the readout:

```
# gated: FG_CATHEDRAL (default 0 -> whole path skipped, byte-identical)
if FG_CATHEDRAL:
    sil = sil_project(cell_hidden)                      # (B, s_max, SIL_DIM), the silhouette
    nb_storage = nb_write(sil[, π-stamp k], nb_storage) # accumulate into K slots (write_o ZERO-INIT)
    read_ctx   = nb_read(cell_hidden[, π-stamp k], nb_storage)  # (B,s_max,H) (read_o ZERO-INIT -> 0 at t=0)
    cell_hidden = cell_hidden + read_ctx                # deep memory injected into the residual
    cell_hidden = cell_hidden * cell_valid.reshape(...) # re-zero pad
```

- **Default-off byte-identical** (`FG_CATHEDRAL=0` → path skipped; the hard mask + residual is the
  permanent fallback). In the JIT cache key.
- **Bootstrap-safe:** the notebook WRITE and READ output projections are ZERO-INIT → the read
  context is exactly 0 at t=0 → the engine is byte-identical at step 0, then the memory "wakes up"
  (the v120 asymmetry: Q/K/V non-zero so dL/dW_o is live from step 1).
- **Params** (`nb_*`, `sil_*`) added to the optimizer ONLY when `FG_CATHEDRAL=1` (mirror the
  PERCEIVER_CELL_MP / FG_HYP param-set gating — no AdamW grad-is-None when off).
- Substrate: no `dtypes.float32` literal in the JIT; single-kernel reductions; π-stamp as a Python
  float scalar baked at compile (NOT a float32 Tensor literal).

---

## 3. The discipline (hard-won — this failed on the perceiver; do it right)

1. **One mechanism at a time, A/B'd against residual-only on the SAME DAG.** Build/test the
   NOTEBOOK first (the core deep memory). Only if it helps the deep-level accuracy, add the
   SILHOUETTE waist (compressed write), then the π-STAMP. Do NOT bundle all three blind — the
   perceiver bundled and got rim+flat with no attribution.
2. **The bar:** the cathedral must BEAT `FG_CATHEDRAL=0` (residual-only) on the **deep-level**
   accuracy (the measured cap), on held-out, on the SAME circuit corpus. If it ties residual-only,
   it is not earning its keep — drop it.
3. **π touches MEMORY only**, never the deduction/cell-MP/THINK attention (π-rope on the deduction
   query died 3× on the perceiver).
4. **Watch the rim / magnitude:** if the notebook drives instability (the perceiver's rim escape),
   the cell renorm (PERCEIVER_CELL_RENORM analogue) + a clamp are the guards.

---

## 4. What to lift from the perceiver (porting, not from-scratch)

From `mycelium/perceiver_poincare.py` §13 (the cathedral, built but on the retired latent path):
- `_nb_init_storage`, `_nb_write` / `_nb_read` (the K-slot ACCUMULATE notebook + multi-head
  cross-attn read), `_sil_project` (the silhouette waist H→SIL_DIM), `_nb_write_sil` / `_nb_read_pi`
  (the π-stamped silhouette write + π-stamped read query). Adapt the QUERY field from latents to
  the cells (variable nodes); everything else (zero-init output projs, the ACCUMULATE storage,
  the π-stamp) carries over.

---

## 5. Delegation

When the trigger fires, delegate the build to Sonnet (GPU-free build + CPU smoke), staged per §3
(notebook → A/B → silhouette → π-stamp), then fire the A/B training runs (cathedral vs residual-only)
on the circuit DAG at the cap depth. The decisive read: does the cathedral lift the deep-level
accuracy that the residual-only engine stalled on?

---

## 6. Risks / open questions

- **Wrong-cap risk (the main one):** the cathedral fixes FORGETTING (resolved-then-degrade); it does
  NOT fix "never resolves deep" (structural propagation). Diagnose the Rung-1 cap signature first (§0).
- **AM K-graph limit:** K=28 hung the AM driver (device hang); the engine runs at K=16. The notebook
  adds graph size — watch for the hang at K=16+notebook; if it hangs, shrink NB_SLOTS/NB_DIM or the
  read heads (the notebook must fit under the AM-driver JIT-graph limit).
- **MCTS is NOT here:** search (PUCT) is a separate track for SEARCH caps (backtracking — hard
  coloring / SAT), not deduction-depth. Circuits are pure deduction; MCTS would not help them. Do
  not bundle.
- **Generality:** the notebook is per-variable-node + per-graph — domain-agnostic (works for any
  factor graph), consistent with the engine's membership-generic design.
