# The Rotational Binding Bus — Architecture & Diagnostic Contract

**Status:** design center of the architecture (Bryce's ruling, 2026-08-28:
"build everything around it"). Research lineage; the deployed stack
(conductor_v2) is untouched. Ledger entries 2026-08-28+ carry the verdicts,
bars, and registrations; this doc is the standing contract.

## 0. THE COMPATIBILITY RULING (2026-08-29, Bryce — governs all selection)

**We do not select the highest-scoring component from each run; we select
the most COMPATIBLE component family.** A component that scores higher but
does not fit the long-term architecture is not chosen. Measurement
discipline is unchanged — bars still pin before fires, verdicts still bank
honestly, kills still kill CLAIMS — but what gets BUILT FORWARD is governed
by fit with the target:

**THE THREE-ROTOR STACK (the canon):**
1. **TOKEN ROTOR** — base trunk RoPE (frozen, vanilla): text-space
   coordinates.
2. **BREATH ROTOR** — the T^3 torus clock: the six helical waves AS THE
   CLOCK, with pi-cycled rotation turning the head's attention space in
   sync with it (multiplicative — geometry, not bias).
3. **BUS ROTOR** — the T^256-rotation bus (256 planes, D_real=512),
   phasor role-binding as an ACTIVE RECURRENT LOOP (upstream layer
   writing/reading bound relations across breaths), not a read-out wire.

Corollary: a verdict like S1's (refined tap killed on the wild bar) closes
an INCUMBENCY CONTEST, not a component family — the refined tap returns
wherever the recurrent loop needs it, carrying its banked caveat (the
two-tap law: preserve the invariant breath-0 read).

## 1. Context & motivation

- **The breakthrough:** relational wiring between slots is not stored as
  static spatial positions (four position-probe designs died there) but as
  **phase rotations** in high-dimensional complex space
  (C^128 ≅ R^256, interleaved-real; FHRR).
- **Key empirical invariance:** the bus is the campaign's first
  **register-invariant** instrument — recovery holds across mint and wild,
  and v3 took the first outright wild crossover (`res` 72.0% vs the pointer
  incumbent's 69.2% on the 143 golds; wild res *exceeds* mint res — the
  strong form).
- **The move to v5:** one monolithic head → **4 dedicated role heads**
  (`arg1`, `arg2`, `res`, `op`) with **binding by construction**
  (structural-entry law: binding enters as structure, never as learned
  behavior).

## 2. The v5 architectural contract

The network does not learn how to bind; it only identifies content.
Binding is an architectural invariant it cannot violate.

| Layer | Responsibility |
|---|---|
| Neural (4 heads) | "Which entity is this?" → logits over codebook \|C\| |
| Codebook C | unit-L2 random complex codes (row norm 1; per-plane modulus varies — truth-maintained 2026-08-29), frozen |
| Structural (phasors) | "Where does it belong?" → fixed rotation e^{iθ_r}, frozen |
| Wire (superposition) | "Transmit relation" → Z_wire = Σ_r v_r ⊙ e^{iθ_r} |
| Cleanup (today) | cosine argmax vs codebook (`bind_read.py`) |
| Cleanup (registered future) | NL-Atlas Mahalanobis resolver — collisions become geometry |

- **Soft-pointer formulation:** head_r outputs logits ℓ_r ∈ R^|C|;
  p_r = softmax(ℓ_r); content vector v_r = Σ_k p_r(k)·c_k. The output space
  is bounded to the codebook hull; CE on ℓ_r is the native loss
  (v3's role-factored objective, now per-head).
- **Zero parameter interference:** four disjoint heads end hidden-capacity
  starvation (the v3 signature: `op` at .93 while args starved at ~.57).
  Precision (ledger addendum): with the native per-role CE there is literally
  zero cross-role gradient; any wire-level aux loss reaches all heads
  through the sum — parameters stay disjoint, gradient traffic is
  conditional on wire-loss.
- **The single-wire fence (NON-NEGOTIABLE):** the four bound vectors SUM
  onto one 256-float wire. Four separate outputs with no superposition is
  pointer heads in a trenchcoat — there is no bus without the wire.
- **Dimensionality convention:** D_real = 256 floats ≡ P_planes = 128
  complex FHRR channels. Code names these explicitly (D_real/P_planes;
  `complex_tensor.py` carries P = D//2).

## 3. The emergent diagnostic: phasor modulus A_r = ‖v_r‖

Codewords are unit-modulus, so a spread softmax suffers destructive phase
interference: ‖Σ_k p_r(k)·c_k‖ ≤ 1, with equality only at one-hot.
An uncertain head writes **dimmer** onto the wire.

- **Phase / amplitude separation:** phase θ_r encodes structural role
  (deterministic, frozen); amplitude ‖v_r‖ is an emergent, uncalibrated
  proxy for pointer concentration.
- **Why it may beat entropy (the honest subtlety):** given the FULL
  distribution p_r, A_r is deterministic — it adds nothing. The audit
  conditions on the scalar H(p_r): amplitude weights uncertainty by
  code-space geometry (mass split across *nearby* codes interferes less
  than mass split across *distant* codes), so A_r is a geometry-aware
  confidence scalar where entropy is geometry-blind. That difference is
  the entire hypothesis.
- **Pre-registered audit:** I(correct; A_r | H(p_r)) > 0 on banked reads.
- **THE GOODHART FENCE:** amplitude is strictly observational — NEVER
  supervised in any loss (a monitored signal in the loss learns concealment,
  not cure). Register with `diagnostic_register.py` before first use.
- **Downstream decoupling:** cleanup normalizes before resolution; the
  discrete pipeline never depends on amplitude.

## 4. Substrate decision rule (awaiting bindbus-joint)

Bars pinned 2026-08-28 before the fire:

- **Val ≥ 0.58 AND the wild res crossover survives (≥ parity with pointers)
  → v5 trains on the JOINT substrate** (the waist has plasticity to meet
  the rotational coordinates without the parse paying).
- **Otherwise → v5 trains FROZEN-PARSE** (the parse geometry is
  crystallized; the 4 soft-pointer heads train as a pure classification
  interface). Val < 0.57 additionally KILLS the joint line outright.
- Constitutional regardless of branch: **the trunk is frozen forever** —
  "frozen parse" refers to the trained head's parse parameters, never to
  the trunk, which no branch of this rule may touch.

## 5. THE ROADMAP (2026-08-29, endorsed; patience ruling attached)

**Governance:** the December deadline is SOFT — extendable at any time
(Bryce's ruling, 2026-08-29). Patience over pressure: stages fire when
their prerequisites are honestly met, never to meet a date. One dial per
fire; cont-controls; bars pin before every measurement; research lineage
only (the deployed stack moves solely by battery + manifest).

**Stage order (each stage holds for its own word):**

- **[burning] v7b — bus width to the canon's T^256** (256 planes,
  D_real=512, square W_bind2). Compatibility note: a bar miss iterates
  WITHIN the T^256 family (longer burn, richer diet), never back to 128.
- **[next, recommended] S2 — the bus rotor goes RECURRENT (v0):** at each
  breath k>=1, wire computed from the REFINED state (the shelved T7
  machinery returns here, per the ruling's corollary), per-role
  demodulated reads injected into breath k+1's queries (the notebook's
  port); external emission keeps reading breath-0 (the two-tap law as a
  design constraint — the invariant wire survives). Zero-init gate at
  entry (structure enters at zero, always). Bars: parse val vs natural
  continuation; wild reads not degraded. Why first: the regularizer
  effect (4-for-4) suggests binding pressure inside the loop presses
  harder; the conditioned-recirculation law is satisfied (parser-state
  recirculates, not frozen-layer blur).
- **[after S2] S3 — the breath rotor goes MULTIPLICATIVE:** pi-cycled
  rotation of the bank's Q/K per breath, six-wave clock as the schedule
  (geometry, not bias — the alpha null is scope-tagged to additive
  entry). Sequenced after S2 so rotation has recirculating structure to
  act on. Cont-controlled pair.
- **[then] THE COMPOSED STACK** — S2+S3 together: the clock turns the
  attention while the bus carries relations through breath-time over the
  frozen token rotor. Judged on the two campaign curves (parse val
  trajectory; the wild gap), not a single bar.
- **[parallel, CPU/Sonnet lane] the substrate road:** Book 3 round 7
  (~184 L3 rows + the retry shelf) — every rotor verdict this week named
  wild competence as the ceiling; the books feed it directly.
- **[the long line] rotor stack matured -> bus crosses the door's
  reopening bar (wild args >= 0.75, res >= 0.85) -> stage-1b rerun with a
  competent witness -> emission door net-positive -> MATH-500 measurement
  cadence on the true chain.**

## 6. THE DUAL-TERMINAL CONTRACT (2026-08-31 — the neuro-symbolic training law)

Two equations govern every symbolic organ in the loop:
dL_downstream/dp |_solver-path = 0  (detach + zero-grad comparisons:
gradient descent is mathematically barred from modifying or gaming the
deduction rules — no STE, no surrogate-gradient deformation)
dL_bind/dp != 0  (the wire-maker's SECOND TERMINAL: the emission CE
independently anchors proposal quality; without it the solver starves on
drift while the optimizer feels nothing — the two-terminal law's shadow).

Taxonomy (our position among hybrid designs): (1) hard snap, no upstream
loss -> upstream drifts, solver starves; (2) hard snap + STE -> the
backward pass lies about the forward pass (surrogate bias); (3) OURS —
dual-terminal independence: exact discrete deduction forward, decoupled
credit backward, proposal quality anchored by an independent loss.

Precision for claims: the runtime story is STATIC-GRAPH CAPTURE (the
micro-solver is fixed dataflow — comparisons, one-hot matmuls, unrolled
sweeps — compiled into the same TinyJit kernels; values vary, structure
never); the interface story is "representations exchanged forward,
credit decoupled backward."

THE COMPLETION (ours): the contract cuts both ways — the gradient cannot
deform the rules, AND the rules cannot compel their own use (gates close
freely; the four preference kills are that freedom exercised). The symbol
is protected from the gradient; the gradient is protected from the
symbol; USE IS EARNED, NEVER ENFORCED. Rhymes with the Goodhart fence
for the same reason: a loss cannot corrupt what it cannot reach.
