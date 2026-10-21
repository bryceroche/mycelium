# Complex Planes — the addressing design (branch `complex-planes`, 2026-10-19)

**Authors:** Bryce + Claude (with Opus's review folded in). **Status:** design. Nothing has been trained, and no bars
have been pinned. The numbers below are zero-GPU toy reads of the addressing math alone (`scripts/toys/complex_addr_*.py`).

## 0. The thesis, credited part by part

Bryce: rebuild the transformer around complex planes. Binding (⊙) and unbinding (⊙ conj) are exact rotations. Composite
keys built from a hierarchy of planes give an effectively unlimited address space that can be searched without loss.

What each ingredient actually buys:

| Claim | Holds? | Where it really comes from |
|---|---|---|
| Exponentially many distinct addresses | **Yes** | The **digit system**: one register per level, each holding one of N symbols, gives N^L addresses. The same count holds for any discrete code. The phases do not create it. |
| Search through them is lossless | **Yes, in layout A** (below) | Cleanup snaps each register to an exact symbol. This is the neural → symbolic handoff in miniature. |
| Robust to an imperfect encoder | **Random noise: yes. Coherent noise: no.** | The phases' **noise margin**: 64 codes spread over 256 planes are nearly orthogonal, so random phase noise averages out. A lean toward a plausible sibling does NOT average out (§2). |
| Values attach to addresses exactly | **Yes** | **Binding**: value ⊙ address is stored, and ⊙ conj(address) returns the value exactly, with zero parameters. |
| "Orthogonality for free" | **No, approximate** | Random unit phasors in C^D overlap by about 1/√D, the same as random real vectors. Real space already has exponentially many nearly orthogonal directions (that is MLP superposition). |
| Many items in one vector | **No: capped** | Superposition crosstalk. The 10-16 clock read measured about 20–30 items per D = 256 register, for every key scheme. |

## 1. Four layouts under random phase noise (`complex_addr_layouts.py`)

D = 256 planes per register; 64 codes per level; 3 levels (262,144 addresses); 500 trials.

| Noise (rad) | **A: one register per level** | B: pure product + resonator | C: weighted prefix sum | D: 26 addresses in one vector |
|---|---|---|---|---|
| 0.0 | **1.000** | 0.096 | 0.992 | 0.964 |
| 0.5 | **1.000** | 0.040 | 0.966 | 0.952 |
| 1.0 | **1.000** | 0.008 | 0.726 | 0.646 |
| 1.5 | **1.000** | 0.000 | 0.242 | 0.230 |

- **B (a pure product L1⊙L2⊙L3 in one vector)** is a lock, not an index. Unbinding the root leaves L2⊙L3, which looks
  random, so there is no partial signal to search by. Reading it is a factorization problem. (The resonator here is a
  quick, untuned one, and 262k combinations at D = 256 is likely past its capacity.)
- **C (Gemini's prefix sum)** is searchable but leaky: the levels share one vector and crosstalk.
- **A** is exact. It searches level by level at 64 comparisons per level (192 in total, not 262,144). The price is
  WIDTH: L × D planes, linear in depth, against N^L addresses.
- **D** shows the storage cap is a separate limit. Address space is lossless; a memory holding many addresses in one
  vector is not.

## 2. Layout A under COHERENT noise (`complex_addr_sibling.py`; Opus's correction)

A real tree: 64 top codes × 8 children × 8 grandchildren = 4,096 valid paths. The encoder leans each level toward a
plausible sibling (another top code at level 1; a sibling under the same parent below) with strength α ~ U(0, αmax),
plus 0.5 rad of random noise. 2,000 trials.

| Sibling lean | independent snaps | greedy descent | **beam-2, joint score** | full joint read over valid paths |
|---|---|---|---|---|
| α ≤ 0.4 | 1.000 | 1.000 | 1.000 | 1.000 |
| α ≤ 0.6 | 0.582 | 0.582 | **0.811** | 0.806 |
| α ≤ 0.8 | 0.236 | 0.236 | **0.506** | 0.486 |
| α ≤ 1.0 | 0.136 | 0.136 | **0.302** | 0.303 |

**Readings:**
1. **Coherent error is a cliff, not a slope.** A level flips exactly when the lean passes the midpoint (α > 0.5); the
   independent column is (1 − p_flip)^3 to three decimals. More planes do not move the cliff, because the margin
   against a blend does not grow with D. The phases protect against random noise only.
2. **Redundancy is the only defense, and the tree supplies it.** A wrong snap at one level is out-voted when the levels
   below it only fit the true parent. Beam-2 with a joint score roughly doubles accuracy in the hard regimes. Greedy
   descent gains nothing over independent snaps: once the top snap is wrong it cannot be undone (the tree's old death).
3. **Beam-2 ≈ the full joint read.** Keeping two candidates per level captures the rescue; scoring every valid path
   adds nothing (the 10-16 tree race's beam-2 result, reproduced).
4. **Implication for the encoder:** what has to be trained is margin against SIBLINGS. That is the twin problem in
   address form: contrastive negatives drawn from the confusable sibling, not random negatives.

## 3. The addressing design

- **One plane group per level.** Each level of the hierarchy owns its own registers (a head group). Superposition is
  never used for the address itself.
- **The MLP is the per-level cleanup.** It soft-snaps each register toward its codebook, keeping the top 2 per level,
  not a hard argmax.
- **Read every register at the first look; the breaths refine.** Do NOT visit one level per breath (that is the
  six-way walk). "Coarse to fine" describes how confidence sharpens across breaths, not the order levels are visited.
  The final pick is held open until the levels agree under the tree's validity.
- **Superposition is reserved for the working set** (≤ ~20 items per register: the entities and quantities in play).
- **Values bind onto addresses** with ⊙ and unbind exactly; the cleanup restores what crosstalk erodes.

## 4. The complex transformer: where the planes go

| Part | Standard | Complex-plane version | Open question |
|---|---|---|---|
| Bridge in | Llama embedding, 2048 real | z = r·e^{iθ}, θ = W_θx, r = softplus(W_r x) | Keep the radius (two-channel law: angle = identity, radius = consolidation). |
| Position | RoPE | Llama already rotates 32 planes per head; ours is RoPE generalized to roles and entities | Share or separate position planes? |
| Score | q·k | Re(qᴴk) = Σ \|q\|\|k\| cos Δθ | Softmax on the real part. |
| Values | weighted sum | weighted phasor sum = superposition | Crosstalk grows with how many tokens a head mixes. |
| Bind / unbind | learned implicitly | ⊙ role / ⊙ conj(role), zero parameters | Which heads unbind which roles? |
| Norm | RMSNorm | normalize the radius, keep the phase | — |
| MLP | SwiGLU | **the cleanup memory** (soft snap to codebook, top 2) or modReLU | The deep surgery. A real nonlinearity applied separately to the real and imaginary parts breaks the phase algebra. |
| Out | linear | unbind by the role's conjugate → codebook match | Readable by construction. |

## 5. The toy ladder (bars to be pinned before any read)

- **Rung 0, synthetic:** toy v1's entity-tracking task (banked arms A / A4 / B / E) plus a NATIVE complex arm.
  Held-out combinations are the test.
- **Rung 1, GSM8K:** tag every token and bind related words. Gold comes mechanically from parse edges, the gold graph's
  quantity roles and the owner. The X-ray: unbind each token by each role and rank its true partner.
- **Controls:** (1) a real twin of equal real parameter count (C^D ≈ R^2D), the same data, learning-rate sweep and
  seeds; (2) a complex arm whose nonlinearity runs separately on the real and imaginary parts (does the phase algebra
  matter, or only the parameterization?).
- CPU only until a rung earns the card.

## 6. What rung 0 changed (2026-10-19, ledger 13:09)

Rung 0 was WEAK (C_clean +0.023 vs A4, ahead at every f). The arm that BROKE the phase algebra (C_split: SiLU on the real
and imaginary parts separately) was the best one, and modReLU hurt. The gain grew as coverage fell (+0.05 to +0.07 at
the f = 0.1 floor) and lived at two hops. The bet, restated (Opus): **weights constrained to respect per-plane rotation
(an equivariance constraint, the family of group-equivariant CNNs) compose better with little data.** Two reads are
in flight: rung 0b (does the gap widen down to the 3.7% floor?) and the width control (at matched parameters the
complex arms are 1.5x WIDER: is it equivariance or width?).

**Both landed (ledger 16:31).** Rung 0b WEAK: at the floor C_split +0.054 and C_clean +0.041 over A4 (A4 loses 0.043
from f = 0.2 to the floor; C_split loses nothing). The width control: **WIDTH EXPLAINS IT** — A4 at width 96 (2.14x the
parameters) holds the floor at 0.906 vs C_split's 0.914 (bar 0.904; margin 0.002). The complex layer's measured gift is
WIDTH AT HALF THE PARAMETERS, a useful weight structure; equivariance and phase algebra are not attributed.

## 7. Rung 0.5: the organs, one ablation each (design; bars are pinned when 0b and the width control land, because
## they decide the base arm: C_split if the constraint earns it, A_wide if width explains it)

Same toy, 48 attributes, the floor cell and f = 0.2; the metric is held-out at hops 2–3 (where composition lives).

| Organ | What it is | The MRI name | Ablation arm |
|---|---|---|---|
| Address registers | R separate complex residual streams (entity / relation / value), each with its own residual; attention writes each through its own output map. Computed from the additive bundle (the fair arm), plus an ORACLE arm where the input writes each field to its register (the ceiling if the encoder separated fields perfectly) | gradient encoding: the field is read off which band it sits in | base + registers |
| Unbinding heads | q = content ⊙ conj(role code), the roles from a fixed phasor codebook, the role choice learned; each head assigned one register | slice selection: excite one band, the rest stay dark | base + unbinding |
| Both | registers + unbinding heads | | base + both |
| k-space mixing (optional) | an FFT token mixer in place of one attention layer (FNet). It mixes POSITIONS, and position is not our problem (Opus): an organ, not a pillar | k-space reconstruction | only if 0.5 earns it |

Rules: parameters matched within 2% per arm; one organ added per arm; bars pinned before training; each organ must
earn ≥ +0.02 at hops 2–3 at the floor to stay.

## 8. Later reads, parked

- **Spin-echo calibration** (Opus): once a trained encoder exists, measure its SYSTEMATIC sibling lean on gold (which
  sibling, how hard), then apply the reverse rotation at read time: a refocusing pulse made from data. It cancels the
  part of the coherent error that recurs; random error cannot be echoed.
- **The independent check** (§2's lesson): an address check only counts if its evidence is decorrelated from the
  encoder's: solver/units and rationale step values (training time); photobooth views detect but do not correct.

## 9. The research line, re-scoped (2026-10-19, Bryce): complex planes are the branch's focus; MATH-500 waits

**What three toys say (v1, rung 0, rung 0b):** a bare complex layer is a tied real layer (2 numbers doing 4's work) and
buys width at half the parameters. None of the three tasks REQUIRED unbinding: width solved them. And none of the arms
had an organ that USES ⊙ conj: the only phase-specific part was the cleanup MLP. The algebra has never been given a
job; it has not yet been tested. "Love" (Bryce) = build the organ that exercises the algebra, then test it fairly.

**The lesson of both earlier task designs:** v1 is solvable by width; v2 (roles from syntax) memorized at toy budget
(i.i.d. val ~0.25 for every arm). The next task sits between them: a feasibility read places it off both the floor and
the ceiling BEFORE any grid.

### Rung 1T — the unbinding task (draft; bars pinned on the word, after the feasibility read)
- **Scene:** K role–filler pairs per record (K in {3, 5}); each role is written with one of 3 alias tokens, the pairs in
  shuffled order. Query "role of X" → the filler. **Multi-hop:** a filler names another record ("role_a of (role_b of
  X)"), hops 1–3. **Held out:** a share of (role, filler) combinations never appears in training (each role and each
  filler is seen; only the combination is new); the aliases of a role are all seen.
- **This is the task FHRR solves by construction:** record = Σ role ⊙ filler; record ⊙ conj(role) → cleanup → filler.
- **Arms (stream width FIXED at 96 reals = 48 planes; the same depth, steps, data, lr sweep, 3 seeds):**
  - **C_clean+U**: the complex net plus UNBINDING HEADS (q = state ⊙ conj(role code), role codes a fixed phasor
    codebook, the role choice learned) and the cleanup MLP. The algebra intact end to end.
  - **C_split+U**: the same organs, the MLP split on Re/Im (the algebra broken inside the layer).
  - **R_width**: a real net at width 96 (about 2x the parameters).
  - **R_params**: a real net at C_clean+U's parameter count.
- **Decision rule (Opus, to pin):** PHASE ALGEBRA EARNS IT = C_clean+U ≥ max(C_split+U, R_width, R_params) + 0.03 on
  held-out hops 2–3. THE FORM IS NULL = C_clean+U ≤ that max + 0.01. Between = WEAK, with ONE pre-registered rescue variant.
  A null closes this FORM (the bare-layer + unbinding-head form), never the line: the next form must name what changed.
- **Feasibility gate (before the bars pin):** a smoke at K = 3 must show i.i.d. val ≥ 0.80 for at least one arm (it
  learns rather than memorizes) and held-out hop 2 ≤ 0.90 for R_width (width does not already solve it).

## 10. Rung 1T: the first trained foothold (2026-10-20, ledger 08:57)

On a task that REQUIRES unbinding (role-filler records, aliases, chains of 1–3 hops, held-out role x filler pairs at
every hop), with the unbinding chain as the only road (form 2: pointer → owner lookup → unbind → soft snap):

| Held-out, hops 2–3 (3 seeds) | |
|---|---|
| **Phase binding, learned write (44.7k params)** | **0.832** |
| Orthogonal role matrices (376k) | 0.767 |
| Phase binding, oracle write | 0.765 |
| Free role matrices (376k) | 0.576 |
| Plain transformer, 6 layers, 1.63M params | 0.566 (hop 3: 0.222; fails hop 3 even in-distribution) |
| Looped plain, 44k | 0.095 |

**Verdict: STRUCTURED LOW-PARAMETER UNITARY BINDING EARNS IT** (+0.065 over orthogonal; bar +0.05). The claim is named
for what it is: a phase rotation is an orthogonal map with very few degrees of freedom, and the fewer degrees of freedom
the binding operator has, the less it can memorize seen pairs, the better it generalizes to unseen ones. Writing is
learnable (the learned entity codes beat fixed phasors). What the history of this line shows: the algebra never failed;
three toys tested equivariance and width, and the first toy that gave the algebra a job (and no bypass) found it.

Open: the owner lookup still degrades on held-out (the memory key carries the bound filler) — form 2b keys the lookup by
owner alone. Then the dictionary supplies real fillers (Llama's features → phasors by random Fourier features).

## 11. Two days on the line (2026-10-20 → 21): from binding to time — the laws, each with its standing

Every number below is in the ledger (`docs/phase1_skeleton_spec.md`, 2026-10-20 11:37 → 2026-10-21 12:03) with its
artifact in `.cache/`. **STANDING** says how much weight a law can bear:
- **BLIND**: bars pinned before any full-length read (a 300-step smoke at most);
- **POST-FEAS**: bars pinned after a seed-0 feasibility read had shown the shape (a three-seed confirmation, not a blind
  prediction);
- **CONSTRUCTION**: passes largely by how the arm is built (the learned part is small);
- **UNPREDICTED**: a finding no pinned prediction asked for;
- **UNTESTED**.

### 11.1 The claims, as they now stand (each narrowed by a measurement)

1. **The phase claim is a statement about binding machines.** Phases beat orthogonal and free matrices
   (K = 3: +0.065; K = 4: ORDER HOLDS, +0.068 / +0.207). But a 46k plain looped block with one soft snap per hop beats
   every binding machine on unseen pairs, in and past the horizon (snap control: hops 4–5 0.965 / 0.980 vs 0.755).
   Phase unbinding is optional in the hybrid.
2. **The clock's job is ADDRESSING, not compression.**
   - Superposed registers TIE per-event slots at equal memory ((who, when) bar c: −0.023).
   - The registers' memory win is real only against equal-memory ATTENTION (+0.450), not against a cache with the same
     interface.
   - Finding by content needs a per-event key in every design: the registers compress values only.
3. **The clock's edge is INTEGRATION, not accuracy (a fit argument, not a measured edge).** Within one turn the wheel
   ties the raw index exactly (race 3 T3), and the odometer is a base-6 counter written in phases. What the clock gives
   is ONE frozen object that addresses the registers, stamps storage and orders events, where a shift is a rotation and
   nothing can drift. That argument belongs to the compatibility ruling (fit builds), not to the bars (bars claim).
4. **'The clock wins the time trial' (Bryce): DOES NOT WIN OUTRIGHT.**
   - Accuracy ties.
   - Parameters are 18× fewer (by construction).
   - Length is +0.083 on short k but **+0.046 on long k against a pinned +0.05**, with full attention holding 16× the
     memory.

### 11.2 The laws

| Law | Evidence | Standing |
|---|---|---|
| **Discretize between steps**: multi-hop extrapolation is carried by a cleanup to a clean symbol after every hop, not by the binding algebra | snap control: TL_snap h5 0.948, TL_snapmem 0.980, vs plain loop 0.300 | BLIND (registered before the full run) |
| **Order past the trained length is a COMPARISON among content candidates, not a score added to content** | races 1–2: every additive order bias fails at 128; race 3: gate, then 'no candidate keyed later' (a product) reads 1.000 to 1024 | 3 races BLIND; the fix itself CONSTRUCTION (the learned part is a clean content gate) |
| **An unbounded position bias swamps the content match** | ALiBi: a sale that occurs ONCE found at 0.529 at 128 and 0.254 at 256, vs 1.000 with no position at all; the slowest wheel alone (a ramp over 0–256) the same, 0.597 / 0.292 | UNPREDICTED (both pinned predictions said ALiBi would tie the clock); race 2 confirmed it, as Opus cautioned |
| **Frozen rates beat learned ones (the Goodhart door, rotor_clock's law)** | the learned slow band sped its periods from 256+ to ~166 until its order horizon equalled the training horizon (~62); frozen gears beat it by +0.213 | BLIND |
| **Bands address, registers store** | bands in one vector lose (CLOCK_ms 0.789, LRU_spread 0.410 on long k); registers routed by a slow band hold 0.999 at 128 | BLIND (a 300-step smoke seen) |
| **Decay chooses which window the capacity covers; it adds no capacity** | the LRU learned decay 0.064 on short k (1.000 at 128), cut it to 0.001 when the task asked far back (bar e), then hit the capacity wall (0.521) | BLIND |
| **One wheel orders for exactly one turn; a locked pass wheel extends it (the odometer)** | one 1296 wheel: two-sale 1.000 at 1024, 0.775 at 1536; with the pass wheel 1.000 at 1536. Read the ANGLE, not the sine (the sine turns back at a quarter turn; Bryce's 'hands moving downward'); gears must be phase-locked so the carries line up | CONSTRUCTION, confirmed (smoke-shaped) |
| **Content lookup needs attention; Fourier filters do position only for SHORT windows** | content: ATT 1.000, gated filter 0.448, plain filter 0.235. The plain filter is perfect on 'k back' for k ≤ 8 (0.989) but reads 0.112 when the query picks k up to 63 | content: POST-FEAS; the short-window limit UNPREDICTED |
| **The router, not the clock, is the length wall** | (who, when): accuracy at 128 tracks the router's anchor hit (0.733 vs 0.731); the oracle tick reads 0.999 | POST-FEAS; decomposed (one sale perfect at every distance; every miss a two-sale order judgment) |
| **The clock carries the read** | scrambled stamps: 0.084, though that arm's router still finds the anchor ~0.7 | POST-FEAS |
| **The rival gets the interface** | plain attention never learns the three-step (who, when) composition in 4000 steps (≤ 0.115); given the router + clock interface, a per-event cache learns it at once (0.997). Same lesson as TL_hop in the depth run | POST-FEAS ('did not learn in 4000 steps', not 'cannot') |
| **A soft content gate leaks with length, (1 − ε)^N; snapping removes it** | the fuzzy arms' gate learned THROUGH σ 1.0 noise (ε → 0), so no leak was induced | **UNTESTED** (not refuted). The smoke also found a second leak the arithmetic missed: N × ε of OWN weight dilutes the anchor |

### 11.3 Earlier reads from the same two days (context)

- The analytic ceiling, extended: 96 planes, K = 3, held-out hops 1–5: 0.993 / 0.982 / 0.964 / 0.949 / 0.951.
- The dictionaries (D4, D7): no single sparse feature carries 'owner' at either depth once syntax and position are
  controlled (the binding theorem's prediction on features).
- FNet fails lookup (0.352).
- The tree rung passed B1–B5:
  - order needs per-level codes;
  - a clock carries order as a frequency, not as an offset;
  - path sums route coarse queries to unseen leaves.
- Correction banked on 2026-10-20: the rung-1T line 'beats plain networks up to 36× larger' (§10) holds only for plain
  nets WITHOUT the per-hop interface.

### 11.4 Open, in order

1. **The fuzzy-gate rung, with irreducible and KNOWN ambiguity** (Opus's design):
   - decoy actions share surface tokens with the queried one;
   - a fraction ε of events carry deliberately ambiguous labels, so even a perfect gate stays soft at ε;
   - sweep ε ∈ {0.001, 0.01, 0.05};
   - the (1 − ε)^N decay and the snap's flattening are checked against a number set in advance.

   This is the version the main line meets: 'sold', 'the sale', 'she got $12 for'.
2. The hybrid as one machine: learned lookup + snap + static memory, with the clock as address, storage stamp and order key.
3. Scope fence: the toys test GIVEN events. The main line's (who, when) failure was DERIVED values (0.013 addressable),
   which have no event of their own; nothing here touches it yet.
4. **HYPOTHESIS (Opus, 10-21; not a run): the fence has a door: the breath stamp as the derived value's address.** T6
   never stamped tokens; it stamps BREATHS, where derived values are born. So 'the result of step j' (the address the main
   line's census found missing: derived values 0.013 addressable) may be (breath tick, slot).
   - **Prior art on the main line:** the ledger already names this address. The (who, when) census (2026-10-15) read the
     planning address as 'mention pointer | derivation register keyed by the producing step'. The breath stamp is a
     candidate IMPLEMENTATION of that step key, not a new address.
   - **Fence 1, the forward requirement:** the gold graph is declarative and solved BACKWARD. The backward-unknown census
     found a third kind of lookup (0.256 'seen-arg' + first-here), so a step key exists only for a FORWARD-planning arm
     (or forward-rewritten gold); the pointer | register pair reached 0.744 of wild lookups.
   - **Fence 2, ticks are not unique:** K3 is three breaths and T6 has six ticks; a problem can have more derivation
     steps than ticks, and one breath can produce several values. The address is (tick, slot), plus a pass wheel if
     steps outrun the ticks: the odometer again.
   - **Toy form, when its turn comes:** some events are COMPUTED (f of two earlier events) and stamped with their producing
     step rather than a token position; does gate-then-compare find them?

Footnote: 'average the vectors, not the angles' (Opus, an engineering note) is the identity behind the router's soft stamp
Σ α_t c(t): a circular mean is taken over phasors, never over raw angles. It was never tested; it is not a law.
