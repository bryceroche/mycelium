# THE MASK DSL — attention-mask rules as checkable programs (2026-09-04)

*Companion to `docs/mask_head_spec.md` (the organ that emits/selects programs)
and the ledger entry "REGISTERED (gut): THE MASK DSL + THE SUDOKU RECREATION
TEST" (2026-09-04). Neural proposes, symbolic disposes — applied to attention
geometry itself: a DSL program is checkable structure, not a soft bias.*

---

## 0. Provenance and scope

The gut's ruling: the dynamic-masking organ (the mask head) should eventually
EMIT structured masking rules in a small domain-specific language, informed by
(a) the frozen trunk's reading of the problem text / rules-of-the-game, (b) the
alternating neural attention state, (c) the solver. Programs are executed
DETERMINISTICALLY by symbolic machinery. The pinned confirmation (§8): present
the system a sudoku in prose; the DSL pipeline must reproduce the v98
hand-crafted mask EXACTLY.

This document contains: the excavated v98/June-era hand-mask rules (§1, ground
truth with file:line), the language (§3), the execution model (§4), the
excavated rules re-written as DSL programs (§5, verified on paper), the
open-only composition law (§6), the staged mask-head interface (§7), the
recreation test (§8), and the v1 exclusion fence (§9).

---

## 1. The excavated rules (ground truth)

Two eras exist. The **v98 era** hand-built per-head boolean masks per domain.
The **general engine era** (June) replaced hand masks with ONE generic builder
driven by a `(membership, latent_type)` factor description per batch —
byte-identical to the v98 KenKen path by construction. The DSL targets the
membership formulation (it subsumes the hand masks), but the recreation test
(§8) is pinned against the v98 sudoku hand tensor.

Common machinery (all domains): the mask is a per-head boolean ALLOW matrix
converted to an additive bias `(1 - allow) * (-1e4)` added to QK^T scores
before softmax (`mycelium/sudoku.py:242`, `mycelium/kenken.py:561`,
`mycelium/factor_graph_engine.py:405`). Heads are SPECIALIZED: different heads
carry different relations (the per-head split IS the specialization; Q/K/V
weights are shared-shape, the mask differs).

### 1.1 Sudoku (v98 hand mask; static)

`mycelium/sudoku.py:53-104` (`_build_sudoku_attention_masks`), attached
precomputed at `mycelium/sudoku.py:612-614` as `model.sudoku_attn_bias`
(shape `(16, 81, 81)`, frozen, no batch dimension — all 81 cells always valid).

Cell index i in 0..80, row-major:
`row(i) = i // 9`, `col(i) = i % 9`, `box(i) = (i//9//3)*3 + (i%9//3)`.

- heads 0–4 (ROW): `allow(i,j) = 1 iff row(i) == row(j)` (∪ eye; a no-op —
  same-row already contains the diagonal)
- heads 5–9 (COL): `allow(i,j) = 1 iff col(i) == col(j)`
- heads 10–14 (BOX): `allow(i,j) = 1 iff box(i) == box(j)`
- head 15 (GLOBAL): `allow(i,j) = 1` (full)

Head split 5/5/5/1 at `sudoku.py:81-88` (`n_heads*5//16` per group, remainder
global).

### 1.2 KenKen (v98 hand mask; fixed rows/cols + per-batch cages)

Fixed part `mycelium/kenken.py:317-369` (`_build_kenken_fixed_masks`); per-batch
assembly `mycelium/kenken.py:574-622` (`build_kenken_attn_bias`). 7×7 grid,
49 cells, `row(i) = i // 7`, `col(i) = i % 7`.

- heads 0–4 (ROW): `allow(i,j) iff row(i) == row(j)` (∪ eye)
- heads 5–9 (COL): `allow(i,j) iff col(i) == col(j)` (∪ eye)
- heads 10–14 (CAGE): per-instance — `allow(i,j) iff cage(i) == cage(j)`
  (the symmetric cage clique `cage_mask (B,49,49)`), ∪ eye (`kenken.py:603-609`)
- head 15 (GLOBAL): all valid cell-pairs

Validity (per batch, `kenken.py:611-618`): padding keys blocked everywhere
(`allow *= valid_key`); padding QUERY rows forced to self-only (no all-blocked
softmax row). Then `(1-allow)*(-1e4)`. Op-type (add/sub/mul/div) is NEVER a
mask channel (the C2-eliminated v100 failure — `kenken.py:20-24`); cage
arithmetic enters via the verification inlet (`kenken.py:428-483`), not the mask.

### 1.3 The general engine (subsumes all of the above)

`mycelium/factor_masks.py:511-607` (`build_factor_attn_bias`). Inputs:
`membership (B, L, s_max)` — row l is factor l's member-indicator over cells —
and `latent_type (B, L)` — a type id per factor. The ONE rule:

```
m_t        = membership * (latent_type == t)          # type-t factors only
allow_t    = (m_t^T @ m_t) > 0                         # co-membership
i.e.  allow_t(i,j) = 1  iff  ∃ factor f of type t with i ∈ f AND j ∈ f
```

Head allocation `mycelium/factor_masks.py:277-328` (`cell_mp_head_allocation`):
`G = max(1, n_heads//16)` global heads at the END; the remaining `R = H - G`
heads split evenly in contiguous blocks across the T types (KenKen T=3, H=16:
5/5/5 + 1 global — v98's exact layout). Validity three-step
(`factor_masks.py:571-582`): (1) block pad keys, (2) pad-query rows self-only,
(3) SELF-EDGE FIX `allow = max(allow, eye*valid_q)` — a valid cell with no
type-t peer still attends to itself (no-op for sudoku/kenken; load-bearing for
coloring's isolated vertices).

KenKen's membership adapter: `mycelium/factor_graph_engine.py:234-306`
(`make_kenken_factor_batch`) — 7 row latents (type 0), 7 col latents (type 1),
one latent per cage from `cell_cage_id` one-hot (type 2). Driven through the
general builder this is byte-identical to §1.2 (the Step-3 GPU anchor).

Multi-task variant `factor_masks.py:416-508` + the head-allocation fix
`factor_masks.py:331-385` (`native_head_alloc_for_present_types`): allocation is
computed per batch over the PRESENT types only (coloring gets 15 edge-heads,
not 2), threaded as tensors. Canonical global type registry:
`mycelium/factor_inlet.py:72-86` (coloring_edge=0, circuit_and/or/not/xor=1-4,
kenken_row/col/cage=5-7).

The hyperbolic generator (`factor_masks.py:610-996`, `kenken.py:625+`) is a
GEOMETRIC RE-DERIVATION of the same boolean target (~1e-3-identical at t=0),
not a different rule; the DSL targets the boolean semantics.

### 1.4 Graph coloring (general-engine era)

`mycelium/graph_coloring_data.py:379-393`. Each EDGE (u,v) is one factor whose
membership row has exactly two 1s (columns u, v); `latent_type = 0` for real
edges (T=1); pad rows carry the global sentinel (=1). Through the general rule:

- 15 heads (EDGE, via native alloc): `allow(i,j) iff (i,j) is an edge, or i==j`
  (self via the self-edge fix — `(m_0^T m_0)(i,i) = degree(i)`, 0 for isolated
  vertices, hence the fix)
- 1 head (GLOBAL): all valid vertex-pairs

T=1 membership encodes the adjacency matrix EXACTLY
(`graph_coloring_data.py:23-32`, self-checked at `:577-615`).

### 1.5 Boolean circuits (general-engine era)

`mycelium/circuit_data.py:600-613`. Each GATE g is one factor whose membership
row has 1s at `{g} ∪ operands(g)` (the gate's local constraint CLIQUE);
`latent_type[g]` = gate-type index (AND=0, OR=1, NOT=2 [, XOR=3]), T=3 (or 4).

- 5 heads per gate type: `allow(i,j) iff i and j co-occur in some type-t gate
  clique` — a gate attends to its operands AND operands attend to their gate
  (bidirectional; deduction, not dataflow — `circuit_data.py:66-84`)
- 1 head GLOBAL

Hierarchy (leaf → gate → gate-of-gate) is deliberately NOT in the mask — the
mask is flat co-membership; depth is carried by `lvl` labels and the type field.

### 1.6 The current dynamic-mask reflex (parser side; what the DSL replaces)

`scripts/phase1_algebra_head.py:1292-1311` (ALG_MASKRE): the snap adjacency
`_A5` (producer→consumer committed edges) is symmetrized and thresholded —
`allow += (( _A5 + _A5^T ) > 0.5)` — ADDITIVE-OPEN over the first-pass base
mask. Zero parameters, zero dedicated heads: the reflex the mask head organ
replaces (`docs/mask_head_spec.md` §0).

---

## 2. Design constraints (the laws that shaped the language)

1. **Neural proposes, symbolic disposes.** The mask head EMITS (or selects, or
   parameterizes) a program; a deterministic executor evaluates it. A program
   is inspectable, hashable, refusable structure — never a soft bias. Soft
   amplitude (the graded `mb` of mask_head_spec §3) remains the head's own
   additive channel; the DSL defines mask SUPPORT (where attention may go),
   not magnitude.
2. **Open-only over the parser's base mask** (mask_head_spec §3; MASKRE's law
   generalized): where a base mask exists, a program's output may only UNION
   with it — A0's grave stays honored. The executor enforces this (§6); it is
   not left to the program author.
3. **No mask-imitation target, ever** (the Goodhart fence): programs are never
   supervised against hand masks. Training signal is downstream parse/solve
   loss only. The sudoku recreation test (§8) is a MEASUREMENT, not a loss.
4. **Groups come from the parse, not from arithmetic.** The DSL has no index
   math (`i // 9` does not appear in any program). Group-membership relations
   (row/col/box/cage/edge/gate-scope) are INPUT BINDINGS produced by the
   symbolic side from the parsed problem — exactly the `(membership,
   latent_type)` contract the June engine already runs on. The binding theorem
   holds: the graph is frame-free; the DSL reads structure, never surface.
5. **Reuse the proven executor.** `same_group(t)` lowers to the validated
   `build_factor_attn_bias` co-membership rule (§1.3) — the DSL is a FRONT-END
   to code with a GPU parity anchor, not a second mask implementation.
6. **Tiny by charter.** v1 is sets, group relations, pairwise predicates,
   union/intersect, head clauses. Everything else is excluded (§9) until a
   specimen demands it (the rank-never-admit discipline).

---

## 3. The language (v1)

A PROGRAM is a header plus an ordered list of HEAD CLAUSES. It describes one
(H, S, S) boolean allow mask over one entity domain.

```
program    := "program" NAME ":" header clause+
header     := "domain" NAME            # entity space (cells / vertices / slots)
              ["mode" ("define"|"open")]   # default: define (no base mask)
              ["when" "kind" "in" "{" KINDID ("," KINDID)* "}"]   # applicability guard
clause     := "heads" (INT | "*") ":" expr
expr       := term ("|" term)*         # union  (OR)
term       := atom ("&" atom)*         # intersect (AND)
atom       := "same_group" "(" NAME ")"    # co-membership in a named group relation
            | "committed"                  # solver committed-edge set (directed)
            | "flip" "(" atom ")"          # transpose of a relation
            | "self"                       # identity (diagonal)
            | "all"                        # full relation
            | "(" expr ")"
```

Semantics (all atoms evaluate to boolean S×S relations):

- `same_group(g)`: `allow(i,j) = 1 iff ∃ factor f in relation g with i ∈ f and
  j ∈ f`. `g` names a group relation in the bindings — a `(membership,
  type-tag)` family. THE workhorse: every excavated rule is this atom.
- `committed`: `allow(i,j) = 1 iff (i→j)` is in the solver's committed-edge
  set at this breath step (booleanized upstream; confidences are the mask
  head's business, not the DSL's). `flip(committed)` is the transpose;
  `committed | flip(committed)` is the symmetrized reflex.
- `self`, `all`: identity / full. (`self` is usually implicit via the
  executor's self-edge fix, §4; it exists for explicitness in `open` mode.)
- `|`, `&`: elementwise OR / AND of relations.
- `heads n: expr` assigns the next n heads (contiguous, in clause order) the
  mask `expr`. `heads *:` takes the remainder (must be the last clause; this
  reproduces `cell_mp_head_allocation`'s global-block-at-the-end layout).
  Head counts must sum to H exactly (executor refuses otherwise).
- `when kind in {...}`: applicability guard read at SELECTION time (which
  program a batch/kind may use, per atlas kind-id); not evaluated pairwise.

That is the whole language. Grammar fits on a card; every program is a few
lines; there are no loops, no recursion, no numbers except head counts.

---

## 4. Execution model

`eval(program, bindings) -> allow (H, S, S) bool` — a PURE FUNCTION.

Bindings (all produced deterministically by the symbolic side):
- group relations: named `(membership (B?, L_g, S), latent-type tags)` families
  from the parsed problem / board encoding;
- `valid (B?, S)`: entity validity (padding);
- `committed (B?, S, S)`: solver committed edges for this step (may be empty);
- H, S.

Pipeline per head clause:
1. Evaluate `expr` to a boolean (B?, S, S) relation. `same_group(g)` lowers to
   `(m_g^T @ m_g) > 0` — the exact §1.3 rule (implementation reuses
   `mycelium/factor_masks.py` machinery; no second mask codepath).
2. VALIDITY POST-PASS (fixed runtime invariant, NOT programmable — programs
   cannot turn it off): (a) block pad keys, (b) pad-query rows self-only,
   (c) self-edge fix `allow = max(allow, eye*valid_q)`. Byte-for-byte the
   three-step of `factor_masks.py:571-582`.
3. Stack clauses into (H, S, S) per the head counts.
4. Bias conversion `(1 - allow) * (-1e4)` happens where it happens today (the
   engine seam); the DSL's contract ends at the boolean allow tensor.

Purity and caching: the result depends only on (program hash, bindings digest).
Static-group programs (sudoku/kenken rows-cols/coloring/circuits) are
cacheable per instance; `committed` changes per breath step, so dynamic
programs cache per (instance, step). No gradient flows through evaluation
(solver facts enter detached — the two-terminal contract of mask_head_spec §4).

Refusal is fail-closed: a program that does not parse, type-check (unknown
group name, head counts ≠ H, `mode define` where a base mask exists), or bind,
is REFUSED loudly and the base mask / v0 reflex runs unchanged. No silent
fallbacks.

---

## 5. The excavated rules as DSL programs (verified on paper)

Sudoku (target of §8; groups row/col/box from the board encoding):

```
program sudoku_v98:
  domain cells
  heads 5: same_group(row)
  heads 5: same_group(col)
  heads 5: same_group(box)
  heads *: all
```

Paper check vs `_build_sudoku_attention_masks(16)`: `same_group(row)` with the
9 row-groups gives `allow(i,j) iff row(i)==row(j)`, which contains the
diagonal, so the hand mask's `max(same_row, eye)` is matched exactly; likewise
col/box; `heads *` = 1 global full head; all 81 cells valid so the validity
pass is identity. Equal, entry for entry. ✓

KenKen (groups row/col from the grid, cage per instance from `cell_cage_id`):

```
program kenken_v98:
  domain cells
  heads 5: same_group(row)
  heads 5: same_group(col)
  heads 5: same_group(cage)
  heads *: all
```

Paper check vs `build_kenken_attn_bias`: cage co-membership = the symmetric
cage clique; the executor's validity pass reproduces `kenken.py:611-618`
(steps a+b; step c is a documented no-op here). ✓ — and identically it matches
the general engine's `make_kenken_factor_batch` + `build_factor_attn_bias`
path, which is byte-identical to v98 by the Step-3 anchor.

Graph coloring (one group family: the edge factors):

```
program coloring_v98:
  domain vertices
  heads 15: same_group(edge)
  heads *: all
```

Paper check: edge factors have exactly two members, so co-membership =
adjacency; isolated vertices get their diagonal from the validity pass's
self-edge fix (exactly why that step exists, §1.4). 15+1 = the native
allocation fix's layout. ✓

Boolean circuits (three group families, one per gate type):

```
program circuit_v98:
  domain nodes
  heads 5: same_group(and_scope)
  heads 5: same_group(or_scope)
  heads 5: same_group(not_scope)
  heads *: all
```

Paper check: gate factor = `{g} ∪ operands(g)`; co-membership = the
bidirectional gate clique of §1.5. ✓ (XOR band: add `heads n:
same_group(xor_scope)` — the language needs nothing new.)

The current dynamic reflex (parser context — the mask head's floor):

```
program maskre_reflex:
  domain slots
  mode open
  heads *: committed | flip(committed)
```

Paper check: `(_A5 + _A5^T) > 0.5` on booleanized edges = the symmetrized
committed relation, unioned (open-only) with the base mask. ✓

Coverage verdict: all excavated rules are expressible in ≤ 5 lines each, with
NO atom beyond `same_group`/`committed`/`flip`/`all`. Nothing resisted
expression. (The KenKen verification inlet — op/target/size features — is NOT
a mask and stays out of the DSL by the same law that kept op-type out of the
v98 masks.)

---

## 6. Composition with the base mask (open-only law)

Two modes, enforced by the executor:

- `mode define`: the program IS the mask. Legal only where no base mask exists
  (game/CSP boards; the engine's membership-driven seam).
- `mode open`: the program's allow is UNIONED with the supplied base allow:
  `final = base | program`. This is the ONLY composition operator in the
  parser context. There is no `mode replace` and no complement/difference atom
  anywhere in the language, so tightening below the first-pass heuristic mask
  is UNREPRESENTABLE — the open-only law is enforced by the grammar, not by a
  runtime check (the strongest form of the fence; the killed tightening road
  cannot be rebuilt by accident).

---

## 7. The mask head ↔ DSL interface (staged; each rung gated per the bring-up ladder)

The mask head (docs/mask_head_spec.md) sits at the breath_step seam — after
solver ping and atlas consult, before the next gather. Its DSL interface grows
in three gated stages; at every stage its soft bias channel (`mb`, open-only)
continues to exist ALONGSIDE the DSL support mask.

- **v1 — SELECT.** A small REGISTERED PROGRAM LIBRARY (the §5 programs + the
  reflex + `all`-only identity). The head emits a selection over the library
  (conditioned on trunk reading, kinds, solver state); argmax is executed.
  Selection is discrete and logged — every mask is attributable to a named
  program. Gate: selecting the identity/base program = bit-identical baseline
  (rung 1 of the ladder); then smoke / twin / fleet per mask_head_spec §6.
- **v2 — PARAMETERIZE.** Templates with typed holes: `heads ?n:
  same_group(?g)` — the head fills group names (from the parse's group
  vocabulary), head counts, and `when kind` guards. The executor type-checks
  fills; ill-typed fills are refused fail-closed. Gate: v2 with holes filled
  to a v1 program = identical to v1's execution.
- **v3 — EMIT.** Token-by-token program emission over the §3 grammar
  (grammar-constrained decoding; the grammar is small enough to mask illegal
  tokens exactly). Parse + type-check + bind before any execution; refusal
  falls back to the v1 selection. Gate: emitting a library program verbatim =
  identical to v1; then the sudoku recreation test (§8) as the capability bar.

Training at every stage: downstream loss only through the re-masked pass
(two-terminal; solver/atlas inputs detached); no mask-imitation target (the
fence, §2.3); warm birth, zero-init output, gentle continuation — all
inherited verbatim from mask_head_spec §4.

---

## 8. THE SUDOKU RECREATION TEST (pinned confirmation; the pass bar)

**Registered 2026-09-04.** The closing of the circle to v98: the machine
re-deriving, from language, the rules we once wrote by hand.

Fixture: a sudoku presented IN PROSE (rules-of-the-game text: "each row
contains 1–9 exactly once; each column …; each 3×3 box …"). The pipeline —
trunk reads the text, the parse yields the group relations (row/col/box over
the 81-cell board encoding), the mask head selects/emits a program, the
executor evaluates it — must produce a mask satisfying:

```
allow_dsl  == allow_v98            # exact tensor equality (np.array_equal)
bias_dsl   == bias_v98             # after (1-allow)*(-1e4)
```

where `allow_v98`/`bias_v98` are `_build_sudoku_attention_masks(16)` /
`model.sudoku_attn_bias` (`mycelium/sudoku.py:53-104, 612-614`), shape
`(16, 81, 81)`, on the SAME board encoding (81 cells row-major, head order
5 row / 5 col / 5 box / 1 global).

Pass bar: EXACT equality — no tolerance, no "close." The bar is pinned before
any training run exists; it does not bend after. Failure modes are diagnostic
by construction: wrong groups = parse-side fault; wrong program = head fault;
wrong tensor from a right program = executor fault (and the executor is
deterministic, so that one is a plain bug). Honest scope: the test certifies
the STRUCTURE channel (prose → rules → program → mask); it says nothing about
solve accuracy, which the engine's own gates measure.

---

## 9. What v1 excludes (the fence against creep)

Excluded until a named specimen demands admission (rank-never-admit; each
admission is a registered ruling):

- **Negation / complement / set difference** — would make tightening
  representable (violates §6) and no excavated rule needs it.
- **Index arithmetic, ranges, coordinate math** — groups come from the parse
  (§2.4); `i // 9` in a program would smuggle surface geometry past the
  binding theorem.
- **Thresholds, weights, real numbers** — the DSL is boolean support; graded
  confidence is the mask head's soft channel, not program text.
- **Quantifiers beyond `same_group`'s built-in ∃, counting, transitive
  closure / paths** — no specimen; closure especially is a solver capability,
  not a mask rule.
- **Per-head learned parameters inside programs** — the program is structure;
  anything learned lives in the head that selects/emits it.
- **Cross-domain relations (slot↔token in one atom)** — the base mask handles
  the slot↔token geometry today; a mixed-domain atom waits for a specimen.
- **Loops, recursion, definitions/macros** — programs are ≤ a handful of
  clauses; a macro layer is the recursion charter's business (books), not the
  mask's.

The grammar in §3 is the whole of v1. If a rule cannot be written in it, that
is a FINDING to bank (a new atom proposed by specimen), not a reason to bend
the executor.
