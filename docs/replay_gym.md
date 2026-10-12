# THE REPLAY GYM — design note (2026-10-11, merge of branch `replay` onto gen-weights)

Written on merging branch `replay` (worktree mycelium-wt8, 6079b855 → 2bab8bc1, forked from
gen-weights at 4859cce3, 2026-10-07) forward onto gen-weights (522fc8e5, 2026-10-11) so THE
REPLAY-TRAINED BREATH (RTB; ledger 2026-10-11 17:21) can be built on top of it. List-don't-build,
per the delegated task: this is the state of the gym as merged, and the gap between it and RTB.

## 1. What a snapshot contains

`ALG_SNAP_AT=<kb>` / `ALG_SNAP_OUT=<path>` (env, read inside `forward()`'s breath loop in
`scripts/phase1_algebra_head.py`) dump, at the **end** of breath `kb` — after that breath's own
`breath_step`, consult-injection (`facts3`/`facts5`), writeback, and wheel updates, i.e. everything
breaths `kb+1..K_B-1` would read — an npz via `_snap_dump`:

- **Top-level**: `B`, `K_B`, `kb_done`, `waist`, `tokmask`, `tree`, `facts3`, `facts5`, `anchor`,
  `amask`, `res_map`, `vst_at` (the per-breath `vst` list up to `kb`).
  `waist` here is `forward()`'s **current** local `waist` tensor at dump time — if the sorting
  room fired (`ALG_SORT` and `"sort0_wq" in p`), `waist` was already rebound to
  `_sort_room()`/`_sort_room_hg()`'s output *before* the bank was built (the mandatory-road law:
  "replaces `waist` outright... every bank() read after this line... closes over this rebound
  tensor"), so the dumped `waist` **is** the re-sorted token tensor — there is no separate
  pre-sort `waist` anywhere in `forward()` to lose. Confirmed empirically by
  `scripts/snap_smoke_sort.py` (THE REPLAY GYM merge gate, part 3): under SRK_241's own env
  (SURF8 + `ALG_SORT=4 ALG_SORT_KEY=1`), the dumped `waist` differs from a by-hand recomputed
  pre-sort `waist_w` projection by a non-trivial margin — it is the sorted tensor, not the raw one.
- **`ctx`** (`_bs_ctx`, minus two excluded live closures — see §4): everything `breath_step`
  reads every breath — `vst`, `res_map`, `ident_tok`, the bus-register constants, and, since the
  owner-key shuffle (`ALG_SORT_KEY=1`), **`tok_key`** (`_sort_tok_key`, the raw per-token 64-d
  owner key computed from the sorted `waist`, before `fat_cur`/`sort_key_gain` touch it) — a
  dict value with no special handling needed; the generic serializer (§4) carries any new key a
  future organ adds the same way, which is exactly why this merge needed no gym-side code change
  for the sorting room.
- **`state`** (`_bs_state`, the full per-breath state dict `breath_step` mutates in place): `cur`,
  **`nb` / `nb_st`** (the notebook, `ALG_NOTEBOOK`/`NB_PERSLOT` — always set by `breath_step`
  whenever notebooks are on, independent of the sorting room; carried through generically,
  confirmed present by the same smoke test), `rb_last`, `rptr_last`, `s4_last`, `rbias_all`,
  `rbias2_all`, `aspan_all`, `ospan_all`, `rcue_all`, `breaths` (the per-breath state list), and,
  when `RINGS`, `m_c`/`anchor`/`cmt_logits`/`x_rel`.

Serialization is generic recursion (`_snap_flatten`/`_snap_unflatten`, ~40 churning keys as of the
original build, more now): dict → `{kind: "dict", keys: [...]}` + one entry per child; list/tuple
similarly; a tinygrad Tensor or ndarray → realized to numpy, stored as an npz array keyed by its
flattened path; bool/int/float/str → inlined in the JSON manifest; anything else **raises**
(`_snap_flatten`'s `TypeError` — "a non-tensor, non-serializable object... must be excluded by the
caller"). This is why the sorting room's new `tok_key` and the notebook needed **zero** gym code
changes to be captured: the serializer doesn't enumerate keys, it walks whatever is there.

**Size**: the original build's estimate was "a few MB per row, ~3-4 MB/row" (2026-10-07, before
the sorting room, hourglass, sine, or the owner-key shuffle existed). `scripts/snap_smoke_sort.py`
re-measures this under the current family env (SRK_241's recipe, kb=2, n=8 rows) and reports
**54 arrays, 26.63 MB total = 3.335 MB/row** — still inside the original ~3-4 MB/row estimate
(THE REPLAY GYM merge gate's part 3, `.cache/replay_gate.log`, 2026-10-11). The extra token-side
tensors (`tok_key` (8,256,64); the notebook `nb`/`nb_st`) did not blow the budget, at least not
under this body (`ALG_SORT_HG=0` — the flat sorting room, not the hourglass's extra level
bookkeeping, which was not exercised by this smoke test and may cost more).

**First caught bug (process note)**: the first run of this smoke test wrote NO snapshot at all,
silently — `forward()` called without `slot_mask` only runs breath 0's grounding read (the
per-breath loop, where `ALG_SNAP_AT`'s hook lives, is gated by `slot_mask is not None` —
`scripts/phase1_algebra_head.py:9220`), exactly THE IDENTITY INSTRUMENT'S BLIND SPOT the ledger
logged *the same day* (2026-10-11 11:16) for a different script. Fixed by building a real
`slot_mask` first (one ungated `forward()` call → `build_slot_masks` → the snapshotting call with
`slot_mask=slot_mask`), the same two-pass pattern `replay_gate_check.py`'s `_final_pass_kwargs`
already uses. Restated here because it is clearly a recurring trap for any NEW read-only script
against this head, not a one-off: **a reader that doesn't pass `slot_mask` has not exercised the
loop, even if it runs without error.**

## 2. How to replay from `kb`, with overrides

`replay(p, snapshot, kb_start, overrides=None)`:
1. Asserts `snapshot["kb_done"] == kb_start` (no silent resume from the wrong breath).
2. Rebuilds `ctx`/`state` from the snapshot (list-valued entries get a **fresh shallow copy** —
   the shallow-copy bug the original build fixed: `dict(state)` alone still shares list objects,
   so a second `replay()` call on the same snapshot would grow them via `breath_step`'s in-place
   appends).
3. Rebuilds `ctx["bank"]` fresh via `_make_bank(p, waist, tokmask, B, sent=sent, tree=tree,
   **bank_kwargs)` — `waist`/`tokmask` default to the snapshot's own, or `overrides["waist"]` /
   `overrides["tokmask"]` when given (this is how an injected chalk block, or any other
   hand-edited token-side tensor, gets into the bank's keys/values: `_make_bank` is never
   recomputed from the snapshot's stale closure). Reattaches `ctx["rot2"]` (always the same
   frozen module-level function, never serialized).
4. Applies `overrides["ctx"]` / `overrides["state"]` (dict merges) on top.
5. Runs `breath_step` for `kb_start+1 .. K_B-1`, replaying the SAME `facts3`/`facts5` injection
   logic `forward()`'s own loop uses (`ALG_WRITEBACK`/`_WHEEL`/the in-graph certificate pass are
   **not** replayed — stated scope limit, unchanged by this merge).
6. Calls the shared `_forward_tail(...)` — after this merge, also threading `dirgold`/`dircue`
   from `overrides["dirgold"]`/`overrides["dircue"]` (both default `None`; see §5 note (a)).

`overrides["chalk3"]`/`["chalk5"]` thread into step 3's `_make_bank` rebuild (the chalkboard's
bank); omit both for the plain bank.

## 3. The decode-exact caveat

Replay matches the straight run's **decoded parse** on every row, not its bits. Max per-head
float delta on the gate's own fixtures: ~9.8e-4 on a census tap, ~1e-5 on the primary heads; a
zero-breath replay (`kb_start = K_B - 1`) still shows `pres` off by ~4e-5 while `args`/`res`/`dig`
are exactly 0 — not a resumption bug, but tinygrad's lazy scheduling re-summing floats in a
different order once a tensor is rematerialized from numpy. **Rule, restated:** a replay verdict
is read with `decode()` (or an equivalent discrete-parse comparison), never `np.array_equal` on
raw head tensors, for anything but a pure engineering gate (where exact equality IS the point —
`replay_gate_check.py`'s own `_compare()` reports both, but its pass/fail line is the
decode-level `n_decode_diff`).

Known gaps, unchanged by this merge (stated, not fixed here): replay never re-derives a consult
live (a snapshot's `facts3`/`facts5`/`cert3`/`cert5`/`rack3`/`rack5` are reused as-is; an
`overrides` change that would alter what a *later* consult concludes leaves those stale);
`ALG_WRITEBACK`/`_WHEEL`/the in-graph certificate pass are not replayed; `ctx["sync"]`/
`["swtick"]` (`ALG_SYNC`/`ALG_SW_TICK`) are refused loudly by `_snap_dump` rather than silently
dropped (neither is armed by any gate body or by SRK_241/RK_241, so this has not bitten anyone
yet, but a future arm of the six-wave tick during a snapshot-based read would need this fixed
first); a snapshot taken under one family env cannot be replayed under a materially different
one (the params `p` must match what trained the checkpoint — never-mix-generations' coordinates).

## 4. What's excluded from the snapshot, and why

`ctx["bank"]` (a live closure over `waist`/`tokmask`/`sent`/`tree` — not data; replay rebuilds it)
and `ctx["rot2"]` (always `rot2_interleaved`, the one frozen module-level function — reattached,
never serialized) are the two explicit exclusions `_snap_dump` makes before flattening. Everything
else in `ctx`/`state` is generic: this is a *feature*, not an oversight — it is exactly what let
this merge carry the sorting room's `tok_key` and confirm the notebook's `nb`/`nb_st` through
unchanged gym code (§1).

## 5. What RTB needs that the gym does not yet have

THE REPLAY-TRAINED BREATH (ledger 2026-10-11 17:21): one breath per training step, with (1) the
breath index/clock phase sampled as the timestep, (2) the state at that breath drawn from a
refreshed replay buffer (breath 0 always on-policy), (3) the solver's feedback for every buffered
snapshot precomputed offline in one map-reduce pass per refresh, (4) the per-breath loss graded
against the same gold as today. Three gaps, listed only (not built here):

(a) **A buffer writer over the diet at every `kb`, not one fixed `kb` per run.** Today
    `ALG_SNAP_AT` is a single integer — one snapshot point per `forward()` call, chosen by the
    caller before the run starts. RTB's buffer needs, for a pass over (a slice of) the diet,
    a snapshot at **every** breath `0..K_B-1` for every row (or a sampled subset), refreshed from
    the *current* body every M steps (on-policy, the exposure-bias fix) — a batched, multi-`kb`
    dump, not today's one-shot single-`kb` hook. (Also needs: a storage/eviction policy — "a few
    MB per row" times every breath times the buffer's row count times its refresh cadence is not
    small; see §1's size note, now bigger than the original estimate.)

(b) **The per-breath loss on a replayed state with a sampled `kb`, as a training step.** The gym's
    `replay()` is a **read** path — it returns `out` (the final heads) for scoring, decode
    comparison, or census taps; it was never wired into `loss_fn`/`do_train`'s optimizer step. RTB
    needs a training step that: samples a breath index as its timestep, resumes from that
    breath's buffered state (not from breath 0), runs the *remaining* loop under TinyJit (today's
    training step JITs the whole `forward()`+`loss_fn`+`backward`+`opt.step()` graph from breath
    0 — resuming mid-loop under the JIT is new plumbing, not a replay() call away), and grades
    against gold the same way the existing per-breath ladder does. This is the largest gap: it is
    a new training-step shape, not an extension of the read-only replay harness.

(c) **The offline feedback pass hook.** Today `facts3`/`facts5`/`cert3`/`cert5`/`rack3`/`rack5`
    are produced by *live* consult calls wired into `forward()`'s own loop (`membrane_rack.py`'s
    `consult3`, called from inside the same process, synchronously, via `alt2_fact_buf`/
    `certifier_bias`/`rack_pack`). RTB's design calls for this to move **out of the training
    step** into a map-reduce pass over the whole buffer, run once per refresh (reusing "the
    solver pipeline's pool" — the registered solver-pipeline multiprocessing work, itself a
    separate prerequisite named in the RTB ledger entry) — i.e. a batch job that annotates every
    buffered snapshot with its solver feedback ahead of time, so the training step only ever
    *reads* a precomputed field, never calls the solver live. No such offline pass exists yet;
    today's consult path is inherently per-step and in-process.

RTB's own prerequisites, named in its ledger entry and outside this gym's scope entirely: the gym
merge (this document), the solver pipeline, and K3 (the three-breath loop-stop schedule).
