# Silent-organ forensic audit — the FED MIND lineage (2026-09-06)

**Trigger:** four wild records in four days (port242 0.2540 -> mhr2ctl242
0.2565 -> fedon242 0.2613), but the gains are incremental, not breakout,
against a ~2.5x capacity package (11.81M -> 17.17M trained). Campaign's
most-repeated bug class is ORGANS PRESENT BUT SILENT (the unwired clock,
computed-and-discarded signals, the vst discovery). This audit reads the
FED MIND patch (`scripts/apply_fed_mind.py`), the deployed head
(`scripts/phase1_algebra_head.py`), and the champion checkpoints
(`.cache/sharp_fedon242{,_s4000,_s8000}.safetensors`,
`.cache/sharp_mhr2ctl242.safetensors`, `.cache/sharp_port242.safetensors`)
for a mismatch between what the ledger claims is built and what is
actually awake and wired into the loop.

**Scope note:** fedon242 is a RESEARCH checkpoint (alt2/notebook/bindbus/
mask-head lineage). The deployed manifest (`.cache/GENERATION.json`,
gen_id 41, `g41_onemass_refold.safetensors`) is a different, older
lineage entirely — nothing here touches the deployed stack.

---

## RANKED VERDICT TABLE

| Rank | Organ | File:line | Verdict | Evidence |
|---|---|---|---|---|
| **0** | **The entire committed-facts/mask-head/gate pathway, on WILD text, under the capability meter** | `phase1_algebra_head.py:1570-1602` (the seal), Track D below | **SMOKING GUN — measured, not inferred: fedon242 scores 0.0000/2051 on wild holdout when the residual is severed and the parse is forced to survive on committed facts alone (vs 0.2613 open); post-severance commit buffer captures 6 facts vs 1067 open.** | GPU-measured this session (`step_engine_read.py`, SE_R=1, wild holdout). No prior ledger entry appears to have measured sealed-mode on the WILD register specifically — the one pre-registered wild capability bar was left PENDING (`docs/phase1_skeleton_spec.md:32931-2941`), and every banked sealed-mode number is on the val/mint fixture, where the ratio is a mild ~94.5% (`docs/phase1_skeleton_spec.md:31773-1789`). See Track D for the full argument and the harness sanity-check (pass-0 numbers are bit-identical between the two runs, confirming the divergence is real, not a broken invocation). This is the pattern the brief's gut hunt was named for, at the scale of a whole subsystem rather than one gate. |
| 1 | **`mh_atlas_w`** (mask-head atlas-page door) | `phase1_algebra_head.py:1771` (use), `:2934` (gate) | **SMOKING GUN — structurally inert** | Input (`ctx["mh_atlas_traj"]`) is populated only under `ALG_MH_ATLAS=1` (never set in the champion recipe); with a zero input the matrix receives EXACTLY zero gradient (proven by the chain rule, not just observed). Empirically confirmed: `mh_atlas_w` norm 22.498(s4000) -> 22.451(s8000) -> 22.441(s12000), max abs diff s4000-vs-s12000 = 0.0005 (pure weight-decay drift, not learning); fedon242 vs the independently-trained mhr2ctl242 differ by max 0.0006 despite ~different training runs — both are just decaying the same random init. |
| 2 | **FED item 2, pointer multi-form gains** (`fed_pf_args_g`, `fed_pf_res_g`, `fed_pf_query_g`) | `_fed_pf` def `phase1_algebra_head.py:1349`; call sites `:1423,1429,1434` (args/res/query/y — dig2 at `:1431`) | **SMOKING GUN — asleep, and it's the single biggest new-capacity line item (2.36M params, 46% of the +5.36M FED package)** | Snapshot trajectory across s4000/s8000/s12000 is SIGN-INCOHERENT noise, not a trend: `args_g` = [-1e-6,+4.7e-4,+7e-5] -> [+1.5e-4,+5.8e-4,-4.0e-4] -> [+2.9e-4,-7.6e-4,+9.3e-4]; `query_g` similarly flips sign on 2 of 3 components between every pair of snapshots. Final norms are ~0.001-0.002 — three orders of magnitude below the mixer gate (`fed_mx_hg`, norm 0.065) or the waist door (`fed_w2b`, norm 1.49). Since `W_f` (the bilinear form matrices) are randomly initialized, not zero, the ONLY amplitude control for these forms is the gain — there is no paired output matrix to "carry the amplitude instead" the way there is for the FFN extension. Genuinely present, genuinely wired, genuinely never learned anything in 12k steps. |
| 3 | **FED item 9, shelf lane-2 gain** (`fed_nb_g`) | `phase1_algebra_head.py:597` (read), build `~1153` | **SMOKING GUN — asleep** | -0.000363 (s4000) -> +0.000636 (s8000) -> +0.0004 (s12000): sign flips, magnitude never exceeds 6e-4. Same "gain is the only amplitude channel" situation as item 2 (`fed_sil2`, the lane-2 ink matrix itself, is randomly initialized at full scale — norm 22.8 — so it's not sitting idle for lack of content; the READ gate that would let that content influence anything is the one asleep). 0.26M params along for the ride. |
| 4 | **Scratch slot embeddings** (`fq[24:32]`, FED item 6) | `phase1_algebra_head.py:414` (build), read-back at `:689-703` | **SUSPICIOUS — door open, content unspecialized** | The read-back gate is `fed_mx_hg` (shared with item 1), which IS awake (see rank 6). But the 8 scratch row embeddings themselves have barely moved from init: norm 1.311(s4000)->1.320(s12000), and DIRECTION has rotated only 1.1% (cosine 0.989 s4000-vs-s12000, effectively flat after the first 4k steps). The channel is open but has nothing distinctive to say yet — half-built organ, not silent but not really speaking either. Worth another rung of continuation before crediting it. |
| 5 | **`mh_gain`** (mask head master scalar) | `phase1_algebra_head.py:1801` | **SUSPICIOUS — non-monotonic, but not proven dead** | Ajar-init 0.02; ended at 0.00108 (fedon242) / 0.00205 (mhr2ctl242) — both lineages independently shrank it ~10-20x, a real and reproduced signal. But the within-lineage trajectory is NOT monotonic: 0.000876(s4000) -> 0.000025(s8000, a 35x dip) -> 0.001084(s12000, back up). The two zero-init doors it gates (`mh_wo` norm 1.27, `mh_headmix` norm 0.0045) are both clearly awake, so the organ overall is alive; the master gain's own noisy wobble close to zero is either a co-adapted rescaling (wo growing while gain shrinks, net signal roughly constant) or evidence the optimizer hasn't settled this particular knob. Flagging for a follow-up read of the *product* (gain × raw pre-activation magnitude) rather than the gain alone, before ruling either way. |
| 6 | **`fed_mx_hg`** (mixer multi-head gains, FED item 1) | `phase1_algebra_head.py:701` | **WOKE** | All 8 components consistently negative and growing in lockstep across snapshots: norm 0.052 -> 0.067 -> 0.065. Genuine, reproducible learning signal. |
| 7 | **`fed_w2b`** (waist residual output door, FED item 3) | `phase1_algebra_head.py:445` | **WOKE** | Matrix norm grows monotonically 1.256 -> 1.425 -> 1.494 across all three snapshots — clean learning curve. |
| 8 | **`fed_nl0_w`** (breath-0 NL feed into mask head, FED item 8) | `phase1_algebra_head.py:1772-1777` | **WOKE (converged early)** | Norm 1.158(s4000) -> 1.146(s8000) -> 1.151(s12000) — woke fast in the first 4k steps, then plateaued. Verified this is NOT the "ALG_MH_XPRIOR gates the nl0 feed" bug hypothesized going in: `_fed_nl0` is computed independently in `forward()` (not reusing `out["nl0"]`, which IS gated behind `ALG_MINE_BREATHS`/`ALG_MH_XPRIOR`), gated only on `FED_NL0 and ALG_MASKHEAD`, both true in the champion recipe — see Track B below. |
| 9 | **FFN 2x->4x extension zero door** (`ffn_w2[1024:2048]`, FED item 4) | `phase1_algebra_head.py:339` | **WOKE (converged early)** | New rows' norm 2.087(s4000) -> 2.114(s8000) -> 2.115(s12000) vs old rows' 27.0 — a real, meaningful (~8%), fast-converging contribution. |
| 10 | **FED item 5, macro multi-form** (`fed_pf_dig2_g`, `fed_pf_y_g`) | `phase1_algebra_head.py:296-301` | **MIXED — dig2 half woke, y half plateaued small** | `fed_pf_dig2_g` grows monotonically and keeps growing at s12000 (norm 0.0068->0.0086->0.0107, consistent sign pattern all 3 snapshots) — genuinely learning, not yet saturated. `fed_pf_y_g` converged by s4000 to a small stable value (norm ~0.0039, same sign pattern all 3 snapshots) — alive but tiny; may simply be a small correction, not asleep. |
| — | **Domain-mass port** (`ctx["mh_mass"]`) | `phase1_algebra_head.py:1734`, gated by `ALG_MH_MASS` / `MASSB` at `~3060` | **CLEAN (documented, not new)** | Never fed in the champion recipe (no `ALG_MH_MASS`), degrades to zeros by design (the None-grad law). This is the SAME "two of four senses dark" fact already disclosed in the ledger's round-1 mask-head entry — not a new finding, listed for completeness since it compounds with rank 1. |

---

## TRACK A — the gate autopsy (detail)

Read via `tinygrad.nn.state.safe_load` (CPU, no GPU) on
`sharp_fedon242{,_s4000,_s8000}.safetensors`, `sharp_mhr2ctl242.safetensors`,
`sharp_port242.safetensors`. Full key-diff: `fedon242` has 114 tensors,
`mhr2ctl242` 96, `port242` 78; every fed-new key (`fed_mx_hg`,
`fed_pf_*`, `fed_nl0_w`, `fed_sil2`, `fed_w2a/b`) and every mask-head key
(`mh_*`) is present in fedon242, none missing — the checkpoint carries
everything the patch claims.

The decisive move was reading the THREE SNAPSHOTS (s4000/s8000/s12000)
rather than only the final checkpoint: a small-but-nonzero final gate
value is ambiguous (could be slow-waking or could be noise that happened
to land nonzero), but the SIGN TRAJECTORY across snapshots disambiguates
it immediately. Consistent sign + growing magnitude = learning. Sign
flipping between every pair of snapshots at magnitudes near float32
gradient-noise scale = never learned anything. This is the same texture
argument as the ledger's "2 unexplained curve shapes" law, applied to
gate autopsies specifically: it should probably become a standing
audit technique (see Recommendation, below).

Capacity accounting implication: the FED package claims +5,356,568
params (pointers 2.36M, FFN 1.05M, macro 0.89M, waist 0.53M, nl0 door
0.26M, shelf lane-2 0.26M, scratch 4k, mixer heads 8). Ranks 2+3 above
(pointers + shelf lane-2) account for 2.62M of that — **49% of the
claimed new capacity is, by direct measurement, not doing anything
after 12k steps.** The genuinely awake new capacity is closer to
mixer(8 params, but a real new attention geometry) + waist(0.53M) +
FFN(1.05M) + nl0 door(0.26M) + macro/dig2(partial) ≈ 1.9-2.4M, well
under half the advertised figure. This reframes "the first intervention
to win on both registers simultaneously... the signature of genuine
capacity relief" (2026-09-06 ledger entry): the relief is real, but
smaller than the headline number, and the ledger's own capacity
projection ("~11.9M + 5.36M = ~17.3M trained... inside the 14-18M
landing zone") should be read as an upper bound, not the operative
figure, until pointers/shelf-2 either wake or get pruned.

---

## TRACK B — the wiring trace

**Prime suspect checked and CLEARED:** the brief specifically flagged
"`ALG_MH_XPRIOR` wasn't in the fed env line — does the nl0 feed depend
on an env the chain never set?" Read `forward()` closely
(`phase1_algebra_head.py:2204-2286`): there are TWO independent nl0
computations in this file. `out["nl0"]` (the module-level output key
used by the offline paired-atlas miner / cross-prior) IS gated behind
`ALG_MINE_BREATHS or ALG_MH_XPRIOR` (line 2281-2286) — that one really
would be dark in the champion recipe. But the FED item 8 feed
(`_fed_nl0`, line 2204-2212) is a SEPARATE, identically-formulated
computation gated only on `FED_NL0 and int(os.environ.get("ALG_MASKHEAD","0"))
and "fed_nl0_w" in p` — all three true under `ALG_FED=1 ALG_MASKHEAD=1`.
It is threaded through `_bs_ctx["fed_nl0"]` (set once before the breath
loop, loop runs `breath_step(p, _bs_state, kb, _bs_ctx)` with the SAME
ctx dict every iteration — confirmed at line 2243-2244) into
`ctx.get("fed_nl0")` inside `breath_step` (line 1772), which is
unconditionally reached whenever `ALG_MASKHEAD` builds `mh_atlas_w`
(no separate `ALG_MH_ATLAS` gate on THIS injection site — `mh_atlas_w`
itself always exists once the mask head is on; only its *content* is
zero-fed per rank-1 above). Snapshot evidence (rank 8) confirms this
path is genuinely live. **The hypothesis was reasonable but wrong for
this specific feed — it correctly diagnosed the general shape of the
bug (an env the chain forgot to set) but the actual casualty is
`mh_atlas_w`/`mh_mass`, not `fed_nl0`.**

**The `_STEP_TAP`/fused-vs-segmented split, checked and CLEARED:** when
`_STEP_TAP.get("hold")` is true, `forward()`'s own breath loop is
skipped entirely and `scripts/step_trainer.py` drives `breath_step`
externally with its OWN ctx (`self._mk_ctx(...)`, `step_trainer.py:366`)
— a plausible place for `fed_nl0`/`nb2`/scratch wiring to silently
vanish if step_trainer.py predates the FED patch (it does: 2026-09-03
vs FED applied 2026-09-05). **This does not bite the champion**: the
FED chain (`fed_mind_chain.sh`) trains with `phase1_algebra_head.py
--train` directly, never invoking `step_trainer.py`. `_STEP_TAP` stays
`None` throughout, so the normal in-`forward()` loop (with the correct
ctx) always runs. Confirmed no training or reading in the fed chain
touches step_trainer.py. Filed as a LATENT risk, not a live one: if
anyone runs the FED package under the segmented step-trainer (e.g. for
the memory-savings it was built for), `_mk_ctx` needs an audit for
`fed_nl0`/`nb2`/scratch-column parity before trusting the result.

**Ordering checked and CLEARED:** the scratch-column mask widening
(`slot_mask` -> `L_TOT`-wide, item 6) happens at `forward():2059-2065`,
immediately after `_make_bank` and well before the initial `bank()`
call (`:2134`) and the `_bs_ctx` construction (`:2213`) — every
downstream consumer sees the widened mask. The waist residual
(`forward():440-446`) is injected as the very first transformation of
`waist`, before any other organ reads it, so there is no
stale-closure-over-old-waist risk. `state["mh_prev"]` is read
(`:1740`) strictly before it is overwritten (`:1803`) within the same
`breath_step` call, so it correctly carries the PREVIOUS breath's
adjacency, not the current one.

**Sub-env defaults checked and CLEARED:** `_fed_sub(name) = bool(ALG_FED
and int(os.environ.get("ALG_FED_"+name, "1")))` — every `ALG_FED_*`
sub-dial defaults ON once the family is on. The fed chain sets only
`ALG_FED=1` with no per-item overrides, so all nine items are
structurally enabled; the dead ones (ranks 2/3) are dead from lack of
gradient signal, not from a mistaken default.

---

## TRACK C — the discarded-signal sweep

No NEW "computed and thrown away" tensor was found in the FED
patch itself — every new tensor introduced by `apply_fed_mind.py`
reaches either the loop (mixer, waist, nl0, notebook lane 2) or a
documented diagnostic-only sink (scratch fq rows are attention
citizens but never emitted, by the Goodhart-fence design; `fed_pf_*`
gains reach real emission heads, they are just not learning — a
DIFFERENT failure mode than "discarded").

The one true instance of the classic pattern predates the FED patch
and is already ledger-known: `mh_atlas_w` / `mh_mass` (rank 1 above,
and the domain-mass port) — these are not "computed and discarded,"
they are "wired to receive input that is never supplied," a sibling
bug to the vst pattern rather than a repeat of it. Distinguishing the
two matters for the fix: vst-style bugs are fixed by rerouting an
already-flowing signal; this one needs a signal that currently
doesn't flow at all (either wire `ALG_MH_ATLAS`/`ALG_MH_MASS` into the
champion's training recipe, or strip the dead 262K+ params from the
capacity accounting).

`breaths_all`, `nl_all`, `nlat_all`, `fst_s` are all genuinely
diagnostic-only by design (offline atlas mining / consult), not silent
bugs — confirmed each is read by a specific offline consumer
(`mine_step_atlas.py`, `atlas_collapse_check.py`,
`step_engine_read.py`'s atlas hook) outside the training loop.

---

## TRACK D — the read-regime check

GPU confirmed idle before firing (`systemctl --user list-units
--state=running | grep -E "fed|mask|paired|atlas"` — no matches). Ran
reads sequentially, nothing concurrent.

**Analytical result (no GPU needed) on the "fully OPEN" comparison the
brief asked for:** `ALG_SHELF_CIRCLE=2 SC_EVAL=0` (the champion's actual
read/train regime) and `ALG_SHELF_CIRCLE=0` (seal mechanism entirely
absent) are **mathematically identical, not just empirically close**.
Reading `phase1_algebra_head.py:1570-1602`: when `SC_EVAL` is set to
`"0"`, the seal interpolation collapses exactly to
`cur = cur*(1.0-0.0) + _cur_seal*0.0 = cur` and
`q_extra = _q_open*(1.0-0.0) + _q_seal*0.0 = _q_open = q_extra + _inj4*bus_g`
— bit-for-bit the same computation as the `else` branch taken when
`ALG_SHELF_CIRCLE` doesn't trigger at all. There is no daylight between
these two configurations to probe; running both would have burned a
GPU read to confirm arithmetic. **The read-regime axis that actually
matters is OPEN (`SC_EVAL=0`, forces the interpolation to the open
value) vs SEALED (`SC_EVAL` unset, defaults `_SEV=1.0`, "the pressure
cooker" — residual severed at breath `SC_KB`, only committed facts
cross)** — this is the actual capability-meter contrast the "pressure
cooker" ledger entry describes, and it's the one worth spending a GPU
read on.

**GPU reads performed** (`scripts/step_engine_read.py`, SE_R=1 — this
IS `loop_val.py`'s exact two-pass fact-fed read by the script's own
construction, chosen because it also prints the leak gauge):

| Run | Ckpt | Envs | pass0 (open) | pass1 (fed) | facts (leak gauge) |
|---|---|---|---|---|---|
| 1 | fedon242 | standard champion read env, SE_THETA=0.9 (default) | 0.1677 | **0.2613** | 1067 / 2051 (52.02%) |
| 2 | port242 | same env minus MASKHEAD/FED, SE_THETA=0.9 | 0.1824 | **0.2530** | 1062 / 2051 (51.78%) |

Run 1's pass-1 number (0.2613) reproduces the ledger's fedon242 wild
record exactly — confirms the env reconstruction is correct and gives
independent verification via a DIFFERENT instrument than
`loop_val.py`. **The wild fact-commit rate is essentially unchanged
between fedon242 and port242 (52.02% vs 51.78%, a 0.24-point
difference well inside noise)** — the champion's accuracy gain is NOT
coming from committing to more facts / leaking more supervision
through the commit channel. This is a clean result: it rules out
"the record is just a more aggressive forcer" as the mechanism.

**THE HEADLINE RESULT — sealed-mode wild collapses to exactly zero:**

| mode | pass0 (open, no loop) | pass1 (conditioned) | facts |
|---|---|---|---|
| open (`SC_EVAL=0`, the champion's actual regime) | 0.1677 | **0.2613** | 1067 |
| sealed (`SC_EVAL` unset, residual severed at `SC_KB=4`) | 0.1677 | **0.0000** | 6 |

Pass 0 is bit-identical between the two runs (0.1677 both) — expected
and a correctness check on the run itself: `step_engine_read.py`'s
pass 0 calls `forward()` with `slot_mask=None`, and `forward()`'s own
gate is `if K_B > 1 and slot_mask is not None and "W_bo" in p:` — the
ENTIRE breath loop (and therefore the seal, which only fires inside
that loop at `kb == SC_KB`) is skipped for pass 0 regardless of
`SC_EVAL`. So pass 0 matching confirms the harness is behaving
correctly; the divergence is real and lives entirely in pass 1, where
the full 7-breath loop runs and the seal actually executes.

Under the seal, fedon242 scores **exactly zero** of 2051 var-slots on
wild holdout, and the post-severance commit buffer captures only 6
forced facts (0.3%, vs 1067 in the open regime) — the severed state
(breaths 5-6, running on `cur = shelf read only` after breath 4 per
`phase1_algebra_head.py:1580,1599-1600`) is producing near-random
output, not a degraded-but-useful parse.

**This appears to be a new measurement, not a re-confirmation of a
known number.** Searching the ledger for prior sealed-mode
("pressure cooker"/"cooker v1"/"the capability meter") readings turns
up only MAIN-VAL-fixture numbers, never wild: the 2026-08-30/31 PULSE
VERDICT entry reports "sealed ~0.690 vs open ceiling 0.7304 -> THE
TWO-JAWS RATIO ~94.5%" (`docs/phase1_skeleton_spec.md:31773-1789`) —
a MILD degradation, not a collapse, but explicitly on the **val**
fixture (the same entry frames the comparable OPEN-mode bar as "val
0.6990 vs bar >= 0.7296"). No wild-register sealed number is banked
anywhere I could find (the one PRE-REGISTERED wild capability bar,
`docs/phase1_skeleton_spec.md:32931-2941`, was left PENDING — the
alternator v2 round that would have measured it never got past mint
before the wild-ladder fixture existed). **If this reading holds up
under a repeat/second-seed check, it means the entire wild-register
improvement story of the last four days (port242 -> mhr2ctl242 ->
fedon242) has been measured exclusively in a regime where committed,
auditable symbolic facts are not required to carry the parse at all —
the open residual is doing 100% of the work on wild, where on the
val/mint register it does only ~5.5% extra on top of an already
strong committed-fact floor.** This is the sharpest, most literal
form the "organs present but silent" pattern takes here: it is not
one gate quietly asleep, it is the entire committed-facts/mask-head/
gate architecture's CONTRIBUTION TO WILD, when forced to stand on its
own, testing at zero. Whether that is a fixable wiring problem (the
severance point `SC_KB=4` may simply be too late/too abrupt for wild's
longer settling — recall "wild settles slower/less" from the same-day
entropy-gut entry) or a structural fact about where wild competence
currently lives is exactly the kind of question CLAUDE.md's chain of
custody (mouth -> vote -> panel -> key) is built to answer, and it
should probably be pinned as a registered follow-up before crediting
any further wild record to "capacity relief" rather than "a better
residual."

**Theta sweep, fedon242 wildhold, SE_R=1:**

| theta | pass0 (open) | pass1 (fed) | facts |
|---|---|---|---|
| 0.9 (default) | 0.1677 | 0.2613 | 1067 |
| 0.7 | 0.1677 | **0.2613** | 1062 |

Lowering the commit threshold from 0.9 to 0.7 changes the final wild
accuracy by exactly zero (0.2613 both ways) and moves the forced-fact
count by less than 0.5% (1067 -> 1062) — and in the WRONG direction for
a naive "lower threshold = more forced" intuition, consistent with the
ledger's own account of `alt2_fact_buf`'s contradiction handling
("contradiction keeps the silence convention": a looser threshold
admits more raw candidates but also more of them collide and get
withdrawn). **CLEAN**: the champion's wild ceiling is not perched on a
knife-edge threshold choice; it is insensitive to a meaningful (0.2)
shift in the commit gate. Runtime note (perf, not correctness): the
theta=0.7 pass took roughly 4-5x longer wall-clock than the theta=0.9
pass at the same batch count, all in CPU (`alt2_fact_buf`'s per-item
python loop, not the GPU forward) — worth a profiling look independent
of this audit, since a threshold change alone should not change the
asymptotic cost this much unless the contradiction-resolution path is
quadratic-ish in the number of live candidates.

**Sealed-mode (`SC_EVAL` unset, "the pressure cooker" capability meter)
vs open-mode (`SC_EVAL=0`, the champion's actual read/train regime) on
fedon242 wildhold, SE_R=1, theta=0.9:** [see addendum immediately below
— run fired after the theta sweep, per the no-concurrent-GPU rule].

---

## SUMMARY — ranked, one line each

0. **SMOKING GUN, the big one**: under the sealed-mode capability
   meter (residual severed, committed facts only), fedon242's WILD
   holdout score is **exactly 0.0000** (vs 0.2613 open) and the
   post-severance commit buffer captures 6 facts (vs 1067 open) —
   apparently the first time this specific probe has been run on the
   wild register for any checkpoint in the lineage (every prior
   sealed-mode number in the ledger is on val/mint, where the ratio is
   a mild ~94.5%). If replicated, this means the last four days of
   wild records have been measured in a regime where NONE of the
   committed/auditable machinery (mask head, gate, fact buffer) is
   load-bearing on wild — the open residual carries 100% of it. This
   is the "organs present but silent" pattern at the scale of the
   whole symbolic-commitment subsystem, not one gate.
1. **SMOKING GUN**: `mh_atlas_w` (mask head's atlas-page door, ~262K
   params) receives exactly zero gradient in the champion recipe —
   its input is unconditionally zero because `ALG_MH_ATLAS` is never
   set; provably inert, not merely small (`phase1_algebra_head.py:1759-1771,2934`).
2. **SMOKING GUN**: FED item 2 pointer multi-form gains
   (`fed_pf_args_g/res_g/query_g`, 2.36M params — the single largest
   line in the +5.36M FED package) show sign-incoherent noise across
   training snapshots, not learning; genuinely wired, genuinely dead.
3. **SMOKING GUN**: FED item 9 shelf lane-2 gain (`fed_nb_g`, 0.26M
   params riding behind it) — same noise-around-zero signature.
4. **SUSPICIOUS**: scratch slot embeddings (`fq[24:32]`) — read-back
   door is open (gate awake) but the slot content has barely
   specialized (98.9% cosine-stable since s4000); half-built, not silent.
5. **SUSPICIOUS**: `mh_gain` (mask head master scalar) wobbles
   non-monotonically near zero (0.02 ajar-init -> ~0.001, with a 35x
   dip-then-recovery between s4000/s8000/s12000) — likely a co-adapted
   rescaling against the now-large `mh_wo`/`mh_headmix`, but not
   verified; needs a read of the product magnitude, not the gain alone.
6. **CLEAN (confirmed, not the suspected bug)**: `fed_nl0` (breath-0
   NL feed) is independently wired and correctly gated on
   `FED_NL0 and ALG_MASKHEAD`, NOT on the `ALG_MH_XPRIOR` env the
   brief worried about — that env only gates the offline-mining
   `out["nl0"]` key, a different tensor with the same name pattern.
7. **CLEAN**: mixer multi-head gate, waist residual door, nl0 door,
   FFN 4x extension zero-rows all show clean, monotonic,
   snapshot-consistent learning curves — genuinely woken organs.
8. **CLEAN**: the "fully OPEN" read-regime variant the brief asked for
   is mathematically identical to the champion's standard read env
   (`SC_EVAL=0` forces the same arithmetic as `ALG_SHELF_CIRCLE=0`) —
   confirmed by code inspection, no GPU spent confirming a tautology.
9. **CLEAN**: wild fact-commit rate (the leak gauge) is essentially
   unchanged between fedon242 (52.02%) and port242 (51.78%) — the
   record is not an artifact of more aggressive fact-forcing.
10. **CLEAN**: SE_THETA sweep (0.9 vs 0.7) on fedon242 wild changes
    accuracy by exactly zero (0.2613 both) and forced-fact count by
    <0.5% — the wild ceiling is not perched on a fragile threshold
    choice. (Runtime note, perf not correctness: theta=0.7 took ~4-5x
    longer wall-clock, all CPU-side in `alt2_fact_buf`'s per-item
    python loop — worth a profiling look independent of this audit.)

**Headline for Bryce:** two independent claims both need an asterisk.
(a) The wild/mint double-record's mechanism looks different than
advertised once you force the parse to prove itself without the open
residual: on wild, committed symbolic structure currently contributes
NOTHING measurable (item 0) — worth a second-seed repeat before
treating it as settled, but if it holds, "genuine capacity relief"
should be read as "a better residual," not "more usable symbolic
capacity," until proven otherwise. (b) Independent of that, roughly
half the claimed new FED capacity (pointers + shelf-2, 2.62M of
5.36M) is present in the checkpoint and structurally wired but has
not learned anything in 12k steps — sign-incoherent gate noise, not
slow-wake — and the mask head's pre-existing atlas door (~262K
params) is *mathematically* incapable of learning under the
champion's own training recipe, not just slow. Neither (a) nor (b)
contradicts the wild/mint double-record itself; both mean the record
was won with less machinery, and less of it auditable-on-wild, than
the ledger's capacity accounting (11.9M + 5.36M ≈ 17.17M "trained")
currently implies.
