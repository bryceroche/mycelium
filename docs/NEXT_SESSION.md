# NEXT SESSION — cold start (HANDOFF 2026-09-22 ~12:30; the box restarts)

**READ FIRST:** CLAUDE.md -> this block -> the ledger's entries from
"2026-09-19 10:20 — THE SURFACE PIVOT" to the end (docs/phase1_skeleton_spec.md;
the last ~60 entries) -> the memory board (MEMORY.md CURRENT BOARD, current).
The reads table: `.venv/bin/python3 scripts/reads.py table wildhold` (assert-on-read).

## STATE AT HANDOFF
- Nothing running: no pc-* units, no agents, no monitors; tree clean at 11ea0c53 (pushed).
- THE CHASSIS (certified neutral on two seeds, +4-6 mint): the 35%-mint from-scratch
  lineage's recipe (form_mix_pm35 / the stamped pm35a, random init, 48k, B=8, the numeral
  mask at the read) + ALG_ROUTER=2 (the bus-native per-field router) + ALG_SPAN_ALL=1
  (the per-breath span loss) on the stamped diet. Bodies: PMS4_241 masked wild 0.3325,
  PMS4_242 0.3320; the lineage PM35_scratch_241/242 0.3291/0.3267 (all vs the 311-row
  wild holdout; SE ~0.009 per read; claims need +0.020 z 2 or a twin pair).
- THE RECORD BODY (one seed, uncertified): PMS8_241 = the chassis + the arg/op/rcue/arcue
  span losses + THE ROLE SIGNATURE pointer (ALG_PTR_SURF=role:add:2.0, BIND_CODES=
  .cache/bindbus_codes512r.npz) on form_mix_pm35c: masked 0.3476 (+0.015 vs PMS4 z 1.78;
  +0.0185 vs the lineage z 1.93 — the claim bar missed by 0.0015), open 0.3198 (z 2.26),
  rows 14/311 masked (PMS4 9), mint flat. Its target cell (wild same-noun twins) moved only
  +0.010 (bar +0.05 missed); the args field flat (0.449).
- THE WALL (unchanged in kind): wild args 0.43-0.45 on every body; rows 5-14/311 (~3-4%).
  The pointer wall is SELECTION among correctly labeled candidates (the target-side
  census), worst in the same sentence; the twins are mostly THE SAME ENTITY at different
  roles/times (the twin-split census: same-noun 25% of the wrong mass, different-noun 9%,
  mixed/no-key 66% = relation-type candidates with no key — UNDECOMPOSED).
- THE SURFACE IS CLOSED for the pointer: pooled clause/value keys, single noun tokens
  (waist AND trunk), one-token lexical identity, sentence-granular priors (the chart),
  the HUD, the span-complete prose renderer — none moves wild. Pass/fail alternation
  (the wheel family) is retired as feedback (THE CHASSIS RULING: value propagation).

## THE QUEUE (all need the word; nothing fires without it)
1. DONE 09-22 15:20 — THE ROLE SIGNATURE IS A CLAIM: PMS8_242 masked 0.3525 (the slot record; +0.0205 vs
   PMS4_242 z 2.42, +0.0258 vs the lineage twin z 2.81; open 0.3276 z 2.65/2.98; rows 13/9; mint flat); the
   pair: +0.018 vs the chassis (combined z 2.97), +0.022 vs the lineage (z 3.35). The args FIELD does not
   carry it (0.429 vs 0.453) — the gain is composition (refusals 96 vs 126). Ledger 09-22 15:2x.
2. THE REGISTER FAMILY (ledger 09-22, seven entries): the register warm +0.018 (z 2.5) on the held-out slot;
   the register from scratch NULL WITH A COST (PMS9_241: masked -0.009 vs PMS8_241, mint -7) — a WARM organ;
   the addressing road NULL; THE VALUE STREAM's two doors NULL WITH A COST (VR masked -0.010; VRA digits flat)
   — the facts-coverage mismatch registered (census on wild first). Branch value-stream (worktree
   /home/bryce/mycelium-wt) UNMERGED: built, gated bit-identical, dead unless set. CARD IDLE.
   09-23 11:15 WORD GIVEN: pc-cont (PMS8c_241 vs CTRLc_241, +0.020 masked paired; + the facts census on wild) then
   pc-pms9r (the fade-in scratch arm, ALG_BUSREG_RAMP=8000, vs PMS8_241 and PMS9_241) — RUNNING; read cont_chain.log
   and pms9r_chain.log. The order of arrival: value passing (after the census) -> atlas -> mask.
3. DONE 09-22 12:50 (scripts/mixed_bucket_census.py -> .cache/mixed_bucket_census.txt): the
   inherited key dissolves the mixed bucket (89% SAME-NOUN); the wall is ~90% same-quantity
   selection, 55-57% of it pointing at DERIVED arguments (k a relation; args-correct 0.58-0.62)
   whose identity is in no token — identity must be PROPAGATED through the bus (ledger).
4. THE DUAL-STREAM IDENTITY PORT: folded into item 2 (the register's identity stream; ident_codes_mint.py).
5. The refusal lever: 40% of wild rows are refusals — the row's other half. The failure-class
   row census DONE 09-22 (row_census.py on four bodies, ledger): DIGITS (the given's value) is the
   largest failure class on every body, 5x args; one-away rows break on digits and ftype first.

## RULES LEARNED THIS WEEK (in the ledger; binding)
- A structural road is verified IN THE LOSS before any arm (a grad norm on its own params
  from that loss alone) — THE LADDER SHADOW: the loss grades out["breaths"], not out["args"];
  the ladder patch applies only under an active ALG_PTR_SURF fusion (unconditional it
  diverged the chassis configs at step ~500).
- A road that enters as structure must be exact on EVERY register in the diet (the
  slot->variable map: positional on prose, first-mention on mint; the res-map scatter).
- A warm gate must have resolution: 20 steps see no divergence at step 300; 44 slots see no
  0.05; gate on the 1,024-row slice (`.cache/form_pm35_slice1024.jsonl`, states pm35slice),
  600+ steps, the twin cell with n in the hundreds (scripts/role_twin_cell_compare.py).
- A masked read must differ from its open read (reads.py flags MASK SANITY on ingest); the
  numeral mask lives in mycelium/rulebook.legal_digit_logits (one rulebook, two doors).
- Every GPU command in a chain takes `.cache/gpu.lock` (tinygrad's AM lock refuses a second
  process; reads collide with training otherwise). PRECOMPUTE_ONLY=<split> when a states
  memmap is hard-linked (a bare --precompute truncates both names).
- Ledger timestamps drift when I write ahead of the clock: `date` before stamping.
- Sonnet builders: small targeted Edits on the head; never `git commit -a` while a builder's
  tree is live (two sweeps this week); never pgrep/pkill a pattern that appears in your own
  command line (exit 144).

## ARTIFACTS (all in .cache; the ledger names each)
Diets: form_mix_pm35{,a,c,v,p,ar,pr}.jsonl (a = arg-mention stamps; c = + role cues;
v = + value spans; p = positional mint (STRUCK); ar/pr = + the rendered 5k). States:
phase1_alg_states_formpm35*_states.npy are HARD LINKS of one 53 GB memmap (pm35pr/ar
are a separate 58 GB one). Checkpoints sharp_<body>.safetensors; per-slot files
ps_{open,legal}_wild_<body>.npz (paired_read.py); dumps dump_wild_<body>.pkl; censuses
.cache/{membrane,args,args_target,twin_key,entity_noun,lexical_identity,twin_split}
_census*.txt. The renderer: scripts/prose_render.py -> render_v1_5k.jsonl. The role codes:
bindbus_codes512r.npz (old keys byte-identical to bindbus_codes512.npz).

---- (the previous cold-start doc, 2026-08-10, follows; superseded above) ----

# NEXT SESSION — cold start (updated 2026-08-10)

**READ FIRST:** CLAUDE.md → this file → the ledger's last ~40 entries
(the tail chain, the record, the rite's first refusal, door #36).

**THE QUEUE (2026-08-11 ruling, the top three = the cathedral's
missing stones):** 1. NAZARÉ-AS-FOCUSING — commit where
BINDING-ENERGY peaks (the rebinding discovery named its quantity);
first step = the binding-energy field read on banked g51 (a read,
on a word). 2. THE TOWER — multi-level IR, spec-first; warrant
renewed (structure transferred at both poles tonight). 3. TRANSFER
— the science over all organs (one affirmative vs three
no-transfers; removal bars ride every organ fire). Behind: anchor
design (row gate ANSWERED: 5/15 carriage @ 0.7% cost), second
cuts, organic gold, the December two-read question (reserved).

**THE DECEMBER CLAIM IS DECIDED (2026-08-07): LEAF A — a system
that never lies; abstention leads; scope in the same breath;
mouth-widening is the claim's guarded gate (trace frontier measured
BEFORE widening).** All roads = coverage growth under the claim.

**GEN-41 IS THE GATE (2026-08-10, PROMOTED; the one-mass fire's
arm):** g41_onemass_refold — both goods on one vehicle (formation +
op-balance on args=[a,a] mass), full dup scope 0/3/-/0, bigtest
1254, alg4 405, **stress certified lies ZERO** (better than 23v5's
baseline 1). Riders AS DATA in the manifest: op-margin thinness
(+0.74 floor — the next mul fire meets this wall) + the refold
clause (successors inherit the REQUIREMENT, not the cure).
ENTOURAGE-41 riding (frontier stands LAST, post-dressing).
Lineage: g23v5 (2026-08-09, the dup-detector adoption) preceded.

**THE TAIL CHAIN — NINE LINKS (2026-08-10):** census → families →
stations → waist → gradient → diffuse → responsive → paying →
**PRICED**.

**PRICED (the ninth link, the decision-relevant fact):** the dose
response is SUB-LINEAR AT DEPTH — threshold-harvesting anatomy: the
dose flips near-boundary rows first (+21 score), the deep band moves
only +0.012 at ~5x effective, the deep tail persists. **The next
increment is priced before purchase: don't buy it blind — buy design
(compounding, E5 co-vehicle) or don't buy.**

The chain's substance: bigtest's 245-tail = size-graded evidence
EROSION under indirection (probe ladder 0.98/0.81/0.48/0.19). The
DOSE FIRE (ration 8x hot-phase, door #35) hit **THE RECORD: bigtest
1259/1500 on g35_size8x** (+21 over promoted 1238). THE METER LAW
carved (berries' third site): an outcome-defined band cannot measure
a cure that changes outcomes; the FROZEN-BAND re-read (baseline
partition) shows BOTH bands rise (pass 0.810→0.830, miss
0.477→0.489) — the record is mechanically honest, state and score
agree.

**THE RECORD'S PROMOTION QUESTION IS OPEN-UNKNOWN — THE
INSTRUMENT CAUGHT ITSELF:** the no-harm battery found dup cells at
5/15/15/15 on g35 (THE CRATER through the deployed head — REAL, the
promotion-era scan instrument; the 4k unlearned the cure) and on
g36 (door #36: freeze + 3x rehearsal — the freeze-fell law's third
measurement; that fire's kill stands procedurally). But BOTH refold
refusals are VOID: the reconstructed rite ABORTS ON THE GATE ITSELF
(g23v5 reads 0.347/1.120 under it vs 1.141/0.801 at adoption — same
bytes, different pools: fresh-minted cells + true-dup-contaminated
wild-forty slots, vs the adoption's fixture cells + wild-ledger
negatives). NEVER MIX INSTRUMENTS' READINGS. Under one instrument
all three arms read alike with identical healthy medians — bulk
separability intact everywhere; the erosion narrative is DISSOLVED.
THE REFOLD LAW STANDS. **THE CALIBRATED RITE (calibrated_rite.py,
control-assert first — the instrument law carved) READ TRUE:
control passes (1.509/0.814 ≈ adoption's numbers); g35 AND g36
both OPEN; folds banked; and the decode scans print THE IDENTITY:
misbind@nd4 = missing slots exactly (12=15−3, 9=15−6) — where the
slot exists the fold cures it. The 4k fire's whole irreducible dup
cost is REL-SLOT FORMATION under crowding** (novel phrasings;
presence grain; no fold reaches it). Formation shows dose response
to rehearsal (nd0 fully protected, nd4 3→6 slots). Neither arm
meets scope (nd4 ≤0): both benched; GEN-23v5 remains the gate.
NEXT DESIGN for the word: formation-targeted rehearsal — crowded
novel-phrasing dup rows minted into the rehearsal set (the mix's
dup rows are familiar low-crowding forms; the hole is exactly
where the mix is empty).

**DOORS #37-#38 (the formation fires) NEARLY PROMOTE — the dup
file's last local residue dissolved INTO the op file:** g38 (bench,
the campaign's best arm): formation cured BOTH ends (nd4 0/15 with
#37's fifteen held; nd0 formation intact), bigtest 1257/alg4 408/
sentinels clean/frontier 40/40 unmoved, widest calibrated gap ever
(1.541/0.609 = 0.932). The sole failing bar: nd0 4/15 = OP-mismatch
on mul-dup at low crowding — branch (c) fired (not coverage-
reachable at 3x); it belongs to the standing op-head item. OWED
NEXT: the op-grain autopsy on the failing four (read, before any
door #39). Door #38's first burn was a REPLAY (ALG_TRAIN_NAME
fossil — states keyed by name; identical-to-the-digit unmasked it;
sha-fence on era's next-touch list).

**DOOR #37 (the formation fire) history:** nd=4 cured
0/15 with the row-grade 15/15 ON THE FENCED SURFACE; bigtest 1259
= the record WITH the cure riding (goods did not trade); alg4 406,
sentinels clean; g37 opens the widest calibrated gap yet
(1.504/0.727). ONE bar failed: nd=0 4/15 — autopsy: formation
INTACT (all rel [0,0] form), residue is OP-GRAIN on mul-dup at the
band the sliver never covered (nd 0-1). g37 benched (best arm ever
benched).

**THE BOARD (one board; supersedes all prior lists, 2026-08-10):**
1. ENTOURAGE-41 PAID (all ten; frontier stood POST-DRESSING
   40/40, stress lies ZERO under the new mouth; dissent ROTATES,
   overlap zero). GEN-41 rules. THE TAIL AFTER THREE ATTACKS (all pre-registered,
   all honest): surface@matched-size = ZERO; size@dose = threshold
   harvest, sub-linear; length op-balanced = CORRELATE-NULL (door
   #42: long band converts SLOWER, 0.135 vs 0.171 — the 3.2x
   enrichment was subsurface's shadow). **The 233 are not
   diet-reachable at the row-selection grain. Unplayed: ALTITUDE
   cards — the floor's supervision form (gold-side) and the second
   clock.** Door #42's side-gains banked: simplex tolerance
   BRACKETED (4.3% safe / 8.6% kills); thinness watch first print
   = RECOVERY (+0.74→+1.29); widest calibrated gap (0.956); g42
   benched panel-eligible. Stress-lie range across arms now 0-3
   (churn; every sheet carries it). DOORS #43-#45 (the transport arc) CLOSED honestly: E-floor
   refuted (probe flat; REF retired — the room was always lit at
   0.992); breaths VACUOUS twice (gates don't self-open; opened
   gates don't fill; loss-scale confound carved: breath arms =
   1.5x hotter LR; nd2's two sightings + the rescue's suggestive
   ordinal lean 3.16 ride the organ file); dialect reader =
   SECOND MAJORITY TONGUE (in-register indirect gold is
   majority-readable; THE REGISTER WALL AT THE CODE GRAIN: tail
   code gold cannot come from the measured fixture). INSTRUMENT
   LAWS gained: slot-aligned reads are train-register-only —
   TEST READS MUST BE SET-BASED (the 85% figure void; the
   station read's slot-station numbers owe a set-based re-audit
   before door #46). NEXT CANDIDATES on words: organic-failure
   dialect gold (specialist pattern at args grain); the set-based
   re-audit; the hot-LR control; the breath bootstrap design.

   THE DIAL COMPLETE (door #53, p=0.75): nd0 CLEARED silent
   (first ever) + alg4 409 > gate; but THE TURN BRACKETED — the
   gift dies at loop-scarcity (lean 2.01@0.5 -> 0.78@0.75; band
   (0.5,0.75)); deep cells dial-independent; engaged loop hollowed
   (1100). ONE WAVE, TWO REGIONS, NO THIRD — the second-mechanism
   clause takes the road: selective (grain-declared), annealed
   (ratchet flag), notebook-gated. All on words.

   BREATH DROPOUT (door #52, p=0.5): BRANCH (b) REFUTED — the
   gift survives the loop's absence (silent lean 2.01); every
   silent axis moved toward deployable (misses 319→255, nd0 15→4,
   silent score 1233, alg4 407 = the gate's number); the grains
   CONVERGE (engaged nd4 repair gone as silent competence grows).
   THE DIAL: silence-heavier coin p≈0.75, on the word. Also
   standing: selective design (grain-declared), texture probe,
   competence map (two-pass).

   THE COMPOSITION CEILING (door #51): the band is EMPTY —
   lean needs >5%, cells never clear <=12% (floor = the pipe's
   residual cost); uniform-global authority exhausted honestly.
   THE FULL CURVE: 0/5/12/50% -> lean -/1.72/2.87/5.17; score
   1254/1250/1237/1133; whisper nearly damage-free (net -2) with
   the probe's FIRST upward twitch (0.210). SUCCESSORS on words:
   (1) SELECTIVE AUTHORITY (leads — the lean is selectivity
   evidence; monitor-family gating); (2) annealed (ratchet flag
   waiting); (3) notebook-gated commit. Plus the breath-0
   competence read (queued).

   THE TITRATION (door #50-R): THE KNOBS SEPARATE — at 12%
   authority the lean holds 2.87 (above the deployable bar) while
   damage decays FASTER (regressions x0.37 vs lean x0.56) and
   converts RISE (22>19); scores 1254/1237/1133 at 0/12/50%.
   Deployable-when-needed within reach; next dial = finer
   titration (word). Queued: breath-0 competence map read.

   THE ORGAN IS REAL (door #50, the bootstrap fire): breathing
   ENGAGED for the first time (norm + warm pipe + open gates) and
   the ordinal lean SURVIVED+AMPLIFIED at true scale — 5.17
   (18/181 vs 1/52), the only mechanism ever to preferentially
   convert the tail's constituency. Price at 50% authority: parse
   overwritten (cells 15/15 all bands, bigtest 1133; sentinels
   alone held). NEXT: AUTHORITY TITRATION (low-blend arm — does
   the lean persist as wreckage recedes), on the word.

   THE WALL-GUARD (door #47, BUILT as policy): the nd0-op wall
   is a BINARY BIFURCATION flipped by data order alone (same
   composition: −2.57 inversion at seed 127, +3.37 gate-plus at
   227). GUARD = (a) post-fire wall check, HARD, with
   REFIRE-ON-BREAK (one jittered refire before any diet verdict);
   (b) the thinness watch as boundary sensor (deep floor = safe;
   thin/inverted = adjacent). g46's fence-kill re-scoped (coin
   flip); its primary miss stands. Depth-diet composition CAN
   seat full scope (jitB 0/3/-/0 deep).

   DOOR #43 (E-floor, first form) history: tail
   probe FLAT 0.190 BUT h_ref transfers 0.882 on bigtest's own
   indirect sites — **THE EVIDENCE IS PRESENT AT THE TOKEN GRAIN
   AND STRANDED IN TRANSPORT (token→slot)**; the successor is
   transport-grain structural entry (slot-side attention/pointer
   supervision onto indirect mention spans; the pointer law's
   family). Open shaping read: ridge on g41's waist — was the
   token info pre-existing (pure transport) or REF-taught (keep
   REF)? No harm (scope held, ring clean, op floor +1.68 — the
   watch's strongest print). Roads on words: the TRANSPORT design
   (door #44 candidate), the second clock, the E5 window
   generation, the op-depth texture question.
2. The dose curve's next increment: priced sub-linear; buy design,
   not volume (compounding / E5 length co-vehicle — one purchase,
   two sheets).
3. The second clock (spatial grain; constituency measured, nut =
   the anchor).
4. The floor's design (downstream of the size road; premise
   corrected — mentions ARE consumed; 10,160 full-joint rows
   already supervised).
5. General-parse residuals: op head + rel-decode under crowding
   (reassigned from dup; op geometry open, unforced).

CLOSED (for the cold reader): the framing session (decided
2026-08-07 → the claim above); the v5 adoption word (given and
executed 2026-08-09 → the gate above); the era manifest (BUILT —
mycelium/era.py, first customer served at next-touch during the
entourage); the invariance proposal (own generation, waits behind
the door); breaths at deployment (REFUSED, −19 alg4, priced).

**THE WEEK'S LAWS (ledger has full forms):** the refold law final
form (refold after ANY training — freeze fell, three measurements);
the meter law (outcome-defined bands); headroom-first (2-for-2 as
decision instrument: dup licensed, op refused; the g35/g36 refusals
VOID — uncalibrated instrument);
band-lives-in-the-heads (waist is band-general); the design space
(mass-sans-aim=0 / aim-sans-band=flicker / aim-with-band=cure);
FIVE spec-time doors; grains-state-conversion + trades-prove-overlap
(on the door); retired vocabulary: slope, under load, trade.

**STANDING RULINGS:** the pre-registration door
(mycelium/preregistration.py) is LIVE — reads register before
existence; bars never rebind. Marker relaxation = fork (b).
Custody-gold, Goodhart fence, conviction index all in force.


## STATE 2026-08-16 (the hill arc — read the ledger's last ~30 entries from "DOOR #61 REGISTERED")
GEN-41 still the gate. The tower/binding war produced: canonical-reorder
receipt (len_asc 159/233), the residue-74 fixture (binding collapse above
solvability), the refusal trigger, the placebo-differential law, the
two-phone law (6 graves, 2 exceptions), the key-quality ladder, and THE
COMMUNICATION CHARTER (7 components — standing registration rider).
THE HILL: native from-birth build (NATIVE_SPEC) — twin protocol printed
ENGAGEMENT +0.192 (first positive carrier ever); differential deferred to
maturity; continuation (+10k both arms) BURNING as natcont; sheet 2 reads
frozen bars + engagement column. Letter-key successor arm 5/5
charter-complete, conditionally worded. Board: len_asc deployment,
tremor-rescue+vote-gate, interleave view, letter-key arm. MEMORY.md board
rebuild still OWED.

## 2026-08-19: THE CONVICTION BOARD
Read docs/CORE_COMPONENTS.md — the 12 core components tiered by receipt
status with refine/extend/titrate lanes. The titration queue and the
December deployment word live there. Best arm: gsb_real (K7+notebook+
dual-channel seed, 30/74@14k val; bigtest cert owed).


## 2026-08-20: THE TRUE CHAIN + THE HONEST ZERO
Deployed stack (manifest: conductor_v2): bigtest 1394/1500 (93.0%), wrongs 24.
MATH-500 under the TRUE chain (mouth first): 0/500, ZERO lies (mouthless
control: 2 rights, 259 lies — agreement is register-bound; the mouth is the
wall's precondition). The road: the register bridge (books/harvest + a
translation lane). Read CORE_COMPONENTS.md + the ledger's last ~40 entries.
MEMORY.md board rebuild still OWED.


## 2026-08-20 CLOSE: BOOK 9 T1 OPEN
The worksheet is banked (.cache/book9_t1_worksheet.json — routed rows,
construction-tagged). NEXT SESSION OPENS AT THE SURGERY: annotate per the
rulebook, then vote+key gate, then the diet. The stack awaits the register.


## 2026-08-20 FINAL: B9T1 READY FOR THE DIET
25 surgery rows KEY-GATED (the L1-vote-on-L3 category error caught and
carved — surgery's gate is the key; the diet pairs are (ORIGINAL prose,
factors)). NEXT: assemble b9t1 into the training-mix format (original
texts + factor gold), sha-fence, fire the continuation with the tranche
in the diet, and measure the head's first steps into the wild register.


## 2026-08-20 EOD: THE BRIDGE MOVES
g42_bridge: 25 wild rows @1% dose -> the 25 originals go 5f/0r -> 16f/2r
(TRAIN-FIT; held-out = t2's opening meter, pre-registered). Ring 1242
(paid 11<20). g41 stays deployed. Disk reclaimed to 679G free. NEXT:
tranche 2 (conflict-store-targeted worksheet; parse-before-train meter),
fdiv schema, then the diet cycle again. The bridge builds.


## THE BIDIRECTIONAL INTERFACE (read CORE_COMPONENTS.md top)
Two arrows, one graph: continuous->discrete (silhouette dialects -> the
registry door -> the core) and discrete->continuous (graph -> masks ->
the settle's topology). The one-graph law (V2's grave): both arrows from
the SAME graph. V3 (worded) builds the dialect atlas with mask provenance.
