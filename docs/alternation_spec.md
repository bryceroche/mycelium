# The Alternation Spec — one setup breath, six alternating breaths, five keys, a predicate mask, a perceiver

**Registered:** 2026-09-24 (Bryce's six-point focus, 09-23 late; the register family's book behind it).
**Status:** SPEC. Nothing here is built until its stage is gated. Bars pin before measurement.
**Governing laws:** the mandatory-road law (a message rides a road with no bypass), the residual-seal law,
the pre/post knob law (census the injection on the WARM body against the port's other terms before any fire),
the chassis ruling (the trunk is a retina, never rewritten), the Goodhart fence (diagnostics are inputs, never
losses), and the week's rules (a pointer organ's cells read on held-out rows; a warm gate's bar is post vs post;
teacher-force only where the teacher matches what the read is handed).

## 0. What the book says going in

Eight organs on the bus and the bank read (09-22/23) — the register, its propagation, the addressing road, the
value pointer read, the presence map, the seal, the fade-in, the identity key at two scales — were exact in their
algebra, in the loss, bit-identical unset, and none moved a held-out decision the right way. Two of them moved
cells the WRONG way at 2.6-2.7 SE. Two facts survive them and govern this spec:

1. **A message must be on the mandatory road.** Every organ that died was an additive term the loss could route
   around; the two that live (the router's bank bias, the role channel) sit on the slots<-tokens read itself.
2. **The pointer wall is role and time among identical identities.** Same-quantity candidates share an identity by
   construction (the twin-split and mixed-bucket censuses); an identity booster on the read blurs which-mention /
   which-time (the tau-32 read). Identity binds a token to a slot ONCE; it never chooses among slots.

And the one lever with a measured slope: **n**. Annotated wild rows moved the binding wall +31% at n=100; nothing
else has. The annotation stream (§5) is a data program, not an organ, and runs beside every stage below.

## 1. Consistency: the rule and the four mismatches

**The rule.** Nothing the loop consumes may be computed one way at training and another at read. Teacher-force only
where the teacher's distribution matches what the read is handed; otherwise the model trains on its own outputs,
detached where they are beliefs, gradient-carrying where they are content.

The four mismatches measured this week, and their fixes:

| what the loop consumes | at training today | at read today | the fix |
|---|---|---|---|
| the slot->variable map (scatter) | gold res_map | predicted from heads | predicted at both (ALG_BUSREG_PREDMAP's form), detached |
| the consult's facts (values) | gold, dense (11/row) | live pass-1 singletons (3.4/row, errors) | live facts from the body's own consult, refreshed per segment (built: ALG_VALREG_LIVE + RESUME segments); in-loop at breath 4 once §2 exists |
| the consult's facts array (ALT2) | one mask-prep pass at run start (empty for scratch) | live pass-1 | the same per-segment refresh |
| the slot masks | the same stale pass | build_slot_masks live | **the predicate mask (§4): computed in-graph from the keys at both times — identical by construction** |

The gate for this stage: with the four fixes on and every organ off, the body's wild reads must not fall (the
consistency fixes are not allowed to cost), and the read-time and train-time forward on the same row must agree
slot-for-slot (a bit-identity check on the val two-pass vs the trainer's forward with the same inputs).

## 2. The lifecycle: one setup breath, six alternating breaths

```
breath 0  SETUP (NL)     retina (Llama L0..L3, frozen, memmapped) -> waist (512) -> the grounding read
                         slots <- tokens on {identity, contextual}; the IDENTITY KEY runs HERE and only here,
                         on the GIVEN channel's read (a slot finds its numeral once); the slots' registers
                         (role, identity) are written once.
breath 1  NL-driven      slots <- tokens on the five keys (§3); slot mixing under the predicate mask (soft);
                         the router's channels; spans graded.
breath 2  math-driven    tokens <- slots: the UPDATED-MATH key = the attention output over slots (no gain,
                         no bypass: the key IS the output); the shared key updated.
breath 3  NL-driven      as breath 1, reading the updated keys.
breath 4  math-driven    + THE CONSULT: the committed graph -> the solver (map over rows in the facts pool,
                         reduce into one fixed device buffer; the JIT stays as two captured graphs, breaths
                         0-3 and 5-6, one host hop between) -> implied values into the slots' value stream.
breath 5  NL-driven      the pickup: slots read tokens on keys that now carry the solver's values; the
                         text-side presence check (a corroborated value pulls the given channel to its numeral).
breath 6  math-driven    finalize under the HARD mask; emission.
```

Six loop breaths = three drop-off / pickup pairs = the six-wave clock's count. The clock sets the ticks; breath
parity is the schedule; both parties know the curb in advance. The ladder grades every breath (both loops in the
loss). The retina is never rewritten: every token carries its keys BESIDE its trunk state.

**Until the in-loop consult exists**, breath 4's values come from the per-segment refresh (§1). The consult's
map-reduce is a perf build with its own wall: the slowest row of a batch of 8 sets the reduce; a per-row timeout
returning "no values" keeps it bounded (a slot with no propagated value reads its stream empty, stated).

The gate for this stage (the skeleton, no new keys, no mask): the even-breath road alone (breath 2/4/6 tokens<-slots
into the updated-math key, read by breaths 3/5) vs the body, warm at 1e-5 with the cont-control; bars: the wild
SAME-NOUN cell paired >= 2 SE (the cell that needs role and time), masked wild paired >= -0.005, rows beside.
Before the fire: the knob census of the updated-math key's spread against the router's bias on the read.

## 3. The five keys, never summed

Every token presents five key streams to the slots<-tokens read; each has its own fixed temperature; none is
summed with another; the softmax sees their sum of SCORES, so no stream can be routed around and no stream can
impersonate another.

| key | what it is | when it changes | temperature |
|---|---|---|---|
| identity | the token's exact lexical tag (the identity table; Llama's input embedding, centred, fixed map, unit) | never | fixed; breath 0 only, given channel only (the tau-32 lesson) |
| contextual | the waist (512-d projection of the retina) | never within a forward | the bank's own |
| updated math | what the slots told this token on the last math-driven breath (the attention output over slots) | breaths 2, 4, 6 | fixed, set by the knob census |
| updated NL | the token's own NL state after the last NL-driven breath (the token-side read of its neighbours under the same keys) | breaths 1, 3, 5 | fixed, set by the knob census |
| updated shared | the interface state both loops write: the running mean of the two updated keys, re-normalised | every breath | fixed |

Queries: a slot's query has one half per key stream, projected from its state for the contextual / updated
streams, and for identity ONLY at breath 0 (the pooled tag of the grounding read). The slot side never carries
an identity query into the loop breaths.

Register vectors (role, identity, value on the bus) stay as built: a language and a channel; they are READ by the
predicate mask (§4), which is the reader the register never had.

## 4. The DSL: dynamic masking, soft to hard over six breaths

The slot mixer's attention (slots<->slots) and the tokens<-slots read (math breaths) run under a MASK that is a
boolean-valued formula over PREDICATES computed exactly from the keys and the registers — no parameters, no
learned mask head (retired), identical at training and read by construction.

Predicates (each a (B, L, L) or (B, T, L) matrix in [0, 1]):

```
same_identity(k, k')   cos(id_k, id_k') > theta_id           (the unbound identity streams; givens only)
same_role(k, k')       cos(role_k, role_k') > theta_role
earlier(k, k')         sign of the precedence term (j - k)/L_FAC, the role signature's own
consumes(k, k')        A[k, k'] > theta_A  (the previous rung's args belief through the slot->variable map, detached)
same_sentence(k, k')   the tokens' sentence ids (the mc_sent skeleton)
forced(k)              the consult marked k's variable a singleton (breath >= 4)
```

The DSL: a mask is a formula in these predicates with AND / OR / NOT, e.g. `consumes OR (same_sentence AND
same_role AND NOT same_identity)` for a relation's candidate set. Masks are specified per breath as text in the
recipe (an env, parsed once), so a change of mask is a change of recipe, never a head edit.

**Soft to hard.** At breath 1 the mask enters as a bias beta_1 * (mask - 1) on the attention logits with beta_1 small;
beta_k rises over the six breaths on a fixed schedule (the fade-in's form, a schedule not a gain) to a value at
breath 6 that is effectively hard (-1e4 on the masked-out cells). The first appearance is the SOFT form at fixed
betas, censused pre and post like every road; the hard form earns its place only if the soft form shows a removal
cost on wild.

Gate: the mask alone on the certified body (warm, cont-control), bars: the wild SAME-NOUN cell paired >= 2 SE;
masked wild paired >= -0.005; the different-noun cell holds. The mask's removal cost at read (sever it on the
trained arm) is quoted beside the bar (the headroom corollary).

## 5. The annotation stream for training data (the data lever)

The books campaign's instrument, resumed: labelled factor and value spans on wild rows, the rulebook as written
(consecutive letters; values <= 300; one fdiv per item; frame-strip flags; knowns explicit, unknowns never gifted),
three lanes (machine-banked / repair / surgery), the 5-view vote + the answer key as the gate, v2-retry before
fresh surgery. Dose law: declare share-of-mix and reps-per-unique. The census pool stays a fixture.

This runs BESIDE every stage above, because it is the only lever with a measured slope. Every organ gate in this
spec reads on the wild holdout; the holdout never enters the diet.

## 6. The perceiver: a monitor that steers the mask, never the state

The perceiver is retired as core and sanctioned as monitor / segmenter (the June ruling). Here it is exactly that:
it reads META-DATA from the loop each breath and outputs the NEXT breath's mask parameters (the betas, the
predicate thresholds, which formula) — it adjusts the trajectory through the mask, never by writing into a state.

Inputs per breath (all registered DIAGNOSTICS — never in a loss; as INPUTS they are lawful):

- distance from the atlas centroid: Welford mean / variance per WL class over the updated-shared key (the
  interface state), the radius as the consolidation clock, the angle as identity;
- the NL certifier's feedback: the text-side presence checks (does a corroborated value sit in the text at the
  address the identity names), per slot, per value — evidence, never pass/fail;
- the math certifier's feedback: the consult's domain mass per variable, contradictions (an emptied domain),
  the number of forced singletons;
- the router's own readings: channel entropies, the same-sentence competitor count per relation.

Output: the mask schedule for the next breath. Fixed rules first (a table: e.g. a contradiction at breath 4 ->
loosen `consumes` at breath 5; a far-from-centroid row -> hold the mask soft one more breath); a learned policy
only after the fixed table shows a removal cost. The perceiver arrives LAST: its inputs mean nothing until the
atlas (Welford over the interface key) and both certifiers exist.

## 7. Build order and the gate for each

| stage | what | gate (all on wild, post vs post, cont-control at 1e-5) |
|---|---|---|
| 1 | consistency: predicted map, live facts, in-graph masks; identity key at breath 0 only | no wild cost; train/read forward agree slot-for-slot |
| 2 | the lifecycle skeleton: the even-breath road (updated-math key) | same-noun cell >= 2 SE; masked >= -0.005 |
| 3 | the five keys complete (updated NL, shared) | same-noun cell holds or rises; digits field beside |
| 4 | the predicate mask, soft; then hard | same-noun cell >= 2 SE; different-noun holds; removal cost quoted |
| 5 | the in-loop consult at breath 4 (map-reduce, two captured graphs) | digits field >= +0.030; step cost quoted |
| 6 | the atlas (Welford on the shared key) and the two certifiers as readings | diagnostics only; no bar — inputs for 7 |
| 7 | the perceiver: the fixed rule table steering the mask | same-noun cell and rows; the table's removal cost |
| n | the annotation stream | the odometer and the disjoint census, as in Book 2 |

Every head edit rides one facts-pass tax and carries THE COMPLEX-TENSOR FENCE (the head's inline rotate asserted
equal to `mycelium/complex_tensor.py`'s rotate and numpy bind in the CPU gate). Every stage's injection is
censused on the warm body before its fire. The word fires each stage; nothing here fires on its own.
