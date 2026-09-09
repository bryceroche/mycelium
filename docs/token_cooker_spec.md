# THE TOKEN COOKER — spec (2026-09-09; registered by the severance probe; the run needs the word)

**Laws:** the mandatory-road law; the headroom corollary (the pot is measured: sever the loop
re-reading and wild falls 0.2511 -> 0.1536, mint 0.9556 -> 0.7704; station 3 carries 2/3 of the
wild headroom, the main bank 1/3; on mint the main bank carries nearly all). **Template:** the
pressure mix / the mask cooker (a per-row (B,1,1) buffer, stable Knuth hash — a THIRD
independent multiplier — flat mix, dose declared, val excluded by an unconditional push,
open rows bit-identical, a read-time door for the meter).

## 1. The design

On a dose of rows (opening dose 0.15), during loop breaths 1..6, BOTH slot->token grounding
attentions — the main bank (`bank(p["fq"], ...)`'s `at`) and ALT21 station 3's (`_sa21`
softmax) — are REPLACED by the MASK HEAD's token-level gate: a per-slot distribution over the
prompt's real tokens, `softmax(g + log tokmask)`, where `g` (B, L_TOT, T) is emitted by a new
token-gate projection of the mask head's per-slot context state (the state the head already
computes per breath from the polar state, the notebook, the solver facts, the committed edges,
the domain mass; the atlas needle joins it at rung 3) against the token bank (a bilinear
score: slot-context @ W_tg @ token-state^T, W_tg zero-born? NO — zero-born makes the gate
uniform AND dead; initialize W_tg as a small random projection so the gate has structure at
birth and a gradient path; the FLOOR is the uniform field, measured: 0.1536 wild). The head's
gate is the only grounding road on sealed rows; open rows keep the banks' q·k bit-for-bit
(per-row blend with m in {0,1}: exact). Breath 0's grounding is never sealed (the floor of
everything).

## 2. Doors

`ALG_TOK_COOK=<share>` (0 = off = byte-identical), `ALG_TOK_COOK_S3=1` (station 3 included;
default 1 — it carries the larger wild share), `TC_EVAL` pushed unconditionally around
_quick_val (val compares OPEN), read-time meter `ALG_TOK_SEAL=head` (a fourth door value:
the head's gate as the grounding on ALL rows, breaths 1..6) beside the existing
`loop`/`all`. Print every door once. Params: W_tg only (H_W x H_W or a factored (H_W x d)(d x
H_W), d = 128 — declare and count).

## 3. Proofs (CPU, staged)

env-inert bit-identical (both polar configs); per-row surgery bitwise; the gate is a proper
distribution (sums to 1 over real tokens, 0 on pads) at every loop breath on sealed rows;
two-terminal: on sealed rows the parse loss reaches W_tg and the mask head's bank and doors
(nonzero) and open rows' gradients equal the pristine head's; the three cookers compose (eight
cells in a batch of 8 exist with independent hashes); `ALG_TOK_SEAL=head` at read == the
all-rows-sealed training path bitwise; --check + idempotence; the guard (`TC_EVAL` unset at
arming; `ALG_TOK_SEAL` barred in do_train as today).

## 4. The twin (needs the word)

Warm from polarsink242, 12k, seed 242, dose 0.15, vs polarsink242c (the control already
exists: 0.2506 / 0.9610; its mask-sealed and token-sealed reads are cheap). BARS: (1) THE HEAD
GROUNDS: token-sealed-wild (`ALG_TOK_SEAL=head`) >= 0.1536 + 0.05 = 0.2036 (the uniform floor
+ 5 points; token-sealed-mint reported vs 0.7704); (2) the census: the head's token gate
carries (a new census organ `tokgate`, post = the gate's KL from uniform per slot, reported);
(3) GUARD: open-wild and mint within -0.02 of the control; (4) the clock stays (probe >= 0.95).
KILL: token-sealed-wild <= 0.1636 after 12k = the head cannot ground from its context;
grounding is content-addressed (q·k), not steerable — and dynamic masking's home is the
bank's own attention, which then gets the cooker's OTHER form (the seal on the RESIDUAL of the
reading: a registered follow-up, not this run).
