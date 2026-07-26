# FIRE 1 DESIGN (draft for countersign — 2026-07-26)
Constraint inheritance: Fire 0's science (pressure cheap, differentiation
unbought; gates not yet load-bearing — trim tabs on an unturned rudder).
The C4 bar (ladder slope <= -0.05) STANDS UNCHANGED — the redesign does
not renegotiate the bar it failed. Refused band pinned by precedent:
(-0.05, -0.0037) = informative, unauthorized.

## 1. The centerpiece: DEPTH-SCHEDULED PER-BREATH SUPERVISION
The honest note first: v98's per-breath CE ladder supervises every breath
on the SAME gold — it asks each breath to improve, not to DIFFER. Fire 0
proved improvement-pressure reaches gates without buying differentiation.
The ask that demands differ-BY-CONSTRUCTION: **breath k's loss masks to
variables whose derivation depth <= d(k), with d increasing across
breaths** — each breath owns targets the previous breath's mask did not
cover; the gradient says "know something new," not "be generally better."
This is the week's depth-axis discovery converted into an objective: the
engine's measured anatomy (values resolving in depth order, ~3.5
breaths/layer) becomes v200's TARGET dynamics rather than a hoped-for
emergent. Schedule d(k): linear map of k in [0,K-1] to depth in
[1, D_max] (K=8, factor-graph depths ~1-4 -> d(k) = 1 + floor(k/2)),
with ALL depths unmasked at the final two breaths (the full-answer
breaths — no content permanently orphaned).
Implementation note: per-var derivation depth = topological depth in the
record's graph (compute at load; records carry graph structure; verify
loader exposure FIRST — if absent, a 20-line topo pass in the loader).

## 2. Retained + inherited
- Improvement-pressure RIDER retained (V200_IMPROVE_W=0.5 — reaches
  gates, costs nothing, demonstrably durable).
- ALIGNED CODEBOOK INIT enters (v98's aligned-init principle; the
  carry-forward table's Fire-0 ablation slot rides Fire 1 instead:
  aligned vs cold arms if budget allows, aligned as default).
- Per-breath markers (breath_embed) already present; two-terminal
  exclusion list expected UNCHANGED (26 names — drift in that list is
  an apparatus alarm).
- Early-kill inherited verbatim (25% budget, excursion + spread).
- SBP arm rides as planned (escape valves during differentiation).

## 3. The new dashboard line: THE STAIRCASE, LIVE
Per-breath MARGINAL accuracy printed at every PB_EVERY: acc(breath k) -
acc(breath k-1) per depth stratum — differentiation watched as a ladder
FORMING, not inferred from a spread statistic post-hoc. The calib
false-friend marking stands.

## 4. Bars (pinned before any code)
- C4 UNCHANGED: eval ladder slope <= -0.05. PASS -> Fire 1's other bars
  read (gates open >= 0.05 excursion held; revision signature per the
  quantitative form; beats frozen 0.410 by 0.15). FAIL inside the
  refused band -> informative, unauthorized, and per the countersign:
  no longer a schedule question — EVIDENCE ABOUT TRAINABILITY AT THIS
  SCALE; the gate's word gets rethought on it.
- Budget: Fire 0's scale (500 steps) until the slope says otherwise.
