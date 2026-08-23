# THE READER POOL MATRIX — round 1 (2026-08-23, word given)

24 candidates for the ~16-reader bank. Every candidate varies MULTIPLE
axes (staggered strides, near-coprime periods — single-axis variants
cluster with the base). Base recipe = G65's verdict (straw version, LR,
steps template). Auditions at 5k steps; hires continue-train to 20k.

THE FENCE: diversity NEVER enters a loss (no disagreement bonus — the
Goodhart front door). Axes only: init lineage, adapter geometry, diet
skew, input convention, objective WEIGHTS (existing gold terms only).

Axes:
- LIN: head-init lineage (g55 / g47 / g51 / g53 / g41 / cap2x*)
- R: LoRA rank (4/8/16/32)
- PROJ: adapted projections (all=wq+wo+wdown / wq / wod=wo+wdown)
- SPAN: adapted layers (A=L0-3 / E=L0-1 / L=L2-3)
- DIET: skew (bal / sys / chain / skel / pert / hum) — shape-sig strata
  for sys/chain; skeleton stratum; perturbation stratum; hum = STRAW_HUMAN 6.0
- IN: input convention (raw / canon)
- OBJW: gold-term emphasis (def / ptr = pointer-heavy / dig = digit-heavy)
  *cap2x rows need ALG_HW=1024 envs; drop to g41 if config friction.

| id  | LIN  | R  | PROJ | SPAN | DIET  | IN    | OBJW |
|-----|------|----|------|------|-------|-------|------|
| R01 | g55  | 16 | all  | A    | bal   | raw   | def  |
| R02 | g47  | 8  | wq   | E    | sys   | canon | ptr  |
| R03 | g51  | 4  | wod  | L    | chain | raw   | dig  |
| R04 | g53  | 32 | all  | A    | skel  | canon | def  |
| R05 | g41  | 16 | wq   | E    | pert  | raw   | ptr  |
| R06 | cap2x| 8  | wod  | L    | hum   | canon | dig  |
| R07 | g55  | 4  | all  | E    | sys   | raw   | dig  |
| R08 | g47  | 32 | wq   | L    | chain | canon | def  |
| R09 | g51  | 16 | wod  | A    | skel  | raw   | ptr  |
| R10 | g53  | 8  | all  | E    | pert  | canon | dig  |
| R11 | g41  | 4  | wq   | L    | hum   | raw   | def  |
| R12 | cap2x| 32 | wod  | A    | bal   | canon | ptr  |
| R13 | g55  | 8  | all  | L    | pert  | raw   | def  |
| R14 | g47  | 16 | wq   | A    | hum   | canon | dig  |
| R15 | g51  | 32 | wod  | E    | bal   | raw   | def  |
| R16 | g53  | 4  | all  | L    | sys   | canon | ptr  |
| R17 | g41  | 8  | wq   | A    | chain | raw   | dig  |
| R18 | cap2x| 16 | wod  | E    | skel  | canon | def  |
| R19 | g55  | 32 | wq   | L    | hum   | raw   | ptr  |
| R20 | g47  | 4  | wod  | A    | pert  | canon | def  |
| R21 | g51  | 8  | all  | E    | chain | canon | ptr  |
| R22 | g53  | 16 | wq   | L    | bal   | raw   | dig  |
| R23 | g41  | 32 | wod  | E    | sys   | canon | def  |
| R24 | g55  | 4  | all  | A    | skel  | raw   | ptr  |

Free pool-fillers: each audition fire's s2000 + final = 2 temporal
candidates (SAME witness axis under the unrelated-witnesses clause).

RECRUITMENT (the covering loop): audition all 24 on the diagnostic
split -> failure-vector disagreement matrix (WL-digest non-isomorphism)
-> greedy max-min hire under the competence floor -> read the UNCOVERED
RESIDUE (items all hires fail) -> design round-2 candidates against the
residue -> repeat until the residue stops shrinking. Even saturation is
a MEASURED OUTCOME, never a design hope.

CONSENSUS GATE (the fingerpost): mouth precondition -> refutation
filter (unsat/degenerate/wall-slack) -> solver -> WL-digest consensus:
>=3 matching digests spanning >=2 witness axes = emit; else refuse.
