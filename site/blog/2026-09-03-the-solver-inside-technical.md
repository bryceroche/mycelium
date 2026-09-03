title: The solver inside (technical)
date: 2026-09-03

# The solver inside: Pythia-410M, structured attention, and Sudoku

*The technical version. A companion post tells the same story in
plainer language.*

## The experiment

Take Pythia-410M — a small open language model — and keep only its
first four transformer layers, frozen. Attach a small trained head.
Then do the one unusual thing: replace free attention with
**hand-crafted attention masks derived from the puzzle's constraint
graph** — each Sudoku cell may attend only to its row, column, and
box peers; the mask *is* the adjacency structure of the constraint
satisfaction problem.

Run this stack for K=16 internal cycles — "breaths" — letting the
masked attention iterate. Each breath, information propagates one hop
along the constraint graph: exactly the message-passing pattern of
classical constraint propagation, but carried in a language model's
frozen features.

## The result

It solved Sudoku — reaching 100% on the solve tier — and the same
frozen slice with the same trained head transferred to graph
coloring, circuit puzzles, and KenKen arithmetic, byte-identical in
behavior to a symbolic oracle on the cases it certified. Deduction
depth scaled with breath count roughly as you'd predict from
parallel propagation (minimum breaths tracking problem depth divided
by the per-breath propagation radius).

The load-bearing observation: **we trained no solver.** The frozen
layers had only ever read text. What we supplied was connectivity —
where attention may look — and a thin head to read answers out. The
propagation machinery those layers used was already in the
pretrained weights, latent, waiting for the right wiring to expose
it. We are careful about the claim's strength: this doesn't prove
language models spontaneously grow complete internal CSP solvers;
it proves the *components* of constraint propagation exist latently
in early pretrained layers, and that structured attention assembles
them into a working solver with almost no additional training.

## The telegraph

A second observation, from our archived attention analyses. Fitting
hidden Markov models to attention-entropy traces recorded from our
own earlier models, we found a strikingly clean two-state structure:
in 93 of 104 traces the best-fitting HMM had exactly two hidden
states, switching with 0.97 stability and genuine dwell times — a
telegraph signal. Attention snapping between a concentrated regime
(working a local neighborhood) and a dispersed regime (surveying
globally), and holding each for a while. We suspect — and state as
hypothesis, not measurement — that the same telegraph runs in much
larger models; verifying against archived large-model traces is on
our list.

## The theory

Put the two observations together and a theory writes itself:

**Deductive reasoning in LLMs is an alternation phenomenon.** The
telegraph is the signature: focus (propagate consequences locally),
survey (choose where to work next), focus again. A model doing
deduction in its weights is running an improvised, unreliable
solver-loop internally — with no exactness guarantees, no explicit
constraint store, everything reconstructed from attention on every
forward pass. Until now, LLMs had to build ALL of it inside their
weights, because the architecture offered them nowhere else to put
it.

Our project is the constructive version of that theory: build the
alternation *explicitly*. A neural parser reads wild language into
typed constraint graphs; an exact CSP solver propagates them; the
two alternate inside the deliberation loop — neural commits, solver
propagates forced consequences, committed structure reshapes the
next cycle's attention masks (dynamic masking — the learned version
of the hand-crafted Sudoku masks that started all this). The
telegraph becomes a designed rhythm rather than an emergent tremor;
the solver becomes a real solver rather than a weight-borne
imitation; and the interface between them is typed, inspectable, and
certified.

Hand-crafted masks proved the latent machinery exists. The telegraph
showed the rhythm it wants to run at. The architecture is us finally
building the instrument that plays it on purpose.
