# Mycelium — Master Design Doc

Oct 11, 2026 · Bryce Roche (drafted with Opus; edited against the ledger by Claude)

## Goal and core theory

Mycelium maps natural-language math text (GSM8K, then MATH-500) to factor graphs that a CSP solver can compute exactly. The answer key gates all training data; MATH-500 and the wild holdout are measured, never trained on.

**The solver inside (the thesis).** Our working theory is that LLMs carry something like a CSP solver inside their weights, and that they alternate between proposing structure and checking it. Deep stacks of MHA blocks give a model the runway to run that alternation. This is the thesis the project is built on, not a settled result; what the ledger has measured so far:

- **June:** a small breathing deducer over four frozen pretrained layers (Pythia L0–L3, shared across 16 breaths) learns propagation that is byte-identical to the KenKen oracle and generalizes across coloring, circuits and KenKen. Trained, not zero-shot; the frozen layers were enough substrate.
- **June, the other half:** on clean CSPs, symbolic search dominates neural guidance (neural-guided clean-CSP search is closed). The neural side's job is reading the text into the graph, not solving the graph.
- **Attention analysis (Bryce's reading):** larger Llama and Qwen models show telegraph-like alternation signals in their attention. We read this as the propose/check alternation happening inside the weights. Not yet a ledger measurement.

**The design consequence.** Mycelium makes the alternation explicit: a loop of breath cycles that alternate between neural blocks and a real CSP solver. *Neural proposes, symbolic disposes* — abstraction may live in annotation and recognition, never in verification.

## Guiding principles

Each breath crosses one bridge: from continuous neural space to discrete solver space and back.

1. **Alternate every breath.** Neural blocks propose, the CSP solver disposes. *Status:* today the solver is consulted after breaths 2 and 4 and at the final decode; every-breath alternation with hard feedback is registered (the six alternations), behind a zero-training pre-read in the replay gym and head diff A (the early breaths lack the role pointer).
2. **Every breath expands and collapses.** The axes:
   - Solver: the main collapse, refuting inconsistent graphs and shrinking domains by forced moves.
   - Tokens: the token states are rewritten each breath (the sorting room; every breath in SRF, on the card now). Written summaries in words do not exist yet: the head points, it does not generate.
   - Slots: the 24 slots that become the equations handed to the solver.
   - Embedding dims: the waist, 384 content dims squeezed to 128 and back each breath.
3. **Dims follow a sine wave** inside the looped transformer, so compression and expansion ratios stay within limits. *Status:* the sine breath (widths 128/224/384/224 per block, the trough inside the crystal) is built; its first arm was confounded by a skip bug and is re-queued with the lawful no-skip form and the FFN sized at 4× each block's width.
4. **Coarse to fine.** Every breath does the same job at a finer granularity, like a diffusion compiler with progressive lowering (MLIR). *Status:* the loss keeps the waist identical every breath and refused a funnel (FG_241); coarse-to-fine lives in the decode (commits, forced moves), not in the state's rank.
5. **Breaths don't specialize; heads do.** Specialized breaths cause a gradient tug-of-war (a negative cross-rung cosine). Supported.
6. **U-Net concepts, not a U-Net.** Compression plus skip-connection upsampling, inside a looped transformer. *Status:* compression holds (the loss defends the squeeze from three sides); a skip across the trough is refused; the lawful skip runs around the peak, and the hourglass on the token side (tokens → clauses → sentences → tokens) is live at 98% and did not move the binding.

## Architecture

The frozen trunk builds the embedding space once; the breath cycle then runs six loop breaths after a grounding read, alternating the neural loops with the solver.

```
text ─► frozen Llama L0–L3 (2048) ─► sorting room: 4 trained MHA blocks (512)   [once]
          │
          ▼
   breath 0: the grounding read (24 slots read the tokens)
   breaths 1..6: ┌ bank read (slots ← tokens) ─ slot-side blocks ─ notebook ─ waist 384→128→384 ┐
                 │                           consults after breaths 2 and 4 (solver → route bias) │
                 └──────────────────── T6 clock rotates the clocked planes each breath ───────────┘
          │
          ▼
   decode (pointers → factor graph) ─► candidates (≤64) ─► perceiver / judge ─► CSP solver ─► answer key
```

The accent line is the perceiver's feedback: dynamic masks and one-hot branch toggles that shape the next breath (registered; the judge's first read is 23 of 311 against a bar of 24). The silhouette of the Dancer is the math loop's bottleneck, the 128-d crystal.

## The organs

The perceiver is the only organ that sees the atlas; everything else feeds it or takes its orders. The factor graph itself is decoded by the pointer heads — kinds are classifiable from a slot's state, bindings are not (the binding theorem) — so the perceiver picks among what the pointers propose; it does not replace the decode.

| Organ | Also called | What it does | Status (2026-10-11) |
| --- | --- | --- | --- |
| Atlas | the bridge | Centroids per kind maintained with Welford stats, frozen at read, each with a DSL tag. Straddles continuous and discrete space. Tolstoy principle: correct operations cluster, wrong ones scatter. | Built (one library per body). The cosine to the own-kind mean predicts slot correctness at AUROC 0.766 — a monitor and a judge's angle, never a decoder (the Goodhart fence). Tree structure registered. |
| Silhouette of the Dancer | the bottleneck | The 128-d waist each breath; the state the loop carries. | Load-bearing (−0.064 at read); the loss keeps it shut from three sides. The sine breath makes it the room where slots meet. |
| Perceiver | the pilot (AlphaZero-style) | Reads the loop's states against the atlas (JSD responsibilities), reviews the candidate graphs, sends accepted ones to the solver, reads the feedback, toggles branches with a one-hot, straight-through circuit breaker. Reads system metadata across the breaths. | The learned judge over candidates reads 23/311 (bar 24) with the atlas angles; the minimality angle joins it. Toggles inside the loop: registered. |
| T6 clock | the conductor | Six helical waves on 64 clocked planes (128 dims). Keeps time for every organ. | Rides as a turned quarter of the bus; separating it into a register (CS_241) kept time (probe 0.996) but cost 0.011 wild and 0.063 mint, confounded with a waist change. |
| T256 bus | relationship tracker | Phasor binding as an active recurrent loop; bands root/branch/leaf by what reads them. | Built. The per-breath descent down a tree (sentence → clause → mention → token) was null in six forms; the band partition with damping cured the drift, not the wall. |
| Math certifier | the solver | The CSP solver (GAC/MRV/LCV, forced-only commits, a decision budget). Runs the proposed graph and returns results. | Built; exact; downstream of the binding (garbage in). A minimality check (every fact needed) adds teeth on time-twin errors: catch 0.41, false 0.23 — an angle. |
| NL certifier | the fingerpost panel | A panel of judges over multiple views, after *An Instance of the Fingerpost*. | nl_certifier v1 at AUROC 0.811; the courtroom's untrained jurors are inert; a trained juror is registered. |
| Notebook | per-breath log | Writes the state each breath in the canonical clock frame and reads it back; the main road at late breaths. | The most load-bearing organ (−0.134 at read). The all-to-all slot mixer beside it is dispensable (HUB_241). |
| Eyes and hands | U-Net + dynamic masking | The U-Net is the perceiver's eyes; dynamic masking is its hands. | Hands: masks and claims are the only feedback that has moved structure. Eyes: the soft-mask form was killed; the form on the partitioned state is untouched. |

Factor-graph node order follows a wave function through the loops: the clock's rotation is the breath-side RoPE.

## Learnings

The model was starved for surface area, not capacity: the wins come from what the loop reads, not how big it is.

- **NL needs more dims and compute than math.** The trained token side (the sorting room) is the slot record, a family of four bodies at about +0.015 over the lineage, with the gain in digits and forward relations; extra slot-side depth was null with every block live.
- **Compression ratio must be limited.** The loss defends the 384 → 128 waist and refuses both a skip around it (WS_241) and a funnel through it (FG_241); widening the waist's domain to 512 cost (CS_241).
- **Alternation must happen every cycle.** Soft feedback has been null repeatedly; hard forms (the legality mask, claims) move structure. The row's fate is set by breath 2, so the consult must come before it.
- **Surface area matters.** The state must connect back to the token embeddings, every breath. SRF_241, the sorting room re-run each breath with the slots as queries, is the arm testing this.
- **The frozen trunk runs once; the loop runs six times.** Four frozen Llama layers plus the trained sorting room build the embedding space once; the breath cycle repeats six times after the grounding read.
- **The wall is supervision, not context, capacity or ownership keys.** The first look picks the twin (the same entity at another role or time) at the same rate on every body; the hourglass gave every token its sentence context and the binding did not move; the mixer groups by owner on token-side bodies without help. The training text's twins are memorized by surface, wild's are not. The data road (unmemorized prose with annotated twins; inverse forms in the wild register) is the open lever.
- **The diet is over-annotated and in-sample.** Pen-annotated book rows are 0.61 minimal (four dead facts per correct row); the valid2 slice is memorized, so calibration needs a third held-out pool. Train, calibrate and measure on disjoint pools.

## Hills to die on

These are convictions to keep testing fairly, not settled results. A null is a verdict on a form, never on the hill.

- **The U-Net is the perceiver's eyes; dynamic masking is its hands.**
- **Search for the answer with MCTS and MDP.** The U-Net is the continuous search (the breath trajectory, sampled by the five views); MCTS over the candidate lattice is the discrete search, with the perceiver as its value and the solver as its rollout.
- **The atlas is tree-structured centroids with Welford stats.**
- **Hierarchical embeddings map to the tree-structured codebook**, perhaps through MLP or MHA blocks.
- **Testing and contracts.** Unit, feature and integration tests plus API contract enforcement. What stands in for it today: CPU bit-identity gates on every head edit, the birth check (identity at every breath under the arm's own env), the knob census (an organ's injection against its port's spread), the pre-arm device smoke, pre-flight contracts on every chain, and a queue dry-run that tests every check against its gate log before a queue is armed. Conventional unit tests are thin; a standing gap.

## Engineering

| Area | Choice |
| --- | --- |
| Framework | tinygrad with TinyJit (not PyTorch); the AM driver, no ROCm; `DEV=PCI+AMD` |
| Hardware | AMD 7900 XTX (24 GB), one card; long work as systemd user units |
| Performance | The solver runs on the host between breaths, outside the JIT graph; consults cost 2.2× the step. Fixed-buffer training steps at ~0.1 s; organs that grow the graph past a threshold need `JIT_BATCH_SIZE=1` |
| Quality | The gates above; one organ per arm, from scratch against a named control, with bars pinned before the read and a seed twin for any claim |

## Open tensions

Places where the vision and the ledger (as of 2026-10-11) don't yet agree.

| Vision says | Measured so far | Open question |
| --- | --- | --- |
| Math loop at 256 or 128 dims with a 2× waist | 384 content dims through a 128 waist; the labels need roughly 104–144 dims | Does a 2× squeeze below 128 cut under what the heads read? The sine breath's crystal attention is the cheap form. |
| T6 clock lives outside the loops at 128 dims | Separating it (CS_241) cost 0.011 wild and 0.063 mint, confounded with the waist change | Is the clean variant (register only, waist unchanged) worth an arm? |
| The solver is the main collapse | Most wrong wild graphs solve and are unique; the minimality check catches 41% of them at a 23% false-refutation rate | Can the check become a filter once the gold graphs are minimal (an admission gate on the diet)? |
| Dims funnel across breaths | The funnel gate drew no envelope: the loss keeps the waist identical every breath | Is the sine breath a better shape? SINE4/ANTI4/SEAL re-fire with the lawful skip. |
| Tokens carry summaries and intermediate results | The head points, it can't write words; chalkboard tokens went unread | What would write them, and as a mandatory road? SRF's re-sorted tokens are the nearest built form. |
| 2–4 Llama blocks run once | Four frozen Llama layers plus a 4-block trained sorting room at 512 | The sorting room is the vision's "2–4 blocks"; a wider one pairs with more data, not more width (mint +0.2 against wild +0.02). |
| The all-to-all mixer binds the slots | Removing it at training (HUB_241) costs nothing on wild and 0.07 on mint | Drop it from the recipe for step time once a twin confirms? |
| The twin problem collapses somewhere | No axis collapses it: the solver passes both readings, the vote agrees with itself, the pool shows both twins | Annotated twins in unmemorized prose — the data road. |
