# The Semantic Abacus


*The physical abacus decomposed arithmetic into mechanical bead manipulation — separating the computation from the understanding. A child who doesn't understand multiplication can still slide beads correctly and get the right answer. The mechanical procedure IS the computation. Understanding is unnecessary.*

*This project builds a semantic abacus for mathematical reasoning. The "beads" are text spans identified by attention dynamics. The "rods" are operation templates discovered by Information Bottleneck compression. The "sliding" is a symbolic executor that performs the actual mathematics. Six small models learn to operate the abacus. None of them understand mathematics. All of them, together, solve it.*

---
## The Semantic Abacus

The physical abacus was one of humanity's great cognitive inventions. Not because it could compute — any arrangement of stones can tally — but because it decomposed computation into a mechanical procedure that required no mathematical understanding to execute. A merchant who couldn't explain why multiplication works could nonetheless compute the price of 48 bolts of silk at 2 drachmas each by sliding beads along rods. The knowledge lived in the device and the procedure, not in the operator.

Our architecture follows the same principle. Consider the problem: "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?"

A large language model solves this by generating chain-of-thought: "48 / 2 = 24 clips in May. 48 + 24 = 72 clips altogether." This works, but requires the model to understand division, understand "half as many," generate coherent text, and compute accurately — all simultaneously, in a single forward pass per token.

The semantic abacus decomposes this:

**Beads.** JSD segmentation identifies the operand-carrying spans in the problem text: `[48 of her friends]`, `[half as many clips]`. These are the beads — discrete units of meaning, identified by where the teacher model's attention shifts during its own reasoning.

**Rods.** Information Bottleneck compression discovers that "half as many" belongs to the same operation template as "a third of," "twice as much," and "four times the amount" — they all reduce to a multiplicative scaling operation. The rod is `MUL(quantity, fraction)`. IB found 10 such rods on GSM8K, and 115 on MATH, without being told what to look for.

**Sliding.** A symbolic executor computes `48 / 2 = 24`, then `48 + 24 = 72`. It doesn't understand the problem. It receives typed arguments and an operation label, and it evaluates. The mechanical procedure IS the computation.

**The operator.** Six small models (0.5B parameters each) learn to operate the abacus — identifying beads, placing them on rods, and deciding when to slide. None of them understand mathematics. The segmenter recognizes span boundaries. The classifier picks the rod. The extractor reads the bead values. They are merchants, not mathematicians.

The physical abacus had a property that made it revolutionary: it was **portable**. The knowledge was in the device, not the operator. Any merchant could use it. The semantic abacus shares this property. The knowledge lives in the IB-discovered template structure and the symbolic executor. The 0.5B models are replaceable operators. Swap the executor for one that does chemistry and the same architecture could work on stoichiometry problems. The abacus doesn't care what you're counting.


## λ — The Hidden Network

Lambda — from the Greek λανθάνω (*lanthano*): to be hidden, to escape notice.

Lambda operates at every level of this project:

**λ the function.** The project recovers hidden functions from text. "Bob bought 5 apples at $2 each" contains an invisible `λ(price, quantity): price × quantity` that the teacher model computes but never names. The chain-of-thought is a trace of lambda evaluations. We extract those lambdas, give them names (IB templates), and distill them into small models that can evaluate them directly.

**λ the IB parameter.** The compression parameter β controls the information-relevance tradeoff — literally a dial that governs how much hidden structure becomes visible. At β = 0, everything is hidden (one template). At β → ∞, everything is visible (each span unique). The plateaus between these extremes are the moments where genuine structure crystallizes from noise. β is the lambda that reveals lambdas.

**λ the bridging operations.** Templates T7 (accumulator), T0 (derived division), T9 (derived multiplication) are anonymous functions in the truest sense. No span in the problem text maps to them. No words describe them. They exist purely as computation — the implicit operations that bridge explicit results into final answers. On GSM8K, 26.2% of all computation steps are these invisible lambdas. IB discovered them from structural signatures alone: late chain position, zero operands from problem text, all derived inputs.

**λ the eigenvalue.** In the IB annealing, the phase transitions where template count jumps — those are bifurcation points. The system's eigenvalues cross zero. Lambda as the parameter that governs when hidden structure becomes visible is literally what β does in the annealing curve. The template count is a step function of β, and each step is a lambda becoming visible.

**λ the mycelium.** Mushrooms appear on the surface — visible, countable, apparently independent. Underground, an invisible network of mycelium connects them, decomposes organic matter into simple molecules, and distributes nutrients across entire forests. The explicit operations are mushrooms. The implicit bridging operations are mycelium. The hidden network beneath the surface, connecting the visible structures, is made of lambdas. IB makes the mycelium visible.

The entire architecture is a lambda-recovery machine. JSD finds where lambdas are evaluated (attention shifts). IB discovers what the lambdas compute (template structure). The executor evaluates them (symbolic computation). The six small models learn to invoke them (span identification + operation classification). And the name — Mycelium — is the hidden network of lambdas that connects everything we can see.


The lambdas are everywhere.