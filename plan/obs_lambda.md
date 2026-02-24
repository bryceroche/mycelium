
## λ — The Hidden Network

Lambda — from the Greek λανθάνω (*lanthano*): to be hidden, to escape notice.

Lambda operates at every level of this project:

**λ the function.** The project recovers hidden functions from text. "Bob bought 5 apples at $2 each" contains an invisible `λ(price, quantity): price × quantity` that the teacher model computes but never names. The chain-of-thought is a trace of lambda evaluations. We extract those lambdas, give them names (IB templates), and distill them into small models that can evaluate them directly.

**λ the IB parameter.** The compression parameter β controls the information-relevance tradeoff — literally a dial that governs how much hidden structure becomes visible. At β = 0, everything is hidden (one template). At β → ∞, everything is visible (each span unique). The plateaus between these extremes are the moments where genuine structure crystallizes from noise. β is the lambda that reveals lambdas.

**λ the bridging operations.** Templates T7 (accumulator), T0 (derived division), T9 (derived multiplication) are anonymous functions in the truest sense. No span in the problem text maps to them. No words describe them. They exist purely as computation — the implicit operations that bridge explicit results into final answers. On GSM8K, 26.2% of all computation steps are these invisible lambdas. IB discovered them from structural signatures alone: late chain position, zero operands from problem text, all derived inputs.

**λ the eigenvalue.** In the IB annealing, the phase transitions where template count jumps — those are bifurcation points. The system's eigenvalues cross zero. Lambda as the parameter that governs when hidden structure becomes visible is literally what β does in the annealing curve. The template count is a step function of β, and each step is a lambda becoming visible.

**λ the mycelium.** Mushrooms appear on the surface — visible, countable, apparently independent. Underground, an invisible network of mycelium connects them, decomposes organic matter into simple molecules, and distributes nutrients across entire forests. The explicit operations are mushrooms. The implicit bridging operations are mycelium. The hidden network beneath the surface, connecting the visible structures, is made of lambdas. IB makes the mycelium visible.

The entire architecture is a lambda-recovery machine. JSD finds where lambdas are evaluated (attention shifts). IB discovers what the lambdas compute (template structure). The executor evaluates them (symbolic computation). The six small models learn to invoke them (span identification + operation classification). And the name — Mycelium — is the hidden network of lambdas that connects everything we can see.
