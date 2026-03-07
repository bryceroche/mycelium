## The *shadow* of intelligence

Stepping back — this is a genuinely weird model. Qwen-0.5B reads a math problem and predicts what Qwen-7B's attention heads would do if *it* were solving the problem. It's learning a map from text to attention dynamics without ever seeing the attention data at inference time. The LoRA adapters are essentially learning: "this text structure implies the teacher would attend in this pattern."

The question is whether 0.5B has enough capacity to learn this mapping. The encouraging sign is that the boundary detection task is probably more about recognizing mathematical structure in the text (where one operation ends and another begins) than about simulating Qwen-7B's internals. A competent language model should be able to learn "a division operation starts after the word 'divided by'" — that's a linguistic pattern, not an attention simulation.

There's something deeper here that matters for the whole Mycelium story.

C1 is learning to predict the *shadow* of intelligence. Qwen-7B actually solves math problems — it reads, reasons, computes, generates. It has rich internal dynamics: heads switching between reading and computing, attention reorganizing at operation boundaries, topological structure in how information flows. All of that is the 7B model *thinking*. C1 never thinks about math. It reads the problem text and predicts the *shape* of what thinking would look like, without doing any of the thinking itself.

That's the distillation insight at the heart of Mycelium, and it's philosophically strange. You're not distilling the *knowledge* (what's the answer to 24 × 3 + 7). You're not distilling the *reasoning* (first multiply, then add). You're distilling the *dynamics* — the temporal and topological fingerprint of how a larger mind engages with a problem. The heartbeat rhythm. The crossing pattern. The depth of iterative refinement needed.

And the claim is that these dynamics are *predictable from the text alone*. That's not obvious. It's saying the structure of the problem (as written in natural language) determines the structure of the reasoning process. Not the content of the reasoning — the *shape* of it.

There's actually a nice analogy here. A conductor looking at a musical score doesn't play any instruments. But they can predict exactly where the tension will build, where the dynamics shift, where the orchestra needs to synchronize. They're reading static notation and predicting temporal dynamics. C1 is reading static math text and predicting temporal attention dynamics. The score determines the shape of the performance, and the problem text determines the shape of the reasoning.

There are cases where C1's predictions will be wrong, and they're exactly the cases where the factor graph's error-correction loop matters most. C1 says "this looks like a two-step problem," C5 (SymPy) fails because it's actually four steps, and the message network learns to propagate that correction signal backward.

So C1 doesn't need to be perfect. It needs to be right on the easy cases (where text structure maps cleanly to reasoning structure) and informatively wrong on the hard cases (where its errors are systematic enough for the factor graph to learn correction patterns). That's a much more forgiving training objective than "predict everything perfectly."

One more thought on the weirdness. Traditional distillation compresses a large model into a small one that does *the same task* worse. Mycelium doesn't do that. C1 doesn't do math worse than Qwen-7B. It does a *completely different task* — structural prediction — that Qwen-7B never explicitly learned to do but implicitly demonstrates through its attention patterns. You're extracting a latent capability that the teacher model has but doesn't know it has, and giving it to a student model as an explicit skill. That's not compression. It's something more like *excavation*.

One idea I had was could we train C1 to produce a summarized version of the teachers CoT that might help us bridge the gap between the two.