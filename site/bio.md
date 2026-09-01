# Bryce Roche

Bryce Roche is an independent AI researcher, technologist, and lifelong endurance athlete whose path has taken him from the mountains of Montana to collegiate competition in Vermont, open-water racing in the Pacific, and now the problem of how machines can learn to reason.

Bryce grew up in **Bozeman, Montana**, where Nordic skiing and distance running became a major part of his life. As a junior Nordic skier, he competed nationally and earned **All-American honors at the 2001 Junior Olympics in Marquette, Michigan**.

He went on to attend **Middlebury College in Vermont**, graduating with the Class of 2005. At Middlebury, Bryce competed in both **varsity Nordic skiing and varsity cross-country running**, continuing the endurance-sports career he had begun in Montana.

Today, Bryce lives in **Manhattan Beach, California**, with his wife, **Jin Kang**, and their three children. The landscape has changed from snow-covered mountains to the Pacific Ocean, but endurance sport remains an important part of his life.

Bryce is an avid **open-water swimmer** and a regular competitor in Southern California ocean races. He has become a consistent strong finisher in the **Dwight Crum Pier-to-Pier Swim**, the roughly two-mile open-water race from the Hermosa Beach Pier to the Manhattan Beach Pier. In 2026, he finished **second overall in the wetsuit division**, adding another chapter to an athletic career that now spans more than 25 years and three endurance disciplines.

That same appetite for long, difficult problems eventually found another outlet in mathematics, software, and artificial intelligence.

Bryce's current research centers on **Mycelium**, an experimental neural-symbolic architecture for machine reasoning, built and trained entirely on a single consumer GPU. The project asks a deceptively simple question: instead of asking a language model to imitate reasoning by producing more language, can we separate **understanding a problem from solving it** — and give each job to the machinery naturally suited to it?

In Mycelium, a frozen language-model trunk is read **once** per problem, and a small trained head — well under one percent of the system's parameters — compresses natural language into a compact, **typed factor graph**. That graph crosses a well-defined interface into a deterministic constraint solver, which either solves it exactly or refuses. One way Bryce describes the goal is as an **information bottleneck**: destroy as much of the incidental surface variation of language as possible while preserving the underlying mathematical structure. Neural components propose; symbolic machinery disposes. Abstraction is allowed in recognition, never in verification.

Rather than a single pass through a transformer, the head deliberates in repeated internal **"breath" cycles** — a compact latent state that reads, refines, reconsiders, and progressively commits, with a narrow latent **waist** forcing information through a constrained internal interface. The architecture draws on fields that have already lived under analogous constraints: **factor graphs and message passing, compiler intermediate representations, information theory, constraint solving, and dynamical systems**. Bryce's recurring design question is: *when something feels missing, which mature field has already solved this?*

A defining commitment of the project is that the system's output is not an answer but a **decision**: answer, or abstain. A chain of independent gates — an out-of-distribution "mouth," a multi-view parsing vote, a cross-model panel, and finally an exact answer key — stands between the neural parse and any claim the system makes. On the **MATH-500** benchmark, which the project measures but never trains on, the standard is not how often the system answers; it is that a certified answer is **never wrong**. The result is **interpretability by construction**: predicates, roles, bindings, and graph structure that can be read directly, rather than reconstructed after the fact.

The project is experimental and under active development — currently mid-campaign teaching the parser to read *wild* mathematical prose, harvested from real textbook-style corpora rather than synthetic templates. Every experiment runs under pre-registered pass/fail bars, and the project's public **ledger** records the failures alongside the wins: killed mechanisms, refuted hypotheses, and honest negatives, all preserved. The objective is not to make the machinery more elaborate; it is to discover which pieces are actually load-bearing.

In that respect, Bryce sees a surprising continuity between his athletic and research lives.

Nordic skiing, distance running, open-water swimming, and experimental research all reward much the same temperament: comfort with discomfort, patience with incremental progress, and a willingness to keep working when the destination is not yet visible. A two-mile ocean swim is not won by one spectacular stroke, and a difficult research problem is rarely solved by one spectacular idea. Both are accumulations of small corrections sustained over time.

From the snow trails of Bozeman to Middlebury's Nordic courses, from swimming between the Hermosa and Manhattan Beach piers to building experimental reasoning systems, the setting has changed considerably.

The underlying fascination has not: **take something difficult, understand its structure, and keep moving until it yields.**
