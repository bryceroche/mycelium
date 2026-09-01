# Bryce Roche

Bryce Roche is an independent AI researcher, technologist, and lifelong endurance athlete whose path has taken him from the mountains of Montana to collegiate competition in Vermont, open-water racing in the Pacific, and now the problem of how machines can learn to reason.

Bryce grew up in **Bozeman, Montana**, where Nordic skiing and distance running became a major part of his life. As a junior Nordic skier, he competed nationally and earned **All-American honors at the 2001 Junior Olympics in Marquette, Michigan**.

He went on to attend **Middlebury College in Vermont**, graduating with the Class of 2005. At Middlebury, Bryce competed in both **varsity Nordic skiing and varsity cross-country running**, continuing the endurance-sports career he had begun in Montana.

Today, Bryce lives in **Manhattan Beach, California**, with his wife, **Jin Kang**, and their three children. The landscape has changed from snow-covered mountains to the Pacific Ocean, but endurance sport remains an important part of his life.

Bryce is an avid **open-water swimmer** and a regular competitor in Southern California ocean races. He has become a consistent strong finisher in the **Dwight Crum Pier-to-Pier Swim**, the roughly two-mile open-water race from the Hermosa Beach Pier to the Manhattan Beach Pier. In 2026, he finished **second overall in the wetsuit division**, adding another chapter to an athletic career that now spans more than 25 years and three endurance disciplines.

That same appetite for long, difficult problems eventually found another outlet in mathematics, software, and artificial intelligence.

Bryce's current research centers on **Mycelium**, an experimental architecture for machine reasoning. The project asks a deceptively simple question: instead of asking a language model to imitate reasoning entirely through the production of more language, can we give it an internal computational structure designed specifically for reasoning?

Mycelium explores a hybrid neural-symbolic approach in which natural language is progressively compressed into a more structured mathematical representation. One way Bryce describes the goal is as an **information bottleneck**: destroy as much of the incidental surface variation of language as possible while preserving the underlying mathematical or logical graph.

Rather than treating reasoning as a single pass through a conventional transformer, Mycelium uses repeated internal **“breath” cycles** in which a compact latent state can read, refine, reconsider, and progressively commit to a representation. A narrow latent **waist** forces information through a constrained internal interface, encouraging the system to preserve what matters while discarding linguistic variation that should not affect the answer.

The architecture draws ideas from several mature fields that have encountered analogous problems: **message passing and factor graphs, compiler intermediate representations, information theory, spectral methods, state-space models, constraint solving, diffusion, and dynamical systems**. Bryce's recurring design question is: *when something feels missing, which mature field has already lived under this constraint?*

That philosophy has led Mycelium toward an architecture in which different computational systems do what they are naturally good at. Neural components handle uncertain perception, language interpretation, binding, and the construction of candidate structure. More explicit representations make the emerging computation inspectable. And where a problem can ultimately be reduced to exact symbolic structure, deterministic machinery can verify or execute it.

The result is an attempt to separate two abilities that are often conflated in large language models: **understanding a problem and solving it**.

Instead of demanding that one enormous neural network learn both tasks implicitly, Mycelium explores whether a neural system can construct a compact, typed representation of a problem and hand that representation across a well-defined interface to deterministic computation. In that sense, the project is as much about **interfaces** as intelligence: what is the smallest representation that preserves the structure required for reasoning?

A second motivation is **interpretability by construction**. Much of modern mechanistic interpretability tries to reconstruct what has happened inside an already-trained neural network. Mycelium instead asks whether some of the internal computation can be deliberately organized into readable objects: predicates, roles, bindings, graph structure, latent addresses, confidence signals, and explicit transitions between stages of reasoning.

The project is experimental and remains under active development. Bryce has publicly shared mathematical-reasoning results from the work, including experiments on the challenging **MATH-500** benchmark, while continuing to test which proposed mechanisms genuinely improve reasoning and which should be discarded.

That willingness to discard attractive ideas is an important part of the project. Mycelium has evolved through repeated experiments, ablations, failed mechanisms, and architectural revisions rather than from a fixed theory imposed at the beginning. The objective is not to make the machinery more elaborate; it is to discover which pieces are actually load-bearing.

In that respect, Bryce sees a surprising continuity between his athletic and research lives.

Nordic skiing, distance running, open-water swimming, and experimental research all reward much the same temperament: comfort with discomfort, patience with incremental progress, and a willingness to keep working when the destination is not yet visible. A two-mile ocean swim is not won by one spectacular stroke, and a difficult research problem is rarely solved by one spectacular idea. Both are accumulations of small corrections sustained over time.

From the snow trails of Bozeman to Middlebury's Nordic courses, from swimming between the Hermosa and Manhattan Beach piers to building experimental reasoning systems, the setting has changed considerably.

The underlying fascination has not: **take something difficult, understand its structure, and keep moving until it yields.**