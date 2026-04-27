# Mycelium Project Update — April 2026

## Quick Reminder

Mycelium teaches a small language model (Llama 1B) to solve math problems it can't solve alone. The model knows individual math facts ("12 × 20 = 240") but can't chain them into multi-step solutions. We give it a "breathing loop" — think, compress, plan, think again — where each breath tackles one piece of the problem.

## Where We Are

The model solves 94.5% of our structured practice problems (3-step word problems with consistent templates). When we moved to real-world math problems (GSM8K — 7,110 grade school math problems written by humans), accuracy dropped to about 20%. We've been diagnosing why.

## What We Just Discovered

The breakthrough came from carefully dissecting HOW our system modifies the base language model. The modification happens through something called LoRA (Low-Rank Adaptation), which adjusts four internal projections in the model's attention mechanism: Q (query), K (key), V (value), and O (output).

We assumed Q and K were doing the important work — steering the model's attention to the right parts of the problem ("look at the number 48" and "focus on 'half as many'"). We assumed V and O might be causing problems by distorting the model's arithmetic.

We were completely wrong.

When we tested each component in isolation:

```
All four (Q,K,V,O):   20.3%  — our best result
V and O only:         13.8%  — V,O does the heavy lifting!
Q and K only:          0.4%  — Q,K alone actually HURTS performance
None (vanilla model):  1.9%  — baseline, generates multiple-choice nonsense
```

The atoms aren't steering attention at all. They're REPROGRAMMING the model. The V,O modification transforms Llama from a "text completion" machine (which defaults to generating multiple-choice format) into a "math computation" machine (which generates equations and solves them). It's not about WHERE the model looks — it's about HOW the model processes what it sees.

## What This Means

This overturns several of our previous approaches:

1. **The "parse loudly, compute quietly" theory was wrong.** We thought the atoms should be strong in early layers (for parsing) and quiet in late layers (to not interfere with arithmetic). Testing proved uniform gentle application works best.

2. **The "split generation" approach was wrong.** We built a system to turn atoms ON for equation setup and OFF for arithmetic. But removing the atoms during arithmetic removes the V,O reprogramming that puts the model in math mode in the first place.

3. **The optimal "volume" for atoms matters enormously.** We discovered that the atoms were set too loud (scale ±3.0). A sweep found the sweet spot at ±0.4 — a gentle nudge that guides without overwhelming:

```
Scale ±0.0:    1.9%   (no help)
Scale ±0.1:    2.7%   (too quiet)
Scale ±0.3:   18.0%   (starting to help)
Scale ±0.4:   20.3%   (sweet spot!)
Scale ±0.7:   16.5%   (too loud)
Scale ±3.0:   14.6%   (way too loud — our old setting)
```

## The Architecture Today

The system has been significantly simplified from where we started:

```
Component                  What It Does
─────────────────────────────────────────────────────────────
Llama 1B (frozen)          The "brain" — reads and computes (1,230M params)
Breathing Controller       Observes what Llama computed, plans next step (190M)
64 LoRA Atoms              Pattern library — reprograms Llama's processing (82M)
Confidence Head            Decides when to stop thinking (2.5M)
─────────────────────────────────────────────────────────────
Total: 1.5B (only 275M trainable, rest frozen)
```

We merged several previously separate components (the "perceiver" compressor and the "hypernetwork" planner) into a single unified controller. This was driven by a diagnostic discovery: the separate planner couldn't see what Llama was actually computing, so it made the same decision regardless of the problem. The unified controller reads Llama's internal states directly and makes informed, problem-specific decisions.

## Key Diagnostic Tools

Two diagnostic tools have been game-changers:

**The Laplace Diagnostic** measures whether the breathing loop is actually "breathing" — producing genuinely different computations at each cycle. We discovered the old architecture collapsed to a fixed point after cycle 1 (DC ratio 0.93 — meaning 93% of the signal was constant). The new controller brought this to 0.54 — the loop genuinely breathes.

**The Atom Inspection** tested each of the 64 atoms individually to see what they do to the model's attention patterns. All 64 are active, they attend to the right things (numbers, operations), and they generalize across different sentence structures. The atoms are healthy.

## Current Challenge

The model gets the right OPERATION (picks multiplication for "times as many") but computes the wrong ANSWER ("12 × 20 = 195" instead of 240). This is because even gentle atom modification (±0.4) slightly disrupts Llama's arithmetic circuits. Vanilla Llama computes "12 × 20 = 240" perfectly. With atoms: "12 × 20 = 195."

We just discovered that V,O projections (not Q,K) are the primary mechanism. Now we're running a 2D sweep to find the INDEPENDENT optimal strength for V,O versus Q,K — they likely want different settings since they do fundamentally different jobs:

```
V,O: reprogram Llama into math mode (needs to be strong enough)
Q,K: fine-tune routing within math mode (should be gentle)

The optimal might be V,O=0.6, Q,K=0.2 rather than both at 0.4.
```

## The Scoreboard

```
Task                    Without Breathing    With Breathing
─────────────────────────────────────────────────────────────
1-step (structured)     18.8%               96.0%
2-step (structured)      6.0%               91.0%
3-step (structured)      0.0%               94.5%
GSM8K (real problems)    1.9%               20.3%
─────────────────────────────────────────────────────────────
Target: MATH-500 benchmark by July 1, 2026
```

## What's Next

1. **Find the optimal V,O vs Q,K balance** — the 2D sweep running now
2. **Train with the optimal atom settings** — the atoms were trained at ±3.0 but work best at ±0.4, so retraining at the right strength should improve further
3. **Graduate from single-step to multi-step GSM8K** — master individual steps before chaining
4. **Hardware: PRC ShadowGlass arriving** — a local workstation with AMD 7900 XTX GPU for 2x faster training at zero hourly cost

## The Big Picture

The core thesis holds: a small model that can't chain reasoning internally learns to chain through external compression loops. The breathing architecture works — 94.5% on structured problems, with the loop genuinely differentiating each cycle.

The gap to real-world math (20% on GSM8K) is about the subtlety of how the atoms modify the base model. We're discovering that the modification is more like "reprogramming" than "steering" — the atoms transform WHAT the model computes, not WHERE it looks. Finding the right balance of this reprogramming is the current frontier.

The model breathes. Each breath focuses on something different. The pattern library guides the decomposition. We're tuning the volume.
