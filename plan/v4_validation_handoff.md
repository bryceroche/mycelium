# v4 Breathing Transformer — Validation Handoff

**Date:** May 6, 2026
**Status:** Day 1 script written, AWS blocked by network timeout, machine restart needed

---

## What Happened This Session

1. Reviewed `~/Downloads/mycelium_v4_final_architecture.md` — the v4 conceptual design (Pythia-160M, 4 looping layers, π cycling, spectral factorization, tinygrad on Shadow Glass)

2. Agreed on a 2-day validation plan on **PyTorch + AWS** before Shadow Glass arrives (~May 8):
   - **Day 1:** Validate looping (do Pythia layers 0-3 improve with reuse?)
   - **Day 2:** Validate π cycling (does per-loop RoPE phase shift improve diversity?)

3. Wrote `scripts/validate_looping.py` — the Day 1 experiment. Self-contained, no dependencies beyond transformers + the existing data generators.

4. AWS CLI timed out reaching `ec2.us-east-1.amazonaws.com` and `sts.us-east-1.amazonaws.com`. Likely local network/DNS issue. Needs machine restart.

---

## What's Ready

### `scripts/validate_looping.py`
- Loads Pythia-160M from HuggingFace
- Takes layers 0-3, loops them 1/2/4/8 times
- Adds small learnable loop embedding between iterations
- Generates 200 L3 problems inline (no data files needed)
- Greedy generation → `####` regex extraction
- Compares against full 12-layer baseline
- Prints VERDICT: helps / hurts / neutral

### To run on AWS:
```bash
# 1. Fix network, then start instance
aws ec2 start-instances --instance-ids i-08c1c295a4113a908

# 2. Get IP
aws ec2 describe-instances --instance-ids i-08c1c295a4113a908 \
  --query 'Reservations[0].Instances[0].PublicIpAddress' --output text

# 3. SSH in
ssh -i ~/.ssh/mycelium-key.pem ubuntu@<IP>

# 4. Pull and run
cd ~/mycelium && git pull
pip install transformers  # if needed
python scripts/validate_looping.py --num_problems 200 --verbose
```

---

## What to Watch For

| Result | Meaning | Next Step |
|--------|---------|-----------|
| Accuracy ↑ with more loops | Looping works! | Write Day 2: π cycling |
| Accuracy flat | Layers tolerate reuse | Weak positive — π cycling may differentiate |
| Accuracy ↓ with more loops | Representation degrades | Revise: more layers per breath, different norm, or abandon looping |
| 4-loop ≈ 12-layer baseline | Very promising | Looping recovers depth from reuse |

---

## Day 2 Script (Not Yet Written)

**π-cycling validation:** If Day 1 passes, add per-loop phase shift to RoPE computation. Same Pythia layers, same L3 problems. Measure:
- Does attention diversity (per-head entropy) increase with phase offset?
- Does accuracy improve over plain looping (no phase shift)?

---

## Key Decisions from Review

- **Start with simple lookup table** — just pattern vectors + cosine matching. Add resonant angles, subtraction masks, coupling matrices only as needed.
- **Keep AWS v3 path warm** — baked Llama at 14% cycle 1 is the baseline to beat.
- **Validate architecture on PyTorch first**, port to tinygrad after mechanism is proven.
- **Shadow Glass arrives ~May 8** — tinygrad port happens after PyTorch validation.

---

## Files That Matter

- `~/Downloads/mycelium_v4_final_architecture.md` — full v4 design doc
- `scripts/validate_looping.py` — Day 1 experiment (NEW)
- `scripts/generate_per_cycle_data.py` — L3 problem generators (used by validation)
- `scripts/controller.py` — v2 controller (reference, not used in v4)
- `plan/mycelium_v2_master_rebuild_handoff.md` — v2 design (superseded by v4)
