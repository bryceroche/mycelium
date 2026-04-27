#!/usr/bin/env python3
"""Inspect what each atom DOES to Llama's attention.

For each atom: activate it alone, measure attention changes.
Categorize atoms as: dead, number-focused, operation-focused, structural.
Test generalization across different sentence structures.
"""
import sys, json, re, torch
sys.path.insert(0, '.')
from scripts.atom_lora import AtomLoRAModel, AtomAdditiveLoRAManager
import torch.nn.functional as F

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = AtomLoRAModel()
model.atoms = model.atoms.to(device=device, dtype=torch.bfloat16)
model.controller = model.controller.to(device=device, dtype=torch.bfloat16)
model.confidence_head = model.confidence_head.to(device)
model.mobius = model.mobius.to(device)

ckpt = torch.load('checkpoints/per_cycle_gsm8k_best.pt', map_location='cpu')
for name in ['atoms', 'controller', 'confidence_head']:
    if name in ckpt:
        obj = getattr(model, name)
        own = obj.state_dict()
        for k, v in ckpt[name].items():
            if k in own and own[k].shape == v.shape:
                own[k] = v
        obj.load_state_dict(own, strict=False)
print(f"Loaded checkpoint", flush=True)
model.eval()

# Test sentences covering different GSM8K patterns
test_sentences = [
    "Natalia sold 48 clips in April and half as many in May",
    "Mark has 20 toys and buys twice as many more",
    "A farmer has 3 times as many chickens as cows with 15 cows",
    "She split 120 cookies equally among 4 friends",
    "The price dropped by 15 percent from 200 dollars",
]


def get_hidden_diff(sentence, atom_idx, scale=0.4):
    """Run sentence with single atom active vs no atoms. Return hidden state diff."""
    inputs = model.tokenizer([sentence], return_tensors='pt', truncation=True, max_length=64)
    ids = inputs['input_ids'].to(device)
    attn = inputs['attention_mask'].to(device)

    with torch.no_grad():
        # Baseline: no atoms
        out_base = model.transformer(ids, attention_mask=attn, output_hidden_states=True)
        h_base = out_base.hidden_states[-1]  # (1, seq, 2048)

        # With single atom
        scales = torch.zeros(1, 64, device=device, dtype=torch.bfloat16)
        scales[0, atom_idx] = scale
        mgr = AtomAdditiveLoRAManager(model.transformer)
        mgr.apply(model.atoms, scales)
        try:
            out_atom = model.transformer(ids, attention_mask=attn, output_hidden_states=True)
            h_atom = out_atom.hidden_states[-1]
        finally:
            mgr.remove()

    diff = (h_atom - h_base).float()  # (1, seq, 2048)
    tokens = [model.tokenizer.decode(t) for t in ids[0]]
    return diff[0], tokens  # (seq, 2048), list of strings


def analyze_atom(atom_idx):
    """Analyze one atom across all test sentences."""
    results = []
    for sent in test_sentences:
        diff, tokens = get_hidden_diff(sent, atom_idx, scale=0.4)
        # Per-token impact: L2 norm of hidden state change
        per_token_impact = diff.norm(dim=-1)  # (seq,)
        total_impact = per_token_impact.mean().item()

        # Which tokens are most affected?
        top_k = min(5, len(tokens))
        top_vals, top_idx = per_token_impact.topk(top_k)

        top_tokens = [(tokens[i], top_vals[j].item()) for j, i in enumerate(top_idx)]
        results.append({
            'sentence': sent[:60],
            'total_impact': total_impact,
            'top_tokens': top_tokens,
        })
    return results


print("\n" + "=" * 70, flush=True)
print("ATOM INSPECTION: what does each atom do to Llama's hidden states?", flush=True)
print("=" * 70, flush=True)

# Phase 1: measure total impact of each atom
print("\nPhase 1: Atom strength (total hidden state change)", flush=True)
print("-" * 50, flush=True)

atom_impacts = []
for a in range(64):
    diff, tokens = get_hidden_diff(test_sentences[0], a, scale=0.4)
    impact = diff.norm(dim=-1).mean().item()
    atom_impacts.append((a, impact))

atom_impacts.sort(key=lambda x: -x[1])

# Categorize by impact
dead = [(a, i) for a, i in atom_impacts if i < 0.1]
weak = [(a, i) for a, i in atom_impacts if 0.1 <= i < 1.0]
moderate = [(a, i) for a, i in atom_impacts if 1.0 <= i < 5.0]
strong = [(a, i) for a, i in atom_impacts if i >= 5.0]

print(f"  Dead (impact < 0.1):     {len(dead)} atoms", flush=True)
print(f"  Weak (0.1 - 1.0):       {len(weak)} atoms", flush=True)
print(f"  Moderate (1.0 - 5.0):   {len(moderate)} atoms", flush=True)
print(f"  Strong (> 5.0):         {len(strong)} atoms", flush=True)
print(f"\n  Top 10 strongest:", flush=True)
for a, imp in atom_impacts[:10]:
    print(f"    Atom {a:2d}: impact = {imp:.2f}", flush=True)
print(f"  Bottom 10 weakest:", flush=True)
for a, imp in atom_impacts[-10:]:
    print(f"    Atom {a:2d}: impact = {imp:.4f}", flush=True)

# Phase 2: What tokens do the top atoms focus on?
print(f"\nPhase 2: Token focus of top 10 atoms across sentences", flush=True)
print("-" * 50, flush=True)

for a, _ in atom_impacts[:10]:
    results = analyze_atom(a)
    print(f"\n  Atom {a}:", flush=True)
    for r in results:
        top_toks = [(t, f"{v:.1f}") for t, v in r['top_tokens'][:3]]
        print(f"    [{r['total_impact']:.2f}] {r['sentence'][:50]}...  top: {top_toks}", flush=True)

# Phase 3: Generalization test — do atoms respond to numbers consistently?
print(f"\nPhase 3: Do atoms generalize across sentence structures?", flush=True)
print("-" * 50, flush=True)

number_sentences = [
    "She has 48 cookies",
    "He bought 48 apples",
    "There are 48 students",
    "The price is 48 dollars",
    "It weighs 48 pounds",
]

for a, _ in atom_impacts[:5]:
    impacts_on_48 = []
    for sent in number_sentences:
        diff, tokens = get_hidden_diff(sent, a, scale=0.4)
        # Find the "48" token
        for i, t in enumerate(tokens):
            if '48' in t:
                impacts_on_48.append(diff[i].norm().item())
                break
    if impacts_on_48:
        mean_impact = sum(impacts_on_48) / len(impacts_on_48)
        std_impact = (sum((x - mean_impact) ** 2 for x in impacts_on_48) / len(impacts_on_48)) ** 0.5
        cv = std_impact / (mean_impact + 1e-8)
        generalize = "GENERALIZES" if cv < 0.3 else "TEMPLATE-SPECIFIC"
        print(f"  Atom {a:2d}: impact on '48' across contexts: mean={mean_impact:.2f} std={std_impact:.2f} CV={cv:.2f} → {generalize}", flush=True)

# Phase 4: Redundancy check — cosine similarity between atom effects
print(f"\nPhase 4: Atom redundancy (cosine similarity of effects)", flush=True)
print("-" * 50, flush=True)

# Get effect vectors for top 20 atoms on first sentence
effect_vecs = []
top20_atoms = [a for a, _ in atom_impacts[:20]]
for a in top20_atoms:
    diff, _ = get_hidden_diff(test_sentences[0], a, scale=0.4)
    effect_vecs.append(diff.reshape(-1))  # flatten to single vector

effect_mat = torch.stack(effect_vecs)  # (20, seq*2048)
effect_norm = F.normalize(effect_mat, dim=-1)
cos_sim = effect_norm @ effect_norm.T  # (20, 20)

# Find highly redundant pairs
redundant_pairs = []
for i in range(20):
    for j in range(i + 1, 20):
        sim = cos_sim[i, j].item()
        if abs(sim) > 0.7:
            redundant_pairs.append((top20_atoms[i], top20_atoms[j], sim))

if redundant_pairs:
    print(f"  Highly similar atom pairs (|cos| > 0.7):", flush=True)
    for a1, a2, sim in sorted(redundant_pairs, key=lambda x: -abs(x[2])):
        print(f"    Atoms {a1:2d} & {a2:2d}: cos = {sim:.3f}", flush=True)
else:
    print(f"  No highly redundant pairs found among top 20", flush=True)

print("\n" + "=" * 70, flush=True)
