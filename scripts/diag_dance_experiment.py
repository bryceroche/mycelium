#!/usr/bin/env python3
"""Per-layer atom strength experiment: the dance hypothesis.

Tests whether atoms should be strong in early layers (parsing)
and quiet in late layers (computation).
"""
import sys, json, re, torch
import torch.nn as nn
sys.path.insert(0, '.')
from scripts.atom_lora import AtomLoRAModel, LoRAAtoms

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

samples = []
with open('data/per_cycle/gsm8k_single_step_eval.jsonl') as f:
    for line in f:
        if line.strip():
            samples.append(json.loads(line.strip()))
print(f'Loaded {len(samples)} eval problems', flush=True)
model.eval()


def extract_answer(text):
    m = re.search(r'####\s*([-]?\d+)', text)
    if m:
        return int(m.group(1))
    nums = re.findall(r'[-]?\d+', text)
    return int(nums[-1]) if nums else None


PROJ_NAMES = ['q_proj', 'k_proj', 'v_proj', 'o_proj']


def apply_per_layer(transformer, atoms, atom_scales, layer_clamps):
    """Apply atom LoRA with per-layer scale clamping. Returns list of originals."""
    originals = {}
    num_layers = len(transformer.model.layers)
    for li in range(num_layers):
        clamp_val = layer_clamps.get(li, 0.0)
        clamped = atom_scales.clamp(-clamp_val, clamp_val)
        originals[li] = {}
        for pn in PROJ_NAMES:
            proj = getattr(transformer.model.layers[li].self_attn, pn)
            originals[li][pn] = proj.forward
            # Must capture variables properly
            _orig = proj.forward
            _atoms = atoms
            _li = li
            _pn = pn
            _cs = clamped

            def make_fwd(orig_f, at, layer, proj_n, scales):
                def fwd(x):
                    base = orig_f(x)
                    lora = at.apply(x.to(dtype=at.A[proj_n].dtype), layer, proj_n, scales)
                    return base + lora.to(dtype=base.dtype)
                return fwd

            proj.forward = make_fwd(_orig, _atoms, _li, _pn, _cs)
    return originals


def remove_lora(transformer, originals):
    """Restore original forwards."""
    for li, lf in originals.items():
        for pn, of in lf.items():
            getattr(transformer.model.layers[li].self_attn, pn).forward = of


def eval_experiment(name, layer_clamps):
    correct = 0
    total = len(samples)
    for s in samples:
        inputs = model.tokenizer([s['problem']], return_tensors='pt',
                                  truncation=True, max_length=192)
        ids = inputs['input_ids'].to(device)
        attn = inputs['attention_mask'].to(device)
        with torch.no_grad():
            (p0, h0, p1, h1, init_s, ns, f1, out) = model.two_pass_cycle1(ids, attn)
            originals = apply_per_layer(model.transformer, model.atoms, init_s, layer_clamps)
            try:
                gen = model.transformer.generate(
                    ids, attention_mask=attn, max_new_tokens=60, do_sample=False)
            finally:
                remove_lora(model.transformer, originals)
            text = model.tokenizer.decode(gen[0][ids.size(1):], skip_special_tokens=True)
            if extract_answer(text) == s['final_answer']:
                correct += 1
    pct = 100 * correct / total
    print(f'  {name:45s}  {correct:3d}/{total} = {pct:5.1f}%', flush=True)
    return pct


print('\nPER-LAYER ATOM STRENGTH EXPERIMENTS', flush=True)
print('=' * 65, flush=True)

# A: Dance (strong early, silent late)
clamps_A = {i: 0.8 for i in range(8)}
clamps_A.update({i: 0.0 for i in range(8, 16)})
eval_experiment('A: Dance (0.8 early, 0.0 late)', clamps_A)

# B: Dance with whisper
clamps_B = {i: 0.8 for i in range(8)}
clamps_B.update({i: 0.1 for i in range(8, 16)})
eval_experiment('B: Dance+whisper (0.8 early, 0.1 late)', clamps_B)

# C: Uniform optimal (baseline)
clamps_C = {i: 0.4 for i in range(16)}
eval_experiment('C: Uniform optimal (0.4 all)', clamps_C)

# D: Inverted dance (control)
clamps_D = {i: 0.1 for i in range(8)}
clamps_D.update({i: 0.8 for i in range(8, 16)})
eval_experiment('D: Inverted (0.1 early, 0.8 late)', clamps_D)

# E: Gradual fade
clamps_E = {i: 0.8 * (1.0 - i / 15.0) for i in range(16)}
eval_experiment('E: Gradual fade (0.8 -> 0.05)', clamps_E)

print('=' * 65, flush=True)
