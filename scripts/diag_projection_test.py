#!/usr/bin/env python3
"""Test which LoRA projections help vs hurt arithmetic.
Q,K modify attention routing. V,O modify value/output computation.
If Q,K-only > Q,K,V,O: V,O corrupt arithmetic."""
import sys, json, re, os, torch
sys.path.insert(0, '.')
from scripts.atom_lora import AtomLoRAModel, AtomAdditiveLoRAManager

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
model.eval()


def extract_answer(text):
    m = re.search(r'####\s*([-]?\d+)', text)
    if m:
        return int(m.group(1))
    nums = re.findall(r'[-]?\d+', text)
    return int(nums[-1]) if nums else None


def eval_with_proj_mask(samples, proj_mask, clamp_val=0.4):
    correct = 0
    for s in samples:
        inputs = model.tokenizer([s['problem']], return_tensors='pt',
                                  truncation=True, max_length=192)
        ids = inputs['input_ids'].to(device)
        attn = inputs['attention_mask'].to(device)
        with torch.no_grad():
            (p0, h0, p1, h1, init_s, ns, f1, out) = model.two_pass_cycle1(ids, attn)
            gs = init_s.clamp(-clamp_val, clamp_val)

            originals = {}
            num_layers = len(model.transformer.model.layers)
            for li in range(num_layers):
                originals[li] = {}
                for pn in ['q_proj', 'k_proj', 'v_proj', 'o_proj']:
                    proj = getattr(model.transformer.model.layers[li].self_attn, pn)
                    originals[li][pn] = proj.forward
                    if pn in proj_mask:
                        def make_fwd(of, atoms, layer, proj_name, scales):
                            def fwd(x):
                                base = of(x)
                                lora = atoms.apply(
                                    x.to(dtype=atoms.A[proj_name].dtype),
                                    layer, proj_name, scales)
                                return base + lora.to(dtype=base.dtype)
                            return fwd
                        proj.forward = make_fwd(proj.forward, model.atoms, li, pn, gs)
            try:
                gen = model.transformer.generate(
                    ids, attention_mask=attn, max_new_tokens=60, do_sample=False)
            finally:
                for li, lf in originals.items():
                    for pn, of in lf.items():
                        getattr(model.transformer.model.layers[li].self_attn, pn).forward = of

            text = model.tokenizer.decode(gen[0][ids.size(1):], skip_special_tokens=True)
            if extract_answer(text) == s['final_answer']:
                correct += 1
    return correct, len(samples)


# Load single-step eval data
samples = []
with open('data/per_cycle/gsm8k_single_step_eval.jsonl') as f:
    for line in f:
        if line.strip():
            samples.append(json.loads(line.strip()))
print(f'Loaded {len(samples)} eval problems', flush=True)

print('\n' + '=' * 60, flush=True)
print('TEST 1: Which LoRA projections help vs hurt?', flush=True)
print('=' * 60, flush=True)

configs = [
    ('Q,K,V,O (all)',    {'q_proj', 'k_proj', 'v_proj', 'o_proj'}),
    ('Q,K only',         {'q_proj', 'k_proj'}),
    ('Q only',           {'q_proj'}),
    ('K only',           {'k_proj'}),
    ('V,O only',         {'v_proj', 'o_proj'}),
    ('None (vanilla)',   set()),
]

for name, mask in configs:
    c, t = eval_with_proj_mask(samples, mask, 0.4)
    pct = 100 * c / t
    bar = '#' * int(pct / 2)
    print(f'  {name:20s}  {c:3d}/{t} = {pct:5.1f}%  {bar}', flush=True)

# Test 2: L4.5 data
print('\n' + '=' * 60, flush=True)
print('TEST 2: Architecture on L4/L4.5 eval data', flush=True)
print('=' * 60, flush=True)

for level in ['L4.5', 'L4', 'L3']:
    path = f'data/per_cycle/{level}_eval.jsonl'
    if os.path.exists(path):
        level_samples = []
        with open(path) as f:
            for line in f:
                if line.strip():
                    level_samples.append(json.loads(line.strip()))
        # Convert to single-step
        single_steps = []
        for s in level_samples:
            for i in range(len(s.get('cycle_targets', []))):
                single_steps.append({
                    'problem': s['problem'],
                    'final_answer': s['cycle_targets'][i],
                })
        n = min(100, len(single_steps))
        c, t = eval_with_proj_mask(single_steps[:n],
                                    {'q_proj', 'k_proj', 'v_proj', 'o_proj'}, 0.4)
        print(f'  {level} at 0.4: {c}/{t} = {100*c/t:.1f}%', flush=True)
        c2, t2 = eval_with_proj_mask(single_steps[:n],
                                      {'q_proj', 'k_proj', 'v_proj', 'o_proj'}, 3.0)
        print(f'  {level} at 3.0: {c2}/{t2} = {100*c2/t2:.1f}%', flush=True)
        break
else:
    print('  No L4.5/L4/L3 eval data found', flush=True)
    for f in os.listdir('data/per_cycle/'):
        if 'eval' in f:
            print(f'    {f}', flush=True)

print('=' * 60, flush=True)
