#!/usr/bin/env python3
"""2D sweep: independent V,O and Q,K scale clamps."""
import sys, json, re, torch
sys.path.insert(0, '.')
from scripts.atom_lora import AtomLoRAModel

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
            if k in own and own[k].shape == v.shape: own[k] = v
        obj.load_state_dict(own, strict=False)
model.eval()

samples = []
with open('data/per_cycle/gsm8k_single_step_eval.jsonl') as f:
    for line in f:
        if line.strip(): samples.append(json.loads(line.strip()))
print(f'Loaded {len(samples)} eval problems', flush=True)


def extract_answer(text):
    m = re.search(r'####\s*([-]?\d+)', text)
    if m: return int(m.group(1))
    nums = re.findall(r'[-]?\d+', text)
    return int(nums[-1]) if nums else None


def eval_2d(vo_clamp, qk_clamp):
    correct = 0
    for s in samples:
        inputs = model.tokenizer([s['problem']], return_tensors='pt',
                                  truncation=True, max_length=192)
        ids = inputs['input_ids'].to(device)
        attn = inputs['attention_mask'].to(device)
        with torch.no_grad():
            (p0, h0, p1, h1, init_s, ns, f1, out) = model.two_pass_cycle1(ids, attn)

            # Build per-projection clamped scales
            vo_scales = init_s.clamp(-vo_clamp, vo_clamp) if vo_clamp > 0 else torch.zeros_like(init_s)
            qk_scales = init_s.clamp(-qk_clamp, qk_clamp) if qk_clamp > 0 else torch.zeros_like(init_s)

            originals = {}
            num_layers = len(model.transformer.model.layers)
            for li in range(num_layers):
                originals[li] = {}
                for pn in ['q_proj', 'k_proj', 'v_proj', 'o_proj']:
                    proj = getattr(model.transformer.model.layers[li].self_attn, pn)
                    originals[li][pn] = proj.forward
                    sc = qk_scales if pn in ('q_proj', 'k_proj') else vo_scales
                    if sc.abs().max() > 0:
                        def make_fwd(of, atoms, layer, proj_name, scales):
                            def fwd(x):
                                base = of(x)
                                lora = atoms.apply(x.to(dtype=atoms.A[proj_name].dtype),
                                                   layer, proj_name, scales)
                                return base + lora.to(dtype=base.dtype)
                            return fwd
                        proj.forward = make_fwd(proj.forward, model.atoms, li, pn, sc)
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
    return correct


qk_vals = [0.0, 0.1, 0.2, 0.3, 0.4]
vo_vals = [0.2, 0.4, 0.6, 0.8, 1.0]

print('\n2D SWEEP: V,O clamp (rows) × Q,K clamp (columns)', flush=True)
print('=' * 60, flush=True)

# Header
header = '       ' + ''.join(f'QK={q:3.1f}  ' for q in qk_vals)
print(header, flush=True)
print('-' * 60, flush=True)

for vo in vo_vals:
    row = f'VO={vo:.1f} '
    for qk in qk_vals:
        c = eval_2d(vo, qk)
        pct = 100 * c / len(samples)
        row += f' {pct:5.1f}% '
        sys.stdout.flush()
    print(row, flush=True)

print('=' * 60, flush=True)
