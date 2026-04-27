#!/usr/bin/env python3
"""Deep single-step diagnostic: categorize WHERE failures happen.

For each problem: what did the model generate, what went wrong, which category?
Runs on the current checkpoint with configurable scale clamp.

Usage:
  python scripts/diag_deep_single_step.py [--clamp 0.4] [--num 50]
"""
import sys, json, re, torch, argparse
sys.path.insert(0, '.')
from scripts.atom_lora import AtomLoRAModel, AtomAdditiveLoRAManager

def extract_answer(text):
    m = re.search(r'####\s*([-]?\d+)', text)
    if m: return int(m.group(1))
    nums = re.findall(r'[-]?\d+', text)
    return int(nums[-1]) if nums else None

def extract_equation(text):
    """Find 'A op B = C' pattern."""
    m = re.search(r'(\d+)\s*([+\-*/x×])\s*(\d+)\s*=\s*(\d+)', text)
    if m:
        return {'a': int(m.group(1)), 'op': m.group(2), 'b': int(m.group(3)), 'c': int(m.group(4))}
    return None

def check_equation(eq):
    """Is the equation self-consistent?"""
    if eq is None: return None
    a, op, b, c = eq['a'], eq['op'], eq['b'], eq['c']
    if op in ('+'): return (a + b) == c
    if op in ('-'): return (a - b) == c
    if op in ('*', 'x', '×'): return (a * b) == c
    if op in ('/'): return b != 0 and (a // b) == c
    return None

def extract_first_number(text):
    """First number in the problem text."""
    nums = re.findall(r'\d+', text)
    return int(nums[0]) if nums else None

def categorize(gen_text, predicted, target, problem_text):
    """Categorize the failure mode."""
    if predicted == target:
        return 'correct'

    if '####' not in gen_text:
        return 'format_error'

    eq = extract_equation(gen_text)
    eq_correct = check_equation(eq)

    first_num = extract_first_number(problem_text)
    if predicted is not None and predicted == first_num:
        return 'copying_input'

    if eq_correct is True:
        return 'self_consistent_wrong'

    if eq is not None and eq_correct is False:
        return 'wrong_arithmetic'

    if predicted is None:
        return 'no_number'

    return 'other_wrong'


def run_diagnostic(clamp_val, num_problems):
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
    print(f"Loaded checkpoint (accuracy={ckpt.get('accuracy', '?')}%)")

    samples = []
    with open('data/per_cycle/gsm8k_single_step_eval.jsonl') as f:
        for line in f:
            if line.strip(): samples.append(json.loads(line.strip()))
    print(f"Loaded {len(samples)} eval problems")
    print(f"Scale clamp: [-{clamp_val}, {clamp_val}]")

    model.eval()

    categories = {}
    details = []

    for idx in range(min(num_problems, len(samples))):
        s = samples[idx]
        problem = s['problem']
        target = s['final_answer']

        inputs = model.tokenizer([problem], return_tensors='pt', truncation=True, max_length=192)
        ids = inputs['input_ids'].to(device)
        attn = inputs['attention_mask'].to(device)

        with torch.no_grad():
            (p0, h0, p1, h1, init_s, ns, f1, out) = model.two_pass_cycle1(ids, attn)

            if clamp_val == 0.0:
                gen = model.transformer.generate(ids, attention_mask=attn, max_new_tokens=60, do_sample=False)
            else:
                gs = init_s.clamp(-clamp_val, clamp_val)
                mgr = AtomAdditiveLoRAManager(model.transformer)
                mgr.apply(model.atoms, gs)
                try:
                    gen = model.transformer.generate(ids, attention_mask=attn, max_new_tokens=60, do_sample=False)
                finally:
                    mgr.remove()

            gen_text = model.tokenizer.decode(gen[0][ids.size(1):], skip_special_tokens=True)
            predicted = extract_answer(gen_text)
            eq = extract_equation(gen_text)
            cat = categorize(gen_text, predicted, target, problem)
            categories[cat] = categories.get(cat, 0) + 1

            detail = {
                'idx': idx,
                'problem': problem[:80],
                'target': target,
                'predicted': predicted,
                'category': cat,
                'gen_text': gen_text[:100],
                'equation': eq,
                'eq_correct': check_equation(eq),
            }
            details.append(detail)

            # Print first 20 with details
            if idx < 20:
                mark = 'OK' if cat == 'correct' else cat.upper()
                print(f"\n[{idx+1}] {mark}")
                print(f"  Problem: {problem[:80]}...")
                print(f"  Target:  {target}")
                print(f"  Gen:     {gen_text[:100]}")
                print(f"  Predicted: {predicted}  Equation: {eq}  EqCorrect: {check_equation(eq)}")

    # Summary
    total = sum(categories.values())
    print(f"\n{'='*60}")
    print(f"CATEGORY BREAKDOWN ({total} problems, clamp={clamp_val})")
    print(f"{'='*60}")
    for cat in sorted(categories.keys(), key=lambda c: -categories[c]):
        count = categories[cat]
        pct = 100 * count / total
        bar = '#' * int(pct / 2)
        print(f"  {cat:30s}  {count:3d}/{total} = {pct:5.1f}%  {bar}")
    print(f"{'='*60}")

    # Detailed equation analysis for wrong_arithmetic
    wrong_arith = [d for d in details if d['category'] == 'wrong_arithmetic']
    if wrong_arith:
        print(f"\nWRONG ARITHMETIC DETAILS ({len(wrong_arith)} cases):")
        for d in wrong_arith[:10]:
            eq = d['equation']
            if eq:
                # What should it have been?
                a, op, b = eq['a'], eq['op'], eq['b']
                if op in ('+'): expected = a + b
                elif op in ('-'): expected = a - b
                elif op in ('*', 'x', '×'): expected = a * b
                elif op in ('/'): expected = a // b if b != 0 else '?'
                else: expected = '?'
                print(f"  {a} {op} {b} = {eq['c']} (should be {expected})")


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--clamp', type=float, default=0.4)
    p.add_argument('--num', type=int, default=50)
    args = p.parse_args()
    run_diagnostic(args.clamp, args.num)
