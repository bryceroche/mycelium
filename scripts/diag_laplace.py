#!/usr/bin/env python3
"""Laplace-like analysis of page trajectories in the breathing loop.

Reveals whether pages converge, oscillate, or stall across cycles.
Runs on the current best checkpoint against eval data.

Usage:
  python scripts/diag_laplace.py [--num_problems 20] [--num_passes 3]
"""
import sys, argparse
sys.path.insert(0, '.')
import json, torch, numpy as np
import torch.nn.functional as F
from scripts.atom_lora import AtomLoRAModel, AtomAdditiveLoRAManager

def load_model(ckpt_path, device):
    model = AtomLoRAModel()
    model.compressor = model.compressor.to(device=device, dtype=torch.bfloat16)
    model.hypernet = model.hypernet.to(device=device, dtype=torch.bfloat16)
    model.atoms = model.atoms.to(device=device, dtype=torch.bfloat16)
    model.confidence_head = model.confidence_head.to(device)
    model.residual_gate = model.residual_gate.to(device=device, dtype=torch.bfloat16)
    model.probe_head = model.probe_head.to(device)
    model.message_generator = model.message_generator.to(device)
    model.ordinal_head = model.ordinal_head.to(device)
    model.mobius = model.mobius.to(device)

    ckpt = torch.load(ckpt_path, map_location='cpu')
    for name in ['compressor', 'atoms', 'hypernet', 'confidence_head',
                 'residual_gate', 'message_generator', 'ordinal_head']:
        if name in ckpt:
            obj = getattr(model, name)
            own = obj.state_dict()
            for k, v in ckpt[name].items():
                if k in own and own[k].shape == v.shape:
                    own[k] = v
            obj.load_state_dict(own, strict=False)
    print(f"Loaded {ckpt_path} (accuracy={ckpt.get('accuracy', '?')}%)")
    return model


def analyze_page_trajectory(pages_list):
    """Analyze the trajectory of pages across cycles.

    Args:
        pages_list: list of (64,) tensors, one per cycle

    Returns dict with diagnostic metrics.
    """
    if len(pages_list) < 2:
        return {}

    pages = torch.stack(pages_list).cpu().float().numpy()  # (num_cycles, 64)
    num_cycles, dim = pages.shape

    # 1. Consecutive cosine similarities
    cos_consecutive = []
    for i in range(1, num_cycles):
        c = np.dot(pages[i], pages[i-1]) / (np.linalg.norm(pages[i]) * np.linalg.norm(pages[i-1]) + 1e-8)
        cos_consecutive.append(c)

    # 2. Skip cosine (k vs k-2) — oscillation detector
    cos_skip = []
    for i in range(2, num_cycles):
        c = np.dot(pages[i], pages[i-2]) / (np.linalg.norm(pages[i]) * np.linalg.norm(pages[i-2]) + 1e-8)
        cos_skip.append(c)

    # 3. Oscillation score: when skip_cos > consecutive_cos, pages oscillate
    oscillation = 0.0
    if cos_skip and cos_consecutive:
        # Compare skip(k, k-2) vs consecutive(k, k-1)
        for i in range(len(cos_skip)):
            oscillation += max(0, cos_skip[i] - cos_consecutive[i + 1])
        oscillation /= len(cos_skip)

    # 4. FFT per dimension — find dominant frequencies
    if num_cycles >= 3:
        spectra = np.abs(np.fft.rfft(pages, axis=0))  # (num_cycles//2+1, 64)
        # DC component (mean) vs AC components (variation)
        dc_power = spectra[0] ** 2
        ac_power = (spectra[1:] ** 2).sum(axis=0)
        total_power = dc_power + ac_power + 1e-8
        dc_ratio = (dc_power / total_power).mean()  # high = stuck, low = changing

        # Dominant non-DC frequency per dim
        if spectra.shape[0] > 1:
            dominant_freqs = np.argmax(spectra[1:], axis=0) + 1
            mean_dominant_freq = dominant_freqs.mean()
        else:
            mean_dominant_freq = 0
    else:
        dc_ratio = 1.0
        mean_dominant_freq = 0

    # 5. Delta norms — are changes getting smaller (converging)?
    delta_norms = []
    for i in range(1, num_cycles):
        delta_norms.append(np.linalg.norm(pages[i] - pages[i-1]))

    converging = False
    if len(delta_norms) >= 2:
        converging = delta_norms[-1] < delta_norms[0]

    # 6. Page norms — are pages staying on the sphere?
    page_norms = [np.linalg.norm(p) for p in pages]

    return {
        'cos_consecutive': cos_consecutive,
        'cos_skip': cos_skip,
        'oscillation_score': oscillation,
        'dc_ratio': dc_ratio,
        'mean_dominant_freq': mean_dominant_freq,
        'delta_norms': delta_norms,
        'converging': converging,
        'page_norms': page_norms,
    }


def run_diagnostic(model, samples, device, num_passes=3, num_problems=20):
    """Run Laplace diagnostic on eval problems."""
    model.eval()

    all_results = []

    for idx in range(min(num_problems, len(samples))):
        sample = samples[idx]
        problem = sample['problem']
        final_answer = sample['final_answer']

        inputs = model.tokenizer(
            problem, return_tensors='pt', truncation=True, max_length=192,
        )
        input_ids = inputs['input_ids'].to(device)
        attention_mask = inputs['attention_mask'].to(device)

        state_pages = []
        mid_states_history = []
        messages = []
        all_scales = []
        all_raw_pages = []

        with torch.no_grad():
            for pass_num in range(num_passes):
                page, scales, mid_states, message, raw_page, _hidden, _focus = model.thinking_pass(
                    input_ids, attention_mask, state_pages, pass_num,
                    prev_mid_states=mid_states_history if mid_states_history else None,
                    messages=messages if messages else None,
                )
                state_pages.append(page)
                mid_states_history.append(mid_states)
                messages.append(message)
                all_scales.append(scales.squeeze(0))
                all_raw_pages.append(raw_page.squeeze(0))

        # Analyze this problem's page trajectory
        pages_squeezed = [p.squeeze(0) for p in state_pages]
        result = analyze_page_trajectory(pages_squeezed)
        result['problem_idx'] = idx
        result['final_answer'] = final_answer

        # Atom scale cosine between cycles
        scale_cos = []
        for i in range(1, len(all_scales)):
            c = F.cosine_similarity(all_scales[i].unsqueeze(0), all_scales[i-1].unsqueeze(0)).item()
            scale_cos.append(c)
        result['scale_cos'] = scale_cos

        # Raw page cosine (before Mobius) between cycles
        raw_page_cos = []
        for i in range(1, len(all_raw_pages)):
            c = F.cosine_similarity(all_raw_pages[i].unsqueeze(0), all_raw_pages[i-1].unsqueeze(0)).item()
            raw_page_cos.append(c)
        result['raw_page_cos'] = raw_page_cos

        all_results.append(result)

    return all_results


def print_report(results, num_passes):
    """Print a summary report."""
    print("=" * 80)
    print(f"LAPLACE PAGE TRAJECTORY DIAGNOSTIC ({len(results)} problems, {num_passes} cycles)")
    print("=" * 80)

    # Aggregate stats
    all_cos_consec = []
    all_cos_skip = []
    all_osc = []
    all_dc = []
    all_delta = [[] for _ in range(num_passes - 1)]
    all_converging = 0

    for r in results:
        all_cos_consec.extend(r.get('cos_consecutive', []))
        all_cos_skip.extend(r.get('cos_skip', []))
        all_osc.append(r.get('oscillation_score', 0))
        all_dc.append(r.get('dc_ratio', 1))
        for i, d in enumerate(r.get('delta_norms', [])):
            if i < len(all_delta):
                all_delta[i].append(d)
        if r.get('converging', False):
            all_converging += 1

    print(f"\n--- Consecutive Cosine (cycle k vs k-1) ---")
    if all_cos_consec:
        print(f"  Mean: {np.mean(all_cos_consec):.4f}  (1.0=identical, 0.0=orthogonal)")
        for i in range(num_passes - 1):
            vals = [r['cos_consecutive'][i] for r in results if i < len(r.get('cos_consecutive', []))]
            if vals:
                print(f"  Cycle {i+1}→{i+2}: {np.mean(vals):.4f} (std={np.std(vals):.4f})")

    print(f"\n--- Skip Cosine (cycle k vs k-2) — oscillation detector ---")
    if all_cos_skip:
        print(f"  Mean: {np.mean(all_cos_skip):.4f}")
        for i in range(num_passes - 2):
            vals = [r['cos_skip'][i] for r in results if i < len(r.get('cos_skip', []))]
            if vals:
                print(f"  Cycle {i+1}→{i+3}: {np.mean(vals):.4f} (std={np.std(vals):.4f})")

    print(f"\n--- Oscillation Score ---")
    print(f"  Mean: {np.mean(all_osc):.4f}  (0.0=no oscillation, >0.1=oscillating)")
    print(f"  Max:  {np.max(all_osc):.4f}")
    osc_problems = sum(1 for o in all_osc if o > 0.05)
    print(f"  Problems oscillating (>0.05): {osc_problems}/{len(results)}")

    print(f"\n--- DC Ratio (stuck detector) ---")
    print(f"  Mean: {np.mean(all_dc):.4f}  (1.0=stuck/constant, <0.5=changing)")

    print(f"\n--- Delta Norms (convergence) ---")
    for i, deltas in enumerate(all_delta):
        if deltas:
            print(f"  Cycle {i+1}→{i+2}: mean={np.mean(deltas):.4f} std={np.std(deltas):.4f}")
    print(f"  Converging (last delta < first delta): {all_converging}/{len(results)}")

    print(f"\n--- Atom Scale Cosine (xpass_cos per cycle pair) ---")
    for i in range(num_passes - 1):
        vals = [r['scale_cos'][i] for r in results if i < len(r.get('scale_cos', []))]
        if vals:
            print(f"  Cycle {i+1}→{i+2}: {np.mean(vals):.4f} (std={np.std(vals):.4f})")
    all_scale_cos = []
    for r in results:
        all_scale_cos.extend(r.get('scale_cos', []))
    if all_scale_cos:
        print(f"  Overall mean: {np.mean(all_scale_cos):.4f}  (1.0=identical scales, <0.5=different)")

    print(f"\n--- Raw Page Cosine (before Mobius warp) ---")
    for i in range(num_passes - 1):
        vals = [r['raw_page_cos'][i] for r in results if i < len(r.get('raw_page_cos', []))]
        if vals:
            print(f"  Cycle {i+1}→{i+2}: {np.mean(vals):.4f} (std={np.std(vals):.4f})")

    print(f"\n--- Page Norms ---")
    for i in range(num_passes):
        norms = [r['page_norms'][i] for r in results if i < len(r.get('page_norms', []))]
        if norms:
            print(f"  Cycle {i+1}: mean={np.mean(norms):.2f} (target={np.sqrt(64):.2f})")

    # Per-problem detail for first 5
    print(f"\n--- Per-Problem Detail (first 5) ---")
    for r in results[:5]:
        idx = r['problem_idx']
        cos_c = [f"{c:.3f}" for c in r.get('cos_consecutive', [])]
        cos_s = [f"{c:.3f}" for c in r.get('cos_skip', [])]
        sc = [f"{c:.3f}" for c in r.get('scale_cos', [])]
        rpc = [f"{c:.3f}" for c in r.get('raw_page_cos', [])]
        deltas = [f"{d:.3f}" for d in r.get('delta_norms', [])]
        osc = r.get('oscillation_score', 0)
        dc = r.get('dc_ratio', 1)
        conv = "YES" if r.get('converging') else "NO"
        print(f"  Problem {idx}: page_cos={cos_c} scale_cos={sc} raw_page_cos={rpc}")
        print(f"    deltas={deltas} osc={osc:.4f} dc={dc:.4f} converging={conv}")

    print("\n" + "=" * 80)

    # Interpretation
    mean_osc = np.mean(all_osc)
    mean_dc = np.mean(all_dc)
    mean_cos = np.mean(all_cos_consec) if all_cos_consec else 0

    print("INTERPRETATION:")
    if mean_cos > 0.8:
        print("  STUCK: Pages barely change between cycles (cos > 0.8)")
        print("  → Cycles are redundant. Need stronger differentiation.")
    elif mean_cos > 0.6:
        print("  CLUSTERED: Pages moderately similar (cos 0.6-0.8)")
        print("  → Some differentiation but not enough. Mobius should help.")
    else:
        print("  DIVERSE: Pages genuinely different (cos < 0.6)")
        print("  → Good cycle differentiation.")

    if mean_osc > 0.05:
        print(f"  OSCILLATING: Score {mean_osc:.3f} > 0.05")
        print("  → Pages bounce between states. Anti-oscillation reg may help.")
    else:
        print(f"  NOT OSCILLATING: Score {mean_osc:.3f}")

    if mean_dc > 0.8:
        print(f"  DC-DOMINANT: Ratio {mean_dc:.3f} > 0.8")
        print("  → Pages are mostly constant across cycles (stuck).")
    elif mean_dc > 0.5:
        print(f"  MIXED: DC ratio {mean_dc:.3f}")
        print("  → Some change, some constant dimensions.")
    else:
        print(f"  AC-DOMINANT: DC ratio {mean_dc:.3f} < 0.5")
        print("  → Pages changing substantially across cycles (good).")

    if all_converging > len(results) * 0.7:
        print(f"  CONVERGING: {all_converging}/{len(results)} problems")
        print("  → Deltas shrink over cycles. Loop is settling.")
    else:
        print(f"  NOT CONVERGING: Only {all_converging}/{len(results)} converge")


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--num_problems', type=int, default=20)
    p.add_argument('--num_passes', type=int, default=3)
    p.add_argument('--ckpt', type=str, default='checkpoints/per_cycle_gsm8k_best.pt')
    args = p.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = load_model(args.ckpt, device)

    samples = []
    with open('data/per_cycle/gsm8k_eval.jsonl') as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    print(f"Loaded {len(samples)} eval problems\n")

    results = run_diagnostic(model, samples, device,
                            num_passes=args.num_passes,
                            num_problems=args.num_problems)
    print_report(results, args.num_passes)
