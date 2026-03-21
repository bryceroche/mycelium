"""
Atom activation spectrogram diagnostic for Fourier smoothness analysis.

Plots a heatmap of atom scales across passes for a set of eval problems.
Shows whether atoms activate in smooth waves (Fourier-like) or arbitrary
jumps (discrete embedding behavior).

Key outputs:
1. Atom spectrogram visualization (atom x pass heatmap)
2. Cross-pass cosine similarity: cos(scales_pass_N, scales_pass_N+1)
3. Smoothness metric: how wave-like are the atom activations?

The smoothness metric measures whether atom activations follow smooth waveforms
across passes (expected with Fourier pass encoding) vs arbitrary jumps
(expected with discrete nn.Embedding).

Usage:
  python scripts/diag_atom_spectrogram.py --checkpoint checkpoints/atom_lora_L3_best.pt --level L3
  python scripts/diag_atom_spectrogram.py --checkpoint checkpoints/atom_lora_L3_best.pt --level L3 --num_problems 50
  python scripts/diag_atom_spectrogram.py --checkpoint checkpoints/atom_lora_gsm8k_best.pt --level L5 --output logs/
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, '/home/ubuntu/mycelium')

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from scripts.atom_lora import AtomLoRAModel, AtomAdditiveLoRAManager
from scripts.train_atom_lora import make_eval_dataset, try_warm_start
from scripts.atom_lora import AnswerHead


def collect_atom_scales(model, dataset, num_problems, num_passes, device):
    """Run thinking loop and collect atom scales for each problem and pass.

    Returns:
        scales: np.ndarray of shape (num_problems, num_passes, num_atoms)
    """
    model.eval()
    all_scales = []
    max_length = 128

    with torch.no_grad():
        for idx in range(min(num_problems, len(dataset))):
            sample = dataset[idx]
            problem = sample['problem']

            inputs = model.tokenizer(
                [problem], return_tensors='pt', padding=True,
                truncation=True, max_length=max_length,
            )
            input_ids = inputs['input_ids'].to(device)
            attention_mask = inputs['attention_mask'].to(device)

            state_pages = []
            problem_scales = []

            mid_states_history = []
            for pass_num in range(num_passes):
                page, atom_scales, current_mid_states = model.thinking_pass(
                    input_ids, attention_mask, state_pages, pass_num,
                    prev_mid_states=mid_states_history if mid_states_history else None,
                )
                state_pages.append(page)
                mid_states_history.append(current_mid_states)
                # atom_scales: (1, num_atoms)
                problem_scales.append(atom_scales[0].float().cpu().numpy())

            all_scales.append(np.stack(problem_scales, axis=0))  # (P, A)

    return np.stack(all_scales, axis=0)  # (N, P, A)


def compute_smoothness_metrics(scales):
    """Compute smoothness metrics for atom activations across passes.

    Smoothness measures how wave-like the atom activations are:
    - High smoothness = smooth waveforms (Fourier encoding)
    - Low smoothness = arbitrary jumps (discrete embedding)

    Args:
        scales: np.ndarray of shape (num_problems, num_passes, num_atoms)

    Returns:
        dict with:
            - consecutive_cosine: mean cosine between pass N and N+1
            - per_atom_smoothness: smoothness score per atom (num_atoms,)
            - overall_smoothness: single scalar summarizing smoothness
            - second_derivative_energy: measures jerkiness (lower = smoother)
    """
    num_problems, num_passes, num_atoms = scales.shape

    # 1. Consecutive-pass cosine similarity
    # cos(scales_pass_N, scales_pass_N+1) averaged across all consecutive pairs
    consecutive_cos = []
    for p in range(num_passes - 1):
        for n in range(num_problems):
            v1 = scales[n, p, :]
            v2 = scales[n, p + 1, :]
            n1 = np.linalg.norm(v1) + 1e-8
            n2 = np.linalg.norm(v2) + 1e-8
            cos = np.dot(v1, v2) / (n1 * n2)
            consecutive_cos.append(cos)
    consecutive_cosine_mean = np.mean(consecutive_cos)
    consecutive_cosine_std = np.std(consecutive_cos)

    # 2. Per-atom smoothness: measure how smooth each atom's trajectory is
    # Use second derivative (acceleration) as a measure of jerkiness
    # Lower second derivative = smoother waveform
    per_atom_smoothness = np.zeros(num_atoms)
    per_atom_second_deriv = np.zeros(num_atoms)

    for a in range(num_atoms):
        atom_trajectories = scales[:, :, a]  # (N, P) - all problems, this atom

        # Second derivative for each problem
        second_derivs = []
        for n in range(num_problems):
            traj = atom_trajectories[n]  # (P,)
            if num_passes >= 3:
                # Second derivative: d2y/dx2 = y[i+1] - 2*y[i] + y[i-1]
                second_deriv = traj[2:] - 2 * traj[1:-1] + traj[:-2]
                second_derivs.append(np.mean(np.abs(second_deriv)))
            else:
                # Not enough passes for second derivative
                second_derivs.append(0.0)

        mean_second_deriv = np.mean(second_derivs)
        per_atom_second_deriv[a] = mean_second_deriv

        # Smoothness = 1 / (1 + second_deriv)
        # Higher smoothness = lower second derivative = smoother waveform
        per_atom_smoothness[a] = 1.0 / (1.0 + mean_second_deriv)

    # 3. Overall smoothness: mean across atoms
    overall_smoothness = np.mean(per_atom_smoothness)

    # 4. Second derivative energy (overall jerkiness measure)
    second_derivative_energy = np.mean(per_atom_second_deriv)

    # 5. Autocorrelation lag-1 (high = smooth transitions)
    # Measures correlation between consecutive pass activations
    autocorr_lag1 = []
    for a in range(num_atoms):
        for n in range(num_problems):
            traj = scales[n, :, a]  # (P,)
            if num_passes >= 2:
                corr = np.corrcoef(traj[:-1], traj[1:])[0, 1]
                if not np.isnan(corr):
                    autocorr_lag1.append(corr)
    autocorr_mean = np.mean(autocorr_lag1) if autocorr_lag1 else 0.0

    return {
        'consecutive_cosine_mean': consecutive_cosine_mean,
        'consecutive_cosine_std': consecutive_cosine_std,
        'per_atom_smoothness': per_atom_smoothness,
        'overall_smoothness': overall_smoothness,
        'second_derivative_energy': second_derivative_energy,
        'autocorr_lag1': autocorr_mean,
        'per_atom_second_deriv': per_atom_second_deriv,
    }


def plot_spectrogram(scales, output_path, smoothness_metrics=None):
    """Plot the atom activation spectrogram.

    Args:
        scales: (num_problems, num_passes, num_atoms)
        output_path: where to save PNG
        smoothness_metrics: dict from compute_smoothness_metrics (optional)
    """
    num_problems, num_passes, num_atoms = scales.shape

    # Average across problems for the main heatmap
    mean_scales = scales.mean(axis=0)  # (P, A)

    fig, axes = plt.subplots(2, 3, figsize=(20, 10))

    # Build title with smoothness metrics if available
    title = f'Atom Activation Spectrogram ({num_problems} problems, {num_passes} passes)'
    if smoothness_metrics:
        title += f'\nSmoothness: {smoothness_metrics["overall_smoothness"]:.3f} | '
        title += f'Consecutive cos: {smoothness_metrics["consecutive_cosine_mean"]:.3f} | '
        title += f'Autocorr lag-1: {smoothness_metrics["autocorr_lag1"]:.3f}'
    fig.suptitle(title, fontsize=14, fontweight='bold')

    # Top-left: mean activation heatmap
    ax = axes[0, 0]
    im = ax.imshow(mean_scales.T, aspect='auto', cmap='RdBu_r',
                   vmin=-1, vmax=1, interpolation='nearest')
    ax.set_xlabel('Pass')
    ax.set_ylabel('Atom index')
    ax.set_title('Mean atom scales (across problems)')
    ax.set_xticks(range(num_passes))
    ax.set_xticklabels([str(i + 1) for i in range(num_passes)])
    plt.colorbar(im, ax=ax, shrink=0.8)

    # Top-middle: std across problems (shows problem-sensitivity)
    ax = axes[0, 1]
    std_scales = scales.std(axis=0)  # (P, A)
    im2 = ax.imshow(std_scales.T, aspect='auto', cmap='viridis',
                    vmin=0, interpolation='nearest')
    ax.set_xlabel('Pass')
    ax.set_ylabel('Atom index')
    ax.set_title('Std of atom scales (across problems)')
    ax.set_xticks(range(num_passes))
    ax.set_xticklabels([str(i + 1) for i in range(num_passes)])
    plt.colorbar(im2, ax=ax, shrink=0.8)

    # Top-right: per-atom smoothness bar chart
    ax = axes[0, 2]
    if smoothness_metrics:
        smoothness = smoothness_metrics['per_atom_smoothness']
        # Color by smoothness: green = smooth, red = jerky
        colors = plt.cm.RdYlGn(smoothness)
        ax.bar(range(num_atoms), smoothness, color=colors, width=1.0, edgecolor='none')
        ax.axhline(y=np.mean(smoothness), color='blue', linestyle='--',
                   label=f'mean={np.mean(smoothness):.3f}')
        ax.set_xlabel('Atom index')
        ax.set_ylabel('Smoothness (1 / (1 + |d2/dx2|))')
        ax.set_title('Per-atom smoothness (higher = wave-like)')
        ax.legend(loc='lower right')
        ax.set_xlim(-0.5, num_atoms - 0.5)
    else:
        ax.text(0.5, 0.5, 'No smoothness metrics', ha='center', va='center')
        ax.set_title('Per-atom smoothness')

    # Bottom-left: per-pass scale distribution (violin-like)
    ax = axes[1, 0]
    for p in range(num_passes):
        pass_scales = scales[:, p, :].flatten()
        ax.violinplot(pass_scales, positions=[p], showmeans=True, widths=0.7)
    ax.set_xlabel('Pass')
    ax.set_ylabel('Scale value')
    ax.set_title('Scale distribution per pass')
    ax.set_xticks(range(num_passes))
    ax.set_xticklabels([str(i + 1) for i in range(num_passes)])
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

    # Bottom-middle: cross-pass cosine similarity
    ax = axes[1, 1]
    # Mean scale vector per pass across problems: (P, A)
    # Cosine similarity matrix between passes
    cos_matrix = np.zeros((num_passes, num_passes))
    for i in range(num_passes):
        for j in range(num_passes):
            vi = mean_scales[i]
            vj = mean_scales[j]
            norm_i = np.linalg.norm(vi) + 1e-8
            norm_j = np.linalg.norm(vj) + 1e-8
            cos_matrix[i, j] = np.dot(vi, vj) / (norm_i * norm_j)
    im3 = ax.imshow(cos_matrix, cmap='coolwarm', vmin=-1, vmax=1)
    ax.set_xlabel('Pass')
    ax.set_ylabel('Pass')
    ax.set_title('Cross-pass cosine similarity (mean scales)')
    ax.set_xticks(range(num_passes))
    ax.set_xticklabels([str(i + 1) for i in range(num_passes)])
    ax.set_yticks(range(num_passes))
    ax.set_yticklabels([str(i + 1) for i in range(num_passes)])
    for i in range(num_passes):
        for j in range(num_passes):
            ax.text(j, i, f'{cos_matrix[i, j]:.2f}', ha='center', va='center',
                    fontsize=9, color='black' if abs(cos_matrix[i, j]) < 0.5 else 'white')
    plt.colorbar(im3, ax=ax, shrink=0.8)

    # Bottom-right: example atom waveforms (show 6 atoms with varying smoothness)
    ax = axes[1, 2]
    if smoothness_metrics:
        smoothness = smoothness_metrics['per_atom_smoothness']
        # Pick atoms: 2 smoothest, 2 medium, 2 jerkiest
        sorted_idx = np.argsort(smoothness)
        sample_atoms = [
            sorted_idx[-1], sorted_idx[-2],  # smoothest
            sorted_idx[num_atoms // 2], sorted_idx[num_atoms // 2 + 1],  # medium
            sorted_idx[0], sorted_idx[1],  # jerkiest
        ]
        colors = ['green', 'limegreen', 'orange', 'gold', 'red', 'darkred']
        labels = ['smoothest', 'smooth', 'medium', 'medium', 'jerky', 'jerkiest']

        for atom_idx, color, label in zip(sample_atoms, colors, labels):
            # Average trajectory across problems
            mean_traj = scales[:, :, atom_idx].mean(axis=0)
            ax.plot(range(1, num_passes + 1), mean_traj, 'o-', color=color,
                    label=f'Atom {atom_idx} ({label})', linewidth=2, markersize=6)

        ax.set_xlabel('Pass')
        ax.set_ylabel('Mean scale')
        ax.set_title('Example atom waveforms')
        ax.legend(loc='best', fontsize=8)
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax.set_xticks(range(1, num_passes + 1))
    else:
        ax.text(0.5, 0.5, 'No smoothness metrics', ha='center', va='center')
        ax.set_title('Example atom waveforms')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"\nSaved spectrogram to {output_path}")


def print_summary(scales, smoothness_metrics=None):
    """Print summary statistics about atom activations.

    Args:
        scales: (num_problems, num_passes, num_atoms)
        smoothness_metrics: dict from compute_smoothness_metrics (optional)
    """
    num_problems, num_passes, num_atoms = scales.shape
    print(f"\n{'='*60}")
    print(f"ATOM ACTIVATION SUMMARY")
    print(f"  Problems: {num_problems}, Passes: {num_passes}, Atoms: {num_atoms}")
    print(f"{'='*60}")

    # Smoothness metrics (key diagnostic for Fourier encoding)
    if smoothness_metrics:
        print(f"\n--- SMOOTHNESS METRICS (Fourier diagnostic) ---")
        print(f"  Overall smoothness:       {smoothness_metrics['overall_smoothness']:.4f}")
        print(f"    (higher = more wave-like, expected with Fourier encoding)")
        print(f"  Consecutive-pass cosine:  {smoothness_metrics['consecutive_cosine_mean']:.4f} "
              f"+/- {smoothness_metrics['consecutive_cosine_std']:.4f}")
        print(f"    (higher = smoother transitions between consecutive passes)")
        print(f"  Autocorrelation lag-1:    {smoothness_metrics['autocorr_lag1']:.4f}")
        print(f"    (higher = atoms maintain activation patterns across passes)")
        print(f"  Second derivative energy: {smoothness_metrics['second_derivative_energy']:.4f}")
        print(f"    (lower = smoother waveforms, less jerkiness)")

        # Per-atom smoothness distribution
        per_atom = smoothness_metrics['per_atom_smoothness']
        print(f"\n  Per-atom smoothness distribution:")
        print(f"    min:  {per_atom.min():.4f}")
        print(f"    25%:  {np.percentile(per_atom, 25):.4f}")
        print(f"    50%:  {np.percentile(per_atom, 50):.4f}")
        print(f"    75%:  {np.percentile(per_atom, 75):.4f}")
        print(f"    max:  {per_atom.max():.4f}")

        # Identify smoothest and jerkiest atoms
        sorted_idx = np.argsort(per_atom)
        print(f"\n  Smoothest atoms: {sorted_idx[-5:][::-1].tolist()} "
              f"(smoothness: {per_atom[sorted_idx[-5:][::-1]].round(3).tolist()})")
        print(f"  Jerkiest atoms:  {sorted_idx[:5].tolist()} "
              f"(smoothness: {per_atom[sorted_idx[:5]].round(3).tolist()})")

    # Active atoms per pass (|scale| > 0.1)
    print(f"\n--- ACTIVATION PATTERNS ---")
    print(f"Active atoms per pass (|scale| > 0.1):")
    for p in range(num_passes):
        pass_scales = scales[:, p, :]  # (N, A)
        active_per_problem = (np.abs(pass_scales) > 0.1).sum(axis=1)  # (N,)
        print(f"  Pass {p+1}: {active_per_problem.mean():.1f} +/- {active_per_problem.std():.1f} "
              f"(min={active_per_problem.min()}, max={active_per_problem.max()})")

    # Cross-pass cosine similarity (per problem, then averaged)
    print(f"\nCross-pass cosine similarity (per-problem mean):")
    consecutive_cos = []
    for i in range(num_passes):
        for j in range(i + 1, num_passes):
            cos_vals = []
            for n in range(num_problems):
                vi = scales[n, i, :]
                vj = scales[n, j, :]
                ni = np.linalg.norm(vi) + 1e-8
                nj = np.linalg.norm(vj) + 1e-8
                cos_vals.append(np.dot(vi, vj) / (ni * nj))
            cos_arr = np.array(cos_vals)
            if j == i + 1:
                consecutive_cos.append(cos_arr.mean())
            print(f"  Pass {i+1} vs {j+1}: {cos_arr.mean():.4f} +/- {cos_arr.std():.4f}")

    # Highlight consecutive-pass cosines
    if consecutive_cos:
        print(f"\n  Consecutive-pass cosines: {[f'{c:.3f}' for c in consecutive_cos]}")
        print(f"  Mean consecutive cos: {np.mean(consecutive_cos):.4f}")

    # Always-on atoms (|mean scale| > 0.3 at every pass)
    mean_abs = np.abs(scales.mean(axis=0))  # (P, A)
    always_on = np.all(mean_abs > 0.3, axis=0)  # (A,)
    always_on_idx = np.where(always_on)[0]
    print(f"\n--- ATOM BEHAVIOR PATTERNS ---")
    print(f"Always-on atoms (|mean| > 0.3 every pass): {len(always_on_idx)}")
    if len(always_on_idx) > 0:
        print(f"  Indices: {always_on_idx.tolist()}")

    # Pass-specific atoms (active in one pass, inactive in others)
    pass_specific = []
    for p in range(num_passes):
        active_this = mean_abs[p] > 0.3
        active_others = np.any(mean_abs[np.arange(num_passes) != p] > 0.3, axis=0)
        specific = active_this & ~active_others
        specific_idx = np.where(specific)[0]
        if len(specific_idx) > 0:
            pass_specific.append((p + 1, specific_idx.tolist()))
    print(f"\nPass-specific atoms (active only in one pass):")
    if pass_specific:
        for pass_num, indices in pass_specific:
            print(f"  Pass {pass_num}: {indices}")
    else:
        print(f"  None found")

    # Dead atoms (|mean scale| < 0.05 at every pass)
    dead = np.all(mean_abs < 0.05, axis=0)
    dead_idx = np.where(dead)[0]
    print(f"\nDead atoms (|mean| < 0.05 every pass): {len(dead_idx)}/{num_atoms}")

    # Scale magnitude stats
    print(f"\n--- SCALE STATISTICS ---")
    print(f"  Overall mean |scale|: {np.abs(scales).mean():.4f}")
    print(f"  Overall max  |scale|: {np.abs(scales).max():.4f}")
    print(f"  Overall std  scale:   {scales.std():.4f}")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(
        description='Atom activation spectrogram diagnostic for Fourier smoothness analysis',
    )
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to atom LoRA checkpoint')
    parser.add_argument('--level', type=str, default='L3',
                        choices=['L3', 'L4', 'L4.5', 'L4.7', 'L4.9', 'L5'],
                        help='Dataset level for eval problems')
    parser.add_argument('--num_problems', type=int, default=20,
                        help='Number of problems to evaluate')
    parser.add_argument('--num_passes', type=int, default=5,
                        help='Number of thinking passes')
    parser.add_argument('--output', type=str, default='logs/',
                        help='Output path (file or directory). If directory, saves atom_spectrogram.png there.')
    parser.add_argument('--num_atoms', type=int, default=64,
                        help='Number of LoRA atoms (must match checkpoint)')
    parser.add_argument('--atom_rank', type=int, default=6,
                        help='Rank of each LoRA atom (must match checkpoint)')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Determine output path
    output_path = args.output
    if os.path.isdir(output_path) or output_path.endswith('/'):
        os.makedirs(output_path, exist_ok=True)
        # Generate filename from checkpoint name
        ckpt_name = os.path.basename(args.checkpoint).replace('.pt', '')
        output_path = os.path.join(output_path, f'atom_spectrogram_{ckpt_name}.png')

    # Load model
    print(f"Loading AtomLoRAModel (atoms={args.num_atoms}, rank={args.atom_rank})...")
    model = AtomLoRAModel(num_atoms=args.num_atoms, atom_rank=args.atom_rank)
    model.compressor = model.compressor.to(device=device, dtype=torch.bfloat16)
    model.hypernet = model.hypernet.to(device=device, dtype=torch.bfloat16)
    model.atoms = model.atoms.to(device=device, dtype=torch.bfloat16)
    model.confidence_head = model.confidence_head.to(device)

    answer_head = AnswerHead(page_size=model.page_size).to(device)

    # Load checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    try_warm_start(model, answer_head, args.checkpoint)

    # Load eval dataset
    print(f"Loading eval dataset: level={args.level}, n={args.num_problems}")
    eval_dataset = make_eval_dataset(args.level, num_samples=args.num_problems)
    actual_n = min(args.num_problems, len(eval_dataset))
    print(f"  Got {actual_n} problems")

    # Collect atom scales
    print(f"Running {args.num_passes} thinking passes on {actual_n} problems...")
    scales = collect_atom_scales(
        model, eval_dataset, actual_n, args.num_passes, device,
    )
    print(f"  Collected scales: {scales.shape}")

    # Compute smoothness metrics
    print(f"Computing smoothness metrics...")
    smoothness_metrics = compute_smoothness_metrics(scales)

    # Plot with smoothness metrics
    plot_spectrogram(scales, output_path, smoothness_metrics)

    # Summary stats with smoothness
    print_summary(scales, smoothness_metrics)

    # Save smoothness metrics to JSON for tracking
    metrics_path = output_path.replace('.png', '_metrics.json')
    metrics_to_save = {
        'checkpoint': args.checkpoint,
        'level': args.level,
        'num_problems': actual_n,
        'num_passes': args.num_passes,
        'overall_smoothness': float(smoothness_metrics['overall_smoothness']),
        'consecutive_cosine_mean': float(smoothness_metrics['consecutive_cosine_mean']),
        'consecutive_cosine_std': float(smoothness_metrics['consecutive_cosine_std']),
        'autocorr_lag1': float(smoothness_metrics['autocorr_lag1']),
        'second_derivative_energy': float(smoothness_metrics['second_derivative_energy']),
        'per_atom_smoothness_mean': float(np.mean(smoothness_metrics['per_atom_smoothness'])),
        'per_atom_smoothness_std': float(np.std(smoothness_metrics['per_atom_smoothness'])),
    }
    with open(metrics_path, 'w') as f:
        json.dump(metrics_to_save, f, indent=2)
    print(f"Saved metrics to {metrics_path}")


if __name__ == '__main__':
    main()
