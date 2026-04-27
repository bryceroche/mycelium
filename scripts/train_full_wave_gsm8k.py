#!/usr/bin/env python3
"""
Full Wave Architecture — GSM8K Training (v24.3)

The complete architecture debut:
  - Haar wavelet preprocessing (2x input compression, 4x attention speedup)
  - 64-atom LoRA (anonymous atoms, tanh scaling, no mode collapse)
  - Fourier pass encoding (smooth rhythmic atom activation)
  - Fourier page encoding (structural identity for page dimensions)
  - Contrastive + anti-copy (pages encode different info per problem)
  - Gradient scaling (earlier cycles get amplified gradients, cap 4x)
  - Fresh data per epoch (augmented GSM8K with number swaps)
  - Answer head + generation (dual evaluation)
  - CoT targets (natural base model completion style)

Target: beat 17.8% (dual LoRA best) and 13.3% (atom baseline).

Usage on AWS g5.xlarge:
  python scripts/train_full_wave_gsm8k.py

Warm-start from L4 if available:
  python scripts/train_full_wave_gsm8k.py --warm checkpoints/atom_lora_L4_best.pt

Monitor with:
  tail -f logs/full_wave_gsm8k.log
"""

import os
import sys
import subprocess
from datetime import datetime

# Ensure we're in the right directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
os.chdir(PROJECT_DIR)
sys.path.insert(0, PROJECT_DIR)


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Full Wave GSM8K Training')
    parser.add_argument('--warm', type=str, default=None,
                        help='Warm-start checkpoint (default: checkpoints/atom_lora_L4_best.pt if exists)')
    parser.add_argument('--epochs', type=int, default=15,
                        help='Number of epochs (default 15)')
    parser.add_argument('--batch_size', type=int, default=12,
                        help='Batch size (default 12 for A10G 24GB)')
    parser.add_argument('--num_passes', type=int, default=5,
                        help='Thinking passes (default 5)')
    parser.add_argument('--eval_size', type=int, default=200,
                        help='Eval problems (default 200, use 500 for final)')
    parser.add_argument('--dry_run', action='store_true',
                        help='Print command without running')
    parser.add_argument('--use_cache', action='store_true',
                        help='Enable page cache (v24.4) for faster training')
    args = parser.parse_args()

    # Auto-detect warm checkpoint
    warm_path = args.warm
    if warm_path is None:
        candidates = [
            'checkpoints/atom_lora_L4_best.pt',
            'checkpoints/atom_lora_L49_best.pt',
            'checkpoints/atom_lora_L47_best.pt',
            'checkpoints/atom_lora_L45_best.pt',
            'checkpoints/atom_lora_L3_best.pt',
        ]
        for c in candidates:
            if os.path.exists(c):
                warm_path = c
                break

    # Build command
    cmd = [
        sys.executable, 'scripts/train_atom_lora.py',
        '--level', 'L5',
        '--epochs', str(args.epochs),
        '--batch_size', str(args.batch_size),
        '--num_passes', str(args.num_passes),
        '--eval_size', str(args.eval_size),
        '--patience', '5',        # longer patience for GSM8K
        '--lam', '0.05',          # contrastive weight
        '--lam_conf', '0.1',      # confidence head weight
        '--lam_answer', '0.3',    # answer head weight
        '--num_atoms', '64',
        '--atom_rank', '6',
        '--num_train', '7473',    # full GSM8K train
    ]

    if args.use_cache:
        cmd.append('--use_cache')

    if warm_path:
        cmd.extend(['--warm', warm_path])

    # Logging setup
    os.makedirs('logs', exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = f'logs/full_wave_gsm8k_{timestamp}.log'

    print("=" * 70)
    print("FULL WAVE ARCHITECTURE — GSM8K TRAINING (v24.3)")
    print("=" * 70)
    print()
    print("Components enabled:")
    print("  [x] Haar wavelet preprocessing (2x compression, 4x attention speedup)")
    print("  [x] 64-atom LoRA (anonymous atoms, tanh scaling)")
    print("  [x] Fourier pass encoding (smooth waveform activation)")
    print("  [x] Fourier page encoding (structural identity)")
    print("  [x] Contrastive + anti-copy loss (page differentiation)")
    print("  [x] Gradient scaling (earlier cycles amplified, cap 4x)")
    print("  [x] Fresh data per epoch (GSM8K number augmentation)")
    print("  [x] Answer head + generation (dual evaluation)")
    print("  [x] Perceiver skip connection (private mid-layer memory)")
    if args.use_cache:
        print("  [x] Page cache + replay buffer (v24.4, up to 2.8x speedup)")
    print()
    print(f"Warm-start: {warm_path or 'None (fresh)'}")
    print(f"Batch size: {args.batch_size}")
    print(f"Passes:     {args.num_passes}")
    print(f"Epochs:     {args.epochs}")
    print(f"Eval size:  {args.eval_size}")
    print(f"Log file:   {log_file}")
    print()
    print("Command:")
    print("  " + " ".join(cmd))
    print("=" * 70)

    if args.dry_run:
        print("\n[DRY RUN] Would execute the above command.")
        return

    print(f"\nStarting training... (logging to {log_file})")
    print("Monitor with: tail -f", log_file)
    print()

    # Run with tee to both console and log file
    with open(log_file, 'w') as f:
        f.write(f"Full Wave GSM8K Training — {timestamp}\n")
        f.write("=" * 70 + "\n")
        f.write(" ".join(cmd) + "\n")
        f.write("=" * 70 + "\n\n")

    # Execute
    env = os.environ.copy()
    env['PYTHONUNBUFFERED'] = '1'

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env,
    )

    with open(log_file, 'a') as f:
        for line in process.stdout:
            print(line, end='')
            f.write(line)
            f.flush()

    process.wait()

    print()
    print("=" * 70)
    if process.returncode == 0:
        print("Training completed successfully!")
    else:
        print(f"Training exited with code {process.returncode}")
    print(f"Log saved to: {log_file}")
    print("=" * 70)


if __name__ == '__main__':
    main()
