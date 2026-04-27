#!/usr/bin/env python3
"""
Quick diagnostic: per-dimension variance of pages across eval problems.
Shows WHERE the page variation is concentrated.

Usage: python3 scripts/diag_page_variance.py --checkpoint checkpoints/sympy_decoder_best.pt
"""
import argparse, json, os, sys, torch
import torch.nn.functional as F

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_DIR)

from scripts.atom_lora import AtomLoRAModel

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='checkpoints/sympy_decoder_best.pt')
    parser.add_argument('--num_problems', type=int, default=50)
    parser.add_argument('--max_passes', type=int, default=5)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load model
    model = AtomLoRAModel(num_atoms=64, atom_rank=6, skip_pass_embed=True)
    model.compressor = model.compressor.to(device=device, dtype=torch.bfloat16)
    model.hypernet = model.hypernet.to(device=device, dtype=torch.bfloat16)
    model.atoms = model.atoms.to(device=device, dtype=torch.bfloat16)
    model.confidence_head = model.confidence_head.to(device)
    model.residual_gate = model.residual_gate.to(device=device, dtype=torch.bfloat16)

    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    for name, sd_key in [('compressor', 'compressor'), ('atoms', 'atoms'),
                          ('hypernet', 'hypernet'), ('residual_gate', 'residual_gate')]:
        if sd_key in ckpt:
            own = getattr(model, name).state_dict()
            for k, v in ckpt[sd_key].items():
                if k in own and own[k].shape == v.shape:
                    own[k] = v
            getattr(model, name).load_state_dict(own, strict=False)

    # Load eval data
    data_path = os.path.join(PROJECT_DIR, 'data', 'gsm8k_sympy_annotations.jsonl')
    problems = []
    with open(data_path) as f:
        for line in f:
            if line.strip():
                problems.append(json.loads(line)['problem'])
                if len(problems) >= args.num_problems:
                    break

    print(f"Running {len(problems)} problems through {args.max_passes} passes...")
    model.eval()
    all_pages = []

    with torch.no_grad():
        for i in range(0, len(problems), 4):
            batch = problems[i:i+4]
            inputs = model.tokenizer(batch, return_tensors='pt', padding=True,
                                     truncation=True, max_length=192)
            input_ids = inputs['input_ids'].to(device)
            attention_mask = inputs['attention_mask'].to(device)

            state_pages = []
            mid_states_history = []
            for pass_num in range(args.max_passes):
                page, _scales, mid = model.thinking_pass(
                    input_ids, attention_mask, state_pages, pass_num,
                    prev_mid_states=mid_states_history if mid_states_history else None,
                )
                state_pages.append(page)
                mid_states_history.append(mid)

            all_pages.append(state_pages[-1].float().cpu())

    all_pages = torch.cat(all_pages, dim=0)  # (N, 64)
    N = all_pages.size(0)

    # Per-dimension statistics
    dim_std = all_pages.std(dim=0)  # (64,)
    dim_mean = all_pages.mean(dim=0)  # (64,)
    dim_range = all_pages.max(dim=0).values - all_pages.min(dim=0).values  # (64,)

    # Overall cosine similarity
    normed = F.normalize(all_pages, dim=-1)
    cos = normed @ normed.T
    off_diag = (cos.sum() - N) / max(N * (N - 1), 1)

    print(f"\n{'='*70}")
    print(f"Page Variance Diagnostic ({N} problems, {args.max_passes} passes)")
    print(f"{'='*70}")
    print(f"Overall page_cos: {off_diag:.4f}")
    print(f"Page norm: {all_pages.norm(dim=-1).mean():.2f}")
    print(f"\nPer-dimension std (higher = more variation):")

    # Print in 8-dim bands aligned with pi-harmonic encoding
    for band in range(8):
        start = band * 8
        end = start + 8
        band_std = dim_std[start:end]
        band_mean_std = band_std.mean().item()
        band_max_std = band_std.max().item()
        bar = '#' * int(band_mean_std * 50)
        print(f"  Band {band} (dims {start:2d}-{end-1:2d}): "
              f"mean_std={band_mean_std:.4f} max_std={band_max_std:.4f} {bar}")

    print(f"\nTop 10 most variable dimensions:")
    top_dims = dim_std.argsort(descending=True)[:10]
    for d in top_dims:
        print(f"  dim {d.item():2d}: std={dim_std[d]:.4f} mean={dim_mean[d]:.4f} "
              f"range={dim_range[d]:.4f}")

    print(f"\nBottom 10 least variable dimensions:")
    bot_dims = dim_std.argsort()[:10]
    for d in bot_dims:
        print(f"  dim {d.item():2d}: std={dim_std[d]:.4f} mean={dim_mean[d]:.4f} "
              f"range={dim_range[d]:.4f}")

    # Concentration: what fraction of total variance is in top 10 dims?
    total_var = (dim_std ** 2).sum()
    top10_var = (dim_std[dim_std.argsort(descending=True)[:10]] ** 2).sum()
    print(f"\nVariance concentration: top 10 dims hold {100*top10_var/total_var:.1f}% of total variance")

if __name__ == '__main__':
    main()
