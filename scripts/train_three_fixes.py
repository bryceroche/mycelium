"""
Three Fixes for the GSM8K Ceiling (v22.3).

Unified training script implementing:
  Fix 1: Gradient scaling per cycle — earlier cycles get amplified gradient
  Fix 2: Fresh data every epoch — procedural levels regenerate, GSM8K augments
  Fix 3: Multi-level curriculum — L4.5 -> L4.7 -> L4.9 -> L5

Usage:
  python scripts/train_three_fixes.py --level L4.5
  python scripts/train_three_fixes.py --level L4.7 --warm checkpoints/three_fixes_L45_best.pt
  python scripts/train_three_fixes.py --level L4.9 --warm checkpoints/three_fixes_L47_best.pt
  python scripts/train_three_fixes.py --level L5   --warm checkpoints/three_fixes_L49_best.pt
"""
import argparse
import re
import sys
import time
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, '/home/ubuntu/mycelium')

from scripts.train_dual_lora import (
    DualLoRAModel, PageConfidenceHead,
    DualAdditiveLoRAManager,
)
from scripts.train_dual_lora_gsm8k import warm_start_dual
from scripts.train_stepping_stones import extract_answer
from src.contrastive_page_loss import per_page_contrastive_loss


# ---------------------------------------------------------------------------
# Fix 1: Gradient scaling
# ---------------------------------------------------------------------------

def scale_gradient(tensor, scale):
    """Scale gradient during backward without affecting forward.

    Forward: returns tensor unchanged.
    Backward: gradient multiplied by *scale*.
    """
    return tensor + (scale - 1.0) * tensor.detach()


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

def make_dataset(level, num_samples, seed):
    """Build the right dataset for *level* with a given seed."""
    if level == 'L4.5':
        from scripts.datasets_L45_L47 import L45TwoStepWordDataset
        return L45TwoStepWordDataset(num_samples=num_samples, seed=seed)
    elif level == 'L4.7':
        from scripts.datasets_L45_L47 import L47ThreeStepWordDataset
        return L47ThreeStepWordDataset(num_samples=num_samples, seed=seed)
    elif level == 'L4.9':
        from scripts.datasets_L49_gsm8k import L49GSM8KEasyDataset
        return L49GSM8KEasyDataset()  # fixed dataset, seed ignored
    elif level == 'L5':
        from scripts.datasets_L49_gsm8k import GSM8KAugmentedDataset
        return GSM8KAugmentedDataset(epoch_seed=seed)
    elif level == 'L4':
        from scripts.train_dual_lora_L4 import L4TwoStepWordDataset
        return L4TwoStepWordDataset(num_samples=num_samples, seed=seed)
    else:
        raise ValueError(f"Unknown level: {level}")


def make_eval_dataset(level):
    """Build a fixed eval dataset for *level*."""
    if level == 'L4.5':
        from scripts.datasets_L45_L47 import L45TwoStepWordDataset
        return L45TwoStepWordDataset(num_samples=500, seed=99999)
    elif level == 'L4.7':
        from scripts.datasets_L45_L47 import L47ThreeStepWordDataset
        return L47ThreeStepWordDataset(num_samples=500, seed=99999)
    elif level == 'L4.9':
        from scripts.datasets_L49_gsm8k import L49GSM8KEasyDataset
        return L49GSM8KEasyDataset(split='test')
    elif level == 'L5':
        from scripts.datasets_L49_gsm8k import GSM8KAugmentedDataset
        # Eval uses the raw GSM8K test set (no augmentation)
        from scripts.train_dual_lora_gsm8k import GSM8KDataset
        return GSM8KDataset(split='test')
    elif level == 'L4':
        from scripts.train_dual_lora_L4 import L4TwoStepWordDataset
        return L4TwoStepWordDataset(num_samples=500, seed=99999)
    else:
        raise ValueError(f"Unknown level: {level}")


def is_procedural(level):
    """True if the level uses procedurally generated data."""
    return level in ('L4', 'L4.5', 'L4.7')


def is_gsm8k(level):
    """True if the level uses GSM8K-style answers (float-tolerant comparison)."""
    return level in ('L4.9', 'L5')


# ---------------------------------------------------------------------------
# Default warm-start checkpoint per level
# ---------------------------------------------------------------------------

DEFAULT_WARM = {
    'L4':   'checkpoints/dual_lora_L3_best.pt',
    'L4.5': 'checkpoints/dual_lora_L4_best.pt',
    'L4.7': 'checkpoints/three_fixes_L45_best.pt',
    'L4.9': 'checkpoints/three_fixes_L47_best.pt',
    'L5':   'checkpoints/three_fixes_L49_best.pt',
}


# ---------------------------------------------------------------------------
# Forward with gradient scaling (Fix 1)
# ---------------------------------------------------------------------------

def forward_train(model, problems, answers, finals_t, num_passes=5,
                  max_length=192, max_answer_length=256):
    """Forward pass with dual LoRA + gradient scaling per cycle."""
    device = model.transformer.device

    inputs = model.tokenizer(
        problems, return_tensors='pt', padding=True,
        truncation=True, max_length=max_length,
    )
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)

    answer_texts = [f" {a}" for a in answers]
    answer_inputs = model.tokenizer(
        answer_texts, return_tensors='pt', padding=True,
        truncation=True, max_length=max_answer_length,
        add_special_tokens=False,
    )
    answer_ids = answer_inputs['input_ids'].to(device)

    batch_size = input_ids.size(0)
    embed_layer = model.transformer.model.embed_tokens
    problem_embeds = embed_layer(input_ids)

    state_pages = []
    blend_history = []
    strategy = torch.zeros(batch_size, model.strategy_size, device=device)

    # Per-cycle gradient norms (logged after backward)
    page_norms = []

    for pass_num in range(num_passes):
        if pass_num == 0:
            outputs = model.transformer(
                inputs_embeds=problem_embeds, attention_mask=attention_mask,
                output_hidden_states=True,
            )
            hidden_states = list(outputs.hidden_states[1:])
            blend = torch.zeros(batch_size, 1, device=device)
        else:
            forward_mods, verify_mods, blend = model.hypernet(
                state_pages, strategy, pass_num=pass_num,
            )
            manager = DualAdditiveLoRAManager(model.transformer)
            manager.apply(forward_mods, verify_mods, blend)
            try:
                outputs = model.transformer(
                    inputs_embeds=problem_embeds, attention_mask=attention_mask,
                    output_hidden_states=True,
                )
                hidden_states = list(outputs.hidden_states[1:])
            finally:
                manager.remove()

        page_delta, strategy = model.compressor(hidden_states, pass_num)
        page = F.normalize(page_delta, dim=-1) * model.page_radius

        # --- FIX 1: Gradient scaling per cycle ---
        # Earlier cycles get amplified gradient to compensate for attenuation.
        # Cycle 0 = num_passes x, cycle 1 = (num_passes-1) x, ..., last = 1x
        # Cap at 4x to avoid destabilizing with many passes.
        grad_scale = min(float(num_passes - pass_num), 4.0)
        page = scale_gradient(page, grad_scale)

        state_pages.append(page)
        blend_history.append(blend)

    # ------- Answer loss (teacher-forced with dual LoRA) -------
    forward_mods, verify_mods, final_blend = model.hypernet(
        state_pages, strategy, pass_num=num_passes,
    )
    manager = DualAdditiveLoRAManager(model.transformer)
    manager.apply(forward_mods, verify_mods, final_blend)
    try:
        answer_embeds = embed_layer(answer_ids)
        full_embeds = torch.cat([problem_embeds, answer_embeds], dim=1)
        outputs = model.transformer(inputs_embeds=full_embeds, use_cache=False)
    finally:
        manager.remove()
    prompt_len = input_ids.size(1)
    logits = outputs.logits[:, prompt_len - 1:-1, :]
    answer_loss = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        answer_ids.reshape(-1),
        ignore_index=model.tokenizer.pad_token_id,
    )

    # ------- Confidence loss -------
    confidence = model.confidence_head(state_pages, blend_history)
    conf_target = torch.ones_like(confidence)
    conf_loss = F.binary_cross_entropy(confidence, conf_target)

    # ------- Contrastive loss -------
    c_loss = per_page_contrastive_loss(
        state_pages, finals_t, cross_target=0.7, within_target=0.3,
    )

    # ------- Diagnostics -------
    with torch.no_grad():
        last_page = state_pages[-1].float()
        normed = F.normalize(last_page, dim=-1)
        sim = normed @ normed.T
        B = sim.size(0)
        off_diag = sim - torch.eye(B, device=sim.device)
        page_cos_mean = off_diag.sum() / (B * (B - 1))
        blend_mean = final_blend.mean()

    return answer_loss, c_loss, conf_loss, page_cos_mean, blend_mean, confidence.mean()


# ---------------------------------------------------------------------------
# Evaluate
# ---------------------------------------------------------------------------

def evaluate(model, eval_dataset, device, num_passes=5, gsm8k_mode=False):
    """Evaluate accuracy via generation.

    gsm8k_mode=True uses float-tolerant comparison and longer max_new_tokens.
    """
    model.eval()
    correct = 0
    total = 0
    eval_batch = 4 if gsm8k_mode else 8
    max_length = 192 if gsm8k_mode else 128
    max_new_tokens = 150 if gsm8k_mode else 50

    with torch.no_grad():
        for i in range(0, len(eval_dataset), eval_batch):
            batch_samples = [
                eval_dataset[j]
                for j in range(i, min(i + eval_batch, len(eval_dataset)))
            ]
            problems = [s['problem'] for s in batch_samples]
            gold_answers = [s['final'] for s in batch_samples]

            inputs = model.tokenizer(
                problems, return_tensors='pt', padding=True,
                truncation=True, max_length=max_length,
            )
            input_ids = inputs['input_ids'].to(device)
            attention_mask = inputs['attention_mask'].to(device)
            batch_size = input_ids.size(0)

            state_pages = []
            blend_history = []
            strategy = torch.zeros(batch_size, model.strategy_size, device=device)

            for pass_num in range(num_passes):
                page, strategy, blend = model.thinking_pass(
                    input_ids, attention_mask, state_pages, strategy, pass_num,
                )
                state_pages.append(page)
                blend_history.append(blend)

            # Generate with dual LoRA
            forward_mods, verify_mods, blend = model.hypernet(
                state_pages, strategy, pass_num=len(state_pages),
            )
            manager = DualAdditiveLoRAManager(model.transformer)
            manager.apply(forward_mods, verify_mods, blend)
            try:
                generated = model.transformer.generate(
                    input_ids=input_ids, attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens, do_sample=False,
                    pad_token_id=model.tokenizer.pad_token_id,
                )
            finally:
                manager.remove()

            for j in range(batch_size):
                prompt_len = input_ids[j].size(0)
                gen_ids = generated[j][prompt_len:]
                gen_text = model.tokenizer.decode(
                    gen_ids, skip_special_tokens=True,
                ).strip()
                pred = extract_answer(gen_text)
                gold = gold_answers[j]

                if pred is not None and gold is not None:
                    if gsm8k_mode:
                        # Float-tolerant comparison
                        try:
                            pred_f = float(pred)
                            gold_f = float(gold)
                            if gold_f == int(gold_f):
                                if pred_f == gold_f:
                                    correct += 1
                            elif abs(pred_f - gold_f) < 0.01:
                                correct += 1
                        except (ValueError, TypeError):
                            pass
                    else:
                        if pred == gold:
                            correct += 1
                total += 1

    acc = 100.0 * correct / total if total > 0 else 0.0
    return acc


# ---------------------------------------------------------------------------
# Per-cycle gradient norm logging
# ---------------------------------------------------------------------------

def log_cycle_grad_norms(model, state_pages):
    """Log gradient norms on each page to verify Fix 1 is working.

    Call AFTER backward(). Reads .grad on pages that retained grad.
    Returns a list of norms (one per cycle).
    """
    norms = []
    for p, page in enumerate(state_pages):
        if page.grad is not None:
            norms.append(page.grad.norm().item())
        else:
            norms.append(0.0)
    return norms


# ---------------------------------------------------------------------------
# Collate function
# ---------------------------------------------------------------------------

def collate_fn(batch):
    return {
        'problem': [s['problem'] for s in batch],
        'answer': [s['answer'] for s in batch],
        'final': [s['final'] for s in batch],
    }


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(args):
    level = args.level
    warm_path = args.warm or DEFAULT_WARM.get(level)

    print("=" * 60)
    print(f"Three Fixes Training -- Level {level}")
    print("=" * 60)
    print(f"  num_passes   = {args.num_passes}")
    print(f"  batch_size   = {args.batch_size}")
    print(f"  epochs       = {args.epochs}")
    print(f"  patience     = {args.patience}")
    print(f"  lam          = {args.lam}")
    print(f"  lam_conf     = {args.lam_conf}")
    print(f"  num_train    = {args.num_train}")
    print(f"  warm         = {warm_path}")
    print(f"  procedural   = {is_procedural(level)}")
    print(f"  gsm8k_mode   = {is_gsm8k(level)}")
    print("=" * 60)

    device = torch.device('cuda')
    model = DualLoRAModel()
    model.compressor = model.compressor.to(device=device, dtype=torch.bfloat16)
    model.hypernet = model.hypernet.to(device=device, dtype=torch.bfloat16)
    model.confidence_head = model.confidence_head.to(device)
    model.probe_head = model.probe_head.to(device)

    if warm_path:
        print(f"\nWarm-starting from {warm_path}")
        warm_start_dual(model, warm_path)

    # Fixed eval dataset (same every epoch for fair comparison)
    eval_dataset = make_eval_dataset(level)
    print(f"\nEval dataset: {len(eval_dataset)} problems")

    # Sequence length settings per level
    if is_gsm8k(level):
        max_length = 192
        max_answer_length = 256
    else:
        max_length = 128
        max_answer_length = 128

    # Optimizer
    optimizer = torch.optim.AdamW([
        {'params': list(model.compressor.parameters()), 'lr': 5e-5},
        # Forward templates
        {'params': (list(model.hypernet.A_forward.parameters())
                    + list(model.hypernet.B_forward.parameters())), 'lr': 5e-4},
        # Verify templates
        {'params': (list(model.hypernet.A_verify.parameters())
                    + list(model.hypernet.B_verify.parameters())), 'lr': 5e-4},
        # Shared hypernetwork
        {'params': (
            list(model.hypernet.page_project.parameters())
            + [model.hypernet.page_query]
            + list(model.hypernet.page_attn.parameters())
            + list(model.hypernet.page_norm.parameters())
            + list(model.hypernet.combine.parameters())
            + list(model.hypernet.pass_embed.parameters())
        ), 'lr': 5e-4},
        # Confidence head
        {'params': list(model.confidence_head.parameters()), 'lr': 1e-3},
    ])
    trainable = (
        list(model.compressor.parameters())
        + list(model.hypernet.parameters())
        + list(model.confidence_head.parameters())
    )
    print(f"Trainable params: {sum(p.numel() for p in trainable):,}")
    print(f"GPU mem after init: {torch.cuda.memory_allocated()/1e9:.2f} GB")

    # Baseline
    base_acc = evaluate(
        model, eval_dataset, device,
        num_passes=args.num_passes, gsm8k_mode=is_gsm8k(level),
    )
    print(f"\nBaseline accuracy (before training): {base_acc:.1f}%\n")

    ckpt_name = f"checkpoints/three_fixes_{level.replace('.', '')}_best.pt"
    best = 0.0
    patience_counter = 0

    for epoch in range(args.epochs):
        model.train()
        t0 = time.time()

        # --- FIX 2: Fresh data every epoch ---
        epoch_seed = epoch * 1000 + 42
        if is_procedural(level):
            # Procedural levels: generate completely new data each epoch
            train_dataset = make_dataset(level, args.num_train, seed=epoch_seed)
            print(f"  [epoch {epoch+1}] Generated {len(train_dataset)} fresh problems (seed={epoch_seed})")
        elif level == 'L5':
            # Full GSM8K: augmented with new seed each epoch
            train_dataset = make_dataset(level, args.num_train, seed=epoch_seed)
            print(f"  [epoch {epoch+1}] Augmented GSM8K dataset (seed={epoch_seed})")
        else:
            # L4.9: fixed GSM8K easy subset (can't procedurally generate)
            if epoch == 0:
                train_dataset = make_dataset(level, args.num_train, seed=epoch_seed)
                print(f"  [epoch {epoch+1}] Loaded {len(train_dataset)} GSM8K easy problems (fixed)")
            # Reuse train_dataset from epoch 0 for L4.9

        train_loader = DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True,
            collate_fn=collate_fn,
        )

        ep_ans = ep_ctr = ep_conf = ep_cos = ep_blend = ep_confval = 0.0
        cycle_grad_norms_sum = [0.0] * args.num_passes
        nb = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False):
            problems = batch['problem']
            answers = batch['answer']

            # Build finals tensor for contrastive loss
            finals_raw = batch['final']
            finals_list = []
            for f in finals_raw:
                try:
                    finals_list.append(int(f))
                except (ValueError, TypeError):
                    finals_list.append(0)
            finals_t = torch.tensor(finals_list, dtype=torch.long, device=device)

            optimizer.zero_grad()
            ans_loss, c_loss, conf_loss, page_cos, blend_mean, conf_mean = forward_train(
                model, problems, answers, finals_t,
                num_passes=args.num_passes,
                max_length=max_length,
                max_answer_length=max_answer_length,
            )
            total_loss = ans_loss + args.lam * c_loss + args.lam_conf * conf_loss
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()

            ep_ans += ans_loss.item()
            ep_ctr += c_loss.item()
            ep_conf += conf_loss.item()
            ep_cos += page_cos.item()
            ep_blend += blend_mean.item()
            ep_confval += conf_mean.item()
            nb += 1

        elapsed = time.time() - t0

        # Eval
        acc = evaluate(
            model, eval_dataset, device,
            num_passes=args.num_passes, gsm8k_mode=is_gsm8k(level),
        )

        if acc > best:
            best = acc
            patience_counter = 0
            torch.save({
                'epoch': epoch + 1,
                'compressor': model.compressor.state_dict(),
                'hypernet': model.hypernet.state_dict(),
                'confidence_head': model.confidence_head.state_dict(),
                'accuracy': acc,
                'level': level,
            }, ckpt_name)
            print(f"  -> saved checkpoint {ckpt_name} (acc={acc:.1f}%)")
        else:
            patience_counter += 1

        print(
            f"Epoch {epoch+1}: ans={ep_ans/nb:.4f} contr={ep_ctr/nb:.4f} "
            f"conf={ep_conf/nb:.4f} page_cos={ep_cos/nb:.4f} "
            f"blend={ep_blend/nb:.3f} conf_val={ep_confval/nb:.3f} | "
            f"Acc={acc:.1f}% best={best:.1f}% base={base_acc:.1f}% "
            f"[{elapsed:.0f}s, {nb/elapsed:.1f} it/s]"
        )

        if patience_counter >= args.patience:
            print(f"\nEarly stopping: no improvement for {args.patience} epochs")
            break

    print(f"\n{'='*60}")
    print(f"Level {level} final: {best:.1f}% (baseline {base_acc:.1f}%)")
    print(f"Checkpoint: {ckpt_name}")
    print(f"{'='*60}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    p = argparse.ArgumentParser(
        description='Three-fixes curriculum training for GSM8K',
    )
    p.add_argument(
        '--level', type=str, required=True,
        choices=['L4', 'L4.5', 'L4.7', 'L4.9', 'L5'],
        help='Curriculum level to train',
    )
    p.add_argument('--warm', type=str, default=None,
                   help='Checkpoint to warm-start from (default per level)')
    p.add_argument('--epochs', type=int, default=12)
    p.add_argument('--batch_size', type=int, default=None,
                   help='Batch size (default: 16 procedural, 8 GSM8K)')
    p.add_argument('--num_passes', type=int, default=5)
    p.add_argument('--patience', type=int, default=3)
    p.add_argument('--lam', type=float, default=0.02,
                   help='Contrastive loss weight')
    p.add_argument('--lam_conf', type=float, default=0.1,
                   help='Confidence loss weight')
    p.add_argument('--num_train', type=int, default=20000,
                   help='Number of training problems per epoch (procedural)')

    args = p.parse_args()

    # Default batch size per level type
    if args.batch_size is None:
        args.batch_size = 16 if is_procedural(args.level) else 8

    train(args)
