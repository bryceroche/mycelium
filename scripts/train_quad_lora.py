"""
QuadLoRA Training (v23).

Four-mode LoRA (Parse, Compute, Verify, Answer) with 4-way softmax blend.
Each thinking pass blends four cognitive modes. The blend trajectory is
learned, not hardcoded — the model discovers when to parse, compute, verify,
and prepare its answer.

Additionally trains an AnswerHead on the last page for direct digit prediction,
bypassing generation + regex extraction.

Usage:
  python scripts/train_quad_lora.py --level L4.5
  python scripts/train_quad_lora.py --level L5 --warm checkpoints/quad_lora_L49_best.pt
  python scripts/train_quad_lora.py --level L4 --warm checkpoints/dual_lora_L3_best.pt
"""
import argparse
import re
import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, '/home/ubuntu/mycelium')

from scripts.quad_lora import (
    QuadLoRAModel,
    QuadAdditiveLoRAManager,
    AnswerHead,
    answer_head_loss,
    warm_start_quad_from_dual,
)
from scripts.train_stepping_stones import extract_answer
from src.contrastive_page_loss import per_page_contrastive_loss


# ---------------------------------------------------------------------------
# Gradient scaling (Fix 1 from v22.3)
# ---------------------------------------------------------------------------

def scale_gradient(tensor, scale):
    """Scale gradient during backward without affecting forward."""
    return tensor + (scale - 1.0) * tensor.detach()


# ---------------------------------------------------------------------------
# Dataset helpers (reused from train_three_fixes.py)
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
        return L49GSM8KEasyDataset()
    elif level == 'L5':
        from scripts.datasets_L49_gsm8k import GSM8KAugmentedDataset
        return GSM8KAugmentedDataset(epoch_seed=seed)
    elif level == 'L4':
        from scripts.train_dual_lora_L4 import L4TwoStepWordDataset
        return L4TwoStepWordDataset(num_samples=num_samples, seed=seed)
    elif level == 'L3':
        from scripts.train_stepping_stones_L3 import L3NamedQtyDataset
        return L3NamedQtyDataset(num_samples=num_samples, seed=seed)
    else:
        raise ValueError(f"Unknown level: {level}")


def make_eval_dataset(level, num_samples=200):
    """Build a fixed eval dataset for *level*."""
    if level == 'L4.5':
        from scripts.datasets_L45_L47 import L45TwoStepWordDataset
        return L45TwoStepWordDataset(num_samples=num_samples, seed=99999)
    elif level == 'L4.7':
        from scripts.datasets_L45_L47 import L47ThreeStepWordDataset
        return L47ThreeStepWordDataset(num_samples=num_samples, seed=99999)
    elif level == 'L4.9':
        from scripts.datasets_L49_gsm8k import L49GSM8KEasyDataset
        return L49GSM8KEasyDataset(split='test')
    elif level == 'L5':
        from scripts.train_dual_lora_gsm8k import GSM8KDataset
        return GSM8KDataset(split='test')
    elif level == 'L4':
        from scripts.train_dual_lora_L4 import L4TwoStepWordDataset
        return L4TwoStepWordDataset(num_samples=num_samples, seed=99999)
    elif level == 'L3':
        from scripts.train_stepping_stones_L3 import L3NamedQtyDataset
        return L3NamedQtyDataset(num_samples=num_samples, seed=123)
    else:
        raise ValueError(f"Unknown level: {level}")


def is_procedural(level):
    return level in ('L3', 'L4', 'L4.5', 'L4.7')


def is_gsm8k(level):
    return level in ('L4.9', 'L5')


# ---------------------------------------------------------------------------
# Default warm-start paths
# ---------------------------------------------------------------------------

DEFAULT_WARM = {
    'L3':   'checkpoints/dual_lora_L3_best.pt',    # dual -> quad upgrade
    'L4':   'checkpoints/quad_lora_L3_best.pt',
    'L4.5': 'checkpoints/quad_lora_L4_best.pt',
    'L4.7': 'checkpoints/quad_lora_L45_best.pt',
    'L4.9': 'checkpoints/quad_lora_L47_best.pt',
    'L5':   'checkpoints/quad_lora_L49_best.pt',
}

MODE_NAMES = ['parse', 'compute', 'verify', 'answer']


# ---------------------------------------------------------------------------
# Forward with QuadLoRA + gradient scaling + answer head
# ---------------------------------------------------------------------------

def forward_train(model, answer_head, problems, answers, finals_t,
                  num_passes=5, max_length=192, max_answer_length=256):
    """Forward pass with quad LoRA, gradient scaling, and answer head."""
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
    blend_history = []   # list of (B, 4) tensors
    strategy = torch.zeros(batch_size, model.strategy_size, device=device)

    for pass_num in range(num_passes):
        if pass_num == 0:
            # First pass: no LoRA (no pages yet)
            outputs = model.transformer(
                inputs_embeds=problem_embeds, attention_mask=attention_mask,
                output_hidden_states=True,
            )
            hidden_states = list(outputs.hidden_states[1:])
            blend = torch.zeros(batch_size, 4, device=device)
        else:
            # Quad LoRA from pages + strategy
            quad_mods, blend = model.hypernet(
                state_pages, strategy, pass_num=pass_num,
            )
            manager = QuadAdditiveLoRAManager(model.transformer)
            manager.apply(quad_mods, blend)
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

        # Gradient scaling per cycle (capped at 4x)
        grad_scale = min(float(num_passes - pass_num), 4.0)
        page = scale_gradient(page, grad_scale)

        state_pages.append(page)
        blend_history.append(blend)

    # ------- Answer loss (teacher-forced with quad LoRA) -------
    quad_mods, final_blend = model.hypernet(
        state_pages, strategy, pass_num=num_passes,
    )
    manager = QuadAdditiveLoRAManager(model.transformer)
    manager.apply(quad_mods, final_blend)
    try:
        answer_embeds = embed_layer(answer_ids)
        full_embeds = torch.cat([problem_embeds, answer_embeds], dim=1)
        outputs = model.transformer(inputs_embeds=full_embeds, use_cache=False)
    finally:
        manager.remove()
    prompt_len = input_ids.size(1)
    logits = outputs.logits[:, prompt_len - 1:-1, :]
    ans_loss = F.cross_entropy(
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

    # ------- Answer head loss (digit prediction from last page) -------
    last_page = state_pages[-1].float()
    ah_loss = answer_head_loss(answer_head, last_page, finals_t)

    # ------- Diagnostics -------
    with torch.no_grad():
        normed = F.normalize(last_page, dim=-1)
        sim = normed @ normed.T
        B = sim.size(0)
        off_diag = sim - torch.eye(B, device=sim.device)
        page_cos_mean = off_diag.sum() / (B * (B - 1))
        # 4-way blend means (from final generation blend)
        blend_means = final_blend.mean(dim=0)  # (4,)

    return (ans_loss, c_loss, conf_loss, ah_loss,
            page_cos_mean, blend_means, confidence.mean())


# ---------------------------------------------------------------------------
# Evaluate (generation-based + answer head)
# ---------------------------------------------------------------------------

def evaluate(model, answer_head, eval_dataset, device, num_passes=5,
             gsm8k_mode=False):
    """Evaluate via generation AND answer head digit prediction."""
    model.eval()
    answer_head.eval()
    gen_correct = 0
    head_correct = 0
    total = 0
    eval_batch = 4 if gsm8k_mode else 16
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

            # --- Generation-based eval ---
            quad_mods, blend = model.hypernet(
                state_pages, strategy, pass_num=len(state_pages),
            )
            manager = QuadAdditiveLoRAManager(model.transformer)
            manager.apply(quad_mods, blend)
            try:
                generated = model.transformer.generate(
                    input_ids=input_ids, attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens, do_sample=False,
                    pad_token_id=model.tokenizer.pad_token_id,
                )
            finally:
                manager.remove()

            # --- Answer head eval ---
            last_page = state_pages[-1].float()
            head_preds = answer_head.decode(last_page)  # (B,) tensor of ints

            for j in range(batch_size):
                gold = gold_answers[j]

                # Generation accuracy
                prompt_len = input_ids[j].size(0)
                gen_ids = generated[j][prompt_len:]
                gen_text = model.tokenizer.decode(
                    gen_ids, skip_special_tokens=True,
                ).strip()
                pred = extract_answer(gen_text)

                if pred is not None and gold is not None:
                    if gsm8k_mode:
                        try:
                            pred_f = float(pred)
                            gold_f = float(gold)
                            if gold_f == int(gold_f):
                                if pred_f == gold_f:
                                    gen_correct += 1
                            elif abs(pred_f - gold_f) < 0.01:
                                gen_correct += 1
                        except (ValueError, TypeError):
                            pass
                    else:
                        if pred == gold:
                            gen_correct += 1

                # Answer head accuracy
                if gold is not None:
                    try:
                        head_pred = head_preds[j].item()
                        gold_int = int(gold)
                        if head_pred == gold_int:
                            head_correct += 1
                    except (ValueError, TypeError):
                        pass

                total += 1

    gen_acc = 100.0 * gen_correct / total if total > 0 else 0.0
    head_acc = 100.0 * head_correct / total if total > 0 else 0.0
    return gen_acc, head_acc


# ---------------------------------------------------------------------------
# Per-mode gradient norm logging
# ---------------------------------------------------------------------------

def log_mode_grad_norms(model):
    """Log gradient norms for each of the 4 template sets.

    Returns dict: {mode_name: float} — average grad norm across A+B params.
    """
    norms = {}
    for mode in MODE_NAMES:
        A_list, B_list = model.hypernet._template_lists[mode]
        A_params = list(A_list.parameters())
        B_params = list(B_list.parameters())
        total_norm = 0.0
        count = 0
        for p in A_params + B_params:
            if p.grad is not None:
                total_norm += p.grad.norm().item()
                count += 1
        norms[mode] = total_norm / max(count, 1)
    return norms


# ---------------------------------------------------------------------------
# Collate
# ---------------------------------------------------------------------------

def collate_fn(batch):
    return {
        'problem': [s['problem'] for s in batch],
        'answer': [s['answer'] for s in batch],
        'final': [s['final'] for s in batch],
    }


# ---------------------------------------------------------------------------
# Warm-start logic
# ---------------------------------------------------------------------------

def try_warm_start(model, answer_head, warm_path):
    """Try quad checkpoint first; fall back to dual checkpoint with upgrade."""
    ckpt = torch.load(warm_path, map_location='cpu', weights_only=True)

    # Detect checkpoint type
    if 'answer_head' in ckpt:
        # Quad checkpoint — load directly
        print(f"  Loading quad checkpoint from {warm_path}")

        # Compressor
        own = model.compressor.state_dict()
        loaded = 0
        for k, v in ckpt['compressor'].items():
            if k in own and own[k].shape == v.shape:
                own[k] = v
                loaded += 1
        model.compressor.load_state_dict(own, strict=False)
        print(f"  compressor: loaded {loaded}/{len(own)}")

        # Hypernet
        if 'hypernet' in ckpt:
            own_h = model.hypernet.state_dict()
            loaded_h = 0
            for k, v in ckpt['hypernet'].items():
                if k in own_h and own_h[k].shape == v.shape:
                    own_h[k] = v
                    loaded_h += 1
            model.hypernet.load_state_dict(own_h, strict=False)
            print(f"  hypernet: loaded {loaded_h}/{len(own_h)}")

        # Confidence head
        if 'confidence_head' in ckpt:
            own_c = model.confidence_head.state_dict()
            loaded_c = 0
            for k, v in ckpt['confidence_head'].items():
                if k in own_c and own_c[k].shape == v.shape:
                    own_c[k] = v
                    loaded_c += 1
            model.confidence_head.load_state_dict(own_c, strict=False)
            print(f"  confidence_head: loaded {loaded_c}/{len(own_c)}")

        # Answer head
        own_a = answer_head.state_dict()
        loaded_a = 0
        for k, v in ckpt['answer_head'].items():
            if k in own_a and own_a[k].shape == v.shape:
                own_a[k] = v
                loaded_a += 1
        answer_head.load_state_dict(own_a, strict=False)
        print(f"  answer_head: loaded {loaded_a}/{len(own_a)}")

    else:
        # Dual checkpoint — upgrade to quad
        print(f"  Upgrading dual checkpoint to quad: {warm_path}")
        warm_start_quad_from_dual(model, ckpt)
        print(f"  answer_head: fresh init (no dual equivalent)")


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(args):
    level = args.level
    warm_path = args.warm or DEFAULT_WARM.get(level)

    print("=" * 60)
    print(f"QuadLoRA Training -- Level {level}")
    print("=" * 60)
    print(f"  num_passes     = {args.num_passes}")
    print(f"  batch_size     = {args.batch_size}")
    print(f"  epochs         = {args.epochs}")
    print(f"  patience       = {args.patience}")
    print(f"  lam            = {args.lam}")
    print(f"  lam_conf       = {args.lam_conf}")
    print(f"  lam_answer     = {args.lam_answer}")
    print(f"  num_train      = {args.num_train}")
    print(f"  warm           = {warm_path}")
    print(f"  procedural     = {is_procedural(level)}")
    print(f"  gsm8k_mode     = {is_gsm8k(level)}")
    print("=" * 60)

    device = torch.device('cuda')
    model = QuadLoRAModel()
    model.compressor = model.compressor.to(device=device, dtype=torch.bfloat16)
    model.hypernet = model.hypernet.to(device=device, dtype=torch.bfloat16)
    model.confidence_head = model.confidence_head.to(device)

    # Answer head — small, kept in float32 for digit classification stability
    answer_head = AnswerHead(page_size=model.page_size).to(device)

    if warm_path:
        print(f"\nWarm-starting from {warm_path}")
        try_warm_start(model, answer_head, warm_path)

    # Fixed eval dataset
    eval_dataset = make_eval_dataset(level, num_samples=args.eval_size)
    print(f"\nEval dataset: {len(eval_dataset)} problems")

    # Sequence length settings
    if is_gsm8k(level):
        max_length = 192
        max_answer_length = 256
    else:
        max_length = 128
        max_answer_length = 128

    # ------- Optimizer: four template sets + shared + heads -------
    template_param_groups = []
    for mode in MODE_NAMES:
        A_list, B_list = model.hypernet._template_lists[mode]
        template_param_groups.append({
            'params': list(A_list.parameters()) + list(B_list.parameters()),
            'lr': 5e-4,
        })

    # Collect shared hypernetwork params (everything that isn't templates)
    template_param_ids = set()
    for mode in MODE_NAMES:
        A_list, B_list = model.hypernet._template_lists[mode]
        for p in A_list.parameters():
            template_param_ids.add(id(p))
        for p in B_list.parameters():
            template_param_ids.add(id(p))
    shared_hypernet_params = [
        p for p in model.hypernet.parameters()
        if id(p) not in template_param_ids
    ]

    optimizer = torch.optim.AdamW(
        [
            {'params': list(model.compressor.parameters()), 'lr': 5e-5},
        ]
        + template_param_groups
        + [
            {'params': shared_hypernet_params, 'lr': 5e-4},
            {'params': list(model.confidence_head.parameters()), 'lr': 1e-3},
            {'params': list(answer_head.parameters()), 'lr': 1e-3},
        ]
    )

    trainable = (
        list(model.compressor.parameters())
        + list(model.hypernet.parameters())
        + list(model.confidence_head.parameters())
        + list(answer_head.parameters())
    )
    print(f"Trainable params: {sum(p.numel() for p in trainable):,}")
    for mode in MODE_NAMES:
        n = sum(
            p.numel()
            for p in list(model.hypernet._template_lists[mode][0].parameters())
            + list(model.hypernet._template_lists[mode][1].parameters())
        )
        print(f"  {mode:8s} templates: {n:,}")
    print(f"  answer_head: {sum(p.numel() for p in answer_head.parameters()):,}")
    print(f"  confidence_head: {sum(p.numel() for p in model.confidence_head.parameters()):,}")
    print(f"GPU mem after init: {torch.cuda.memory_allocated()/1e9:.2f} GB")

    # Baseline
    base_gen, base_head = evaluate(
        model, answer_head, eval_dataset, device,
        num_passes=args.num_passes, gsm8k_mode=is_gsm8k(level),
    )
    print(f"\nBaseline: gen={base_gen:.1f}% head={base_head:.1f}%\n")

    ckpt_name = f"checkpoints/quad_lora_{level.replace('.', '')}_best.pt"
    best = 0.0
    best_head = 0.0
    patience_counter = 0

    for epoch in range(args.epochs):
        model.train()
        answer_head.train()
        t0 = time.time()

        # Fresh data per epoch (Fix 2)
        epoch_seed = epoch * 1000 + 42
        if is_procedural(level):
            train_dataset = make_dataset(level, args.num_train, seed=epoch_seed)
            print(f"  [epoch {epoch+1}] Generated {len(train_dataset)} fresh problems (seed={epoch_seed})")
        elif level == 'L5':
            train_dataset = make_dataset(level, args.num_train, seed=epoch_seed)
            print(f"  [epoch {epoch+1}] Augmented GSM8K (seed={epoch_seed})")
        else:
            if epoch == 0:
                train_dataset = make_dataset(level, args.num_train, seed=epoch_seed)
                print(f"  [epoch {epoch+1}] Loaded {len(train_dataset)} problems (fixed)")

        train_loader = DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True,
            collate_fn=collate_fn,
        )

        ep_ans = ep_ctr = ep_conf = ep_head = ep_cos = 0.0
        ep_blend = torch.zeros(4)
        nb = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False):
            problems = batch['problem']
            answers = batch['answer']

            finals_raw = batch['final']
            finals_list = []
            for f in finals_raw:
                try:
                    finals_list.append(int(f))
                except (ValueError, TypeError):
                    finals_list.append(0)
            finals_t = torch.tensor(finals_list, dtype=torch.long, device=device)

            optimizer.zero_grad()
            (ans_loss, c_loss, conf_loss, ah_loss,
             page_cos, blend_means, conf_mean) = forward_train(
                model, answer_head, problems, answers, finals_t,
                num_passes=args.num_passes,
                max_length=max_length,
                max_answer_length=max_answer_length,
            )

            total_loss = (ans_loss
                          + args.lam * c_loss
                          + args.lam_conf * conf_loss
                          + args.lam_answer * ah_loss)
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()

            ep_ans += ans_loss.item()
            ep_ctr += c_loss.item()
            ep_conf += conf_loss.item()
            ep_head += ah_loss.item()
            ep_cos += page_cos.item()
            ep_blend += blend_means.detach().cpu()
            nb += 1

        elapsed = time.time() - t0

        # Per-mode gradient norms (from last batch)
        mode_norms = log_mode_grad_norms(model)

        # Eval
        gen_acc, head_acc = evaluate(
            model, answer_head, eval_dataset, device,
            num_passes=args.num_passes, gsm8k_mode=is_gsm8k(level),
        )

        # Track both metrics — early stop only when BOTH plateau
        improved = False
        if gen_acc > best:
            best = gen_acc
            improved = True
        if head_acc > best_head:
            best_head = head_acc
            improved = True

        if improved:
            patience_counter = 0
            torch.save({
                'epoch': epoch + 1,
                'compressor': model.compressor.state_dict(),
                'hypernet': model.hypernet.state_dict(),
                'confidence_head': model.confidence_head.state_dict(),
                'answer_head': answer_head.state_dict(),
                'accuracy': gen_acc,
                'head_accuracy': head_acc,
                'level': level,
            }, ckpt_name)
            print(f"  -> saved checkpoint {ckpt_name} (gen={gen_acc:.1f}% head={head_acc:.1f}%)")
        else:
            patience_counter += 1

        # Blend averages
        bm = ep_blend / nb
        P, C, V, A = bm[0].item(), bm[1].item(), bm[2].item(), bm[3].item()

        print(
            f"Epoch {epoch+1}: "
            f"ans={ep_ans/nb:.4f} contr={ep_ctr/nb:.2f} "
            f"conf={ep_conf/nb:.2f} head={ep_head/nb:.2f} "
            f"page_cos={ep_cos/nb:.2f} "
            f"P={P:.2f} C={C:.2f} V={V:.2f} A={A:.2f} | "
            f"Acc={gen_acc:.1f}% head={head_acc:.1f}% "
            f"best={best:.1f}% [{elapsed:.0f}s]"
        )
        # Per-mode gradient norms
        gn = mode_norms
        print(
            f"  grad norms: "
            f"parse={gn['parse']:.4f} compute={gn['compute']:.4f} "
            f"verify={gn['verify']:.4f} answer={gn['answer']:.4f}"
        )

        if patience_counter >= args.patience:
            print(f"\nEarly stopping: no improvement for {args.patience} epochs")
            break

    print(f"\n{'='*60}")
    print(f"Level {level} final: gen={best:.1f}% head={best_head:.1f}% (baseline gen={base_gen:.1f}% head={base_head:.1f}%)")
    print(f"Checkpoint: {ckpt_name}")
    print(f"{'='*60}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    p = argparse.ArgumentParser(
        description='QuadLoRA (4-mode) training with answer head',
    )
    p.add_argument(
        '--level', type=str, required=True,
        choices=['L3', 'L4', 'L4.5', 'L4.7', 'L4.9', 'L5'],
        help='Curriculum level to train',
    )
    p.add_argument('--warm', type=str, default=None,
                   help='Checkpoint to warm-start from (default per level)')
    p.add_argument('--epochs', type=int, default=12)
    p.add_argument('--batch_size', type=int, default=None,
                   help='Batch size (default: 16 procedural, 8 GSM8K)')
    p.add_argument('--num_passes', type=int, default=5)
    p.add_argument('--patience', type=int, default=3)
    p.add_argument('--lam', type=float, default=0.05,
                   help='Contrastive loss weight')
    p.add_argument('--lam_conf', type=float, default=0.1,
                   help='Confidence loss weight')
    p.add_argument('--lam_answer', type=float, default=0.3,
                   help='Answer head loss weight')
    p.add_argument('--num_train', type=int, default=20000,
                   help='Number of training problems per epoch (procedural)')
    p.add_argument('--eval_size', type=int, default=200,
                   help='Number of eval problems (200 quick, 500 thorough)')

    args = p.parse_args()

    if args.batch_size is None:
        args.batch_size = 32 if is_procedural(args.level) else 8

    train(args)
