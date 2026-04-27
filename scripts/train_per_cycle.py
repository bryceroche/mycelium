"""
Per-Cycle Generation-Only Training (v25.0).

Generation-only training: no answer head. Each cycle generates intermediate
results as text, and answers are extracted via regex (#### marker or last number).

Key differences from v24.9:
  - No answer head -- generation loss is the only training signal
  - Eval uses greedy generation + regex extraction
  - JSONL data format with cycle_targets per problem
  - Confidence-based early stopping at inference
  - Variable-length cycle targets with masking

Data format (JSONL):
  {"problem": "...", "cycle_targets": [48, 24, 72], "final_answer": 72, "num_steps": 3}

Usage:
  python scripts/train_per_cycle.py --level L3 --data_dir data/per_cycle/ --epochs 8 --num_passes 3
  python scripts/train_per_cycle.py --level L4 --warm_from checkpoints/per_cycle_L3_best.pt --epochs 6 --num_passes 3
"""
import argparse
import json
import os
import random
import re
import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from scripts.atom_lora import (
    AtomLoRAModel,
    AtomAdditiveLoRAManager,
    warm_start_atom_from_checkpoint,
    encode_text_context,
    scale_diversity_loss,
    DifferentiableBypass,
)
from src.contrastive_page_loss import per_page_contrastive_loss


# ---------------------------------------------------------------------------
# Answer extraction from generated text
# ---------------------------------------------------------------------------

def extract_answer_from_text(text):
    """Extract number after #### marker."""
    match = re.search(r'####\s*([-]?\d+)', text)
    if match:
        return int(match.group(1))
    # Fallback: last number in text
    nums = re.findall(r'[-]?\d+', text)
    return int(nums[-1]) if nums else None


# ---------------------------------------------------------------------------
# Gradient scaling (Fix 1 from v22.3)
# ---------------------------------------------------------------------------

def scale_gradient(tensor, scale):
    """Scale gradient during backward without affecting forward."""
    return tensor + (scale - 1.0) * tensor.detach()


# ---------------------------------------------------------------------------
# Generation target dropout — force computation instead of copying
# ---------------------------------------------------------------------------

def dropout_gen_target(gen_text, target_number, dropout_prob=0.3, rng=None):
    """Randomly mask equation results so the model must compute, not copy.

    With probability `dropout_prob`, activates masking on this example.
    When active, each equation result (the Z in "X op Y = Z") independently
    has a 50% chance of being replaced with "___".

    The #### marker is NEVER masked — it's the extraction target.

    Args:
        gen_text: generation target string (e.g. "She has 5 + 3 = 8 apples. #### 8")
        target_number: the final answer (unused, reserved for safety checks)
        dropout_prob: probability that any masking happens at all (default 0.3)
        rng: random.Random instance (created fresh if None)

    Returns:
        gen_text with some equation results replaced by "___", or unchanged.
    """
    if rng is None:
        rng = random.Random()

    # With probability (1 - dropout_prob), return unchanged
    if rng.random() >= dropout_prob:
        return gen_text

    # Split off the #### marker so we never touch it
    marker_pattern = r'(####\s*[-]?\d+)'
    marker_match = re.search(marker_pattern, gen_text)
    if marker_match:
        body = gen_text[:marker_match.start()]
        tail = gen_text[marker_match.start():]
    else:
        body = gen_text
        tail = ''

    # Mask equation results in the body only
    eq_pattern = r'(\d+)\s*([+\-*/×÷])\s*(\d+)\s*(=)\s*(\d+)'

    def _mask_result(m):
        if rng.random() < 0.5:
            return f'{m.group(1)} {m.group(2)} {m.group(3)} {m.group(4)} ___'
        return m.group(0)

    body = re.sub(eq_pattern, _mask_result, body)

    return body + tail


# ---------------------------------------------------------------------------
# Number augmentation (anti-memorization)
# ---------------------------------------------------------------------------

def extract_base_numbers(question):
    """Find all whole numbers in question text, return list of (match, int)."""
    return [(m.group(), int(m.group())) for m in re.finditer(r'\b(\d+)\b', question)]


def replace_numbers_in_text(text, number_map):
    """Replace old->new numbers in text.

    Sorts replacements by length descending so '160' is replaced before '16'.
    """
    # Sort by length of old string descending, then by value descending
    sorted_pairs = sorted(number_map.items(), key=lambda kv: (-len(kv[0]), kv[0]))
    for old_str, new_str in sorted_pairs:
        text = text.replace(old_str, new_str)
    return text


def simple_augment(problem, rng):
    """Randomize all numbers in a problem by a shared scale factor.

    Args:
        problem: dict with 'question', 'cycle_targets', 'cycle_gen_targets', etc.
        rng: random.Random instance for reproducibility.

    Returns:
        New dict with scaled numbers (original is not mutated).
    """
    question = problem.get('question', problem.get('problem', ''))
    matches = extract_base_numbers(question)
    if not matches:
        return problem

    # Pick a single scale factor so ratios are preserved
    scale = rng.uniform(0.5, 2.0)

    # Build old_str -> new_str map for the question
    number_map = {}
    for match_str, match_int in matches:
        new_val = max(1, round(match_int * scale))
        number_map[match_str] = str(new_val)

    new_question = replace_numbers_in_text(question, number_map)

    # Scale cycle_targets by the same factor
    old_targets = problem.get('cycle_targets', [])
    new_targets = [max(1, round(t * scale)) for t in old_targets]

    # Scale cycle_gen_targets: replace numbers in text AND update #### markers
    old_gen_targets = problem.get('cycle_gen_targets', [])
    new_gen_targets = []
    for i, gt in enumerate(old_gen_targets):
        new_text = replace_numbers_in_text(gt, number_map)
        # Also replace any previously-computed target numbers that appear in
        # later gen targets (chain of intermediate results)
        for j, ot in enumerate(old_targets):
            old_t_str = str(ot)
            new_t_str = str(new_targets[j])
            if old_t_str != new_t_str:
                new_text = new_text.replace(old_t_str, new_t_str)
        # Update #### marker to match the new target for this cycle
        if i < len(new_targets):
            new_text = re.sub(r'####\s*[-]?\d+', f'#### {new_targets[i]}', new_text)
        new_gen_targets.append(new_text)

    # Build augmented problem (don't mutate original)
    augmented = dict(problem)
    # Write back using whichever key the data uses
    q_key = 'question' if 'question' in problem else 'problem'
    augmented[q_key] = new_question
    augmented['cycle_targets'] = new_targets
    if old_gen_targets:
        augmented['cycle_gen_targets'] = new_gen_targets
    # Update final_answer if present
    if 'final_answer' in problem:
        augmented['final_answer'] = new_targets[-1] if new_targets else problem['final_answer']

    return augmented


# ---------------------------------------------------------------------------
# Per-cycle JSONL Dataset
# ---------------------------------------------------------------------------

class PerCycleDataset(Dataset):
    """Load per-cycle training data from JSONL files.

    Each line: {"problem": "...", "cycle_targets": [48, 24, 72],
                "final_answer": 72, "num_steps": 3}

    When augment=True, numbers are randomized each epoch (anti-memorization)
    and generation targets get equation-result dropout (anti-copying).
    """

    def __init__(self, jsonl_path: str, max_passes: int = 5,
                 augment: bool = False, dropout_prob: float = 0.3):
        self.samples = []
        self.max_passes = max_passes
        self.augment = augment
        self.dropout_prob = dropout_prob
        self.epoch = 0  # set externally each epoch
        with open(jsonl_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                self.samples.append(item)

    def set_epoch(self, epoch):
        """Call before each epoch to change augmentation seed."""
        self.epoch = epoch

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]

        # Number augmentation: different random numbers each epoch
        if self.augment:
            aug_rng = random.Random(self.epoch * 12345 + idx)
            item = simple_augment(item, aug_rng)

        cycle_targets = item['cycle_targets']
        # Truncate to max_passes if needed
        cycle_targets = cycle_targets[:self.max_passes]
        # Generation targets: equation strings (e.g. "160 - 63 = 97")
        # Falls back to stringified numbers if not present
        gen_targets = item.get('cycle_gen_targets',
                               [str(ct) for ct in cycle_targets])
        gen_targets = gen_targets[:self.max_passes]

        # Generation target dropout: mask equation results
        if self.augment:
            drop_rng = random.Random(self.epoch * 67890 + idx)
            for i in range(len(gen_targets)):
                target_num = cycle_targets[i] if i < len(cycle_targets) else 0
                gen_targets[i] = dropout_gen_target(
                    gen_targets[i], target_num,
                    dropout_prob=self.dropout_prob, rng=drop_rng,
                )

        return {
            'problem': item['problem'],
            'cycle_targets': cycle_targets,
            'cycle_gen_targets': gen_targets,
            'final_answer': item['final_answer'],
            'num_steps': min(item.get('num_steps', len(cycle_targets)), self.max_passes),
        }


def per_cycle_collate_fn(batch):
    """Collate with variable-length cycle_targets.

    Pads cycle_targets to max length in batch and creates a mask.
    """
    problems = [s['problem'] for s in batch]
    finals = [s['final_answer'] for s in batch]
    num_steps_list = [s['num_steps'] for s in batch]
    max_steps = max(num_steps_list)

    # Pad cycle_targets to max_steps, track mask
    padded_targets = []
    mask = []
    # Collect gen_targets per cycle position: list of max_steps lists
    gen_targets_by_cycle = [[] for _ in range(max_steps)]
    for s in batch:
        ct = s['cycle_targets']
        gt = s.get('cycle_gen_targets', [str(c) for c in ct])
        pad_len = max_steps - len(ct)
        padded_targets.append(ct + [0] * pad_len)
        mask.append([1.0] * len(ct) + [0.0] * pad_len)
        for i in range(max_steps):
            if i < len(gt):
                gen_targets_by_cycle[i].append(gt[i])
            else:
                gen_targets_by_cycle[i].append("0")  # placeholder for padded

    return {
        'problem': problems,
        'cycle_targets': torch.tensor(padded_targets, dtype=torch.long),
        'cycle_mask': torch.tensor(mask, dtype=torch.float32),
        'cycle_gen_targets': gen_targets_by_cycle,  # list of max_steps x list of B strings
        'final_answer': finals,
        'num_steps': num_steps_list,
        'max_steps': max_steps,
    }


# ---------------------------------------------------------------------------
# Forward with per-cycle generation-only loss
# ---------------------------------------------------------------------------

def per_cycle_target_weight(final_accuracy, cycle, total_cycles):
    """Smoothly fade intermediate targets as accuracy climbs.
    Final cycle: always 1.0. Intermediate cycles: sigmoid fade centered at 80%.
    Dormant below 70%. Fully faded above 90%."""
    if cycle == total_cycles - 1:
        return 1.0  # final cycle always fully supervised
    fade = torch.sigmoid(torch.tensor((final_accuracy - 0.80) * 15.0))
    return float(1.0 - fade)


def forward_train_per_cycle(model, problems, cycle_targets, cycle_mask,
                            finals_t, cycle_gen_targets=None,
                            num_passes=5, max_length=192,
                            final_accuracy=0.0):
    """Forward pass with generation-only per-cycle loss.

    Each cycle:
    1. Thinks (transformer + perceiver -> page)
    2. Generates the intermediate result as short text (e.g. "48") with LoRA ON

    The generation loss is the sole training signal (no answer head).

    Args:
        model:         AtomLoRAModel
        problems:      list of problem strings
        cycle_targets: (B, max_steps) integer tensor of per-cycle targets
        cycle_mask:    (B, max_steps) float mask (1.0 = valid, 0.0 = padded)
        finals_t:      (B,) final answer tensor (for contrastive loss)
        num_passes:    int, number of thinking passes
        max_length:    int, max tokenization length

    Returns:
        (gen_loss, c_loss, conf_loss, scale_reg_loss, iso_loss,
         page_cos_mean, active_atoms_mean, scale_std, cross_pass_cos,
         confidence_mean, state_pages, per_cycle_preds)
    """
    device = model.transformer.device

    inputs = model.tokenizer(
        problems, return_tensors='pt', padding=True,
        truncation=True, max_length=max_length,
    )
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)

    batch_size = input_ids.size(0)
    embed_layer = model.transformer.model.embed_tokens
    problem_embeds = embed_layer(input_ids)
    prompt_len = input_ids.size(1)

    cycle_targets = cycle_targets.to(device)
    cycle_mask = cycle_mask.to(device)

    state_pages = []
    raw_pages = []
    messages = []
    prev_predictions = []  # list of (B,) tensors -- accumulated gen predictions
    all_scales = []      # for diversity loss
    bypass_vectors = []  # for differentiable bypass
    atom_scales_history = []
    mid_states_history = []
    pre_tanh_history = []
    per_cycle_gen_loss = torch.tensor(0.0, device=device)
    per_cycle_preds = []  # list of (B,) tensors for diagnostics
    cycle_gen_logits_list = []  # collect gen_logits per cycle for context injection
    valid_cycles = 0

    for pass_num in range(num_passes):
        # --- For cycle 2+: inject ALL previous answers as text context ---
        if pass_num > 0 and len(state_pages) > 0:
            # Extract prediction from last cycle's gen_logits via argmax + decode
            with torch.no_grad():
                prev_logits = cycle_gen_logits_list[-1]  # (B, T, vocab)
                pred_tokens = prev_logits.argmax(dim=-1)  # (B, T)
                latest_preds = []
                for b in range(batch_size):
                    text = model.tokenizer.decode(pred_tokens[b], skip_special_tokens=True)
                    val = extract_answer_from_text(text)
                    latest_preds.append(val if val is not None else 0)
                latest_preds_t = torch.tensor(latest_preds, dtype=torch.long, device=device)
                prev_predictions.append(latest_preds_t)
            # Build cumulative context: "Step 1 result: 263\nStep 2 result: 346\n"
            context_strs = []
            for b in range(batch_size):
                ctx = ""
                for step_i, preds in enumerate(prev_predictions):
                    ctx += f"Step {step_i + 1} result: {int(preds[b].item())}\n"
                context_strs.append(ctx)
            augmented = [ctx + prob for ctx, prob in zip(context_strs, problems)]
            aug_inputs = model.tokenizer(
                augmented, return_tensors='pt', padding=True,
                truncation=True, max_length=max_length + 20 * len(prev_predictions),
            )
            cycle_input_ids = aug_inputs['input_ids'].to(device)
            cycle_attention_mask = aug_inputs['attention_mask'].to(device)
            cycle_embeds = embed_layer(cycle_input_ids)
            cycle_prompt_len = cycle_input_ids.size(1)
        else:
            cycle_embeds = problem_embeds
            cycle_attention_mask = attention_mask
            cycle_prompt_len = prompt_len

        # --- Build text context for hypernetwork (breaks fixed-point collapse) ---
        text_ctx_list = []
        for b in range(batch_size):
            prev_nums = []
            for prev_preds in prev_predictions:
                prev_nums.append(int(prev_preds[b].item()))
            text_ctx_list.append(encode_text_context(prev_nums, device=device))
        text_context = torch.stack(text_ctx_list)  # (B, 16)

        # --- Compute bypass summary for hypernetwork ---
        if bypass_vectors and len(bypass_vectors) > 0:
            bypass_summary = torch.stack(bypass_vectors, dim=0).mean(dim=0)
        else:
            bypass_summary = None

        # --- Get atom scales for this cycle (pages + messages) ---
        if pass_num == 0 and len(state_pages) == 0:
            hyper_dtype = next(model.hypernet.parameters()).dtype
            zero_page = torch.zeros(batch_size, model.page_size,
                                    device=device, dtype=hyper_dtype)
            atom_scales, pre_tanh, _focus = model.hypernet(
                [zero_page], pass_num=0, return_pre_tanh=True,
                messages=messages if messages else None,
                text_context=text_context,
                bypass_summary=bypass_summary,
            )
        else:
            atom_scales, pre_tanh, _focus = model.hypernet(
                state_pages, pass_num=pass_num, return_pre_tanh=True,
                messages=messages if messages else None,
                text_context=text_context,
                bypass_summary=bypass_summary,
            )

        # --- Think: run transformer with this cycle's LoRA ---
        manager = AtomAdditiveLoRAManager(model.transformer)
        manager.apply(model.atoms, atom_scales)
        try:
            outputs = model.transformer(
                inputs_embeds=cycle_embeds, attention_mask=cycle_attention_mask,
                output_hidden_states=True,
            )
            hidden_states = list(outputs.hidden_states[1:])
        finally:
            manager.remove()

        # --- Perceiver: compress to page ---
        page_delta, _strategy, current_mid_states = model.compressor(
            hidden_states, pass_num,
            prev_mid_states=mid_states_history if mid_states_history else None,
        )
        mid_states_history.append(current_mid_states)
        raw_pages.append(page_delta)
        page = F.normalize(page_delta, dim=-1) * model.page_radius

        # Page noise during training
        if model.training:
            page = page + torch.randn_like(page) * 0.05

        # Gradient scaling per cycle (capped at 4x)
        grad_scale = min(float(num_passes - pass_num), 4.0)
        page = scale_gradient(page, grad_scale)

        # Pages always stay in the computation graph -- later cycles need
        # the gradient highway through earlier pages. Graduation works by
        # skipping the graduated cycle's LOSS, not by detaching the page.
        state_pages.append(page)

        # Generate message: direct signal from last layer, bypasses perceiver
        message = model.message_generator(hidden_states[-1])
        messages.append(message)

        # Differentiable bypass: rich signal from hidden states to hypernetwork
        hidden_pool = hidden_states[-1].mean(dim=1).float()  # (B, d_model)
        if hasattr(model, 'bypass'):
            bypass_vec = model.bypass(hidden_pool)
            all_scales.append(atom_scales)
            bypass_vectors.append(bypass_vec)
        else:
            all_scales.append(atom_scales)

        # Atom dropout during training: randomly zero 10% of atom scales
        if model.training and pass_num > 0:
            atom_mask = (torch.rand_like(atom_scales) > 0.1).float()
            atom_scales = atom_scales * atom_mask

        atom_scales_history.append(atom_scales)
        pre_tanh_history.append(pre_tanh)

        # ------- Per-cycle generation loss -------
        if pass_num < cycle_targets.size(1):
            cycle_target = cycle_targets[:, pass_num]   # (B,)
            mask_val = cycle_mask[:, pass_num]           # (B,)

            if mask_val.sum() > 0:
                # === GENERATION LOSS ===
                # Tokenize short target: just the number as text (e.g. "48")
                # Use equation gen targets if available (e.g. "160 - 63 = 97")
                # Otherwise fall back to stringified numbers
                if cycle_gen_targets is not None and pass_num < len(cycle_gen_targets):
                    target_strs = cycle_gen_targets[pass_num]  # list of B strings
                else:
                    target_strs = [str(ct.item()) for ct in cycle_target]

                # Append #### answer marker to each target
                # Format: "{natural sentence} #### {number}"
                # The number comes from cycle_target for this cycle
                cycle_numbers = [str(ct.item()) for ct in cycle_target]
                target_strs = [f"{text} #### {num}" for text, num in zip(target_strs, cycle_numbers)]

                target_inputs = model.tokenizer(
                    target_strs, return_tensors='pt', padding=True,
                    add_special_tokens=False,
                )
                target_ids = target_inputs['input_ids'].to(device)  # (B, T)
                # Append EOS token so model learns when to stop
                eos_id = model.tokenizer.eos_token_id
                if eos_id is not None:
                    eos_col = torch.full((target_ids.size(0), 1), eos_id,
                                         dtype=target_ids.dtype, device=device)
                    target_ids = torch.cat([target_ids, eos_col], dim=1)

                # Teacher-forced generation with THIS cycle's LoRA
                # Use augmented input (with injected context) for cycle 2+
                target_embeds = embed_layer(target_ids)
                full_embeds = torch.cat([cycle_embeds, target_embeds], dim=1)
                target_attn = (target_ids != model.tokenizer.pad_token_id).long()
                full_attn = torch.cat([cycle_attention_mask, target_attn], dim=1)
                manager = AtomAdditiveLoRAManager(model.transformer)
                manager.apply(model.atoms, atom_scales)
                try:
                    gen_outputs = model.transformer(
                        inputs_embeds=full_embeds,
                        attention_mask=full_attn,
                        use_cache=False,
                    )
                finally:
                    manager.remove()

                # Cross-entropy on target tokens only
                gen_logits = gen_outputs.logits[:, cycle_prompt_len - 1:-1, :]
                # Weight EOS 5x: stopping at the right place is more important
                # than getting any individual word right
                eos_id = model.tokenizer.eos_token_id
                vocab_size = gen_logits.size(-1)
                token_weights = torch.ones(vocab_size, device=device, dtype=gen_logits.dtype)
                if eos_id is not None and eos_id < vocab_size:
                    token_weights[eos_id] = 5.0
                gen_loss_this = F.cross_entropy(
                    gen_logits.reshape(-1, gen_logits.size(-1)),
                    target_ids.reshape(-1),
                    weight=token_weights,
                    ignore_index=model.tokenizer.pad_token_id,
                    reduction='none',
                    label_smoothing=0.05,
                )
                # Mask per sample, then average
                target_mask = (target_ids != model.tokenizer.pad_token_id).float()
                gen_loss_per_sample = (gen_loss_this.view(batch_size, -1) * target_mask).sum(dim=1)
                gen_loss_per_sample = gen_loss_per_sample / target_mask.sum(dim=1).clamp(min=1)
                # Apply cycle mask
                gen_loss_this = (gen_loss_per_sample * mask_val).sum() / mask_val.sum()

                # Store gen_logits for this cycle (for text injection into next cycle)
                cycle_gen_logits_list.append(gen_logits.detach())

                # Smooth fading: intermediate cycle targets fade as accuracy climbs
                total_supervised = cycle_targets.size(1)
                fade_w = per_cycle_target_weight(final_accuracy, pass_num, total_supervised)

                # Three-tier per-sample gating:
                #   Correct new target → 1.0 (full reward)
                #   Wrong but trying   → 0.1 (reduced, still learns language)
                #   Copying consumed   → 0.0 (zero reward for repeating)
                if pass_num == 0:
                    # Cycle 1: always full weight (nothing to copy yet)
                    per_cycle_gen_loss = per_cycle_gen_loss + gen_loss_this * fade_w
                else:
                    with torch.no_grad():
                        pred_tokens = gen_logits.argmax(dim=-1)
                        per_sample_weights = torch.ones(batch_size, device=device)
                        for b in range(batch_size):
                            text = model.tokenizer.decode(pred_tokens[b], skip_special_tokens=True)
                            pred_val = extract_answer_from_text(text)
                            if pred_val is not None:
                                # Check if copying a consumed target
                                is_copy = False
                                for prev_preds in prev_predictions:
                                    if int(prev_preds[b].item()) == pred_val:
                                        is_copy = True
                                        break
                                if is_copy:
                                    per_sample_weights[b] = 0.0  # copying → zero reward
                                elif pred_val == int(cycle_target[b].item()):
                                    per_sample_weights[b] = 1.0  # correct → full reward
                                else:
                                    per_sample_weights[b] = 0.1  # wrong → reduced
                            else:
                                per_sample_weights[b] = 0.1  # no extraction → reduced
                    weighted_gen = (gen_loss_per_sample * per_sample_weights * mask_val).sum() / mask_val.sum()
                    per_cycle_gen_loss = per_cycle_gen_loss + weighted_gen * fade_w
                valid_cycles += 1

            # Record predictions for diagnostics (extract from gen_logits)
            with torch.no_grad():
                if cycle_gen_logits_list:
                    last_logits = cycle_gen_logits_list[-1]
                    pred_tokens = last_logits.argmax(dim=-1)  # (B, T)
                    preds_list = []
                    for b in range(batch_size):
                        text = model.tokenizer.decode(pred_tokens[b], skip_special_tokens=True)
                        val = extract_answer_from_text(text)
                        # Clamp to prevent overflow in torch.long
                        v = val if val is not None else 0
                        v = max(min(v, 999999999), -999999999)
                        preds_list.append(v)
                    preds = torch.tensor(preds_list, dtype=torch.long, device=device)
                else:
                    preds = torch.zeros(batch_size, dtype=torch.long, device=device)
                per_cycle_preds.append(preds)

    # Normalize losses by number of valid cycles
    if valid_cycles > 0:
        per_cycle_gen_loss = per_cycle_gen_loss / valid_cycles

    # ------- Confidence head: correctness signal + entropy regularization -------
    conf_loss = torch.tensor(0.0, device=device)
    exit_probs = []
    for pg_idx, pg in enumerate(state_pages):
        conf_pred = model.confidence_head(state_pages[:pg_idx+1])  # reads first k pages
        exit_probs.append(conf_pred.mean())  # average confidence across batch
        with torch.no_grad():
            # Check correctness via generation extraction
            if pg_idx < len(per_cycle_preds):
                pred = per_cycle_preds[pg_idx]
            else:
                pred = torch.zeros(batch_size, dtype=torch.long, device=device)
            target = (pred == finals_t).float().unsqueeze(-1)  # (B, 1)
        conf_loss = conf_loss + F.binary_cross_entropy(conf_pred, target)
    conf_loss = conf_loss / max(len(state_pages), 1)

    # Entropy reg: prevent collapse to "always stop at cycle N"
    if len(exit_probs) > 1:
        exit_dist = torch.stack(exit_probs)
        exit_dist = exit_dist / (exit_dist.sum() + 1e-8)
        entropy = -(exit_dist * torch.log(exit_dist + 1e-8)).sum()
        conf_loss = conf_loss - 0.01 * entropy  # maximize entropy

    # ------- Contrastive loss -------
    c_loss = per_page_contrastive_loss(
        state_pages, finals_t, cross_target=0.7, within_target=0.3,
    )

    # ------- Scale diversity loss (computed ONCE after all cycles) -------
    diversity_loss = scale_diversity_loss(all_scales, target_cos=0.3)

    # ------- Atom diagnostics -------
    with torch.no_grad():
        last_page = state_pages[-1].float()
        normed = F.normalize(last_page, dim=-1)
        sim = normed @ normed.T
        B = sim.size(0)
        off_diag = sim - torch.eye(B, device=sim.device)
        page_cos_mean = off_diag.sum() / max(B * (B - 1), 1)

        diag_scales = torch.stack(atom_scales_history, dim=0)  # (P, B, A)
        active_per_pass = (diag_scales.abs() > 0.1).float().sum(dim=-1)
        active_atoms_mean = active_per_pass.mean()

        scale_std = diag_scales.std(dim=-1).mean()

        if len(atom_scales_history) >= 2:
            s0 = F.normalize(atom_scales_history[0], dim=-1)
            sN = F.normalize(atom_scales_history[-1], dim=-1)
            cross_pass_cos = (s0 * sN).sum(dim=-1).mean()
        else:
            cross_pass_cos = torch.tensor(0.0, device=device)

    # ------- Pre-tanh regularization -------
    all_pre_tanh = torch.cat(pre_tanh_history, dim=0)
    scale_reg_loss = (all_pre_tanh ** 2).mean()

    # ------- Isotropic regularization (raw pages before normalization) -------
    if raw_pages:
        raw_pages_flat = torch.cat(raw_pages, dim=0)  # (cycles*B, 64)
        iso_loss = model.isotropic_reg(raw_pages_flat.float())
    else:
        iso_loss = torch.tensor(0.0, device=device)

    return (per_cycle_gen_loss,
            c_loss, conf_loss, scale_reg_loss, iso_loss,
            page_cos_mean, active_atoms_mean, scale_std, cross_pass_cos,
            conf_loss.detach(), state_pages, per_cycle_preds, diversity_loss)


# ---------------------------------------------------------------------------
# Evaluate (generation + regex extraction, per-cycle accuracy)
# ---------------------------------------------------------------------------

def evaluate_per_cycle(model, eval_dataset, device,
                       num_passes=5, max_length=192):
    """Evaluate via generation + regex extraction at each cycle.

    Returns:
        per_cycle_acc: list of floats, accuracy at each cycle
        final_acc:     float, accuracy of last cycle (should match final_answer)
    """
    model.eval()
    eval_batch = 16
    max_steps_seen = 0

    per_cycle_correct = {}  # pass_num -> count
    per_cycle_total = {}    # pass_num -> count
    final_correct = 0
    total = 0

    with torch.no_grad():
        for i in range(0, len(eval_dataset), eval_batch):
            batch_samples = [
                eval_dataset[j]
                for j in range(i, min(i + eval_batch, len(eval_dataset)))
            ]
            problems = [s['problem'] for s in batch_samples]
            cycle_targets_list = [s['cycle_targets'] for s in batch_samples]
            gold_finals = [s['final_answer'] for s in batch_samples]

            inputs = model.tokenizer(
                problems, return_tensors='pt', padding=True,
                truncation=True, max_length=max_length,
            )
            input_ids = inputs['input_ids'].to(device)
            attention_mask = inputs['attention_mask'].to(device)
            batch_size = input_ids.size(0)

            state_pages = []
            eval_messages = []
            eval_prev_predictions = []
            mid_states_history = []
            bypass_vectors = []
            embed_layer = model.transformer.model.embed_tokens
            eval_pred_vals = []  # list of lists: per cycle, per sample predicted values

            for pass_num in range(num_passes):
                # For cycle 2+: inject ALL previous answers as text context
                if pass_num > 0 and len(state_pages) > 0 and eval_prev_predictions:
                    context_strs = []
                    for b in range(batch_size):
                        ctx = ""
                        for step_i, preds in enumerate(eval_prev_predictions):
                            ctx += f"Step {step_i + 1} result: {preds[b]}\n"
                        context_strs.append(ctx)
                    augmented = [ctx + prob for ctx, prob in zip(context_strs, problems)]
                    aug_inputs = model.tokenizer(
                        augmented, return_tensors='pt', padding=True,
                        truncation=True, max_length=max_length + 20 * len(eval_prev_predictions),
                    )
                    eval_ids = aug_inputs['input_ids'].to(device)
                    eval_mask = aug_inputs['attention_mask'].to(device)
                else:
                    eval_ids = input_ids
                    eval_mask = attention_mask

                # Build text context for hypernetwork
                text_ctx_list = []
                for b in range(batch_size):
                    prev_nums = []
                    for prev_preds in eval_prev_predictions:
                        prev_nums.append(int(prev_preds[b]))
                    text_ctx_list.append(encode_text_context(prev_nums, device=device))
                text_context = torch.stack(text_ctx_list)  # (B, 16)

                page, _scales, current_mid_states, message, _raw_page, hidden_pool, _focus, bypass_vec = model.thinking_pass(
                    eval_ids, eval_mask, state_pages, pass_num,
                    prev_mid_states=mid_states_history if mid_states_history else None,
                    messages=eval_messages if eval_messages else None,
                    text_context=text_context,
                    bypass_vectors=bypass_vectors,
                )
                state_pages.append(page)
                eval_messages.append(message)
                mid_states_history.append(current_mid_states)
                bypass_vectors.append(bypass_vec)

                # Generate text with LoRA applied
                prompt_len = eval_ids.size(1)

                # Get atom scales for generation
                if bypass_vectors and len(bypass_vectors) > 0:
                    eval_bypass_summary = torch.stack(bypass_vectors, dim=0).mean(dim=0)
                else:
                    eval_bypass_summary = None

                if pass_num == 0 and len(state_pages) == 1:
                    hyper_dtype = next(model.hypernet.parameters()).dtype
                    zero_page = torch.zeros(batch_size, model.page_size,
                                            device=device, dtype=hyper_dtype)
                    atom_scales, _, _focus = model.hypernet(
                        [zero_page], pass_num=0, return_pre_tanh=True,
                        messages=eval_messages if eval_messages else None,
                        text_context=text_context,
                        bypass_summary=eval_bypass_summary,
                    )
                else:
                    atom_scales, _, _focus = model.hypernet(
                        state_pages, pass_num=pass_num, return_pre_tanh=True,
                        messages=eval_messages if eval_messages else None,
                        text_context=text_context,
                        bypass_summary=eval_bypass_summary,
                    )

                manager = AtomAdditiveLoRAManager(model.transformer)
                manager.apply(model.atoms, atom_scales)
                try:
                    gen_out = model.transformer.generate(
                        input_ids=eval_ids, attention_mask=eval_mask,
                        max_new_tokens=60, do_sample=False,
                        pad_token_id=model.tokenizer.pad_token_id,
                        eos_token_id=model.tokenizer.eos_token_id,
                    )
                finally:
                    manager.remove()

                # Extract predictions from generated text
                cycle_preds = []  # list of int or None for this cycle
                for b in range(batch_size):
                    gen_text = model.tokenizer.decode(gen_out[b][prompt_len:], skip_special_tokens=False)
                    pred_val = extract_answer_from_text(gen_text)
                    cycle_preds.append(pred_val)
                eval_pred_vals.append(cycle_preds)
                eval_prev_predictions.append(cycle_preds)

                # Per-cycle accuracy
                for j in range(batch_size):
                    ct = cycle_targets_list[j]
                    if pass_num < len(ct):
                        gold_cycle = ct[pass_num]
                        pred_val = cycle_preds[j]

                        if pass_num not in per_cycle_correct:
                            per_cycle_correct[pass_num] = 0
                            per_cycle_total[pass_num] = 0
                        per_cycle_total[pass_num] += 1

                        try:
                            if pred_val is not None and int(pred_val) == int(gold_cycle):
                                per_cycle_correct[pass_num] += 1
                        except (ValueError, TypeError):
                            pass

                        max_steps_seen = max(max_steps_seen, pass_num + 1)

            # Final accuracy: use the LAST SUPERVISED cycle's prediction
            for j in range(batch_size):
                gold = gold_finals[j]
                ct = cycle_targets_list[j]
                last_supervised = min(len(ct), len(eval_pred_vals)) - 1
                pred_val = eval_pred_vals[last_supervised][j]
                try:
                    if pred_val is not None and int(pred_val) == int(gold):
                        final_correct += 1
                except (ValueError, TypeError):
                    pass
                total += 1

    # Build per-cycle accuracy list
    per_cycle_acc = []
    for p in range(max_steps_seen):
        if p in per_cycle_total and per_cycle_total[p] > 0:
            per_cycle_acc.append(
                100.0 * per_cycle_correct.get(p, 0) / per_cycle_total[p]
            )
        else:
            per_cycle_acc.append(0.0)

    final_acc = 100.0 * final_correct / total if total > 0 else 0.0
    return per_cycle_acc, final_acc


# ---------------------------------------------------------------------------
# Inference with confidence-based early stopping
# ---------------------------------------------------------------------------

def infer_single(model, problem_text, device,
                 max_passes=5, confidence_threshold=0.9, max_length=192):
    """Infer answer for a single problem using generation + confidence stopping.

    Returns:
        answer: int or None, predicted answer
        stopped_at: int, pass number where inference stopped
    """
    model.eval()

    with torch.no_grad():
        inputs = model.tokenizer(
            [problem_text], return_tensors='pt', padding=True,
            truncation=True, max_length=max_length,
        )
        input_ids = inputs['input_ids'].to(device)
        attention_mask = inputs['attention_mask'].to(device)

        state_pages = []
        mid_states_history = []
        bypass_vectors = []
        embed_layer = model.transformer.model.embed_tokens
        last_pred = None
        infer_prev_predictions = []  # list of extracted ints per cycle

        for pass_num in range(max_passes):
            # Build text context for hypernetwork
            text_context = encode_text_context(
                infer_prev_predictions, device=device,
            ).unsqueeze(0)  # (1, 16) — single sample

            page, _scales, current_mid_states, _message, _raw_page, hidden_pool, _focus, bypass_vec = model.thinking_pass(
                input_ids, attention_mask, state_pages, pass_num,
                prev_mid_states=mid_states_history if mid_states_history else None,
                text_context=text_context,
                bypass_vectors=bypass_vectors,
            )
            state_pages.append(page)
            mid_states_history.append(current_mid_states)
            bypass_vectors.append(bypass_vec)

            # Get atom scales for generation
            if bypass_vectors and len(bypass_vectors) > 0:
                infer_bypass_summary = torch.stack(bypass_vectors, dim=0).mean(dim=0)
            else:
                infer_bypass_summary = None

            if pass_num == 0:
                hyper_dtype = next(model.hypernet.parameters()).dtype
                zero_page = torch.zeros(1, model.page_size,
                                        device=device, dtype=hyper_dtype)
                atom_scales, _, _focus = model.hypernet(
                    [zero_page], pass_num=0, return_pre_tanh=True,
                    text_context=text_context,
                    bypass_summary=infer_bypass_summary,
                )
            else:
                atom_scales, _, _focus = model.hypernet(
                    state_pages, pass_num=pass_num, return_pre_tanh=True,
                    text_context=text_context,
                    bypass_summary=infer_bypass_summary,
                )

            # Generate with LoRA applied
            manager = AtomAdditiveLoRAManager(model.transformer)
            manager.apply(model.atoms, atom_scales)
            try:
                gen_out = model.transformer.generate(
                    input_ids=input_ids, attention_mask=attention_mask,
                    max_new_tokens=60, do_sample=False,
                    pad_token_id=model.tokenizer.pad_token_id,
                    eos_token_id=model.tokenizer.eos_token_id,
                )
            finally:
                manager.remove()

            prompt_len = input_ids.size(1)
            gen_text = model.tokenizer.decode(gen_out[0][prompt_len:], skip_special_tokens=False)
            last_pred = extract_answer_from_text(gen_text)
            infer_prev_predictions.append(last_pred if last_pred is not None else 0)

            # Confidence-based early stopping (after at least 1 pass)
            if pass_num >= 1:
                conf = model.confidence_head(state_pages)
                if conf.mean().item() > confidence_threshold:
                    break

        return last_pred, pass_num + 1


# ---------------------------------------------------------------------------
# Atom gradient norm logging
# ---------------------------------------------------------------------------

def log_atom_grad_norms(model):
    """Log gradient norms for atom A params, atom B params, and hypernetwork."""
    norms = {}

    a_norm = 0.0
    a_count = 0
    for name, param in model.atoms.A.items():
        if param.grad is not None:
            a_norm += param.grad.norm().item()
            a_count += 1
    norms['atoms_A'] = a_norm / max(a_count, 1)

    b_norm = 0.0
    b_count = 0
    for name, param in model.atoms.B.items():
        if param.grad is not None:
            b_norm += param.grad.norm().item()
            b_count += 1
    norms['atoms_B'] = b_norm / max(b_count, 1)

    h_norm = 0.0
    h_count = 0
    for p in model.hypernet.parameters():
        if p.grad is not None:
            h_norm += p.grad.norm().item()
            h_count += 1
    norms['hypernet'] = h_norm / max(h_count, 1)

    return norms


# ---------------------------------------------------------------------------
# Per-dimension page variance diagnostic
# ---------------------------------------------------------------------------

def log_page_variance(state_pages, device):
    """Log per-dimension variance across a batch of pages.

    Returns: mean variance across dimensions, and number of 'dead' dims (var < 0.01).
    """
    if not state_pages:
        return 0.0, 0
    last_page = state_pages[-1].float()  # (B, 64)
    per_dim_var = last_page.var(dim=0)   # (64,)
    mean_var = per_dim_var.mean().item()
    dead_dims = (per_dim_var < 0.01).sum().item()
    return mean_var, dead_dims


# ---------------------------------------------------------------------------
# Warm-start logic
# ---------------------------------------------------------------------------

def try_warm_start(model, warm_path, skip_perceiver=False):
    """Try atom checkpoint first; fall back to perceiver-only warm start.
    Returns optimizer state dict if present in checkpoint, else None."""
    ckpt = torch.load(warm_path, map_location='cpu', weights_only=True)

    if 'atoms' in ckpt:
        print(f"  Loading atom checkpoint from {warm_path}")

        if skip_perceiver:
            print(f"  compressor: SKIPPED (--skip_perceiver flag)")
        else:
            own = model.compressor.state_dict()
            loaded = 0
            for k, v in ckpt['compressor'].items():
                if k in own and own[k].shape == v.shape:
                    own[k] = v
                    loaded += 1
            model.compressor.load_state_dict(own, strict=False)
            print(f"  compressor: loaded {loaded}/{len(own)}")

        if 'atoms' in ckpt:
            own_a = model.atoms.state_dict()
            loaded_a = 0
            for k, v in ckpt['atoms'].items():
                if k in own_a and own_a[k].shape == v.shape:
                    own_a[k] = v
                    loaded_a += 1
            model.atoms.load_state_dict(own_a, strict=False)
            print(f"  atoms: loaded {loaded_a}/{len(own_a)}")

        if 'hypernet' in ckpt:
            own_h = model.hypernet.state_dict()
            loaded_h = 0
            for k, v in ckpt['hypernet'].items():
                if k in own_h and own_h[k].shape == v.shape:
                    own_h[k] = v
                    loaded_h += 1
            model.hypernet.load_state_dict(own_h, strict=False)
            print(f"  hypernet: loaded {loaded_h}/{len(own_h)}")

        if 'confidence_head' in ckpt:
            own_c = model.confidence_head.state_dict()
            loaded_c = 0
            for k, v in ckpt['confidence_head'].items():
                if k in own_c and own_c[k].shape == v.shape:
                    own_c[k] = v
                    loaded_c += 1
            model.confidence_head.load_state_dict(own_c, strict=False)
            print(f"  confidence_head: loaded {loaded_c}/{len(own_c)}")

        # Skip answer_head -- removed from architecture

        if 'residual_gate' in ckpt:
            own_rg = model.residual_gate.state_dict()
            loaded_rg = 0
            for k, v in ckpt['residual_gate'].items():
                if k in own_rg and own_rg[k].shape == v.shape:
                    own_rg[k] = v
                    loaded_rg += 1
            model.residual_gate.load_state_dict(own_rg, strict=False)
            print(f"  residual_gate: loaded {loaded_rg}/{len(own_rg)}")

        if 'ordinal_head' in ckpt:
            own_oh = model.ordinal_head.state_dict()
            loaded_oh = 0
            for k, v in ckpt['ordinal_head'].items():
                if k in own_oh and own_oh[k].shape == v.shape:
                    own_oh[k] = v
                    loaded_oh += 1
            model.ordinal_head.load_state_dict(own_oh, strict=False)
            print(f"  ordinal_head: loaded {loaded_oh}/{len(own_oh)}")

        if 'message_generator' in ckpt:
            own_mg = model.message_generator.state_dict()
            loaded_mg = 0
            for k, v in ckpt['message_generator'].items():
                if k in own_mg and own_mg[k].shape == v.shape:
                    own_mg[k] = v
                    loaded_mg += 1
            model.message_generator.load_state_dict(own_mg, strict=False)
            print(f"  message_generator: loaded {loaded_mg}/{len(own_mg)}")

        if 'bypass' in ckpt and hasattr(model, 'bypass'):
            own_bp = model.bypass.state_dict()
            loaded_bp = 0
            for k, v in ckpt['bypass'].items():
                if k in own_bp and own_bp[k].shape == v.shape:
                    own_bp[k] = v
                    loaded_bp += 1
            model.bypass.load_state_dict(own_bp, strict=False)
            print(f"  bypass: loaded {loaded_bp}/{len(own_bp)}")

        if 'optimizer' in ckpt:
            print(f"  optimizer state: found")
            return ckpt['optimizer']
        return None

    else:
        print(f"  Warm-starting perceiver only from {warm_path}")
        warm_start_atom_from_checkpoint(model, ckpt)
        print(f"  atoms: fresh init (no atom equivalent in checkpoint)")
        print(f"  hypernet: fresh init (new architecture)")
        return None


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(args):
    level = args.level
    warm_path = args.warm_from

    print("=" * 60)
    print(f"Per-Cycle Generation-Only Training -- Level {level}")
    print("=" * 60)
    print(f"  num_passes     = {args.num_passes}")
    print(f"  batch_size     = {args.batch_size}")
    print(f"  epochs         = {args.epochs}")
    print(f"  patience       = {args.patience}")
    print(f"  lam_gen        = {args.lam_gen}")
    print(f"  lam_contrastive= {args.lam_contrastive}")
    print(f"  lam_conf       = {args.lam_conf}")
    print(f"  lam_scale_reg  = {args.lam_scale_reg}")
    print(f"  lam_diversity  = {args.lam_diversity}")
    print(f"  skip_perceiver = {args.skip_perceiver}")
    print(f"  num_atoms      = {args.num_atoms}")
    print(f"  atom_rank      = {args.atom_rank}")
    print(f"  data_dir       = {args.data_dir}")
    print(f"  warm_from      = {warm_path}")
    print("=" * 60)

    device = torch.device('cuda')
    model = AtomLoRAModel(
        num_atoms=args.num_atoms,
        atom_rank=args.atom_rank,
        skip_pass_embed=True,  # v24.6: pages are the only input
    )
    model.compressor = model.compressor.to(device=device, dtype=torch.bfloat16)
    model.hypernet = model.hypernet.to(device=device, dtype=torch.bfloat16)
    model.atoms = model.atoms.to(device=device, dtype=torch.bfloat16)
    model.confidence_head = model.confidence_head.to(device)  # fp32 (small)
    model.residual_gate = model.residual_gate.to(device=device, dtype=torch.bfloat16)
    model.probe_head = model.probe_head.to(device)
    model.message_generator = model.message_generator.to(device)  # fp32 (small)
    model.ordinal_head = model.ordinal_head.to(device)  # fp32 (small)

    # Differentiable bypass: create if not already on the model
    if not hasattr(model, 'bypass'):
        model.bypass = DifferentiableBypass(hidden_dim=2048, bypass_dim=512)
    model.bypass = model.bypass.to(device)  # fp32 (small)

    saved_optim_state = None
    if warm_path:
        print(f"\nWarm-starting from {warm_path}")
        if args.skip_perceiver:
            print("  (--skip_perceiver: perceiver will stay random)")
        saved_optim_state = try_warm_start(model, warm_path, skip_perceiver=args.skip_perceiver)

    # --- Data ---
    train_path = os.path.join(args.data_dir, f'{level}_train.jsonl')
    eval_path = os.path.join(args.data_dir, f'{level}_eval.jsonl')

    if not os.path.exists(train_path):
        print(f"ERROR: Training data not found at {train_path}")
        print(f"Expected JSONL format: {{\"problem\": \"...\", \"cycle_targets\": [...], \"final_answer\": N, \"num_steps\": N}}")
        sys.exit(1)

    train_dataset = PerCycleDataset(train_path, max_passes=args.num_passes,
                                     augment=args.augment,
                                     dropout_prob=args.dropout_prob)
    aug_str = f" (augment={args.augment}, dropout={args.dropout_prob})" if args.augment else ""
    print(f"\nTrain dataset: {len(train_dataset)} problems from {train_path}{aug_str}")

    if os.path.exists(eval_path):
        # Eval is NEVER augmented
        eval_dataset = PerCycleDataset(eval_path, max_passes=args.num_passes)
        print(f"Eval dataset: {len(eval_dataset)} problems from {eval_path}")
    else:
        # Fall back to using last 200 from training data
        print(f"  Eval file not found at {eval_path}, using last 200 of training data")
        eval_size = min(200, len(train_dataset))
        eval_dataset = torch.utils.data.Subset(
            train_dataset,
            range(len(train_dataset) - eval_size, len(train_dataset)),
        )

    max_length = 192

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        collate_fn=per_cycle_collate_fn,
    )

    # ------- Optimizer: atoms A/B + hypernet + compressor + heads -------
    atom_A_params = list(model.atoms.A.values()) if hasattr(model.atoms.A, 'values') else list(model.atoms.A.parameters())
    atom_B_params = list(model.atoms.B.values()) if hasattr(model.atoms.B, 'values') else list(model.atoms.B.parameters())

    s = args.lr_scale
    optimizer = torch.optim.AdamW([
        {'params': list(model.compressor.parameters()), 'lr': 1e-4 * s, 'weight_decay': 0.01},
        {'params': atom_A_params, 'lr': 1e-4 * s, 'weight_decay': 0.05},
        {'params': atom_B_params, 'lr': 1e-4 * s, 'weight_decay': 0.05},
        {'params': list(model.hypernet.parameters()), 'lr': 3e-4 * s, 'weight_decay': 0.1},
        {'params': list(model.confidence_head.parameters()), 'lr': 1e-3 * s, 'weight_decay': 0.01},
        {'params': list(model.message_generator.parameters()), 'lr': 1e-3 * s, 'weight_decay': 0.01},
        {'params': list(model.ordinal_head.parameters()), 'lr': 1e-3 * s, 'weight_decay': 0.01},
        {'params': list(model.bypass.parameters()), 'lr': 1e-4 * s, 'weight_decay': 0.01},
    ])

    # Note: optimizer state loading disabled -- architecture changes between
    # checkpoints cause shape mismatches in momentum buffers. Fresh optimizer
    # with lr_scale is sufficient for warm starts.
    if saved_optim_state is not None:
        print("  optimizer state: skipped (architecture may have changed)")

    trainable = (
        list(model.compressor.parameters())
        + list(model.atoms.parameters())
        + list(model.hypernet.parameters())
        + list(model.confidence_head.parameters())
        + list(model.message_generator.parameters())
        + list(model.ordinal_head.parameters())
        + list(model.bypass.parameters())
    )
    print(f"Trainable params: {sum(p.numel() for p in trainable):,}")
    print(f"  atoms: {sum(p.numel() for p in model.atoms.parameters()):,} "
          f"({args.num_atoms} atoms, rank {args.atom_rank})")
    print(f"  hypernet: {sum(p.numel() for p in model.hypernet.parameters()):,}")
    print(f"  compressor: {sum(p.numel() for p in model.compressor.parameters()):,}")
    print(f"  confidence_head: {sum(p.numel() for p in model.confidence_head.parameters()):,}")
    print(f"  message_gen: {sum(p.numel() for p in model.message_generator.parameters()):,}")
    print(f"  bypass: {sum(p.numel() for p in model.bypass.parameters()):,}")
    print(f"GPU mem after init: {torch.cuda.memory_allocated()/1e9:.2f} GB")

    # Baseline eval
    per_cycle_acc, final_acc = evaluate_per_cycle(
        model, eval_dataset, device,
        num_passes=args.num_passes, max_length=max_length,
    )
    print(f"\nBaseline: final={final_acc:.1f}%  per_cycle={[f'{a:.1f}%' for a in per_cycle_acc]}\n")

    ckpt_name = f"checkpoints/per_cycle_{level.replace('.', '')}_best.pt"
    best_final = 0.0
    # Graduation parked -- disrupts equilibrium worse than oscillation.
    # All cycles train at full weight. Track peak accuracy.
    patience_counter = 0

    for epoch in range(args.epochs):
        model.train()
        t0 = time.time()

        # Update augmentation seed for this epoch (different numbers each epoch)
        train_dataset.set_epoch(epoch)

        # Re-shuffle data each epoch (DataLoader shuffle=True handles this)
        ep_gen = ep_ctr = ep_conf = ep_scale_reg = ep_iso = ep_cos = 0.0
        ep_active = ep_std = ep_xpass = 0.0
        ep_div = 0.0
        ep_page_var = 0.0
        ep_dead_dims = 0.0
        nb = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False):
            problems = batch['problem']
            cycle_targets = batch['cycle_targets']  # (B, max_steps)
            cycle_mask = batch['cycle_mask']          # (B, max_steps)
            cycle_gen_targets = batch.get('cycle_gen_targets')  # list of max_steps x B strings

            finals_raw = batch['final_answer']
            finals_list = []
            for f in finals_raw:
                try:
                    finals_list.append(int(f))
                except (ValueError, TypeError):
                    finals_list.append(0)
            finals_t = torch.tensor(finals_list, dtype=torch.long, device=device)

            optimizer.zero_grad()

            (gen_loss, c_loss, conf_loss, scale_reg, iso_loss,
             page_cos, active_atoms, s_std, xpass_cos,
             conf_mean, state_pages, per_cycle_preds, div_loss) = forward_train_per_cycle(
                model, problems, cycle_targets, cycle_mask,
                finals_t, cycle_gen_targets=cycle_gen_targets,
                num_passes=args.num_passes, max_length=max_length,
                final_accuracy=best_final / 100.0,  # smooth fading uses 0-1 scale
            )

            total_loss = (args.lam_gen * gen_loss
                          + args.lam_contrastive * c_loss
                          + args.lam_conf * conf_loss
                          + args.lam_scale_reg * scale_reg
                          + 0.01 * iso_loss
                          + args.lam_diversity * div_loss)
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()

            ep_gen += gen_loss.item()
            ep_ctr += c_loss.item()
            ep_conf += conf_loss.item()
            ep_scale_reg += scale_reg.item()
            ep_iso += iso_loss.item()
            ep_div += div_loss.item()
            ep_cos += page_cos.item()
            ep_active += active_atoms.item()
            ep_std += s_std.item()
            ep_xpass += xpass_cos.item()

            # Page variance diagnostic
            with torch.no_grad():
                pv, dd = log_page_variance(state_pages, device)
                ep_page_var += pv
                ep_dead_dims += dd

            nb += 1

        elapsed = time.time() - t0

        # Gradient norms (from last batch)
        grad_norms = log_atom_grad_norms(model)

        # Eval
        per_cycle_acc, final_acc = evaluate_per_cycle(
            model, eval_dataset, device,
            num_passes=args.num_passes, max_length=max_length,
        )

        # Gradient coupling blends (v24.7)
        hypernet_blend = torch.sigmoid(model.hypernet.blend_logit).item()
        compressor_blend = torch.sigmoid(model.compressor.blend_logit).item()

        improved = False
        if final_acc > best_final:
            best_final = final_acc
            improved = True

        if improved:
            patience_counter = 0
            torch.save({
                'epoch': epoch + 1,
                'compressor': model.compressor.state_dict(),
                'atoms': model.atoms.state_dict(),
                'hypernet': model.hypernet.state_dict(),
                'confidence_head': model.confidence_head.state_dict(),
                'residual_gate': model.residual_gate.state_dict(),
                'message_generator': model.message_generator.state_dict(),
                'ordinal_head': model.ordinal_head.state_dict(),
                'bypass': model.bypass.state_dict(),
                'optimizer': optimizer.state_dict(),
                'accuracy': final_acc,
                'per_cycle_acc': per_cycle_acc,
                'level': level,
                'num_atoms': args.num_atoms,
                'atom_rank': args.atom_rank,
                'hypernet_blend': hypernet_blend,
                'compressor_blend': compressor_blend,
            }, ckpt_name)
            print(f"  -> saved checkpoint {ckpt_name} (final={final_acc:.1f}%)")
        else:
            patience_counter += 1

        print(
            f"Epoch {epoch+1}: "
            f"gen={ep_gen/nb:.4f} contr={ep_ctr/nb:.2f} "
            f"conf={ep_conf/nb:.2f} scale_reg={ep_scale_reg/nb:.2f} iso={ep_iso/nb:.4f} "
            f"div={ep_div/nb:.4f} "
            f"page_cos={ep_cos/nb:.2f} "
            f"active={ep_active/nb:.1f}/{args.num_atoms} "
            f"scale_std={ep_std/nb:.3f} "
            f"xpass_cos={ep_xpass/nb:.2f} | "
            f"Final={final_acc:.1f}% best={best_final:.1f}% [{elapsed:.0f}s]"
        )
        # Per-cycle accuracy breakdown
        print(
            f"  per_cycle_acc: {[f'{a:.1f}%' for a in per_cycle_acc]}"
        )
        # Page variance diagnostic
        print(
            f"  page_var={ep_page_var/nb:.4f} dead_dims={ep_dead_dims/nb:.1f}/64"
        )
        # Gradient coupling blends
        print(
            f"  blends: hypernet={hypernet_blend:.3f} compressor={compressor_blend:.3f} "
            f"(1.0=direct, 0.0=contextual)"
        )
        # Gradient norms
        gn = grad_norms
        print(
            f"  grad norms: "
            f"atoms_A={gn['atoms_A']:.4f} atoms_B={gn['atoms_B']:.4f} "
            f"hypernet={gn['hypernet']:.4f}"
        )

        if patience_counter >= args.patience:
            print(f"\nEarly stopping: no improvement for {args.patience} epochs")
            break

    print(f"\n{'='*60}")
    print(f"Level {level} final: {best_final:.1f}% "
          f"(baseline final={per_cycle_acc[-1] if per_cycle_acc else 0:.1f}%)")
    print(f"Checkpoint: {ckpt_name}")
    print(f"{'='*60}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    p = argparse.ArgumentParser(
        description='Per-cycle generation-only training (no answer head)',
    )
    p.add_argument(
        '--level', type=str, required=True,
        choices=['L3', 'L4', 'L4.5', 'L4.7', 'L4.9', 'L5', 'gsm8k'],
        help='Curriculum level to train',
    )
    p.add_argument('--data_dir', type=str, default='data/per_cycle/',
                   help='Directory containing JSONL data files (default: data/per_cycle/)')
    p.add_argument('--warm_from', type=str, default=None,
                   help='Checkpoint to warm-start from')
    p.add_argument('--epochs', type=int, default=12)
    p.add_argument('--batch_size', type=int, default=8,
                   help='Batch size (default: 8, A10G memory constraint)')
    p.add_argument('--num_passes', type=int, default=3,
                   help='Number of thinking passes per problem')
    p.add_argument('--patience', type=int, default=3)
    p.add_argument('--lam_gen', type=float, default=1.0,
                   help='Per-cycle generation loss weight')
    p.add_argument('--lam_contrastive', type=float, default=0.05,
                   help='Contrastive loss weight')
    p.add_argument('--lam_conf', type=float, default=0.1,
                   help='Confidence loss weight')
    p.add_argument('--lam_scale_reg', type=float, default=0.1,
                   help='Pre-tanh scale regularization weight')
    p.add_argument('--lam_diversity', type=float, default=0.1,
                   help='Scale diversity loss weight (default: 0.1)')
    p.add_argument('--lr_scale', type=float, default=1.0,
                   help='Scale all learning rates (use 0.7 when resuming to protect cycle 1)')
    p.add_argument('--num_atoms', type=int, default=64,
                   help='Number of LoRA atoms')
    p.add_argument('--atom_rank', type=int, default=6,
                   help='Rank of each LoRA atom')
    p.add_argument('--skip_perceiver', action='store_true',
                   help='Skip loading perceiver from warm checkpoint')
    p.add_argument('--augment', action='store_true',
                   help='Enable number augmentation + gen target dropout (anti-memorization)')
    p.add_argument('--dropout_prob', type=float, default=0.15,
                   help='Generation target dropout probability (default: 0.15)')

    args = p.parse_args()
    train(args)
