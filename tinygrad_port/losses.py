"""
Loss functions for per-cycle answer head training, ported to tinygrad.

All losses operate on tinygrad Tensors and return scalar Tensors.
"""
from tinygrad import Tensor, dtypes
import math


# ---------------------------------------------------------------------------
# Answer head loss (per-digit classification)
# ---------------------------------------------------------------------------

def answer_head_loss(answer_head, page, gold_answers, cycle_num=0):
    """Per-digit classification loss for the AnswerHead.

    Ported from atom_lora.py answer_head_loss().

    Args:
        answer_head: AnswerHead module (tinygrad)
        page:        Tensor (B, page_size) -- raw page floats
        gold_answers: list of int or Tensor (B,) -- integer gold answers
        cycle_num:   int -- which reasoning cycle (0-indexed)

    Returns:
        loss: scalar Tensor
    """
    sign_logits, length_logits, digit_logits = answer_head(page, cycle_num=cycle_num)
    batch_size = page.shape[0]
    max_digits = answer_head.max_digits

    # Convert gold_answers to python list if needed
    if isinstance(gold_answers, Tensor):
        gold_list = gold_answers.numpy().tolist()
    else:
        gold_list = list(gold_answers)

    gold_abs = [abs(int(v)) for v in gold_list]
    gold_sign = [1 if int(v) < 0 else 0 for v in gold_list]
    gold_strings = [str(v) for v in gold_abs]

    # Gold length (0-indexed for CE: "72" has 2 digits -> index 1)
    gold_length = [min(len(s) - 1, max_digits - 1) for s in gold_strings]

    # Gold digit matrix (left-aligned, most significant first)
    gold_digit_matrix = []
    for s in gold_strings:
        s = s[:max_digits]
        row = [int(ch) for ch in s] + [0] * (max_digits - len(s))
        gold_digit_matrix.append(row)

    # Sign loss
    sign_target = Tensor(gold_sign, dtype=dtypes.int32)
    loss = sign_logits.log_softmax(axis=-1).gather(sign_target.unsqueeze(-1), dim=-1).squeeze(-1).mean().neg()

    # Length loss
    length_target = Tensor(gold_length, dtype=dtypes.int32)
    loss = loss + length_logits.log_softmax(axis=-1).gather(length_target.unsqueeze(-1), dim=-1).squeeze(-1).mean().neg()

    # Per-digit losses (masked by actual digit count)
    for i in range(max_digits):
        mask_vals = [1.0 if i < len(gold_strings[b]) else 0.0 for b in range(batch_size)]
        mask_sum = sum(mask_vals)
        if mask_sum > 0:
            digit_target = Tensor([gold_digit_matrix[b][i] for b in range(batch_size)], dtype=dtypes.int32)
            mask_t = Tensor(mask_vals)
            # Cross entropy per sample
            log_probs = digit_logits[i].log_softmax(axis=-1)
            per_sample = log_probs.gather(digit_target.unsqueeze(-1), dim=-1).squeeze(-1).neg()
            loss = loss + (per_sample * mask_t).sum() / mask_sum

    return loss


# ---------------------------------------------------------------------------
# Generation loss (teacher-forced CE)
# ---------------------------------------------------------------------------

def generation_loss(logits, target_ids, target_mask, prompt_len):
    """Teacher-forced generation cross-entropy.

    Ported from the gen loss computation in train_per_cycle.py forward_train_per_cycle().

    Args:
        logits:      Tensor (B, prompt_len + target_len, vocab) -- full model output
        target_ids:  Tensor (B, T) -- token ids for the target
        target_mask: Tensor (B, T) -- 1.0 where target token is real, 0.0 for padding
        prompt_len:  int -- length of the prompt portion

    Returns:
        loss: scalar Tensor -- masked cross-entropy over target tokens only
    """
    batch_size = target_ids.shape[0]
    target_len = target_ids.shape[1]

    # Slice logits to align: position prompt_len-1 predicts first target token
    gen_logits = logits[:, prompt_len - 1:prompt_len - 1 + target_len, :]  # (B, T, V)

    # Flatten for cross-entropy
    vocab_size = gen_logits.shape[-1]
    flat_logits = gen_logits.reshape(-1, vocab_size)  # (B*T, V)
    flat_targets = target_ids.reshape(-1)  # (B*T,)

    # Per-token cross-entropy
    log_probs = flat_logits.log_softmax(axis=-1)
    per_token_loss = log_probs.gather(flat_targets.unsqueeze(-1), dim=-1).squeeze(-1).neg()  # (B*T,)
    per_token_loss = per_token_loss.reshape(batch_size, target_len)  # (B, T)

    # Mask and average per sample, then across batch
    per_sample = (per_token_loss * target_mask).sum(axis=1) / target_mask.sum(axis=1).maximum(Tensor(1.0))
    return per_sample.mean()


# ---------------------------------------------------------------------------
# Per-page contrastive loss (cross-problem + within-problem diversity)
# ---------------------------------------------------------------------------

def per_page_contrastive_loss(pages, gold_answers, cross_target=0.7, within_target=0.3):
    """Three-term contrastive loss for multi-page reasoning.

    Ported from contrastive_page_loss.py per_page_contrastive_loss().

    Term 1+2 (cross-problem): each page should differ across problems with
        different answers. Applied to ALL pages.
    Term 3 (within-problem): pages within same problem should differ.

    Args:
        pages:        list of Tensor, each (B, 64)
        gold_answers: list of int or Tensor (B,) -- integer gold answers
        cross_target: float, target cosine for diff-answer pairs (0.7)
        within_target: float, target cosine for within-problem pages (0.3)

    Returns:
        loss: scalar Tensor
    """
    if isinstance(gold_answers, Tensor):
        ga_np = gold_answers.numpy().tolist()
    else:
        ga_np = list(gold_answers)

    B = pages[0].shape[0]

    # Build same/diff masks as tensors
    same_vals = [[1.0 if ga_np[i] == ga_np[j] else 0.0 for j in range(B)] for i in range(B)]
    eye_vals = [[1.0 if i == j else 0.0 for j in range(B)] for i in range(B)]
    same_t = Tensor(same_vals)
    eye_t = Tensor(eye_vals)
    same_no_diag = same_t * (Tensor(1.0) - eye_t)
    diff_no_diag = (Tensor(1.0) - same_t) * (Tensor(1.0) - eye_t)

    loss = Tensor(0.0)

    # Terms 1 & 2: cross-problem differentiation for each page
    for page in pages:
        page_f = page.float()
        normed = page_f / (page_f.square().sum(axis=-1, keepdim=True).sqrt() + 1e-8)
        sim = normed.matmul(normed.T)  # (B, B)

        same_count = same_no_diag.sum().maximum(Tensor(1.0))
        diff_count = diff_no_diag.sum().maximum(Tensor(1.0))

        pos_loss = ((Tensor(1.0) - sim) * same_no_diag).sum() / same_count
        neg_loss = (((sim - cross_target) ** 2) * diff_no_diag).sum() / diff_count
        loss = loss + pos_loss + neg_loss

    # Term 3: within-problem page diversity
    for i in range(len(pages)):
        for j in range(i + 1, len(pages)):
            pi = pages[i].float()
            pj = pages[j].float()
            pi_n = pi / (pi.square().sum(axis=-1, keepdim=True).sqrt() + 1e-8)
            pj_n = pj / (pj.square().sum(axis=-1, keepdim=True).sqrt() + 1e-8)
            within_cos = (pi_n * pj_n).sum(axis=-1)  # (B,)
            within_loss = (within_cos - within_target).relu().mean()
            loss = loss + within_loss

    return loss


# ---------------------------------------------------------------------------
# Isotropic regularizer (raw pages before normalization)
# ---------------------------------------------------------------------------

def isotropic_regularizer(raw_pages, target_var=1.0, corr_weight=0.1):
    """Force isotropic Gaussian distribution on raw pages.

    Ported from atom_lora.py IsotropicRegularizer.

    Args:
        raw_pages:   Tensor (N, 64) -- raw perceiver output before normalization
        target_var:  float, target per-dimension variance (1.0)
        corr_weight: float, weight on off-diagonal correlation penalty (0.1)

    Returns:
        loss: scalar Tensor
    """
    if raw_pages.shape[0] < 4:
        return Tensor(0.0)

    dim_means = raw_pages.mean(axis=0)  # (64,)
    mean_loss = (dim_means ** 2).mean()

    dim_vars = raw_pages.var(axis=0)  # (64,)
    var_loss = ((dim_vars - target_var) ** 2).mean()

    # Off-diagonal correlation penalty
    normalized = (raw_pages - dim_means) / (dim_vars.sqrt() + 1e-8)
    batch_size = raw_pages.shape[0]
    correlation = normalized.T.matmul(normalized) / batch_size  # (64, 64)

    page_dim = raw_pages.shape[1]
    identity = Tensor.eye(page_dim)
    off_diagonal = correlation - identity
    corr_loss = (off_diagonal ** 2).mean()

    return mean_loss + var_loss + corr_weight * corr_loss


# ---------------------------------------------------------------------------
# Confidence + entropy loss
# ---------------------------------------------------------------------------

def confidence_entropy_loss(confidence_head, answer_head, state_pages, finals_t):
    """Per-cycle correctness signal + entropy regularization on exit distribution.

    Ported from train_per_cycle.py confidence head section.

    Args:
        confidence_head: ConfidenceHead module (tinygrad) -- pages -> (B, 1) sigmoid
        answer_head:     AnswerHead module (tinygrad) -- page -> digit prediction
        state_pages:     list of Tensor, each (B, page_size)
        finals_t:        list of int or Tensor (B,) -- final gold answers

    Returns:
        conf_loss: scalar Tensor
    """
    if isinstance(finals_t, Tensor):
        finals_list = finals_t.numpy().tolist()
    else:
        finals_list = list(finals_t)

    conf_loss = Tensor(0.0)
    exit_probs = []

    for pg_idx, pg in enumerate(state_pages):
        conf_pred = confidence_head(state_pages[:pg_idx + 1])  # (B, 1)
        exit_probs.append(conf_pred.mean())

        # Target: 1.0 if answer_head.decode matches gold, else 0.0
        # Decode is non-differentiable, compute outside graph
        with Tensor.no_grad:
            preds = answer_head.decode(pg.float(), cycle_num=pg_idx)
            pred_list = preds.numpy().tolist()
            target_vals = [[1.0 if int(pred_list[b]) == int(finals_list[b]) else 0.0]
                           for b in range(pg.shape[0])]
        target = Tensor(target_vals)  # (B, 1)

        # Binary cross-entropy: -[t*log(p) + (1-t)*log(1-p)]
        p = conf_pred.clip(1e-7, 1.0 - 1e-7)
        bce = -(target * p.log() + (Tensor(1.0) - target) * (Tensor(1.0) - p).log())
        conf_loss = conf_loss + bce.mean()

    conf_loss = conf_loss / max(len(state_pages), 1)

    # Entropy regularization: prevent collapse to "always stop at cycle N"
    if len(exit_probs) > 1:
        exit_stack = Tensor.stack(*exit_probs)
        exit_dist = exit_stack / (exit_stack.sum() + 1e-8)
        entropy = -(exit_dist * (exit_dist + 1e-8).log()).sum()
        conf_loss = conf_loss - 0.01 * entropy  # maximize entropy

    return conf_loss


# ---------------------------------------------------------------------------
# Per-cycle target weight (smooth sigmoid fading)
# ---------------------------------------------------------------------------

def per_cycle_target_weight(final_accuracy, cycle, total_cycles):
    """Smoothly fade intermediate targets as accuracy climbs.

    Ported from train_per_cycle.py per_cycle_target_weight().

    Final cycle: always 1.0.
    Intermediate cycles: sigmoid fade centered at 80%.
    Dormant below 70%. Fully faded above 90%.

    Args:
        final_accuracy: float, current best final accuracy in [0, 1]
        cycle:          int, current cycle index (0-indexed)
        total_cycles:   int, total number of supervised cycles

    Returns:
        weight: float in [0, 1]
    """
    if cycle == total_cycles - 1:
        return 1.0  # final cycle always fully supervised
    # Sigmoid centered at 0.80, steepness 15
    x = (final_accuracy - 0.80) * 15.0
    fade = 1.0 / (1.0 + math.exp(-x))
    return 1.0 - fade
