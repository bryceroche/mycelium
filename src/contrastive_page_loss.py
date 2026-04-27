"""
Contrastive page loss — force pages to differ across problems and within problems.

Three loss functions (evolution):

1. `contrastive_page_loss`: margin-based — fragile, knife-edge λ. Historical.

2. `target_cos_page_loss`: bidirectional quadratic attractor. Stable but
   requires choosing a target cosine (arbitrary). Historical.

3. `breathing_contrastive_loss` (current): supervised contrastive (SupCon)
   applied to ALL pages independently + soft quadratic anti-copying penalty
   for within-problem page diversity. No target cosine — temperature controls
   geometry. Two proven failure modes closed:
     - Fixed-point collapse: pages constant across problems (SupCon fixes)
     - Page copying: pages 2-3 identical within a problem (anti-copy fixes)

Intended companion to a generation/answer loss:
    total = gen_loss + lam * breathing_contrastive_loss(all_pages, gold)
"""
import torch
import torch.nn.functional as F


def contrastive_page_loss(
    last_pages: torch.Tensor,
    gold_answers: torch.Tensor,
    margin: float = 0.2,
) -> torch.Tensor:
    """
    last_pages:   (B, D) — float, one vector per problem
    gold_answers: (B,)   — any dtype that supports equality
    margin:       push different-answer pairs to cos sim < margin

    Returns a scalar loss.
    """
    normed = F.normalize(last_pages.float(), dim=-1)          # (B, D)
    sim = normed @ normed.T                                   # (B, B)

    same = (gold_answers.unsqueeze(0) == gold_answers.unsqueeze(1)).float()
    # Ignore diagonal (self-similarity is always 1).
    eye = torch.eye(same.size(0), device=same.device)
    same = same * (1 - eye)
    diff = (1 - same) * (1 - eye)

    # Pull same-answer pairs together (1 - cos).
    pos_loss = (1.0 - sim) * same
    # Push different-answer pairs apart below margin.
    neg_loss = F.relu(sim - margin) * diff

    num_pos = same.sum().clamp(min=1.0)
    num_neg = diff.sum().clamp(min=1.0)
    return pos_loss.sum() / num_pos + neg_loss.sum() / num_neg


def target_cos_page_loss(
    last_pages: torch.Tensor,
    gold_answers: torch.Tensor,
    target_cos: float = 0.4,
) -> torch.Tensor:
    """
    Bidirectional page loss with a fixed-point attractor at cos=target_cos
    for different-answer pairs. Same-answer pairs are pulled together as in
    the margin variant.

    - pos term: (1 - sim) for pairs with matching gold answers
    - neg term: (sim - target_cos)² per diff-answer pair (quadratic, bidirectional)

    A diff pair at cos > target is pushed down; a diff pair at cos < target is
    pulled up. No ReLU, no one-sided gradient. Stable under a small constant λ.
    """
    normed = F.normalize(last_pages.float(), dim=-1)
    sim = normed @ normed.T                                    # (B, B)

    same = (gold_answers.unsqueeze(0) == gold_answers.unsqueeze(1)).float()
    eye = torch.eye(same.size(0), device=same.device)
    same = same * (1 - eye)
    diff = (1 - same) * (1 - eye)

    pos_loss = ((1.0 - sim) * same).sum() / same.sum().clamp(min=1.0)
    neg_loss = (((sim - target_cos) ** 2) * diff).sum() / diff.sum().clamp(min=1.0)
    return pos_loss + neg_loss


def per_page_contrastive_loss(
    all_pages: list,
    gold_answers: torch.Tensor,
    cross_target: float = 0.7,
    within_target: float = 0.3,
) -> torch.Tensor:
    """
    Three-term contrastive loss for multi-page reasoning:

    Term 1+2 (cross-problem): each page should differ across problems with
        different answers. Applied to ALL pages, not just the last one.
        Prevents fixed-point collapse AND static parsing pages.

    Term 3 (within-problem): pages within the same problem should differ from
        each other. Page 1 (parsing) != page 2 (computing) != page 3 (answering).
        Prevents page copying where later passes just repeat earlier ones.

    Args:
        all_pages:    list of num_passes tensors, each (B, 64)
        gold_answers: (B,) integer tensor
        cross_target: target cosine for diff-answer pairs across problems (0.7)
        within_target: target cosine for different pages within same problem (0.3)

    Returns scalar loss.
    """
    loss = torch.tensor(0.0, device=all_pages[0].device)

    # Terms 1 & 2: cross-problem differentiation for each page
    for page in all_pages:
        normed = F.normalize(page.float(), dim=-1)
        sim = normed @ normed.T
        same = (gold_answers.unsqueeze(0) == gold_answers.unsqueeze(1)).float()
        eye = torch.eye(same.size(0), device=sim.device)
        same = same * (1 - eye)
        diff = (1 - same) * (1 - eye)
        pos_loss = ((1.0 - sim) * same).sum() / same.sum().clamp(1)
        neg_loss = (((sim - cross_target) ** 2) * diff).sum() / diff.sum().clamp(1)
        loss = loss + pos_loss + neg_loss

    # Term 3: within-problem page diversity (notebook pages should differ)
    # With appended pages (no blending), push cosine below within_target
    for i in range(len(all_pages)):
        for j in range(i + 1, len(all_pages)):
            page_i = F.normalize(all_pages[i].float(), dim=-1)
            page_j = F.normalize(all_pages[j].float(), dim=-1)
            within_cos = (page_i * page_j).sum(dim=-1)  # (B,)
            # Penalize if pages are too similar — each page should capture different info
            within_loss = F.relu(within_cos - within_target).mean()
            loss = loss + within_loss

    return loss


def supervised_contrastive(
    pages: torch.Tensor,
    gold_answers: torch.Tensor,
    temperature: float = 0.1,
) -> torch.Tensor:
    """
    Standard supervised contrastive (SupCon) loss.
    Positives: same answer. Negatives: different answer.

    pages:        (B, D) — one vector per problem
    gold_answers: (B,)   — integer gold answers
    temperature:  controls cluster tightness (lower = tighter)
    """
    normed = F.normalize(pages.float(), dim=-1)
    sim = normed @ normed.T / temperature  # (B, B)

    self_mask = 1.0 - torch.eye(sim.size(0), device=sim.device)
    same = (gold_answers.unsqueeze(0) == gold_answers.unsqueeze(1)).float()
    same = same * self_mask

    # Log-softmax over each row (numerically stable)
    sim_max = sim.max(dim=-1, keepdim=True).values.detach()
    exp_sim = torch.exp(sim - sim_max) * self_mask
    log_prob = (sim - sim_max) - torch.log(exp_sim.sum(dim=-1, keepdim=True) + 1e-8)

    # Average log probability of positive (same-answer) pairs
    num_positives = same.sum(dim=-1).clamp(min=1)
    pos_log_prob = (log_prob * same).sum(dim=-1) / num_positives

    return -pos_log_prob.mean()


def breathing_contrastive_loss(
    all_pages: list,
    gold_answers: torch.Tensor,
    temperature: float = 0.3,
    anti_copy_threshold: float = 0.7,
):
    """
    Two-term contrastive loss for multi-page breathing.
    Returns (supcon_loss, anti_copy_loss) separately for independent lambdas.

    Term 1 (per-page SupCon): each page independently must differentiate
        across problems. Same answer → pull together. Different answer →
        push apart. No target cosine — temperature controls geometry.

    Term 2 (anti-copying): pages within the same problem must not be
        identical. Free below threshold (default 0.7). Soft quadratic
        penalty above. The model discovers its own within-problem geometry.

    Args:
        all_pages:           list of num_passes tensors, each (B, 64)
        gold_answers:        (B,) integer tensor
        temperature:         SupCon temperature (0.3 = moderate clusters)
        anti_copy_threshold: cosine above this gets penalized (0.7)

    Returns (supcon_loss, anti_copy_loss) — apply separate lambdas.
    """
    supcon_loss = torch.tensor(0.0, device=all_pages[0].device)
    anti_copy_loss = torch.tensor(0.0, device=all_pages[0].device)

    # Term 1: per-page supervised contrastive
    for page in all_pages:
        supcon_loss = supcon_loss + supervised_contrastive(page, gold_answers, temperature)
    supcon_loss = supcon_loss / len(all_pages)

    # Term 2: anti-copying (soft quadratic above threshold)
    n_pairs = 0
    for i in range(len(all_pages)):
        for j in range(i + 1, len(all_pages)):
            page_i = F.normalize(all_pages[i].float(), dim=-1)
            page_j = F.normalize(all_pages[j].float(), dim=-1)
            within_cos = (page_i * page_j).sum(dim=-1)  # (B,)
            anti_copy_loss = anti_copy_loss + (F.relu(within_cos - anti_copy_threshold) ** 2).mean()
            n_pairs += 1
    if n_pairs > 0:
        anti_copy_loss = anti_copy_loss / n_pairs

    return supcon_loss, anti_copy_loss


if __name__ == "__main__":
    torch.manual_seed(0)
    # Sanity: identical pages → pos=0, neg large (if answers differ).
    pages = torch.ones(4, 64)
    golds = torch.tensor([1, 2, 1, 3])
    print("identical pages, mixed answers:",
          contrastive_page_loss(pages, golds).item())
    # Orthogonal per-sample pages → pos large (if answers match), neg=0.
    pages = torch.eye(4, 64)
    golds = torch.tensor([1, 1, 2, 2])
    print("orthogonal pages, matched answers:",
          contrastive_page_loss(pages, golds).item())
    # Random → somewhere in between.
    pages = torch.randn(8, 64)
    golds = torch.tensor([1, 2, 3, 4, 1, 2, 3, 4])
    print("random pages (margin):",
          contrastive_page_loss(pages, golds).item())
    print("random pages (target_cos=0.4):",
          target_cos_page_loss(pages, golds).item())
    # Sanity: pages spread around target should give low loss.
    torch.manual_seed(0)
    pages = torch.randn(16, 64)
    golds = torch.arange(16)  # all different
    print("all-diff random (target_cos=0.4):",
          target_cos_page_loss(pages, golds).item())
