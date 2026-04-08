"""
Contrastive page loss — force last pages to differ across problems.

Two variants:

1. `contrastive_page_loss`: margin-based — pull same-answer together, push
   diff-answer below a cosine margin. Strong kick but one-sided — it only
   penalizes diff pairs that are TOO similar, nothing stops them going
   orthogonal. Fragile: λ=0.3 overshoots, λ=0.05 undershoots, sweet spot
   is a knife edge.

2. `target_cos_page_loss` (preferred): bidirectional — pull same-answer
   together AND pull diff-answer cosine TOWARD a target (default 0.4),
   per-pair quadratic. Fixed-point attractor at cos=target, self-stabilizing
   in both directions, can't collapse or over-separate. Use a small constant
   λ (~0.05); no schedule needed.

Intended companion to a generation/answer loss:
    total = gen_loss + lam * target_cos_page_loss(last_page, gold)
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
