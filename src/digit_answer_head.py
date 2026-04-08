"""
DigitAnswerHead: categorical per-digit readout from the last page.

Answer in [0, 999] is decomposed into length ∈ {1,2,3} + up to 3 digits.
Each head is a softmax classification (cross-entropy), so the head
*structurally cannot* collapse to the dataset mean the way a regression
MSE head can.

  "72"  → length=2, digits=[7, 2, _]  (third digit ignored)
  "5"   → length=1, digits=[5, _, _]
  "199" → length=3, digits=[1, 9, 9]

Total params (page_size=64):
  length_head: 64 * 3 + 3 = 195
  3 digit heads: 3 * (64 * 10 + 10) = 1950
  total: 2145 (vs 130 for log-head — still trivial)

For answers ≥ 1000 or negative, caller should skip / clip (curriculum
levels are bounded to [1, 200], so this is safe there).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


NUM_DIGITS = 3  # supports 0-999


class DigitAnswerHead(nn.Module):
    def __init__(self, page_size: int = 64, num_digits: int = NUM_DIGITS):
        super().__init__()
        self.num_digits = num_digits
        self.length_head = nn.Linear(page_size, num_digits)  # 1..num_digits
        self.digit_heads = nn.ModuleList([
            nn.Linear(page_size, 10) for _ in range(num_digits)
        ])
        for m in [self.length_head, *self.digit_heads]:
            nn.init.normal_(m.weight, std=0.01)
            nn.init.zeros_(m.bias)

    def forward(self, last_page: torch.Tensor):
        length_logits = self.length_head(last_page)              # (B, num_digits)
        digit_logits = torch.stack(
            [h(last_page) for h in self.digit_heads], dim=1,
        )                                                        # (B, num_digits, 10)
        return length_logits, digit_logits

    @staticmethod
    def encode(gold: torch.Tensor, num_digits: int = NUM_DIGITS):
        """
        gold: (B,) non-negative integers < 10**num_digits
        returns:
          length_idx: (B,) in [0, num_digits-1]  (length - 1)
          digit_idx:  (B, num_digits) in [0, 9], right-padded MSD-first
          mask:       (B, num_digits) bool — True where digit is meaningful
        """
        gold = gold.long().clamp(min=0, max=10**num_digits - 1)
        lengths = torch.where(
            gold == 0,
            torch.ones_like(gold),
            torch.floor(torch.log10(gold.float().clamp(min=1)) + 1).long(),
        ).clamp(max=num_digits)
        length_idx = (lengths - 1).clamp(min=0)

        digit_idx = torch.zeros(gold.size(0), num_digits, dtype=torch.long, device=gold.device)
        mask = torch.zeros(gold.size(0), num_digits, dtype=torch.bool, device=gold.device)
        for b in range(gold.size(0)):
            L = int(lengths[b].item())
            s = str(int(gold[b].item())).zfill(L)[-L:]
            for i, ch in enumerate(s):
                digit_idx[b, i] = int(ch)
                mask[b, i] = True
        return length_idx, digit_idx, mask

    def compute_loss(self, last_page: torch.Tensor, gold: torch.Tensor):
        length_logits, digit_logits = self.forward(last_page)
        length_idx, digit_idx, mask = self.encode(gold, self.num_digits)

        length_loss = F.cross_entropy(length_logits, length_idx)

        # Per-digit CE, only where mask is True
        # Flatten: (B*num_digits, 10) and (B*num_digits,)
        flat_logits = digit_logits.reshape(-1, 10)
        flat_targets = digit_idx.reshape(-1)
        flat_mask = mask.reshape(-1)
        if flat_mask.any():
            digit_loss = F.cross_entropy(
                flat_logits[flat_mask], flat_targets[flat_mask],
            )
        else:
            digit_loss = torch.zeros((), device=last_page.device)

        total = length_loss + digit_loss
        return total, length_loss.item(), digit_loss.item()

    def decode(self, last_page: torch.Tensor) -> torch.Tensor:
        """Greedy decode → (B,) integer tensor."""
        length_logits, digit_logits = self.forward(last_page)
        lengths = length_logits.argmax(dim=-1) + 1                 # (B,)
        digits = digit_logits.argmax(dim=-1)                       # (B, num_digits)
        out = torch.zeros(last_page.size(0), dtype=torch.long, device=last_page.device)
        for b in range(last_page.size(0)):
            L = int(lengths[b].item())
            val = 0
            for i in range(L):
                val = val * 10 + int(digits[b, i].item())
            out[b] = val
        return out


if __name__ == "__main__":
    h = DigitAnswerHead()
    page = torch.randn(4, 64)
    gold = torch.tensor([72.0, 199.0, 5.0, 0.0])
    loss, ll, dl = h.compute_loss(page, gold)
    pred = h.decode(page)
    print(f"loss={loss.item():.4f} len={ll:.4f} digit={dl:.4f}")
    print(f"pred={pred.tolist()} gold={gold.long().tolist()}")
    print("OK")
