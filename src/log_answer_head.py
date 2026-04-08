"""
LogAnswerHead: read the final numeric answer directly out of the last page.

Two tiny linear heads on the 64-float final page:
  - log_mag_head:  predicts log10(|answer| + 1)
  - sign_head:     predicts P(answer >= 0)  via sigmoid

Decoded answer:
  sign      = +1 if sign_prob > 0.5 else -1
  magnitude = 10^log_mag - 1
  answer    = sign * magnitude   (rounded if gold was integer)

Why log space:
  GSM8K answers range from 1 to ~1M. Raw MSE explodes (we saw 1.26B).
  log10 flattens: 1 → 0, 100 → 2, 1_000_000 → 6. Bounded, learnable.
  The "+1" keeps log(0) safe and smooths the low end.

Why separate sign:
  Most GSM8K answers are positive, but "how much did X lose" problems can
  produce negatives. log10(|x|) throws away sign, so we recover it with a
  tiny binary head.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LogAnswerHead(nn.Module):
    def __init__(self, page_size: int = 64):
        super().__init__()
        self.log_mag_head = nn.Linear(page_size, 1)
        self.sign_head = nn.Linear(page_size, 1)

        # Small init — let the page drive the prediction, not the bias
        nn.init.normal_(self.log_mag_head.weight, std=0.01)
        nn.init.zeros_(self.log_mag_head.bias)
        nn.init.normal_(self.sign_head.weight, std=0.01)
        nn.init.zeros_(self.sign_head.bias)

    def forward(self, last_page: torch.Tensor):
        """
        last_page: (batch, 64)
        returns: (log_mag, sign_logit) each (batch,)
        """
        log_mag = self.log_mag_head(last_page).squeeze(-1)
        sign_logit = self.sign_head(last_page).squeeze(-1)
        return log_mag, sign_logit

    @staticmethod
    def target_log_mag(gold: torch.Tensor) -> torch.Tensor:
        """log10(|gold| + 1) — smooth, zero-safe."""
        return torch.log10(gold.abs() + 1.0)

    @staticmethod
    def target_sign(gold: torch.Tensor) -> torch.Tensor:
        """1.0 if gold >= 0 else 0.0."""
        return (gold >= 0).float()

    def compute_loss(self, last_page: torch.Tensor, gold: torch.Tensor,
                     sign_weight: float = 0.1):
        log_mag, sign_logit = self.forward(last_page)
        mag_loss = F.mse_loss(log_mag, self.target_log_mag(gold))
        sign_loss = F.binary_cross_entropy_with_logits(
            sign_logit, self.target_sign(gold),
        )
        return mag_loss + sign_weight * sign_loss, mag_loss.item(), sign_loss.item()

    def decode(self, last_page: torch.Tensor) -> torch.Tensor:
        """
        Returns predicted answer as float tensor. Caller rounds if gold was int.
        """
        log_mag, sign_logit = self.forward(last_page)
        mag = torch.pow(10.0, log_mag) - 1.0
        sign = torch.where(torch.sigmoid(sign_logit) >= 0.5,
                           torch.ones_like(mag), -torch.ones_like(mag))
        return sign * mag


if __name__ == '__main__':
    h = LogAnswerHead()
    page = torch.randn(4, 64)
    gold = torch.tensor([72.0, 1_000_000.0, -5.0, 0.5])
    loss, ml, sl = h.compute_loss(page, gold)
    pred = h.decode(page)
    print(f"loss={loss.item():.4f} mag={ml:.4f} sign={sl:.4f}")
    print(f"pred={pred.tolist()}")
    print("OK")
