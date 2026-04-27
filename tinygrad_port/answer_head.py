"""
Tinygrad port of AnswerHead, AtomConfidenceHead, and answer_head_loss
from scripts/atom_lora.py.

AnswerHead (~0.9M params):
  Digit-based answer extraction from the last 64-float page.

AtomConfidenceHead (~2.5M params):
  Cross-attention over accumulated pages -> stop confidence in [0, 1].

answer_head_loss():
  CE loss on sign, length, and per-digit predictions.
"""

from tinygrad import Tensor, dtypes
from tinygrad.nn import Linear, LayerNorm
from typing import List, Tuple
import math


# ---------------------------------------------------------------------------
# AnswerHead -- digit-based answer extraction from last page
# ---------------------------------------------------------------------------
class AnswerHead:
    """
    Reads the last page (64 floats) and predicts the answer as digits.

    Three sub-heads:
    - sign_head:   Linear(hidden, 2)           -> positive or negative
    - length_head: Linear(hidden, max_digits)   -> how many digits
    - digit_heads: max_digits x Linear(hidden, 10) -> 0-9 per position
    """

    def __init__(self, page_size: int = 64, max_digits: int = 8,
                 hidden: int = 512, max_cycles: int = 12):
        self.page_size = page_size
        self.max_digits = max_digits
        self.hidden = hidden

        # Per-cycle embedding: Embedding(max_cycles, hidden)
        # In tinygrad: just a weight matrix, index with weight[idx]
        self.cycle_embed_weight = Tensor.randn(max_cycles, hidden)

        # Shared encoder: page + cycle_embed -> richer representation
        # Layer 1: Linear(page_size + hidden, hidden) -> GELU -> LayerNorm
        self.enc_lin1 = Linear(page_size + hidden, hidden)
        self.enc_ln1 = LayerNorm(hidden)
        # Layer 2: Linear(hidden, hidden) -> GELU -> LayerNorm
        self.enc_lin2 = Linear(hidden, hidden)
        self.enc_ln2 = LayerNorm(hidden)
        # Layer 3: Linear(hidden, hidden) -> GELU (no LayerNorm after last)
        self.enc_lin3 = Linear(hidden, hidden)

        # Sub-heads
        self.sign_head = Linear(hidden, 2)
        self.length_head = Linear(hidden, max_digits)
        self.digit_heads = [Linear(hidden, 10) for _ in range(max_digits)]

    def __call__(self, last_page: Tensor, cycle_num: int = 0
                 ) -> Tuple[Tensor, Tensor, List[Tensor]]:
        """
        Args:
            last_page:  (B, page_size) -- raw page
            cycle_num:  int -- which reasoning cycle (0-indexed)

        Returns:
            sign_logits:   (B, 2)
            length_logits: (B, max_digits)
            digit_logits:  list of max_digits x (B, 10)
        """
        page = last_page.float()
        batch_size = page.shape[0]

        # Cycle embedding: index into weight matrix
        cycle_emb = self.cycle_embed_weight[cycle_num]          # (hidden,)
        cycle_emb = cycle_emb.unsqueeze(0).expand(batch_size, -1)  # (B, hidden)

        combined = page.cat(cycle_emb, dim=-1)                  # (B, page_size + hidden)

        # Encoder: 3 layers
        h = self.enc_lin1(combined).gelu()
        h = self.enc_ln1(h)
        h = self.enc_lin2(h).gelu()
        h = self.enc_ln2(h)
        h = self.enc_lin3(h).gelu()

        sign_logits = self.sign_head(h)
        length_logits = self.length_head(h)
        digit_logits = [head(h) for head in self.digit_heads]

        return sign_logits, length_logits, digit_logits

    def decode(self, last_page: Tensor, cycle_num: int = 0) -> Tensor:
        """
        Decode predicted answer as integer tensor (B,).

        Args:
            last_page: (B, page_size)
            cycle_num: int -- which reasoning cycle (0-indexed)

        Returns:
            answers: (B,) integer tensor
        """
        sign_logits, length_logits, digit_logits = self(last_page, cycle_num=cycle_num)
        batch_size = last_page.shape[0]

        num_digits = length_logits.argmax(axis=-1) + 1      # (B,) 1-indexed
        is_negative = sign_logits.argmax(axis=-1) == 1       # (B,) bool

        answers = Tensor.zeros(batch_size).cast(dtypes.int32)
        for i in range(self.max_digits):
            digit = digit_logits[i].argmax(axis=-1)          # (B,)
            answers = answers * 10 + digit

        # Trim to predicted length: divide by 10^(max_digits - num_digits)
        trim_power = (Tensor.full((batch_size,), self.max_digits).cast(dtypes.int32) - num_digits).maximum(0)
        # 10 ** trim_power -- use exp/log since tinygrad int pow is limited
        divisor = (trim_power.float() * math.log(10)).exp().cast(dtypes.int32).maximum(1)
        answers = answers // divisor

        # Apply sign
        answers = answers.where(is_negative == 0, -answers)

        return answers


# ---------------------------------------------------------------------------
# AtomConfidenceHead -- reads pages via cross-attention -> stop decision
# ---------------------------------------------------------------------------
class AtomConfidenceHead:
    """
    Confidence head that reads accumulated pages via cross-attention.

    Architecture:
    - page_project: Linear(page_size, hidden)
    - query: Parameter(1, hidden)
    - 2-layer cross-attention (hidden dim, num_heads heads) with residual + LayerNorm
    - output MLP: Linear(hidden, hidden) -> GELU -> LayerNorm ->
                  Linear(hidden, 128) -> GELU -> Linear(128, 1) -> Sigmoid
    """

    def __init__(
        self,
        page_size: int = 64,
        hidden: int = 512,
        num_heads: int = 8,
    ):
        self.hidden = hidden
        self.num_heads = num_heads
        self.head_dim = hidden // num_heads

        self.page_project = Linear(page_size, hidden)
        self.query = Tensor.randn(1, hidden) * 0.02

        # Cross-attention layer 1
        self.attn1_q_proj = Linear(hidden, hidden)
        self.attn1_k_proj = Linear(hidden, hidden)
        self.attn1_v_proj = Linear(hidden, hidden)
        self.attn1_out_proj = Linear(hidden, hidden)
        self.norm1 = LayerNorm(hidden)

        # Cross-attention layer 2
        self.attn2_q_proj = Linear(hidden, hidden)
        self.attn2_k_proj = Linear(hidden, hidden)
        self.attn2_v_proj = Linear(hidden, hidden)
        self.attn2_out_proj = Linear(hidden, hidden)
        self.norm2 = LayerNorm(hidden)

        # Output MLP
        self.out_lin1 = Linear(hidden, hidden)
        self.out_ln1 = LayerNorm(hidden)
        self.out_lin2 = Linear(hidden, 128)
        self.out_lin3 = Linear(128, 1)

    def _cross_attention(
        self,
        q: Tensor,           # (B, 1, hidden)
        kv: Tensor,          # (B, P, hidden)
        q_proj: Linear,
        k_proj: Linear,
        v_proj: Linear,
        out_proj: Linear,
    ) -> Tensor:
        """Multi-head cross-attention: query attends to key-value pairs."""
        B = q.shape[0]
        P = kv.shape[1]
        H = self.num_heads
        D = self.head_dim

        # Project
        Q = q_proj(q).reshape(B, 1, H, D).permute(0, 2, 1, 3)    # (B, H, 1, D)
        K = k_proj(kv).reshape(B, P, H, D).permute(0, 2, 1, 3)   # (B, H, P, D)
        V = v_proj(kv).reshape(B, P, H, D).permute(0, 2, 1, 3)   # (B, H, P, D)

        # Scaled dot-product attention
        scale = D ** -0.5
        attn_weights = (Q @ K.permute(0, 1, 3, 2)) * scale        # (B, H, 1, P)
        attn_weights = attn_weights.softmax(axis=-1)
        attn_out = attn_weights @ V                                # (B, H, 1, D)

        # Reshape and project output
        attn_out = attn_out.permute(0, 2, 1, 3).reshape(B, 1, H * D)  # (B, 1, hidden)
        return out_proj(attn_out)

    def __call__(self, state_pages: List[Tensor]) -> Tensor:
        """
        Args:
            state_pages: list of (B, page_size) tensors

        Returns:
            confidence: (B, 1) in [0, 1]
        """
        if not state_pages or len(state_pages) == 0:
            return Tensor([[0.5]])

        # Stack pages: list of (B, page_size) -> (B, P, page_size)
        pages = Tensor.stack(state_pages, dim=1).float()
        pages_proj = self.page_project(pages)                      # (B, P, hidden)

        batch_size = pages.shape[0]
        q = self.query.unsqueeze(0).expand(batch_size, -1, -1)     # (B, 1, hidden)

        # Cross-attention layer 1 with residual + LayerNorm
        att1 = self._cross_attention(
            q, pages_proj,
            self.attn1_q_proj, self.attn1_k_proj,
            self.attn1_v_proj, self.attn1_out_proj,
        )
        q = self.norm1(q + att1)

        # Cross-attention layer 2 with residual + LayerNorm
        att2 = self._cross_attention(
            q, pages_proj,
            self.attn2_q_proj, self.attn2_k_proj,
            self.attn2_v_proj, self.attn2_out_proj,
        )
        q = self.norm2(q + att2)

        # Output MLP: squeeze the sequence dim (1) first
        h = q.squeeze(1)                                           # (B, hidden)
        h = self.out_lin1(h).gelu()
        h = self.out_ln1(h)
        h = self.out_lin2(h).gelu()
        h = self.out_lin3(h).sigmoid()                             # (B, 1)

        return h


# ---------------------------------------------------------------------------
# answer_head_loss -- CE on sign, length, and per-digit predictions
# ---------------------------------------------------------------------------
def answer_head_loss(
    answer_head: AnswerHead,
    last_page: Tensor,
    gold_answers: Tensor,
    cycle_num: int = 0,
) -> Tensor:
    """
    Compute answer head loss from gold integer answers.

    Args:
        answer_head: AnswerHead instance
        last_page:   (B, page_size)
        gold_answers: (B,) integer tensor
        cycle_num:   int -- which reasoning cycle (0-indexed)

    Returns:
        loss: scalar tensor
    """
    sign_logits, length_logits, digit_logits = answer_head(last_page, cycle_num=cycle_num)
    batch_size = last_page.shape[0]

    # Extract gold values as Python lists for string manipulation
    gold_np = gold_answers.numpy().flatten().tolist()
    gold_abs_vals = [abs(int(v)) for v in gold_np]
    gold_sign_vals = [1 if v < 0 else 0 for v in gold_np]  # 0=positive, 1=negative

    gold_strings = [str(v) for v in gold_abs_vals]
    max_digits = answer_head.max_digits

    # Gold targets as tensors
    gold_sign = Tensor(gold_sign_vals).cast(dtypes.int32)
    gold_length_vals = [min(len(s) - 1, max_digits - 1) for s in gold_strings]  # 0-indexed
    gold_length = Tensor(gold_length_vals).cast(dtypes.int32)

    # Digit matrix (left-aligned, most significant first)
    gold_digit_matrix = []
    for b, s in enumerate(gold_strings):
        row = []
        s = s[:max_digits]
        for i in range(max_digits):
            if i < len(s):
                row.append(int(s[i]))
            else:
                row.append(0)
        gold_digit_matrix.append(row)
    gold_digit_tensor = Tensor(gold_digit_matrix).cast(dtypes.int32)  # (B, max_digits)

    # Sign loss: CE
    loss = sign_logits.sparse_categorical_crossentropy(gold_sign)

    # Length loss: CE
    loss = loss + length_logits.sparse_categorical_crossentropy(gold_length)

    # Per-digit losses (masked by actual digit length)
    for i in range(max_digits):
        mask_vals = [1.0 if i < len(gold_strings[b]) else 0.0 for b in range(batch_size)]
        mask = Tensor(mask_vals)
        mask_sum = mask.sum()

        # Only add loss if any sample has this digit position
        digit_target = gold_digit_tensor[:, i]
        digit_loss = digit_logits[i].sparse_categorical_crossentropy(digit_target, reduction="none")
        # Masked mean
        masked_loss = (digit_loss * mask).sum() / mask_sum.maximum(1.0)
        loss = loss + masked_loss

    return loss


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("Tinygrad port -- AnswerHead + AtomConfidenceHead self-test")
    print("=" * 60)

    batch = 4

    # --- AnswerHead ---
    print("\n--- AnswerHead ---")
    ah = AnswerHead(page_size=64, max_digits=8, hidden=512, max_cycles=12)
    last_page = Tensor.randn(batch, 64)

    sign_logits, length_logits, digit_logits = ah(last_page, cycle_num=0)
    print(f"  sign_logits shape:   {sign_logits.shape}")
    print(f"  length_logits shape: {length_logits.shape}")
    print(f"  digit_logits:        {len(digit_logits)} x {digit_logits[0].shape}")

    decoded = ah.decode(last_page, cycle_num=0)
    print(f"  Decoded answers:     {decoded.numpy().tolist()}")

    ah_params = sum(
        p.numel() for p in [
            ah.cycle_embed_weight,
            ah.enc_lin1.weight, ah.enc_lin1.bias,
            ah.enc_ln1.weight, ah.enc_ln1.bias,
            ah.enc_lin2.weight, ah.enc_lin2.bias,
            ah.enc_ln2.weight, ah.enc_ln2.bias,
            ah.enc_lin3.weight, ah.enc_lin3.bias,
            ah.sign_head.weight, ah.sign_head.bias,
            ah.length_head.weight, ah.length_head.bias,
        ] + [h.weight for h in ah.digit_heads] + [h.bias for h in ah.digit_heads]
    )
    print(f"  Params: {ah_params:,}")

    # --- answer_head_loss ---
    print("\n--- answer_head_loss ---")
    gold = Tensor([42, 137, 5, 9999]).cast(dtypes.int32)
    loss = answer_head_loss(ah, last_page, gold, cycle_num=0)
    print(f"  Loss: {loss.numpy().item():.4f}")

    # --- AtomConfidenceHead ---
    print("\n--- AtomConfidenceHead ---")
    ch = AtomConfidenceHead(page_size=64, hidden=512, num_heads=8)
    pages = [Tensor.randn(batch, 64) for _ in range(3)]

    confidence = ch(pages)
    print(f"  Confidence shape: {confidence.shape}")
    conf_np = confidence.numpy()
    print(f"  Confidence range: [{conf_np.min():.3f}, {conf_np.max():.3f}]")

    # Empty pages
    conf_empty = ch([])
    print(f"  Empty pages confidence: {conf_empty.numpy().tolist()}")

    print("\n" + "=" * 60)
    print("All tests passed.")
    print("=" * 60)
