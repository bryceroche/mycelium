"""Phase 0 training step + evaluation + greedy sampling.

The breathing transformer trains by next-token CE on the post-final-LN integrated
representation. We never generate tokens between breaths (Copy Machine Principle):
sampling does N internal breaths and emits ONE token, then repeats with the
extended prefix.
"""
from typing import List
import numpy as np
from tinygrad import Tensor, Device, dtypes


def forward_loss(model, tokens: Tensor, n_loops: int) -> Tensor:
    """Next-token CE on the integrated representation. Tokens (B, S) int.

    Returns scalar loss tensor. Caller must run .backward() BEFORE reading the
    value (tinygrad consumes the lazy graph on .realize()/.numpy()).
    """
    h = model(tokens, n_loops)                       # (B, S, hidden) — post final LN
    logits = (h @ model.embed_out).cast(dtypes.float)  # (B, S, vocab) FP32 for stable CE
    pred = logits[:, :-1, :]
    targ = tokens[:, 1:]
    return pred.sparse_categorical_crossentropy(targ, reduction="mean")


def train_step(model, opt, tokens: Tensor, n_loops: int) -> float:
    Tensor.training = True
    opt.zero_grad()
    loss = forward_loss(model, tokens, n_loops)
    loss.backward()
    opt.step()
    Device[Device.DEFAULT].synchronize()
    return float(loss.numpy())


def eval_loss(model, tokens: Tensor, n_loops: int) -> float:
    Tensor.training = False
    loss = forward_loss(model, tokens, n_loops)
    return float(loss.realize().numpy())


def sample_text(model, prompt_ids: List[int], n_new_tokens: int, n_loops: int,
                temperature: float = 0.0, vocab_active: int = 50277,
                rng: np.random.Generator | None = None) -> List[int]:
    """Generate n_new_tokens via repeated full-breath forwards.

    temperature=0 -> greedy. The model breathes n_loops times per emitted token,
    using the full prefix each time. KV cache cannot be reused across emissions
    because π-cycled RoPE rotates the attention geometry every loop.
    """
    Tensor.training = False
    out = list(prompt_ids)
    if rng is None and temperature > 0:
        rng = np.random.default_rng()

    for _ in range(n_new_tokens):
        # Truncate context to model's max_seq_len if needed.
        ctx = out[-model.cfg.max_seq_len :]
        tokens = Tensor([ctx], dtype=dtypes.int).realize()
        h = model(tokens, n_loops)                                # (1, S, hidden)
        last = h[:, -1, :]                                        # (1, hidden)
        logits = (last @ model.embed_out).cast(dtypes.float)      # (1, vocab)
        # Mask tokens above the actual active vocab (Pythia has padded vocab 50304 vs active 50277)
        logits = logits[:, :vocab_active]
        if temperature == 0.0:
            next_id = int(logits.argmax(axis=-1).realize().numpy()[0])
        else:
            probs = (logits / temperature).softmax(-1).realize().numpy()[0]
            next_id = int(rng.choice(probs.shape[0], p=probs / probs.sum()))
        out.append(next_id)

    return out[len(prompt_ids):]
