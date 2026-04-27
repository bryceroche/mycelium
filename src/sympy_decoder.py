"""
SymPy Decoder v3 — 8 Frequency Band Tokens + Scheduled Sampling (v24.9)

A dedicated decoder that reads 64-float pages as 8 FREQUENCY BAND tokens
(8 floats per band) and emits SymPy expressions.

v2 problem: 64 single-float tokens, but diagnostic showed 68% of variance
in 10 dims. 54 of 64 tokens were noise drowning the signal.

v3 fix: 8 bands of 8 floats each. Each band is a frequency band aligned
with pi-harmonic encoding. Linear(8, 256) has 8x richer input signal per
token. Decoder cross-attends over 8 meaningful bands, not 64 thin tokens.

Architecture:
  - Tiny math vocabulary (~50 tokens): digits, operators, variables, assignment
  - 3-layer transformer decoder, d_model=256, 4 heads
  - Cross-attends over 8 frequency band tokens + optional previous results
  - Scheduled sampling: gradually replace teacher tokens with own predictions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union


class SymPyVocab:
    """
    Tiny vocabulary for mathematical expressions.
    ~50 tokens total. NOT the LLM's 128K vocabulary.
    """

    def __init__(self):
        self.tokens = [
            # Special
            '<bos>', '<eos>', '<pad>',

            # Digits
            '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',

            # Decimal point
            '.',

            # Operators
            '+', '-', '*', '/', '//', '%', '**',

            # Parentheses
            '(', ')',

            # Assignment
            '=',

            # Separators
            ';', ',',

            # Variables (up to 12 intermediates)
            'v1', 'v2', 'v3', 'v4', 'v5', 'v6',
            'v7', 'v8', 'v9', 'v10', 'v11', 'v12',

            # Special variable
            'answer',

            # Common functions
            'abs', 'max', 'min', 'round',

            # Common constants
            'Rational',
        ]

        self.token_to_id = {t: i for i, t in enumerate(self.tokens)}
        self.id_to_token = {i: t for i, t in enumerate(self.tokens)}

        self.bos_id = self.token_to_id['<bos>']
        self.eos_id = self.token_to_id['<eos>']
        self.pad_id = self.token_to_id['<pad>']

    def __len__(self):
        return len(self.tokens)

    def encode(self, expression_str: str) -> List[int]:
        """
        Tokenize a SymPy expression string.

        "v1 = 48; v2 = v1 / 2" → [v1, =, 4, 8, ;, v2, =, v1, /, 2]

        Multi-character tokens are matched greedily (longest first).
        """
        tokens = []
        i = 0
        expr = expression_str.strip()

        while i < len(expr):
            # Skip whitespace
            if expr[i] == ' ':
                i += 1
                continue

            # Try multi-char tokens first (longest match: 'answer'=6, 'Rational'=8)
            matched = False
            for length in [8, 6, 5, 3, 2]:
                candidate = expr[i:i + length]
                if candidate in self.token_to_id:
                    tokens.append(self.token_to_id[candidate])
                    i += length
                    matched = True
                    break

            if not matched:
                # Single character token
                char = expr[i]
                if char in self.token_to_id:
                    tokens.append(self.token_to_id[char])
                i += 1

        return tokens

    def encode_with_bos_eos(self, expression_str: str) -> List[int]:
        """Encode with BOS/EOS tokens for teacher forcing."""
        return [self.bos_id] + self.encode(expression_str) + [self.eos_id]

    def decode(self, token_ids: List[int]) -> str:
        """Convert token IDs back to a SymPy expression string."""
        tokens = [
            self.id_to_token.get(t, '?') for t in token_ids
            if t not in (self.bos_id, self.eos_id, self.pad_id)
        ]

        # Join with smart spacing
        result = []
        for t in tokens:
            if t in ('=', '+', '-', '*', '/', '//', '**', '%', ';'):
                result.append(f' {t} ')
            else:
                result.append(t)

        return ''.join(result).strip()

    def batch_encode(self, expressions: List[str], device: torch.device,
                     max_len: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Batch encode expressions with padding.

        Returns:
            token_ids: (batch, max_seq) padded token IDs (with BOS/EOS)
            mask: (batch, max_seq) attention mask (1 = real, 0 = pad)
        """
        encoded = [self.encode_with_bos_eos(e) for e in expressions]
        lengths = [len(e) for e in encoded]

        if max_len is None:
            max_len = max(lengths)

        token_ids = torch.full((len(encoded), max_len), self.pad_id,
                               dtype=torch.long, device=device)
        mask = torch.zeros(len(encoded), max_len, dtype=torch.bool, device=device)

        for i, (enc, length) in enumerate(zip(encoded, lengths)):
            actual_len = min(length, max_len)
            token_ids[i, :actual_len] = torch.tensor(enc[:actual_len], dtype=torch.long)
            mask[i, :actual_len] = True

        return token_ids, mask


class SymPyDecoder(nn.Module):
    """
    v3: Decoder reads page as 8 FREQUENCY BAND tokens (8 floats each).

    Diagnostic showed 68% of page variance is in 10 dims, 54 dims are constant.
    With 64 single-float tokens, 54 are noise drowning 10 signals.
    With 8 bands of 8 floats each, every band carries meaningful signal
    through its Linear(8, 256) projection.

    Aligned with pi-harmonic page encoding:
      Band 0 (dims 0-7):   lowest frequency — problem type, category
      Band 1 (dims 8-15):  low freq — magnitude, structure
      Band 2 (dims 16-23): mid-low — key quantities
      Band 3 (dims 24-31): mid — relationships
      Band 4 (dims 32-39): mid-high — operations
      Band 5 (dims 40-47): high — specific numbers
      Band 6 (dims 48-55): higher — exact values
      Band 7 (dims 56-63): highest — corrections, precision

    3-layer transformer decoder, 256 dim.
    Includes scheduled sampling to bridge the teacher forcing gap.
    """

    def __init__(self, page_size: int = 64, d_model: int = 256, nhead: int = 4,
                 num_layers: int = 3, max_tokens: int = 40, dropout: float = 0.1,
                 num_bands: int = 8):
        super().__init__()
        self.d_model = d_model
        self.max_tokens = max_tokens
        self.page_size = page_size
        self.num_bands = num_bands
        self.band_size = page_size // num_bands  # 8 floats per band

        # Math-specific vocabulary
        self.vocab = SymPyVocab()
        vocab_size = len(self.vocab)

        # Token embedding (for decoder's own tokens)
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_tokens, d_model)

        # === PAGE AS 8 FREQUENCY BANDS (v3 key change) ===
        # Each band is 8 floats → d_model (8x richer input than single-float tokens)
        self.band_embed = nn.Sequential(
            nn.Linear(self.band_size, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.band_pos = nn.Embedding(num_bands, d_model)
        self.page_norm = nn.LayerNorm(d_model)

        # Project accumulated SymPy results as additional memory token
        self.result_project = nn.Sequential(
            nn.Linear(page_size, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

        # 3-layer transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        # Output projection
        self.output = nn.Linear(d_model, vocab_size)

        # Layer norm
        self.norm = nn.LayerNorm(d_model)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.embed.weight)
        nn.init.xavier_uniform_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def _build_memory(self, page: torch.Tensor,
                      result_embedding: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Convert page to 8 frequency band tokens for cross-attention.

        page: (batch, 64) → reshape to (batch, 8, 8) → project to (batch, 8, d_model)
        result_embedding: (batch, 64) → (batch, 1, d_model)

        Returns: (batch, 8 or 9, d_model)
        """
        device = page.device

        # Split page into 8 frequency bands of 8 floats each
        bands = page.view(page.size(0), self.num_bands, self.band_size)  # (batch, 8, 8)
        band_tokens = self.band_embed(bands)  # (batch, 8, d_model)
        band_tokens = band_tokens + self.band_pos(
            torch.arange(self.num_bands, device=device)
        )
        band_tokens = self.page_norm(band_tokens)

        if result_embedding is not None:
            result_mem = self.result_project(result_embedding).unsqueeze(1)  # (batch, 1, d_model)
            memory = torch.cat([band_tokens, result_mem], dim=1)  # (batch, 9, d_model)
        else:
            memory = band_tokens  # (batch, 8, d_model)

        return memory

    def forward(self, page: torch.Tensor,
                result_embedding: Optional[torch.Tensor] = None,
                target_tokens: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass for training (teacher forcing) or inference.

        Args:
            page: (batch, 64) — current page from perceiver
            result_embedding: (batch, 64) — encoded previous SymPy results (optional)
            target_tokens: (batch, seq) — teacher-forced target IDs including BOS (training only)

        Returns:
            logits: (batch, seq-1, vocab_size) if teacher forcing
            List[str] if inference
        """
        memory = self._build_memory(page, result_embedding)

        if target_tokens is not None:
            # Teacher forcing — shift right: input is target[:-1], labels are target[1:]
            tgt_input = target_tokens[:, :-1]  # (batch, seq-1)
            seq_len = tgt_input.size(1)

            positions = torch.arange(seq_len, device=page.device)
            tgt = self.embed(tgt_input) + self.pos_embed(positions)
            tgt = self.norm(tgt)

            causal_mask = nn.Transformer.generate_square_subsequent_mask(
                seq_len, device=page.device
            )
            tgt_key_padding_mask = (tgt_input == self.vocab.pad_id)

            out = self.decoder(
                tgt, memory,
                tgt_mask=causal_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
            )
            return self.output(out)  # (batch, seq-1, vocab_size)
        else:
            return self._generate(memory, page.device)

    def forward_scheduled_sampling(self, page: torch.Tensor,
                                   target_tokens: torch.Tensor,
                                   result_embedding: Optional[torch.Tensor] = None,
                                   sample_rate: float = 0.0) -> torch.Tensor:
        """
        Scheduled sampling: with probability sample_rate, use decoder's own
        prediction instead of teacher token. Bridges train/inference gap.

        sample_rate=0.0: pure teacher forcing (early training)
        sample_rate=0.2: 80% teacher, 20% self (mid training)
        sample_rate=0.5: half and half (late training)

        Args:
            page: (batch, 64)
            target_tokens: (batch, seq) — full sequence including BOS and EOS
            result_embedding: (batch, 64) — optional
            sample_rate: probability of using own prediction vs teacher token

        Returns:
            logits: (batch, seq-1, vocab_size)
        """
        device = page.device
        memory = self._build_memory(page, result_embedding)
        batch_size, seq_len = target_tokens.shape
        input_tokens = target_tokens[:, :1]  # BOS
        all_logits = []

        for t in range(1, seq_len):
            tgt = self.embed(input_tokens) + self.pos_embed(
                torch.arange(input_tokens.size(1), device=device))
            tgt = self.norm(tgt)
            causal_mask = nn.Transformer.generate_square_subsequent_mask(
                input_tokens.size(1), device=device)
            out = self.decoder(tgt, memory, tgt_mask=causal_mask)
            logits = self.output(out[:, -1:, :])  # (batch, 1, vocab_size)
            all_logits.append(logits)

            # Scheduled sampling: sometimes use own prediction
            if self.training and torch.rand(1).item() < sample_rate:
                next_token = logits.squeeze(1).argmax(dim=-1, keepdim=True)  # (batch, 1)
            else:
                next_token = target_tokens[:, t:t + 1]  # (batch, 1)
            input_tokens = torch.cat([input_tokens, next_token], dim=1)

        return torch.cat(all_logits, dim=1)  # (batch, seq-1, vocab_size)

    def _generate(self, memory: torch.Tensor, device: torch.device,
                  temperature: float = 0.0, min_tokens: int = 5) -> List[str]:
        """Autoregressive generation with EOS blocking for first min_tokens."""
        batch_size = memory.size(0)
        results = []

        for b in range(batch_size):
            mem = memory[b:b + 1]  # (1, N, d_model)
            tokens = [self.vocab.bos_id]

            for step in range(self.max_tokens):
                token_ids = torch.tensor([tokens], device=device)
                seq_len = token_ids.size(1)

                positions = torch.arange(seq_len, device=device)
                tgt = self.embed(token_ids) + self.pos_embed(positions)
                tgt = self.norm(tgt)

                causal_mask = nn.Transformer.generate_square_subsequent_mask(
                    seq_len, device=device
                )

                out = self.decoder(tgt, mem, tgt_mask=causal_mask)
                logits = self.output(out[:, -1, :])  # (1, vocab_size)

                # Block EOS for first min_tokens steps
                if step < min_tokens:
                    logits[:, self.vocab.eos_id] = -1e9

                if temperature == 0:
                    next_token = logits.argmax(dim=-1).item()
                else:
                    probs = torch.softmax(logits / temperature, dim=-1)
                    next_token = torch.multinomial(probs, 1).item()

                tokens.append(next_token)

                if next_token == self.vocab.eos_id:
                    break

            results.append(self.vocab.decode(tokens[1:]))  # strip BOS (decode strips EOS)

        return results

    def compute_loss(self, page: torch.Tensor,
                     target_tokens: torch.Tensor,
                     result_embedding: Optional[torch.Tensor] = None,
                     sample_rate: float = 0.0) -> torch.Tensor:
        """
        Compute cross-entropy loss. Uses scheduled sampling if sample_rate > 0.

        Args:
            page: (batch, 64)
            target_tokens: (batch, seq) — full sequence including BOS and EOS
            result_embedding: (batch, 64) — optional
            sample_rate: scheduled sampling rate (0.0 = pure teacher forcing)

        Returns:
            loss: scalar tensor
        """
        if sample_rate > 0 and self.training:
            logits = self.forward_scheduled_sampling(
                page, target_tokens, result_embedding, sample_rate=sample_rate,
            )
        else:
            logits = self.forward(page, result_embedding, target_tokens=target_tokens)

        # logits: (batch, seq-1, vocab_size)
        # labels: target_tokens[:, 1:]
        labels = target_tokens[:, 1:]  # (batch, seq-1)

        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            labels.reshape(-1),
            ignore_index=self.vocab.pad_id,
        )
        return loss
