"""
SymPy Decoder — Separated Comprehension and Computation (v24.9)

A small dedicated decoder (~5M params) that reads 64-float pages and emits
SymPy expressions token by token. Completely separate from the LLM's
generation path — no vocabulary contamination possible.

The LLM comprehends. The decoder formulates. SymPy computes.

Architecture:
  - Tiny math vocabulary (~50 tokens): digits, operators, variables, assignment
  - 2-layer transformer decoder, d_model=256, 4 heads
  - Cross-attends to page (projected to memory) + optional previous results
  - Autoregressive generation at inference, teacher forcing at training

Key insight: The SymPy decoder INHERENTLY needs per-problem pages because
different problems need different formulas. Page diversity falls out naturally,
unlike the LLM generation loss which killed it.
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
    Small decoder that reads a 64-float page and emits a SymPy expression.

    Tiny math-specific vocabulary (~50 tokens).
    2-layer transformer decoder (~5M params).
    Completely separate from the LLM's generation path.
    """

    def __init__(self, page_size: int = 64, d_model: int = 256, nhead: int = 4,
                 num_layers: int = 2, max_tokens: int = 40, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.max_tokens = max_tokens
        self.page_size = page_size

        # Math-specific vocabulary
        self.vocab = SymPyVocab()
        vocab_size = len(self.vocab)

        # Token embedding
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_tokens, d_model)

        # Page projection (page → decoder memory)
        # Project 64-float page to multiple memory positions for richer cross-attention
        self.page_project = nn.Sequential(
            nn.Linear(page_size, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model * 4),  # → reshape to (4, d_model)
        )

        # Also project accumulated SymPy results as additional memory
        self.result_project = nn.Sequential(
            nn.Linear(page_size, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

        # Transformer decoder
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
        """Initialize weights for stable training."""
        nn.init.xavier_uniform_(self.embed.weight)
        nn.init.xavier_uniform_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def _build_memory(self, page: torch.Tensor,
                      result_embedding: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Build cross-attention memory from page (+ optional previous results).

        page: (batch, 64) → projected to (batch, 4, d_model) for richer attention
        result_embedding: (batch, 64) → projected to (batch, 1, d_model)

        Returns: (batch, N, d_model) where N = 4 or 5
        """
        batch_size = page.size(0)

        # Project page to 4 memory positions
        page_flat = self.page_project(page)  # (batch, d_model * 4)
        page_mem = page_flat.view(batch_size, 4, self.d_model)  # (batch, 4, d_model)

        if result_embedding is not None:
            result_mem = self.result_project(result_embedding).unsqueeze(1)  # (batch, 1, d_model)
            memory = torch.cat([page_mem, result_mem], dim=1)  # (batch, 5, d_model)
        else:
            memory = page_mem  # (batch, 4, d_model)

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
            logits: (batch, seq, vocab_size)
        """
        memory = self._build_memory(page, result_embedding)

        if target_tokens is not None:
            # Teacher forcing — shift right: input is target[:-1], labels are target[1:]
            tgt_input = target_tokens[:, :-1]  # (batch, seq-1)
            seq_len = tgt_input.size(1)

            positions = torch.arange(seq_len, device=page.device)
            tgt = self.embed(tgt_input) + self.pos_embed(positions)
            tgt = self.norm(tgt)

            # Causal mask
            causal_mask = nn.Transformer.generate_square_subsequent_mask(
                seq_len, device=page.device
            )

            # Padding mask for target
            tgt_key_padding_mask = (tgt_input == self.vocab.pad_id)

            out = self.decoder(
                tgt, memory,
                tgt_mask=causal_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
            )
            return self.output(out)  # (batch, seq-1, vocab_size)
        else:
            # Autoregressive generation (inference)
            return self._generate(memory, page.device)

    def _generate(self, memory: torch.Tensor, device: torch.device,
                  temperature: float = 0.0) -> List[str]:
        """
        Autoregressive generation of SymPy expressions.

        Args:
            memory: (batch, N, d_model) cross-attention memory
            device: torch device
            temperature: 0 for greedy

        Returns:
            List of decoded SymPy expression strings (one per batch item)
        """
        batch_size = memory.size(0)
        results = []

        for b in range(batch_size):
            mem = memory[b:b + 1]  # (1, N, d_model)
            tokens = [self.vocab.bos_id]

            for step in range(self.max_tokens):
                token_ids = torch.tensor([tokens], device=device)  # (1, seq)
                seq_len = token_ids.size(1)

                positions = torch.arange(seq_len, device=device)
                tgt = self.embed(token_ids) + self.pos_embed(positions)
                tgt = self.norm(tgt)

                causal_mask = nn.Transformer.generate_square_subsequent_mask(
                    seq_len, device=device
                )

                out = self.decoder(tgt, mem, tgt_mask=causal_mask)
                logits = self.output(out[:, -1, :])  # (1, vocab_size)

                # Block EOS for first min_tokens steps — must produce an expression
                if step < 5:
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
                     result_embedding: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute cross-entropy loss for teacher forcing.

        Args:
            page: (batch, 64)
            target_tokens: (batch, seq) — full sequence including BOS and EOS
            result_embedding: (batch, 64) — optional

        Returns:
            loss: scalar tensor
        """
        logits = self.forward(page, result_embedding, target_tokens=target_tokens)
        # logits: (batch, seq-1, vocab_size) — predictions for positions 1..seq-1
        # labels: target_tokens[:, 1:] — actual tokens at positions 1..seq-1

        labels = target_tokens[:, 1:]  # (batch, seq-1)

        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            labels.reshape(-1),
            ignore_index=self.vocab.pad_id,
        )
        return loss
