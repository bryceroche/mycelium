"""
Custom GTS (Goal-driven Tree-structured) Model Loader

Loads a pre-trained GTS model from MWPToolkit without the MWPToolkit dependency.
The model architecture is reconstructed based on the saved state dict.

Architecture (from state dict analysis):
- embedder: Embedding(vocab_size=1032, embedding_dim=128)
- encoder: 2-layer bidirectional GRU (hidden=512, bidirectional output=1024)
- decoder: Tree decoder with attention mechanism
- merge: Merges left/right subtree representations
- node_generater: Generates operator/operand predictions

Reference: "Goal-Driven Tree-Structured Neural Model for Math Word Problems"
https://www.ijcai.org/proceedings/2019/0736.pdf
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

logger = logging.getLogger(__name__)


@dataclass
class GTSConfig:
    """Configuration for GTS model."""

    vocab_size: int
    output_size: int  # Number of output symbols (operators + NUM tokens)
    generate_size: int  # Number of generated constants
    embedding_size: int = 128
    hidden_size: int = 512
    num_layers: int = 2
    dropout_ratio: float = 0.5
    beam_size: int = 5
    max_output_len: int = 30

    @classmethod
    def from_config_dict(
        cls, config: dict, input_vocab: dict, output_vocab: dict
    ) -> GTSConfig:
        """Create config from loaded JSON config."""
        # Get the final config dict which has all merged values
        final = config.get("final_config_dict", config)

        # Count output symbols
        out_symbols = output_vocab.get("out_idx2symbol", [])
        temp_symbols = output_vocab.get("temp_idx2symbol", [])

        # Input vocab size
        in_vocab = input_vocab.get("in_idx2word", [])

        return cls(
            vocab_size=len(in_vocab),
            output_size=len(out_symbols),
            generate_size=len(temp_symbols),
            embedding_size=final.get("embedding_size", 128),
            hidden_size=final.get("hidden_size", 512),
            num_layers=final.get("num_layers", 2),
            dropout_ratio=final.get("dropout_ratio", 0.5),
            beam_size=final.get("beam_size", 5),
            max_output_len=final.get("max_output_len", 30),
        )


class GTSEmbedder(nn.Module):
    """Embedding layer for input tokens."""

    def __init__(self, vocab_size: int, embedding_size: int, dropout: float = 0.5):
        super().__init__()
        self.embedder = nn.Embedding(vocab_size, embedding_size, padding_idx=0)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_ids: (batch, seq_len) token indices

        Returns:
            (batch, seq_len, embedding_size) embeddings
        """
        return self.dropout(self.embedder(input_ids))


class GTSEncoder(nn.Module):
    """Bidirectional GRU encoder for problem text."""

    def __init__(
        self,
        embedding_size: int,
        hidden_size: int,
        num_layers: int = 2,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.encoder = nn.GRU(
            embedding_size,
            hidden_size,
            num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0,
        )

    def forward(
        self, embedded: torch.Tensor, lengths: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            embedded: (batch, seq_len, embedding_size) embedded tokens
            lengths: (batch,) sequence lengths

        Returns:
            outputs: (batch, seq_len, hidden_size*2) encoder outputs
            hidden: (batch, hidden_size) final hidden state
        """
        # Pack for efficient RNN processing
        packed = pack_padded_sequence(
            embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        outputs, hidden = self.encoder(packed)
        outputs, _ = pad_packed_sequence(outputs, batch_first=True)

        # Combine forward and backward hidden states
        # hidden: (num_layers*2, batch, hidden_size)
        # Take last layer's forward and backward, concatenate
        hidden = torch.cat(
            [hidden[-2, :, :], hidden[-1, :, :]], dim=1
        )  # (batch, hidden_size*2)

        return outputs, hidden


class TreeAttn(nn.Module):
    """Attention mechanism for tree decoder.

    Note: The checkpoint has attn: (512, 1024) and score: (1, 512) with bias.
    The input is encoder_outputs which is hidden_size*2 = 1024.
    """

    def __init__(self, hidden_size: int):
        super().__init__()
        # Input is encoder_outputs (hidden_size*2), output is hidden_size
        self.attn = nn.Linear(hidden_size * 2, hidden_size)
        # Note: checkpoint has bias=True for score layer
        self.score = nn.Linear(hidden_size, 1, bias=True)

    def forward(
        self,
        hidden: torch.Tensor,
        encoder_outputs: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            hidden: (batch, hidden_size) current decoder hidden state (unused in this impl)
            encoder_outputs: (batch, seq_len, hidden_size*2) encoder outputs
            mask: (batch, seq_len) attention mask (1=valid, 0=padding)

        Returns:
            context: (batch, hidden_size*2) attention-weighted context
        """
        batch_size, seq_len, _ = encoder_outputs.size()

        # Compute attention scores directly on encoder outputs
        # attn: (batch, seq_len, hidden_size*2) -> (batch, seq_len, hidden_size)
        energy = torch.tanh(self.attn(encoder_outputs))
        # score: (batch, seq_len, hidden_size) -> (batch, seq_len, 1) -> (batch, seq_len)
        scores = self.score(energy).squeeze(2)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        weights = F.softmax(scores, dim=1)  # (batch, seq_len)

        # Weighted sum of encoder outputs
        context = torch.bmm(weights.unsqueeze(1), encoder_outputs).squeeze(
            1
        )  # (batch, hidden*2)

        return context


class Score(nn.Module):
    """Scoring module for selecting tokens."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.attn = nn.Linear(hidden_size * 3, hidden_size)
        self.score = nn.Linear(hidden_size, 1, bias=False)

    def forward(
        self, hidden: torch.Tensor, num_embeddings: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            hidden: (batch, hidden_size*2) decoder hidden state
            num_embeddings: (batch, num_nums, hidden_size) number embeddings

        Returns:
            scores: (batch, num_nums) scores for each number
        """
        batch_size, num_nums, _ = num_embeddings.size()

        hidden_expanded = hidden.unsqueeze(1).expand(-1, num_nums, -1)
        combined = torch.cat([hidden_expanded, num_embeddings], dim=2)

        energy = torch.tanh(self.attn(combined))
        scores = self.score(energy).squeeze(2)

        return scores


class Merge(nn.Module):
    """Merge left and right subtree representations."""

    def __init__(self, hidden_size: int, embedding_size: int):
        super().__init__()
        # hidden_size*2 + embedding_size = 512*2 + 128 = 1152
        self.merge = nn.Linear(hidden_size * 2 + embedding_size, hidden_size)
        self.merge_g = nn.Linear(hidden_size * 2 + embedding_size, hidden_size)

    def forward(
        self, left: torch.Tensor, right: torch.Tensor, op_embedding: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            left: (batch, hidden_size) left subtree representation
            right: (batch, hidden_size) right subtree representation
            op_embedding: (batch, embedding_size) operator embedding

        Returns:
            merged: (batch, hidden_size) merged representation
        """
        combined = torch.cat([left, right, op_embedding], dim=1)
        return torch.tanh(self.merge(combined)) * torch.sigmoid(self.merge_g(combined))


class NodeGenerator(nn.Module):
    """Generates child nodes in the expression tree."""

    def __init__(self, hidden_size: int, op_nums: int, embedding_size: int):
        super().__init__()
        self.embeddings = nn.Embedding(op_nums, embedding_size)
        # Input: hidden*2 + embedding = 1024 + 128 = 1152
        self.generate_l = nn.Linear(hidden_size * 2 + embedding_size, hidden_size)
        self.generate_lg = nn.Linear(hidden_size * 2 + embedding_size, hidden_size)
        self.generate_r = nn.Linear(hidden_size * 2 + embedding_size, hidden_size)
        self.generate_rg = nn.Linear(hidden_size * 2 + embedding_size, hidden_size)

    def forward(
        self, node_embedding: torch.Tensor, context: torch.Tensor, op_idx: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            node_embedding: (batch, hidden_size) current node embedding
            context: (batch, hidden_size*2) attention context
            op_idx: (batch,) operator indices

        Returns:
            left_child: (batch, hidden_size) left child embedding
            right_child: (batch, hidden_size) right child embedding
        """
        op_embed = self.embeddings(op_idx)  # (batch, embedding_size)
        combined = torch.cat([node_embedding, context, op_embed], dim=1)

        left = torch.tanh(self.generate_l(combined)) * torch.sigmoid(
            self.generate_lg(combined)
        )
        right = torch.tanh(self.generate_r(combined)) * torch.sigmoid(
            self.generate_rg(combined)
        )

        return left, right


class GTSDecoder(nn.Module):
    """Tree-structured decoder for GTS model."""

    def __init__(self, hidden_size: int, op_nums: int, generate_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.op_nums = op_nums  # Number of operators (+, -, *, /, ^, =)

        # Attention for context
        self.attn = TreeAttn(hidden_size)

        # Score for selecting numbers from problem
        self.score = Score(hidden_size)

        # Operator prediction (concat of node + context)
        self.ops = nn.Linear(hidden_size * 2, op_nums)

        # For generating left/right children
        self.concat_l = nn.Linear(hidden_size, hidden_size)
        self.concat_lg = nn.Linear(hidden_size, hidden_size)
        self.concat_r = nn.Linear(hidden_size * 2, hidden_size)
        self.concat_rg = nn.Linear(hidden_size * 2, hidden_size)

        # Embedding for generated constants (not from problem)
        # Shape: (1, generate_size, hidden_size) for constants like 1.0, 100.0, etc.
        self.embedding_weight = nn.Parameter(torch.zeros(1, generate_size, hidden_size))

    def forward(
        self,
        node: torch.Tensor,
        context: torch.Tensor,
        num_embeddings: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            node: (batch, hidden_size) current goal embedding
            context: (batch, hidden_size*2) context from encoder
            num_embeddings: (batch, num_nums, hidden_size) problem number embeddings
            mask: attention mask

        Returns:
            op_scores: (batch, op_nums) scores for operators
            num_scores: (batch, num_nums + generate_size) scores for numbers
        """
        # Predict operator
        combined = torch.cat([node, context], dim=1)
        op_scores = self.ops(combined)

        # Score numbers from problem + generated constants
        batch_size = num_embeddings.size(0)
        gen_embeds = self.embedding_weight.expand(batch_size, -1, -1)
        all_num_embeds = torch.cat([num_embeddings, gen_embeds], dim=1)

        num_scores = self.score(combined, all_num_embeds)

        return op_scores, num_scores


class GTSModel(nn.Module):
    """
    Goal-driven Tree-structured Neural Model for Math Word Problems.

    This model generates prefix-notation mathematical expressions from
    natural language math word problems.
    """

    def __init__(self, config: GTSConfig, input_vocab: dict, output_vocab: dict):
        super().__init__()
        self.config = config
        self.input_vocab = input_vocab
        self.output_vocab = output_vocab

        # Build vocab mappings
        self.idx2word = input_vocab.get("in_idx2word", [])
        self.word2idx = {w: i for i, w in enumerate(self.idx2word)}

        self.idx2symbol = output_vocab.get("out_idx2symbol", [])
        self.symbol2idx = {s: i for i, s in enumerate(self.idx2symbol)}

        self.temp_idx2symbol = output_vocab.get("temp_idx2symbol", [])

        # Count operators (first 6 symbols: +, -, *, /, ^, =)
        self.op_nums = 6

        # Model components
        self.embedder = GTSEmbedder(
            config.vocab_size, config.embedding_size, config.dropout_ratio
        )

        self.encoder = GTSEncoder(
            config.embedding_size,
            config.hidden_size,
            config.num_layers,
            config.dropout_ratio,
        )

        # generate_size is 12 based on the trained model checkpoint
        # This is x + 11 constants (0.01, 1.0, 3.0, 12.0, 100.0, 4.0, 7.0, 2.0, 0.5, 0.25, 0.1)
        self.generate_size = 12
        self.decoder = GTSDecoder(
            config.hidden_size, self.op_nums, self.generate_size
        )

        self.merge = Merge(config.hidden_size, config.embedding_size)

        self.node_generater = NodeGenerator(
            config.hidden_size, self.op_nums, config.embedding_size
        )

    @classmethod
    def from_pretrained(cls, model_path: str | Path) -> GTSModel:
        """
        Load a pre-trained GTS model from directory.

        Args:
            model_path: Path to directory containing:
                - config.json: Model configuration
                - input_vocab.json: Input vocabulary
                - output_vocab.json: Output vocabulary
                - model.pth: PyTorch state dict

        Returns:
            Loaded GTSModel instance
        """
        model_path = Path(model_path)

        logger.info(f"Loading GTS model from {model_path}")

        # Load config and vocab files
        with open(model_path / "config.json") as f:
            config_dict = json.load(f)
        with open(model_path / "input_vocab.json") as f:
            input_vocab = json.load(f)
        with open(model_path / "output_vocab.json") as f:
            output_vocab = json.load(f)

        # Create config
        config = GTSConfig.from_config_dict(config_dict, input_vocab, output_vocab)
        logger.info(f"Model config: {config}")

        # Build model
        model = cls(config, input_vocab, output_vocab)

        # Load state dict
        checkpoint = torch.load(model_path / "model.pth", map_location="cpu")

        # The checkpoint has a 'model' key containing the actual state dict
        state_dict = checkpoint.get("model", checkpoint)

        # Load weights
        model.load_state_dict(state_dict)
        model.eval()

        logger.info("Model loaded successfully")
        return model

    def tokenize(self, text: str) -> list[int]:
        """
        Tokenize input text to token indices.

        Args:
            text: Input problem text

        Returns:
            List of token indices
        """
        tokens = text.lower().split()
        indices = []
        unk_idx = self.word2idx.get("<UNK>", 3)

        for token in tokens:
            idx = self.word2idx.get(token, unk_idx)
            indices.append(idx)

        return indices

    def encode(
        self, input_ids: torch.Tensor, lengths: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Encode input sequence.

        Args:
            input_ids: (batch, seq_len) token indices
            lengths: (batch,) sequence lengths

        Returns:
            encoder_outputs: (batch, seq_len, hidden_size*2)
            hidden: (batch, hidden_size*2)
        """
        embedded = self.embedder(input_ids)
        return self.encoder(embedded, lengths)

    def generate(
        self,
        problem_text: str,
        num_values: list[float] | None = None,
        beam_size: int | None = None,
    ) -> str:
        """
        Generate a prefix expression for a math word problem.

        Args:
            problem_text: The math word problem text
            num_values: Optional list of number values extracted from problem
            beam_size: Beam size for beam search (default: config.beam_size)

        Returns:
            Prefix notation expression (e.g., "+ NUM_0 NUM_1")
        """
        if beam_size is None:
            beam_size = self.config.beam_size

        # Tokenize
        tokens = self.tokenize(problem_text)

        # Convert to tensor
        input_ids = torch.tensor([tokens], dtype=torch.long)
        lengths = torch.tensor([len(tokens)], dtype=torch.long)

        # Encode
        with torch.no_grad():
            encoder_outputs, hidden = self.encode(input_ids, lengths)

        # For now, return a placeholder - full beam search decoding is complex
        # and requires the full tree generation algorithm
        logger.warning(
            "Full beam search decoding not yet implemented. "
            "Use encode() for embeddings only."
        )

        return "<decoding_not_implemented>"

    def get_state_dict_info(self) -> dict[str, Any]:
        """Get information about the model's state dict for debugging."""
        info = {}
        for name, param in self.named_parameters():
            info[name] = {
                "shape": list(param.shape),
                "dtype": str(param.dtype),
                "requires_grad": param.requires_grad,
            }
        return info


def load_gts_model(model_path: str | Path = "trained_model/GTS-mawps") -> GTSModel:
    """
    Convenience function to load the GTS model.

    Args:
        model_path: Path to model directory

    Returns:
        Loaded GTSModel
    """
    return GTSModel.from_pretrained(model_path)


# For testing
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Print expected vs actual state dict keys
    model_path = Path("trained_model/GTS-mawps")

    if model_path.exists():
        print("Loading model...")
        model = load_gts_model(model_path)

        print("\nModel architecture:")
        print(model)

        print("\nState dict info:")
        for name, info in model.get_state_dict_info().items():
            print(f"  {name}: {info['shape']}")
    else:
        print(f"Model path not found: {model_path}")
