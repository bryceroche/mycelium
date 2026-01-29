"""Tests for the custom GTS model loader."""

import json
from pathlib import Path

import pytest
import torch

from mycelium.gts_model import (
    GTSConfig,
    GTSModel,
    load_gts_model,
)


MODEL_PATH = Path("trained_model/GTS-mawps")


@pytest.fixture
def model_files():
    """Check if model files exist."""
    if not MODEL_PATH.exists():
        pytest.skip("Model files not found at trained_model/GTS-mawps")
    return MODEL_PATH


class TestVocabLoading:
    """Tests for vocabulary loading."""

    def test_input_vocab_loading(self, model_files):
        """Test that input vocabulary loads correctly."""
        with open(model_files / "input_vocab.json") as f:
            input_vocab = json.load(f)

        # Check expected structure
        assert "in_idx2word" in input_vocab
        vocab_list = input_vocab["in_idx2word"]

        # Check expected special tokens
        assert vocab_list[0] == "<PAD>"
        assert vocab_list[1] == "<SOS>"
        assert vocab_list[2] == "<EOS>"
        assert vocab_list[3] == "<UNK>"

        # Check vocab size matches model
        assert len(vocab_list) == 1032

    def test_output_vocab_loading(self, model_files):
        """Test that output vocabulary loads correctly."""
        with open(model_files / "output_vocab.json") as f:
            output_vocab = json.load(f)

        # Check expected structure
        assert "out_idx2symbol" in output_vocab
        assert "temp_idx2symbol" in output_vocab

        out_symbols = output_vocab["out_idx2symbol"]
        temp_symbols = output_vocab["temp_idx2symbol"]

        # Check operators are first
        assert out_symbols[0] == "+"
        assert out_symbols[1] == "-"
        assert out_symbols[2] == "*"
        assert out_symbols[3] == "/"
        assert out_symbols[4] == "^"
        assert out_symbols[5] == "="

        # Check NUM tokens exist
        assert "NUM_0" in out_symbols
        assert "NUM_1" in out_symbols


class TestStateDictKeys:
    """Tests for verifying state dict keys match."""

    def test_state_dict_keys_documented(self, model_files):
        """Test that we have documented all state dict keys."""
        checkpoint = torch.load(model_files / "model.pth", map_location="cpu")
        state_dict = checkpoint.get("model", checkpoint)

        expected_prefixes = {
            "embedder.",
            "encoder.",
            "decoder.",
            "merge.",
            "node_generater.",
        }

        for key in state_dict.keys():
            prefix = key.split(".")[0] + "."
            assert prefix in expected_prefixes, f"Unexpected key prefix: {key}"

    def test_embedder_keys(self, model_files):
        """Test embedder layer keys."""
        checkpoint = torch.load(model_files / "model.pth", map_location="cpu")
        state_dict = checkpoint.get("model", checkpoint)

        assert "embedder.embedder.weight" in state_dict
        assert state_dict["embedder.embedder.weight"].shape == (1032, 128)

    def test_encoder_keys(self, model_files):
        """Test encoder layer keys."""
        checkpoint = torch.load(model_files / "model.pth", map_location="cpu")
        state_dict = checkpoint.get("model", checkpoint)

        # 2-layer bidirectional GRU should have these keys
        encoder_keys = [k for k in state_dict if k.startswith("encoder.")]
        assert len(encoder_keys) == 16  # 4 weights * 2 layers * 2 directions

    def test_decoder_keys(self, model_files):
        """Test decoder layer keys."""
        checkpoint = torch.load(model_files / "model.pth", map_location="cpu")
        state_dict = checkpoint.get("model", checkpoint)

        # Check key decoder components
        assert "decoder.embedding_weight" in state_dict
        assert state_dict["decoder.embedding_weight"].shape == (1, 12, 512)

        assert "decoder.ops.weight" in state_dict
        assert state_dict["decoder.ops.weight"].shape == (6, 1024)  # 6 operators

        # Attention layers
        assert "decoder.attn.attn.weight" in state_dict
        assert "decoder.attn.score.weight" in state_dict
        assert "decoder.attn.score.bias" in state_dict  # Has bias!

    def test_merge_keys(self, model_files):
        """Test merge layer keys."""
        checkpoint = torch.load(model_files / "model.pth", map_location="cpu")
        state_dict = checkpoint.get("model", checkpoint)

        # Merge has gated linear layers
        assert "merge.merge.weight" in state_dict
        assert "merge.merge_g.weight" in state_dict
        # Input: hidden*2 + embedding = 512*2 + 128 = 1152
        assert state_dict["merge.merge.weight"].shape == (512, 1152)

    def test_node_generater_keys(self, model_files):
        """Test node_generater layer keys."""
        checkpoint = torch.load(model_files / "model.pth", map_location="cpu")
        state_dict = checkpoint.get("model", checkpoint)

        # Note the typo "generater" is intentional (matches checkpoint)
        assert "node_generater.embeddings.weight" in state_dict
        assert state_dict["node_generater.embeddings.weight"].shape == (6, 128)

        assert "node_generater.generate_l.weight" in state_dict
        assert "node_generater.generate_r.weight" in state_dict


class TestModelLoading:
    """Tests for model loading."""

    def test_model_loading_succeeds(self, model_files):
        """Test that GTSModel.from_pretrained loads without error."""
        model = load_gts_model(model_files)

        assert model is not None
        assert isinstance(model, GTSModel)

    def test_model_in_eval_mode(self, model_files):
        """Test that loaded model is in eval mode."""
        model = load_gts_model(model_files)
        assert not model.training

    def test_model_config_correct(self, model_files):
        """Test that model config is loaded correctly."""
        model = load_gts_model(model_files)

        assert model.config.vocab_size == 1032
        assert model.config.embedding_size == 128
        assert model.config.hidden_size == 512
        assert model.config.num_layers == 2
        assert model.config.beam_size == 5

    def test_vocab_mappings_created(self, model_files):
        """Test that vocabulary mappings are created."""
        model = load_gts_model(model_files)

        # Input vocab
        assert len(model.idx2word) == 1032
        assert len(model.word2idx) == 1032
        assert model.word2idx["<PAD>"] == 0
        assert model.word2idx["<UNK>"] == 3

        # Output vocab
        assert len(model.idx2symbol) > 0
        assert model.symbol2idx["+"] == 0

    def test_state_dict_shapes_match(self, model_files):
        """Test that all state dict shapes match between model and checkpoint."""
        # Load checkpoint
        checkpoint = torch.load(model_files / "model.pth", map_location="cpu")
        checkpoint_state = checkpoint.get("model", checkpoint)

        # Load model
        model = load_gts_model(model_files)
        model_state = model.state_dict()

        # Every key in checkpoint should be in model with same shape
        for key, checkpoint_tensor in checkpoint_state.items():
            assert key in model_state, f"Key {key} not found in model"
            assert (
                model_state[key].shape == checkpoint_tensor.shape
            ), f"Shape mismatch for {key}: model={model_state[key].shape}, checkpoint={checkpoint_tensor.shape}"


class TestModelEncoding:
    """Tests for model encoding functionality."""

    def test_tokenize(self, model_files):
        """Test tokenization works."""
        model = load_gts_model(model_files)

        tokens = model.tokenize("a waiter had NUM customers")
        assert isinstance(tokens, list)
        assert all(isinstance(t, int) for t in tokens)
        assert len(tokens) == 5

    def test_tokenize_unknown_words(self, model_files):
        """Test that unknown words get UNK token."""
        model = load_gts_model(model_files)

        tokens = model.tokenize("xyzfoobar")  # Unknown word
        assert len(tokens) == 1
        assert tokens[0] == model.word2idx["<UNK>"]

    def test_encode_produces_output(self, model_files):
        """Test that encoding produces output tensors."""
        model = load_gts_model(model_files)

        tokens = model.tokenize("a waiter had NUM customers")
        input_ids = torch.tensor([tokens], dtype=torch.long)
        lengths = torch.tensor([len(tokens)], dtype=torch.long)

        with torch.no_grad():
            encoder_outputs, hidden = model.encode(input_ids, lengths)

        # Check output shapes
        assert encoder_outputs.shape == (1, len(tokens), 1024)  # hidden*2
        assert hidden.shape == (1, 1024)  # hidden*2


class TestGTSConfig:
    """Tests for GTSConfig."""

    def test_config_from_config_dict(self, model_files):
        """Test config creation from JSON files."""
        with open(model_files / "config.json") as f:
            config_dict = json.load(f)
        with open(model_files / "input_vocab.json") as f:
            input_vocab = json.load(f)
        with open(model_files / "output_vocab.json") as f:
            output_vocab = json.load(f)

        config = GTSConfig.from_config_dict(config_dict, input_vocab, output_vocab)

        assert config.vocab_size == 1032
        assert config.embedding_size == 128
        assert config.hidden_size == 512
        assert config.num_layers == 2
        assert config.dropout_ratio == 0.5
        assert config.beam_size == 5
        assert config.max_output_len == 30
