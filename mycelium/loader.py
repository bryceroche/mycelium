"""Pythia-410M -> mycelium weight loader.

Maps the EleutherAI Pythia state dict onto:
  - mycelium.pythia.PythiaStack (baseline: 4 unmodified layers)
  - mycelium.breathing.BreathingTransformer (sharing + π cycling)

Pythia stores Linear weights as PyTorch convention (out_dim, in_dim). Our linears
use (in_dim, out_dim) for direct x @ w, so we transpose. Q/K/V are fused into a
single query_key_value weight that we split per-head.
"""
import os
from typing import Dict
from tinygrad import Tensor, dtypes, Device
from tinygrad.nn.state import safe_load

from mycelium.config import Config
from mycelium.pythia import PythiaStack, PythiaLayer
from mycelium.breathing import BreathingTransformer, BreathingLayer, SharedWeights


_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PYTHIA_CACHE = os.path.join(_PROJECT_ROOT, ".cache", "pythia-410m", "model.safetensors")


def _load_state(path: str = PYTHIA_CACHE) -> Dict[str, Tensor]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Pythia weights not found at {path}. Download with curl first.")
    return safe_load(path)


def _split_qkv(qkv_w_gpu: Tensor, qkv_b_gpu: Tensor, cfg: Config):
    """Split Pythia's fused QKV (already materialized on GPU) into per-projection
    weights/biases. Returns each weight in (in, out) layout.
    """
    H, hd, nh = cfg.hidden, cfg.head_dim, cfg.n_heads
    w = qkv_w_gpu.reshape(nh, 3, hd, H)
    q_w = w[:, 0].reshape(H, H)
    k_w = w[:, 1].reshape(H, H)
    v_w = w[:, 2].reshape(H, H)
    b = qkv_b_gpu.reshape(nh, 3, hd)
    q_b = b[:, 0].reshape(H)
    k_b = b[:, 1].reshape(H)
    v_b = b[:, 2].reshape(H)
    return q_w.T.contiguous(), k_w.T.contiguous(), v_w.T.contiguous(), q_b, k_b, v_b


def _gpu(t: Tensor) -> Tensor:
    """Materialize a safetensors-backed tensor on the GPU. DISK has no kernel
    renderer, so any compute (transpose, reshape, cast) must happen post-transfer.
    """
    return t.to(Device.DEFAULT).realize()


def _assign(dst: Tensor, src: Tensor) -> None:
    """Copy src values into dst, preserving dst's tensor identity so optimizer's
    requires_grad_ marking and autograd graph stay attached. src is assumed to be
    already on the GPU (use _gpu() first if loaded from safetensors).
    """
    if src.shape != dst.shape:
        src = src.reshape(dst.shape)
    if src.dtype != dst.dtype:
        src = src.cast(dst.dtype)
    if src.device != dst.device:
        src = src.to(dst.device)
    dst.assign(src).realize()


def _load_pythia_layer_weights(layer: PythiaLayer, sd: Dict[str, Tensor], i: int, cfg: Config):
    """Load Pythia layer i into a PythiaLayer via .assign() so tensor identity
    is preserved (optimizer/autograd wiring stays attached)."""
    p = f"gpt_neox.layers.{i}"

    q_w, k_w, v_w, q_b, k_b, v_b = _split_qkv(
        _gpu(sd[f"{p}.attention.query_key_value.weight"]),
        _gpu(sd[f"{p}.attention.query_key_value.bias"]),
        cfg,
    )
    _assign(layer.wq, q_w)
    _assign(layer.wk, k_w)
    _assign(layer.wv, v_w)
    _assign(layer.bq, q_b)
    _assign(layer.bk, k_b)
    _assign(layer.bv, v_b)

    _assign(layer.wo, _gpu(sd[f"{p}.attention.dense.weight"]).T)
    _assign(layer.bo, _gpu(sd[f"{p}.attention.dense.bias"]))

    _assign(layer.w_in, _gpu(sd[f"{p}.mlp.dense_h_to_4h.weight"]).T)
    _assign(layer.b_in, _gpu(sd[f"{p}.mlp.dense_h_to_4h.bias"]))
    _assign(layer.w_out, _gpu(sd[f"{p}.mlp.dense_4h_to_h.weight"]).T)
    _assign(layer.b_out, _gpu(sd[f"{p}.mlp.dense_4h_to_h.bias"]))

    _assign(layer.in_ln_g, _gpu(sd[f"{p}.input_layernorm.weight"]))
    _assign(layer.in_ln_b, _gpu(sd[f"{p}.input_layernorm.bias"]))
    _assign(layer.post_ln_g, _gpu(sd[f"{p}.post_attention_layernorm.weight"]))
    _assign(layer.post_ln_b, _gpu(sd[f"{p}.post_attention_layernorm.bias"]))


def load_pythia_baseline(cfg: Config, n_layers: int = 4, sd: Dict[str, Tensor] | None = None) -> PythiaStack:
    """Build a PythiaStack with the first n_layers of Pythia-410M loaded."""
    sd = sd or _load_state()
    model = PythiaStack(cfg, n_layers)

    _assign(model.embed.weight, _gpu(sd["gpt_neox.embed_in.weight"]))

    for i in range(n_layers):
        _load_pythia_layer_weights(model.layers[i], sd, i, cfg)

    _assign(model.ln_f_g, _gpu(sd["gpt_neox.final_layer_norm.weight"]))
    _assign(model.ln_f_b, _gpu(sd["gpt_neox.final_layer_norm.bias"]))
    return model


def load_breathing(cfg: Config, sd: Dict[str, Tensor] | None = None) -> BreathingTransformer:
    """Build a BreathingTransformer with weights mapped per the v4 spec:
      - Phase i's Q, K, FFN-in (and biases) come from Pythia layer i
      - Shared V, O, FFN-out, LNs come from Pythia layer 0
      - Embedding from gpt_neox.embed_in; output head from embed_out
    """
    sd = sd or _load_state()
    assert cfg.n_phases <= 4, "Pythia-410M only has 4 layers worth of phase-specific weights"

    model = BreathingTransformer(cfg)
    _assign(model.embed.weight, _gpu(sd["gpt_neox.embed_in.weight"]))

    # Shared (from Pythia layer 0)
    p0 = "gpt_neox.layers.0"
    _, _, v_w, _, _, v_b = _split_qkv(
        _gpu(sd[f"{p0}.attention.query_key_value.weight"]),
        _gpu(sd[f"{p0}.attention.query_key_value.bias"]),
        cfg,
    )
    sw: SharedWeights = model.block.shared
    _assign(sw.wv, v_w)
    _assign(sw.bv, v_b)
    _assign(sw.wo, _gpu(sd[f"{p0}.attention.dense.weight"]).T)
    _assign(sw.bo, _gpu(sd[f"{p0}.attention.dense.bias"]))
    _assign(sw.w_out, _gpu(sd[f"{p0}.mlp.dense_4h_to_h.weight"]).T)
    _assign(sw.b_out, _gpu(sd[f"{p0}.mlp.dense_4h_to_h.bias"]))
    _assign(sw.in_ln_g, _gpu(sd[f"{p0}.input_layernorm.weight"]))
    _assign(sw.in_ln_b, _gpu(sd[f"{p0}.input_layernorm.bias"]))
    _assign(sw.post_ln_g, _gpu(sd[f"{p0}.post_attention_layernorm.weight"]))
    _assign(sw.post_ln_b, _gpu(sd[f"{p0}.post_attention_layernorm.bias"]))

    # Phase-specific (Q, K, FFN-in from layers 0..3 respectively)
    for i, layer in enumerate(model.block.layers):
        p = f"gpt_neox.layers.{i}"
        q_w, k_w, _, q_b, k_b, _ = _split_qkv(
            _gpu(sd[f"{p}.attention.query_key_value.weight"]),
            _gpu(sd[f"{p}.attention.query_key_value.bias"]),
            cfg,
        )
        _assign(layer.wq, q_w)
        _assign(layer.wk, k_w)
        _assign(layer.bq, q_b)
        _assign(layer.bk, k_b)
        _assign(layer.w_in, _gpu(sd[f"{p}.mlp.dense_h_to_4h.weight"]).T)
        _assign(layer.b_in, _gpu(sd[f"{p}.mlp.dense_h_to_4h.bias"]))

    _assign(model.ln_f_g, _gpu(sd["gpt_neox.final_layer_norm.weight"]))
    _assign(model.ln_f_b, _gpu(sd["gpt_neox.final_layer_norm.bias"]))
    _assign(model.embed_out, _gpu(sd["embed_out.weight"]).T)
    return model
