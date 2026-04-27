"""Tinygrad nn utilities — wrappers that ensure requires_grad on all parameters."""
from tinygrad import Tensor
from tinygrad.nn import Linear as _Linear, LayerNorm as _LayerNorm, Embedding as _Embedding


def Linear(in_features: int, out_features: int, bias=True):
    """Linear layer with requires_grad on all parameters."""
    lin = _Linear(in_features, out_features, bias=bias)
    lin.weight.requires_grad_()
    if bias and lin.bias is not None:
        lin.bias.requires_grad_()
    return lin


def LayerNorm(normalized_shape, eps=1e-5):
    """LayerNorm with requires_grad on all parameters."""
    ln = _LayerNorm(normalized_shape, eps=eps)
    if hasattr(ln, 'weight') and ln.weight is not None:
        ln.weight.requires_grad_()
    if hasattr(ln, 'bias') and ln.bias is not None:
        ln.bias.requires_grad_()
    return ln


def Embedding(num_embeddings: int, embedding_dim: int):
    """Embedding with requires_grad."""
    emb = _Embedding(num_embeddings, embedding_dim)
    emb.weight.requires_grad_()
    return emb
