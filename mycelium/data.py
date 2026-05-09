"""Tokenizer + corpus loader for Phase 0 training.

Wraps Pythia's BPE tokenizer (HF `tokenizers` lib) and provides a simple
random-window batch sampler over a tokenized text corpus.
"""
import os
from typing import List
import numpy as np
from tokenizers import Tokenizer
from tinygrad import Tensor, dtypes


_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOKENIZER_PATH = os.path.join(_PROJECT_ROOT, ".cache", "pythia-410m", "tokenizer.json")
WIKITEXT_DIR = os.path.join(_PROJECT_ROOT, ".cache", "wikitext-2")


def load_tokenizer() -> Tokenizer:
    return Tokenizer.from_file(TOKENIZER_PATH)


def _tokenize_parquet_to_npy(parquet_path: str, npy_path: str, tok: Tokenizer) -> np.ndarray:
    import pyarrow.parquet as pq
    rows = pq.read_table(parquet_path).column("text").to_pylist()
    text = "\n".join(s for s in rows if s.strip())
    ids = np.asarray(tok.encode(text).ids, dtype=np.int32)
    np.save(npy_path, ids)
    return ids


def load_wikitext(tok: Tokenizer | None = None, split: str = "train") -> np.ndarray:
    """Return token ids for wikitext-2 split, cached on disk as .npy."""
    if tok is None:
        tok = load_tokenizer()
    npy_path = os.path.join(WIKITEXT_DIR, f"{split}.npy")
    if os.path.exists(npy_path):
        return np.load(npy_path)
    parquet = os.path.join(WIKITEXT_DIR, f"{split}.parquet")
    return _tokenize_parquet_to_npy(parquet, npy_path, tok)


def sample_batch(ids: np.ndarray, batch: int, seq: int, rng: np.random.Generator) -> Tensor:
    """Sample `batch` random context windows of length `seq` from `ids`. Each
    window is a contiguous slice. Returned as int32 tensor (B, seq).
    """
    n = ids.shape[0]
    starts = rng.integers(0, n - seq - 1, size=batch)
    out = np.empty((batch, seq), dtype=np.int32)
    for i, s in enumerate(starts):
        out[i] = ids[s : s + seq]
    return Tensor(out, dtype=dtypes.int).realize()
