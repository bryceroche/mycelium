"""Phase 1 NL Parser: DistilBERT + classification heads.

Architecture:
  - Backbone: DistilBERT-base-uncased (66M, frozen optionally)
  - Input:  question text + variable descriptions (concatenated per-var)
  - Heads (per variable):
      leaf_head:   32-way softmax (IB leaf class)
      op_head:     5-way softmax (none/add/sub/mul/div)
  - Loss: CE(leaf) + CE(op), both standard cross-entropy
    Class-weighted CE for leaf to handle the severe imbalance (DIV.1 = 7331,
    ADD.1.0 = 24).

Encoding strategy:
  We encode each (question, var_description) pair independently:
    [CLS] question [SEP] var_description [SEP]
  The [CLS] token embedding is fed to both heads.

  This is simpler and more memory-efficient than encoding all variables
  simultaneously for the ~3-14 variables per problem.  At inference, we
  batch all variables from one problem into a single forward pass.

  Alternative (span extraction) is NOT used: variable descriptions in the
  factor graph are NOT necessarily verbatim substrings of the question —
  they are Haiku-generated semantic labels.  Span extraction would require
  exact match or fuzzy alignment; classification over description text is
  cleaner.

Usage:
  from mycelium.phase1_classifier import Phase1Classifier, Phase1Config
  model = Phase1Classifier(config)
  out = model(input_ids, attention_mask)    # out.leaf_logits, out.op_logits
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

# ---------------------------------------------------------------------------
# Constants (loaded from metadata file at import if available)
# ---------------------------------------------------------------------------

_META_PATH = Path(__file__).resolve().parent.parent / ".cache" / "phase1_classifier_meta.json"

def _load_meta() -> tuple[list[str], dict[str, int], dict[str, int]]:
    if _META_PATH.exists():
        m = json.loads(_META_PATH.read_text())
        return m["leaf_ids"], m["leaf_to_int"], m["op_to_int"]
    # Fallback canonical order
    leaf_ids = sorted([
        "ADD.0.0.0","ADD.0.0.1","ADD.0.0.2","ADD.0.1",
        "ADD.1.0","ADD.1.1.0","ADD.1.1.1","ADD.1.1.2",
        "ADD.1.2.0","ADD.1.2.1",
        "DIV.0.0.0","DIV.0.0.1","DIV.0.0.2",
        "DIV.0.1.0","DIV.0.1.1","DIV.0.1.2",
        "DIV.0.2.0","DIV.0.2.1","DIV.0.2.2","DIV.1",
        "MUL.0.0.0","MUL.0.0.1","MUL.0.0.2",
        "MUL.0.1.0","MUL.0.1.1","MUL.1",
        "SUB.0.0.0","SUB.0.0.1","SUB.0.1",
        "SUB.0.2.0","SUB.0.2.1","SUB.1",
    ])
    leaf_to_int = {l: i for i, l in enumerate(leaf_ids)}
    op_to_int = {"none": 0, "add": 1, "sub": 2, "mul": 3, "div": 4}
    return leaf_ids, leaf_to_int, op_to_int

LEAF_IDS, LEAF_TO_INT, OP_TO_INT = _load_meta()
N_LEAVES = len(LEAF_IDS)   # 32
N_OPS    = 5               # none/add/sub/mul/div


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class Phase1Config:
    backbone:          str   = "distilbert-base-uncased"
    hidden_size:       int   = 768       # DistilBERT hidden dim
    dropout_p:         float = 0.1
    freeze_backbone:   bool  = False     # set True to train heads only first
    max_seq_len:       int   = 128       # per (question, var) pair; 128 is safe
    # Loss weights
    leaf_loss_weight:  float = 1.0
    op_loss_weight:    float = 0.5       # op is easier (4-way vs 32-way)
    # Class weights strategy: "uniform" | "inverse_sqrt" | "inverse"
    leaf_weight_strategy: str = "inverse_sqrt"
    # Leaf counts for weighting (filled by training script from data stats)
    leaf_counts: list[int] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Model outputs
# ---------------------------------------------------------------------------

@dataclass
class Phase1Output:
    leaf_logits: Tensor    # (N_vars, 32)
    op_logits:   Tensor    # (N_vars, 5)
    cls_hidden:  Tensor    # (N_vars, hidden_size) — for diagnostics


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class Phase1Classifier(nn.Module):
    """DistilBERT backbone + leaf-classification head + op-classification head."""

    def __init__(self, config: Phase1Config):
        super().__init__()
        self.config = config
        self._load_backbone()

        H = config.hidden_size
        self.dropout = nn.Dropout(config.dropout_p)

        # Leaf head: 2-layer MLP, 32-way output
        self.leaf_head = nn.Sequential(
            nn.Linear(H, H // 2),
            nn.GELU(),
            nn.Dropout(config.dropout_p),
            nn.Linear(H // 2, N_LEAVES),
        )

        # Op head: 1-layer, 5-way output (simpler — easier task)
        self.op_head = nn.Sequential(
            nn.Linear(H, H // 4),
            nn.GELU(),
            nn.Dropout(config.dropout_p),
            nn.Linear(H // 4, N_OPS),
        )

        # Build class weights for leaf CE loss
        self._leaf_weights: Optional[Tensor] = None
        if config.leaf_counts:
            self._leaf_weights = self._build_leaf_weights(
                config.leaf_counts, config.leaf_weight_strategy
            )

    # ------------------------------------------------------------------
    # Backbone loading
    # ------------------------------------------------------------------

    def _load_backbone(self) -> None:
        """Load DistilBERT; tolerate import errors gracefully."""
        try:
            from transformers import DistilBertModel
            self.backbone = DistilBertModel.from_pretrained(
                self.config.backbone,
                # Suppress "some weights were not used" noise
                ignore_mismatched_sizes=False,
            )
            if self.config.freeze_backbone:
                for p in self.backbone.parameters():
                    p.requires_grad = False
        except Exception as e:
            raise ImportError(
                f"Could not load DistilBERT backbone '{self.config.backbone}': {e}\n"
                "Install: pip install transformers torch"
            ) from e

    # ------------------------------------------------------------------
    # Class-weight helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_leaf_weights(counts: list[int], strategy: str) -> Tensor:
        """Build per-class weights for unbalanced CE loss."""
        import numpy as np
        counts_arr = np.array(counts, dtype=np.float32)
        counts_arr = np.maximum(counts_arr, 1.0)    # avoid div-by-zero
        if strategy == "inverse":
            weights = 1.0 / counts_arr
        elif strategy == "inverse_sqrt":
            weights = 1.0 / np.sqrt(counts_arr)
        else:
            weights = np.ones_like(counts_arr)
        # Normalize so mean weight = 1
        weights = weights / weights.mean()
        return torch.tensor(weights, dtype=torch.float32)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        input_ids: Tensor,        # (B, L)
        attention_mask: Tensor,   # (B, L)
    ) -> Phase1Output:
        """
        B = batch size (one entry per variable across all problems in batch).
        L = sequence length (max_seq_len).
        """
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        # DistilBERT returns last_hidden_state (B, L, H); [CLS] is token 0
        cls_hidden = out.last_hidden_state[:, 0, :]   # (B, H)
        cls_hidden = self.dropout(cls_hidden)

        leaf_logits = self.leaf_head(cls_hidden)   # (B, 32)
        op_logits   = self.op_head(cls_hidden)     # (B, 5)

        return Phase1Output(
            leaf_logits=leaf_logits,
            op_logits=op_logits,
            cls_hidden=cls_hidden,
        )

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------

    def compute_loss(
        self,
        leaf_logits: Tensor,   # (B, 32)
        op_logits:   Tensor,   # (B, 5)
        leaf_labels: Tensor,   # (B,) int64
        op_labels:   Tensor,   # (B,) int64
    ) -> dict[str, Tensor]:
        """Compute per-head CE losses; return dict with individual + total."""
        device = leaf_logits.device

        # Leaf CE — with optional class weighting
        leaf_weights = None
        if self._leaf_weights is not None:
            leaf_weights = self._leaf_weights.to(device)
        leaf_loss = nn.functional.cross_entropy(
            leaf_logits, leaf_labels, weight=leaf_weights
        )

        # Op CE — uniform weights (4+1 classes, not too imbalanced)
        op_loss = nn.functional.cross_entropy(op_logits, op_labels)

        cfg = self.config
        total = cfg.leaf_loss_weight * leaf_loss + cfg.op_loss_weight * op_loss

        return {
            "loss":      total,
            "leaf_loss": leaf_loss,
            "op_loss":   op_loss,
        }

    # ------------------------------------------------------------------
    # Inference helper
    # ------------------------------------------------------------------

    @torch.no_grad()
    def predict(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
    ) -> dict[str, list]:
        """Run inference, return decoded leaf_id + op strings."""
        self.eval()
        out = self(input_ids, attention_mask)
        leaf_pred = out.leaf_logits.argmax(dim=-1).tolist()
        op_pred   = out.op_logits.argmax(dim=-1).tolist()

        INT_TO_OP = {v: k for k, v in OP_TO_INT.items()}
        return {
            "leaf_ids": [LEAF_IDS[i] for i in leaf_pred],
            "ops":      [INT_TO_OP[i] for i in op_pred],
            "leaf_probs": out.leaf_logits.softmax(dim=-1).tolist(),
            "op_probs":   out.op_logits.softmax(dim=-1).tolist(),
        }

    # ------------------------------------------------------------------
    # Parameter count
    # ------------------------------------------------------------------

    def param_count(self) -> dict[str, int]:
        backbone_p  = sum(p.numel() for p in self.backbone.parameters())
        head_p      = sum(p.numel() for p in self.leaf_head.parameters())
        head_p     += sum(p.numel() for p in self.op_head.parameters())
        total       = backbone_p + head_p
        trainable   = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            "backbone": backbone_p,
            "heads":    head_p,
            "total":    total,
            "trainable": trainable,
        }
