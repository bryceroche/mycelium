#!/usr/bin/env python3
"""
C3 Source Classifier

Simple classifier that predicts WHERE each operand comes from:
- TEXT: from problem text
- PRIOR: from a prior computation result
- IMPLICIT: implied value ("half"→0.5, "dozen"→12)
- CONSTANT: domain constant (60 min/hr, etc.)

Uses MiniLM-22M (same as C2) - tiny task, tiny model.
Key fix: slot embeddings so each head knows which operand it's classifying.
"""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

# MiniLM-22M - same backbone as C2
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

SOURCE_TYPES = ['TEXT', 'PRIOR', 'IMPLICIT', 'CONSTANT', 'NONE']
SOURCE_TYPE_TO_IDX = {s: i for i, s in enumerate(SOURCE_TYPES)}
MAX_OPERANDS = 4


class C3SourceClassifier(nn.Module):
    """Classifies operand source types with slot-aware heads."""

    def __init__(self, backbone_name: str = MODEL_NAME, dropout: float = 0.1):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(backbone_name, trust_remote_code=True)
        hidden_size = self.backbone.config.hidden_size  # 384 for MiniLM

        # Slot embeddings - tells each head which operand it's classifying
        self.slot_embeddings = nn.Embedding(MAX_OPERANDS, hidden_size)

        # Shared classifier (slot embedding differentiates)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, len(SOURCE_TYPES))
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)

        # Mean pooling over sequence
        hidden = outputs.last_hidden_state  # (batch, seq, hidden)
        mask_expanded = attention_mask.unsqueeze(-1).float()
        pooled = (hidden * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1)

        batch_size = pooled.size(0)
        device = pooled.device

        # Classify each operand slot with slot-aware representation
        source_logits = []
        for i in range(MAX_OPERANDS):
            # Add slot embedding so classifier knows which operand
            slot_emb = self.slot_embeddings(torch.tensor(i, device=device))
            slot_aware = pooled + slot_emb.unsqueeze(0).expand(batch_size, -1)
            logits = self.classifier(slot_aware)  # (batch, 5)
            source_logits.append(logits)

        # Stack: (batch, max_operands, 5)
        source_logits = torch.stack(source_logits, dim=1)

        return {'source_type_logits': source_logits}


def load_c3_source(checkpoint_path: str = None, device: str = "cuda"):
    """Load C3 source classifier from checkpoint."""
    model = C3SourceClassifier()

    if checkpoint_path:
        state = torch.load(checkpoint_path, map_location=device)
        if 'model_state_dict' in state:
            model.load_state_dict(state['model_state_dict'])
        else:
            model.load_state_dict(state)

    model = model.to(device)
    model.eval()
    return model


def predict_sources(model, tokenizer, problem_text: str, template: str):
    """Predict source types for a problem."""
    input_text = f"[TEMPLATE: {template}] {problem_text}"

    encodings = tokenizer(
        input_text,
        truncation=True,
        max_length=384,
        padding='max_length',
        return_tensors='pt'
    )

    device = next(model.parameters()).device
    input_ids = encodings['input_ids'].to(device)
    attention_mask = encodings['attention_mask'].to(device)

    with torch.no_grad():
        outputs = model(input_ids, attention_mask)
        probs = torch.softmax(outputs['source_type_logits'], dim=-1)
        preds = probs.argmax(dim=-1)[0]  # (max_operands,)
        confs = probs.max(dim=-1).values[0]  # (max_operands,)

    results = []
    for i in range(MAX_OPERANDS):
        source_type = SOURCE_TYPES[preds[i].item()]
        confidence = confs[i].item()
        if source_type != 'NONE':
            results.append({
                'slot': i,
                'source_type': source_type,
                'confidence': confidence
            })

    return results
