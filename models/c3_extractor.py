#!/usr/bin/env python3
"""
C3 Span Extractor (with Slot Embeddings)

SQuAD-style extractive QA for operand extraction.
Input: [TEMPLATE: DIV] [PRIOR_0: 24] problem text...
Output: start/end token positions per operand slot

SLOT EMBEDDINGS: Each operand slot gets a learned embedding that tells
the model "you're extracting operand 0 (dividend) vs operand 1 (divisor)".
This breaks the symmetry that caused duplicate span predictions.
"""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from typing import List, Dict, Any, Tuple

MODEL_NAME = "Qwen/Qwen2-0.5B"

MAX_OPERANDS = 4
MAX_SEQ_LEN = 384

# Word-to-value mapping for C4's resolve_operand
WORD_MAP = {
    'half': 0.5, 'quarter': 0.25, 'third': 1/3,
    'twice': 2, 'double': 2, 'triple': 3,
    'dozen': 12, 'score': 20, 'hundred': 100,
    'thousand': 1000, 'million': 1000000,
}


class C3SpanExtractor(nn.Module):
    """
    SQuAD-style span extraction with slot embeddings.
    
    Key change: slot_embeddings tell the model which operand slot
    it's extracting, breaking the symmetry that caused duplicate spans.
    """

    def __init__(self, backbone_name: str = MODEL_NAME):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(
            backbone_name,
            trust_remote_code=True,
            torch_dtype=torch.float32
        )
        hidden_size = self.backbone.config.hidden_size

        # Slot embeddings - learned per-slot context
        self.slot_embeddings = nn.Embedding(MAX_OPERANDS, hidden_size)
        
        # Single start/end heads - slot info comes from embeddings
        self.start_head = nn.Linear(hidden_size, 1)
        self.end_head = nn.Linear(hidden_size, 1)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            input_ids: (batch, seq_len)
            attention_mask: (batch, seq_len)

        Returns:
            start_logits: (batch, seq_len, max_operands)
            end_logits: (batch, seq_len, max_operands)
        """
        hidden = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask
        ).last_hidden_state  # (batch, seq, hidden)

        batch_size, seq_len, hidden_size = hidden.shape
        device = hidden.device

        # Compute logits per slot with slot-aware context
        start_logits_list = []
        end_logits_list = []
        
        for slot in range(MAX_OPERANDS):
            # Add slot embedding to hidden states
            slot_emb = self.slot_embeddings(torch.tensor(slot, device=device))  # (hidden,)
            slot_aware_hidden = hidden + slot_emb.unsqueeze(0).unsqueeze(0)  # (batch, seq, hidden)
            
            # Compute start/end logits for this slot
            start_logits_list.append(self.start_head(slot_aware_hidden))  # (batch, seq, 1)
            end_logits_list.append(self.end_head(slot_aware_hidden))
        
        # Stack: (batch, seq, max_operands)
        start_logits = torch.cat(start_logits_list, dim=-1)
        end_logits = torch.cat(end_logits_list, dim=-1)

        # Mask padding
        mask = attention_mask.unsqueeze(-1)  # (batch, seq, 1)
        start_logits = start_logits.masked_fill(mask == 0, float('-inf'))
        end_logits = end_logits.masked_fill(mask == 0, float('-inf'))

        return start_logits, end_logits


def load_c3_extractor(checkpoint_path: str = None, device: str = "cuda"):
    """Load C3 span extractor from checkpoint."""
    model = C3SpanExtractor()

    if checkpoint_path:
        state = torch.load(checkpoint_path, map_location=device)
        if 'model_state_dict' in state:
            model.load_state_dict(state['model_state_dict'], strict=False)
        else:
            model.load_state_dict(state, strict=False)

    model = model.to(device)
    model.eval()
    return model


def format_input_with_priors(
    problem_text: str,
    template: str,
    prior_results: List[float] = None
) -> str:
    """
    Format input with template and prior results as extractable spans.
    
    Uses 0-indexed PRIORs to match training format.
    """
    parts = [f"[TEMPLATE: {template}]"]

    if prior_results:
        for i, result in enumerate(prior_results):
            if isinstance(result, float) and result == int(result):
                result = int(result)
            parts.append(f"[PRIOR_{i}: {result}]")

    parts.append(problem_text)
    return " ".join(parts)


def extract_spans(
    model: C3SpanExtractor,
    tokenizer,
    problem_text: str,
    template: str,
    num_operands: int = 2,
    prior_results: List[float] = None,
) -> List[Dict[str, Any]]:
    """Extract operand spans from input text."""
    input_text = format_input_with_priors(problem_text, template, prior_results)

    encodings = tokenizer(
        input_text,
        truncation=True,
        max_length=MAX_SEQ_LEN,
        padding='max_length',
        return_tensors='pt',
    )

    device = next(model.parameters()).device
    input_ids = encodings['input_ids'].to(device)
    attention_mask = encodings['attention_mask'].to(device)

    with torch.no_grad():
        start_logits, end_logits = model(input_ids, attention_mask)

    start_logits = start_logits[0].transpose(0, 1)
    end_logits = end_logits[0].transpose(0, 1)

    results = []
    for i in range(min(num_operands, MAX_OPERANDS)):
        start_probs = torch.softmax(start_logits[i], dim=-1)
        end_probs = torch.softmax(end_logits[i], dim=-1)

        start_idx = start_logits[i].argmax().item()
        end_idx = end_logits[i].argmax().item()

        if end_idx < start_idx:
            end_idx = start_idx

        tokens = input_ids[0, start_idx:end_idx+1]
        span_text = tokenizer.decode(tokens, skip_special_tokens=True).strip()

        confidence = (start_probs[start_idx] * end_probs[end_idx]).item()

        results.append({
            'slot': i,
            'text': span_text,
            'start': start_idx,
            'end': end_idx,
            'confidence': confidence,
        })

    return results


def resolve_operand(span_text: str) -> Any:
    """C4's operand resolution - convert span to value."""
    span_text = span_text.strip().lower()

    try:
        return float(span_text)
    except ValueError:
        pass

    if span_text in WORD_MAP:
        return WORD_MAP[span_text]

    import sympy
    if len(span_text) == 1 and span_text.isalpha():
        return sympy.Symbol(span_text)

    return span_text
