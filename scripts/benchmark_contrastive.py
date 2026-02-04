#!/usr/bin/env python3
"""
Benchmark GSM8K using contrastive-trained MiniLM embeddings.

This bypasses the Qwen attention signals entirely and uses pure
embedding similarity with operation-aware embeddings.
"""

import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
from dataclasses import dataclass
import re


@dataclass
class Template:
    """A DSL template with embedding."""
    template_id: str
    operation: str
    custom_dsl: str
    embedding: np.ndarray
    span_examples: List[str]


class ContrastiveEncoder(nn.Module):
    """Load and use the contrastive-trained encoder."""

    def __init__(self, checkpoint_path: str, device: str = "cuda"):
        super().__init__()
        from transformers import AutoModel, AutoTokenizer

        self.device = device

        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

        # Get model name from args if available
        model_name = checkpoint.get('args', {}).get('model_name', 'sentence-transformers/all-MiniLM-L6-v2')

        # Load base model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.transformer = AutoModel.from_pretrained(model_name)
        self.embedding_dim = self.transformer.config.hidden_size

        # Load projection head (same architecture as training)
        self.projection = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.ReLU(),
            nn.Linear(self.embedding_dim, 128),
        )

        # Load trained weights
        # The checkpoint contains 'model_state_dict' with both transformer and projection
        state_dict = checkpoint['model_state_dict']

        # Filter for projection weights
        projection_state = {k.replace('projection.', ''): v for k, v in state_dict.items() if k.startswith('projection.')}
        self.projection.load_state_dict(projection_state)

        # Filter for transformer weights
        transformer_state = {k.replace('transformer.', ''): v for k, v in state_dict.items() if k.startswith('transformer.')}
        self.transformer.load_state_dict(transformer_state, strict=False)

        self.to(device)
        self.eval()

        print(f"Loaded contrastive encoder from {checkpoint_path}")
        print(f"  Metrics: {checkpoint.get('metrics', {})}")

    def mean_pooling(self, model_output, attention_mask):
        """Mean pooling - take attention mask into account."""
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def encode(self, texts: List[str], use_projection: bool = False) -> np.ndarray:
        """Encode texts to embeddings."""
        with torch.no_grad():
            encoded = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors='pt'
            )
            encoded = {k: v.to(self.device) for k, v in encoded.items()}

            output = self.transformer(**encoded)
            embeddings = self.mean_pooling(output, encoded['attention_mask'])
            embeddings = F.normalize(embeddings, p=2, dim=1)

            if use_projection:
                embeddings = self.projection(embeddings)
                embeddings = F.normalize(embeddings, p=2, dim=1)

        return embeddings.cpu().numpy()


def load_templates(templates_path: str, encoder: ContrastiveEncoder) -> List[Template]:
    """Load templates and re-embed with contrastive encoder."""
    print(f"Loading templates from {templates_path}")

    with open(templates_path) as f:
        data = json.load(f)

    raw_templates = data.get('templates', data) if isinstance(data, dict) else data
    if isinstance(raw_templates, dict):
        raw_templates = list(raw_templates.values())

    print(f"  Found {len(raw_templates)} templates")

    # Re-embed span examples with contrastive encoder
    templates = []
    for t in raw_templates:
        spans = t.get('span_examples', [])
        if not spans:
            continue

        # Encode span examples and average
        embeddings = encoder.encode(spans[:10])  # Use up to 10 examples
        centroid = embeddings.mean(axis=0)
        centroid = centroid / np.linalg.norm(centroid)

        templates.append(Template(
            template_id=t.get('template_id', 'unknown'),
            operation=t.get('operation', t.get('operation_type', 'unknown')),
            custom_dsl=t.get('custom_dsl', 'value'),
            embedding=centroid,
            span_examples=spans,
        ))

    print(f"  Re-embedded {len(templates)} templates with contrastive encoder")
    return templates


def find_best_template(text: str, templates: List[Template], encoder: ContrastiveEncoder) -> Optional[Template]:
    """Find best matching template by embedding similarity."""
    text_emb = encoder.encode([text])[0]

    best_template = None
    best_sim = -1

    for t in templates:
        sim = np.dot(text_emb, t.embedding)
        if sim > best_sim:
            best_sim = sim
            best_template = t

    return best_template


WORD_TO_NUM = {
    'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4,
    'five': 5, 'six': 6, 'seven': 7, 'eight': 8, 'nine': 9,
    'ten': 10, 'eleven': 11, 'twelve': 12, 'thirteen': 13,
    'fourteen': 14, 'fifteen': 15, 'sixteen': 16, 'seventeen': 17,
    'eighteen': 18, 'nineteen': 19, 'twenty': 20, 'thirty': 30,
    'forty': 40, 'fifty': 50, 'sixty': 60, 'seventy': 70,
    'eighty': 80, 'ninety': 90, 'hundred': 100, 'thousand': 1000,
    'million': 1000000, 'billion': 1000000000,
    'half': 0.5, 'third': 1/3, 'quarter': 0.25, 'fourth': 0.25,
    'twice': 2, 'double': 2, 'triple': 3,
    'a': 1, 'an': 1,  # "a dozen" = 1 dozen
    'dozen': 12, 'couple': 2, 'few': 3, 'several': 4,
}

def word_to_number(text: str) -> str:
    """Convert word numbers to digits in text."""
    text_lower = text.lower()

    # Handle compound numbers like "twenty-three"
    for tens in ['twenty', 'thirty', 'forty', 'fifty', 'sixty', 'seventy', 'eighty', 'ninety']:
        for ones in ['one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']:
            compound = f"{tens}-{ones}"
            if compound in text_lower:
                value = WORD_TO_NUM[tens] + WORD_TO_NUM[ones]
                text = re.sub(compound, str(value), text, flags=re.IGNORECASE)
            compound_space = f"{tens} {ones}"
            if compound_space in text_lower:
                value = WORD_TO_NUM[tens] + WORD_TO_NUM[ones]
                text = re.sub(f"{tens}\\s+{ones}", str(value), text, flags=re.IGNORECASE)

    # Handle simple word numbers (sort by length to match longer words first)
    for word in sorted(WORD_TO_NUM.keys(), key=len, reverse=True):
        if word in ['a', 'an']:  # Skip these as they're too common
            continue
        pattern = r'\b' + word + r'\b'
        if re.search(pattern, text, re.IGNORECASE):
            text = re.sub(pattern, str(WORD_TO_NUM[word]), text, flags=re.IGNORECASE)

    return text


def extract_numbers(text: str) -> List[float]:
    """Extract numbers from text, including word numbers."""
    # First convert word numbers to digits
    text = word_to_number(text)

    # Match numbers: $80,000 or 80,000 or 80000 or 3.14
    # Handle currency symbols and comma-separated thousands
    pattern = r'[\$]?\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?'
    matches = re.findall(pattern, text)

    numbers = []
    for m in matches:
        try:
            # Remove $ and commas
            clean = m.replace('$', '').replace(',', '')
            numbers.append(float(clean))
        except:
            pass
    return numbers


def has_number(text: str) -> bool:
    """Check if text contains a number (digit or word)."""
    # Check for digits
    if re.search(r'\d', text):
        return True
    # Check for word numbers
    text_lower = text.lower()
    for word in WORD_TO_NUM.keys():
        if word in ['a', 'an']:
            continue
        if re.search(r'\b' + word + r'\b', text_lower):
            return True
    return False


def detect_reference_pattern(sentence: str) -> Optional[str]:
    """Detect if sentence refers to a previous result."""
    sentence_lower = sentence.lower()

    # Patterns that indicate reference to previous value
    ref_patterns = [
        (r'half\s+(?:that|as)\s+(?:much|many)', 'ref / 2'),
        (r'twice\s+(?:that|as)\s+(?:much|many)', 'ref * 2'),
        (r'double\s+(?:that|the)', 'ref * 2'),
        (r'triple\s+(?:that|the)', 'ref * 3'),
        (r'the\s+(?:same|rest|remainder|remaining)', 'ref'),
        (r'that\s+(?:much|many|amount)', 'ref'),
    ]

    for pattern, dsl in ref_patterns:
        if re.search(pattern, sentence_lower):
            return dsl
    return None


def solve_with_dsl(sentence: str, dsl: str, prev_result: Optional[float] = None) -> Optional[float]:
    """Execute a DSL expression."""
    # Check for reference patterns first
    ref_pattern = detect_reference_pattern(sentence)
    if ref_pattern and prev_result is not None:
        if ref_pattern == 'ref / 2':
            return prev_result / 2
        elif ref_pattern == 'ref * 2':
            return prev_result * 2
        elif ref_pattern == 'ref * 3':
            return prev_result * 3
        elif ref_pattern == 'ref':
            # Just reference, combine with extracted number if any
            numbers = extract_numbers(sentence)
            if numbers:
                return numbers[0]
            return prev_result

    numbers = extract_numbers(sentence)

    if not numbers:
        return None

    # Get the primary value (usually first or last number depending on context)
    value = numbers[-1] if len(numbers) > 0 else 0

    # For reference-based DSLs, use previous result
    if 'ref' in dsl and prev_result is not None:
        entity = prev_result
    else:
        entity = numbers[0] if len(numbers) > 1 else value

    try:
        if dsl == 'value':
            return value
        elif dsl == 'entity + value':
            return entity + value if len(numbers) > 1 else value
        elif dsl == 'entity - value':
            return entity - value if len(numbers) > 1 else -value
        elif dsl == 'entity * value':
            return entity * value if len(numbers) > 1 else value
        elif dsl == 'entity / value':
            return entity / value if len(numbers) > 1 and value != 0 else value
        elif dsl == 'ref + value':
            return (prev_result or 0) + value
        elif dsl == 'ref - value':
            return (prev_result or 0) - value
        elif dsl == 'ref * 2':
            return (prev_result or 0) * 2
        elif dsl == 'ref / 2':
            return (prev_result or 0) / 2
        elif dsl == 'entity / 2':
            return entity / 2
        else:
            return value
    except:
        return value


def split_into_clauses(text: str) -> List[str]:
    """Split text into clauses, handling 'and', commas, etc."""
    # First split by sentence boundaries
    sentences = re.split(r'[.!?]', text)

    clauses = []
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        # Split on " and " only if both sides have numbers
        if ' and ' in sentence.lower():
            parts = re.split(r'\s+and\s+', sentence, flags=re.IGNORECASE)
            for part in parts:
                part = part.strip()
                if part and has_number(part):
                    clauses.append(part)
        elif has_number(sentence):
            clauses.append(sentence)

    return clauses


def solve_problem(question: str, templates: List[Template], encoder: ContrastiveEncoder) -> Optional[float]:
    """Solve a GSM8K problem using contrastive template matching."""
    # Split into clauses (handles "and" within sentences)
    clauses = split_into_clauses(question)

    if not clauses:
        return None

    result = None
    for clause in clauses:
        template = find_best_template(clause, templates, encoder)
        if template:
            step_result = solve_with_dsl(clause, template.custom_dsl, result)
            if step_result is not None:
                result = step_result

    return result


def main():
    import argparse
    from datasets import load_dataset

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="models/minilm_contrastive.pt")
    parser.add_argument("--templates", default="operation_separated_templates.json")
    parser.add_argument("--num-problems", type=int, default=100)
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent

    # Load encoder
    device = "cuda" if torch.cuda.is_available() else "cpu"
    encoder = ContrastiveEncoder(str(project_root / args.checkpoint), device)

    # Load and re-embed templates
    templates = load_templates(str(project_root / args.templates), encoder)

    # Count by operation
    by_op = defaultdict(int)
    for t in templates:
        by_op[t.operation] += 1
    print(f"Templates by operation: {dict(by_op)}")

    # Load GSM8K
    print(f"\nLoading GSM8K test set...")
    ds = load_dataset('openai/gsm8k', 'main', split='test')

    # Benchmark
    print(f"\nRunning benchmark on {args.num_problems} problems...")
    correct = 0
    total = 0

    for i in range(args.num_problems):
        problem = ds[i]
        question = problem['question']
        answer_str = problem['answer'].split('####')[-1].strip()

        try:
            expected = float(answer_str.replace(',', ''))
        except:
            continue

        total += 1
        predicted = solve_problem(question, templates, encoder)

        if predicted is not None and abs(predicted - expected) < 0.01:
            correct += 1
            status = "CORRECT"
        else:
            status = f"WRONG (got {predicted})"

        if i < 10 or (i + 1) % 20 == 0:
            print(f"{i+1:3d}. {status} (expected={expected})")

    print(f"\nResults: {correct}/{total} = {100*correct/total:.1f}%")


if __name__ == "__main__":
    main()
