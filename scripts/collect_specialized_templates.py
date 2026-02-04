#!/usr/bin/env python3
"""
Collect specialized templates from GSM8K and MATH datasets.

For EVERY unique span we encounter:
1. Genericize it (names -> [NAME], numbers -> [N], etc.)
2. Compute embedding (MiniLM fine-tuned)
3. Extract attention features (entropy, received, connection) from Qwen-7B
4. Infer custom DSL expression
5. Create specialized template

Output: specialized_templates.json with one template per unique generic pattern.

USAGE:
    # Full extraction on VM with GPU:
    python scripts/collect_specialized_templates.py --dataset both --output specialized_templates.json

    # Quick test:
    python scripts/collect_specialized_templates.py --test-mode --num-samples 50
"""

import argparse
import json
import re
import os
import sys
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class CollectionConfig:
    """Configuration for template collection."""
    # Dataset
    dataset: str = "both"  # "gsm8k", "math", or "both"
    math_levels: List[str] = field(default_factory=lambda: ["Level 1", "Level 2"])
    num_samples: Optional[int] = None

    # Models
    qwen_model: str = "Qwen/Qwen2.5-Math-7B-Instruct"
    minilm_model: str = "models/minilm_attention_finetuned.pt"
    quantize: Optional[str] = "4bit"  # None, "4bit", "8bit"

    # Output
    output_path: str = "specialized_templates.json"

    # Processing
    batch_size: int = 1
    max_seq_length: int = 512


# =============================================================================
# Entity Genericization
# =============================================================================

# Common names to replace with [NAME]
COMMON_NAMES = [
    'James', 'John', 'Mary', 'Lisa', 'Tom', 'Sarah', 'Mike', 'Emma', 'Jake',
    'Janet', 'Bob', 'Alice', 'Sam', 'Amy', 'Tim', 'Jane', 'Mark', 'Anna',
    'David', 'Susan', 'Chris', 'Karen', 'Paul', 'Laura', 'Peter', 'Nancy',
    'Betty', 'Carol', 'Daniel', 'Helen', 'George', 'Ruth', 'Joseph', 'Sharon',
    'Brian', 'Donna', 'Ronald', 'Michelle', 'Kevin', 'Dorothy', 'Jason',
    'Melissa', 'Gary', 'Deborah', 'Timothy', 'Stephanie', 'Jose', 'Rebecca',
    'Larry', 'Sandra', 'Bobbie', 'Cynthia', 'Daragh', 'Rachelle', 'Gretchen',
    'Rocky', 'Harold', 'Zizi', 'Milton', 'Cary', 'Jamie', 'Biff', 'Kenneth',
    'Maria', 'Martha', 'Jennifer', 'Elizabeth', 'Linda', 'Barbara', 'Patricia',
    'Jessica', 'Angela', 'Brenda', 'Katherine', 'Nicole', 'Samantha', 'Daria',
    'Annie', 'Leo', 'Ella', 'Veronica', 'Velma', 'Nada', 'Wendi', 'Ryan',
    'Kelly', 'Dave', 'Jerry', 'Cyrus', 'Gunner', 'Faith', 'Natalie', 'Charles',
    'Andrea', 'Louie', 'Hendrix', 'Michiko', 'Morio', 'Danivan', 'Smith',
]

# Items/objects to replace
COMMON_ITEMS = [
    'apples', 'oranges', 'cookies', 'candies', 'marbles', 'dollars', 'books',
    'pencils', 'pens', 'toys', 'cars', 'bikes', 'eggs', 'cupcakes', 'flowers',
    'stamps', 'stickers', 'coins', 'balls', 'blocks', 'cards', 'shirts',
]


def genericize_span(text: str) -> str:
    """Convert a specific span to a generic pattern.

    "John sold 5 apples to Mary" -> "[NAME] sold [N] [ITEM] to [NAME]"
    """
    result = text

    # Replace numbers first (before names, in case names contain numbers)
    # Match: $80,000 or 80,000 or 3.14 or 42
    result = re.sub(r'\$[\d,]+\.?\d*', '[N]', result)
    result = re.sub(r'\b\d{1,3}(?:,\d{3})+\.?\d*\b', '[N]', result)  # Comma numbers
    result = re.sub(r'\b\d+\.?\d*\b', '[N]', result)  # Plain numbers

    # Replace names with [NAME]
    for name in COMMON_NAMES:
        result = re.sub(rf'\b{name}\b', '[NAME]', result, flags=re.IGNORECASE)

    # Replace pronouns
    for p in ['He', 'She', 'They', 'It', 'We', 'I']:
        result = re.sub(rf'\b{p}\b', '[SUBJ]', result)
        result = re.sub(rf'\b{p.lower()}\b', '[SUBJ]', result)
    for p in ['him', 'her', 'them', 'us', 'me']:
        result = re.sub(rf'\b{p}\b', '[OBJ]', result)
    for p in ['his', 'her', 'their', 'its', 'our', 'my', "John's", "Mary's"]:
        result = re.sub(rf'\b{p}\b', '[POSS]', result, flags=re.IGNORECASE)

    # Replace common items with [ITEM]
    for item in COMMON_ITEMS:
        result = re.sub(rf'\b{item}\b', '[ITEM]', result, flags=re.IGNORECASE)
        # Also singular form
        if item.endswith('s'):
            result = re.sub(rf'\b{item[:-1]}\b', '[ITEM]', result, flags=re.IGNORECASE)

    # Clean up multiple consecutive placeholders
    result = re.sub(r'\[NAME\]\s*\[NAME\]', '[NAME]', result)
    result = re.sub(r'\[N\]\s*\[N\]', '[N]', result)
    result = re.sub(r'\[ITEM\]\s*\[ITEM\]', '[ITEM]', result)

    return result


# =============================================================================
# DSL Inference
# =============================================================================

def infer_dsl(pattern: str, examples: List[str]) -> Tuple[str, str]:
    """Infer operation type and custom DSL from pattern.

    Returns: (operation_type, dsl_expr)
    """
    text = pattern.lower() + " " + " ".join(ex.lower() for ex in examples[:3])

    # === Price/revenue calculations (MUL) ===
    if re.search(r'(sells?|sold)\s+.*for\s+\[n\]', text):
        return ("MUL", "entity * value")
    if re.search(r'for\s+\[n\].*\b(each|per)\b', text):
        return ("MUL", "entity * value")
    if re.search(r'each\s+\w+\s+has\s+\[n\]', text):
        return ("MUL", "entity * value")

    # === Subtraction verbs ===
    sub_verbs = ['sold', 'gave', 'spent', 'lost', 'ate', 'used', 'took', 'baked',
                 'threw', 'lent', 'traded', 'donated', 'paid', 'drank', 'dropped']
    for verb in sub_verbs:
        if verb in text:
            return ("SUB", "entity - value")

    # === Addition verbs ===
    add_verbs = ['found', 'received', 'earned', 'won', 'bought', 'got', 'gained',
                 'collected', 'picked', 'gathered', 'saved', 'harvested']
    for verb in add_verbs:
        if verb in text:
            return ("ADD", "entity + value")

    # === Multiplication patterns ===
    if 'times' in text or 'doubled' in text or 'tripled' in text:
        return ("MUL", "entity * value")

    # === Division patterns ===
    if 'shared' in text and 'equally' in text:
        return ("DIV", "entity / value")
    if 'split' in text or 'divided' in text:
        return ("DIV", "entity / value")
    if 'half of' in text:
        return ("DIV", "entity / 2")

    # === Comparison patterns ===
    if 'more than' in text:
        return ("ADD", "ref + value")
    if 'less than' in text or 'fewer than' in text:
        return ("SUB", "ref - value")
    if 'twice as' in text:
        return ("MUL", "ref * 2")

    # === Set/initial patterns ===
    set_verbs = ['has', 'have', 'had', 'starts', 'started', 'owns', 'contains']
    for verb in set_verbs:
        if verb in text:
            return ("SET", "value")

    # Default
    return ("SET", "value")


# =============================================================================
# Span Extraction (sentence segmentation)
# =============================================================================

def extract_spans(problem_text: str) -> List[str]:
    """Extract operational spans from a problem.

    Splits on sentence boundaries and compound clauses.
    Filters out questions.
    """
    # Split on sentence endings and compound clauses
    parts = re.split(r'[.!?]|\band\b|\bthen\b', problem_text)

    spans = []
    for part in parts:
        part = part.strip()
        if not part:
            continue

        # Skip questions
        part_lower = part.lower()
        if any(q in part_lower for q in ['how many', 'how much', 'what is', 'what are', 'what was']):
            continue

        # Must have a number to be operational
        if re.search(r'\d+', part):
            spans.append(part)

    return spans


# =============================================================================
# Template Collection
# =============================================================================

@dataclass
class SpecializedTemplate:
    """A specialized template for a unique generic pattern."""
    template_id: str
    pattern: str  # Generic pattern: "[NAME] sold [N] [ITEM]"
    operation_type: str  # SET, ADD, SUB, MUL, DIV
    dsl_expr: str  # Custom DSL: "entity - value"

    # Dual-signal features
    embedding_centroid: List[float] = field(default_factory=list)  # 384-dim
    attention_entropy: float = 0.0  # How spread attention is
    attention_received: float = 0.0  # How much attention tokens receive
    attention_connection: float = 0.0  # Connectivity strength

    # Examples
    span_examples: List[str] = field(default_factory=list)
    count: int = 0

    # Welford stats for online learning
    welford_count: int = 0
    welford_mean: float = 0.0
    welford_m2: float = 0.0


def collect_templates(config: CollectionConfig) -> Dict[str, SpecializedTemplate]:
    """Collect specialized templates from datasets.

    Returns: Dict of pattern -> SpecializedTemplate
    """
    import torch
    from datasets import load_dataset

    print("=" * 60)
    print("Specialized Template Collection")
    print("=" * 60)

    # Load MiniLM for embeddings
    print("\nLoading MiniLM for embeddings...")
    try:
        from mycelium.dual_signal_templates import SpanDetector

        if os.path.exists(config.minilm_model):
            detector = SpanDetector(model_path=config.minilm_model)
            print(f"  Loaded fine-tuned MiniLM from {config.minilm_model}")
        else:
            detector = SpanDetector(model_path=None)
            print("  Using base MiniLM (no fine-tuned weights)")
    except Exception as e:
        print(f"  Warning: Could not load MiniLM: {e}")
        detector = None

    # Load Qwen for attention (optional, GPU required)
    qwen_model = None
    qwen_tokenizer = None
    try:
        if torch.cuda.is_available():
            print("\nLoading Qwen-7B for attention extraction...")
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

            model_kwargs = {
                "output_attentions": True,
                "trust_remote_code": True,
                "attn_implementation": "eager",
            }

            if config.quantize == "4bit":
                model_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                )
                model_kwargs["device_map"] = "auto"
            elif config.quantize == "8bit":
                model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
                model_kwargs["device_map"] = "auto"
            else:
                model_kwargs["torch_dtype"] = torch.float16
                model_kwargs["device_map"] = "auto"

            qwen_tokenizer = AutoTokenizer.from_pretrained(
                config.qwen_model, trust_remote_code=True
            )
            qwen_model = AutoModelForCausalLM.from_pretrained(
                config.qwen_model, **model_kwargs
            )
            qwen_model.eval()
            print(f"  Loaded {config.qwen_model}")
        else:
            print("\nNo GPU available - skipping Qwen attention extraction")
    except Exception as e:
        print(f"  Warning: Could not load Qwen: {e}")

    # Load datasets
    print("\nLoading datasets...")
    problems = []

    if config.dataset in ["gsm8k", "both"]:
        gsm = load_dataset("openai/gsm8k", "main", split="train")
        for i, item in enumerate(gsm):
            if config.num_samples and i >= config.num_samples // 2:
                break
            problems.append({"id": f"gsm8k_{i}", "text": item["question"]})
        print(f"  Loaded {len(problems)} GSM8K problems")

    if config.dataset in ["math", "both"]:
        for cfg in ['algebra', 'prealgebra', 'number_theory']:
            try:
                math_ds = load_dataset("EleutherAI/hendrycks_math", cfg, split="train")
                for i, item in enumerate(math_ds):
                    if item.get("level") in config.math_levels:
                        problems.append({"id": f"math_{cfg}_{i}", "text": item["problem"]})
            except Exception as e:
                print(f"  Warning: Could not load MATH/{cfg}: {e}")
        print(f"  Total problems: {len(problems)}")

    if config.num_samples:
        problems = problems[:config.num_samples]

    # Collect templates
    print(f"\nProcessing {len(problems)} problems...")
    templates: Dict[str, SpecializedTemplate] = {}

    for problem in tqdm(problems, desc="Collecting"):
        spans = extract_spans(problem["text"])

        for span in spans:
            # Genericize
            pattern = genericize_span(span)

            # Skip if pattern is too short or just placeholders
            if len(pattern) < 10 or pattern.count('[') > len(pattern.split()) // 2:
                continue

            # Create template ID from pattern
            pattern_id = re.sub(r'[^a-z0-9]', '_', pattern.lower())[:50]

            if pattern_id not in templates:
                # Infer operation and DSL
                op_type, dsl_expr = infer_dsl(pattern, [span])

                # Create new template
                templates[pattern_id] = SpecializedTemplate(
                    template_id=pattern_id,
                    pattern=pattern,
                    operation_type=op_type,
                    dsl_expr=dsl_expr,
                    span_examples=[span],
                    count=1,
                )

                # Compute embedding
                if detector:
                    try:
                        embedding, attention, _ = detector.extract_features(span)
                        embedding = embedding / (np.linalg.norm(embedding) + 1e-8)
                        templates[pattern_id].embedding_centroid = embedding.tolist()

                        # Compute attention features from MiniLM
                        if attention is not None and attention.size > 0:
                            att_flat = attention.flatten()
                            att_abs = np.abs(att_flat)
                            att_sum = np.sum(att_abs)
                            if att_sum > 0:
                                att_prob = att_abs / att_sum
                                att_prob = att_prob[att_prob > 1e-10]
                                entropy = -np.sum(att_prob * np.log(att_prob))
                                max_entropy = np.log(len(att_flat))
                                templates[pattern_id].attention_entropy = float(entropy / max_entropy) if max_entropy > 0 else 0
                            templates[pattern_id].attention_received = float(np.mean(attention))
                            templates[pattern_id].attention_connection = float(np.std(attention))
                    except Exception as e:
                        pass

                # Extract Qwen attention (optional)
                if qwen_model and qwen_tokenizer:
                    try:
                        inputs = qwen_tokenizer(span, return_tensors="pt", max_length=256, truncation=True)
                        inputs = {k: v.to(qwen_model.device) for k, v in inputs.items()}

                        with torch.no_grad():
                            outputs = qwen_model(**inputs, output_attentions=True)

                        # Get attention from middle layers (more semantic)
                        attentions = outputs.attentions
                        mid_layers = attentions[len(attentions)//3 : 2*len(attentions)//3]
                        att_stack = torch.stack(mid_layers).mean(dim=(0, 1, 2))  # Avg across layers, batch, heads
                        att_np = att_stack.cpu().numpy()

                        # Update attention features with Qwen's attention
                        att_flat = att_np.flatten()
                        att_abs = np.abs(att_flat)
                        att_sum = np.sum(att_abs)
                        if att_sum > 0:
                            att_prob = att_abs / att_sum
                            att_prob = att_prob[att_prob > 1e-10]
                            entropy = -np.sum(att_prob * np.log(att_prob))
                            max_entropy = np.log(len(att_flat))
                            templates[pattern_id].attention_entropy = float(entropy / max_entropy) if max_entropy > 0 else 0
                        templates[pattern_id].attention_received = float(np.mean(att_np))
                        templates[pattern_id].attention_connection = float(np.std(att_np))

                        del outputs
                        torch.cuda.empty_cache()
                    except Exception as e:
                        pass
            else:
                # Update existing template
                templates[pattern_id].span_examples.append(span)
                templates[pattern_id].count += 1

                # Update embedding centroid (running average)
                if detector and templates[pattern_id].embedding_centroid:
                    try:
                        embedding, _, _ = detector.extract_features(span)
                        embedding = embedding / (np.linalg.norm(embedding) + 1e-8)
                        old_centroid = np.array(templates[pattern_id].embedding_centroid)
                        n = templates[pattern_id].count
                        new_centroid = old_centroid * (n-1)/n + embedding / n
                        templates[pattern_id].embedding_centroid = new_centroid.tolist()
                    except:
                        pass

    print(f"\nCollected {len(templates)} unique templates")

    # Print distribution
    op_counts = defaultdict(int)
    for t in templates.values():
        op_counts[t.operation_type] += 1
    print("\nOperation distribution:")
    for op, count in sorted(op_counts.items()):
        print(f"  {op}: {count}")

    return templates


def save_templates(templates: Dict[str, SpecializedTemplate], output_path: str):
    """Save templates to JSON."""
    data = {tid: asdict(t) for tid, t in templates.items()}

    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"\nSaved {len(templates)} templates to {output_path}")


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Collect specialized templates")

    parser.add_argument("--dataset", choices=["gsm8k", "math", "both"], default="both")
    parser.add_argument("--num-samples", type=int, help="Limit samples")
    parser.add_argument("--output", "-o", default="specialized_templates.json")
    parser.add_argument("--quantize", choices=["4bit", "8bit"], default="4bit")
    parser.add_argument("--test-mode", action="store_true")

    args = parser.parse_args()

    config = CollectionConfig(
        dataset=args.dataset,
        num_samples=args.num_samples if not args.test_mode else 50,
        output_path=args.output,
        quantize=args.quantize,
    )

    templates = collect_templates(config)
    save_templates(templates, config.output_path)

    print("\nDone!")


if __name__ == "__main__":
    main()
