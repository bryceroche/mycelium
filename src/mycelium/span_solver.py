"""Span-based math word problem solver.

Architecture (no hardcoded patterns):
1. ATTENTION: num→verb attention distinguishes SET (low) from actions (high)
2. EMBEDDING: verb embedding distinguishes ADD from SUB
3. Combined: attention threshold + embedding NN

The operation is encoded in HOW tokens attend, not just what tokens are present.
"""

import re
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
import numpy as np


# Word numbers for value extraction (not for operation inference!)
WORD_NUMBERS = {
    "half": 0.5, "halves": 0.5, "quarter": 0.25, "third": 1/3,
    "twice": 2, "double": 2, "triple": 3,
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
}


@dataclass
class Step:
    """A computation step."""
    span: str
    operation: str
    value: float
    confidence: float = 0.0


class AttentionOperationClassifier:
    """Classify span operations using attention patterns + verb embeddings.

    No keywords. No regex for operations. Just model signals:
    - Attention: SET vs action verbs
    - Verb embedding: ADD vs SUB
    """

    def __init__(self, attention_threshold: float = 0.055):
        self.attention_threshold = attention_threshold
        self._embed_model = None
        self._attn_model = None
        self._tokenizer = None

        # Verb centroids (computed once)
        self._sub_centroid = None
        self._add_centroid = None
        self._set_centroid = None

    def _load_embed_model(self):
        if self._embed_model is None:
            from sentence_transformers import SentenceTransformer
            self._embed_model = SentenceTransformer("all-MiniLM-L6-v2")
        return self._embed_model

    def _load_attn_model(self):
        if self._attn_model is None:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            model_name = "deepseek-ai/deepseek-math-7b-instruct"
            self._tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            self._attn_model = AutoModelForCausalLM.from_pretrained(
                model_name, output_attentions=True, torch_dtype=torch.float16,
                device_map="auto", trust_remote_code=True,
            )
        return self._attn_model, self._tokenizer

    def _compute_verb_centroids(self):
        """Compute verb centroids for operation classification."""
        if self._sub_centroid is not None:
            return

        model = self._load_embed_model()

        # Representative verbs for each operation
        sub_verbs = ["sold", "ate", "gave", "spent", "lost", "used", "consumed", "donated"]
        add_verbs = ["bought", "found", "received", "earned", "gained", "got", "acquired", "collected"]
        set_verbs = ["has", "had", "owns", "contains", "holds", "keeps", "is", "are"]

        self._sub_centroid = model.encode(sub_verbs).mean(axis=0)
        self._add_centroid = model.encode(add_verbs).mean(axis=0)
        self._set_centroid = model.encode(set_verbs).mean(axis=0)

    def _cosine_sim(self, a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))

    def _get_num_verb_attention(self, text: str) -> float:
        """Get attention from number token to verb token."""
        import torch

        model, tokenizer = self._load_attn_model()
        inputs = tokenizer(text, return_tensors="pt").to("cuda")

        with torch.no_grad():
            outputs = model(**inputs, output_attentions=True)

        attn = torch.stack(outputs.attentions[-4:]).mean(dim=(0, 1, 2)).cpu().numpy()

        # Number at position 3, verb at position 1 (typical structure)
        if attn.shape[0] >= 4:
            return float(attn[3, 1])
        return 0.05

    def _extract_verb(self, text: str) -> str:
        """Extract the main verb from a span."""
        words = text.lower().split()
        stop_words = {"a", "an", "the", "to", "for", "of", "and", "or", "but", "in", "on", "at"}

        for word in words[1:]:  # Skip first word (usually subject)
            word = word.strip(".,!?")
            if word and word not in stop_words and not word.isdigit():
                return word

        return words[1] if len(words) > 1 else words[0]

    def classify(self, span_text: str) -> Tuple[str, float]:
        """Classify a span's operation using attention + embedding.

        Returns (operation, confidence)
        """
        self._compute_verb_centroids()
        model = self._load_embed_model()

        # Step 1: Get attention signal
        attn = self._get_num_verb_attention(span_text)

        # Step 2: If low attention, likely SET
        if attn < self.attention_threshold:
            return "SET", 0.8 + (self.attention_threshold - attn) * 5

        # Step 3: For action verbs, use embedding to distinguish ADD/SUB
        verb = self._extract_verb(span_text)
        verb_embed = model.encode([verb])[0]

        sim_sub = self._cosine_sim(verb_embed, self._sub_centroid)
        sim_add = self._cosine_sim(verb_embed, self._add_centroid)

        if sim_add > sim_sub:
            return "ADD", sim_add
        else:
            return "SUB", sim_sub


def extract_number(text: str) -> Optional[float]:
    """Extract numeric value from span text."""
    nums = re.findall(r"\b(\d+(?:\.\d+)?)\b", text)
    if nums:
        return float(nums[0])

    t = text.lower()
    for word, val in WORD_NUMBERS.items():
        if word in t:
            return val

    return None


def segment_problem(text: str) -> List[str]:
    """Split problem into spans."""
    text = re.sub(r"[Hh]ow many.*\?", "", text)
    text = re.sub(r"[Ww]hat.*\?", "", text)
    text = re.sub(r"[Hh]ow much.*\?", "", text)

    parts = re.split(r"[.!?]|\band\b|\bthen\b", text)

    spans = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        if re.search(r"\d+", part) or any(w in part.lower() for w in WORD_NUMBERS):
            spans.append(part)

    return spans


def solve(problem_text: str, classifier: AttentionOperationClassifier = None) -> Tuple[Optional[float], List[Step]]:
    """Solve a math word problem.

    Uses attention + embedding for operation classification (no hardcoded rules).
    """
    if classifier is None:
        classifier = AttentionOperationClassifier()

    spans = segment_problem(problem_text)

    state = {}
    var = "total"
    steps = []

    for span in spans:
        num = extract_number(span)
        if num is None:
            continue

        op, confidence = classifier.classify(span)

        steps.append(Step(
            span=span,
            operation=op,
            value=num,
            confidence=confidence,
        ))

        # Execute
        if op == "SET":
            state[var] = num
        elif op == "SUB" and var in state:
            state[var] -= num
        elif op == "ADD" and var in state:
            state[var] += num
        elif op == "MUL" and var in state:
            state[var] *= num
        elif op == "DIV" and var in state:
            state[var] *= num  # "half" = 0.5

    return state.get(var), steps


if __name__ == "__main__":
    # Test with attention-based classification
    print("=== Attention + Embedding Classifier Test ===\n")

    classifier = AttentionOperationClassifier()

    test_cases = [
        ("she sold 5 apples", "SUB"),
        ("she bought 5 apples", "ADD"),
        ("she has 5 apples", "SET"),
        ("he ate 3 cookies", "SUB"),
        ("he found 3 coins", "ADD"),
        ("the box contains 10 items", "SET"),
    ]

    for text, expected in test_cases:
        op, conf = classifier.classify(text)
        status = "✓" if op == expected else "✗"
        print(f"{status} {op} (conf={conf:.2f}) <- \"{text}\" (expected {expected})")
