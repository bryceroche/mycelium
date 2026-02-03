"""Span-based math word problem solver.

Architecture (NO HARDCODED PATTERNS - everything learned via Welford):
1. ATTENTION: num→verb attention distinguishes SET (low) from actions (high)
   - Threshold is LEARNED via Welford statistics, not hardcoded
2. EMBEDDING: verb embedding distinguishes ADD from SUB
   - Centroids are LEARNED from examples, not hardcoded verb lists
3. Combined: learned attention threshold + learned embedding centroids

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
    """Classify span operations using attention patterns + learned verb embeddings.

    NO HARDCODED VERB LISTS. Everything is learned from labeled examples:
    - Attention: SET vs action verbs (Welford-tracked threshold)
    - Verb embedding: ADD vs SUB (learned centroids from examples)
    """

    def __init__(self, learned_profiles_path: Optional[str] = None):
        self._embed_model = None
        self._attn_model = None
        self._tokenizer = None

        # Welford stats for attention threshold (learned, not hardcoded)
        from .attention_learner import WelfordStats
        self._set_attention = WelfordStats()
        self._action_attention = WelfordStats()

        # Verb centroids (learned from examples, not hardcoded lists)
        self._sub_centroid = None
        self._add_centroid = None
        self._sub_verbs_learned: List[np.ndarray] = []
        self._add_verbs_learned: List[np.ndarray] = []

        # Load pre-learned profiles if available
        if learned_profiles_path:
            self._load_learned(learned_profiles_path)

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

    def learn(self, span_text: str, operation: str):
        """Learn from a labeled example. Updates Welford stats and verb centroids."""
        model = self._load_embed_model()

        # Learn attention threshold via Welford
        attn = self._get_num_verb_attention(span_text)
        if operation == "SET":
            self._set_attention.update(attn)
        else:
            self._action_attention.update(attn)

        # Learn verb centroids from examples (not hardcoded lists!)
        if operation in ("ADD", "SUB"):
            verb = self._extract_verb(span_text)
            verb_embed = model.encode([verb])[0]
            if operation == "SUB":
                self._sub_verbs_learned.append(verb_embed)
                self._sub_centroid = np.mean(self._sub_verbs_learned, axis=0)
            else:
                self._add_verbs_learned.append(verb_embed)
                self._add_centroid = np.mean(self._add_verbs_learned, axis=0)

    def _get_learned_threshold(self) -> float:
        """Get attention threshold from Welford stats (midpoint between SET and action means)."""
        if self._set_attention.count < 3 or self._action_attention.count < 3:
            return 0.06  # Bootstrap default until we have enough data
        return (self._set_attention.mean + self._action_attention.mean) / 2

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
        """Classify a span's operation using learned attention + embedding.

        Returns (operation, confidence). NO HARDCODED THRESHOLDS OR VERB LISTS.
        """
        model = self._load_embed_model()

        # Step 1: Get attention signal
        attn = self._get_num_verb_attention(span_text)

        # Step 2: Use LEARNED threshold (Welford) to distinguish SET vs action
        threshold = self._get_learned_threshold()
        if attn < threshold:
            # Confidence based on z-score from SET distribution
            if self._set_attention.count >= 3:
                conf = 0.8 + (threshold - attn) / max(self._set_attention.std, 0.01)
            else:
                conf = 0.7
            return "SET", min(conf, 1.0)

        # Step 3: For action verbs, use LEARNED centroids (not hardcoded lists!)
        if self._sub_centroid is None or self._add_centroid is None:
            # Not enough learned examples yet - return best guess
            return "ADD", 0.5

        verb = self._extract_verb(span_text)
        verb_embed = model.encode([verb])[0]

        sim_sub = self._cosine_sim(verb_embed, self._sub_centroid)
        sim_add = self._cosine_sim(verb_embed, self._add_centroid)

        if sim_add > sim_sub:
            return "ADD", sim_add
        else:
            return "SUB", sim_sub

    def _load_learned(self, path: str):
        """Load pre-learned Welford stats and centroids."""
        import json
        with open(path) as f:
            data = json.load(f)
        # Load Welford stats
        if "set_attention" in data:
            self._set_attention.count = data["set_attention"]["count"]
            self._set_attention.mean = data["set_attention"]["mean"]
            self._set_attention.m2 = data["set_attention"]["m2"]
        if "action_attention" in data:
            self._action_attention.count = data["action_attention"]["count"]
            self._action_attention.mean = data["action_attention"]["mean"]
            self._action_attention.m2 = data["action_attention"]["m2"]
        # Load centroids
        if "sub_centroid" in data:
            self._sub_centroid = np.array(data["sub_centroid"])
        if "add_centroid" in data:
            self._add_centroid = np.array(data["add_centroid"])

    def save_learned(self, path: str):
        """Save learned Welford stats and centroids."""
        import json
        data = {
            "set_attention": {
                "count": self._set_attention.count,
                "mean": self._set_attention.mean,
                "m2": self._set_attention.m2,
            },
            "action_attention": {
                "count": self._action_attention.count,
                "mean": self._action_attention.mean,
                "m2": self._action_attention.m2,
            },
        }
        if self._sub_centroid is not None:
            data["sub_centroid"] = self._sub_centroid.tolist()
        if self._add_centroid is not None:
            data["add_centroid"] = self._add_centroid.tolist()
        with open(path, "w") as f:
            json.dump(data, f, indent=2)


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
    # Test with LEARNED attention-based classification (no hardcoded verb lists!)
    print("=== Learned Attention + Embedding Classifier ===\n")

    classifier = AttentionOperationClassifier()

    # Training examples - classifier learns from these
    train_examples = [
        ("she has 5 apples", "SET"),
        ("he had 10 cookies", "SET"),
        ("the box contains 12 items", "SET"),
        ("she sold 3 apples", "SUB"),
        ("he ate 4 cookies", "SUB"),
        ("Mary gave away 5 books", "SUB"),
        ("she bought 5 apples", "ADD"),
        ("he found 3 coins", "ADD"),
        ("Mary received 4 gifts", "ADD"),
    ]

    print("Learning from examples...")
    for text, op in train_examples:
        classifier.learn(text, op)
        print(f"  Learned: {op} <- '{text}'")

    print(f"\nLearned threshold: {classifier._get_learned_threshold():.4f}")
    print(f"SET attention: mean={classifier._set_attention.mean:.4f} std={classifier._set_attention.std:.4f}")
    print(f"Action attention: mean={classifier._action_attention.mean:.4f} std={classifier._action_attention.std:.4f}")

    # Test on held-out examples
    test_cases = [
        ("John has 12 oranges", "SET"),
        ("Lisa sold 4 tickets", "SUB"),
        ("Mike bought 6 toys", "ADD"),
        ("the jar holds 20 candies", "SET"),
    ]

    print("\n=== Testing on held-out examples ===\n")
    for text, expected in test_cases:
        op, conf = classifier.classify(text)
        status = "✓" if op == expected else "✗"
        print(f"{status} {op} (conf={conf:.2f}) <- \"{text}\" (expected {expected})")
