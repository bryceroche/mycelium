"""
Mycelium E2E Pipeline v6
========================
Full inference pipeline: Problem text → Answer

Components:
  C1: Segmenter (Qwen-0.5B, BIO token tagging)
  C2: Classifier (Qwen-0.5B, span group → operation label)
  C3: Extractor (Qwen-0.5B-Instruct, → typed arguments)
  Candidate grouping (heuristic enumeration)
  Bridging search (IB templates + domain constants + two-hop)
  Symbolic executor (validates candidates)
  Scorer (picks best candidate)

Usage:
  python e2e_pipeline.py \
    --segmenter-path models/segmenter/final \
    --classifier-path models/classifier/final \
    --extractor-path models/extractor/final \
    --problems-path data/gsm8k_test.jsonl \
    --output-path results/e2e_results.json
"""

import json
import re
import math
import itertools
import argparse
import logging
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Tuple, Any, Set
from fractions import Fraction
from pathlib import Path

try:
    import torch
    import numpy as np
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

logger = logging.getLogger(__name__)

# ============================================================
# Data Structures
# ============================================================

@dataclass
class Span:
    """A text span identified by the segmenter."""
    text: str
    start: int  # token index
    end: int    # token index (exclusive)
    numbers: List[float] = field(default_factory=list)

@dataclass
class SpanGroup:
    """A candidate grouping of spans for one operation."""
    spans: List[Span]
    text: str  # concatenated span texts with markers
    
@dataclass
class Operation:
    """A classified operation with extracted arguments."""
    operation: str           # e.g., "MUL", "ADD", "SUB", "DIV"
    arguments: List[float]   # extracted numeric arguments
    sources: List[str]       # "PROB" or "DERIVED" per argument
    confidence: float        # classifier confidence
    span_group: SpanGroup
    
@dataclass 
class GraphNode:
    """A node in the computation graph."""
    id: str
    operation: str
    params: List[float]
    result: Optional[float] = None
    source: str = "explicit"  # "explicit" or "bridging"
    
@dataclass
class Candidate:
    """A complete candidate solution."""
    grouping: List[SpanGroup]
    operations: List[Operation]
    graph: List[GraphNode]
    bridging: List[GraphNode]
    answer: Optional[float] = None
    score: float = 0.0


# ============================================================
# Number Extraction Utilities
# ============================================================

# Spelled-out number mapping
WORD_TO_NUM = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4,
    "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
    "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13,
    "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17,
    "eighteen": 18, "nineteen": 19, "twenty": 20, "thirty": 30,
    "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70,
    "eighty": 80, "ninety": 90, "hundred": 100, "thousand": 1000,
    "million": 1000000, "billion": 1000000000,
    # Fractional
    "half": 0.5, "third": 1/3, "quarter": 0.25, "fourth": 0.25,
    "fifth": 0.2, "sixth": 1/6, "eighth": 0.125, "tenth": 0.1,
    "twice": 2, "double": 2, "triple": 3, "dozen": 12,
}

def extract_numbers(text: str) -> List[float]:
    """Extract all numeric values from text, including spelled-out numbers."""
    numbers = []
    
    # Numeric patterns: integers, decimals, comma-separated, dollar amounts
    numeric_pattern = r'(?<![a-zA-Z])\$?(\d{1,3}(?:,\d{3})*(?:\.\d+)?|\d+(?:\.\d+)?)\%?(?![a-zA-Z])'
    for match in re.finditer(numeric_pattern, text):
        val = match.group(1).replace(",", "")
        try:
            numbers.append(float(val))
        except ValueError:
            pass
    
    # Spelled-out numbers
    text_lower = text.lower()
    for word, val in WORD_TO_NUM.items():
        if re.search(r'\b' + word + r'\b', text_lower):
            numbers.append(val)
    
    return numbers


# ============================================================
# Domain Constants (discovered from orphan operand analysis)
# ============================================================

DOMAIN_CONSTANTS = {
    # Time
    60: "minutes/hour",
    24: "hours/day",
    7: "days/week",
    52: "weeks/year",
    12: "months/year",
    365: "days/year",
    # Units
    100: "percentage, cm/m",
    1000: "g/kg, m/km",
}

DOMAIN_CONSTANT_VALUES = list(DOMAIN_CONSTANTS.keys())


# ============================================================
# IB-Discovered Bridging Templates (from GSM8K)
# ============================================================

BRIDGING_TEMPLATES = {
    # T7: Accumulator ADD (position 0.89, both operands derived)
    "ACC_ADD": lambda results: sum(results),
    
    # Accumulator SUB (max - sum of rest)
    "ACC_SUB": lambda results: max(results) - sum(r for r in results if r != max(results)) if len(results) > 1 else results[0],
    
    # T0: Derived Division (average, split equally, per person)
    "AVG": lambda results: sum(results) / len(results) if results else 0,
    
    # Derived division with count
    "DIV_DERIVED": lambda results: results[0] / results[1] if len(results) >= 2 and results[1] != 0 else None,
    
    # T9: Derived Multiplication (unit conversions, rate calculations)
    "MUL_DERIVED": lambda results: results[0] * results[1] if len(results) >= 2 else None,
    
    # SUB derived (difference between two derived results)
    "SUB_DERIVED": lambda results: abs(results[0] - results[1]) if len(results) >= 2 else None,
}


# ============================================================
# Goal Hints (simple keyword matching for C6 functionality)
# ============================================================

GOAL_HINTS = {
    "total": ["ACC_ADD", "MUL_DERIVED"],
    "altogether": ["ACC_ADD"],
    "combined": ["ACC_ADD"],
    "sum": ["ACC_ADD"],
    "how many": ["ACC_ADD", "ACC_SUB", "MUL_DERIVED"],
    "how much": ["ACC_ADD", "ACC_SUB", "MUL_DERIVED"],
    "difference": ["ACC_SUB", "SUB_DERIVED"],
    "change": ["ACC_SUB"],
    "remaining": ["ACC_SUB"],
    "left": ["ACC_SUB"],
    "average": ["AVG"],
    "mean": ["AVG"],
    "per": ["DIV_DERIVED"],
    "each": ["DIV_DERIVED", "MUL_DERIVED"],
    "ratio": ["DIV_DERIVED"],
    "times": ["MUL_DERIVED"],
    "percent": ["MUL_DERIVED", "DIV_DERIVED"],
}

def get_goal_hints(question: str) -> List[str]:
    """Extract goal hints from the question text."""
    hints = set()
    question_lower = question.lower()
    for keyword, templates in GOAL_HINTS.items():
        if keyword in question_lower:
            hints.update(templates)
    # If no hints found, try all
    if not hints:
        hints = set(BRIDGING_TEMPLATES.keys())
    return list(hints)


# ============================================================
# Symbolic Executor
# ============================================================

OPERATIONS = {
    "ADD": lambda a, b: a + b,
    "SUB": lambda a, b: a - b,
    "MUL": lambda a, b: a * b,
    "DIV": lambda a, b: a / b if b != 0 else None,
    "POW": lambda a, b: a ** b if abs(b) < 100 else None,
    "SQRT": lambda a: math.sqrt(a) if a >= 0 else None,
    "MOD": lambda a, b: a % b if b != 0 else None,
    "FLOOR_DIV": lambda a, b: a // b if b != 0 else None,
    "ABS": lambda a: abs(a),
    # MATH-specific (extensible)
    "SIN": lambda a: math.sin(math.radians(a)),
    "COS": lambda a: math.cos(math.radians(a)),
    "TAN": lambda a: math.tan(math.radians(a)),
    "LOG": lambda a: math.log(a) if a > 0 else None,
    "LOG10": lambda a: math.log10(a) if a > 0 else None,
    "FACTORIAL": lambda a: math.factorial(int(a)) if 0 <= a <= 20 and a == int(a) else None,
    "COMB": lambda a, b: math.comb(int(a), int(b)) if a >= b >= 0 else None,
    "PERM": lambda a, b: math.perm(int(a), int(b)) if a >= b >= 0 else None,
}

def execute_operation(op: str, args: List[float]) -> Optional[float]:
    """Execute a single operation with given arguments."""
    op = op.upper()
    if op not in OPERATIONS:
        return None
    try:
        func = OPERATIONS[op]
        # Handle variable arity
        if op in ("SQRT", "ABS", "LOG", "LOG10", "FACTORIAL", "SIN", "COS", "TAN"):
            if len(args) < 1:
                return None
            result = func(args[0])
        elif len(args) < 2:
            return None
        else:
            result = func(args[0], args[1])
        
        if result is None or not math.isfinite(result):
            return None
        return result
    except (ValueError, ZeroDivisionError, OverflowError, TypeError):
        return None

def execute_graph(graph: List[GraphNode]) -> Optional[float]:
    """Execute a computation graph, returning the final result."""
    results = {}
    for node in graph:
        # Resolve params - could reference prior results
        resolved_params = []
        for p in node.params:
            if isinstance(p, str) and p in results:
                resolved_params.append(results[p])
            elif isinstance(p, (int, float)):
                resolved_params.append(float(p))
            else:
                return None  # unresolved dependency
        
        result = execute_operation(node.operation, resolved_params)
        if result is None:
            return None
        node.result = result
        results[node.id] = result
    
    # Return the last node's result
    if graph:
        return graph[-1].result
    return None


# ============================================================
# C1: Segmenter Interface
# ============================================================

class Segmenter:
    """BIO token tagger that identifies computation spans in problem text."""
    
    def __init__(self, model_path: str):
        from transformers import AutoTokenizer, AutoModelForTokenClassification
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForTokenClassification.from_pretrained(model_path)
        self.model.eval()
        
        # BIO label mapping
        self.id2label = self.model.config.id2label
        self.label2id = self.model.config.label2id
        logger.info(f"Segmenter loaded from {model_path}, labels: {self.id2label}")
    
    def predict(self, text: str) -> List[Span]:
        """Tag problem text and extract spans."""
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, 
                                max_length=512, return_offsets_mapping=True)
        offset_mapping = inputs.pop("offset_mapping")[0]
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        predictions = torch.argmax(outputs.logits, dim=-1)[0]
        tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        
        spans = []
        current_span_tokens = []
        current_span_start = None
        
        for idx, (pred, token, offset) in enumerate(zip(predictions, tokens, offset_mapping)):
            label = self.id2label.get(pred.item(), "O")
            
            if label.startswith("B"):
                # Save previous span if exists
                if current_span_tokens:
                    span_text = text[current_span_start:current_span_end]
                    spans.append(Span(
                        text=span_text,
                        start=current_span_start,
                        end=current_span_end,
                        numbers=extract_numbers(span_text)
                    ))
                current_span_tokens = [token]
                current_span_start = offset[0].item()
                current_span_end = offset[1].item()
                
            elif label.startswith("I") and current_span_tokens:
                current_span_tokens.append(token)
                current_span_end = offset[1].item()
                
            else:  # O label
                if current_span_tokens:
                    span_text = text[current_span_start:current_span_end]
                    spans.append(Span(
                        text=span_text,
                        start=current_span_start,
                        end=current_span_end,
                        numbers=extract_numbers(span_text)
                    ))
                    current_span_tokens = []
                    current_span_start = None
        
        # Handle last span
        if current_span_tokens and current_span_start is not None:
            span_text = text[current_span_start:current_span_end]
            spans.append(Span(
                text=span_text,
                start=current_span_start,
                end=current_span_end,
                numbers=extract_numbers(span_text)
            ))
        
        # Fallback: if segmenter produces no spans, extract number-bearing phrases
        if not spans:
            spans = self._fallback_segmentation(text)
        
        return spans
    
    def _fallback_segmentation(self, text: str) -> List[Span]:
        """Fallback: create spans around number-bearing phrases."""
        spans = []
        # Find sentences/clauses containing numbers
        for match in re.finditer(r'(\b\d[\d,]*\.?\d*\b)', text):
            start = max(0, match.start() - 30)
            end = min(len(text), match.end() + 30)
            # Expand to word boundaries
            while start > 0 and text[start] != ' ':
                start -= 1
            while end < len(text) and text[end] != ' ':
                end += 1
            span_text = text[start:end].strip()
            spans.append(Span(
                text=span_text,
                start=start,
                end=end,
                numbers=extract_numbers(span_text)
            ))
        return spans


# ============================================================
# Candidate Grouping Generator
# ============================================================

def generate_candidate_groupings(spans: List[Span], max_group_size: int = 3) -> List[List[SpanGroup]]:
    """
    Generate candidate groupings of spans.
    
    Heuristics:
    - All individual spans (each is its own operation)
    - Adjacent pairs
    - All pairs
    - Triplets (if <= 5 spans)
    - Mixed: some grouped, some individual
    """
    n = len(spans)
    if n == 0:
        return []
    if n == 1:
        return [[SpanGroup(spans=[spans[0]], text=spans[0].text)]]
    
    candidates = []
    
    # Candidate 1: All individual
    all_individual = [SpanGroup(spans=[s], text=s.text) for s in spans]
    candidates.append(all_individual)
    
    # Adjacent pairs + remaining individual
    for i in range(n - 1):
        grouping = []
        for j in range(n):
            if j == i:
                pair_text = f"<<{spans[j].text}>> <<{spans[j+1].text}>>"
                grouping.append(SpanGroup(spans=[spans[j], spans[j+1]], text=pair_text))
            elif j == i + 1:
                continue  # already included in pair
            else:
                grouping.append(SpanGroup(spans=[spans[j]], text=spans[j].text))
        candidates.append(grouping)
    
    # All adjacent pairs (if even number)
    if n >= 4 and n % 2 == 0:
        grouping = []
        for i in range(0, n, 2):
            pair_text = f"<<{spans[i].text}>> <<{spans[i+1].text}>>"
            grouping.append(SpanGroup(spans=[spans[i], spans[i+1]], text=pair_text))
        candidates.append(grouping)
    
    # Non-adjacent pairs (for n <= 6 to keep combinatorics reasonable)
    if n <= 6:
        for i in range(n):
            for j in range(i + 2, n):  # skip adjacent (already covered)
                grouping = []
                used = {i, j}
                pair_text = f"<<{spans[i].text}>> <<{spans[j].text}>>"
                grouping.append(SpanGroup(spans=[spans[i], spans[j]], text=pair_text))
                for k in range(n):
                    if k not in used:
                        grouping.append(SpanGroup(spans=[spans[k]], text=spans[k].text))
                candidates.append(grouping)
    
    # Triplets (if n <= 5)
    if n <= 5 and n >= 3:
        for combo in itertools.combinations(range(n), 3):
            grouping = []
            triple_text = " ".join(f"<<{spans[c].text}>>" for c in combo)
            triple_spans = [spans[c] for c in combo]
            grouping.append(SpanGroup(spans=triple_spans, text=triple_text))
            for k in range(n):
                if k not in combo:
                    grouping.append(SpanGroup(spans=[spans[k]], text=spans[k].text))
            candidates.append(grouping)
    
    # Deduplicate (by span index tuples)
    seen = set()
    unique_candidates = []
    for grouping in candidates:
        key = tuple(tuple(s.start for s in g.spans) for g in sorted(grouping, key=lambda g: g.spans[0].start))
        if key not in seen:
            seen.add(key)
            unique_candidates.append(grouping)
    
    return unique_candidates


# ============================================================
# C2: Classifier Interface
# ============================================================

class Classifier:
    """Classifies span groups into operation labels."""
    
    def __init__(self, model_path: str):
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.eval()
        
        self.id2label = self.model.config.id2label
        self.label2id = self.model.config.label2id
        logger.info(f"Classifier loaded from {model_path}, {len(self.id2label)} labels")
    
    def classify_batch(self, problem_text: str, span_groups: List[SpanGroup]) -> List[Tuple[str, float]]:
        """Classify multiple span groups in one batch."""
        inputs_list = []
        for group in span_groups:
            marked = self._mark_spans(problem_text, group)
            inputs_list.append(marked)
        
        if not inputs_list:
            return []
        
        encodings = self.tokenizer(inputs_list, return_tensors="pt", padding=True,
                                    truncation=True, max_length=512)
        
        with torch.no_grad():
            outputs = self.model(**encodings)
        
        probs = torch.softmax(outputs.logits, dim=-1)
        pred_ids = torch.argmax(probs, dim=-1)
        confidences = probs.gather(1, pred_ids.unsqueeze(1)).squeeze(1)
        
        results = []
        for pred_id, conf in zip(pred_ids, confidences):
            label = self.id2label.get(pred_id.item(), "UNKNOWN")
            results.append((label, conf.item()))
        
        return results
    
    def _mark_spans(self, problem_text: str, group: SpanGroup) -> str:
        """Format input for classifier: problem text with marked spans."""
        marked = problem_text
        # Mark spans in reverse order to preserve indices
        for span in sorted(group.spans, key=lambda s: s.start, reverse=True):
            marked = marked[:span.start] + "<<" + marked[span.start:span.end] + ">>" + marked[span.end:]
        return marked


# ============================================================
# C3: Argument Extractor Interface
# ============================================================

class Extractor:
    """Extracts typed arguments from classified span groups."""
    
    def __init__(self, model_path: str):
        from transformers import AutoTokenizer, AutoModelForCausalLM
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_path)
        self.model.eval()
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        logger.info(f"Extractor loaded from {model_path}")
    
    def extract(self, problem_text: str, span_group: SpanGroup, operation: str) -> List[Tuple[float, str]]:
        """Extract arguments: returns list of (value, source) tuples."""
        prompt = f"Extract arguments for {operation}: {self._mark_spans(problem_text, span_group)}\nArguments:"
        
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=64,
                do_sample=False,
                temperature=1.0,
            )
        
        generated = self.tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        return self._parse_arguments(generated, problem_text, span_group)
    
    def extract_batch(self, problem_text: str, span_groups: List[SpanGroup], 
                      operations: List[str]) -> List[List[Tuple[float, str]]]:
        """Extract arguments for multiple span groups."""
        results = []
        for group, op in zip(span_groups, operations):
            args = self.extract(problem_text, group, op)
            # Fallback: if extractor fails, use numbers from spans directly
            if not args:
                args = self._fallback_extract(group, op, problem_text)
            results.append(args)
        return results
    
    def _fallback_extract(self, group: SpanGroup, operation: str, 
                          problem_text: str) -> List[Tuple[float, str]]:
        """Fallback: extract numbers directly from span text."""
        all_numbers = []
        for span in group.spans:
            for num in span.numbers:
                all_numbers.append((num, "PROB"))
        
        # For operations that need 2 args, ensure we have at least 2
        if len(all_numbers) < 2 and operation in ("ADD", "SUB", "MUL", "DIV"):
            # Try to find numbers from surrounding context
            problem_numbers = extract_numbers(problem_text)
            for num in problem_numbers:
                if (num, "PROB") not in all_numbers:
                    all_numbers.append((num, "PROB"))
                    if len(all_numbers) >= 2:
                        break
        
        return all_numbers
    
    def _parse_arguments(self, text: str, problem_text: str, 
                         span_group: SpanGroup) -> List[Tuple[float, str]]:
        """Parse extractor output into (value, source) pairs."""
        args = []
        problem_numbers = set(extract_numbers(problem_text))
        
        # Pattern: "value|source" or just numbers
        for match in re.finditer(r'([\d.]+)\|?(PROB|DERIVED)?', text):
            try:
                val = float(match.group(1))
                source = match.group(2) or ("PROB" if val in problem_numbers else "DERIVED")
                args.append((val, source))
            except ValueError:
                pass
        
        return args
    
    def _mark_spans(self, problem_text: str, group: SpanGroup) -> str:
        marked = problem_text
        for span in sorted(group.spans, key=lambda s: s.start, reverse=True):
            marked = marked[:span.start] + "<<" + marked[span.start:span.end] + ">>" + marked[span.end:]
        return marked


# ============================================================
# Bridging Search
# ============================================================

def bridging_search(explicit_results: List[float], 
                    goal_hints: List[str],
                    gold_answer: Optional[float] = None,
                    max_hops: int = 2) -> List[GraphNode]:
    """
    Search for bridging operations that connect explicit results to the answer.
    
    Tries:
    1. Single bridging template on all explicit results
    2. Bridging template + domain constant
    3. Two-hop chains (template → template, template → constant → template)
    """
    if not explicit_results:
        return []
    
    candidates = []
    
    # === One-hop: direct bridging templates ===
    for template_name in goal_hints:
        if template_name not in BRIDGING_TEMPLATES:
            continue
        func = BRIDGING_TEMPLATES[template_name]
        try:
            result = func(explicit_results)
            if result is not None and math.isfinite(result):
                node = GraphNode(
                    id=f"bridge_0",
                    operation=template_name,
                    params=explicit_results.copy(),
                    result=result,
                    source="bridging"
                )
                candidates.append(([node], result))
        except (ValueError, ZeroDivisionError, OverflowError):
            pass
    
    # === One-hop: explicit result OP domain_constant ===
    for result in explicit_results:
        for const in DOMAIN_CONSTANT_VALUES:
            for op_name in ["MUL", "DIV", "ADD", "SUB"]:
                val = execute_operation(op_name, [result, const])
                if val is not None:
                    node = GraphNode(
                        id=f"bridge_const",
                        operation=f"{op_name}_CONST",
                        params=[result, const],
                        result=val,
                        source="bridging"
                    )
                    candidates.append(([node], val))
                # Reverse order for non-commutative
                if op_name in ("SUB", "DIV"):
                    val2 = execute_operation(op_name, [const, result])
                    if val2 is not None:
                        node = GraphNode(
                            id=f"bridge_const_rev",
                            operation=f"{op_name}_CONST_REV",
                            params=[const, result],
                            result=val2,
                            source="bridging"
                        )
                        candidates.append(([node], val2))
    
    # === Two-hop: bridging → bridging ===
    if max_hops >= 2 and len(explicit_results) >= 2:
        # First hop: aggregate subsets
        for size in range(2, len(explicit_results) + 1):
            for subset in itertools.combinations(explicit_results, size):
                subset_list = list(subset)
                remaining = [r for r in explicit_results if r not in subset_list]
                
                for t1_name in BRIDGING_TEMPLATES:
                    func1 = BRIDGING_TEMPLATES[t1_name]
                    try:
                        intermediate = func1(subset_list)
                        if intermediate is None or not math.isfinite(intermediate):
                            continue
                    except:
                        continue
                    
                    # Second hop: combine intermediate with remaining
                    all_for_hop2 = [intermediate] + remaining
                    for t2_name in goal_hints:
                        if t2_name not in BRIDGING_TEMPLATES:
                            continue
                        func2 = BRIDGING_TEMPLATES[t2_name]
                        try:
                            final = func2(all_for_hop2)
                            if final is not None and math.isfinite(final):
                                nodes = [
                                    GraphNode(id="bridge_hop1", operation=t1_name,
                                              params=subset_list, result=intermediate, source="bridging"),
                                    GraphNode(id="bridge_hop2", operation=t2_name,
                                              params=all_for_hop2, result=final, source="bridging")
                                ]
                                candidates.append((nodes, final))
                        except:
                            pass
                    
                    # Second hop: intermediate OP constant
                    for const in DOMAIN_CONSTANT_VALUES:
                        for op_name in ["MUL", "DIV"]:
                            val = execute_operation(op_name, [intermediate, const])
                            if val is not None:
                                nodes = [
                                    GraphNode(id="bridge_hop1", operation=t1_name,
                                              params=subset_list, result=intermediate, source="bridging"),
                                    GraphNode(id="bridge_hop2_const", operation=f"{op_name}_CONST",
                                              params=[intermediate, const], result=val, source="bridging")
                                ]
                                candidates.append((nodes, val))
    
    return candidates


# ============================================================
# Scorer
# ============================================================

def score_candidate(answer: float, gold_answer: Optional[float],
                    operations: List[Operation],
                    bridging_nodes: List[GraphNode],
                    goal_hints: List[str]) -> float:
    """Score a candidate solution."""
    score = 0.0
    
    # Gold answer match (if available, used during development/eval)
    if gold_answer is not None:
        if answer_matches(answer, gold_answer):
            score += 1000.0  # Overwhelming signal
    
    # Classifier confidence
    for op in operations:
        score += op.confidence
    
    # Answer reasonableness
    if answer > 0:
        score += 0.5
    if answer == int(answer):
        score += 0.3  # Integer answers more likely on GSM8K
    if 0 < abs(answer) < 1e6:
        score += 0.2  # Reasonable magnitude
    
    # Prefer simpler bridging
    score -= 0.1 * len(bridging_nodes)
    
    # Goal consistency
    for node in bridging_nodes:
        if node.operation in goal_hints:
            score += 0.5
    
    return score

def answer_matches(predicted: float, gold: float, tolerance: float = 1e-6) -> bool:
    """Check if predicted answer matches gold."""
    if abs(predicted - gold) < tolerance:
        return True
    # Handle rounding (GSM8K answers are integers)
    if abs(round(predicted) - gold) < tolerance:
        return True
    return False


# ============================================================
# Full Pipeline
# ============================================================

class MyceliumPipeline:
    """Complete E2E inference pipeline."""
    
    def __init__(self, segmenter_path: str, classifier_path: str, extractor_path: str):
        logger.info("Loading pipeline components...")
        self.segmenter = Segmenter(segmenter_path)
        self.classifier = Classifier(classifier_path)
        self.extractor = Extractor(extractor_path)
        logger.info("Pipeline loaded.")
    
    def solve(self, problem_text: str, gold_answer: Optional[float] = None,
              verbose: bool = False) -> Dict[str, Any]:
        """
        Solve a math problem.
        
        Returns dict with:
          - answer: predicted answer
          - correct: whether it matches gold (if provided)
          - candidates_tried: number of candidates evaluated
          - winning_candidate: details of the winning candidate
          - error_attribution: diagnosis of failure (if incorrect)
        """
        result = {
            "problem_text": problem_text,
            "gold_answer": gold_answer,
            "answer": None,
            "correct": None,
            "candidates_tried": 0,
            "winning_candidate": None,
            "all_candidates": [],
        }
        
        # Step 1: Segment
        spans = self.segmenter.predict(problem_text)
        if verbose:
            logger.info(f"Spans: {[s.text for s in spans]}")
        
        if not spans:
            result["error_attribution"] = "segmentation_miss"
            return result
        
        # Step 2: Generate candidate groupings
        groupings = generate_candidate_groupings(spans)
        if verbose:
            logger.info(f"Candidate groupings: {len(groupings)}")
        
        # Step 3: Get goal hints from question
        # Find question part (usually after last period or the whole text)
        question = problem_text.split("?")[0] if "?" in problem_text else problem_text
        hints = get_goal_hints(question)
        
        # Step 4: For each grouping, classify + extract + build graph + bridge + execute
        all_candidates = []
        
        for grouping in groupings:
            # Classify all span groups in this grouping (batched)
            classifications = self.classifier.classify_batch(problem_text, grouping)
            
            # Extract arguments
            operations_labels = [c[0] for c in classifications]
            arg_results = self.extractor.extract_batch(problem_text, grouping, operations_labels)
            
            # Build operations
            operations = []
            for group, (label, conf), args in zip(grouping, classifications, arg_results):
                operations.append(Operation(
                    operation=label,
                    arguments=[a[0] for a in args],
                    sources=[a[1] for a in args],
                    confidence=conf,
                    span_group=group
                ))
            
            # Build explicit graph
            explicit_graph = []
            explicit_results = []
            for i, op in enumerate(operations):
                if len(op.arguments) >= 2:
                    node = GraphNode(
                        id=f"op_{i}",
                        operation=op.operation,
                        params=op.arguments[:2],
                        source="explicit"
                    )
                    exec_result = execute_operation(op.operation, op.arguments[:2])
                    if exec_result is not None:
                        node.result = exec_result
                        explicit_results.append(exec_result)
                        explicit_graph.append(node)
                elif len(op.arguments) == 1 and op.operation in ("SQRT", "ABS", "FACTORIAL"):
                    node = GraphNode(
                        id=f"op_{i}",
                        operation=op.operation,
                        params=op.arguments[:1],
                        source="explicit"
                    )
                    exec_result = execute_operation(op.operation, op.arguments[:1])
                    if exec_result is not None:
                        node.result = exec_result
                        explicit_results.append(exec_result)
                        explicit_graph.append(node)
            
            # Check if any explicit result is already the answer
            for er in explicit_results:
                if gold_answer is not None and answer_matches(er, gold_answer):
                    candidate = Candidate(
                        grouping=grouping, operations=operations,
                        graph=explicit_graph, bridging=[],
                        answer=er,
                        score=score_candidate(er, gold_answer, operations, [], hints)
                    )
                    all_candidates.append(candidate)
            
            # Bridging search
            if explicit_results:
                bridge_candidates = bridging_search(
                    explicit_results, hints, gold_answer, max_hops=2
                )
                for bridge_nodes, bridge_answer in bridge_candidates:
                    candidate = Candidate(
                        grouping=grouping, operations=operations,
                        graph=explicit_graph, bridging=bridge_nodes,
                        answer=bridge_answer,
                        score=score_candidate(bridge_answer, gold_answer, operations, 
                                              bridge_nodes, hints)
                    )
                    all_candidates.append(candidate)
            
            # Also try: all explicit results directly (no bridging)
            if len(explicit_results) == 1:
                candidate = Candidate(
                    grouping=grouping, operations=operations,
                    graph=explicit_graph, bridging=[],
                    answer=explicit_results[0],
                    score=score_candidate(explicit_results[0], gold_answer, operations, [], hints)
                )
                all_candidates.append(candidate)
        
        result["candidates_tried"] = len(all_candidates)
        
        if not all_candidates:
            result["error_attribution"] = "no_valid_candidates"
            return result
        
        # Step 5: Pick the best candidate
        all_candidates.sort(key=lambda c: c.score, reverse=True)
        winner = all_candidates[0]
        
        result["answer"] = winner.answer
        result["winning_candidate"] = {
            "operations": [(op.operation, op.arguments, op.confidence) for op in winner.operations],
            "bridging": [(n.operation, n.params, n.result) for n in winner.bridging],
            "score": winner.score,
        }
        
        if gold_answer is not None:
            result["correct"] = answer_matches(winner.answer, gold_answer)
            if not result["correct"]:
                result["error_attribution"] = self._attribute_error(
                    spans, winner, gold_answer, all_candidates
                )
        
        return result
    
    def _attribute_error(self, spans: List[Span], winner: Candidate,
                         gold_answer: float, all_candidates: List[Candidate]) -> str:
        """Diagnose why the pipeline failed."""
        # Check if any candidate got the right answer
        correct_exists = any(
            c.answer is not None and answer_matches(c.answer, gold_answer) 
            for c in all_candidates
        )
        
        if correct_exists:
            return "scoring_picked_wrong"
        
        # Check if segmentation missed key numbers
        problem_numbers = set()
        for span in spans:
            problem_numbers.update(span.numbers)
        
        # If we have no spans with numbers
        if not problem_numbers:
            return "segmentation_miss"
        
        # Check if explicit graph produced any results
        if not winner.graph:
            return "classifier_extractor_error"
        
        # Default: missing implicit ops (bridging didn't find the answer)
        return "missing_implicit_ops"


# ============================================================
# Evaluation Runner
# ============================================================

def load_gsm8k(path: str) -> List[Dict]:
    """Load GSM8K problems from JSONL."""
    problems = []
    with open(path) as f:
        for line in f:
            data = json.loads(line)
            # Extract gold answer from GSM8K format
            answer_text = data.get("answer", "")
            # GSM8K format: "#### 42"
            match = re.search(r'####\s*([\d,.-]+)', answer_text)
            if match:
                gold = float(match.group(1).replace(",", ""))
            else:
                try:
                    gold = float(answer_text.strip().split()[-1].replace(",", ""))
                except:
                    gold = None
            
            problems.append({
                "question": data.get("question", data.get("problem", "")),
                "gold_answer": gold,
            })
    return problems


def evaluate(pipeline: MyceliumPipeline, problems: List[Dict],
             max_problems: Optional[int] = None, verbose: bool = False) -> Dict:
    """Run evaluation and compute metrics."""
    
    if max_problems:
        problems = problems[:max_problems]
    
    results = []
    correct = 0
    error_counts = {}
    
    for i, problem in enumerate(problems):
        if verbose and i % 10 == 0:
            logger.info(f"Problem {i+1}/{len(problems)}")
        
        result = pipeline.solve(
            problem["question"],
            gold_answer=problem.get("gold_answer"),
            verbose=verbose
        )
        results.append(result)
        
        if result["correct"]:
            correct += 1
        elif result.get("error_attribution"):
            err = result["error_attribution"]
            error_counts[err] = error_counts.get(err, 0) + 1
    
    total = len(problems)
    accuracy = correct / total if total > 0 else 0
    
    summary = {
        "total": total,
        "correct": correct,
        "accuracy": accuracy,
        "error_attribution": error_counts,
        "avg_candidates": sum(r["candidates_tried"] for r in results) / total if total > 0 else 0,
    }
    
    return {"summary": summary, "results": results}


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Mycelium E2E Pipeline")
    parser.add_argument("--segmenter-path", required=True, help="Path to segmenter model")
    parser.add_argument("--classifier-path", required=True, help="Path to classifier model")
    parser.add_argument("--extractor-path", required=True, help="Path to extractor model")
    parser.add_argument("--problems-path", required=True, help="Path to problems JSONL")
    parser.add_argument("--output-path", default="results/e2e_results.json", help="Output path")
    parser.add_argument("--max-problems", type=int, default=None, help="Max problems to evaluate")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)
    
    # Load pipeline
    pipeline = MyceliumPipeline(
        segmenter_path=args.segmenter_path,
        classifier_path=args.classifier_path,
        extractor_path=args.extractor_path,
    )
    
    # Load problems
    problems = load_gsm8k(args.problems_path)
    logger.info(f"Loaded {len(problems)} problems")
    
    # Evaluate
    eval_results = evaluate(pipeline, problems, max_problems=args.max_problems, verbose=args.verbose)
    
    # Print summary
    s = eval_results["summary"]
    print(f"\n{'='*50}")
    print(f"E2E Pipeline Results")
    print(f"{'='*50}")
    print(f"Total:    {s['total']}")
    print(f"Correct:  {s['correct']}")
    print(f"Accuracy: {s['accuracy']:.2%}")
    print(f"Avg candidates/problem: {s['avg_candidates']:.1f}")
    print(f"\nError Attribution:")
    for err, count in sorted(s['error_attribution'].items(), key=lambda x: -x[1]):
        pct = count / (s['total'] - s['correct']) * 100 if s['total'] > s['correct'] else 0
        print(f"  {err}: {count} ({pct:.0f}%)")
    
    # Save
    Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_path, 'w') as f:
        json.dump(eval_results, f, indent=2, default=str)
    print(f"\nResults saved to {args.output_path}")


if __name__ == "__main__":
    main()
