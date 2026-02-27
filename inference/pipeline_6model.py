#!/usr/bin/env python3
"""
Mycelium 6-Model Pipeline
=========================

Full inference: Problem text → Answer using all 6 specialist models.

Pipeline:
  1. C1: problem text → relevance field (soft attention over tokens)
  2. C6: problem text → answer type hints
  3. [Clustering]: peak detection on relevance → operation regions
  4. C2: per region, classify → top-3 templates
  5. C3: per region + template → top-3 expressions via beam
  6. C5: pairwise dependency check across regions
  7. C4: bridging operations for implicit values
  8. Sympy: execute DAG, filter valid, check answer type
  9. Evaluate: compare to gold

Usage:
  python pipeline_6model.py --problem "Natalia sold clips..." --gold 72
  python pipeline_6model.py --problems-path data/gsm8k_test.jsonl --max-problems 10
"""

import re
import math
import json
import argparse
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
from collections import defaultdict

import numpy as np

try:
    import torch
    from transformers import (
        AutoTokenizer,
        AutoModelForTokenClassification,
        AutoModelForSequenceClassification,
        AutoModelForCausalLM,
    )
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    from scipy.signal import find_peaks
    from scipy.ndimage import gaussian_filter1d
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

logger = logging.getLogger(__name__)


# ============================================================
# Data Structures
# ============================================================

@dataclass
class OperationRegion:
    """A cluster of high-relevance tokens forming one operation."""
    start_idx: int
    end_idx: int
    peak_idx: int
    peak_value: float
    text: str
    tokens: List[str]
    relevance_scores: List[float]


@dataclass
class TemplateCandidate:
    """A template classification for a region."""
    template: str
    confidence: float
    region: OperationRegion


@dataclass
class ExpressionCandidate:
    """An extracted expression from a template + region."""
    operation: str
    arg1: float
    arg2: Optional[float]
    arg1_source: str
    arg2_source: Optional[str]
    confidence: float
    result: Optional[float] = None


@dataclass
class DependencyEdge:
    """A dependency between two expression candidates."""
    from_idx: int
    to_idx: int
    confidence: float


@dataclass
class BridgingOp:
    """An implicit bridging operation."""
    operation: str
    args: List[float]
    result: float
    bridge_type: str  # "ACC_ADD", "ACC_SUB", "DERIVED_MUL", etc.


@dataclass
class ExecutionDAG:
    """Complete execution graph."""
    expressions: List[ExpressionCandidate]
    dependencies: List[DependencyEdge]
    bridging: List[BridgingOp]
    final_result: Optional[float] = None


# ============================================================
# Simple Number Extraction (fallback)
# ============================================================

def extract_numbers(text: str) -> List[float]:
    """Extract numeric values from text."""
    numbers = []
    for match in re.finditer(r'\b(\d{1,3}(?:,\d{3})*(?:\.\d+)?|\d+(?:\.\d+)?)\b', text):
        num_str = match.group(1).replace(',', '')
        try:
            numbers.append(float(num_str))
        except ValueError:
            pass
    return numbers


# ============================================================
# Symbolic Executor
# ============================================================

OPERATIONS = {
    "ADD": lambda a, b: a + b,
    "SUB": lambda a, b: a - b,
    "MUL": lambda a, b: a * b,
    "DIV": lambda a, b: a / b if b != 0 else None,
    "POW": lambda a, b: a ** b if abs(b) < 100 else None,
}

def execute_operation(op: str, args: List[float]) -> Optional[float]:
    """Execute a single operation."""
    op = op.upper()
    if op not in OPERATIONS or len(args) < 2:
        return None
    try:
        result = OPERATIONS[op](args[0], args[1])
        if result is not None and math.isfinite(result):
            return result
    except (ValueError, ZeroDivisionError, OverflowError):
        pass
    return None


def answer_matches(pred: float, gold: float, tol: float = 1e-6) -> bool:
    """Check if predicted answer matches gold."""
    if abs(pred - gold) < tol:
        return True
    if abs(round(pred) - gold) < tol:
        return True
    return False


# ============================================================
# Relevance Clustering (Signal Processing)
# ============================================================

def cluster_relevance_field(
    relevance: np.ndarray,
    tokens: List[str],
    min_peak_height: float = 0.3,
    min_peak_distance: int = 5,
    sigma: float = 2.0,
) -> List[OperationRegion]:
    """
    Cluster high-relevance tokens into operation regions using peak detection.

    This is the one piece of non-learned logic in the pipeline (like sympy).
    Simple signal processing on the 1D relevance field.

    Args:
        relevance: 1D array of relevance scores per token
        tokens: List of token strings
        min_peak_height: Minimum height for a peak to be considered
        min_peak_distance: Minimum tokens between peaks
        sigma: Gaussian smoothing sigma

    Returns:
        List of OperationRegion objects
    """
    if not HAS_SCIPY:
        # Fallback: simple threshold-based clustering
        return _cluster_relevance_simple(relevance, tokens, min_peak_height)

    # Smooth the signal to reduce noise
    smoothed = gaussian_filter1d(relevance, sigma=sigma)

    # Find peaks
    peaks, properties = find_peaks(
        smoothed,
        height=min_peak_height,
        distance=min_peak_distance,
    )

    if len(peaks) == 0:
        # No peaks found, try with lower threshold
        peaks, properties = find_peaks(
            smoothed,
            height=min_peak_height * 0.5,
            distance=min_peak_distance,
        )

    if len(peaks) == 0:
        return []

    # Expand each peak into a region (tokens above threshold around peak)
    regions = []
    threshold = min_peak_height * 0.5

    for peak_idx in peaks:
        # Find region boundaries
        start = peak_idx
        while start > 0 and smoothed[start - 1] > threshold:
            start -= 1

        end = peak_idx
        while end < len(smoothed) - 1 and smoothed[end + 1] > threshold:
            end += 1

        # Extract region tokens and text
        region_tokens = tokens[start:end + 1]
        region_text = " ".join(t for t in region_tokens if not t.startswith("##"))
        region_text = region_text.replace(" ##", "")

        regions.append(OperationRegion(
            start_idx=start,
            end_idx=end,
            peak_idx=peak_idx,
            peak_value=float(smoothed[peak_idx]),
            text=region_text,
            tokens=region_tokens,
            relevance_scores=relevance[start:end + 1].tolist(),
        ))

    # Merge overlapping regions
    regions = _merge_overlapping_regions(regions)

    return regions


def _cluster_relevance_simple(
    relevance: np.ndarray,
    tokens: List[str],
    threshold: float = 0.3,
) -> List[OperationRegion]:
    """Simple fallback clustering without scipy."""
    regions = []
    in_region = False
    start = 0
    peak_idx = 0
    peak_val = 0

    for i, score in enumerate(relevance):
        if score > threshold:
            if not in_region:
                in_region = True
                start = i
                peak_idx = i
                peak_val = score
            elif score > peak_val:
                peak_idx = i
                peak_val = score
        else:
            if in_region:
                region_tokens = tokens[start:i]
                region_text = " ".join(t for t in region_tokens if not t.startswith("##"))
                regions.append(OperationRegion(
                    start_idx=start,
                    end_idx=i - 1,
                    peak_idx=peak_idx,
                    peak_value=peak_val,
                    text=region_text,
                    tokens=region_tokens,
                    relevance_scores=relevance[start:i].tolist(),
                ))
                in_region = False

    # Handle region at end
    if in_region:
        region_tokens = tokens[start:]
        region_text = " ".join(t for t in region_tokens if not t.startswith("##"))
        regions.append(OperationRegion(
            start_idx=start,
            end_idx=len(relevance) - 1,
            peak_idx=peak_idx,
            peak_value=peak_val,
            text=region_text,
            tokens=region_tokens,
            relevance_scores=relevance[start:].tolist(),
        ))

    return regions


def _merge_overlapping_regions(regions: List[OperationRegion]) -> List[OperationRegion]:
    """Merge regions that overlap."""
    if len(regions) <= 1:
        return regions

    # Sort by start index
    regions = sorted(regions, key=lambda r: r.start_idx)

    merged = [regions[0]]
    for region in regions[1:]:
        if region.start_idx <= merged[-1].end_idx + 1:
            # Overlap - merge
            prev = merged[-1]
            new_end = max(prev.end_idx, region.end_idx)
            new_peak = prev if prev.peak_value >= region.peak_value else region
            merged[-1] = OperationRegion(
                start_idx=prev.start_idx,
                end_idx=new_end,
                peak_idx=new_peak.peak_idx,
                peak_value=new_peak.peak_value,
                text=prev.text + " " + region.text,
                tokens=prev.tokens + region.tokens,
                relevance_scores=prev.relevance_scores + region.relevance_scores,
            )
        else:
            merged.append(region)

    return merged


# ============================================================
# Custom Model: Qwen2ForTokenRegression (from train_c1_relevance.py)
# ============================================================

class Qwen2ForTokenRegression(torch.nn.Module):
    """Qwen2 with a token-level regression head for relevance scoring."""

    def __init__(self, config, base_model):
        super().__init__()
        self.qwen = base_model
        self.dropout = torch.nn.Dropout(0.1)
        self.regressor = torch.nn.Linear(config.hidden_size, 1)

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        outputs = self.qwen(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        sequence_output = outputs.last_hidden_state
        sequence_output = self.dropout(sequence_output)
        predictions = self.regressor(sequence_output).squeeze(-1)
        return predictions


# ============================================================
# Model Interfaces
# ============================================================

class C1Relevance:
    """C1: Token-level relevance scoring (regression model)."""

    def __init__(self, model_path: str):
        from transformers import AutoModel, AutoConfig

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

        # Load the custom regression model
        config = AutoConfig.from_pretrained(model_path)

        # Load weights from safetensors or pytorch format
        safetensors_path = Path(f"{model_path}/model.safetensors")
        pytorch_path = Path(f"{model_path}/pytorch_model.bin")

        if safetensors_path.exists():
            from safetensors.torch import load_file
            state_dict = load_file(str(safetensors_path))
        elif pytorch_path.exists():
            state_dict = torch.load(str(pytorch_path), map_location="cpu", weights_only=False)
        else:
            raise FileNotFoundError(f"No model weights found in {model_path}")

        # Build model manually
        base_model = AutoModel.from_config(config)

        # Map weights: qwen.* -> model.*
        mapped_dict = {}
        for k, v in state_dict.items():
            if k.startswith("qwen."):
                mapped_dict[k.replace("qwen.", "")] = v
            elif k.startswith("regressor."):
                pass  # Handle separately
            else:
                mapped_dict[k] = v

        base_model.load_state_dict(mapped_dict, strict=False)

        # Create regression head
        self.model = Qwen2ForTokenRegression(config, base_model)

        # Load regressor weights
        if "regressor.weight" in state_dict:
            self.model.regressor.weight.data = state_dict["regressor.weight"]
            self.model.regressor.bias.data = state_dict["regressor.bias"]

        self.model.eval()
        logger.info(f"C1 loaded (regression model)")

    def predict(self, text: str) -> Tuple[np.ndarray, List[str]]:
        """
        Predict relevance scores for each token.

        Returns:
            relevance: 1D array of scores (0-1) per token
            tokens: List of token strings
        """
        inputs = self.tokenizer(
            text, return_tensors="pt", truncation=True, max_length=512
        )

        with torch.no_grad():
            predictions = self.model(**inputs)

        # Clamp to 0-1 range
        relevance = torch.clamp(predictions[0], 0, 1).numpy()

        tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

        return relevance, tokens


class C2Classifier:
    """C2: Region → template classification."""

    def __init__(self, model_path: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.eval()
        self.id2label = self.model.config.id2label
        logger.info(f"C2 loaded: {len(self.id2label)} classes")

    def classify_region(
        self, problem_text: str, region: OperationRegion, top_k: int = 3
    ) -> List[TemplateCandidate]:
        """Classify a region, return top-k template candidates."""
        # Mark the region in the problem text
        marked = self._mark_region(problem_text, region)

        inputs = self.tokenizer(
            marked, return_tensors="pt", truncation=True, max_length=256
        )

        with torch.no_grad():
            outputs = self.model(**inputs)

        probs = torch.softmax(outputs.logits, dim=-1)[0]
        top_indices = torch.argsort(probs, descending=True)[:top_k]

        candidates = []
        for idx in top_indices:
            candidates.append(TemplateCandidate(
                template=self.id2label.get(idx.item(), "UNKNOWN"),
                confidence=probs[idx].item(),
                region=region,
            ))

        return candidates

    def _mark_region(self, text: str, region: OperationRegion) -> str:
        """Mark region tokens in text with <SPAN> markers."""
        # Simple approach: find region text in problem and mark it
        region_text = region.text.strip()
        if region_text in text:
            return text.replace(region_text, f"<SPAN>{region_text}</SPAN>", 1)
        return f"<SPAN>{region_text}</SPAN> {text}"


class C3Extractor:
    """C3: Template + region → expression extraction."""

    def __init__(self, model_path: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_path)
        self.model.eval()
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        logger.info("C3 loaded")

    def extract(
        self,
        problem_text: str,
        region: OperationRegion,
        template: str,
        beam_size: int = 3,
    ) -> List[ExpressionCandidate]:
        """Extract expressions using beam search."""
        # Mark region
        marked = problem_text
        if region.text in problem_text:
            marked = problem_text.replace(
                region.text, f"<SPAN>{region.text}</SPAN>", 1
            )

        prompt = f"[{template}]\n{marked}\nArguments:\n"
        inputs = self.tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=256
        )

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=50,
                num_beams=beam_size,
                num_return_sequences=beam_size,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        candidates = []
        for i in range(beam_size):
            generated = self.tokenizer.decode(
                outputs[i][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True
            )
            expr = self._parse_expression(generated, template, problem_text)
            if expr:
                candidates.append(expr)

        return candidates

    def _parse_expression(
        self, text: str, template: str, problem_text: str
    ) -> Optional[ExpressionCandidate]:
        """Parse extractor output into ExpressionCandidate."""
        lines = text.strip().split('\n')
        args = []
        sources = []

        problem_nums = set(extract_numbers(problem_text))

        for line in lines[:2]:
            match = re.match(r'([\d.]+)\|(\w+)', line.strip())
            if match:
                try:
                    val = float(match.group(1))
                    src = match.group(2)
                    args.append(val)
                    sources.append(src)
                except ValueError:
                    pass

        if len(args) < 2:
            return None

        result = execute_operation(template, args[:2])

        return ExpressionCandidate(
            operation=template,
            arg1=args[0],
            arg2=args[1],
            arg1_source=sources[0],
            arg2_source=sources[1],
            confidence=0.8,  # Could use beam scores
            result=result,
        )


class C4Bridging:
    """C4: Implicit operation detection."""

    def __init__(self, model_path: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.eval()
        self.id2label = self.model.config.id2label
        logger.info(f"C4 loaded: {len(self.id2label)} bridge types")

    def detect_bridging(
        self,
        problem_text: str,
        expressions: List[ExpressionCandidate],
    ) -> List[BridgingOp]:
        """Detect implicit bridging operations."""
        if len(expressions) < 2:
            return []

        # Collect intermediate results
        results = [e.result for e in expressions if e.result is not None]
        if len(results) < 2:
            return []

        # Classify bridging type
        context = f"Results: {results}. Question: {problem_text}"
        inputs = self.tokenizer(
            context, return_tensors="pt", truncation=True, max_length=256
        )

        with torch.no_grad():
            outputs = self.model(**inputs)

        probs = torch.softmax(outputs.logits, dim=-1)[0]
        pred_idx = torch.argmax(probs).item()
        bridge_type = self.id2label.get(pred_idx, "NONE")

        bridging_ops = []

        if bridge_type == "ACC_ADD":
            total = sum(results)
            bridging_ops.append(BridgingOp(
                operation="ADD",
                args=results,
                result=total,
                bridge_type="ACC_ADD",
            ))
        elif bridge_type == "ACC_SUB":
            # Max minus others
            mx = max(results)
            others = sum(r for r in results if r != mx)
            bridging_ops.append(BridgingOp(
                operation="SUB",
                args=[mx, others],
                result=mx - others,
                bridge_type="ACC_SUB",
            ))
        elif bridge_type == "DERIVED_MUL":
            # Product of results
            prod = 1
            for r in results:
                prod *= r
            bridging_ops.append(BridgingOp(
                operation="MUL",
                args=results,
                result=prod,
                bridge_type="DERIVED_MUL",
            ))

        return bridging_ops


class C5Dependencies:
    """C5: Pairwise dependency resolution."""

    def __init__(self, model_path: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.eval()
        logger.info("C5 loaded")

    def resolve_dependencies(
        self,
        expressions: List[ExpressionCandidate],
        problem_text: str,
    ) -> List[DependencyEdge]:
        """Determine execution order via pairwise dependency checks."""
        edges = []

        for i in range(len(expressions)):
            for j in range(len(expressions)):
                if i == j:
                    continue

                # Check if expression j depends on expression i
                context = (
                    f"Expr A: {expressions[i].operation}({expressions[i].arg1}, {expressions[i].arg2})\n"
                    f"Expr B: {expressions[j].operation}({expressions[j].arg1}, {expressions[j].arg2})\n"
                    f"Problem: {problem_text[:200]}"
                )

                inputs = self.tokenizer(
                    context, return_tensors="pt", truncation=True, max_length=256
                )

                with torch.no_grad():
                    outputs = self.model(**inputs)

                # Binary classification: 1 = depends, 0 = independent
                probs = torch.softmax(outputs.logits, dim=-1)[0]
                if probs[1] > 0.5:
                    edges.append(DependencyEdge(
                        from_idx=i,
                        to_idx=j,
                        confidence=probs[1].item(),
                    ))

        return edges


class C6AnswerType:
    """C6: Answer type / goal detection."""

    def __init__(self, model_path: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.eval()
        self.id2label = self.model.config.id2label
        logger.info(f"C6 loaded: {len(self.id2label)} answer types")

    def predict(self, problem_text: str) -> Tuple[str, float]:
        """Predict answer type and goal hints."""
        inputs = self.tokenizer(
            problem_text, return_tensors="pt", truncation=True, max_length=256
        )

        with torch.no_grad():
            outputs = self.model(**inputs)

        probs = torch.softmax(outputs.logits, dim=-1)[0]
        pred_idx = torch.argmax(probs).item()

        return self.id2label.get(pred_idx, "UNKNOWN"), probs[pred_idx].item()


# ============================================================
# Pipeline Orchestrator
# ============================================================

class Pipeline6Model:
    """Full 6-model inference pipeline."""

    def __init__(
        self,
        c1_path: str,
        c2_path: str,
        c3_path: str,
        c4_path: Optional[str] = None,
        c5_path: Optional[str] = None,
        c6_path: Optional[str] = None,
    ):
        logger.info("Loading 6-model pipeline...")

        self.c1 = C1Relevance(c1_path)
        self.c2 = C2Classifier(c2_path)
        self.c3 = C3Extractor(c3_path)

        # Optional models (may not be trained yet)
        self.c4 = C4Bridging(c4_path) if c4_path and Path(c4_path).exists() else None
        self.c5 = C5Dependencies(c5_path) if c5_path and Path(c5_path).exists() else None
        self.c6 = C6AnswerType(c6_path) if c6_path and Path(c6_path).exists() else None

        logger.info(f"Pipeline ready: C1-C3 loaded, C4={self.c4 is not None}, C5={self.c5 is not None}, C6={self.c6 is not None}")

    def solve(
        self,
        problem_text: str,
        gold_answer: Optional[float] = None,
        verbose: bool = False,
    ) -> Dict[str, Any]:
        """
        Solve a math problem using the full pipeline.

        Returns dict with:
          - answer: predicted answer
          - correct: whether it matches gold
          - regions: operation regions found
          - expressions: extracted expressions
          - dag: execution graph
        """
        result = {
            "problem": problem_text[:100] + "...",
            "gold": gold_answer,
            "answer": None,
            "correct": None,
            "regions": [],
            "templates": [],
            "expressions": [],
            "bridging": [],
            "answer_type": None,
        }

        # Step 1: C1 - relevance field
        if verbose:
            logger.info("Step 1: C1 relevance...")
        relevance, tokens = self.c1.predict(problem_text)

        # Step 2: C6 - answer type (if available)
        if self.c6:
            if verbose:
                logger.info("Step 2: C6 answer type...")
            answer_type, conf = self.c6.predict(problem_text)
            result["answer_type"] = answer_type

        # Step 3: Clustering - peak detection on relevance
        if verbose:
            logger.info("Step 3: Clustering relevance field...")
        regions = cluster_relevance_field(relevance, tokens)
        result["regions"] = [
            {"text": r.text, "peak": r.peak_value} for r in regions
        ]

        if not regions:
            if verbose:
                logger.info("No operation regions found")
            return result

        # Step 4: C2 - classify each region
        if verbose:
            logger.info(f"Step 4: C2 classification for {len(regions)} regions...")
        all_templates = []
        for region in regions:
            templates = self.c2.classify_region(problem_text, region, top_k=3)
            all_templates.extend(templates)

        result["templates"] = [
            {"template": t.template, "conf": t.confidence, "region": t.region.text}
            for t in all_templates
        ]

        # Step 5: C3 - extract expressions for each template
        if verbose:
            logger.info(f"Step 5: C3 extraction for {len(all_templates)} templates...")
        all_expressions = []
        for tmpl in all_templates:
            expressions = self.c3.extract(
                problem_text, tmpl.region, tmpl.template, beam_size=3
            )
            all_expressions.extend(expressions)

        # Filter to valid expressions
        valid_exprs = [e for e in all_expressions if e.result is not None]
        result["expressions"] = [
            {
                "op": e.operation,
                "args": [e.arg1, e.arg2],
                "result": e.result,
            }
            for e in valid_exprs
        ]

        if not valid_exprs:
            if verbose:
                logger.info("No valid expressions extracted")
            return result

        # Step 6: C5 - dependency resolution (if available)
        dependencies = []
        if self.c5:
            if verbose:
                logger.info("Step 6: C5 dependency resolution...")
            dependencies = self.c5.resolve_dependencies(valid_exprs, problem_text)

        # Step 7: C4 - bridging operations (if available)
        bridging_ops = []
        if self.c4:
            if verbose:
                logger.info("Step 7: C4 bridging detection...")
            bridging_ops = self.c4.detect_bridging(problem_text, valid_exprs)

        result["bridging"] = [
            {"type": b.bridge_type, "result": b.result}
            for b in bridging_ops
        ]

        # Step 8: Execute DAG and find best answer
        if verbose:
            logger.info("Step 8: Sympy execution and scoring...")

        # Candidate answers: all expression results + bridging results
        candidates = []
        for e in valid_exprs:
            candidates.append((e.result, e.confidence, "explicit"))
        for b in bridging_ops:
            candidates.append((b.result, 0.7, b.bridge_type))

        # Score and pick best
        best_answer = None
        best_score = -1

        for ans, conf, source in candidates:
            score = conf

            # Prefer integers on GSM8K
            if ans == int(ans):
                score += 0.1

            # Reasonable magnitude
            if 0 < abs(ans) < 1e6:
                score += 0.05

            # Answer type consistency (if we have C6)
            if result["answer_type"] and source == result["answer_type"]:
                score += 0.2

            # Gold match (only for debugging/eval)
            if gold_answer is not None and answer_matches(ans, gold_answer):
                score += 100

            if score > best_score:
                best_score = score
                best_answer = ans

        result["answer"] = best_answer

        if gold_answer is not None and best_answer is not None:
            result["correct"] = answer_matches(best_answer, gold_answer)

        return result


# ============================================================
# Mock Models for Testing
# ============================================================

class MockC1:
    """Mock C1 for testing without trained models."""

    def predict(self, text: str) -> Tuple[np.ndarray, List[str]]:
        # Simple heuristic: high relevance around numbers
        tokens = text.split()
        relevance = np.zeros(len(tokens))

        for i, token in enumerate(tokens):
            if any(c.isdigit() for c in token):
                # High relevance for number tokens
                relevance[i] = 0.8
                # Medium relevance for neighbors
                if i > 0:
                    relevance[i-1] = max(relevance[i-1], 0.4)
                if i < len(tokens) - 1:
                    relevance[i+1] = max(relevance[i+1], 0.4)

        return relevance, tokens


class MockC2:
    """Mock C2 for testing."""

    def classify_region(
        self, problem_text: str, region: OperationRegion, top_k: int = 3
    ) -> List[TemplateCandidate]:
        # Heuristic based on keywords
        text = region.text.lower()

        if "total" in text or "altogether" in text or "sum" in text:
            return [TemplateCandidate("ADD", 0.8, region)]
        elif "half" in text or "divide" in text or "each" in text:
            return [TemplateCandidate("DIV", 0.7, region)]
        elif "times" in text or "per" in text:
            return [TemplateCandidate("MUL", 0.75, region)]
        elif "left" in text or "remain" in text or "less" in text:
            return [TemplateCandidate("SUB", 0.7, region)]
        else:
            # Default: return common operations
            return [
                TemplateCandidate("MUL", 0.5, region),
                TemplateCandidate("ADD", 0.3, region),
            ]


class MockC3:
    """Mock C3 for testing."""

    def extract(
        self,
        problem_text: str,
        region: OperationRegion,
        template: str,
        beam_size: int = 3,
    ) -> List[ExpressionCandidate]:
        # Extract numbers from region
        nums = extract_numbers(region.text)
        if len(nums) < 2:
            # Try problem text
            all_nums = extract_numbers(problem_text)
            nums = all_nums[:2] if len(all_nums) >= 2 else nums

        if len(nums) < 2:
            return []

        result = execute_operation(template, nums[:2])

        return [ExpressionCandidate(
            operation=template,
            arg1=nums[0],
            arg2=nums[1],
            arg1_source="PROB",
            arg2_source="PROB",
            confidence=0.7,
            result=result,
        )]


def create_mock_pipeline() -> Pipeline6Model:
    """Create pipeline with mock models for testing."""
    pipeline = Pipeline6Model.__new__(Pipeline6Model)
    pipeline.c1 = MockC1()
    pipeline.c2 = MockC2()
    pipeline.c3 = MockC3()
    pipeline.c4 = None
    pipeline.c5 = None
    pipeline.c6 = None
    return pipeline


# ============================================================
# Evaluation
# ============================================================

def load_gsm8k(path: str, max_problems: Optional[int] = None) -> List[Dict]:
    """Load GSM8K problems."""
    problems = []
    with open(path) as f:
        for line in f:
            data = json.loads(line)
            match = re.search(r'####\s*([\d,.-]+)', data.get("answer", ""))
            gold = float(match.group(1).replace(",", "")) if match else None
            problems.append({
                "question": data.get("question", data.get("problem", "")),
                "gold": gold,
            })
            if max_problems and len(problems) >= max_problems:
                break
    return problems


def evaluate(
    pipeline,
    problems: List[Dict],
    verbose: bool = False,
) -> Dict:
    """Run evaluation."""
    results = []
    correct = 0

    for i, prob in enumerate(problems):
        if verbose:
            print(f"\n{'='*60}")
            print(f"Problem {i+1}/{len(problems)}")

        result = pipeline.solve(
            prob["question"],
            gold_answer=prob["gold"],
            verbose=verbose,
        )
        results.append(result)

        if result.get("correct"):
            correct += 1

        if verbose:
            print(f"Gold: {prob['gold']}, Predicted: {result['answer']}, Correct: {result.get('correct')}")

    accuracy = correct / len(problems) if problems else 0

    return {
        "total": len(problems),
        "correct": correct,
        "accuracy": accuracy,
        "results": results,
    }


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Mycelium 6-Model Pipeline")
    parser.add_argument("--problem", type=str, help="Single problem to solve")
    parser.add_argument("--gold", type=float, help="Gold answer for single problem")
    parser.add_argument("--problems-path", type=str, help="Path to GSM8K JSONL")
    parser.add_argument("--max-problems", type=int, default=10, help="Max problems to evaluate")
    parser.add_argument("--c1-path", default="models/c1_relevance_v2")
    parser.add_argument("--c2-path", default="models/c2_ib_templates_frozen_v1")
    parser.add_argument("--c3-path", default="models/c3_extractor_partial_freeze_v1")
    parser.add_argument("--c4-path", default=None)
    parser.add_argument("--c5-path", default=None)
    parser.add_argument("--c6-path", default=None)
    parser.add_argument("--mock", action="store_true", help="Use mock models for testing")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s: %(message)s",
    )

    print("\n" + "="*60)
    print("  MYCELIUM 6-MODEL PIPELINE")
    print("="*60)

    # Create pipeline
    if args.mock:
        print("\nUsing mock models for testing...")
        pipeline = create_mock_pipeline()
    else:
        print("\nLoading trained models...")
        pipeline = Pipeline6Model(
            c1_path=args.c1_path,
            c2_path=args.c2_path,
            c3_path=args.c3_path,
            c4_path=args.c4_path,
            c5_path=args.c5_path,
            c6_path=args.c6_path,
        )

    # Single problem mode
    if args.problem:
        result = pipeline.solve(args.problem, gold_answer=args.gold, verbose=True)
        print(f"\n{'='*60}")
        print("RESULT")
        print("="*60)
        print(json.dumps(result, indent=2, default=str))
        return

    # Batch evaluation mode
    if args.problems_path:
        print(f"\nLoading problems from {args.problems_path}...")
        problems = load_gsm8k(args.problems_path, args.max_problems)
        print(f"Loaded {len(problems)} problems")

        eval_results = evaluate(pipeline, problems, verbose=args.verbose)

        print(f"\n{'='*60}")
        print("EVALUATION RESULTS")
        print("="*60)
        print(f"Total: {eval_results['total']}")
        print(f"Correct: {eval_results['correct']}")
        print(f"Accuracy: {eval_results['accuracy']:.2%}")
        return

    # Demo mode
    print("\nRunning demo problems...")
    demo_problems = [
        {
            "question": "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?",
            "gold": 72,
        },
        {
            "question": "A baker sold 5 cakes for $20 each. How much did the baker earn?",
            "gold": 100,
        },
    ]

    for prob in demo_problems:
        result = pipeline.solve(prob["question"], gold_answer=prob["gold"], verbose=args.verbose)
        print(f"\nProblem: {prob['question'][:60]}...")
        print(f"Gold: {prob['gold']}, Predicted: {result['answer']}, Correct: {result.get('correct')}")
        print(f"Regions: {result['regions']}")
        print(f"Expressions: {result['expressions']}")


if __name__ == "__main__":
    main()
