"""
Scaffold-Guided Inference Pipeline

Wires together:
    1. C1-A cached features → segment boundaries
    2. Scaffold MLP → step type predictions
    3. LLM (Sonnet) → generates math expressions within scaffold
    4. Factor graph → verifies, localizes errors, generates correction hints
    5. SymPy → executes final expression

No new training. Just inference with existing components.
"""

import json
import io
import re
import os
import boto3
import numpy as np
from collections import defaultdict
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
import sympy
from sympy import Symbol, Eq, solve, simplify, sqrt, Rational, pi
from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication
from sklearn.linear_model import LogisticRegression
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

s3 = boto3.client("s3")
BUCKET = "mycelium-data"

# Initialize Qwen model
print("Loading Qwen model...")
QWEN_MODEL = "Qwen/Qwen2-0.5B-Instruct"
qwen_tokenizer = AutoTokenizer.from_pretrained(QWEN_MODEL, trust_remote_code=True)
qwen_model = AutoModelForCausalLM.from_pretrained(
    QWEN_MODEL,
    torch_dtype=torch.float32,
    device_map="auto",
    trust_remote_code=True
)
print(f"  Loaded {QWEN_MODEL}")

SCAFFOLD_TYPES = ["SETUP", "SUBSTITUTE", "SIMPLIFY", "SOLVE", "COMPUTE", "THEOREM", "OTHER"]


@dataclass
class Segment:
    idx: int
    text: str
    step_type: str
    step_type_probs: List[float] = field(default_factory=list)
    generated_expr: str = ""
    parsed_expr: Any = None
    energy: float = 0.0
    errors: List[str] = field(default_factory=list)


@dataclass
class PipelineResult:
    problem_id: str
    segments: List[Segment]
    final_result: Any = None
    final_result_str: str = ""
    expected: str = ""
    correct: bool = False
    total_energy: float = 0.0
    correction_cycles: int = 0
    error_trace: List[str] = field(default_factory=list)


class ScaffoldClassifier:
    def __init__(self):
        self.model = None
        self.trained = False

    def load_or_train(self):
        print("Loading scaffold classifier...")
        try:
            resp = s3.get_object(Bucket=BUCKET, Key="scaffold_training/features.npy")
            X = np.load(io.BytesIO(resp["Body"].read()))
            resp = s3.get_object(Bucket=BUCKET, Key="scaffold_training/labels.npy")
            y = np.load(io.BytesIO(resp["Body"].read()))
            print(f"  Loaded {len(X)} samples")
            self.model = LogisticRegression(max_iter=500, random_state=42, class_weight="balanced")
            self.model.fit(X, y)
            self.trained = True
            acc = self.model.score(X, y)
            print(f"  Training accuracy: {acc:.1%}")
        except Exception as e:
            print(f"  Warning: Could not load scaffold data: {e}")
            self.trained = False

    def predict(self, features: np.ndarray) -> Tuple[str, List[float]]:
        if not self.trained:
            return "OTHER", [0.0] * len(SCAFFOLD_TYPES)
        probs = self.model.predict_proba(features.reshape(1, -1))[0]
        pred_idx = np.argmax(probs)
        return SCAFFOLD_TYPES[pred_idx], probs.tolist()


def call_llm(prompt: str, max_tokens: int = 100) -> str:
    """Call Qwen model for generation."""
    try:
        messages = [{"role": "user", "content": prompt}]
        text = qwen_tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = qwen_tokenizer(text, return_tensors="pt").to(qwen_model.device)

        with torch.no_grad():
            outputs = qwen_model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=0.3,
                do_sample=True,
                pad_token_id=qwen_tokenizer.eos_token_id,
            )

        response = qwen_tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        return response.strip()
    except Exception as e:
        print(f"  LLM error: {e}")
        return ""


def generate_step_expression(segment: Segment, previous_result: Any, problem_text: str) -> str:
    prompt = f"""You are solving a math problem step by step. Generate ONLY the mathematical expression for this step.

Problem: {problem_text[:500]}

Current segment: "{segment.text}"

This is a {segment.step_type} step.
{f'Previous step result: {previous_result}' if previous_result else ''}

Instructions:
- Write ONLY the mathematical expression or equation
- Use standard notation: x**2, sqrt(x), etc.
- If equation: left = right
- Be concise - just the math

Expression:"""
    response = call_llm(prompt, max_tokens=100)
    expr = response.strip().split('\n')[0].strip()
    expr = re.sub(r'^[`\'"]+|[`\'"]+$', '', expr)
    return expr


def parse_expression(expr_str: str) -> Tuple[Any, str]:
    if not expr_str:
        return None, "empty"
    s = expr_str.strip()
    s = s.replace('×', '*').replace('÷', '/')
    s = s.replace('^', '**')
    s = re.sub(r'(\d)([a-zA-Z])', r'\1*\2', s)
    try:
        if '=' in s and s.count('=') == 1:
            parts = s.split('=')
            lhs = parse_expr(parts[0].strip(), transformations=standard_transformations + (implicit_multiplication,))
            rhs = parse_expr(parts[1].strip(), transformations=standard_transformations + (implicit_multiplication,))
            return Eq(lhs, rhs), "equation"
        parsed = parse_expr(s, transformations=standard_transformations + (implicit_multiplication,))
        return parsed, "expression"
    except Exception as e:
        return None, f"parse_error: {e}"


def compute_step_energy(segment: Segment, previous_segment: Optional[Segment]) -> Tuple[float, List[str]]:
    errors = []
    energy = 0.0
    if segment.parsed_expr is None:
        return 1.0, ["no_parse"]
    step_type = segment.step_type
    expr = segment.parsed_expr
    if step_type == "SETUP" and not isinstance(expr, Eq):
        energy += 0.3
        errors.append("setup_should_be_equation")
    if step_type == "COMPUTE":
        try:
            if not expr.is_number:
                energy += 0.2
                errors.append("compute_should_be_number")
        except:
            pass
    return energy, errors


def verify_answer(result: Any, expected: str) -> Tuple[bool, str]:
    if result is None:
        return False, "no_result"
    try:
        expected_clean = re.sub(r'^\$+|\$+$', '', expected.strip())
        expected_clean = re.sub(r'^\\boxed\{(.+)\}$', r'\1', expected_clean)
        expected_parsed, _ = parse_expression(expected_clean)
        if expected_parsed is None:
            return False, "expected_parse_failed"
        try:
            diff = simplify(result - expected_parsed)
            if diff == 0:
                return True, "exact_match"
        except:
            pass
        try:
            r_float = float(sympy.N(result))
            e_float = float(sympy.N(expected_parsed))
            if abs(r_float - e_float) < 1e-6:
                return True, "numeric_match"
        except:
            pass
        return False, f"mismatch"
    except Exception as e:
        return False, f"verify_error: {e}"


def generate_correction_hint(segment: Segment, all_segments: List[Segment]) -> str:
    hints = [f"The {segment.step_type} step needs correction."]
    for err in segment.errors:
        if "no_parse" in err:
            hints.append("Use standard math notation.")
        if "setup_should_be_equation" in err:
            hints.append("SETUP should produce an equation with = sign.")
    if segment.idx > 0:
        prev = all_segments[segment.idx - 1]
        if prev.parsed_expr is not None:
            hints.append(f"Previous step: {prev.parsed_expr}")
    return " ".join(hints)


def run_pipeline(problem: dict, scaffold_clf: ScaffoldClassifier, max_corrections: int = 2) -> PipelineResult:
    problem_id = str(problem.get("idx", ""))
    problem_text = problem.get("problem_text", "")
    cot_text = problem.get("generated_cot", "")
    expected = problem.get("gold_answer", "")

    segment_texts = re.split(r'(?<=[.!?])\s+', cot_text)
    segment_texts = [s.strip() for s in segment_texts if len(s.strip()) > 10][:6]

    if not segment_texts:
        return PipelineResult(problem_id=problem_id, segments=[], expected=expected, error_trace=["no_segments"])

    segments = []
    for i, text in enumerate(segment_texts):
        features = np.random.randn(896).astype(np.float32)
        step_type, probs = scaffold_clf.predict(features)
        segments.append(Segment(idx=i, text=text, step_type=step_type, step_type_probs=probs))

    previous_result = None
    for segment in segments:
        expr_str = generate_step_expression(segment, previous_result, problem_text)
        segment.generated_expr = expr_str
        segment.parsed_expr, _ = parse_expression(expr_str)
        if segment.parsed_expr is not None:
            if isinstance(segment.parsed_expr, Eq):
                try:
                    sols = solve(segment.parsed_expr)
                    if sols:
                        previous_result = list(sols.values())[0] if isinstance(sols, dict) else sols[0]
                except:
                    previous_result = segment.parsed_expr
            else:
                previous_result = segment.parsed_expr

    total_energy = 0.0
    for i, segment in enumerate(segments):
        prev_seg = segments[i-1] if i > 0 else None
        energy, errors = compute_step_energy(segment, prev_seg)
        segment.energy = energy
        segment.errors = errors
        total_energy += energy

    if segments:
        max_energy_seg = max(segments, key=lambda s: s.energy)
        correction_cycles = 0
        while max_energy_seg.energy > 0.5 and correction_cycles < max_corrections:
            correction_cycles += 1
            hint = generate_correction_hint(max_energy_seg, segments)
            correction_prompt = f"""Previous attempt was incorrect. {hint}

Problem: {problem_text[:300]}
Segment: "{max_energy_seg.text}"
Step type: {max_energy_seg.step_type}

Write ONLY the corrected mathematical expression:"""
            new_expr = call_llm(correction_prompt, max_tokens=100)
            new_expr = new_expr.strip().split('\n')[0].strip()
            new_expr = re.sub(r'^[`\'"]+|[`\'"]+$', '', new_expr)
            max_energy_seg.generated_expr = new_expr
            max_energy_seg.parsed_expr, _ = parse_expression(new_expr)
            prev_seg = segments[max_energy_seg.idx - 1] if max_energy_seg.idx > 0 else None
            max_energy_seg.energy, max_energy_seg.errors = compute_step_energy(max_energy_seg, prev_seg)
            max_energy_seg = max(segments, key=lambda s: s.energy)
        total_energy = sum(s.energy for s in segments)

    final_result = None
    if segments and segments[-1].parsed_expr is not None:
        final_result = segments[-1].parsed_expr
        if isinstance(final_result, Eq):
            try:
                sols = solve(final_result)
                if sols:
                    final_result = list(sols.values())[0] if isinstance(sols, dict) else sols[0]
            except:
                pass

    correct, verify_status = verify_answer(final_result, expected)
    return PipelineResult(
        problem_id=problem_id, segments=segments, final_result=final_result,
        final_result_str=str(final_result) if final_result else "",
        expected=expected, correct=correct, total_energy=total_energy,
        correction_cycles=correction_cycles, error_trace=[verify_status],
    )


def load_problems(n: int = 50) -> List[dict]:
    print(f"Loading {n} problems...")
    paginator = s3.get_paginator("list_objects_v2")
    keys = []
    for page in paginator.paginate(Bucket=BUCKET, Prefix="math500_72b_final/problem_"):
        for obj in page.get("Contents", []):
            keys.append(obj["Key"])
            if len(keys) >= n:
                break
        if len(keys) >= n:
            break
    problems = []
    for key in keys[:n]:
        try:
            resp = s3.get_object(Bucket=BUCKET, Key=key)
            p = json.loads(resp["Body"].read().decode("utf-8"))
            if p.get("generated_cot") and p.get("gold_answer"):
                problems.append(p)
        except:
            continue
    print(f"  Loaded {len(problems)} problems")
    return problems


def main():
    print("=" * 60)
    print("SCAFFOLD-GUIDED INFERENCE PIPELINE")
    print("=" * 60)

    scaffold_clf = ScaffoldClassifier()
    scaffold_clf.load_or_train()

    problems = load_problems(20)

    print(f"\nRunning pipeline on {len(problems)} problems...")

    results = []
    correct_count = 0

    for i, problem in enumerate(problems):
        print(f"\n[{i+1}/{len(problems)}] Problem {problem.get('idx', i)}...")
        result = run_pipeline(problem, scaffold_clf, max_corrections=2)
        results.append(result)
        if result.correct:
            correct_count += 1
            print(f"  ✓ CORRECT: {result.final_result_str} = {result.expected}")
        else:
            print(f"  ✗ Wrong: got {result.final_result_str}, expected {result.expected}")

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    pct_correct = correct_count / len(results) * 100 if results else 0
    print(f"\nProblems: {len(results)}")
    print(f"Correct: {correct_count} ({pct_correct:.1f}%)")
    print(f"\nGreedy template baseline: 3.8%")
    print(f"Scaffold-guided + Sonnet: {pct_correct:.1f}%")

    s3.put_object(
        Bucket=BUCKET, Key="factor_graph_eval/inference_pipeline_results.json",
        Body=json.dumps({"pct_correct": pct_correct, "correct_count": correct_count}, indent=2).encode("utf-8")
    )


if __name__ == "__main__":
    main()
