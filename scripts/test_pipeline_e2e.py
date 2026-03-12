#!/usr/bin/env python3
"""
End-to-End Pipeline Test for Mycelium v7

Tests the full pipeline on 10 MATH problems:
1. Stage 1: C1-A → boundaries + scaffold types
2. Stage 2: slot filler → rough arguments (SETUP/SUBSTITUTE/THEOREM)
            templates → deterministic (EXPAND/SIMPLIFY/SOLVE/COMPUTE/ANSWER)
3. Stage 3: rough_to_sympy → SymPy execute → verify
4. Compare final answer to gold via parse_latex
5. Print: correct/total, per-step success, per-scaffold-type breakdown

Usage:
    # Quick test with mock C1-A (no C1-A model needed)
    python scripts/test_pipeline_e2e.py --mock-c1a --slot-filler-model /path/to/model

    # Full test with C1-A model
    python scripts/test_pipeline_e2e.py --slot-filler-model /path/to/model
"""

import os
import sys
import json
import argparse
import random
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from collections import defaultdict

# Add src and scripts to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.dirname(__file__))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from oracle import SympyOracle, OracleResult
from templates import execute_template

# Import C1-A model from stage1 script
try:
    from stage1_c1a_inference import C1AModel, C1AConfig
    C1A_AVAILABLE = True
except ImportError:
    C1A_AVAILABLE = False


# ============================================================================
# Configuration
# ============================================================================

SCAFFOLD_TYPES = ["SETUP", "SUBSTITUTE", "SIMPLIFY", "SOLVE", "COMPUTE", "THEOREM", "ANSWER", "EXPAND"]
SLOT_FILLER_SCAFFOLDS = {"SETUP", "SUBSTITUTE", "THEOREM"}
TEMPLATE_SCAFFOLDS = {"EXPAND", "SIMPLIFY", "SOLVE", "COMPUTE", "ANSWER"}


@dataclass
class TestConfig:
    n_problems: int = 10
    seed: int = 42
    max_input_length: int = 256
    max_output_length: int = 32
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================================
# MATH Problem Loading
# ============================================================================

def load_math_problems(n_problems: int, seed: int = 42) -> List[Dict]:
    """
    Load n MATH problems for testing.

    Returns list of dicts with:
        - problem_id: str
        - problem_text: str
        - expected_answer: str
    """
    # Try to load from S3 or use sample problems
    sample_problems = [
        {
            "problem_id": "test_001",
            "problem_text": "Compute $2 + 3 \\times 4$.",
            "expected_answer": "14"
        },
        {
            "problem_id": "test_002",
            "problem_text": "Solve for $x$: $x^2 - 9 = 0$.",
            "expected_answer": "3"
        },
        {
            "problem_id": "test_003",
            "problem_text": "Simplify $\\frac{x^2 - 1}{x - 1}$ for $x \\neq 1$.",
            "expected_answer": "x + 1"
        },
        {
            "problem_id": "test_004",
            "problem_text": "Expand $(x + 2)^2$.",
            "expected_answer": "x^2 + 4x + 4"
        },
        {
            "problem_id": "test_005",
            "problem_text": "What is $\\sqrt{144}$?",
            "expected_answer": "12"
        },
        {
            "problem_id": "test_006",
            "problem_text": "If $f(x) = 2x + 1$, what is $f(3)$?",
            "expected_answer": "7"
        },
        {
            "problem_id": "test_007",
            "problem_text": "Compute $\\frac{1}{2} + \\frac{1}{3}$.",
            "expected_answer": "5/6"
        },
        {
            "problem_id": "test_008",
            "problem_text": "Solve $2x + 4 = 10$.",
            "expected_answer": "3"
        },
        {
            "problem_id": "test_009",
            "problem_text": "What is $3^4$?",
            "expected_answer": "81"
        },
        {
            "problem_id": "test_010",
            "problem_text": "Simplify $\\frac{6x^2}{2x}$.",
            "expected_answer": "3x"
        },
    ]

    random.seed(seed)
    random.shuffle(sample_problems)

    return sample_problems[:n_problems]


def load_real_math_problems(n_problems: int = 10, seed: int = 42) -> List[Dict]:
    """
    Load real MATH500 problems from S3 with gold answers.
    """
    import subprocess
    import json
    import random

    random.seed(seed)

    # List problem files and sample
    result = subprocess.run(
        'aws s3 ls s3://mycelium-data/math500_7b/ | grep problem_ | head -100',
        shell=True, capture_output=True, text=True
    )

    problem_files = []
    for line in result.stdout.strip().split('\n'):
        if 'problem_' in line:
            parts = line.split()
            if parts:
                problem_files.append(parts[-1])

    # Sample n problems
    sampled = random.sample(problem_files, min(n_problems, len(problem_files)))

    problems = []
    for fname in sampled:
        result = subprocess.run(
            f'aws s3 cp s3://mycelium-data/math500_7b/{fname} -',
            shell=True, capture_output=True, text=True
        )
        if result.returncode != 0:
            continue

        data = json.loads(result.stdout)
        problems.append({
            'problem_id': str(data.get('idx', len(problems))),
            'problem_text': data.get('question', ''),
            'expected_answer': data.get('gold_answer', ''),
        })

    return problems


# ============================================================================
# Stage 1: C1-A (or Mock)
# ============================================================================

def mock_c1a_inference(problem: Dict) -> Dict:
    """
    Mock C1-A inference for testing without the C1-A model.

    Uses a generic scaffold sequence - no keyword heuristics.
    In production, the C1-A model learns to predict scaffold types from
    hidden state representations, not from keyword matching.
    """
    # Generic 3-step scaffold sequence for testing
    # SETUP -> COMPUTE -> ANSWER is a reasonable default
    # The real C1-A model will learn the appropriate scaffolds
    scaffolds = ["SETUP", "COMPUTE", "ANSWER"]

    return {
        "problem_id": problem['problem_id'],
        "problem_text": problem['problem_text'],
        "boundaries": list(range(len(scaffolds))),
        "scaffold_types": scaffolds,
        "scaffold_confidences": [0.8] * len(scaffolds),
    }


def run_stage1(problems: List[Dict], use_mock: bool = True, c1a_model: Any = None) -> List[Dict]:
    """
    Run Stage 1: C1-A boundary detection and scaffold classification.
    """
    print("\n=== Stage 1: C1-A Inference ===")
    print(f"  Mode: {'Mock' if use_mock else 'Real C1-A'}")

    results = []
    for problem in problems:
        if use_mock:
            result = mock_c1a_inference(problem)
        else:
            # Use real C1-A model
            if c1a_model is None:
                raise ValueError("C1-A model required when not using mock")

            c1a_result = c1a_model.inference(problem['problem_text'])

            # Convert C1-A output to our format
            # C1-A returns scaffold types per window - we take the unique sequence
            scaffold_types = []
            prev_type = None
            for stype in c1a_result.get('scaffold_types', []):
                if stype != prev_type:
                    scaffold_types.append(stype)
                    prev_type = stype

            # Ensure we have at least SETUP and ANSWER
            if not scaffold_types:
                scaffold_types = ["SETUP", "COMPUTE", "ANSWER"]
            elif scaffold_types[0] != "SETUP":
                scaffold_types.insert(0, "SETUP")
            if scaffold_types[-1] != "ANSWER":
                scaffold_types.append("ANSWER")

            result = {
                "problem_id": problem['problem_id'],
                "problem_text": problem['problem_text'],
                "boundaries": c1a_result.get('boundaries', list(range(len(scaffold_types)))),
                "scaffold_types": scaffold_types,
                "scaffold_confidences": c1a_result.get('scaffold_confidences', [0.5] * len(scaffold_types)),
            }

        results.append(result)
        print(f"  {problem['problem_id']}: {result['scaffold_types']}")

    return results


# ============================================================================
# Stage 2: Slot Filler Inference
# ============================================================================

class SlotFillerModel:
    """Slot filler model wrapper for inference."""

    def __init__(self, model_path: str, device: str = "cpu"):
        self.model_path = model_path
        self.device = device
        self.model = None
        self.tokenizer = None

    def load(self):
        """Load the model."""
        print(f"  Loading slot filler from {self.model_path}...")

        # Check if it's a LoRA adapter or merged model
        if os.path.exists(os.path.join(self.model_path, 'adapter_config.json')):
            # LoRA adapter - need base model
            print("    Loading base model + LoRA adapter...")
            base_model = AutoModelForCausalLM.from_pretrained(
                "Qwen/Qwen2.5-0.5B",
                torch_dtype=torch.float32,
                trust_remote_code=True,
            )
            self.model = PeftModel.from_pretrained(base_model, self.model_path)
            self.tokenizer = AutoTokenizer.from_pretrained(
                "Qwen/Qwen2.5-0.5B",
                trust_remote_code=True
            )
        else:
            # Merged model
            print("    Loading merged model...")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.float32,
                trust_remote_code=True,
            )
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                trust_remote_code=True
            )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = self.model.to(self.device)
        self.model.eval()
        print("    Model loaded!")

    def generate(self, scaffold_type: str, raw_text: str) -> str:
        """Generate rough output for a segment."""
        # Build prompt in training format
        prompt = f"[{scaffold_type.upper()}] {raw_text}\n[OUTPUT]"

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=256,
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=64,
                temperature=0.0,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        # Decode and extract output
        full_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract the part after [OUTPUT]
        if "[OUTPUT]" in full_text:
            output = full_text.split("[OUTPUT]")[-1].strip()
        else:
            output = full_text[len(prompt):].strip()

        # Clean up - stop at newline
        output = output.split("\n")[0].strip()

        # For SETUP: extract first equation (stop at period if followed by text)
        # e.g., "x^2 - 9 = 0. This is..." -> "x^2 - 9 = 0"
        if ". " in output and "=" in output:
            # Get part before first period followed by space
            output = output.split(". ")[0]

        return output

    def unload(self):
        """Unload model from memory."""
        del self.model
        del self.tokenizer
        self.model = None
        self.tokenizer = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def run_stage2(
    stage1_results: List[Dict],
    problems: List[Dict],
    slot_filler: Optional[SlotFillerModel] = None
) -> List[Dict]:
    """
    Run Stage 2: Slot filler inference for SETUP/SUBSTITUTE/THEOREM,
                 template preparation for EXPAND/SIMPLIFY/SOLVE/COMPUTE/ANSWER.
    """
    print("\n=== Stage 2: Slot Filler Inference ===")

    results = []

    for stage1, problem in zip(stage1_results, problems):
        segments = []

        for i, scaffold_type in enumerate(stage1['scaffold_types']):
            segment = {
                "segment_idx": i,
                "scaffold_type": scaffold_type,
            }

            if scaffold_type in SLOT_FILLER_SCAFFOLDS:
                # Need model inference
                segment["source"] = "slot_filler"

                if slot_filler is not None:
                    # Run slot filler
                    rough_output = slot_filler.generate(
                        scaffold_type,
                        problem['problem_text']
                    )
                    segment["rough_output"] = rough_output
                else:
                    # Mock output
                    if scaffold_type == "SETUP":
                        segment["rough_output"] = problem['problem_text']
                    elif scaffold_type == "SUBSTITUTE":
                        segment["rough_output"] = "x, 0"
                    elif scaffold_type == "THEOREM":
                        segment["rough_output"] = "algebra, general"
            else:
                # Template scaffold
                segment["source"] = "template"
                segment["rough_output"] = None

            segments.append(segment)

        result = {
            "problem_id": problem['problem_id'],
            "problem_text": problem['problem_text'],
            "expected_answer": problem.get('expected_answer'),
            "segments": segments,
        }
        results.append(result)

        print(f"  {problem['problem_id']}:")
        for seg in segments:
            source = seg['source']
            rough = seg.get('rough_output', 'N/A')[:30] if seg.get('rough_output') else 'template'
            print(f"    [{seg['scaffold_type']}] ({source}): {rough}")

    return results


# ============================================================================
# Stage 3: ODE Refinement and SymPy Execution
# ============================================================================

def rough_to_sympy(scaffold_type: str, rough: str, prev_result: Any = None) -> Tuple[Any, Optional[str]]:
    """
    Convert rough slot filler output to SymPy using SymPy's own parsers.

    No regex on mathematical content. Let sympify and parse_latex do the work.
    """
    import sympy
    from sympy import sympify
    from sympy.parsing.latex import parse_latex

    if not rough:
        return None, "Empty input"

    rough = rough.strip()

    # NO REGEX. Slot filler outputs SymPy-friendly notation.
    # sympify() parses it. parse_latex() handles LaTeX fallback.

    def try_sympify(s):
        """Try sympify first (SymPy-friendly notation), then parse_latex (LaTeX)."""
        if not s:
            return None

        # 1. Try sympify directly (slot filler should output SymPy-friendly notation)
        try:
            return sympify(s)
        except:
            pass

        # 2. Try parse_latex (for legacy LaTeX outputs)
        try:
            return parse_latex(s)
        except:
            pass

        # 3. For equations with =, try parsing as Eq()
        if '=' in s:
            parts = s.split('=')
            # Try last part (might be the answer in a chain)
            for part in reversed(parts):
                part = part.strip()
                if part:
                    try:
                        result = sympify(part)
                        if result.is_number:
                            return result
                    except:
                        pass
                    try:
                        return parse_latex(part)
                    except:
                        pass

        return None

    if scaffold_type == "SETUP":
        result = try_sympify(rough)
        if result is not None:
            return result, None
        return None, f"Could not parse: {rough[:50]}"

    elif scaffold_type == "SUBSTITUTE":
        # "expr, value" format
        parts = rough.split(',', 1)
        if len(parts) != 2:
            return None, f"Expected 'expr, value' format, got: {rough}"

        try:
            expr = sympify(parts[0].strip())
            value = sympify(parts[1].strip())

            if prev_result is not None:
                return prev_result.subs(expr, value), None
            else:
                return (expr, value), None
        except Exception as e:
            return None, f"Could not parse substitution: {e}"

    elif scaffold_type == "THEOREM":
        # Keywords only - no execution
        return rough, None

    else:
        return None, f"Unknown scaffold type: {scaffold_type}"


def run_stage3(stage2_results: List[Dict], oracle: SympyOracle) -> List[Dict]:
    """
    Run Stage 3: ODE refinement and SymPy execution.
    """
    print("\n=== Stage 3: SymPy Execution ===")

    results = []

    for problem in stage2_results:
        problem_result = {
            "problem_id": problem['problem_id'],
            "expected_answer": problem.get('expected_answer'),
            "segments": [],
            "final_answer": None,
            "correct": None,
        }

        context = {"prev_result": None}

        for segment in problem['segments']:
            scaffold_type = segment['scaffold_type']
            rough_output = segment.get('rough_output')

            seg_result = {
                "scaffold_type": scaffold_type,
                "rough_output": rough_output,
                "sympy_expr": None,
                "result": None,
                "success": False,
                "error": None,
            }

            # Template scaffolds
            if scaffold_type in TEMPLATE_SCAFFOLDS:
                # Use previous result or problem text as input
                prev = context.get('prev_result')
                expr = prev if prev is not None else problem['problem_text']

                success, result, message = execute_template(scaffold_type, str(expr))

                seg_result["success"] = success
                seg_result["result"] = str(result) if result is not None else None
                seg_result["sympy_expr"] = str(expr)
                if not success:
                    seg_result["error"] = message

                if success and result is not None:
                    context['prev_result'] = result

            # Slot filler scaffolds
            elif scaffold_type in SLOT_FILLER_SCAFFOLDS:
                if rough_output:
                    sympy_expr, error = rough_to_sympy(
                        scaffold_type,
                        rough_output,
                        context.get('prev_result')
                    )

                    if error:
                        seg_result["error"] = error
                    else:
                        seg_result["sympy_expr"] = str(sympy_expr)

                        if scaffold_type == "THEOREM":
                            # THEOREM just stores keywords
                            seg_result["result"] = str(sympy_expr)
                            seg_result["success"] = True
                        elif scaffold_type == "SETUP":
                            # For equations, auto-solve if next step is ANSWER
                            import sympy
                            from sympy import solve, simplify, N
                            result = sympy_expr

                            # If it's an equation and there's only ANSWER after, solve it
                            if isinstance(sympy_expr, sympy.Eq):
                                try:
                                    # Find free symbols
                                    free_syms = list(sympy_expr.free_symbols)
                                    if free_syms:
                                        solutions = solve(sympy_expr, free_syms[0])
                                        if solutions:
                                            result = solutions[0] if len(solutions) == 1 else solutions
                                except:
                                    pass

                            # Try to evaluate if it's arithmetic
                            try:
                                evaluated = N(result)
                                if evaluated.is_number:
                                    result = evaluated
                            except:
                                pass

                            seg_result["result"] = str(result) if result is not None else None
                            seg_result["success"] = True
                            context['prev_result'] = result
                        else:
                            # Try to simplify
                            try:
                                from sympy import simplify
                                result = simplify(sympy_expr) if sympy_expr is not None else None
                                seg_result["result"] = str(result) if result is not None else None
                                seg_result["success"] = True
                                context['prev_result'] = result
                            except Exception as e:
                                seg_result["error"] = str(e)
                else:
                    seg_result["error"] = "No rough output"

            problem_result["segments"].append(seg_result)

        # Get final answer
        for seg in reversed(problem_result["segments"]):
            if seg["success"] and seg["result"]:
                problem_result["final_answer"] = seg["result"]
                break

        # Check correctness using SymPy - NO REGEX
        if problem_result["final_answer"] and problem_result["expected_answer"]:
            final = str(problem_result["final_answer"])
            expected = str(problem_result["expected_answer"])

            from sympy import sympify, simplify, N
            from sympy.parsing.latex import parse_latex

            def try_parse(s):
                """Try sympify, then parse_latex."""
                try:
                    return sympify(s)
                except:
                    pass
                try:
                    return parse_latex(s)
                except:
                    pass
                return None

            # Try direct oracle comparison first
            problem_result["correct"] = oracle.compare(final, expected)

            # If failed, try SymPy symbolic comparison: simplify(a - b) == 0
            if not problem_result["correct"]:
                final_sym = try_parse(final)
                expected_sym = try_parse(expected)

                if final_sym is not None and expected_sym is not None:
                    try:
                        if simplify(final_sym - expected_sym) == 0:
                            problem_result["correct"] = True
                    except:
                        pass

                    # For lists like [-3, 3], check if expected is in the list
                    if not problem_result["correct"] and hasattr(final_sym, '__iter__'):
                        try:
                            for item in final_sym:
                                if simplify(item - expected_sym) == 0:
                                    problem_result["correct"] = True
                                    break
                        except:
                            pass

        results.append(problem_result)

        # Print summary
        status = "✓" if problem_result["correct"] else "✗"
        print(f"  {problem['problem_id']}: {status}")
        print(f"    Final: {problem_result['final_answer']}")
        print(f"    Expected: {problem_result['expected_answer']}")

    return results


# ============================================================================
# Results Summary
# ============================================================================

def print_summary(results: List[Dict]):
    """Print detailed summary of test results."""
    print("\n" + "=" * 60)
    print("                    PIPELINE TEST RESULTS")
    print("=" * 60)

    # Overall accuracy
    n_correct = sum(1 for r in results if r.get('correct'))
    n_with_answer = sum(1 for r in results if r.get('correct') is not None)

    print(f"\n📊 OVERALL ACCURACY: {n_correct}/{n_with_answer}")
    if n_with_answer > 0:
        print(f"   Percentage: {100 * n_correct / n_with_answer:.1f}%")

    # Per-step success
    total_segments = 0
    successful_segments = 0

    for r in results:
        for seg in r.get('segments', []):
            total_segments += 1
            if seg.get('success'):
                successful_segments += 1

    print(f"\n📈 PER-STEP SUCCESS: {successful_segments}/{total_segments}")
    if total_segments > 0:
        print(f"   Percentage: {100 * successful_segments / total_segments:.1f}%")

    # Per-scaffold-type breakdown
    scaffold_stats = defaultdict(lambda: {"total": 0, "success": 0})

    for r in results:
        for seg in r.get('segments', []):
            scaffold_type = seg.get('scaffold_type', 'UNKNOWN')
            scaffold_stats[scaffold_type]["total"] += 1
            if seg.get('success'):
                scaffold_stats[scaffold_type]["success"] += 1

    print("\n📋 PER-SCAFFOLD-TYPE BREAKDOWN:")
    for scaffold_type in sorted(scaffold_stats.keys()):
        stats = scaffold_stats[scaffold_type]
        pct = 100 * stats["success"] / stats["total"] if stats["total"] > 0 else 0
        source = "slot_filler" if scaffold_type in SLOT_FILLER_SCAFFOLDS else "template"
        print(f"   {scaffold_type:12} ({source:10}): {stats['success']:2}/{stats['total']:2} ({pct:5.1f}%)")

    # Error summary
    errors = []
    for r in results:
        for seg in r.get('segments', []):
            if seg.get('error'):
                errors.append({
                    "problem_id": r['problem_id'],
                    "scaffold_type": seg['scaffold_type'],
                    "error": seg['error']
                })

    if errors:
        print(f"\n⚠️  ERRORS ({len(errors)} total):")
        for err in errors[:5]:  # Show first 5
            print(f"   [{err['problem_id']}] {err['scaffold_type']}: {err['error'][:50]}")
        if len(errors) > 5:
            print(f"   ... and {len(errors) - 5} more")

    print("\n" + "=" * 60)


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="End-to-End Pipeline Test")
    parser.add_argument(
        '--slot-filler-model',
        type=str,
        default=None,
        help='Path to trained slot filler model (LoRA adapter or merged)'
    )
    parser.add_argument(
        '--mock-c1a',
        action='store_true',
        help='Use mock C1-A instead of real model'
    )
    parser.add_argument(
        '--c1a-model',
        type=str,
        default='s3://mycelium-data/models/c1a_coarse_v6_aux_telegraph/best_checkpoint/',
        help='Path to C1-A model (S3 or local)'
    )
    parser.add_argument(
        '--n-problems',
        type=int,
        default=10,
        help='Number of problems to test (default: 10)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=None,  # Random seed each run
        help='Random seed for problem selection (default: random)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device to run on (default: cuda if available)'
    )
    parser.add_argument(
        '--mock-problems',
        action='store_true',
        help='Use mock problems instead of real MATH500 from S3'
    )

    args = parser.parse_args()

    print("=" * 60)
    print("         MYCELIUM v7 END-TO-END PIPELINE TEST")
    print("=" * 60)
    print(f"Problems: {args.n_problems}")
    print(f"Device: {args.device}")
    print(f"Mock C1-A: {args.mock_c1a}")
    print(f"Slot Filler: {args.slot_filler_model or 'mock'}")

    # Initialize oracle
    oracle = SympyOracle(timeout=5)

    # Load problems - use real MATH500 by default, random seed each run
    import time
    seed = args.seed if args.seed is not None else int(time.time()) % 10000
    print(f"\n📚 Loading MATH problems (seed={seed})...")
    if args.mock_problems:
        problems = load_math_problems(args.n_problems, seed)
        print(f"   Loaded {len(problems)} mock problems")
    else:
        problems = load_real_math_problems(args.n_problems, seed)
        print(f"   Loaded {len(problems)} REAL MATH500 problems from S3")

    # Load C1-A model if not using mock
    c1a_model = None
    if not args.mock_c1a:
        if not C1A_AVAILABLE:
            print("ERROR: C1-A model not available. Install dependencies or use --mock-c1a")
            sys.exit(1)

        print("\n🧠 Loading C1-A model...")
        # Update config with S3 paths
        c1a_config = C1AConfig(
            lora_path=args.c1a_model.rstrip('/') + '/lora_adapters/',
            head_weights_path=args.c1a_model.rstrip('/') + '/head_weights.pt',
            device=args.device,
        )
        c1a_model = C1AModel(c1a_config)
        c1a_model.load()

    # Load slot filler model if provided
    slot_filler = None
    if args.slot_filler_model:
        slot_filler = SlotFillerModel(args.slot_filler_model, args.device)
        slot_filler.load()

    try:
        # Stage 1: C1-A
        stage1_results = run_stage1(problems, use_mock=args.mock_c1a, c1a_model=c1a_model)

        # Stage 2: Slot Filler
        stage2_results = run_stage2(stage1_results, problems, slot_filler)

        # Stage 3: SymPy Execution
        stage3_results = run_stage3(stage2_results, oracle)

        # Summary
        print_summary(stage3_results)

        # Save results
        output_path = "/tmp/pipeline_test_results.json"
        with open(output_path, 'w') as f:
            json.dump(stage3_results, f, indent=2, default=str)
        print(f"\nResults saved to: {output_path}")

    finally:
        # Cleanup
        if slot_filler:
            slot_filler.unload()
        if c1a_model:
            c1a_model.unload()


if __name__ == '__main__':
    main()
