#!/usr/bin/env python3
"""
Stage 1: C1-A Batch Inference

Runs the frozen C1-A model (Qwen-0.5B + LoRA) on all problems to extract:
- boundaries: list of boundary positions (from coarse windows)
- scaffold_types: 7-class prediction per segment (SETUP, SUBSTITUTE, SIMPLIFY, SOLVE, COMPUTE, THEOREM, OTHER)
- hidden_states: 896-dim cached states for downstream ODE processing

Architecture:
- Base: Qwen/Qwen2-0.5B
- LoRA: r=16, targets q_proj, k_proj, v_proj, o_proj
- Window head: 896 -> 64 -> 1 (boundary detection)
- Scaffold head: 896 -> 128 -> 7 (scaffold classification)
- Window size: 16 tokens, stride: 8 tokens (50% overlap)

GPU Rules:
- Uses torch.float32 (A10G doesn't support bfloat16)
- Designed for g5.xlarge (24GB VRAM)

Usage:
    python scripts/stage1_c1a_inference.py --input-file problems.jsonl --output-dir s3://mycelium-data-v7/inference/stage1/
"""

import argparse
import json
import os
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import boto3
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# Conditional imports
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
except ImportError:
    print("ERROR: transformers and peft required. Install with:")
    print("  pip install transformers peft")
    sys.exit(1)


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class C1AConfig:
    """C1-A model configuration."""
    # Model paths
    base_model: str = "Qwen/Qwen2-0.5B"
    lora_path: str = "s3://mycelium-data-v7/models/c1a/best_checkpoint/lora_adapters/"
    head_weights_path: str = "s3://mycelium-data-v7/models/c1a/best_checkpoint/head_weights.pt"
    scaffold_data_path: str = "s3://mycelium-data/scaffold_training/"

    # Architecture
    hidden_size: int = 896
    window_size: int = 16
    stride: int = 8
    n_scaffold_classes: int = 7

    # Scaffold types (order matters - matches training labels)
    scaffold_types: List[str] = field(default_factory=lambda: [
        "SETUP", "SUBSTITUTE", "SIMPLIFY", "SOLVE", "COMPUTE", "THEOREM", "OTHER"
    ])

    # Inference
    batch_size: int = 8
    boundary_threshold: float = 0.5
    max_length: int = 512

    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: torch.dtype = torch.float32  # A10G doesn't support bfloat16


# ============================================================================
# S3 Utilities
# ============================================================================

def download_from_s3(s3_path: str, local_path: str) -> str:
    """Download file or directory from S3."""
    if not s3_path.startswith("s3://"):
        return s3_path  # Already local

    s3 = boto3.client('s3')

    # Parse S3 path
    parts = s3_path.replace("s3://", "").split("/", 1)
    bucket = parts[0]
    key = parts[1] if len(parts) > 1 else ""

    # Check if it's a directory (ends with / or no extension in last part)
    is_dir = s3_path.endswith("/") or "." not in key.split("/")[-1]

    if is_dir:
        # List and download all files
        os.makedirs(local_path, exist_ok=True)
        paginator = s3.get_paginator('list_objects_v2')
        prefix = key.rstrip("/") + "/" if key else ""

        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            for obj in page.get('Contents', []):
                obj_key = obj['Key']
                rel_path = obj_key[len(prefix):] if prefix else obj_key
                if rel_path:
                    local_file = os.path.join(local_path, rel_path)
                    os.makedirs(os.path.dirname(local_file), exist_ok=True)
                    s3.download_file(bucket, obj_key, local_file)
        return local_path
    else:
        # Single file
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        s3.download_file(bucket, key, local_path)
        return local_path


def upload_to_s3(local_path: str, s3_path: str):
    """Upload file to S3."""
    if not s3_path.startswith("s3://"):
        # Copy locally instead
        import shutil
        os.makedirs(os.path.dirname(s3_path), exist_ok=True)
        shutil.copy2(local_path, s3_path)
        return

    s3 = boto3.client('s3')
    parts = s3_path.replace("s3://", "").split("/", 1)
    bucket = parts[0]
    key = parts[1]
    s3.upload_file(local_path, bucket, key)


def load_from_s3_bytes(s3_path: str) -> bytes:
    """Load file bytes from S3."""
    s3 = boto3.client('s3')
    parts = s3_path.replace("s3://", "").split("/", 1)
    bucket = parts[0]
    key = parts[1]
    obj = s3.get_object(Bucket=bucket, Key=key)
    return obj['Body'].read()


# ============================================================================
# LaTeX Preprocessing (must match training)
# ============================================================================

def preprocess_latex(text: str) -> str:
    """
    Deterministic LaTeX normalization using SymPy.
    Uses parse_latex for proper parsing, with string-based fallback for display math delimiters.
    """
    if not text:
        return text

    # Import SymPy's latex parser
    from sympy.parsing.latex import parse_latex
    from sympy import latex, sympify

    # Remove display math delimiters (structural, not math parsing)
    # These are document-level delimiters, not mathematical content
    text = text.replace('\\[', '').replace('\\]', '')
    text = text.replace('$$', '')

    # Try to parse with SymPy and convert back to normalized string
    try:
        # Check if it looks like LaTeX math
        if '\\' in text or ('{' in text and '}' in text):
            parsed = parse_latex(text)
            # Convert back to a normalized string representation
            return str(parsed)
        else:
            # For non-LaTeX text, try sympify
            parsed = sympify(text, evaluate=False)
            return str(parsed)
    except Exception:
        # If parsing fails, return cleaned text as-is
        # Clean up multiple spaces using simple split/join
        return ' '.join(text.split())


# ============================================================================
# Neural Network Heads
# ============================================================================

class WindowHead(nn.Module):
    """
    Boundary detection head for coarse windows.
    Architecture: Linear(896, 64) -> ReLU -> Dropout -> Linear(64, 1)
    """
    def __init__(self, hidden_size: int = 896):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, hidden_size)
        Returns:
            (batch, seq_len) boundary logits
        """
        return self.mlp(x).squeeze(-1)


class TelegraphHead(nn.Module):
    """
    Auxiliary telegraph prediction head (used during training only).
    Architecture: Linear(896, 128) -> ReLU -> Dropout -> Linear(128, 1)
    """
    def __init__(self, hidden_size: int = 896):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x).squeeze(-1)


class ScaffoldHead(nn.Module):
    """
    Scaffold type classifier head.
    Trained on window-level hidden states to predict scaffold types.
    Architecture: Linear(896, 128) -> ReLU -> Dropout -> Linear(128, 7)
    """
    def __init__(self, hidden_size: int = 896, n_classes: int = 7):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, n_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, hidden_size) or (batch, n_windows, hidden_size)
        Returns:
            (batch, n_classes) or (batch, n_windows, n_classes) logits
        """
        return self.mlp(x)


# ============================================================================
# C1-A Model
# ============================================================================

class C1AModel:
    """
    C1-A inference model.

    Combines:
    - Qwen-0.5B backbone
    - LoRA adapters
    - Window head (boundary detection)
    - Scaffold head (scaffold type classification)
    """

    def __init__(self, config: C1AConfig):
        self.config = config
        self.device = config.device
        self.dtype = config.dtype

        self.model = None
        self.tokenizer = None
        self.window_head = None
        self.scaffold_head = None

    def load(self, cache_dir: Optional[str] = None):
        """Load the model and heads."""
        print(f"Loading C1-A model on {self.device} with dtype={self.dtype}")

        if cache_dir is None:
            cache_dir = tempfile.mkdtemp(prefix="c1a_cache_")

        # Load tokenizer
        print("  Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.base_model,
            trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load base model
        print("  Loading base model...")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.base_model,
            torch_dtype=self.dtype,
            trust_remote_code=True,
            output_hidden_states=True
        )

        # Load LoRA adapters
        print("  Loading LoRA adapters...")
        lora_local = os.path.join(cache_dir, "lora_adapters")
        download_from_s3(self.config.lora_path, lora_local)
        self.model = PeftModel.from_pretrained(
            self.model,
            lora_local,
            torch_dtype=self.dtype
        )

        # Move to device
        self.model = self.model.to(self.device)
        self.model.eval()

        # Load window head
        print("  Loading window head...")
        head_bytes = load_from_s3_bytes(self.config.head_weights_path)
        import io
        head_weights = torch.load(io.BytesIO(head_bytes), map_location='cpu')

        self.window_head = WindowHead(self.config.hidden_size)
        # Handle both formats: full window_head state_dict or just mlp state_dict
        window_head_weights = head_weights['window_head']
        if any(k.startswith('mlp.') for k in window_head_weights.keys()):
            self.window_head.load_state_dict(window_head_weights)
        else:
            self.window_head.mlp.load_state_dict(window_head_weights)
        self.window_head = self.window_head.to(self.device).to(self.dtype)
        self.window_head.eval()

        # Load or train scaffold head
        print("  Loading scaffold head...")
        self.scaffold_head = self._load_or_train_scaffold_head(cache_dir)

        print("  C1-A model loaded successfully!")

    def _load_or_train_scaffold_head(self, cache_dir: str) -> ScaffoldHead:
        """Load scaffold head from S3 or train from scaffold data."""
        scaffold_head = ScaffoldHead(
            self.config.hidden_size,
            self.config.n_scaffold_classes
        )

        # Check if scaffold head exists in S3 (use v2 trained on problem text)
        scaffold_head_path = "s3://mycelium-data-v7/models/c1a/best_checkpoint/scaffold_head_v2.pt"
        try:
            head_bytes = load_from_s3_bytes(scaffold_head_path)
            import io
            state_dict = torch.load(io.BytesIO(head_bytes), map_location='cpu')
            scaffold_head.load_state_dict(state_dict)
            print("    Loaded pre-trained scaffold head from S3")
        except Exception as e:
            print(f"    No pre-trained scaffold head found, training from scratch...")
            scaffold_head = self._train_scaffold_head(cache_dir)

            # Save for future use
            try:
                with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
                    torch.save(scaffold_head.state_dict(), f.name)
                    upload_to_s3(f.name, scaffold_head_path)
                    os.unlink(f.name)
                print("    Saved scaffold head to S3")
            except Exception as save_err:
                print(f"    Warning: Could not save scaffold head: {save_err}")

        scaffold_head = scaffold_head.to(self.device).to(self.dtype)
        scaffold_head.eval()
        return scaffold_head

    def _train_scaffold_head(self, cache_dir: str) -> ScaffoldHead:
        """Train scaffold head from scaffold training data."""
        import io

        # Download scaffold training data
        print("    Downloading scaffold training data...")
        scaffold_dir = os.path.join(cache_dir, "scaffold_training")
        download_from_s3(self.config.scaffold_data_path, scaffold_dir)

        # Load features and labels
        features = np.load(os.path.join(scaffold_dir, "features.npy"))
        labels = np.load(os.path.join(scaffold_dir, "labels.npy"))

        print(f"    Training scaffold head on {len(features)} samples...")

        # Convert to tensors
        X = torch.from_numpy(features).float()
        y = torch.from_numpy(labels).long()

        # Split train/val
        n_train = int(0.9 * len(X))
        indices = torch.randperm(len(X))
        train_idx = indices[:n_train]
        val_idx = indices[n_train:]

        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        # Create model and optimizer
        scaffold_head = ScaffoldHead(self.config.hidden_size, self.config.n_scaffold_classes)
        optimizer = torch.optim.AdamW(scaffold_head.parameters(), lr=1e-3, weight_decay=0.01)
        criterion = nn.CrossEntropyLoss()

        # Training loop
        scaffold_head.train()
        best_val_acc = 0
        best_state = None

        for epoch in range(50):
            # Mini-batch training
            perm = torch.randperm(len(X_train))
            total_loss = 0
            n_batches = 0

            for i in range(0, len(X_train), 64):
                batch_idx = perm[i:i+64]
                batch_X = X_train[batch_idx]
                batch_y = y_train[batch_idx]

                optimizer.zero_grad()
                logits = scaffold_head(batch_X)
                loss = criterion(logits, batch_y)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                n_batches += 1

            # Validation
            scaffold_head.eval()
            with torch.no_grad():
                val_logits = scaffold_head(X_val)
                val_preds = val_logits.argmax(dim=-1)
                val_acc = (val_preds == y_val).float().mean().item()
            scaffold_head.train()

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state = {k: v.clone() for k, v in scaffold_head.state_dict().items()}

            if (epoch + 1) % 10 == 0:
                print(f"      Epoch {epoch+1}: loss={total_loss/n_batches:.4f}, val_acc={val_acc:.4f}")

        print(f"    Best validation accuracy: {best_val_acc:.4f}")

        # Load best state
        scaffold_head.load_state_dict(best_state)
        scaffold_head.eval()

        return scaffold_head

    def unload(self):
        """Unload model from memory."""
        print("Unloading C1-A model...")
        del self.model
        del self.tokenizer
        del self.window_head
        del self.scaffold_head
        self.model = None
        self.tokenizer = None
        self.window_head = None
        self.scaffold_head = None

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("  Model unloaded")

    @torch.no_grad()
    def get_hidden_states(self, text: str) -> Tuple[torch.Tensor, List[int]]:
        """
        Get hidden states for text.

        Returns:
            hidden_states: (seq_len, hidden_size) tensor
            token_ids: list of token IDs
        """
        # Preprocess
        text = preprocess_latex(text)

        # Tokenize
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.max_length,
            padding=False
        )
        input_ids = inputs['input_ids'].to(self.device)

        # Forward pass
        outputs = self.model(
            input_ids=input_ids,
            output_hidden_states=True
        )

        # Get last hidden state
        hidden_states = outputs.hidden_states[-1][0]  # (seq_len, hidden_size)

        return hidden_states, input_ids[0].tolist()

    @torch.no_grad()
    def extract_window_features(
        self,
        hidden_states: torch.Tensor
    ) -> Tuple[torch.Tensor, List[Tuple[int, int]]]:
        """
        Extract window-level features using coarse windowing.

        Args:
            hidden_states: (seq_len, hidden_size)

        Returns:
            window_features: (n_windows, hidden_size)
            window_ranges: list of (start, end) for each window
        """
        seq_len = hidden_states.shape[0]
        window_size = self.config.window_size
        stride = self.config.stride

        window_features = []
        window_ranges = []

        for start in range(0, seq_len - window_size + 1, stride):
            end = start + window_size
            # Mean pool over window
            window_feat = hidden_states[start:end].mean(dim=0)
            window_features.append(window_feat)
            window_ranges.append((start, end))

        # Handle final partial window if needed
        if len(window_ranges) == 0 or window_ranges[-1][1] < seq_len:
            start = max(0, seq_len - window_size)
            end = seq_len
            if end > start:
                window_feat = hidden_states[start:end].mean(dim=0)
                window_features.append(window_feat)
                window_ranges.append((start, end))

        if not window_features:
            # Edge case: very short sequence
            window_feat = hidden_states.mean(dim=0)
            window_features.append(window_feat)
            window_ranges.append((0, seq_len))

        return torch.stack(window_features), window_ranges

    @torch.no_grad()
    def predict_boundaries(
        self,
        window_features: torch.Tensor
    ) -> Tuple[List[int], List[float]]:
        """
        Predict boundaries from window features.

        Args:
            window_features: (n_windows, hidden_size)

        Returns:
            boundary_indices: list of window indices with boundaries
            boundary_probs: list of boundary probabilities
        """
        logits = self.window_head(window_features.unsqueeze(0))[0]  # (n_windows,)
        probs = torch.sigmoid(logits).cpu().numpy()

        boundary_indices = []
        boundary_probs = []

        for i, prob in enumerate(probs):
            if prob > self.config.boundary_threshold:
                boundary_indices.append(i)
                boundary_probs.append(float(prob))

        return boundary_indices, boundary_probs

    @torch.no_grad()
    def predict_scaffold_types(
        self,
        window_features: torch.Tensor
    ) -> Tuple[List[str], List[float]]:
        """
        Predict scaffold types for each window.

        Args:
            window_features: (n_windows, hidden_size)

        Returns:
            scaffold_types: list of scaffold type names
            confidences: list of prediction confidences
        """
        logits = self.scaffold_head(window_features)  # (n_windows, n_classes)
        probs = F.softmax(logits, dim=-1)

        pred_indices = logits.argmax(dim=-1).cpu().numpy()
        confidences = probs.max(dim=-1).values.cpu().numpy()

        scaffold_types = [self.config.scaffold_types[i] for i in pred_indices]

        return scaffold_types, [float(c) for c in confidences]

    @torch.no_grad()
    def inference(self, problem_text: str) -> Dict[str, Any]:
        """
        Run full C1-A inference on a problem.

        Args:
            problem_text: The problem text

        Returns:
            Dict with:
                - boundaries: list of boundary token positions
                - boundary_probs: list of boundary probabilities
                - scaffold_types: list of scaffold type per segment
                - scaffold_confidences: list of confidence scores
                - hidden_states: (n_windows, hidden_size) numpy array
                - window_ranges: list of (start, end) for each window
                - n_tokens: number of tokens in input
        """
        # Get hidden states
        hidden_states, token_ids = self.get_hidden_states(problem_text)
        n_tokens = len(token_ids)

        # Extract window features
        window_features, window_ranges = self.extract_window_features(hidden_states)

        # Predict boundaries
        boundary_indices, boundary_probs = self.predict_boundaries(window_features)

        # Convert window indices to token positions (use window midpoint)
        boundary_positions = []
        for idx in boundary_indices:
            start, end = window_ranges[idx]
            midpoint = (start + end) // 2
            boundary_positions.append(midpoint)

        # Predict scaffold types
        scaffold_types, scaffold_confidences = self.predict_scaffold_types(window_features)

        return {
            "boundaries": boundary_positions,
            "boundary_probs": boundary_probs,
            "scaffold_types": scaffold_types,
            "scaffold_confidences": scaffold_confidences,
            "hidden_states": window_features.cpu().numpy(),
            "window_ranges": window_ranges,
            "n_tokens": n_tokens
        }

    @torch.no_grad()
    def batch_inference(
        self,
        problems: List[Dict],
        show_progress: bool = True
    ) -> Dict[str, Dict[str, Any]]:
        """
        Run inference on a batch of problems.

        Args:
            problems: List of dicts with 'problem_id' and 'problem_text'
            show_progress: Whether to show progress bar

        Returns:
            Dict mapping problem_id -> inference results
        """
        results = {}

        iterator = tqdm(problems, desc="C1-A inference") if show_progress else problems

        for problem in iterator:
            problem_id = str(problem.get('problem_id', problem.get('problem_idx', '')))
            problem_text = problem.get('problem_text', problem.get('problem', ''))

            if not problem_text:
                print(f"Warning: Empty problem text for {problem_id}")
                continue

            try:
                result = self.inference(problem_text)
                results[problem_id] = result
            except Exception as e:
                print(f"Error processing problem {problem_id}: {e}")
                results[problem_id] = {
                    "error": str(e),
                    "boundaries": [],
                    "scaffold_types": [],
                    "hidden_states": None
                }

        return results


# ============================================================================
# Data Loading
# ============================================================================

def load_problems(input_path: str) -> List[Dict]:
    """Load problems from file or S3."""
    if input_path.startswith("s3://"):
        # Download from S3
        data = load_from_s3_bytes(input_path).decode('utf-8')
        lines = data.strip().split('\n')
    else:
        with open(input_path, 'r') as f:
            lines = f.readlines()

    problems = []
    for line in lines:
        if line.strip():
            problems.append(json.loads(line))

    return problems


def save_results(
    results: Dict[str, Dict],
    output_dir: str,
    save_hidden_states: bool = True
):
    """
    Save inference results.

    Creates:
        - results.json: Main results without hidden states
        - hidden_states.npz: Numpy archive with hidden states (optional)
    """
    # Separate hidden states from main results
    main_results = {}
    hidden_states_dict = {}

    for problem_id, result in results.items():
        main_result = {k: v for k, v in result.items() if k != 'hidden_states'}
        main_results[problem_id] = main_result

        if save_hidden_states and result.get('hidden_states') is not None:
            hidden_states_dict[problem_id] = result['hidden_states']

    # Save main results
    results_json = json.dumps(main_results, indent=2)

    if output_dir.startswith("s3://"):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write(results_json)
            f.flush()
            upload_to_s3(f.name, os.path.join(output_dir.rstrip('/'), 'results.json'))
            os.unlink(f.name)
    else:
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, 'results.json'), 'w') as f:
            f.write(results_json)

    # Save hidden states
    if save_hidden_states and hidden_states_dict:
        if output_dir.startswith("s3://"):
            with tempfile.NamedTemporaryFile(suffix='.npz', delete=False) as f:
                np.savez_compressed(f.name, **hidden_states_dict)
                upload_to_s3(f.name, os.path.join(output_dir.rstrip('/'), 'hidden_states.npz'))
                os.unlink(f.name)
        else:
            np.savez_compressed(
                os.path.join(output_dir, 'hidden_states.npz'),
                **hidden_states_dict
            )

    print(f"Saved results to {output_dir}")
    print(f"  - {len(main_results)} problems processed")
    if save_hidden_states:
        print(f"  - {len(hidden_states_dict)} hidden state arrays saved")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Stage 1: C1-A Batch Inference")
    parser.add_argument(
        "--input-file",
        type=str,
        default="s3://mycelium-data/c1_training_v6/merged_training.jsonl",
        help="Path to input JSONL file (local or s3://)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="s3://mycelium-data-v7/inference/stage1/",
        help="Output directory (local or s3://)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for inference"
    )
    parser.add_argument(
        "--no-hidden-states",
        action="store_true",
        help="Don't save hidden states (saves disk space)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of problems (for testing)"
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Directory to cache downloaded models"
    )

    args = parser.parse_args()

    # Initialize config
    config = C1AConfig(batch_size=args.batch_size)

    # Load model
    model = C1AModel(config)
    model.load(cache_dir=args.cache_dir)

    try:
        # Load problems
        print(f"\nLoading problems from {args.input_file}...")
        problems = load_problems(args.input_file)

        if args.limit:
            problems = problems[:args.limit]

        print(f"Loaded {len(problems)} problems")

        # Run inference
        print("\nRunning C1-A inference...")
        results = model.batch_inference(problems)

        # Save results
        print(f"\nSaving results to {args.output_dir}...")
        save_results(
            results,
            args.output_dir,
            save_hidden_states=not args.no_hidden_states
        )

        # Print summary
        n_with_boundaries = sum(1 for r in results.values() if r.get('boundaries'))
        n_errors = sum(1 for r in results.values() if r.get('error'))

        print("\n=== Summary ===")
        print(f"Total problems: {len(problems)}")
        print(f"Processed successfully: {len(results) - n_errors}")
        print(f"Problems with boundaries: {n_with_boundaries}")
        print(f"Errors: {n_errors}")

        # Show sample result
        if results:
            sample_id = list(results.keys())[0]
            sample = results[sample_id]
            print(f"\nSample result (problem {sample_id}):")
            print(f"  Boundaries: {sample.get('boundaries', [])}")
            print(f"  Scaffold types: {sample.get('scaffold_types', [])[:5]}...")
            if sample.get('hidden_states') is not None:
                print(f"  Hidden states shape: {sample['hidden_states'].shape}")

    finally:
        # Always unload model
        model.unload()


if __name__ == "__main__":
    main()
