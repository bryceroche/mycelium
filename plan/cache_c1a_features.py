"""
Cache C1-A Hidden States for C2/C3 Training

Runs C1-A inference on all training problems and saves per-window hidden states.

Usage:
    python cache_c1a_features.py  # Run on GPU (g5.xlarge)
"""

import json
import io
import argparse
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from peft import PeftModel
import boto3
from botocore.config import Config

config = Config(read_timeout=120, connect_timeout=30, retries={'max_attempts': 3})
s3 = boto3.client("s3", config=config)
BUCKET = "mycelium-data"

# Window parameters
W = 16  # Window size
S = 8   # Stride

# C1-A model path
MODEL_NAME = "Qwen/Qwen2-0.5B"
ADAPTER_PATH = "s3://mycelium-data/models/c1a_coarse_v6_aux_telegraph/"


def preprocess_latex(text: str) -> str:
    """Apply LaTeX preprocessing to match C1-A training."""
    import re
    text = text.replace("\\left", "").replace("\\right", "")
    text = text.replace("\\begin{array}", "").replace("\\end{array}", "")
    text = re.sub(r'\\text\{([^}]*)\}', r'\1', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def load_c1a_model(device):
    """Load C1-A model with LoRA adapter."""
    print("Loading C1-A model...")

    # Download adapter from S3
    adapter_local = Path("/tmp/c1a_adapter")
    adapter_local.mkdir(exist_ok=True)

    # List adapter files
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=BUCKET, Prefix="models/c1a_coarse_v6_aux_telegraph/"):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            filename = key.split("/")[-1]
            if filename:
                local_path = adapter_local / filename
                s3.download_file(BUCKET, key, str(local_path))
                print(f"  Downloaded {filename}")

    # Load base model
    print("  Loading base model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    base_model = AutoModel.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        torch_dtype=torch.float32,  # A10G doesn't support bfloat16
    )

    # Load LoRA adapter
    print("  Loading LoRA adapter...")
    model = PeftModel.from_pretrained(base_model, str(adapter_local))
    model = model.to(device)
    model.eval()

    print(f"  Model loaded on {device}")
    return model, tokenizer


def load_problems():
    """Load C1 training problems."""
    print("Loading problems...")
    resp = s3.get_object(Bucket=BUCKET, Key="c1_training_v6/merged_training.jsonl")

    problems = []
    for line in resp["Body"].iter_lines():
        p = json.loads(line.decode("utf-8"))
        problems.append({
            "problem_idx": str(p["problem_idx"]),
            "problem_text": p.get("original_text", p.get("problem_text", "")),
        })

    print(f"  Loaded {len(problems)} problems")
    return problems


def cache_features(model, tokenizer, problems, device, batch_size=1):
    """Run inference and cache per-window features."""
    print("\nCaching features...")

    all_features = {}

    for i, problem in enumerate(problems):
        pid = problem["problem_idx"]
        text = preprocess_latex(problem["problem_text"])

        # Tokenize
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=False
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            hidden_states = outputs.last_hidden_state  # (1, seq_len, hidden_dim)

        seq_len = hidden_states.shape[1]
        hidden_dim = hidden_states.shape[2]

        # Extract per-window features (full token features, not pooled)
        windows = []
        for w_start in range(0, seq_len, S):
            w_end = min(w_start + W, seq_len)
            if w_end - w_start < W // 2:
                break

            # Get window token features
            window_feats = hidden_states[0, w_start:w_end, :]  # (actual_len, hidden_dim)

            # Pad to W if needed
            actual_len = window_feats.shape[0]
            if actual_len < W:
                pad = torch.zeros(W - actual_len, hidden_dim, device=device)
                window_feats = torch.cat([window_feats, pad], dim=0)

            windows.append(window_feats.cpu().numpy())

        if windows:
            all_features[pid] = np.stack(windows)  # (n_windows, W, hidden_dim)
        else:
            # Edge case: very short problem
            all_features[pid] = np.zeros((1, W, hidden_dim), dtype=np.float32)

        if (i + 1) % 100 == 0:
            print(f"  [{i+1}/{len(problems)}] {pid}: {len(windows)} windows")

    return all_features


def save_features(all_features):
    """Save features to S3."""
    print("\nSaving features to S3...")

    output_prefix = "c2c3_training_ready/cached_features/"

    # Save each problem's features as separate .npy file
    for pid, features in all_features.items():
        buffer = io.BytesIO()
        np.save(buffer, features)
        buffer.seek(0)

        s3.put_object(
            Bucket=BUCKET,
            Key=f"{output_prefix}{pid}.npy",
            Body=buffer.read()
        )

    # Save manifest
    manifest = {
        "n_problems": len(all_features),
        "problems": list(all_features.keys()),
        "window_size": W,
        "stride": S,
    }
    s3.put_object(
        Bucket=BUCKET,
        Key=f"{output_prefix}manifest.json",
        Body=json.dumps(manifest, indent=2).encode("utf-8")
    )

    print(f"  Saved {len(all_features)} feature files to s3://{BUCKET}/{output_prefix}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"Using device: {device}")

    # Load model
    model, tokenizer = load_c1a_model(device)

    # Load problems
    problems = load_problems()

    # Cache features
    all_features = cache_features(model, tokenizer, problems, device)

    # Save to S3
    save_features(all_features)

    print("\nDone!")


if __name__ == "__main__":
    main()
