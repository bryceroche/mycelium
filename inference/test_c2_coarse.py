#!/usr/bin/env python3
"""Test C2 coarse-grain model (14 labels)."""

import torch
import torch.nn as nn
import boto3
from transformers import AutoTokenizer, AutoModel

# Coarse-grain labels (14)
LABELS = ['FACTORIAL', 'LOG', 'TRIG', 'MOD', 'SQRT', 'CUBE', 'FRAC_POW',
          'HIGH_POW', 'SQUARE', 'EQUATION', 'DIV', 'MUL', 'ADD', 'OTHER']

BACKBONE = "sentence-transformers/all-MiniLM-L6-v2"


class C2Classifier(nn.Module):
    """Multi-label classifier for operation detection with heartbeat head."""
    def __init__(self, backbone_name, num_labels):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(backbone_name)
        hidden_size = self.backbone.config.hidden_size  # 384 for MiniLM

        # Match the saved model architecture (hidden/2 = 192 for intermediate)
        self.classifier = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size // 2),  # 384 -> 192
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, num_labels)    # 192 -> 14
        )

        self.heartbeat_head = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size // 4),  # 384 -> 96
            nn.ReLU(),
            nn.Linear(hidden_size // 4, 1)             # 96 -> 1
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.pooler_output  # Use pooler output
        logits = self.classifier(pooled)
        heartbeat = self.heartbeat_head(pooled)
        return logits, heartbeat


def load_model():
    """Load C2 model from S3."""
    s3 = boto3.client('s3')

    # Download model
    local_path = "/tmp/c2_heartbeat_model.pt"
    s3.download_file("mycelium-data", "models/c2_heartbeat/model.pt", local_path)

    # Load checkpoint
    checkpoint = torch.load(local_path, map_location='cpu', weights_only=False)

    # Get labels from checkpoint
    id2label = checkpoint.get('id2label', {i: l for i, l in enumerate(LABELS)})
    num_labels = checkpoint.get('num_labels', len(LABELS))
    print(f"Checkpoint labels: {num_labels}")
    print(f"Metrics: {checkpoint.get('metrics', {})}")

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(BACKBONE)
    model = C2Classifier(BACKBONE, num_labels)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    return model, tokenizer, id2label


def predict(model, tokenizer, id2label, text, threshold=0.5):
    """Predict operations for a math problem."""
    inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=512)

    with torch.no_grad():
        logits, heartbeat = model(inputs['input_ids'], inputs['attention_mask'])
        probs = torch.sigmoid(logits).squeeze()
        hb_prob = torch.sigmoid(heartbeat).item()

    predictions = []
    for i, prob in enumerate(probs):
        label = id2label.get(i, id2label.get(str(i), f"LABEL_{i}"))
        if prob > threshold:
            predictions.append((label, prob.item()))

    return sorted(predictions, key=lambda x: -x[1]), hb_prob


# Test examples
TEST_CASES = [
    ("What is 5 + 3?", ["ADD"]),
    ("Calculate 12 * 7.", ["MUL"]),
    ("Find the square root of 144.", ["SQRT"]),
    ("What is 5! (5 factorial)?", ["FACTORIAL"]),
    ("Solve for x: 2x + 3 = 7", ["ADD", "MUL"]),
    ("Find sin(30°) + cos(60°)", ["TRIG", "ADD"]),
    ("Calculate log base 2 of 8", ["LOG"]),
    ("What is 17 mod 5?", ["MOD"]),
    ("Find the cube of 4.", ["CUBE"]),
    ("Calculate 2^10", ["HIGH_POW"]),
]


def main():
    print("Loading C2 coarse-grain model...")
    model, tokenizer, id2label = load_model()
    print(f"Model loaded. Labels: {list(id2label.values())}")
    print()

    correct = 0
    total = 0

    for text, expected in TEST_CASES:
        predictions, hb = predict(model, tokenizer, id2label, text, threshold=0.3)
        pred_labels = [p[0] for p in predictions]

        # Check if all expected labels are in predictions
        hit = all(e in pred_labels for e in expected)
        correct += hit
        total += 1

        status = "✓" if hit else "✗"
        print(f"{status} Input: {text[:60]}...")
        print(f"  Expected: {expected}")
        print(f"  Predicted: {predictions[:5]}")
        print(f"  Heartbeat prob: {hb:.3f}")
        print()

    print(f"Accuracy: {correct}/{total} ({100*correct/total:.1f}%)")


if __name__ == "__main__":
    main()
