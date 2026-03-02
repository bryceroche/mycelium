"""
C1: Heartbeat Counter - Predicts pulse count range from problem text

Outputs Gaussian range (mu, sigma) for adaptive C2 budget:
- mu = predicted pulse count
- sigma = uncertainty (learned per-problem)
- range = [floor(mu - sigma), ceil(mu + sigma)]

C2 uses this range as operation budget hint. MCTS uses range width for search:
- Tight range → greedy is enough
- Wide range → explore more
"""
import json
import math
import torch
import torch.nn as nn
import numpy as np
import subprocess
from transformers import (
    AutoTokenizer,
    AutoModel,
    TrainingArguments,
    Trainer,
    EvalPrediction
)
from torch.utils.data import Dataset
from dataclasses import dataclass

MAX_SEQ_LEN = 256
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DATA_PATH = "s3://mycelium-data/c2_training/c2_train_with_heartbeats.json"

# Clamp log_var to prevent degenerate variance
LOG_VAR_MIN = -4.0  # min sigma = exp(-2) ≈ 0.14
LOG_VAR_MAX = 4.0   # max sigma = exp(2) ≈ 7.4


class C1HeartbeatRegressor(nn.Module):
    """
    Predicts pulse count as Gaussian: mu ± sigma.

    Separate heads for mean and log-variance to allow independent capacity.
    """

    def __init__(self):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(MODEL_NAME)
        hidden_size = self.encoder.config.hidden_size  # 384 for MiniLM

        # Separate heads for mu and log_var
        self.mu_head = nn.Linear(hidden_size, 1)
        self.logvar_head = nn.Linear(hidden_size, 1)

    def forward(self, input_ids, attention_mask, labels=None):
        # Encode
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        token_embeddings = outputs.last_hidden_state

        # Mean pooling
        mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        pooled = torch.sum(token_embeddings * mask_expanded, 1) / torch.clamp(mask_expanded.sum(1), min=1e-9)

        # Predict mu and log_var
        mu = self.mu_head(pooled).squeeze(-1)
        log_var = self.logvar_head(pooled).squeeze(-1)

        # Clamp log_var to prevent degenerate variance
        log_var = torch.clamp(log_var, LOG_VAR_MIN, LOG_VAR_MAX)

        loss = None
        if labels is not None:
            # Gaussian NLL loss
            var = torch.exp(log_var)
            loss = 0.5 * (log_var + (labels - mu) ** 2 / var).mean()

        return {
            'loss': loss,
            'mu': mu,
            'log_var': log_var,
            'logits': torch.stack([mu, log_var], dim=-1)  # For Trainer compatibility
        }

    def predict_range(self, input_ids, attention_mask):
        """Inference: returns (mu, sigma, range_low, range_high)."""
        with torch.no_grad():
            out = self.forward(input_ids, attention_mask)
            mu = out['mu']
            sigma = torch.exp(0.5 * out['log_var'])
            range_low = torch.floor(mu - sigma)
            range_high = torch.ceil(mu + sigma)
        return mu, sigma, range_low, range_high


class C1Dataset(Dataset):
    """Dataset for heartbeat count regression.

    Expects format: {"problem_text": "...", "pulse_count": N}
    Also supports legacy format: {"text": "...", "heartbeat_count": N}
    """

    def __init__(self, examples, tokenizer, max_len=MAX_SEQ_LEN):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]

        # Support both new and legacy format
        text = ex.get('problem_text', ex.get('text', ''))
        pulse_count = ex.get('pulse_count', ex.get('heartbeat_count', 0))

        encoding = self.tokenizer(
            text,
            max_length=self.max_len,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(float(pulse_count), dtype=torch.float32)
        }


@dataclass
class C1DataCollator:
    def __call__(self, features):
        return {k: torch.stack([f[k] for f in features]) for k in features[0].keys()}


class C1Trainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        outputs = model(**inputs)
        return (outputs['loss'], outputs) if return_outputs else outputs['loss']

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        inputs = self._prepare_inputs(inputs)
        with torch.no_grad():
            outputs = model(**inputs)
        if prediction_loss_only:
            return (outputs['loss'], None, None)
        return (outputs['loss'], outputs['logits'], inputs['labels'])


def compute_metrics(eval_pred: EvalPrediction):
    """Compute range-aware metrics for C1."""
    logits = eval_pred.predictions  # (N, 2) = [mu, log_var]
    labels = eval_pred.label_ids

    mu = logits[:, 0]
    log_var = logits[:, 1]
    sigma = np.exp(0.5 * log_var)

    # Point estimate metrics
    errors = np.abs(mu - labels)
    mse = ((mu - labels) ** 2).mean()
    mae = errors.mean()

    # Range metrics
    range_low = np.floor(mu - sigma)
    range_high = np.ceil(mu + sigma)
    range_width = range_high - range_low

    # Range accuracy: target falls within [mu - sigma, mu + sigma]
    in_range = (labels >= range_low) & (labels <= range_high)
    range_accuracy = in_range.mean()

    # Tight range accuracy: width <= 2 AND target in range
    tight_and_correct = (range_width <= 2) & in_range
    tight_range_accuracy = tight_and_correct.mean()

    # Calibration: does sigma correlate with actual error?
    uncertainty_error_corr = np.corrcoef(sigma, errors)[0, 1]
    if np.isnan(uncertainty_error_corr):
        uncertainty_error_corr = 0.0

    # Prediction-target correlation (discrimination ability)
    pred_target_corr = np.corrcoef(mu, labels)[0, 1]
    if np.isnan(pred_target_corr):
        pred_target_corr = 0.0

    return {
        'mae': float(mae),
        'rmse': float(np.sqrt(mse)),
        'range_accuracy': float(range_accuracy),
        'tight_range_accuracy': float(tight_range_accuracy),
        'mean_sigma': float(sigma.mean()),
        'mean_range_width': float(range_width.mean()),
        'uncertainty_error_corr': float(uncertainty_error_corr),
        'pred_target_corr': float(pred_target_corr),
    }


if __name__ == "__main__":
    print(f"Downloading training data from {DATA_PATH}...")
    subprocess.run(['aws', 's3', 'cp', DATA_PATH, '/tmp/c1_train.json'], check=True)

    print("Loading data...")
    with open('/tmp/c1_train.json') as f:
        data = json.load(f)

    examples = data['examples']
    print(f"Loaded {len(examples)} examples")

    # Analyze target distribution
    pulse_counts = [ex.get('pulse_count', ex.get('heartbeat_count', 0)) for ex in examples]
    print(f"Pulse stats: min={min(pulse_counts)}, max={max(pulse_counts)}, "
          f"mean={np.mean(pulse_counts):.2f}, std={np.std(pulse_counts):.2f}")

    print(f"Loading tokenizer and model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = C1HeartbeatRegressor()
    model = model.cuda()
    print(f"Total params: {sum(p.numel() for p in model.parameters()):,}")

    # Train/eval split
    np.random.seed(42)
    indices = np.random.permutation(len(examples))
    train_size = int(0.9 * len(examples))
    train_examples = [examples[i] for i in indices[:train_size]]
    eval_examples = [examples[i] for i in indices[train_size:]]

    train_dataset = C1Dataset(train_examples, tokenizer)
    eval_dataset = C1Dataset(eval_examples, tokenizer)
    print(f"Train: {len(train_dataset)}, Eval: {len(eval_dataset)}")

    training_args = TrainingArguments(
        output_dir="./c1_checkpoints",
        num_train_epochs=10,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        warmup_steps=100,
        weight_decay=0.01,
        learning_rate=2e-5,
        logging_steps=50,
        eval_strategy="epoch",
        save_strategy="epoch",
        metric_for_best_model="range_accuracy",
        greater_is_better=True,
        load_best_model_at_end=True,
        report_to="none",
        fp16=True,
        dataloader_num_workers=4,
    )

    trainer = C1Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=C1DataCollator(),
        compute_metrics=compute_metrics,
    )

    print("Training C1 heartbeat regressor (Gaussian range)...")
    trainer.train()

    results = trainer.evaluate()
    print(f"\n{'='*50}")
    print("C1 HEARTBEAT REGRESSOR RESULTS")
    print('='*50)
    print(f"MAE: {results['eval_mae']:.2f} pulses")
    print(f"RMSE: {results['eval_rmse']:.2f}")
    print(f"Pred-Target Correlation: {results['eval_pred_target_corr']:.3f}")
    print()
    print("RANGE METRICS:")
    print(f"  Range Accuracy (target in ±1σ): {results['eval_range_accuracy']*100:.1f}%")
    print(f"  Tight Range Accuracy (width≤2 & correct): {results['eval_tight_range_accuracy']*100:.1f}%")
    print(f"  Mean σ: {results['eval_mean_sigma']:.2f}")
    print(f"  Mean Range Width: {results['eval_mean_range_width']:.1f}")
    print()
    print("CALIBRATION:")
    print(f"  Uncertainty-Error Correlation: {results['eval_uncertainty_error_corr']:.3f}")
    print(f"  (Higher = model knows when it's uncertain)")

    # Save model
    torch.save(model.state_dict(), "./c1_heartbeat_best.pt")
    print(f"\nModel saved to ./c1_heartbeat_best.pt")

    # Save to S3
    print("Uploading to S3...")
    subprocess.run(['aws', 's3', 'cp', './c1_heartbeat_best.pt',
                    's3://mycelium-data/models/c1_heartbeat/model.pt'], check=True)
    print("Done!")
