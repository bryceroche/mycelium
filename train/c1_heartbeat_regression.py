"""
C1: Heartbeat Counter - Predicts pulse count from problem text

Regression with learned uncertainty: predicts mean + variance per problem.
Output "4.1 ± 0.8" gives per-problem search budget directly from uncertainty.

Why regression over binned classification:
- Pulse count is ordinal (3 is between 2 and 4)
- MSE respects ordering; classification treats 2→3 error same as 2→7
- Continuous output useful as soft constraint for C2
"""
import json
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
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"  # 22M params, same family as C2
DATA_PATH = "s3://mycelium-data/c2_training/c2_train_with_heartbeats.json"


class C1HeartbeatRegressor(nn.Module):
    """Predicts heartbeat count from problem text."""

    def __init__(self, predict_variance=False):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(MODEL_NAME)
        hidden_size = self.encoder.config.hidden_size  # 384 for MiniLM
        self.predict_variance = predict_variance

        if predict_variance:
            # Predict mean and log_variance for uncertainty
            self.head = nn.Linear(hidden_size, 2)
        else:
            self.head = nn.Linear(hidden_size, 1)

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        # Mean pooling
        token_embeddings = outputs.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        pooled = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

        out = self.head(pooled)

        loss = None
        if labels is not None:
            if self.predict_variance:
                mean, log_var = out[:, 0], out[:, 1]
                # Gaussian NLL loss
                var = torch.exp(log_var)
                loss = 0.5 * (log_var + (labels - mean)**2 / var).mean()
                predictions = mean
            else:
                predictions = out.squeeze(-1)
                loss = nn.MSELoss()(predictions, labels)
        else:
            predictions = out[:, 0] if self.predict_variance else out.squeeze(-1)

        return {'loss': loss, 'predictions': predictions, 'logits': out}


class C1Dataset(Dataset):
    """Dataset for heartbeat count regression."""

    def __init__(self, examples, tokenizer, max_len=MAX_SEQ_LEN):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        encoding = self.tokenizer(
            ex['text'],
            max_length=self.max_len,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(float(ex['heartbeat_count']), dtype=torch.float32)
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
        # Return full logits (mean, log_var) for uncertainty analysis
        return (outputs['loss'], outputs['logits'], inputs['labels'])


def compute_metrics(eval_pred: EvalPrediction):
    logits = eval_pred.predictions
    labels = eval_pred.label_ids

    # Handle both variance and non-variance modes
    if logits.ndim == 2 and logits.shape[1] == 2:
        # Variance mode: logits = (mean, log_var)
        preds = logits[:, 0]
        log_var = logits[:, 1]
        std = np.exp(0.5 * log_var)  # predicted uncertainty
    else:
        preds = logits.squeeze()
        std = None

    # MSE and MAE
    mse = ((preds - labels) ** 2).mean()
    mae = np.abs(preds - labels).mean()
    errors = np.abs(preds - labels)

    # Accuracy within ±1 and ±2
    acc_1 = (errors <= 1).mean()
    acc_2 = (errors <= 2).mean()

    # Correlation of predictions with labels
    corr = np.corrcoef(preds, labels)[0, 1]

    metrics = {
        'mse': float(mse),
        'mae': float(mae),
        'rmse': float(np.sqrt(mse)),
        'acc_within_1': float(acc_1),
        'acc_within_2': float(acc_2),
        'correlation': float(corr)
    }

    # Uncertainty calibration metrics (if variance mode)
    if std is not None:
        # Does predicted uncertainty correlate with actual error?
        uncertainty_corr = np.corrcoef(std, errors)[0, 1]
        # What % of predictions fall within predicted ±1σ?
        within_1sigma = (errors <= std).mean()
        # Mean predicted std
        mean_std = std.mean()

        metrics.update({
            'mean_predicted_std': float(mean_std),
            'uncertainty_error_corr': float(uncertainty_corr),
            'within_1sigma': float(within_1sigma),  # Should be ~68% if calibrated
        })

    return metrics


if __name__ == "__main__":
    print(f"Downloading training data from {DATA_PATH}...")
    subprocess.run(['aws', 's3', 'cp', DATA_PATH, '/tmp/c1_train.json'], check=True)

    print("Loading data...")
    with open('/tmp/c1_train.json') as f:
        data = json.load(f)

    examples = data['examples']
    print(f"Loaded {len(examples)} examples")

    # Analyze target distribution
    heartbeats = [ex['heartbeat_count'] for ex in examples]
    print(f"Heartbeat stats: min={min(heartbeats)}, max={max(heartbeats)}, mean={np.mean(heartbeats):.2f}, std={np.std(heartbeats):.2f}")

    print(f"Loading tokenizer and model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = C1HeartbeatRegressor(predict_variance=True)
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
        metric_for_best_model="mae",
        greater_is_better=False,
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

    print("Training C1 heartbeat regressor...")
    trainer.train()

    results = trainer.evaluate()
    print(f"\n=== C1 RESULTS ===")
    print(f"MAE: {results['eval_mae']:.2f} heartbeats")
    print(f"RMSE: {results['eval_rmse']:.2f}")
    print(f"Acc within ±1: {results['eval_acc_within_1']*100:.1f}%")
    print(f"Acc within ±2: {results['eval_acc_within_2']*100:.1f}%")
    print(f"Correlation: {results['eval_correlation']:.3f}")
    if 'eval_mean_predicted_std' in results:
        print(f"\n=== UNCERTAINTY CALIBRATION ===")
        print(f"Mean predicted σ: {results['eval_mean_predicted_std']:.2f}")
        print(f"Uncertainty-error correlation: {results['eval_uncertainty_error_corr']:.3f}")
        print(f"Within ±1σ: {results['eval_within_1sigma']*100:.1f}% (ideal: 68%)")

    torch.save(model.state_dict(), "./c1_heartbeat_best.pt")
    print("Saved!")
