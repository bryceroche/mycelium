"""
Qwen 0.5B + Slot Embeddings for C3 v2
Uses pre-tokenized training data from RoBERTa-large format
"""
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

MAX_OPERANDS = 4
MAX_SEQ_LEN = 512
MODEL_NAME = "Qwen/Qwen2.5-0.5B"
DATA_PATH = "s3://mycelium-data/c3_span_training/c3_train_roberta_large.pt"

class QwenSpanExtractorWithSlots(nn.Module):
    def __init__(self, num_slots=MAX_OPERANDS):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(
            MODEL_NAME, 
            torch_dtype=torch.float32,
            trust_remote_code=True
        )
        hidden_size = self.encoder.config.hidden_size
        self.slot_embeddings = nn.Embedding(num_slots, hidden_size)
        self.qa_outputs = nn.Linear(hidden_size, 2)
        
    def forward(self, input_ids, attention_mask, slot_ids=None, start_positions=None, end_positions=None):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        if slot_ids is not None:
            slot_emb = self.slot_embeddings(slot_ids).unsqueeze(1)
            sequence_output = sequence_output + slot_emb
        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)
        
        loss = None
        if start_positions is not None and end_positions is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = (loss_fct(start_logits, start_positions) + loss_fct(end_logits, end_positions)) / 2
        
        return {'loss': loss, 'start_logits': start_logits, 'end_logits': end_logits}


class C3PreTokenizedDataset(Dataset):
    """Dataset using pre-tokenized .pt data."""

    def __init__(self, data_dict):
        self.input_ids = data_dict['input_ids']
        self.attention_mask = data_dict['attention_mask']
        self.slot_ids = data_dict['slot_ids']
        self.start_positions = data_dict['start_positions']
        self.end_positions = data_dict['end_positions']
        print(f"Loaded {len(self.input_ids)} training pairs")

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {
            'input_ids': self.input_ids[idx],
            'attention_mask': self.attention_mask[idx],
            'slot_ids': self.slot_ids[idx],
            'start_positions': self.start_positions[idx],
            'end_positions': self.end_positions[idx]
        }


@dataclass
class DataCollatorWithSlots:
    def __call__(self, features):
        return {k: torch.stack([f[k] for f in features]) for k in features[0].keys()}


class SlotQATrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        outputs = model(**inputs)
        return (outputs['loss'], outputs) if return_outputs else outputs['loss']
    
    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        inputs = self._prepare_inputs(inputs)
        with torch.no_grad():
            outputs = model(**inputs)
        if prediction_loss_only:
            return (outputs['loss'], None, None)
        return (outputs['loss'], (outputs['start_logits'], outputs['end_logits']), 
                (inputs['start_positions'], inputs['end_positions']))


def compute_metrics(eval_pred: EvalPrediction):
    start_preds = np.argmax(eval_pred.predictions[0], axis=-1)
    end_preds = np.argmax(eval_pred.predictions[1], axis=-1)
    start_labels, end_labels = eval_pred.label_ids
    
    exact_match = ((start_preds == start_labels) & (end_preds == end_labels)).mean()
    return {
        'exact_match': float(exact_match),
        'start_accuracy': float((start_preds == start_labels).mean()),
        'end_accuracy': float((end_preds == end_labels).mean())
    }


if __name__ == "__main__":
    print(f"Downloading pre-tokenized data from {DATA_PATH}...")
    subprocess.run(['aws', 's3', 'cp', DATA_PATH, '/tmp/c3_train.pt'], check=True)

    print("Loading pre-tokenized data...")
    data_dict = torch.load('/tmp/c3_train.pt')

    print("Loading Qwen model with slot embeddings...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = QwenSpanExtractorWithSlots(num_slots=MAX_OPERANDS)
    model = model.cuda()
    print(f"Total params: {sum(p.numel() for p in model.parameters()):,}")

    dataset = C3PreTokenizedDataset(data_dict)
    train_size = int(0.9 * len(dataset))
    train_dataset, eval_dataset = torch.utils.data.random_split(dataset, [train_size, len(dataset) - train_size])
    print(f"Train: {len(train_dataset)}, Eval: {len(eval_dataset)}")

    training_args = TrainingArguments(
        output_dir="./qwen_slots_v2_checkpoints",
        num_train_epochs=10,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        warmup_steps=500,
        weight_decay=0.01,
        learning_rate=2e-5,
        logging_steps=100,
        eval_strategy="epoch",
        save_strategy="epoch",
        metric_for_best_model="exact_match",
        greater_is_better=True,
        load_best_model_at_end=True,
        report_to="none",
        fp16=False,
        dataloader_num_workers=4,
    )

    trainer = SlotQATrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=DataCollatorWithSlots(),
        compute_metrics=compute_metrics,
    )

    print("Training Qwen + slot embeddings v2...")
    trainer.train()

    results = trainer.evaluate()
    print(f"\n=== QWEN + SLOTS v2 RESULTS ===")
    print(f"Exact Match: {results['eval_exact_match']*100:.1f}%")
    print(f"Start Acc: {results['eval_start_accuracy']*100:.1f}%")
    print(f"End Acc: {results['eval_end_accuracy']*100:.1f}%")

    torch.save(model.state_dict(), "./qwen_slots_v2_best.pt")
    print("Saved!")
