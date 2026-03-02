"""
Qwen 0.5B + Slot Embeddings for C3 v2
Uses rebuilt training data with PRIOR injection (93.5% coverage vs 24%)
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

MAX_OPERANDS = 4
MAX_SEQ_LEN = 512
MODEL_NAME = "Qwen/Qwen2.5-0.5B"

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


class C3QwenDatasetV2(Dataset):
    """Dataset using rebuilt data with PRIOR injection."""
    
    def __init__(self, path, tokenizer, max_len=MAX_SEQ_LEN):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.examples = []
        
        with open(path) as f:
            for line in f:
                ex = json.loads(line)
                
                # Use pre-built input_text with PRIORs injected
                input_text = ex.get('input_text', '')
                if not input_text:
                    continue
                
                for slot_idx, span in enumerate(ex.get('spans', [])[:MAX_OPERANDS]):
                    # Skip spans with invalid positions
                    if span.get('span_start', -1) < 0:
                        continue
                    
                    self.examples.append({
                        'input_text': input_text,
                        'span_start': span['span_start'],
                        'span_end': span['span_end'],
                        'span_text': span.get('span_text', ''),
                        'slot_id': slot_idx
                    })
        
        print(f"Loaded {len(self.examples)} training pairs")
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        ex = self.examples[idx]
        encoding = self.tokenizer(
            ex['input_text'],
            max_length=self.max_len, truncation=True,
            padding='max_length', return_offsets_mapping=True, return_tensors='pt'
        )
        
        offset_mapping = encoding['offset_mapping'][0].tolist()
        start_char = ex['span_start']
        end_char = ex['span_end']
        
        # Convert char positions to token positions
        start_token = 0
        end_token = 0
        for i, (s, e) in enumerate(offset_mapping):
            if s <= start_char < e:
                start_token = i
            if s < end_char <= e:
                end_token = i
                break
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'slot_ids': torch.tensor(ex['slot_id']),
            'start_positions': torch.tensor(start_token),
            'end_positions': torch.tensor(end_token)
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


print("Downloading rebuilt data with PRIOR injection...")
subprocess.run(['aws', 's3', 'cp', 's3://mycelium-data/c3_span_training/c3_train_with_priors.jsonl', '/tmp/c3_train.jsonl'], check=True)

print("Loading Qwen model with slot embeddings...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = QwenSpanExtractorWithSlots(num_slots=MAX_OPERANDS)
model = model.cuda()
print(f"Total params: {sum(p.numel() for p in model.parameters()):,}")

dataset = C3QwenDatasetV2('/tmp/c3_train.jsonl', tokenizer)
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

print("Training Qwen + slot embeddings v2 (with PRIOR injection)...")
trainer.train()

results = trainer.evaluate()
print(f"\n=== QWEN + SLOTS v2 RESULTS ===")
print(f"Exact Match: {results['eval_exact_match']*100:.1f}%")
print(f"Start Acc: {results['eval_start_accuracy']*100:.1f}%")
print(f"End Acc: {results['eval_end_accuracy']*100:.1f}%")

torch.save(model.state_dict(), "./qwen_slots_v2_best.pt")
print("Saved!")
