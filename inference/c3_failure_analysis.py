"""
C3 Failure Analysis: Understand why 45% of span predictions fail

Error categories:
1. Off-by-one (start or end)
2. Wrong span but overlapping
3. Completely wrong span
4. Per-slot breakdown
5. Error vs span length correlation
"""
import torch
import torch.nn as nn
import numpy as np
from collections import defaultdict
from transformers import AutoModel, AutoTokenizer
import subprocess

MAX_OPERANDS = 4
MODEL_NAME = "Qwen/Qwen2.5-0.5B"
CHECKPOINT_PATH = "s3://mycelium-data/c3_checkpoints/qwen_slots_v2/checkpoint-1950/"
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

    def forward(self, input_ids, attention_mask, slot_ids=None):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        if slot_ids is not None:
            slot_emb = self.slot_embeddings(slot_ids).unsqueeze(1)
            sequence_output = sequence_output + slot_emb
        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        return start_logits.squeeze(-1), end_logits.squeeze(-1)


def categorize_error(pred_start, pred_end, true_start, true_end):
    """Categorize the type of error."""
    if pred_start == true_start and pred_end == true_end:
        return "correct"

    # Off-by-one errors
    start_off = abs(pred_start - true_start)
    end_off = abs(pred_end - true_end)

    if start_off <= 1 and end_off <= 1 and (start_off + end_off) > 0:
        if start_off == 0:
            return "off_by_one_end"
        elif end_off == 0:
            return "off_by_one_start"
        else:
            return "off_by_one_both"

    # Check overlap
    pred_range = set(range(pred_start, pred_end + 1))
    true_range = set(range(true_start, true_end + 1))
    overlap = pred_range & true_range

    if overlap:
        overlap_ratio = len(overlap) / len(true_range)
        if overlap_ratio >= 0.5:
            return "partial_overlap_good"
        else:
            return "partial_overlap_bad"

    return "completely_wrong"


def analyze_failures(model, data_dict, tokenizer, device, num_samples=None):
    """Run predictions and analyze failures."""
    model.eval()

    input_ids = data_dict['input_ids']
    attention_mask = data_dict['attention_mask']
    slot_ids = data_dict['slot_ids']
    start_positions = data_dict['start_positions']
    end_positions = data_dict['end_positions']

    # Use last 10% as eval (matching training split)
    n = len(input_ids)
    eval_start = int(0.9 * n)

    input_ids = input_ids[eval_start:]
    attention_mask = attention_mask[eval_start:]
    slot_ids = slot_ids[eval_start:]
    start_positions = start_positions[eval_start:]
    end_positions = end_positions[eval_start:]

    if num_samples:
        input_ids = input_ids[:num_samples]
        attention_mask = attention_mask[:num_samples]
        slot_ids = slot_ids[:num_samples]
        start_positions = start_positions[:num_samples]
        end_positions = end_positions[:num_samples]

    print(f"Analyzing {len(input_ids)} eval samples...")

    # Track errors
    error_counts = defaultdict(int)
    slot_errors = defaultdict(lambda: defaultdict(int))
    span_length_errors = defaultdict(list)
    examples = defaultdict(list)

    batch_size = 32
    all_results = []

    with torch.no_grad():
        for i in range(0, len(input_ids), batch_size):
            batch_input = input_ids[i:i+batch_size].to(device)
            batch_mask = attention_mask[i:i+batch_size].to(device)
            batch_slots = slot_ids[i:i+batch_size].to(device)

            start_logits, end_logits = model(batch_input, batch_mask, batch_slots)

            pred_starts = start_logits.argmax(dim=-1).cpu()
            pred_ends = end_logits.argmax(dim=-1).cpu()

            for j in range(len(batch_input)):
                idx = i + j
                ps = pred_starts[j].item()
                pe = pred_ends[j].item()
                ts = start_positions[idx].item()
                te = end_positions[idx].item()
                slot = slot_ids[idx].item()

                error_type = categorize_error(ps, pe, ts, te)
                error_counts[error_type] += 1
                slot_errors[slot][error_type] += 1

                span_length = te - ts + 1
                span_length_errors[span_length].append(error_type)

                all_results.append({
                    'idx': idx,
                    'pred_start': ps,
                    'pred_end': pe,
                    'true_start': ts,
                    'true_end': te,
                    'slot': slot,
                    'error_type': error_type,
                    'input_ids': input_ids[idx],
                    'span_length': span_length
                })

                # Collect examples (up to 3 per error type)
                if error_type != "correct" and len(examples[error_type]) < 3:
                    examples[error_type].append({
                        'input_ids': input_ids[idx],
                        'pred': (ps, pe),
                        'true': (ts, te),
                        'slot': slot
                    })

    return error_counts, slot_errors, span_length_errors, examples, all_results


def print_analysis(error_counts, slot_errors, span_length_errors, examples, tokenizer):
    """Print analysis results."""
    total = sum(error_counts.values())

    print("\n" + "="*60)
    print("C3 FAILURE ANALYSIS")
    print("="*60)

    # Overall error breakdown
    print("\n### ERROR TYPE BREAKDOWN ###")
    for error_type in ["correct", "off_by_one_start", "off_by_one_end", "off_by_one_both",
                       "partial_overlap_good", "partial_overlap_bad", "completely_wrong"]:
        count = error_counts.get(error_type, 0)
        pct = 100 * count / total if total > 0 else 0
        print(f"  {error_type:25s}: {count:5d} ({pct:5.1f}%)")

    # Recoverable errors (off-by-one + good overlap)
    recoverable = (error_counts.get("off_by_one_start", 0) +
                   error_counts.get("off_by_one_end", 0) +
                   error_counts.get("off_by_one_both", 0) +
                   error_counts.get("partial_overlap_good", 0))
    print(f"\n  Recoverable (off-by-one + good overlap): {recoverable} ({100*recoverable/total:.1f}%)")

    # Per-slot breakdown
    print("\n### PER-SLOT ACCURACY ###")
    for slot in sorted(slot_errors.keys()):
        slot_total = sum(slot_errors[slot].values())
        slot_correct = slot_errors[slot].get("correct", 0)
        slot_acc = 100 * slot_correct / slot_total if slot_total > 0 else 0
        print(f"  Slot {slot}: {slot_correct}/{slot_total} ({slot_acc:.1f}%)")

    # Span length analysis
    print("\n### ERROR RATE BY SPAN LENGTH ###")
    length_stats = []
    for length in sorted(span_length_errors.keys()):
        errors = span_length_errors[length]
        total_len = len(errors)
        correct = sum(1 for e in errors if e == "correct")
        if total_len >= 10:  # Only show lengths with enough samples
            length_stats.append((length, total_len, correct))

    for length, total_len, correct in length_stats[:15]:
        acc = 100 * correct / total_len
        bar = "█" * int(acc / 5)
        print(f"  Length {length:2d}: {correct:4d}/{total_len:4d} ({acc:5.1f}%) {bar}")

    # Show examples
    print("\n### ERROR EXAMPLES ###")
    for error_type in ["off_by_one_start", "off_by_one_end", "completely_wrong"]:
        if error_type in examples and examples[error_type]:
            print(f"\n--- {error_type} ---")
            for ex in examples[error_type][:2]:
                tokens = tokenizer.convert_ids_to_tokens(ex['input_ids'].tolist())
                pred_span = tokens[ex['pred'][0]:ex['pred'][1]+1]
                true_span = tokens[ex['true'][0]:ex['true'][1]+1]
                print(f"  Slot {ex['slot']}")
                print(f"  Predicted: {' '.join(pred_span[:20])}{'...' if len(pred_span) > 20 else ''}")
                print(f"  Actual:    {' '.join(true_span[:20])}{'...' if len(true_span) > 20 else ''}")
                print()


def analyze_prediction_distribution(all_results):
    """Check if model is collapsing to certain positions."""
    print("\n### PREDICTION DISTRIBUTION ###")

    pred_starts = [r['pred_start'] for r in all_results]
    pred_ends = [r['pred_end'] for r in all_results]
    true_starts = [r['true_start'] for r in all_results]
    true_ends = [r['true_end'] for r in all_results]

    print(f"  Pred start: mean={np.mean(pred_starts):.1f}, std={np.std(pred_starts):.1f}, "
          f"min={min(pred_starts)}, max={max(pred_starts)}")
    print(f"  True start: mean={np.mean(true_starts):.1f}, std={np.std(true_starts):.1f}, "
          f"min={min(true_starts)}, max={max(true_starts)}")
    print(f"  Pred end:   mean={np.mean(pred_ends):.1f}, std={np.std(pred_ends):.1f}, "
          f"min={min(pred_ends)}, max={max(pred_ends)}")
    print(f"  True end:   mean={np.mean(true_ends):.1f}, std={np.std(true_ends):.1f}, "
          f"min={min(true_ends)}, max={max(true_ends)}")

    # Check for position collapse
    from collections import Counter
    start_counter = Counter(pred_starts)
    end_counter = Counter(pred_ends)

    print(f"\n  Most common pred starts: {start_counter.most_common(5)}")
    print(f"  Most common pred ends:   {end_counter.most_common(5)}")

    # Check start > end (invalid spans)
    invalid = sum(1 for r in all_results if r['pred_start'] > r['pred_end'])
    print(f"\n  Invalid spans (start > end): {invalid} ({100*invalid/len(all_results):.1f}%)")


if __name__ == "__main__":
    print("Downloading checkpoint...")
    subprocess.run(['aws', 's3', 'cp', '--recursive', CHECKPOINT_PATH, '/tmp/checkpoint/'], check=True)

    print("Downloading data...")
    subprocess.run(['aws', 's3', 'cp', DATA_PATH, '/tmp/c3_data.pt'], check=True)

    print("Loading model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = QwenSpanExtractorWithSlots(num_slots=MAX_OPERANDS)

    # Load checkpoint
    checkpoint = torch.load('/tmp/checkpoint/pytorch_model.bin', map_location='cpu')
    model.load_state_dict(checkpoint)
    model = model.to(device)
    print(f"Model loaded. Params: {sum(p.numel() for p in model.parameters()):,}")

    print("Loading data...")
    data_dict = torch.load('/tmp/c3_data.pt')
    print(f"Data loaded: {len(data_dict['input_ids'])} samples")

    # Run analysis
    error_counts, slot_errors, span_length_errors, examples, all_results = analyze_failures(
        model, data_dict, tokenizer, device
    )

    print_analysis(error_counts, slot_errors, span_length_errors, examples, tokenizer)
    analyze_prediction_distribution(all_results)

    print("\nDone!")
