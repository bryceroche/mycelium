# Lambda Handoff: Parallel MATH IAF + IB Template Discovery

## Immediate Actions

### Action 1: Kill GSM8K IAF extraction
Stop the current run. We have enough GSM8K data (8,986 C3 training pairs). Switch all resources to MATH.

### Action 2: Run IB on existing 1,516 MATH IAF files (NOW)
While the VMs spin up, run Information Bottleneck template discovery on the 1,516 MATH IAF files we already have. This gives us MATH operation templates within ~30 minutes.

IB setup:
- Input X: CoT span embeddings from the 1,516 IAF files
- Target Y: Execution result type (operation category from LaTeX parser)
- Anneal β from low (hot) to high (cold)
- Track template count at each β — look for plateaus
- Expected: 50-100 MATH templates (vs 86 GSM8K templates)
- Report: template list, frequency distribution, example spans per template

### Action 3: Spin up 3 additional VMs for parallel MATH IAF extraction

**VM allocation:**
```
VM 1 (existing): Problems 0-1,960      (~1,961 problems)
VM 2 (new):      Problems 1,961-3,921   (~1,961 problems)
VM 3 (new):      Problems 3,922-5,882   (~1,961 problems)
VM 4 (new):      Problems 5,883-7,842   (~1,960 problems)
```

**Each VM needs:**
- GPU: T4 or A10 (smallest that fits 7B with attention hooks)
- Model: Qwen2.5-Math-7B-Instruct
- Data: The 7B CoT outputs (already generated — just distribute the indices)
- Script: IAF extraction script (same as current, just different index range)
- Output: Save IAF files to local disk, rsync to main VM when done

**Script per VM:**
```bash
python extract_iaf.py \
  --cot_data /path/to/7b_cot_results.json \
  --start_idx $START \
  --end_idx $END \
  --output_dir /home/ubuntu/math_iaf/ \
  --model Qwen/Qwen2.5-Math-7B-Instruct
```

**Estimated time:** ~3-4 hours per VM (7,842 / 4 = ~1,960 problems each, vs 19 hours for all 7,842 on one VM)

### Action 4: When extraction finishes (~3-4 hours)

1. **Merge IAF files** from all 4 VMs into one directory
2. **Run IB again** on the full 7,842 MATH IAF dataset
3. **Compare templates** to the preliminary 1,516-sample IB
4. **Build C3 training data** from all MATH IAF pairs
5. **Retrain C3** on combined GSM8K + MATH data
6. **Run Track 2 E2E eval**

### Action 5: Shut down extra VMs
Once extraction is complete and data is merged, terminate the 3 extra VMs. Only keep the main VM for training and eval.

## VM Setup Script (for each new VM)

```bash
#!/bin/bash
# Quick setup for MATH IAF extraction VM

# Install dependencies
pip install torch transformers scipy numpy --break-system-packages

# Download model (will cache)
python -c "from transformers import AutoModelForCausalLM, AutoTokenizer; \
  AutoTokenizer.from_pretrained('Qwen/Qwen2.5-Math-7B-Instruct'); \
  AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-Math-7B-Instruct', \
    torch_dtype='auto', device_map='auto')"

# Copy CoT data and extraction script from main VM
scp ubuntu@MAIN_VM_IP:/home/ubuntu/7b_cot_results.json /home/ubuntu/
scp ubuntu@MAIN_VM_IP:/home/ubuntu/extract_iaf.py /home/ubuntu/

# Run extraction with assigned index range
python extract_iaf.py \
  --start_idx $START_IDX \
  --end_idx $END_IDX \
  --output_dir /home/ubuntu/math_iaf/
```

## Priority Order
1. Kill GSM8K IAF (immediate)
2. Start IB on 1,516 MATH IAF (immediate, runs on existing VM)
3. Spin up VMs and start MATH IAF extraction (ASAP)
4. C3 v6 training on GSM8K data can continue in parallel
5. When MATH IAF finishes: IB on full data, rebuild C3 training, retrain, eval

## What We'll Have by End of Day
- MATH operation template taxonomy from IB (50-100 templates)
- Full MATH IAF data (7,842 files)
- C3 trained on GSM8K + MATH combined data
- Track 2 E2E number with MATH-native templates and full training data
