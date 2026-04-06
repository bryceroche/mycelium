from datasets import load_dataset
ds = load_dataset("openai/gsm8k", "main", split="train[:5]")
for i, ex in enumerate(ds):
    q = ex["question"]
    a = ex["answer"]
    print(f"=== Example {i} ===")
    print(f"QUESTION: {q}")
    print(f"ANSWER:\n{a}")
    print()
