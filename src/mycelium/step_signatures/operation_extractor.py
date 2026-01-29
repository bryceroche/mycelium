"""Operation Extraction from Problem Text.

Per CLAUDE.md: Route by what operations DO, not what they SOUND LIKE.

This module provides the prompt template for operation extraction.
Actual embedding is done through embedding_cache.py (per "New Favorite Pattern").

Flow:
    Problem text → LLM extracts operation → cached_embed_batch() → Compare to graph embeddings

Example:
    "Calculate 25% of 200" → "multiply percentage by base then divide by 100"
    → embed → matches MUL(DIV(param_0, CONST(100)), param_1)

Note: Async embedding functions were removed per mycelium-ocal as they bypassed
the centralized embedding cache. Use cached_embed() / cached_embed_batch() from
embedding_cache.py for all embedding needs.
"""

import logging

logger = logging.getLogger(__name__)

# Prompt for operation extraction
OPERATION_EXTRACTION_PROMPT = '''You are an operation extractor for a math problem solver.

Given a math problem or step, extract WHAT COMPUTATION is needed - not the specific numbers or context.

## Rules

1. Focus on the OPERATION TYPE, not the specific values
2. Use generic terms: "multiply", "divide", "add", "subtract", "raise to power", "find GCD", etc.
3. Describe the computation structure: "multiply two numbers then add a third"
4. Be concise - just the operation, not a full explanation
5. If multiple operations, describe the sequence: "divide A by B, then multiply by C"

## Examples

Problem: "Calculate 25% of 200"
Operation: "divide percentage by 100, multiply by base value"

Problem: "Find 2^8"
Operation: "raise base to exponent power"

Problem: "What is the GCD of 24 and 36?"
Operation: "find greatest common divisor of two numbers"

Problem: "If a train travels 60 mph for 3 hours, how far does it go?"
Operation: "multiply rate by time"

Problem: "A rectangle has length 5 and width 3. What is its area?"
Operation: "multiply length by width"

Problem: "What is 15% more than 200?"
Operation: "multiply base by percentage divided by 100, add to base"

Problem: "Simplify (3x + 2) - (x - 4)"
Operation: "subtract polynomials by distributing negative and combining like terms"

Now extract the operation for this problem:

Problem: {problem_text}
Operation:'''
