"""LLM API utilities for calling OpenAI and Anthropic."""

import os


def call_openai(prompt: str, model: str = "gpt-4o") -> str:
    """Call OpenAI API.

    Args:
        prompt: The prompt to send
        model: Model name (default: gpt-4o)

    Returns:
        Response text
    """
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("pip install openai")

    client = OpenAI()  # Uses OPENAI_API_KEY env var

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You output valid JSON only. No markdown, no explanation."},
            {"role": "user", "content": prompt}
        ],
        response_format={"type": "json_object"},
        temperature=0.1,
    )

    return response.choices[0].message.content


def call_anthropic(prompt: str, model: str = "claude-sonnet-4-20250514") -> str:
    """Call Anthropic API.

    Args:
        prompt: The prompt to send
        model: Model name (default: claude-sonnet-4-20250514)

    Returns:
        Response text
    """
    try:
        from anthropic import Anthropic
    except ImportError:
        raise ImportError("pip install anthropic")

    client = Anthropic()  # Uses ANTHROPIC_API_KEY env var

    response = client.messages.create(
        model=model,
        max_tokens=1024,
        messages=[
            {"role": "user", "content": prompt + "\n\nRespond with valid JSON only, no markdown code blocks."}
        ],
        temperature=0.1,
    )

    return response.content[0].text
