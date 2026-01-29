"""DSL Templates stub - minimal for cleanup."""

def infer_dsl_for_signature(step_type: str, step_text: str = None, **kwargs) -> tuple[str, str]:
    """Stub - return basic arithmetic DSL based on step_type.

    Returns:
        (dsl_script, dsl_type) tuple
    """
    op_map = {
        "add": ("a + b", "math"),
        "subtract": ("a - b", "math"),
        "multiply": ("a * b", "math"),
        "divide": ("a / b", "math"),
    }
    search_text = (step_type or "") + " " + (step_text or "")
    for key, (dsl, dsl_type) in op_map.items():
        if key in search_text.lower():
            return dsl, dsl_type
    return "a + b", "math"  # Default
