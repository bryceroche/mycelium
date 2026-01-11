"""PromptTemplate registry for centralized prompt management.

Provides:
- Template validation (ensures required fields are filled)
- Registry for easy lookup by name
- Consistent formatting across the codebase
"""

import re
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class PromptTemplate:
    """A prompt template with named placeholders.

    Placeholders use {name} syntax similar to Python's str.format().
    """
    name: str
    template: str
    description: str = ""
    required_fields: list[str] = field(default_factory=list)

    def __post_init__(self):
        # Auto-detect required fields from template if not provided
        if not self.required_fields:
            self.required_fields = self._extract_fields()

    def _extract_fields(self) -> list[str]:
        """Extract placeholder names from template."""
        # Match {field_name} but not {{escaped}} (doubled braces are literal in str.format)
        # Use negative lookbehind (?<!\{) and negative lookahead (?!\})
        pattern = r'(?<!\{)\{(\w+)\}(?!\})'
        return list(set(re.findall(pattern, self.template)))

    def format(self, **kwargs) -> str:
        """Format template with given values.

        Raises:
            ValueError: If required fields are missing
        """
        missing = set(self.required_fields) - set(kwargs.keys())
        if missing:
            raise ValueError(
                f"Template '{self.name}' missing required fields: {missing}"
            )
        return self.template.format(**kwargs)

    def format_safe(self, **kwargs) -> tuple[str, list[str]]:
        """Format template, returning missing fields instead of raising.

        Returns:
            Tuple of (formatted_string, list_of_missing_fields)
            If fields are missing, returns template with placeholders intact.
        """
        missing = [f for f in self.required_fields if f not in kwargs]
        if missing:
            # Partial format - only substitute available fields
            result = self.template
            for key, value in kwargs.items():
                result = result.replace(f"{{{key}}}", str(value))
            return result, missing
        return self.template.format(**kwargs), []


class PromptRegistry:
    """Registry of named prompt templates."""

    _instance: Optional["PromptRegistry"] = None

    def __init__(self):
        self._templates: dict[str, PromptTemplate] = {}

    @classmethod
    def get_instance(cls) -> "PromptRegistry":
        """Get singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
            cls._instance._register_defaults()
        return cls._instance

    def register(self, template: PromptTemplate) -> None:
        """Register a template."""
        self._templates[template.name] = template

    def get(self, name: str) -> PromptTemplate:
        """Get template by name.

        Raises:
            KeyError: If template not found
        """
        if name not in self._templates:
            raise KeyError(f"Template '{name}' not found. Available: {list(self._templates.keys())}")
        return self._templates[name]

    def format(self, name: str, **kwargs) -> str:
        """Convenience method to get and format a template."""
        return self.get(name).format(**kwargs)

    def list_templates(self) -> list[str]:
        """List all registered template names."""
        return list(self._templates.keys())

    def _register_defaults(self) -> None:
        """Register default templates used across the codebase."""
        # Step solver templates
        self.register(PromptTemplate(
            name="step_solver_with_method",
            template="""You are solving ONE step of a multi-step math problem.

**Context (original problem + previous results):**
{context}

**Your task for this step:**
{task}

**Recommended approach** (from similar problems):
{method_template}

Apply this approach, keeping the original problem in mind. Be precise.
Output your result as: RESULT: [your result for this step]
""",
            description="Solve a step with an injected method template",
        ))

        self.register(PromptTemplate(
            name="step_solver_with_superposition",
            template="""You are solving ONE step of a multi-step math problem.

**Context (original problem + previous results):**
{context}

**Your task for this step:**
{task}

**Multiple approaches available** (from similar problems, weighted by relevance):

{method_superposition}

Consider these approaches and choose the most appropriate for this specific step.
Keep the original problem in mind for context.
Be precise and show your work.
Output your result as: RESULT: [your result for this step]
""",
            description="Solve a step with multiple weighted method options",
        ))

        self.register(PromptTemplate(
            name="step_solver_plain",
            template="""You are solving ONE step of a multi-step math problem.

**Context (original problem + previous results):**
{context}

**Your task for this step:**
{task}

Solve ONLY this step, keeping the original problem in mind for context.
Be precise and show your work.
Output your result as: RESULT: [your result for this step]
""",
            description="Solve a step without method injection",
        ))

        self.register(PromptTemplate(
            name="step_solver_with_io_schema",
            template="""You are solving ONE step of a multi-step math problem.

**Available inputs** (from previous steps):
{formatted_inputs}

**Your task for this step:**
{task}

**Recommended approach** (from similar problems):
{method_template}

**Expected output format:** {output_format}

Apply the approach to solve this step using the inputs above.
Output your result as: RESULT: [your result in the expected format]
""",
            description="Solve a step with structured I/O schema",
        ))

        # Formula mode: Python-evaluable expression
        self.register(PromptTemplate(
            name="step_solver_with_formula",
            template="""You are solving ONE step of a multi-step math problem.

**Inputs:**
{formatted_inputs}

**Task:** {task}

**Execute this formula:**
```
{formula}
```

Substitute the input values and compute the result.
Output your result as: RESULT: [numeric value]
""",
            description="Solve a step by evaluating a formula",
        ))

        # Procedure mode: ordered steps to follow
        self.register(PromptTemplate(
            name="step_solver_with_procedure",
            template="""You are solving ONE step of a multi-step math problem.

**Inputs:**
{formatted_inputs}

**Task:** {task}

**Follow this procedure:**
{procedure_steps}

Execute each step in order.
Output your result as: RESULT: [your result]
""",
            description="Solve a step by following a procedure",
        ))

        # Guidance mode: natural language hint
        self.register(PromptTemplate(
            name="step_solver_with_guidance",
            template="""You are solving ONE step of a multi-step math problem.

**Inputs:**
{formatted_inputs}

**Task:** {task}

**Approach:** {guidance}

Apply this approach. Be precise.
Output your result as: RESULT: [your result]
""",
            description="Solve a step with natural language guidance",
        ))

        self.register(PromptTemplate(
            name="step_solver_plain_with_output_format",
            template="""You are solving ONE step of a multi-step math problem.

**Context (original problem + previous results):**
{context}

**Your task for this step:**
{task}

**Expected output format:** {output_format}

Solve ONLY this step, keeping the original problem in mind.
Be precise and show your work.
Output your result as: RESULT: [your result in the expected format]
""",
            description="Solve a step with output format hint only",
        ))

        # Synthesis templates
        self.register(PromptTemplate(
            name="final_synthesizer",
            template="""You are synthesizing the final answer from completed steps.

Problem: {problem}

Step results:
{step_results}

Combine these results to give the final answer.
Output as: ANSWER: [final answer]
""",
            description="Synthesize final answer from all step results",
        ))

        self.register(PromptTemplate(
            name="incremental_synthesizer",
            template="""You are merging intermediate results from dependent steps.

These steps have been completed:
{step_results}

Synthesize these into a single coherent intermediate result that captures all the information.
Be concise but preserve all important values and relationships.
Output as: MERGED: [merged result]
""",
            description="Merge intermediate results at synthesis points",
        ))

        # Direct solve template
        self.register(PromptTemplate(
            name="direct_solve",
            template="""You are a mathematical problem solver. Solve the problem step by step, showing your work clearly.

At the end, provide your final answer in the format:
ANSWER: [your answer]

Be precise with mathematical notation.""",
            description="Solve a problem directly without decomposition",
        ))

        # Tagger template (from solver.py)
        self.register(PromptTemplate(
            name="tagger",
            template="""You are a math problem tagger. Given a math problem, extract relevant tags that describe the problem's characteristics.

Tags should be loose and descriptive - think of keywords that would help find similar problems.
Include things like:
- Math concepts involved (e.g., "algebra", "derivative", "integral", "quadratic", "linear")
- Operations needed (e.g., "solve_for_x", "simplify", "factor", "compute")
- Problem features (e.g., "word_problem", "multi_step", "geometry", "probability")

Output 3-7 tags, one per line, lowercase with underscores:
TAGS:
tag1
tag2
tag3
...
""",
            description="Extract tags from a math problem",
        ))

        # Method injection template (from solver.py)
        self.register(PromptTemplate(
            name="method_injection",
            template="""You are a mathematical problem solver.

## Suggested Approach
Based on similar problems, this approach has worked well:

**{method_name}**
{method_template}

## Instructions
1. Consider if this approach fits the given problem
2. Adapt it as needed for the specific details
3. Show your work step by step
4. Provide your final answer as: ANSWER: [your answer]
""",
            description="Inject a method template for problem solving",
        ))

        # Generic solver template
        self.register(PromptTemplate(
            name="solver",
            template="""You are a mathematical problem solver. Solve the problem step by step, showing your work clearly.

At the end, provide your final answer in the format:
ANSWER: [your answer]

Be precise with mathematical notation.""",
            description="Generic math problem solver",
        ))

        # ============================================================
        # JSON OUTPUT TEMPLATES - For structured output mode
        # ============================================================

        self.register(PromptTemplate(
            name="step_solver_json",
            template="""You are solving ONE step of a multi-step math problem.

**Context (original problem + previous results):**
{context}

**Your task for this step:**
{task}

Solve ONLY this step. Output your response as JSON:
{{"reasoning": "your step-by-step reasoning", "result": <the numeric/symbolic result>}}

IMPORTANT:
- "result" should be the direct value (number, expression, or equation)
- Use a number for numeric results: {{"reasoning": "...", "result": 42}}
- Use a string for expressions: {{"reasoning": "...", "result": "x^2 - 4"}}
""",
            description="Solve a step with JSON output",
        ))

        self.register(PromptTemplate(
            name="step_solver_json_with_guidance",
            template="""You are solving ONE step of a multi-step math problem.

**Inputs:**
{formatted_inputs}

**Task:** {task}

**Approach:** {guidance}

Apply this approach. Output your response as JSON:
{{"reasoning": "your step-by-step reasoning", "result": <the numeric/symbolic result>}}

IMPORTANT:
- "result" should be the direct value (number, expression, or equation)
- Use a number for numeric results: {{"reasoning": "...", "result": 42}}
- Use a string for expressions: {{"reasoning": "...", "result": "x^2 - 4"}}
""",
            description="Solve a step with guidance and JSON output",
        ))

        self.register(PromptTemplate(
            name="step_solver_json_with_procedure",
            template="""You are solving ONE step of a multi-step math problem.

**Inputs:**
{formatted_inputs}

**Task:** {task}

**Follow this procedure:**
{procedure_steps}

Execute each step in order. Output your response as JSON:
{{"reasoning": "your step-by-step reasoning", "result": <the numeric/symbolic result>}}

IMPORTANT:
- "result" should be the direct value (number, expression, or equation)
- Use a number for numeric results: {{"reasoning": "...", "result": 42}}
- Use a string for expressions: {{"reasoning": "...", "result": "x^2 - 4"}}
""",
            description="Solve a step with procedure and JSON output",
        ))

        self.register(PromptTemplate(
            name="final_synthesizer_json",
            template="""You are synthesizing the final answer from completed steps.

Problem: {problem}

Step results:
{step_results}

Combine these results to give the final answer.
Output as JSON: {{"reasoning": "how you combined the results", "answer": <final answer>}}

Use a number for numeric answers, string otherwise.
""",
            description="Synthesize final answer with JSON output",
        ))


# Convenience function to get the global registry
def get_registry() -> PromptRegistry:
    """Get the global prompt registry."""
    return PromptRegistry.get_instance()


# Convenience function to format a template
def format_prompt(name: str, **kwargs) -> str:
    """Format a registered template by name."""
    return get_registry().format(name, **kwargs)
