"""
Specialized DSL Templates - Generated from Qwen-7B Attention Analysis

Each template represents a unique semantic pattern discovered from GSM8K + MATH.
Templates are organized by operation type and include:
- Generic pattern (with placeholders)
- DSL expression template
- Parameter extraction hints
- Attention-based importance weights
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Callable
from enum import Enum

class Operation(Enum):
    SET = "SET"      # Direct value assignment
    ADD = "ADD"      # Accumulation
    SUB = "SUB"      # Reduction
    MUL = "MUL"      # Multiplication/scaling
    DIV = "DIV"      # Division/splitting


@dataclass
class DSLTemplate:
    """A specialized DSL template for a semantic pattern."""
    template_id: str
    operation: Operation
    pattern: str                    # Generic pattern with [N], [NAME], [ITEM], etc.
    dsl_expr: str                   # DSL expression template
    param_slots: List[str]          # Parameters to extract: ["value", "entity", "rate"]
    verb_triggers: List[str]        # Verbs that activate this template
    examples: List[str] = field(default_factory=list)
    attention_weight: float = 1.0   # Importance from Qwen attention

    def to_dict(self) -> Dict:
        return {
            "template_id": self.template_id,
            "operation": self.operation.value,
            "pattern": self.pattern,
            "dsl_expr": self.dsl_expr,
            "param_slots": self.param_slots,
            "verb_triggers": self.verb_triggers,
            "examples": self.examples,
            "attention_weight": self.attention_weight,
        }


# =============================================================================
# SET Operations - Direct Value Assignment
# =============================================================================

SET_TEMPLATES = [
    DSLTemplate(
        template_id="set_quantity",
        operation=Operation.SET,
        pattern="[ENTITY] has [N] [ITEM]",
        dsl_expr="quantity = value",
        param_slots=["value"],
        verb_triggers=["has", "have", "had", "having"],
        examples=["A large pizza has 16 slices", "She has $16 left"],
        attention_weight=1.0,
    ),
    DSLTemplate(
        template_id="set_age",
        operation=Operation.SET,
        pattern="[ENTITY] is [N] years old",
        dsl_expr="age = value",
        param_slots=["value"],
        verb_triggers=["is", "was", "are", "were"],
        examples=["Tom is 25 years old", "She was 30 years old"],
        attention_weight=0.95,
    ),
    DSLTemplate(
        template_id="set_price",
        operation=Operation.SET,
        pattern="[ITEM] costs [N]",
        dsl_expr="price = value",
        param_slots=["value"],
        verb_triggers=["costs", "cost", "priced"],
        examples=["A concert ticket costs $50", "The shirt costs $25"],
        attention_weight=0.98,
    ),
    DSLTemplate(
        template_id="set_rate",
        operation=Operation.SET,
        pattern="[ENTITY] earns [N] per [UNIT]",
        dsl_expr="rate = value",
        param_slots=["value", "unit"],
        verb_triggers=["earns", "makes", "pays", "charges"],
        examples=["Weng earns $12 an hour", "She makes $15 per hour"],
        attention_weight=0.96,
    ),
    DSLTemplate(
        template_id="set_duration",
        operation=Operation.SET,
        pattern="[ENTITY] [ACTION] for [N] [TIME_UNIT]",
        dsl_expr="duration = value",
        param_slots=["value", "time_unit"],
        verb_triggers=["did", "worked", "ran", "walked", "studied"],
        examples=["She did 50 minutes of babysitting", "He worked for 8 hours"],
        attention_weight=0.92,
    ),
    DSLTemplate(
        template_id="set_distance",
        operation=Operation.SET,
        pattern="[ENTITY] [MOTION] [N] [DISTANCE_UNIT]",
        dsl_expr="distance = value",
        param_slots=["value", "distance_unit"],
        verb_triggers=["runs", "walks", "drives", "travels"],
        examples=["He runs 5 miles", "She drives 30 km"],
        attention_weight=0.90,
    ),
    DSLTemplate(
        template_id="set_target",
        operation=Operation.SET,
        pattern="[ENTITY] needs [N] [ITEM]",
        dsl_expr="target = value",
        param_slots=["value"],
        verb_triggers=["needs", "wants", "requires"],
        examples=["She needs to gain 10 pounds", "He wants 5 more"],
        attention_weight=0.88,
    ),
    DSLTemplate(
        template_id="set_capacity",
        operation=Operation.SET,
        pattern="[CONTAINER] holds [N] [ITEM]",
        dsl_expr="capacity = value",
        param_slots=["value"],
        verb_triggers=["holds", "contains", "fits"],
        examples=["The tank holds 50 gallons", "The box contains 24 items"],
        attention_weight=0.85,
    ),
    DSLTemplate(
        template_id="set_count",
        operation=Operation.SET,
        pattern="there are [N] [ITEM]",
        dsl_expr="count = value",
        param_slots=["value"],
        verb_triggers=["are", "were", "is"],
        examples=["There are 25 students", "There were 10 apples"],
        attention_weight=0.87,
    ),
    DSLTemplate(
        template_id="set_percentage",
        operation=Operation.SET,
        pattern="[N]% of [ENTITY]",
        dsl_expr="percentage = value / 100",
        param_slots=["value"],
        verb_triggers=["percent", "%"],
        examples=["25% of the students", "80% of the flowers"],
        attention_weight=0.93,
    ),
]


# =============================================================================
# ADD Operations - Accumulation/Increase
# =============================================================================

ADD_TEMPLATES = [
    DSLTemplate(
        template_id="add_purchase",
        operation=Operation.ADD,
        pattern="[ENTITY] bought [N] [ITEM]",
        dsl_expr="total = entity + value",
        param_slots=["entity", "value"],
        verb_triggers=["bought", "purchased", "acquired"],
        examples=["Bella bought 3 snowflakes", "He purchased 5 shirts"],
        attention_weight=0.97,
    ),
    DSLTemplate(
        template_id="add_collection",
        operation=Operation.ADD,
        pattern="[ENTITY] collected [N] [ITEM]",
        dsl_expr="total = entity + value",
        param_slots=["entity", "value"],
        verb_triggers=["collected", "gathered", "picked"],
        examples=["They collected 50 bananas", "She gathered 12 eggs"],
        attention_weight=0.94,
    ),
    DSLTemplate(
        template_id="add_earning",
        operation=Operation.ADD,
        pattern="[ENTITY] earned [N]",
        dsl_expr="total = entity + value",
        param_slots=["entity", "value"],
        verb_triggers=["earned", "made", "received"],
        examples=["Leah earned $40", "He received $100"],
        attention_weight=0.95,
    ),
    DSLTemplate(
        template_id="add_discovery",
        operation=Operation.ADD,
        pattern="[ENTITY] found [N] [ITEM]",
        dsl_expr="total = entity + value",
        param_slots=["entity", "value"],
        verb_triggers=["found", "discovered"],
        examples=["She found 5 coins", "They discovered 3 artifacts"],
        attention_weight=0.88,
    ),
    DSLTemplate(
        template_id="add_savings",
        operation=Operation.ADD,
        pattern="[ENTITY] saved [N]",
        dsl_expr="total = entity + value",
        param_slots=["entity", "value"],
        verb_triggers=["saved", "put aside", "set aside"],
        examples=["She saved $200", "He put aside $50"],
        attention_weight=0.91,
    ),
    DSLTemplate(
        template_id="add_harvest",
        operation=Operation.ADD,
        pattern="[ENTITY] harvested [N] [ITEM]",
        dsl_expr="total = entity + value",
        param_slots=["entity", "value"],
        verb_triggers=["harvested", "reaped", "picked"],
        examples=["He harvested 100 bales", "They picked 50 apples"],
        attention_weight=0.86,
    ),
    DSLTemplate(
        template_id="add_more_than",
        operation=Operation.ADD,
        pattern="[N] more than [REFERENCE]",
        dsl_expr="total = ref + value",
        param_slots=["ref", "value"],
        verb_triggers=["more"],
        examples=["5 more than yesterday", "$10 more than the first"],
        attention_weight=0.92,
    ),
    DSLTemplate(
        template_id="add_percentage_increase",
        operation=Operation.ADD,
        pattern="[N]% more than [REFERENCE]",
        dsl_expr="total = ref * (1 + value/100)",
        param_slots=["ref", "value"],
        verb_triggers=["more", "increase"],
        examples=["20% more than last year", "15% increase"],
        attention_weight=0.94,
    ),
]


# =============================================================================
# SUB Operations - Reduction/Decrease
# =============================================================================

SUB_TEMPLATES = [
    DSLTemplate(
        template_id="sub_consumption",
        operation=Operation.SUB,
        pattern="[ENTITY] ate [N] [ITEM]",
        dsl_expr="remaining = entity - value",
        param_slots=["entity", "value"],
        verb_triggers=["ate", "consumed", "drank", "used"],
        examples=["She ate 3 slices", "He drank 2 liters"],
        attention_weight=0.96,
    ),
    DSLTemplate(
        template_id="sub_sale",
        operation=Operation.SUB,
        pattern="[ENTITY] sold [N] [ITEM]",
        dsl_expr="remaining = entity - value",
        param_slots=["entity", "value"],
        verb_triggers=["sold", "traded"],
        examples=["Natalia sold 48 clips", "He sold 5 cars"],
        attention_weight=0.95,
    ),
    DSLTemplate(
        template_id="sub_gift",
        operation=Operation.SUB,
        pattern="[ENTITY] gave [N] [ITEM]",
        dsl_expr="remaining = entity - value",
        param_slots=["entity", "value"],
        verb_triggers=["gave", "donated", "gifted"],
        examples=["She gave 10 away", "He donated $50"],
        attention_weight=0.93,
    ),
    DSLTemplate(
        template_id="sub_spending",
        operation=Operation.SUB,
        pattern="[ENTITY] spent [N]",
        dsl_expr="remaining = entity - value",
        param_slots=["entity", "value"],
        verb_triggers=["spent", "paid", "used"],
        examples=["She spent $30 on pants", "He paid $15"],
        attention_weight=0.97,
    ),
    DSLTemplate(
        template_id="sub_loss",
        operation=Operation.SUB,
        pattern="[ENTITY] lost [N] [ITEM]",
        dsl_expr="remaining = entity - value",
        param_slots=["entity", "value"],
        verb_triggers=["lost", "dropped", "misplaced"],
        examples=["She lost 5 pounds", "He dropped 2 items"],
        attention_weight=0.90,
    ),
    DSLTemplate(
        template_id="sub_removal",
        operation=Operation.SUB,
        pattern="[ENTITY] took [N] [ITEM]",
        dsl_expr="remaining = entity - value",
        param_slots=["entity", "value"],
        verb_triggers=["took", "removed", "took away"],
        examples=["He took 3 away", "She removed 5 items"],
        attention_weight=0.89,
    ),
    DSLTemplate(
        template_id="sub_usage",
        operation=Operation.SUB,
        pattern="[ENTITY] used [N] [ITEM]",
        dsl_expr="remaining = entity - value",
        param_slots=["entity", "value"],
        verb_triggers=["used", "utilized", "consumed"],
        examples=["He used 20% of the budget", "She used 5 gallons"],
        attention_weight=0.91,
    ),
    DSLTemplate(
        template_id="sub_baking",
        operation=Operation.SUB,
        pattern="[ENTITY] baked [N] [ITEM]",
        dsl_expr="remaining = entity - value",
        param_slots=["entity", "value"],
        verb_triggers=["baked", "cooked", "made"],
        examples=["He baked 12 cookies", "She made 6 pies"],
        attention_weight=0.85,
    ),
    DSLTemplate(
        template_id="sub_less_than",
        operation=Operation.SUB,
        pattern="[N] less than [REFERENCE]",
        dsl_expr="total = ref - value",
        param_slots=["ref", "value"],
        verb_triggers=["less", "fewer"],
        examples=["3 less than Tuesday", "$5 less than before"],
        attention_weight=0.92,
    ),
    DSLTemplate(
        template_id="sub_percentage_decrease",
        operation=Operation.SUB,
        pattern="[N]% less than [REFERENCE]",
        dsl_expr="total = ref * (1 - value/100)",
        param_slots=["ref", "value"],
        verb_triggers=["less", "decrease", "discount"],
        examples=["20% less than usual", "15% discount"],
        attention_weight=0.93,
    ),
]


# =============================================================================
# MUL Operations - Multiplication/Scaling
# =============================================================================

MUL_TEMPLATES = [
    DSLTemplate(
        template_id="mul_times",
        operation=Operation.MUL,
        pattern="[N] times [REFERENCE]",
        dsl_expr="total = entity * value",
        param_slots=["entity", "value"],
        verb_triggers=["times", "multiplied"],
        examples=["5 times that much", "3 times as many"],
        attention_weight=0.98,
    ),
    DSLTemplate(
        template_id="mul_unit_price",
        operation=Operation.MUL,
        pattern="[N] [ITEM] for [PRICE] each",
        dsl_expr="total = quantity * unit_price",
        param_slots=["quantity", "unit_price"],
        verb_triggers=["for", "each", "per"],
        examples=["5 shirts for $10 each", "3 pizzas at $15 each"],
        attention_weight=0.97,
    ),
    DSLTemplate(
        template_id="mul_rate_time",
        operation=Operation.MUL,
        pattern="[RATE] per [UNIT] for [N] [UNIT]",
        dsl_expr="total = rate * time",
        param_slots=["rate", "time"],
        verb_triggers=["per", "for"],
        examples=["$12 per hour for 8 hours", "5 miles per day for 7 days"],
        attention_weight=0.96,
    ),
    DSLTemplate(
        template_id="mul_fraction",
        operation=Operation.MUL,
        pattern="[N]/[N] of [ENTITY]",
        dsl_expr="result = entity * (numerator / denominator)",
        param_slots=["entity", "numerator", "denominator"],
        verb_triggers=["of"],
        examples=["2/3 of the apples", "3/4 of the money"],
        attention_weight=0.95,
    ),
    DSLTemplate(
        template_id="mul_twice",
        operation=Operation.MUL,
        pattern="twice as [ADJ] as [REFERENCE]",
        dsl_expr="total = ref * 2",
        param_slots=["ref"],
        verb_triggers=["twice", "double"],
        examples=["twice as many", "double the amount"],
        attention_weight=0.94,
    ),
    DSLTemplate(
        template_id="mul_triple",
        operation=Operation.MUL,
        pattern="three times as [ADJ] as [REFERENCE]",
        dsl_expr="total = ref * 3",
        param_slots=["ref"],
        verb_triggers=["three times", "triple"],
        examples=["three times as much", "triple the distance"],
        attention_weight=0.92,
    ),
    DSLTemplate(
        template_id="mul_days_week",
        operation=Operation.MUL,
        pattern="[N] per day for [N] days",
        dsl_expr="total = daily * days",
        param_slots=["daily", "days"],
        verb_triggers=["per", "day", "days"],
        examples=["500 movies a day for 5 days", "10 miles per day for 7 days"],
        attention_weight=0.93,
    ),
    DSLTemplate(
        template_id="mul_percentage_of",
        operation=Operation.MUL,
        pattern="[N]% of [ENTITY]",
        dsl_expr="result = entity * (value / 100)",
        param_slots=["entity", "value"],
        verb_triggers=["percent", "%", "of"],
        examples=["25% of the total", "10% of her salary"],
        attention_weight=0.96,
    ),
]


# =============================================================================
# DIV Operations - Division/Splitting
# =============================================================================

DIV_TEMPLATES = [
    DSLTemplate(
        template_id="div_split",
        operation=Operation.DIV,
        pattern="[ENTITY] split into [N] [PARTS]",
        dsl_expr="part = entity / divisor",
        param_slots=["entity", "divisor"],
        verb_triggers=["split", "divided", "shared"],
        examples=["Split into 4 sections", "Divided among 5 people"],
        attention_weight=0.94,
    ),
    DSLTemplate(
        template_id="div_half",
        operation=Operation.DIV,
        pattern="half of [ENTITY]",
        dsl_expr="result = entity / 2",
        param_slots=["entity"],
        verb_triggers=["half"],
        examples=["Half of the apples", "Half the distance"],
        attention_weight=0.96,
    ),
    DSLTemplate(
        template_id="div_per_unit",
        operation=Operation.DIV,
        pattern="[TOTAL] for [N] [ITEM]",
        dsl_expr="unit_price = total / quantity",
        param_slots=["total", "quantity"],
        verb_triggers=["for", "per"],
        examples=["$100 for 5 shirts", "$60 for 4 pizzas"],
        attention_weight=0.91,
    ),
    DSLTemplate(
        template_id="div_ratio",
        operation=Operation.DIV,
        pattern="[ENTITY] is [N] times [REFERENCE]",
        dsl_expr="ref = entity / multiplier",
        param_slots=["entity", "multiplier"],
        verb_triggers=["times"],
        examples=["She is 3 times his age", "This is 4 times that"],
        attention_weight=0.89,
    ),
    DSLTemplate(
        template_id="div_share",
        operation=Operation.DIV,
        pattern="shared equally among [N] [ENTITY]",
        dsl_expr="share = total / count",
        param_slots=["total", "count"],
        verb_triggers=["shared", "distributed"],
        examples=["Shared among 4 friends", "Distributed to 10 people"],
        attention_weight=0.92,
    ),
    DSLTemplate(
        template_id="div_fraction_inverse",
        operation=Operation.DIV,
        pattern="[RESULT] is [N]/[N] of [ENTITY]",
        dsl_expr="entity = result * (denominator / numerator)",
        param_slots=["result", "numerator", "denominator"],
        verb_triggers=["of"],
        examples=["12 is 2/3 of the total", "15 is 3/4 of the amount"],
        attention_weight=0.88,
    ),
]


# =============================================================================
# Combined Template Registry
# =============================================================================

ALL_TEMPLATES: List[DSLTemplate] = (
    SET_TEMPLATES + ADD_TEMPLATES + SUB_TEMPLATES + MUL_TEMPLATES + DIV_TEMPLATES
)

TEMPLATE_BY_ID: Dict[str, DSLTemplate] = {t.template_id: t for t in ALL_TEMPLATES}

TEMPLATES_BY_OPERATION: Dict[Operation, List[DSLTemplate]] = {
    Operation.SET: SET_TEMPLATES,
    Operation.ADD: ADD_TEMPLATES,
    Operation.SUB: SUB_TEMPLATES,
    Operation.MUL: MUL_TEMPLATES,
    Operation.DIV: DIV_TEMPLATES,
}


def find_matching_templates(
    text: str,
    operation: Optional[Operation] = None
) -> List[DSLTemplate]:
    """Find templates that match the given text based on verb triggers."""
    text_lower = text.lower()
    matches = []

    templates = TEMPLATES_BY_OPERATION.get(operation, ALL_TEMPLATES) if operation else ALL_TEMPLATES

    for template in templates:
        for verb in template.verb_triggers:
            if verb in text_lower:
                matches.append(template)
                break

    # Sort by attention weight
    matches.sort(key=lambda t: -t.attention_weight)
    return matches


def get_template_stats() -> Dict:
    """Get statistics about the template library."""
    return {
        "total_templates": len(ALL_TEMPLATES),
        "by_operation": {op.value: len(ts) for op, ts in TEMPLATES_BY_OPERATION.items()},
        "avg_attention_weight": sum(t.attention_weight for t in ALL_TEMPLATES) / len(ALL_TEMPLATES),
    }


# Quick test
if __name__ == "__main__":
    print("Specialized DSL Template Library")
    print("=" * 50)
    stats = get_template_stats()
    print(f"Total templates: {stats['total_templates']}")
    for op, count in stats['by_operation'].items():
        print(f"  {op}: {count}")
    print(f"Avg attention weight: {stats['avg_attention_weight']:.2f}")

    print("\nExample matching:")
    test_texts = [
        "She bought 5 apples",
        "He sold 3 cars",
        "The pizza costs $15",
        "She earned $100",
        "Split among 4 friends",
    ]
    for text in test_texts:
        matches = find_matching_templates(text)
        if matches:
            print(f"  '{text}' -> {matches[0].template_id} ({matches[0].dsl_expr})")
