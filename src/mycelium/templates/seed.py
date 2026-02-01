"""Seed templates to bootstrap the system."""
from .models import Template, ComputeGraph, Example
from .db import save_template, save_example, get_template_by_name
from mycelium.embedding_cache import cached_embed


SEED_TEMPLATES = [
    # Basic arithmetic - subtraction
    Template(
        name="subtraction",
        description="Subtract Y from X to get remainder",
        pattern="[AGENT] has [X] [OBJECT]. Gives/loses [Y]. How many left?",
        slots=["X", "Y"],
        graph=ComputeGraph(
            nodes=["X", "Y", "answer"],
            edges=[{"op": "sub", "inputs": ["X", "Y"], "output": "answer"}]
        )
    ),

    # Basic arithmetic - addition
    Template(
        name="addition",
        description="Add X and Y together",
        pattern="[AGENT] has [X] [OBJECT] and gets [Y] more. How many total?",
        slots=["X", "Y"],
        graph=ComputeGraph(
            nodes=["X", "Y", "answer"],
            edges=[{"op": "add", "inputs": ["X", "Y"], "output": "answer"}]
        )
    ),

    # Multiplication
    Template(
        name="multiplication",
        description="Multiply X by Y",
        pattern="[X] items at [Y] each. What is total?",
        slots=["X", "Y"],
        graph=ComputeGraph(
            nodes=["X", "Y", "answer"],
            edges=[{"op": "mul", "inputs": ["X", "Y"], "output": "answer"}]
        )
    ),

    # Division
    Template(
        name="division",
        description="Divide X by Y",
        pattern="[X] items split among [Y] people. How many each?",
        slots=["X", "Y"],
        graph=ComputeGraph(
            nodes=["X", "Y", "answer"],
            edges=[{"op": "div", "inputs": ["X", "Y"], "output": "answer"}]
        )
    ),

    # Circle radius (complete the square)
    Template(
        name="circle_radius",
        description="Find radius of circle by completing the square",
        pattern="Circle equation x^2 + [A]x + y^2 + [B]y = [C]. Find radius.",
        slots=["A", "B", "C"],
        graph=ComputeGraph(
            nodes=["A", "B", "C", "h_sq", "k_sq", "r_sq", "answer"],
            edges=[
                {"op": "div", "inputs": ["A", "2"], "output": "h"},
                {"op": "pow", "inputs": ["h", "2"], "output": "h_sq"},
                {"op": "div", "inputs": ["B", "2"], "output": "k"},
                {"op": "pow", "inputs": ["k", "2"], "output": "k_sq"},
                {"op": "add", "inputs": ["h_sq", "k_sq"], "output": "sum_sq"},
                {"op": "add", "inputs": ["sum_sq", "C"], "output": "r_sq"},
                {"op": "sqrt", "inputs": ["r_sq"], "output": "answer"}
            ]
        )
    ),

    # Midpoint
    Template(
        name="midpoint",
        description="Find unknown point given midpoint and other point",
        pattern="Midpoint of ([X1],[Y1]) and ([X2],[Y2]) is ([MX],[MY]). Find unknown.",
        slots=["X1", "Y1", "MX", "MY"],
        graph=ComputeGraph(
            nodes=["X1", "Y1", "MX", "MY", "X2", "Y2", "answer"],
            edges=[
                {"op": "mul", "inputs": ["MX", "2"], "output": "mx2"},
                {"op": "sub", "inputs": ["mx2", "X1"], "output": "X2"},
                {"op": "mul", "inputs": ["MY", "2"], "output": "my2"},
                {"op": "sub", "inputs": ["my2", "Y1"], "output": "Y2"},
            ]
        )
    ),

    # System of 3 equations (a+b, b+c, a+c)
    Template(
        name="system_three",
        description="Solve a+b=P, b+c=Q, a+c=R for abc",
        pattern="a+b=[P], b+c=[Q], a+c=[R]. Find abc.",
        slots=["P", "Q", "R"],
        graph=ComputeGraph(
            nodes=["P", "Q", "R", "sum", "a", "b", "c", "answer"],
            edges=[
                # 2(a+b+c) = P+Q+R, so a+b+c = (P+Q+R)/2
                {"op": "add", "inputs": ["P", "Q"], "output": "pq"},
                {"op": "add", "inputs": ["pq", "R"], "output": "total"},
                {"op": "div", "inputs": ["total", "2"], "output": "sum"},
                # a = sum - (b+c) = sum - Q
                {"op": "sub", "inputs": ["sum", "Q"], "output": "a"},
                # b = sum - (a+c) = sum - R
                {"op": "sub", "inputs": ["sum", "R"], "output": "b"},
                # c = sum - (a+b) = sum - P
                {"op": "sub", "inputs": ["sum", "P"], "output": "c"},
                # abc = a * b * c
                {"op": "mul", "inputs": ["a", "b"], "output": "ab"},
                {"op": "mul", "inputs": ["ab", "c"], "output": "answer"}
            ]
        )
    ),

    # Vieta's sum of roots
    Template(
        name="vieta_sum",
        description="Sum of roots of ax^2 + bx + c = 0 is -b/a",
        pattern="Quadratic [A]x^2 + [B]x + [C] = 0. Sum of roots?",
        slots=["A", "B"],
        graph=ComputeGraph(
            nodes=["A", "B", "answer"],
            edges=[
                {"op": "neg", "inputs": ["B"], "output": "neg_b"},
                {"op": "div", "inputs": ["neg_b", "A"], "output": "answer"}
            ]
        )
    ),

    # Exponent same-base
    Template(
        name="exponent_equation",
        description="Solve exponential equations by matching bases",
        pattern="[BASE1]^[EXP1] = [BASE2]^[EXP2], solve for n",
        slots=["RESULT"],  # LLM computes the answer directly
        graph=ComputeGraph(
            nodes=["RESULT", "answer"],
            edges=[{"op": "add", "inputs": ["RESULT", "0"], "output": "answer"}]  # Identity
        )
    ),
]


SEED_EXAMPLES = [
    # Subtraction examples
    ("Tim has 10 apples. He gives 3 to Mary. How many left?", "subtraction"),
    ("Sarah has 15 books. She loses 7. How many does she have?", "subtraction"),
    ("A store has 100 items. They sell 40. How many remain?", "subtraction"),
    ("John had $50. He spent $23 on lunch. How much money does he have left?", "subtraction"),
    ("There were 85 people at the party. 32 people left early. How many people stayed?", "subtraction"),
    ("A movie is 120 minutes long. We have watched 45 minutes. How many minutes are left?", "subtraction"),
    ("The bakery made 200 cupcakes. They sold 156 by noon. How many cupcakes remain?", "subtraction"),

    # Addition examples
    ("Tom has 5 marbles. He finds 8 more. How many total?", "addition"),
    ("Lisa has 12 stickers. Her friend gives her 6. How many now?", "addition"),
    ("A train traveled 120 miles. It has 80 more miles to go. What is the total distance?", "addition"),
    ("Mark earned $45 on Monday and $67 on Tuesday. How much did he earn in total?", "addition"),
    ("The library had 340 books. They received a donation of 125 new books. How many books does the library have now?", "addition"),
    ("In the morning, 28 birds were in the tree. In the afternoon, 15 more birds arrived. How many birds are there now?", "addition"),
    # Money addition with "finds" and "more"
    ("Sarah has $30. She finds $10 more. How much money does she have now?", "addition"),
    ("Mike has $25. He receives $15 more. How much money does he have?", "addition"),
    ("The jar has 50 coins. We add 25 more coins. How many coins are there now?", "addition"),
    ("Jake has 20 cards. He gets 8 more cards. How many cards does he have now?", "addition"),

    # Multiplication examples
    ("There are 6 boxes with 8 items each. How many total items?", "multiplication"),
    ("A farmer has 5 rows with 12 plants each. How many plants?", "multiplication"),
    ("A movie theater has 15 rows with 24 seats in each row. How many seats are there in total?", "multiplication"),
    ("Each package contains 8 batteries. If you buy 7 packages, how many batteries do you have?", "multiplication"),
    ("A parking lot has 9 levels with 45 cars on each level. How many cars are parked in total?", "multiplication"),

    # Division examples
    ("24 cookies divided among 6 children. How many each?", "division"),
    ("A class of 30 students split into 5 groups. How many per group?", "division"),
    ("There are 72 pencils to be shared equally among 8 students. How many pencils does each student get?", "division"),
    ("A chef has 96 ounces of sauce to divide into 12 equal portions. How many ounces per portion?", "division"),
    ("The company earned $144,000 to be split equally among 6 partners. How much does each partner receive?", "division"),

    # Circle examples
    ("Find the radius of the circle with equation x^2 + 8x + y^2 - 6y = 0.", "circle_radius"),
    ("What is the radius of x^2 + 4x + y^2 - 2y = 11?", "circle_radius"),

    # Midpoint examples
    ("The midpoint of (x,y) and (-9,1) is (3,-5). Find (x,y).", "midpoint"),

    # System of 3 examples
    ("If a+b=8, b+c=-3, and a+c=-5, what is abc?", "system_three"),

    # Vieta examples
    ("What is the sum of roots of 2x^2 - 10x + 3 = 0?", "vieta_sum"),
]


def seed_database():
    """Seed the database with initial templates and examples."""
    print("Seeding templates...")

    # Save templates
    template_ids = {}
    for template in SEED_TEMPLATES:
        # Check if already exists
        existing = get_template_by_name(template.name)
        if existing:
            template_ids[template.name] = existing.id
            print(f"  Template '{template.name}' already exists (id={existing.id})")
        else:
            tid = save_template(template)
            template_ids[template.name] = tid
            print(f"  Created template '{template.name}' (id={tid})")

    print("\nSeeding examples...")

    # Save examples
    for problem, template_name in SEED_EXAMPLES:
        template_id = template_ids.get(template_name)
        if not template_id:
            print(f"  Skipping example - template '{template_name}' not found")
            continue

        embedding = cached_embed(problem)
        if embedding is None:
            print(f"  Skipping example - failed to embed: {problem[:50]}...")
            continue

        example = Example(
            problem_text=problem,
            embedding=embedding,
            template_id=template_id,
        )
        save_example(example)
        print(f"  Added example for '{template_name}': {problem[:50]}...")

    print("\nSeeding complete!")


if __name__ == "__main__":
    seed_database()
