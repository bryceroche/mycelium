"""Seed templates to bootstrap the system with coarse-grained reasoning patterns."""
from .models import Template, Example
from .db import save_template, save_example, get_template_by_name
from mycelium.embedding_cache import cached_embed


# Prompt template builder - JSON braces are escaped with quadruple braces
# since the template is formatted twice (once here, once at solve time)
def _make_prompt_template(name: str, guidance: str) -> str:
    """Build a prompt template with proper escaping."""
    return f"""You are solving a {name} math problem.

{guidance}

Problem: {{problem}}

Extract the values and write a Python expression to compute the answer.
Output JSON only:
{{"expression": "<python expression>", "answer": <computed result>}}"""


SEED_TEMPLATES = [
    # Sequential - most common pattern in GSM8K
    Template(
        name="sequential",
        description="Do arithmetic operations in sequence",
        guidance="Extract values, perform operations in order (add, subtract, multiply, divide)",
        prompt_template=_make_prompt_template(
            "sequential",
            "Extract values, perform operations in order (add, subtract, multiply, divide)"
        ),
    ),

    # Complement - percentage/fraction remaining
    Template(
        name="complement",
        description="Find the complement (100-X%)",
        guidance="If X% did something, (100-X)% did the opposite",
        prompt_template=_make_prompt_template(
            "complement",
            "If X% did something, (100-X)% did the opposite"
        ),
    ),

    # Algebra - set up and solve equations
    Template(
        name="algebra",
        description="Set up and solve an equation",
        guidance="Define unknown as x, set up equation, solve for x",
        prompt_template=_make_prompt_template(
            "algebra",
            "Define unknown as x, set up equation, solve for x"
        ),
    ),

    # Conditional - tiered rates or thresholds
    Template(
        name="conditional",
        description="Apply different rates based on thresholds",
        guidance="Split at threshold, apply different rates to each part",
        prompt_template=_make_prompt_template(
            "conditional",
            "Split at threshold, apply different rates to each part"
        ),
    ),

    # Ratio - proportional reasoning
    Template(
        name="ratio",
        description="Work with ratios and proportions",
        guidance="Convert ratio to parts, find value per part, scale",
        prompt_template=_make_prompt_template(
            "ratio",
            "Convert ratio to parts, find value per part, scale"
        ),
    ),

    # Inversion - work backwards
    Template(
        name="inversion",
        description="Work backwards from result",
        guidance="Start from final value, reverse operations to find original",
        prompt_template=_make_prompt_template(
            "inversion",
            "Start from final value, reverse operations to find original"
        ),
    ),
]


SEED_EXAMPLES = [
    # ============================================================
    # SEQUENTIAL - Most common pattern (many examples for coverage)
    # ============================================================
    # Classic GSM8K-style multi-step problems
    ("Janet's ducks lay 16 eggs per day. She eats 3 for breakfast and bakes muffins with 4. She sells the rest at $2 each. How much does she make?", "sequential"),
    ("Tim has 10 apples. He gives 3 to Mary and buys 5 more. How many does he have?", "sequential"),
    ("A store has 100 items. They sell 40 in morning and 25 in afternoon. How many left?", "sequential"),
    ("Sarah earns $15 per hour. She works 8 hours on Monday and 6 hours on Tuesday. How much did she earn?", "sequential"),
    ("A baker makes 120 cookies. He sells 45 in the morning, gives away 20, and sells 30 more in the evening. How many are left?", "sequential"),
    ("John has $50. He spends $12 on lunch, $8 on coffee, and receives $25 from a friend. How much does he have now?", "sequential"),
    ("A farm has 200 chickens. 30 are sold, 15 die, and 45 new chicks are born. How many chickens are there now?", "sequential"),
    ("Maria reads 25 pages on Monday, 32 pages on Tuesday, and 18 pages on Wednesday. How many pages did she read in total?", "sequential"),
    ("A bus has 45 passengers. At the first stop, 12 get off and 8 get on. At the second stop, 15 get off and 20 get on. How many passengers are on the bus now?", "sequential"),
    ("Tom bought 3 notebooks at $4 each and 5 pens at $2 each. How much did he spend in total?", "sequential"),
    ("A factory produces 500 widgets per day. They ship 200 to Store A and 150 to Store B. How many widgets remain?", "sequential"),
    ("Lisa has 80 stickers. She gives 15 to her sister, 20 to her friend, and buys 30 more. How many stickers does she have?", "sequential"),
    ("A restaurant served 120 customers on Friday. On Saturday they served 45 more than Friday. How many customers did they serve on both days?", "sequential"),
    ("Mike earns $12 per hour for 40 hours. After taxes of $96, how much does he take home?", "sequential"),
    ("A tree had 250 apples. Wind knocked down 30, birds ate 15, and farmers picked 180. How many apples remain on the tree?", "sequential"),

    # Simple two-step problems
    ("There are 6 boxes with 8 items each. 10 items are removed. How many items remain?", "sequential"),
    ("A parking lot has 5 rows with 20 spaces each. 35 cars are parked. How many spaces are empty?", "sequential"),
    ("Jake has 3 bags with 12 marbles each. He loses 7 marbles. How many does he have left?", "sequential"),
    ("A library has 8 shelves with 25 books each. 50 books are checked out. How many books are on the shelves?", "sequential"),
    ("Emma buys 4 packs of gum with 5 pieces each. She chews 6 pieces. How many pieces are left?", "sequential"),

    # Money calculations
    ("Coffee costs $4 and a muffin costs $3. If you buy 2 coffees and 3 muffins, how much do you spend?", "sequential"),
    ("A shirt costs $25 and pants cost $40. With a $10 coupon, how much do you pay for both?", "sequential"),
    ("Movie tickets are $12 each. Popcorn is $8 and drinks are $5. How much for 2 tickets, 1 popcorn, and 2 drinks?", "sequential"),
    ("A meal costs $15. With 20% tip and $2 delivery fee, what's the total?", "sequential"),
    ("Books cost $8 each. Buy 5 books and get $10 off. How much do you pay?", "sequential"),

    # ============================================================
    # COMPLEMENT - Percentage remaining
    # ============================================================
    ("40% of students got below average. How many percent got average or above?", "complement"),
    ("30% of the cookies were eaten. What percent remain?", "complement"),
    ("A survey shows 65% of people prefer coffee. What percent prefer something else?", "complement"),
    ("25% of the budget was spent on marketing. What percent was spent on other things?", "complement"),
    ("If 15% of applicants were rejected, what percent were accepted?", "complement"),
    ("72% of voters supported the measure. What percent opposed it?", "complement"),
    ("A shirt is 35% cotton. What percent is other materials?", "complement"),
    ("If 88% of students passed the test, what percent failed?", "complement"),

    # ============================================================
    # ALGEBRA - Set up and solve equations
    # ============================================================
    ("A number increased by 20% equals 60. What was the original number?", "algebra"),
    ("After spending $15, Tom has twice what he started with minus $15. He now has $45. How much did he start with?", "algebra"),
    ("Three times a number minus 7 equals 20. What is the number?", "algebra"),
    ("The sum of two consecutive numbers is 37. What are the numbers?", "algebra"),
    ("A rectangle's length is 3 more than its width. The perimeter is 26. Find the dimensions.", "algebra"),
    ("John is 5 years older than Mary. In 3 years, their ages will sum to 35. How old is Mary now?", "algebra"),
    ("Twice a number plus 8 equals the number plus 20. What is the number?", "algebra"),
    ("A number divided by 4 then increased by 6 equals 15. What is the number?", "algebra"),

    # ============================================================
    # CONDITIONAL - Tiered rates and thresholds
    # ============================================================
    ("Workers earn $10/hour for first 40 hours and $15/hour for overtime. How much for 50 hours?", "conditional"),
    ("A phone plan charges $0.05/minute for the first 100 minutes and $0.03/minute after. What's the cost for 150 minutes?", "conditional"),
    ("Shipping is $5 for orders under $50 and free for orders $50 or more. What's the total for a $45 order?", "conditional"),
    ("Tax is 10% on income up to $10,000 and 20% on income above that. What's the tax on $15,000?", "conditional"),
    ("A gym charges $30/month for up to 10 visits and $2 per visit after that. What's the cost for 15 visits?", "conditional"),
    ("Electricity costs $0.10/kWh for the first 500 kWh and $0.15/kWh after. What's the bill for 700 kWh?", "conditional"),
    ("A salesperson earns 5% commission on sales up to $1000 and 8% on sales above that. What's the commission on $1500?", "conditional"),

    # ============================================================
    # RATIO - Proportional reasoning
    # ============================================================
    ("Boys to girls ratio is 2:3. If there are 10 boys, how many girls?", "ratio"),
    ("The ratio of cats to dogs is 3:4. If there are 21 cats, how many dogs are there?", "ratio"),
    ("A recipe calls for flour and sugar in a 5:2 ratio. If you use 10 cups of flour, how much sugar?", "ratio"),
    ("Red and blue marbles are in ratio 3:7. If there are 28 blue marbles, how many red?", "ratio"),
    ("Workers to managers ratio is 8:1. If there are 72 workers, how many managers?", "ratio"),
    ("The ratio of fiction to non-fiction books is 5:3. If there are 40 fiction books, how many non-fiction?", "ratio"),
    ("Apples to oranges ratio is 4:5. With 36 pieces of fruit total, how many apples?", "ratio"),
    ("Paint mix requires red and white in 2:5 ratio. For 14 liters total, how much red paint?", "ratio"),

    # ============================================================
    # INVERSION - Work backwards
    # ============================================================
    ("After a 20% discount, the price was $80. What was the original price?", "inversion"),
    ("A number was tripled and then 10 was subtracted, giving 50. What was the original number?", "inversion"),
    ("After adding 25% tax, the total was $75. What was the pre-tax price?", "inversion"),
    ("Maria spent half her money and then $10 more, leaving her with $15. How much did she start with?", "inversion"),
    ("A population doubled and then increased by 100, reaching 500. What was the original population?", "inversion"),
    ("After a 15% raise, salary became $5750. What was the original salary?", "inversion"),
    ("A number was divided by 4, then 5 was added, giving 12. What was the original number?", "inversion"),
    ("The price after 30% off is $35. What was the original price?", "inversion"),
]


def seed_database():
    """Seed the database with initial templates and examples."""
    print("Seeding reasoning pattern templates...")

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


def get_prompt_for_template(template_name: str, problem: str) -> str:
    """Generate a prompt for the given template and problem."""
    guidance_map = {
        "sequential": "Extract values, perform operations in order (add, subtract, multiply, divide)",
        "complement": "If X% did something, (100-X)% did the opposite",
        "algebra": "Define unknown as x, set up equation, solve for x",
        "conditional": "Split at threshold, apply different rates to each part",
        "ratio": "Convert ratio to parts, find value per part, scale",
        "inversion": "Start from final value, reverse operations to find original",
    }

    guidance = guidance_map.get(template_name, "Solve the problem step by step")

    return PROMPT_TEMPLATE.format(
        name=template_name,
        guidance=guidance,
        problem=problem
    )


if __name__ == "__main__":
    seed_database()
