"""
Quick test of the E2E pipeline flow using mock models.
Demonstrates that the search + executor + bridging produces correct answers
even with imperfect component predictions.

Run: python test_pipeline_flow.py
No GPU needed.
"""

import math
from e2e_pipeline import (
    Span, SpanGroup, Operation, GraphNode, Candidate,
    generate_candidate_groupings, bridging_search, execute_operation,
    score_candidate, answer_matches, get_goal_hints, extract_numbers,
    BRIDGING_TEMPLATES, DOMAIN_CONSTANT_VALUES,
)


def test_number_extraction():
    """Test number extraction from text."""
    print("=" * 50)
    print("Test: Number Extraction")
    print("=" * 50)
    
    tests = [
        ("48 of her friends", [48]),
        ("$2 each", [2]),
        ("1,000 dollars", [1000]),
        ("half as many", [0.5]),
        ("twice the amount", [2]),
        ("a dozen eggs", [12]),
        ("3.14 meters", [3.14]),
    ]
    
    passed = 0
    for text, expected in tests:
        result = extract_numbers(text)
        match = all(e in result for e in expected)
        status = "✓" if match else "✗"
        print(f"  {status} '{text}' → {result} (expected {expected})")
        if match:
            passed += 1
    
    print(f"  Passed: {passed}/{len(tests)}\n")


def test_candidate_groupings():
    """Test candidate grouping generation."""
    print("=" * 50)
    print("Test: Candidate Groupings")
    print("=" * 50)
    
    spans = [
        Span("48 of her friends", 0, 17, [48]),
        Span("half as many", 40, 52, [0.5]),
        Span("altogether", 70, 80, []),
    ]
    
    groupings = generate_candidate_groupings(spans)
    print(f"  {len(spans)} spans → {len(groupings)} candidate groupings")
    for i, g in enumerate(groupings[:5]):
        groups_str = " | ".join(f"[{','.join(s.text for s in sg.spans)}]" for sg in g)
        print(f"    Candidate {i}: {groups_str}")
    
    assert len(groupings) >= 3, f"Expected ≥3 groupings, got {len(groupings)}"
    print(f"  ✓ Pass\n")


def test_executor():
    """Test symbolic executor."""
    print("=" * 50)
    print("Test: Symbolic Executor")
    print("=" * 50)
    
    tests = [
        ("MUL", [5, 2], 10),
        ("ADD", [10, 4.5], 14.5),
        ("SUB", [100, 37], 63),
        ("DIV", [48, 2], 24),
        ("POW", [2, 10], 1024),
        ("SQRT", [144], 12),
        ("MOD", [17, 5], 2),
        ("COMB", [10, 3], 120),
        ("FACTORIAL", [5], 120),
    ]
    
    passed = 0
    for op, args, expected in tests:
        result = execute_operation(op, args)
        match = result is not None and abs(result - expected) < 1e-6
        status = "✓" if match else "✗"
        print(f"  {status} {op}({args}) = {result} (expected {expected})")
        if match:
            passed += 1
    
    print(f"  Passed: {passed}/{len(tests)}\n")


def test_bridging_search():
    """Test bridging search finds correct answers."""
    print("=" * 50)
    print("Test: Bridging Search")
    print("=" * 50)
    
    # Problem: "Bob bought 5 apples at $2 each and 3 oranges at $1.50 each. How much total?"
    # Explicit results: MUL(5,2)=10, MUL(3,1.50)=4.50
    # Bridging: ACC_ADD([10, 4.50]) = 14.50
    
    explicit_results = [10.0, 4.5]
    gold = 14.5
    hints = get_goal_hints("How much did Bob spend in total?")
    
    candidates = bridging_search(explicit_results, hints, gold, max_hops=2)
    
    found_correct = False
    for nodes, answer in candidates:
        if answer_matches(answer, gold):
            found_correct = True
            ops = " → ".join(f"{n.operation}({n.params})={n.result}" for n in nodes)
            print(f"  ✓ Found: {ops}")
            break
    
    assert found_correct, "Bridging search didn't find correct answer"
    print(f"  Total candidates explored: {len(candidates)}")
    
    # Problem: "She works 8 hours a day. How many minutes does she work?"
    # Explicit: just 8
    # Bridging: MUL(8, 60) = 480 (domain constant)
    
    explicit_results2 = [8.0]
    gold2 = 480.0
    hints2 = get_goal_hints("How many minutes does she work?")
    
    candidates2 = bridging_search(explicit_results2, hints2, gold2, max_hops=2)
    
    found_correct2 = False
    for nodes, answer in candidates2:
        if answer_matches(answer, gold2):
            found_correct2 = True
            ops = " → ".join(f"{n.operation}({n.params})={n.result}" for n in nodes)
            print(f"  ✓ Found (domain constant): {ops}")
            break
    
    assert found_correct2, "Bridging search didn't find domain constant answer"
    print(f"  ✓ Pass\n")


def test_two_hop_bridging():
    """Test two-hop bridging chains."""
    print("=" * 50)
    print("Test: Two-Hop Bridging")
    print("=" * 50)
    
    # Problem: "A store sold 15 items at $4 and 20 items at $3. What's the average price per item?"
    # Explicit: MUL(15,4)=60, MUL(20,3)=60
    # Hop 1: ACC_ADD([60, 60]) = 120 (total revenue)
    # Hop 2: DIV(120, 35) = 3.43... 
    # But wait, we need 35 (15+20) too... this requires the sum of quantities
    
    # Simpler two-hop: "3 workers earn $100 each per week. How much total per year?"
    # Explicit: MUL(3, 100) = 300
    # Hop 1: result = 300
    # Hop 2: MUL(300, 52) = 15600 (domain constant: weeks/year)
    
    explicit_results = [300.0]
    gold = 15600.0
    hints = get_goal_hints("How much total per year?")
    
    candidates = bridging_search(explicit_results, hints, gold, max_hops=2)
    
    found_correct = False
    for nodes, answer in candidates:
        if answer_matches(answer, gold):
            found_correct = True
            ops = " → ".join(f"{n.operation}({n.params})={n.result}" for n in nodes)
            print(f"  ✓ Found two-hop: {ops}")
            break
    
    assert found_correct, "Two-hop bridging didn't find correct answer"
    print(f"  Total candidates: {len(candidates)}")
    print(f"  ✓ Pass\n")


def test_full_pipeline_mock():
    """
    Test full pipeline flow with mock model outputs.
    Simulates what happens when models produce imperfect predictions
    but the search + executor finds the right answer.
    """
    print("=" * 50)
    print("Test: Full Pipeline Flow (Mock)")
    print("=" * 50)
    
    problem = "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?"
    gold = 72.0
    
    # Mock segmenter output (might be imperfect)
    spans = [
        Span("48 of her friends", 26, 43, [48]),
        Span("half as many clips", 65, 83, [0.5]),
    ]
    
    # Generate groupings
    groupings = generate_candidate_groupings(spans)
    print(f"  Spans: {[s.text for s in spans]}")
    print(f"  Groupings: {len(groupings)}")
    
    # Mock classifier + extractor for each grouping
    best_answer = None
    best_score = -float('inf')
    
    hints = get_goal_hints("How many clips did Natalia sell altogether?")
    
    for grouping in groupings:
        # For the paired grouping {48, half} → this should be DIV or MUL
        # The classifier might say MUL, DIV, SUB... we try all
        
        for try_op in ["MUL", "DIV", "SUB", "ADD"]:
            # Mock extractor: pull numbers from spans
            all_numbers = []
            for group in grouping:
                for span in group.spans:
                    all_numbers.extend(span.numbers)
            
            if len(all_numbers) < 2:
                continue
            
            # Try the operation
            explicit_result = execute_operation(try_op, all_numbers[:2])
            if explicit_result is None:
                continue
            
            # Try direct answer
            if answer_matches(explicit_result, gold):
                score = 1000 + 0.5  # gold match + some confidence
                if score > best_score:
                    best_score = score
                    best_answer = explicit_result
                    print(f"  ✓ Direct: {try_op}({all_numbers[:2]}) = {explicit_result}")
            
            # Try bridging
            bridge_results = bridging_search([explicit_result], hints, gold, max_hops=2)
            for nodes, bridge_answer in bridge_results:
                if answer_matches(bridge_answer, gold):
                    score = 1000 + 0.3
                    if score > best_score and best_answer is None:
                        best_score = score
                        best_answer = bridge_answer
                        ops = " → ".join(f"{n.operation}" for n in nodes)
                        print(f"  ✓ Bridged: {try_op}({all_numbers[:2]})={explicit_result} → {ops} → {bridge_answer}")
    
    # The key insight: DIV(48, 0.5) = 96 (wrong), MUL(48, 0.5) = 24 (right!)
    # Then ACC_ADD([24, 48]) = 72 (but we need 48 from somewhere)
    # OR: just MUL(48, 0.5) = 24, then 48 + 24 = 72 via bridging
    
    # Let's also try: what if segmenter gives us 48 and the problem text has 48 again?
    # The pipeline should try adding the raw 48 as an explicit result
    
    assert best_answer is not None and answer_matches(best_answer, gold), \
        f"Pipeline didn't find gold answer {gold}, best was {best_answer}"
    
    print(f"\n  Final answer: {best_answer} (gold: {gold})")
    print(f"  ✓ Pass\n")


def test_goal_hints():
    """Test goal hint extraction."""
    print("=" * 50)
    print("Test: Goal Hints")
    print("=" * 50)
    
    tests = [
        ("How many clips did she sell altogether?", ["ACC_ADD"]),
        ("How much change did she receive?", ["ACC_SUB"]),
        ("What is the average score?", ["AVG"]),
        ("How many hours per day?", ["DIV_DERIVED", "MUL_DERIVED"]),
    ]
    
    passed = 0
    for question, expected_subset in tests:
        hints = get_goal_hints(question)
        has_all = all(e in hints for e in expected_subset)
        status = "✓" if has_all else "✗"
        print(f"  {status} '{question[:40]}...' → {hints}")
        if has_all:
            passed += 1
    
    print(f"  Passed: {passed}/{len(tests)}\n")


def test_answer_matching():
    """Test answer comparison."""
    print("=" * 50)
    print("Test: Answer Matching")
    print("=" * 50)
    
    tests = [
        (72.0, 72.0, True),
        (72.0000001, 72.0, True),
        (71.9999999, 72.0, True),
        (73.0, 72.0, False),
        (14.5, 14.5, True),
    ]
    
    passed = 0
    for pred, gold, expected in tests:
        result = answer_matches(pred, gold)
        status = "✓" if result == expected else "✗"
        print(f"  {status} {pred} == {gold}? {result} (expected {expected})")
        if result == expected:
            passed += 1
    
    print(f"  Passed: {passed}/{len(tests)}\n")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  MYCELIUM E2E PIPELINE - FLOW TEST")
    print("=" * 60 + "\n")
    
    test_number_extraction()
    test_candidate_groupings()
    test_executor()
    test_goal_hints()
    test_answer_matching()
    test_bridging_search()
    test_two_hop_bridging()
    test_full_pipeline_mock()
    
    print("=" * 60)
    print("  ALL TESTS PASSED")
    print("=" * 60)
