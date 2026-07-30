"""gate_fixtures.py — THE ONE AUTHORITY for the regression roster
(2026-07-30, the sweep's #4/#5 findings): the nine-fixture list
previously lived as independently-maintained literals in every
battery_genNN.py AND every genNN_verdict.py — eight generations of
copy/paste with no cross-check, and the PROMOTE/KILL fixture path
duplicated between the battery that scores and the verdict that
grades. Both now import from here (the two-home law: one authority,
zero copies). Adding a fixture = one edit, visible to both consumers
mechanically."""

FIXTURES = [("bigtest", ".cache/algebra_nl_bigtest.jsonl"),
            ("alg4test", ".cache/algebra4_nl_test.jsonl"),
            ("alg2test", ".cache/algebra2_nl_test.jsonl"),
            ("vtest", ".cache/algv_test_verbose.jsonl"),
            ("dagtest", ".cache/dag_test.jsonl"),
            ("dag7btest", ".cache/dag7b_test.jsonl"),
            ("dag8test", ".cache/dag8_test.jsonl"),
            ("h3held", ".cache/gen17_hundreds_held.jsonl"),
            ("adupheld", ".cache/gen17_adup_held.jsonl")]
FIXTURE_PATHS = dict(FIXTURES)
BIGTEST = FIXTURE_PATHS["bigtest"]
