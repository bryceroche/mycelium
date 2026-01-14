# Tree Size Estimation

## Current Gaps in Function Registry

| Category | Missing | Examples |
|----------|---------|----------|
| **Linear Algebra** | ~15-20 | dot, cross, matmul, transpose, inverse, det, eigenvalues |
| **Complex Numbers** | ~5 | real, imag, conjugate, phase |
| **More Number Theory** | ~10 | is_prime, prime_factors, divisors, mod_pow, totient |
| **Probability** | ~10-15 | binomial, normal_cdf, expected_value, bayes |
| **Geometry** | ~10-15 | distance, area_triangle, area_circle, volume, angle_between |
| **Sequences** | ~5-10 | arithmetic_sum, geometric_sum, fibonacci |
| **Financial** | ~5 | compound_interest, simple_interest, pv, fv |
| **Competition Math** | ~20-30 | chinese_remainder, stirling, catalan, polynomial_roots |

---

## Rough Estimate

- **Current:** 57 functions
- **Comprehensive:** ~150-200 functions

---

## But Here's the Key Insight

The **tree** isn't just functions - it's **signatures** (step_type + function + learned embedding).

One function can have multiple signatures:

```
add:
├── "combine two prices"        → signature_42
├── "sum the quantities"        → signature_87
└── "total distance traveled"   → signature_156
```

---

## Tree Size at Maturity

- ~150-200 functions
- ~2-5 signatures per function (semantic variants)
- = **~300-1000 leaf signatures**

Plus umbrella routers for hierarchical routing.
