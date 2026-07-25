# SPEC-TIGHTEN SPECIMEN SHEET (2026-07-25, for Bryce's pass)

## Flag 1 — k=1 legality: THE SET IS EMPTY, TOTALLY
0 k=1 emissions in 1,710,655 banked mod/fdiv factors (every jsonl corpus);
0 k=1 in the 3,800 failure parses. ONE k=0 emission exists in failure space
(slot grain, no text linkage banked). The empirical branch of the framework:
no source semantics has ever compiled to k=1, so the law may forbid it at
zero retroactive cost — and the verifier's k>=2 check would guard only
FUTURE emitters (the organ). Ruling needed: forbid (tighten to corpus) or
permit-degenerate (tighten to intent — 'one group of five' semantics).

## Flag 2 — pct's result-less form: 8 specimens
THE SOLVER'S READING (csp_domains, LTYPE_PCT): pct(args=[a,b], p) asserts
a*100 == p*b — a pure RELATION between two existing vars via the literal p.
No third variable is produced; there is nothing for a result slot to name.

### Specimen 1
TEXT: The following facts hold about a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, r, s, t, u. It is known that j is 6. Adding k to l gives m. the first number exceeds b by d. When e is divided by 4, the remainder is p. T
PCT:  {"ftype": "pct", "args": [16, 17], "p": 10}
GRAPH (21 factors): rel:add; rel:add; given; given; rel:add; rel:add; given; given; rel:add; given; rel:add; rel:mul

### Specimen 2
TEXT: The following facts hold about a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, r. e and f together make g. e minus the sixth number equals h. The remainder when i is divided by 8 is j. The remainder when i is divided 
PCT:  {"ftype": "pct", "args": [16, 17], "p": 200}
GRAPH (19 factors): rel:add; rel:add; given; given; rel:add; rel:add; given; given; mod; mod; given; given

### Specimen 3
TEXT: Consider the numbers l, p, a, o, d, c, j, i, n, g, f, b, k, e, h, m. It is known that i is 6. When l is divided by 7, the remainder is n. It is known that a is 42. the fourth number is 4. The sum of d and c is j. the fou
PCT:  {"ftype": "pct", "args": [14, 15], "p": 50}
GRAPH (16 factors): rel:add; rel:add; given; given; rel:add; rel:add; given; given; mod; pct; rel:add; given

### Specimen 4
TEXT: Let a, b, c, d, e, f, g, h, i, j, k, l, m, n be whole numbers. Dividing e by 8 leaves a remainder of g. e is what you get from a plus f. It is known that f is 32. The value of the fourth number is 7. c is 23. The sum of 
PCT:  {"ftype": "pct", "args": [12, 13], "p": 75}
GRAPH (14 factors): rel:add; rel:add; given; given; rel:add; given; mod; rel:add; given; given; rel:add; given

### Specimen 5
TEXT: The following facts hold about a, b, c, d, e, f, g, h, i. It is known that c is 19. a exceeds b by d. c is the total of a and b. It is known that d is 7. f equals 20. g has the value 27. When g is divided by 4, the quoti
PCT:  {"ftype": "pct", "args": [4, 5], "p": 50}
GRAPH (9 factors): rel:add; rel:add; given; given; given; given; fdiv; mod; pct

### Specimen 6
TEXT: Let j, i, b, g, h, d, e, m, n, k, l, c, o, f, a be whole numbers. m is h reduced by d. It is known that g is 19. e has the value 32. Adding j to k gives n. m equals 2. k equals 7. e is what you get from h plus d. g is j 
PCT:  {"ftype": "pct", "args": [10, 11], "p": 50}
GRAPH (15 factors): rel:add; rel:add; given; given; rel:add; rel:add; given; given; rel:add; given; given; pct

### Specimen 7
TEXT: Consider the numbers m, j, o, d, p, r, c, l, k, i, g, e, q, b, n, a, h, s, f. The product of c and l is i. m plus j equals o. e leaves remainder b when divided by 9. The remainder when e is divided by 7 is q. The value o
PCT:  {"ftype": "pct", "args": [14, 15], "p": 20}
GRAPH (20 factors): rel:add; rel:add; given; given; rel:mul; given; rel:add; rel:mul; given; given; sel; mod

### Specimen 8
TEXT: The following facts hold about a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, r, s. g is 11. The difference between a and b is d. c is the total of a and b. i leaves remainder j when divided by 9. i is whichever of e
PCT:  {"ftype": "pct", "args": [14, 15], "p": 10}
GRAPH (19 factors): rel:add; rel:add; given; given; rel:add; rel:mul; given; given; sel; mod; given; fdiv

## Flag 3 — loc: no specimens needed (ruled in principle at #69).
Design point for the writing: derived nodes (canonicalizer folds,
solver intermediates) carry loc:derived with parent spans — provenance
never silently vanishes.

## The fifty's annotation schema (machine-readable, per the ruling)
```json
{"id": int, "stratum": {"zone": "...", "species": "...", "vintage": "..."},
 "text": "...", "graph": [...],
 "altitude": "schema | assembly",
 "confidence": "high | low",
 "note": "free text ONLY for binary-resisting specimens"}
```
The binary-resisters are themselves data on whether two altitudes suffice.