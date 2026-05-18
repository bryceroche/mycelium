# Mycelium curriculum — sample problems

Generated 2026-05-18 from `mycelium/l3_data.py:generate_math(seed=7, digit_spacing=True)`.
Three samples per curriculum level, drawn from the actual training distribution
the breathing transformer sees.

---

## ARITH — bare arithmetic (no narrative)

```
Q: 6 * 8 =
A: 4 8 .
   gold: 48

Q: 4 4 / 2 =
A: 2 2 .
   gold: 22

Q: 1 4 7 + 2 9 =
A: 1 7 6 .
   gold: 176
```

---

## L3 — single-cycle word problem (one arithmetic step)

```
Q: Alex had 2 2 rocks. Alex doubled the collection.
   How many rocks does Alex have now?
A: Alex had 2 2 rocks and doubled them. 2 2 * 2 = 4 4 rocks now.
   gold: 44

Q: Emma had 8 4 crayons. Emma tripled the collection.
   How many crayons does Emma have now?
A: Emma had 8 4 crayons and tripled them. 8 4 * 3 = 2 5 2 crayons now.
   gold: 252

Q: Ryan collected 1 9 cards in the morning and 2 7 in the afternoon.
   How many cards did Ryan collect in total?
A: Ryan had 1 9 and got 2 7 more. 1 9 + 2 7 = 4 6 cards total.
   gold: 46
```

---

## L4_MIXED — two-cycle multi-step word problem (the current training level)

```
Q: Alex had 2 2 rocks. Alex doubled the collection, then gave 5 away.
   How many rocks does Alex have now?
A: Alex had 2 2 rocks and doubled them. 2 2 * 2 = 4 4 rocks now.
   Then Alex gave 5 away. 4 4 - 5 = 3 9 rocks remaining.
   gold: 39

Q: A bakery had 1 9 1 crayons. They threw out 7 5 on Monday
   and gave away 2 8 on Tuesday. How many crayons are left?
A: They started with 1 9 1 crayons and threw out 7 5.
   1 9 1 - 7 5 = 1 1 6 crayons remaining.
   Then they gave away 2 8 more. 1 1 6 - 2 8 = 8 8 crayons left.
   gold: 88

Q: Sarah had 5 3 stickers. Sarah doubled the collection, then gave 3 7 away.
   How many stickers does Sarah have now?
A: Sarah had 5 3 stickers and doubled them. 5 3 * 2 = 1 0 6 stickers now.
   Then Sarah gave 3 7 away. 1 0 6 - 3 7 = 6 9 stickers remaining.
   gold: 69
```

---

## Grade level

| Level     | Description                                  | Approx. grade   |
| --------- | -------------------------------------------- | --------------- |
| ARITH     | 2-3 digit add/sub/mul/div                    | late 2nd – 3rd  |
| L3        | single-cycle word problem                    | ~3rd            |
| L4_MIXED  | 2-cycle multi-step word problem (current)    | ~3rd – 4th      |
| L4.5      | 3-cycle multi-step word problem              | ~4th            |
| **GSM8K** | natural-English grade-school math (target)   | **~5th – 7th**  |
| **MATH-500** | competition-style algebra/precalc (Sep 1 target) | **HS / early college** |

## Note on digit spacing

All problems use **digit-spaced tokenization**: "5 8 0" rather than "580".
This makes arithmetic dramatically easier — each digit is a separate token,
so the model can learn carry/borrow as per-position operations instead of
fighting against BPE tokenizers that merge multi-digit numbers into single
subwords.

Big LLMs evaluated on **raw GSM8K** (with default tokenization, no digit
spacing) hit this exact wall: most multi-digit arithmetic errors trace
back to the model never seeing "1 4 1" as three tokens — it sees something
like "14" + "1" or "141" as one token, and arithmetic operations don't
have clean per-position semantics there.

The breathing transformer's bet is that **iterative refinement (looping
4 layers up to 8 times)** combined with **digit-spaced arithmetic + a
curriculum that teaches multi-step structure** generalizes better than
single-pass forward at the same param budget. Whether that holds up on
natural-English GSM8K and MATH-500 is the open question — that's what
September 1 is the deadline for.

## Current best performance (v45 take 3, step 1000)

On the L4_MIXED held-out set (100 examples, multi-cycle digit-spaced):

| Loop count | Accuracy |
| ---------- | -------- |
| A=1        | 96 %     |
| A=4        | 94 %     |
| A=8        | 93 %     |

127M params (4 looped Pythia-410M layers), trained on AMD 7900 XTX
via tinygrad + AM driver (no ROCm, no PyTorch).
