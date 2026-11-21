# Leaderboard log

Partial leaderboard (a subset of the 296 test rows). The "final grade" the platform shows (v1 41.93, best 48.48) is a linear rescale of this score: about 1.98 grade points per score point, with grade 0 at about 35.2 and grade 100 at about 85.6. That's inferred from two points.

| Submission | What | Out-of-fold (OOF) clarity / evasion | Partial score |
|---|---|---|---|
| v1 | run-1 trio, plain average | 0.663 / 0.395 | 56.33 |
| v2 | run-1 + run-2 trios, geometric mean | 0.676 / 0.403 | 58.49 |
| v3 | probe: v1's evasion + v2's clarity | 0.676 / 0.395 | 59.63 |
| v4 | recipe V4: v2's six + run 4's seed-43 pair (8 models), geometric mean | 0.680 / 0.406 | 58.95 |
| v5 | recipe R3: every large lr 1e-5 model (runs 1-2, 4, 6) + run 3's head+tail pair (9 models), geometric mean | 0.685 / 0.405 | 57.48 |

**What the probe shows:** v3's evasion answers are identical to v1's. Only 41 clarity answers changed, and the score rose by 3.30. So **clarity counts.** A 50/50 formula (score = 50 × clarity F1 + 50 × evasion F1) fits: v1's OOF predicts 52.9, and task 8's band for the partial board was about 46–59. If clarity counts equally, the probe's +3.30 means +0.066 clarity F1 on the scored rows, and v2 against v3 means −0.023 evasion F1. Both are inside the subset noise.

**Consequences:**
- Choose ensembles on **clarity + evasion** (`--objective sum`), not evasion first.
- Clarity and evasion can come from **different ensembles**. Each subtask's rows are scored separately, which is exactly what v3 did.
- v2 vs v3 (−1.14) is inside the noise. On OOF, v2 is higher (1.079 against 1.071).
