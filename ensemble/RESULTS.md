# Task 7: ensemble selection

## Run 1 only
```
.venv\Scripts\python select_ensemble.py --runs results/run1
```
Out-of-fold macro-F1 over all 3,278 train rows.

| ensemble | clarity | evasion | sum |
|---|---|---|---|
| v1 trio (submission v1) | 0.6626 | 0.3950 | 1.0576 |
| all 4 configs averaged | 0.6553 | 0.3958 | 1.0511 |
| greedy selection, in-sample | 0.6733 | 0.4004 | 1.0737 |
| **greedy selection, nested** | 0.6542 | 0.3931 | **1.0473** |
| run1/xlm-roberta-large_p0 | 0.6717 | 0.3764 | 1.0481 |
| run1/bert-base-romanian-cased-v1_p0 | 0.6346 | 0.3666 | 1.0012 |
| run1/RoBERT-large_p0 | 0.6308 | 0.3607 | 0.9916 |
| run1/bert-base-romanian-cased-v1_p32 | 0.6146 | 0.3377 | 0.9524 |

- **The in-sample bag** is XLM-R ×3, RoBERT-large ×1 and Romanian BERT ×1. It beats v1 by +0.016, but in-sample selection overstates by **+0.027**.
- **Nested, greedy selection is −0.010 against v1.** The 5 nested folds pick 2 different bags (XLM-R + RoBERT, or the trio with XLM-R weighted 2–3×).
- **Verdict for run 1:** no new submission; v1 stays.
- **Caveat:** v1 itself was picked by looking at run-1 out-of-fold scores (the p32 config was dropped), so it's not a nested number either. The comparison slightly favours v1.

## Run 1, evasion-first objective (Rares's rule, now the default)
`select_ensemble.py` now maximises OOF **evasion** macro-F1 while clarity stays within 0.01 of v1 (`--objective evasion --max_clarity_drop 0.01`). `--objective sum` gives the table above.

| ensemble | clarity | evasion | sum |
|---|---|---|---|
| v1 trio | 0.6626 | 0.3950 | 1.0576 |
| greedy, in-sample (XLM-R ×2 + RoBERT + RO-BERT) | 0.6688 | 0.4041 | 1.0729 |
| **greedy, nested** | 0.6490 | **0.3806** | 1.0296 |

- **Nested against v1:** evasion −0.014, clarity −0.013. The bar (nested evasion +0.02 with clarity drop ≤ 0.01) is **not met**, so v1 stays.
- **It overfits more than the sum objective:** in-sample overstates the nested score by +0.043, against +0.027. Evasion macro-F1 depends on a handful of rare-class rows, so the 5 nested folds pick 4 different bags.

## Run 2
To be filled when `results/run2/` lands:
```
.venv\Scripts\python select_ensemble.py --runs results/run1 results/run2
.venv\Scripts\python stack.py --runs results/run1 results/run2
```
