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

## Runs 1 + 2 (8 configs)
```
.venv\Scripts\python select_ensemble.py --runs results/run1 results/run2
.venv\Scripts\python stack.py --runs results/run1 results/run2 --C 0.1 1
```

| ensemble | clarity | evasion | sum |
|---|---|---|---|
| v1 trio | 0.6626 | 0.3950 | 1.0576 |
| all 8 averaged | 0.6634 | 0.3942 | 1.0576 |
| greedy, in-sample (XLM-R run 1 ×3 + XLM-R run 2) | 0.6743 | 0.4045 | 1.0787 |
| **greedy, nested** | 0.6448 | **0.3744** | 1.0192 |
| stack, class_weight None, C=0.1 (nested) | 0.6681 | 0.3775 | 1.0456 |
| **R1: all large lr 1e-5 models, fixed (4)** | **0.6742** | **0.3954** | **1.0696** |
| R2: R1 + v1's RO-BERT base (5) | 0.6740 | 0.3963 | 1.0703 |
| R4: XLM-R large only (2) | 0.6690 | 0.3899 | 1.0589 |

- **Greedy selection overfits badly now.** In-sample overstates the nested score by **+0.060**, and the 5 nested folds pick 5 unrelated bags. Nested it's −0.021 evasion against v1.
- **Stacking still loses on evasion:** −0.017 against the plain average in its best setting.
- **The bar** (nested evasion +0.02 with clarity drop ≤ 0.01) is not met. **v1 stays.**

### Fixed recipes: the plan for runs 3–6
Learned selection can't beat the noise with 3,278 rows. The recipes in `select_ensemble.py` (`RECIPES`) are fixed equal-weight averages, defined now, before runs 3–6 land. Configs from later runs join automatically by name pattern, and with no selection step their OOF score needs no nesting.
- **R1 = every XLM-R large and RoBERT-large model at lr 1e-5, any run or seed.** Today that's 4 models: evasion level with v1 (+0.0004) and **clarity +0.012**. Runs 4, 5 and 6 add seeds 43, 44 and 45 of exactly these, up to 9 models. More seeds is the one lever Rares's run-2 notes call reliable.
- **On test, R1 and v1 agree** on 85.1% of evasion and 91.2% of clarity labels (`leaderboard_noise.py --compare`). That's the 80–90% range task 8 suggested for a hedged pair of finals.

**Proposed rule (Rares decides):**
- when runs 4–6 are in, score R1 again;
- if R1's evasion ≥ v1 + 0.02 with clarity drop ≤ 0.01, **v2 = R1**;
- if not, keep v1 as the main final and consider R1 as the second, hedging final. It's at least as good on OOF, it's a different set of models, and the visible leaderboard can't tell them apart anyway (task 9).

To write it:
```
.venv\Scripts\python select_ensemble.py --runs results/run1 results/run2 results/run4 results/run5 results/run6 --submission submissions/v2/submission_v2.csv --submit_recipe "R1 all large lr1e-5 models"
.venv\Scripts\python make_submission_py.py submissions/v2/submission_v2.csv
```
