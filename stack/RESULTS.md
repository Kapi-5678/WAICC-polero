# Task 6: stacking result (run 1)

**Verdict: don't use it.** A logistic-regression stacker loses to the plain probability average on evasion in every setting, and gains at most +0.006 on clarity.

Command (repo root):
```
.venv\Scripts\python stack.py --C 0.03 0.1 0.3 1 3
```

Setup:
- **Features:** the concatenated OOF probabilities (9 evasion + 3 clarity) of the three `submissions/v1` configs, at their best epochs (36 features), standardised.
- **Model:** one multinomial logistic regression for evasion and one for clarity.
- **Scoring:** nested, i.e. fitted on 4 folds' OOF rows (`common.folds`) and scored on the 5th, rotating through all 5.
- **Baseline:** `evaluate.py`'s plain average, 0.3950 evasion and 0.6626 clarity.

| class_weight | C | Evasion | Δ | Clarity | Δ |
|---|---|---|---|---|---|
| None | 0.03 | 0.3569 | −0.038 | 0.6638 | +0.001 |
| None | 0.1 | 0.3585 | −0.037 | 0.6650 | +0.002 |
| None | 0.3 | 0.3628 | −0.032 | 0.6650 | +0.002 |
| None | 1 | 0.3623 | −0.033 | 0.6651 | +0.003 |
| None | 3 | 0.3620 | −0.033 | 0.6682 | +0.006 |
| balanced | 0.03 | 0.3536 | −0.041 | 0.6388 | −0.024 |
| balanced | 0.1 | 0.3459 | −0.049 | 0.6423 | −0.020 |
| balanced | 1 | 0.3358 | −0.059 | 0.6386 | −0.024 |

Per-fold evasion gain for None, C=1: −0.032 ± 0.033. Log-probability features (`--features logprobs`) are worse still: −0.042 to −0.080 on evasion.

## Why it loses
The stacker does what it optimises for. It's **more accurate** (0.605 vs 0.580) and has **lower log-loss** (1.199 vs 1.219). But a calibrated model leans towards the big classes. It predicts Explicit 2,386 times instead of 1,866, and Deflection only 22 times instead of 130. That costs macro-F1, where each of the 9 classes counts the same.

| | accuracy | macro-F1 | Implicit | General | Partial | Deflection | Clarification |
|---|---|---|---|---|---|---|---|
| Plain average | 0.580 | 0.395 | .26 | .33 | .00 | .14 | .56 |
| Stack | 0.605 | 0.362 | .20 | .23 | .07 | .04 | .48 |
| Stack, balanced | 0.471 | 0.336 | .23 | .29 | .08 | .14 | .27 |

`class_weight=balanced` overshoots the other way: it predicts Partial 353 times for 56 real cases.

I also tried **stacking + task 5's per-class offsets**, double-nested: offsets fitted on inner-CV stacker probabilities within each outer training split. The best result is still −0.004 on evasion (C=0.1), so no improvement there either.

The plain average of the fine-tuned models is already a good combination for macro-F1. With three similar encoders, the stacker has little to learn beyond re-weighting the classes, and that's where it does harm.

## Run 2
Re-run once `results/run2/` lands:
```
.venv\Scripts\python stack.py --runs results/run2
.venv\Scripts\python stack.py --runs results/run1 results/run2
```
With more than one run folder, or any folder other than run 1, it uses every complete 5-fold config it finds. Configs are named `<run folder>/<config>`; pick a subset with `--use`. If a setting beats the plain average by at least 0.02, write the submission in the new layout:
```
.venv\Scripts\python stack.py --runs results/run1 results/run2 --submission submissions/v3/submission_v3.csv
```
Check the test prediction counts it prints. On run 1 the stacker predicted no Partial and no Deflection on test.
