# Task 8: how much can the leaderboard tell us?

All numbers come from the run-1 OOF predictions only (`results/run1/`); test data isn't touched.

```
.venv\Scripts\python leaderboard_noise.py
```

**Method:**
- 5,000 bootstrap draws of 296 train rows with the train class mix (stratified by evasion label, with replacement).
- Per-draw quota: Explicit 163, Implicit 25, General 40, Partial 5, Dodging 26, Deflection 16, Declining 13, Claims ignorance 5, Clarification 3.
- Macro-F1 follows sklearn: averaged over the classes present in the draw's gold labels or the predictions.

## 1. Is 41.93 just bad luck? No, if the score is 50 × clarity + 50 × evasion

| v1 on a 296-row draw (50/50 formula) | |
|---|---|
| full OOF score | 52.88 |
| draws: mean ± sd | 52.46 ± 3.53 |
| 5th – 95th percentile | 46.52 – 58.15 |
| 1st percentile / min | 44.33 / 41.39 |
| leaderboard 41.93 | z = −3.0, **3 of 5,000 draws** at or below it |

Sampling noise, including the rare classes, can't explain 41.93 under the 50/50 formula. So either the test set differs from train, or the formula is different. These are the candidates I could check using train only:

| explanation | expected score (mean, 5th–95th) | P(score ≤ 41.93) |
|---|---|---|
| 50 × clarity F1 + 50 × evasion F1 (assumed) | 52.5 (46.5 – 58.1) | 0.000 |
| **100 × evasion macro-F1 only** (subtask 2 alone) | **38.9 (31.1 – 46.2)** | **0.73** |
| 100 × clarity macro-F1 only | 66.2 (60.1 – 71.9) | 0.000 |
| 50/50 but clarity mapped from evasion | 52.3 (45.9 – 58.4) | 0.003 |
| 50 × evasion F1 + 50 × evasion accuracy | 48.5 (43.0 – 53.4) | 0.022 |
| 50/50, test drawn only from Iohannis + Dan + Bolojan | 50.5 (45.1 – 55.8) | 0.005 |
| 50/50, test drawn only from Dan + Bolojan | 49.4 (44.6 – 54.2) | 0.005 |

**The only explanation that fits is a leaderboard that shows subtask 2 (evasion) alone,** or one that weights it heavily. Then 41.93 is an ordinary draw, right in the middle of the expected range. A test set from recent presidents only, where the model is weaker, lowers the 50/50 score by just 2–3 points, which isn't enough.

**Action for Rares:** check on the competition page how the leaderboard number is computed. Is it subtask 2 only, the two subtasks shown separately, or their average? If it's evasion only, nothing is wrong with v1, and the test set looks like train. If it really is the 50/50 sum, then the test labels differ from train in a way we can't see. One possibility is that test is labelled by 3 annotators plus an expert, while each train row has a single annotator. In that case, OOF scores still rank models, but their absolute level is about 11 points too optimistic.

## 2. What leaderboard difference is real?
These are paired differences: both ensembles are scored on the same 296-row draws, other − v1, in leaderboard points. "Agree" is the share of train rows where the two give the same evasion label.

| other ensemble | agree | 50/50: full-OOF diff | 50/50: sd | 50/50: 95% range | evasion only: sd | evasion only: 95% range |
|---|---|---|---|---|---|---|
| all 4 run-1 configs | 92.6% | −0.32 | 1.38 | −3.3 … +2.2 | 2.17 | −5.2 … +3.9 |
| greedy (XLM-R ×3 + RoBERT + RO-BERT) | 89.4% | +0.81 | 1.84 | −2.4 … +4.8 | 2.99 | −4.5 … +7.4 |
| XLM-R large alone | 79.3% | −0.47 | 2.79 | −6.0 … +5.2 | 4.03 | −9.5 … +6.5 |
| RO-BERT base alone | 79.7% | −2.82 | 2.50 | −7.8 … +2.0 | 3.68 | −10.7 … +3.8 |

The noise depends mostly on how many test predictions two submissions share, not on their OOF gap. Agreement on test matches agreement on OOF: the run-1 greedy ensemble agrees with v1 on 90.2% of test evasion labels, against 89.4% out-of-fold. So measure agreement first:
```
.venv\Scripts\python leaderboard_noise.py --compare submissions/v1/submission_v1.csv submissions/v2/submission_v2.csv
```
This compares only the predicted labels in our two files; no test answers are involved.

### Decision rule for picking the finals
A leaderboard gap between v2 and v1 counts as real only if it's bigger than the band for their agreement. The bands are about 2 sd, rounded up:

| v2 agrees with v1 on test evasion labels | 50/50 formula | evasion-only formula |
|---|---|---|
| ≥ 88% | **±4 points** | **±6 points** |
| ~80% | ±6 points | ±9 points |
| < 75% | use the OOF difference; the leaderboard says little | same |

- **A v2 that is more than the band *below* v1 on the leaderboard is a real drop.** Don't pick it, even if its OOF score is higher.
- **Anything inside the band is noise.** Pick by nested OOF score (task 7's ≥ 0.02 rule), not by which leaderboard number is higher.
- **A v2 more than the band *above* v1 is a real gain,** but don't tune further against the leaderboard.
- With a v1 / v2 pair as finals, choose two that **disagree enough to hedge** (around 80–90% agreement), not two near-copies. Two near-copies score within noise of each other.

## 3. Which classes drive the variance
v1's evasion F1 per class across the draws. "sd in points" is the class's share of the 50/50 score, sd × 50 / 9.

| class | rows per draw | mean F1 | sd F1 | sd in points | F1 = 0 in draws |
|---|---|---|---|---|---|
| Explicit | 163 | .768 | .023 | 0.13 | 0.0% |
| Implicit | 25 | .261 | .080 | 0.44 | 0.1% |
| General | 40 | .327 | .066 | 0.37 | 0.0% |
| Partial/half-answer | 5 | .000 | .000 | 0.00 | 100% (v1 never gets one right) |
| Dodging | 26 | .385 | .081 | 0.45 | 0.0% |
| Deflection | 16 | .133 | .088 | 0.49 | 13.8% |
| Declining to answer | 13 | .582 | .103 | 0.57 | 0.0% |
| Claims ignorance | 5 | .522 | .181 | 1.00 | 1.2% |
| **Clarification** | **3** | .517 | .277 | **1.54** | **14.5%** |

- **The three rarest classes dominate the noise.** With 3 rows per draw, a single Clarification hit or miss moves the score by about 1.5 points on its own, and Claims ignorance by about 1.
- **Explicit**, with 163 rows, contributes almost nothing.
- **Partial** has zero variance only because v1 never predicts it correctly. Its F1 is 0 in every draw, and it still counts as one of the nine classes. It has the most room to gain: a Partial F1 of 0.3 would add about 1.7 points (0.3 × 50 / 9). With only 5 rows per draw, that gain would itself be noisy.
