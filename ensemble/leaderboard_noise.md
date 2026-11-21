# Task 8: how much can the leaderboard tell us?

All numbers come from the run-1 OOF predictions only (`results/run1/`); test data isn't touched. The only exception is section 5, which reads the test set's `president`, `date` and `context` metadata, never its text.

> **Update (task 9):** the visible leaderboard is scored on a fixed **subset** of the 296 test rows. Full scores come only after the deadline, and only for the 2 finals. The subset size isn't published, so sections 4 and 5 redo the numbers for 148 rows (half) and 100 rows (pessimistic). **Use the decision table in section 4, not the 296-row one below.**

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
| ≥ 88% | ±4 points | ±6 points |
| ~80% | ±6 points | ±9 points |
| < 75% | use the OOF difference; the leaderboard says little | same |

*This table assumes all 296 rows are scored. For the real, partial leaderboard, see section 4.*

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

## 4. The visible leaderboard is a subset: wider bands
```
.venv\Scripts\python leaderboard_noise.py --n 148
.venv\Scripts\python leaderboard_noise.py --n 100
```

### Is 41.93 bad luck on a subset?
| rows scored | v1 50/50: mean ± sd | 5th – 95th | P(≤ 41.93), 50/50 | P(≤ 41.93), evasion only |
|---|---|---|---|---|
| 296 | 52.5 ± 3.5 | 46.5 – 58.2 | 0.0006 | 0.73 |
| 148 | 52.6 ± 4.7 | 44.7 – 60.2 | **0.015** | 0.67 |
| 100 | 52.2 ± 5.9 | 42.2 – 61.7 | **0.046** | 0.67 |

The subset makes the 50/50 formula less implausible, but on a half-size subset it's still a 1-in-70 result, and 1-in-22 on 100 rows. Evasion-only stays an ordinary draw at every size. Rows drawn to match the test's metadata mix (section 5) instead of the class mix give P(≤ 41.93) of 0.03–0.04 at 148 rows and 0.07–0.08 at 100. Those draws don't fix the class quota, so they're noisier, and either way it's unlikely but not rare.

### Decision rule for the partial leaderboard (use this one)
The bands are about 2 sd of the paired v2 − v1 difference, rounded up, from the same draws.

| v2 agrees with v1 on test evasion labels | 148 rows, 50/50 | 148 rows, evasion only | 100 rows, 50/50 | 100 rows, evasion only |
|---|---|---|---|---|
| ≥ 88% | **±5 points** | **±8 points** | ±7 points | ±11 points |
| ~80% | ±8 points | ±11 points | ±10 points | ±14 points |

Paired-difference sd behind the table:

| other − v1 | agree | 148: 50/50 | 148: evasion | 100: 50/50 | 100: evasion |
|---|---|---|---|---|---|
| all 4 run-1 configs | 92.6% | 1.87 | 2.97 | 2.25 | 3.59 |
| greedy (XLM-R ×3 + RoBERT + RO-BERT) | 89.4% | 2.50 | 4.00 | 3.12 | 5.23 |
| XLM-R large alone | 79.3% | 3.81 | 5.37 | 4.68 | 6.90 |
| RO-BERT base alone | 79.7% | 3.31 | 4.77 | 4.08 | 5.91 |

**For picking the 2 finals tonight:**
- **The visible leaderboard can't tell apart submissions that agree on ~90% of rows** unless they differ by 5–11 points, and real ensemble improvements are 1–3 points. So **choose by nested OOF score** (task 7). Use the leaderboard only to veto a v2 that falls further below v1 than the band.
- **Hedge:** pick two finals that differ, about 80–90% agreement. Only the finals get full 296-row scores, and those are less noisy than what we see now.

### Rare classes on a subset
On 148 rows a draw has 2 Clarification, 3 Claims ignorance and 3 Partial rows:
- **Clarification:** F1 = 0 in 27.5% of draws, and it moves the score by about 2 points on its own.
- **Deflection:** F1 = 0 in 38.7% of draws.

On 100 rows it's 1 Clarification row: F1 = 0 in 52.8% of draws, about 2.6 points either way.

## 5. Test-set makeup (metadata only: `president`, `date`, `context`)
Only these three columns of `test_data.csv` were read; no question or answer text.

| president | train % | test % (n) | v1 OOF evasion accuracy |
|---|---|---|---|
| Traian Băsescu | 34.2 | 31.8 (94) | 0.638 |
| Klaus Iohannis | 30.4 | **34.8** (103) | 0.522 |
| Ion Iliescu | 23.1 | 20.6 (61) | 0.635 |
| Nicușor Dan | 10.9 | 10.5 (31) | 0.478 |
| Ilie Bolojan | 1.5 | 2.4 (7) | 0.327 |

| context | train % | test % (n) | v1 OOF evasion accuracy |
|---|---|---|---|
| Interview | 53.7 | 48.6 (144) | 0.638 |
| Declaration | 33.6 | **37.8** (112) | 0.524 |
| Press Conference | 12.7 | 13.5 (40) | 0.480 |

| years | train % | test % (n) |
|---|---|---|
| 2001–2004 | 23.1 | 20.6 (61) |
| 2005–2009 | 33.4 | 31.1 (92) |
| 2010–2014 | 0.8 | 0.7 (2) |
| 2015–2019 | 17.9 | 16.6 (49) |
| 2020–2024 | 12.5 | **17.9** (53) |
| 2025–2026 | 12.4 | 13.2 (39) |

Train dates come in two formats: `YYYY-MM-DD` up to Băsescu, `DD.MM.YYYY` from Iohannis on. Both parse.

**Test leans slightly towards the harder slices:**
- 47.7% of test rows are from 2015 or later, against 42.8% in train;
- Iohannis's second term is over-represented;
- Declarations make up 37.8% of test, against 33.6% in train.

**How much of the gap the mix explains.** Expected v1 50/50 score on train rows drawn to match the test's **president × context** mix (12 cells), against the train mix, 3,000 draws each:

| rows scored | train mix | test president × context mix | test year-range × context mix |
|---|---|---|---|
| 296 | 52.3 | 52.0 (−0.3) | 51.8 (−0.3) |
| 148 | 51.7 | 51.4 (−0.3) | 51.0 (−0.6) |
| 100 | 51.2 | 50.8 (−0.5) | 50.5 (−0.6) |

The test mix costs about **0.3–0.6 points** of the 11-point gap between OOF (about 52.9) and the leaderboard (41.93), so about 3–5% of it. **The metadata doesn't explain the gap.** The likely explanations, in order:
1. **The leaderboard shows evasion only** (fits at every subset size, section 4);
2. **The test labels differ from train:** test was labelled by 3 annotators plus expert adjudication, while train rows had a single annotator;
3. **a bad draw of a small subset:** 1.5–5% likely under 50/50.
