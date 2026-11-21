# Team tasks

## Team
| Role | Who | GitHub |
|---|---|---|
| Team leader | Rares | Kapi-5678 |
| Colleague 1 | TreiPaDus | TreiPaDus |
| Colleague 2 | notneagu | notneagu |

Rares merges every PR and makes every competition submission. Questions go in a GitHub issue. Rares answers them.

Main track (Rares + Claude): 5-fold fine-tuning of XLM-R large, Romanian BERT and RoBERT-large on Kaggle GPU, then an ensemble. `evaluate.py` scores everything on out-of-fold (OOF) predictions over the full train set. That OOF score is the yardstick every idea below is judged by.

## Ground rules
- Work on your own branch (`<name>/<task>`) and only in your task's folder. Open a PR, Rares merges.
- Data CSVs are not in git. Download them from the competition page into `data/`.
- Never read or hand-label `test_data.csv` answers to make predictions. Test is only for the final model run.
- Report every result as a 5-fold OOF macro-F1 (or on the fixed sample in task 3), never as a score on the test set.

## 1. Rules and reference scores (first, ~30 min, anyone)
From the competition page, write down in `notes/rules.md`:
- the evaluation metric (macro-F1? accuracy?), the deadline and the submission limit;
- whether external datasets (the English QEvasion corpus) are allowed;
- whether external LLM APIs, or only open models, are allowed.

From the PolERo paper (EMNLP 2026, 2026.emnlp-main.2688), copy the best reported clarity and evasion scores per method (TF-IDF, Romanian encoders, multi-head sliding window, EN+RO joint training, LLM zero/few-shot) into `notes/reference_scores.md`. These scores tell us what "good" looks like.

Tasks 2 and 3 depend on these answers.

## Who does what (updated 14:10)
- **Colleague 1 (TreiPaDus):** tasks 1 and 5 done. Now: **task 4a** (error export, ~30 min), then **task 6** (stacking).
- **Colleague 2:** task 4b (reading the errors) on arrival. Task 4a's files will be ready for them.
- **Tasks 2 and 3 are dropped.** Per `notes/rules.md`, the rules allow neither external data nor LLM APIs.
- **Never download or open the public PolERo release** (`github.com/gabriel-stefan/polero`). It most likely contains our test labels.

## 2. English QEvasion data (DROPPED: no external data)
- Load QEvasion from Hugging Face (`ailsntua/QEvasion`) and write `data_prep/qevasion.py`. It outputs `data/qevasion_en.csv` with the columns `interview_question, interview_answer, clarity_label, evasion_label`.
- The label strings must match exactly the 9 evasion and 3 clarity labels in `common.py`. Print a label count table and list any label that doesn't map.
- Deliverable: the script plus label counts. Claude then adds an `--extra_en` training flag.

## 3b. Open-weight LLM on Kaggle (OFF for now, Rares' decision)
- Same as task 3 below, but with an open-weight model running on a Kaggle GPU, with no API calls. In the paper, Qwen3.6-35B-A3B few-shot got the best evasion score (.682 vs .545 for the best encoder). On T4 x2 it needs 4-bit quantisation, so try the smaller Gemma or Qwen variants if that doesn't fit.
- Score it on the fold-0 sample first, and run test only if it helps. Hard cut-off: no usable sample score by 20:00 means it's dropped.

## 3. LLM classifier via API (DROPPED: no LLM APIs)
- Write `llm/classify.py`. It prompts an LLM with the taxonomy definitions and boundary rules from the task description, the question and the answer, and asks for exactly one of the 9 evasion labels as JSON. Clarity is derived through the mapping in `common.py`.
- Evaluate it on the fixed sample: the fold-0 validation rows from `common.folds(train.evasion_label.values)[0][1]` (656 rows). Report the evasion and clarity macro-F1.
- Try zero-shot first, then few-shot with 2 to 3 examples per class drawn from the other folds only.
- Cap the spend at EUR 10 until results justify more. Run it on test only after it beats or usefully complements the encoder on the sample.
- Output: `runs_llm/<name>.csv` with `datapointID, evasion_label`.

## 4a. Error export (Colleague 1, now, ~30 min)
- Write `analysis/export_errors.py`. It loads the OOF probabilities of the run-1 ensemble (xlm-roberta-large_p0, RoBERT-large_p0, bert-base-romanian-cased-v1_p0) from `results/run1/` (see `evaluate.py` for how they're assembled) and writes:
  - `analysis/errors.csv`: one row per OOF evasion error, with datapointID, question, answer, gold label, predicted label and the predicted probability. Sort it so the focus classes (Partial/half-answer, Deflection, Implicit, General) come first. It's git-ignored because it contains the competition text. Colleague 2 runs the script locally.
  - `analysis/confusion.md`: the 9x9 OOF confusion matrix and per-class precision, recall and F1. This one gets committed.
- Open a PR. Once it's merged, colleague 2 can start reading straight away.

## 4b. Reading the errors (Colleague 2, native speaker, on arrival)
- Run `analysis/export_errors.py`, then read about 100 errors, focusing on Partial/half-answer, Deflection and Implicit. Task 5 showed these need better models, not thresholds. Tag each one as: gold label looks wrong / genuinely ambiguous / model clearly wrong (and why).
- Deliverable: `analysis/notes.md` with the 3 to 5 patterns a feature, a rule or more context could fix (for example, answers opening with "Nu știu..." that should be Claims ignorance).

## 6. Stacking (Colleague 1, after 4a)
- Write `stack.py`. Instead of averaging the models' probabilities, learn the combination: a multinomial logistic regression whose features are the concatenated OOF probabilities (9 evasion + 3 clarity per model) of the chosen configs. Fit it separately for evasion and clarity.
- Same honesty rule as task 5: fit the stacker on 4 folds' OOF rows and score on the 5th, rotating through all 5. Compare with the plain average from `evaluate.py`. Try `class_weight` None vs balanced.
- Run it on run 1 now. When run 2 lands in `results/run2/` (~18:30), re-run it on the best run-2 configs and on run 1 + run 2 together.
- If it beats the plain average by at least 0.02, add `--submission <path>`, which writes the test predictions as `submissions/<vN>/submission_<vN>.csv`.

## 5. Class-bias tuning for macro-F1 (after Kaggle run 1, anyone comfortable with numpy)
- Argmax favours the big classes, while macro-F1 rewards the small ones. Write `tune_bias.py`. It learns one additive log-probability offset per class on the OOF ensemble probabilities (coordinate ascent on macro-F1), separately for evasion (9) and clarity (3).
- To avoid fooling ourselves, fit the offsets on 4 folds and score on the 5th, rotating through all 5. Report the gain over plain argmax.
- If the gain is at least 0.02, the offsets get applied in the final submission.

## 7. Ensemble selection and the milestone-2 submission (Colleague 1, now until the end)
This is the final model choice, and it's Colleague 1's to own. Claude handles the Kaggle runs and commits their outputs to `results/runN/`. Rares reviews and picks the 2 finals.

**Now, until run 2 lands (~17:15): build `select_ensemble.py`**
- Inputs: one or more run folders (`--runs results/run1 results/run2 ...`). Take every complete 5-fold config at its best epoch, chosen as in `evaluate.py`. Config names need the run as a prefix (`run1/...`), as `stack.py` already does.
- Greedy forward selection with replacement (Caruana et al.): start empty, then repeatedly add the config whose inclusion most improves OOF clarity + evasion macro-F1 of the probability average. Stop when nothing improves. Clarity is chosen as in `evaluate.py` (best of combined / mapped / head).
- Honesty check: also run the selection nested (select on 4 folds' OOF rows, score on the 5th). Report how much the in-sample score overstates it.
- Print a comparison table: v1 trio, all configs averaged, greedy selection (in-sample and nested), and each single config.
- `--submission submissions/vN/submission_vN.csv` writes the chosen ensemble's test predictions in the same row order as `data/sample_output.csv`, with the same checks as `evaluate.py`. Then `python make_submission_py.py submissions/vN/submission_vN.csv` writes the matching input `.py`.
- Test it on `results/run1` now. It should reproduce or beat v1's 0.663 / 0.395.

**When run 2 lands in `results/run2/` (~17:15)**
1. Write `results/run2/SCORES.md`, the per-config table in the same format as `results/run1/SCORES.md`.
2. Run `select_ensemble.py --runs results/run1 results/run2`. Also re-run `stack.py` on the same set, as task 6 says.
3. If the best ensemble beats v1 by at least 0.02 (clarity + evasion, nested), produce `submissions/v2/` (csv + py) and open a PR. If not, say so: v1 stays the final.

**When run 3 lands in `results/run3/` (~19:30):** repeat with all three runs, which produces `submissions/v3/` if it wins.

## 8. How much can the leaderboard tell us? (Colleague 1, now until run 2 lands)
v1 scored **41.93 / 100** on the leaderboard (296 test rows), against about 53 expected from OOF if the score is 50 × clarity macro-F1 + 50 × evasion macro-F1. Two questions, both answered from the v1 OOF predictions in `results/run1/` alone (never look at test):
1. **Is 41.9 just bad luck?** Draw many random 296-row samples from the OOF predictions with the train class mix (bootstrap, at least 2,000 draws) and compute 50 × clarity F1 + 50 × evasion F1 for each. Report the mean and the 5th–95th percentile. If 41.9 is far below the 5th percentile, the test set really differs from train (for example, its 3-annotator labels), and that's not just rare-class noise.
2. **What leaderboard difference is real?** Using the same draws, score v1 and the run-1 greedy ensemble (or any second ensemble) on the *same* 296 rows, and report the spread of their score *difference*. That gives a rule like "a v2 that is more than X points below v1 on the leaderboard is a real drop, otherwise it's noise". Rares uses that rule to pick the 2 finals tonight.
- Also report which classes drive the variance: with only a handful of Partial or Clarification rows in a 296-row draw, how often is a class's F1 exactly 0?
- Deliverable: `ensemble/leaderboard_noise.md` with the two numbers, the decision rule, and the script (`leaderboard_noise.py`). PR before run 2 lands, ~17:00.
- Follow-up: the visible leaderboard is *partial* (a subset of test), so re-run with `--n 148` and `--n 100` and add the wider bands.

## 9. Before run 2 lands (Colleague 1, ~15:30–17:00)
1. **Test-set makeup, from metadata only:** compare test vs train by `president`, year (from `date`) and `context`, using only those columns and never the question or answer text. Combine it with the per-president OOF accuracy from `analysis/notes.md` to estimate how much of the leaderboard gap the test mix explains. Add it to `ensemble/leaderboard_noise.md`.
2. **Independent bug review of the pipeline:** read `finetune.py`, `evaluate.py`, `select_ensemble.py` and `make_submission_py.py`, looking for anything that could make a submission wrong. Examples: test rows misaligned with IDs, the wrong epoch's test probabilities, clarity and evasion mixed up, folds not matching across runs, the `--q_mode` / `--president` flags changing the default behaviour. Report findings in a new issue, each with file:line and how to reproduce it. **Don't edit those files.** Rares decides on fixes. "Found nothing" is a fine result.
3. At ~17:00, switch to task 7 as soon as `results/run2/` is committed.

## 10. Run 4 on a second Kaggle GPU (Colleague 2, now, ~2.5 h unattended)
Rares' Kaggle account runs one GPU session at a time (run 2, then run 3), so run 4 goes on Colleague 2's own account.
1. On kaggle.com: verify your phone number (Settings > Phone verification). Without it there is no GPU and no internet.
2. Send Rares your Kaggle username in issue #2. He shares the private dataset `lucazanotti/polero-data` with you.
3. Create a new notebook. **Add Input** > the `polero-data` dataset. Session options: **Accelerator GPU T4 x2**, **Internet on**.
4. Paste all of `kaggle/run.py` from main (it already says `RUN = "run4"`) into one code cell. Click **Save Version** > **Save & Run All (Commit)**. Close the tab: it runs on its own, about 2.5 hours, 10 jobs.
5. When the version is done, open it > **Output**, and download `runs/` (10 `.npz` files plus logs). Check each log line says `ok`.
6. On branch `notneagu/run4`, put the `.npz` files in `results/run4/` and the notebook log as `results/run4/kaggle.log`. Open a PR. Don't commit anything from `/kaggle/input` or any CSV.
- If anything fails (a job says FAILED, out of memory, no GPU), post the log line in issue #2 instead of retrying blindly.

## Schedule (deadline: Sun 22 Nov, 09:00)
| By | What |
|---|---|
| Sat 13:00 | Task 1 done: rules and reference scores in `notes/` |
| Sat 13:20 (done) | Kaggle run 1 finished. `submissions/v1/` written (milestone 1, safe fallback). Predictions and scores in `results/run1/` for tasks 4 and 5 |
| Sat ~14:00 | Kaggle run 2 starts with the paper's training settings (10+ epochs, 512 tokens, lr 2e-5 for RoBERT-large). Takes about 4 to 5 h |
| Sat 18:00 | Tasks 4 and 5 done: error-analysis notes, bias-tuning result |
| Sat ~18:30 | Run 2 finishes. Keep only changes that beat run 1's OOF score by at least 0.02. Write `submissions/v2/` (milestone 2) |
| Sat 23:30 | **Hard stop on experiments.** Rares uploads the final files and picks the **2 final submissions** (budget: 30 submissions in total) |
| Sun 09:00 | Deadline, Romania time (EET). The night is buffer for a rerun if something breaks |

Rules for stopping:
- An add-on is kept only if it improves the ensemble's OOF macro-F1 (clarity + evasion) by at least 0.02. Smaller gains are within noise: there are only 36 Clarification and 56 Partial examples.
- Whatever isn't ready by its cut-off is dropped.
- Don't tune against the leaderboard. With 296 test rows, a few rows swing the score.
