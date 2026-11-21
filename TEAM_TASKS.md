# Team tasks

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

## 2. English QEvasion data (if external data is allowed)
- Load QEvasion from Hugging Face (`ailsntua/QEvasion`) and write `data_prep/qevasion.py`. It outputs `data/qevasion_en.csv` with the columns `interview_question, interview_answer, clarity_label, evasion_label`.
- The label strings must match exactly the 9 evasion and 3 clarity labels in `common.py`. Print a label count table and list any label that doesn't map.
- Deliverable: the script plus label counts. Claude then adds an `--extra_en` training flag.

## 3. LLM classifier (if LLM APIs are allowed and someone has an API key)
- Write `llm/classify.py`. It prompts an LLM with the taxonomy definitions and boundary rules from the task description, the question and the answer, and asks for exactly one of the 9 evasion labels as JSON. Clarity is derived through the mapping in `common.py`.
- Evaluate it on the fixed sample: the fold-0 validation rows from `common.folds(train.evasion_label.values)[0][1]` (656 rows). Report the evasion and clarity macro-F1.
- Try zero-shot first, then few-shot with 2 to 3 examples per class drawn from the other folds only.
- Cap the spend at EUR 10 until results justify more. Run it on test only after it beats or usefully complements the encoder on the sample.
- Output: `runs_llm/<name>.csv` with `datapointID, evasion_label`.

## 4. Error analysis (native speaker, after Kaggle run 1)
- Write `analysis/export_errors.py`. It loads the OOF probabilities from the run-1 `.npz` files in `results/run1/` (see `evaluate.py` for how they're assembled) and writes `analysis/errors.csv` with the question, answer, gold label and predicted label for the OOF errors, plus a 9x9 confusion matrix.
- Read about 100 errors, focusing on the biggest confusions (Explicit vs Implicit, General vs Partial, Dodging vs Deflection). Tag each one as: gold label looks wrong / genuinely ambiguous / model clearly wrong (and why).
- Deliverable: `analysis/notes.md` with the 3 to 5 patterns a feature, a rule or more context could fix (for example, answers opening with "Nu știu..." that should be Claims ignorance).

## 5. Class-bias tuning for macro-F1 (after Kaggle run 1, anyone comfortable with numpy)
- Argmax favours the big classes, while macro-F1 rewards the small ones. Write `tune_bias.py`. It learns one additive log-probability offset per class on the OOF ensemble probabilities (coordinate ascent on macro-F1), separately for evasion (9) and clarity (3).
- To avoid fooling ourselves, fit the offsets on 4 folds and score on the 5th, rotating through all 5. Report the gain over plain argmax.
- If the gain is at least 0.02, the offsets get applied in the final submission.

## Schedule (deadline: Sun 22 Nov, 09:00)
| By | What |
|---|---|
| Sat 13:00 | Task 1 done: rules and reference scores in `notes/` |
| Sat 13:20 (done) | Kaggle run 1 finished. `submissions/submission_v1.csv` written (milestone 1, safe fallback). Predictions and scores in `results/run1/` for tasks 4 and 5 |
| Sat 17:00 | Tasks 2 and 3 done: English data converted, LLM scored on the fold-0 sample. PRs open |
| Sat 18:00 | Tasks 4 and 5 done: error-analysis notes, bias-tuning result |
| Sat 18:30 | Kaggle run 2 starts with the add-ons that look promising (English data, previous-turn context, extra seeds of the best config) |
| Sat ~22:30 | Run 2 finishes. Keep only add-ons that beat run 1's OOF score by at least 0.02. Write `submission_final.csv` (milestone 2) |
| Sat 23:30 | **Hard stop on experiments.** Final file checked and uploaded by Rares |
| Sun 09:00 | Deadline. The night is buffer for a rerun if something breaks |

Rules for stopping:
- An add-on is kept only if it improves the ensemble's OOF macro-F1 (clarity + evasion) by at least 0.02. Smaller gains are within noise: there are only 36 Clarification and 56 Partial examples.
- Whatever isn't ready by 18:30 misses run 2 and is dropped.
- Don't tune against the leaderboard. With 296 test rows, a few rows swing the score.
