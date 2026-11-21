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

## Who does what (updated 13:30 after task 1)
- **TreiPaDus:** task 1 done (PR #1, rule answers in issue #2). Now on **task 5**.
- **Colleague 2:** task 4 on arrival.
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
| Sat ~14:00 | Kaggle run 2 starts with the paper's training settings (10+ epochs, 512 tokens, lr 2e-5 for RoBERT-large). Takes about 4 to 5 h |
| Sat 18:00 | Tasks 4 and 5 done: error-analysis notes, bias-tuning result |
| Sat ~18:30 | Run 2 finishes. Keep only changes that beat run 1's OOF score by at least 0.02. Write `submission_final.csv` (milestone 2) |
| Sat 23:30 | **Hard stop on experiments.** Rares uploads the final files and picks the **2 final submissions** (budget: 30 submissions in total) |
| Sun 09:00 | Deadline, Romania time (EET). The night is buffer for a rerun if something breaks |

Rules for stopping:
- An add-on is kept only if it improves the ensemble's OOF macro-F1 (clarity + evasion) by at least 0.02. Smaller gains are within noise: there are only 36 Clarification and 56 Partial examples.
- Whatever isn't ready by its cut-off is dropped.
- Don't tune against the leaderboard. With 296 test rows, a few rows swing the score.
