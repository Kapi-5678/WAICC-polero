# Competition rules

**Status:** external data and LLM APIs are settled (both **not allowed**, confirmed by the team). The metric, deadline and submission limit are still marked **TODO**: I couldn't find the competition page with a web search, so whoever has the link should check them. Everything else comes from our repo and the PolERo paper, with the source noted.

## Evaluation metric
- **TODO: confirm on the competition page.**
- Likely **macro-F1**, scored separately for Subtask 1 (clarity, 3 classes) and Subtask 2 (evasion, 9 classes). That's the primary metric in the PolERo paper (Section 3.1) and in SemEval-2026 Task 6, which this task copies.
- Submission format, as `common.write_submission` writes it (check it against `sample_output.csv`): one CSV with columns `subtaskID, datapointID, answer`. Rows with `subtaskID=1` hold the clarity label and rows with `subtaskID=2` the evasion label, for each of the 296 test rows.
- **TODO:** is the leaderboard score the average of the two subtasks, or are they ranked separately? Is the public leaderboard computed on a subset of test?

## Deadline
- **TODO: confirm on the competition page**, including the time zone.
- Our working assumption (`TEAM_TASKS.md`): **Sun 22 Nov 2026, 09:00**, with experiments stopping Sat 21 Nov 23:30.

## Submission limit
- **TODO:** submissions per day, the total limit, and how many final submissions can be selected.

## External data (English QEvasion corpus)
- **Not allowed** (confirmed by the team, 21 Nov).
- So **task 2 is dropped.** No QEvasion, no EN+RO joint training (the paper's best fine-tuned evasion setup, .570), and no machine-translated English data (EN2RO).
- Pretrained model weights (XLM-R, RoBERT, Romanian BERT) are what run 1 already uses, and we treat them as allowed. Only extra *labelled data* is ruled out.
- That includes the public PolERo release (`github.com/gabriel-stefan/polero`). Its test split is most likely our test set (both have n=296), so it almost certainly holds the test gold labels. **Don't download or use its data files.**

## LLM APIs vs. open models
- **External LLM APIs are not allowed** (confirmed by the team, 21 Nov).
- So **task 3 as written (an API classifier) is dropped.**
- **Open question:** may we run an open-weight LLM ourselves (for example Qwen3.6-35B-A3B or Gemma-4-31B-it on a Kaggle GPU)? "No API" doesn't settle that. In the paper, few-shot Qwen3.6-35B-A3B got the best Romanian evasion score (.682 vs .545 for the best encoder trained on our data alone). The catch: a ~35B model on Kaggle T4s needs 4-bit quantisation, and there is little time left. Only pursue it if it's explicitly allowed and someone can own it.

## Other rules to check while on the page
- **TODO:** team size and whether teams can merge; hand-labelling or other use of the test set (we never do this anyway); a requirement to submit code or a system description.
