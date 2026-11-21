# Competition rules

**Status: incomplete.** I couldn't find the competition page with a web search (no public Kaggle, Codabench or CodaLab page for PolERo turned up). Every line marked **TODO** has to be checked on the competition page by someone who has the link. What's filled in comes from our own repo and the PolERo paper, with the source noted.

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
- **TODO: is external data allowed?** Task 2 depends on this.
- Licence check: QEvasion / CLARITY (Thomas et al., 2024; `ailsntua/QEvasion` on Hugging Face) is a public research dataset. The PolERo paper trains EN+RO jointly with it.
- Related: the PolERo dataset itself is public (CC BY 4.0, `github.com/gabriel-stefan/polero`), and its test split is most likely our test set (both have n=296). Even if external data is allowed, **don't use the PolERo release's data files**, because they almost certainly contain the test gold labels.

## LLM APIs vs. open models
- **TODO: are external LLM APIs (OpenAI, Anthropic, Google, …) allowed, or only open-weight models? Is there a model-size or compute limit?** Task 3 depends on this.
- For context: in the paper, open-weight Qwen3.6-35B-A3B (few-shot) is as good as or better than GPT-5.4 on Romanian evasion (.682 vs .666). If only open models are allowed, task 3 can still run Qwen3.6-35B-A3B or Gemma-4-31B-it, for example on a Kaggle GPU.

## Other rules to check while on the page
- **TODO:** team size and whether teams can merge; hand-labelling or other use of the test set (we never do this anyway); a requirement to submit code or a system description.
