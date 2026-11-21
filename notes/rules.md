# Competition rules

**Status:** all questions answered by Rares in issue #2 (21 Nov). The competition page doesn't name the metric, so the formula below is our working assumption.

## Evaluation metric
- **Each subtask is worth 50 points.** The task text doesn't name the metric. We optimise **macro-F1 per subtask**, as in the PolERo paper (Section 3.1) and SemEval-2026 Task 6, and assume score = 50 × clarity macro-F1 + 50 × evasion macro-F1.
- **The 50/50 formula is confirmed by a probe** (`submissions/LEADERBOARD.md`, 21 Nov ~17:35).
  - v3 kept v1's evasion answers exactly and changed only 41 clarity answers. Its partial score rose from 56.33 to 59.63, so **clarity counts**.
  - v1's partial score of 56.33 fits 50 × clarity F1 + 50 × evasion F1: OOF predicts about 52.9, and the band for a partial board is about 46–59.
- **The earlier "41.93" was the platform's rescaled grade, not the score.** It's a linear rescale, about 1.98 grade points per score point. The "evasion only" reading in the first version of `ensemble/leaderboard_noise.md` is therefore wrong.
- **Selection rule (Rares, after the probe):**
  - choose on **clarity + evasion** (`select_ensemble.py --objective sum`), with the bar at a nested **sum** gain of +0.02 over v1;
  - clarity and evasion answers may come from **different ensembles** (`split_submission.py`), each chosen on its own OOF score.
- **Submission format:** as `common.write_submission` / `evaluate.py` write it. One CSV with columns `subtaskID, datapointID, answer`, in `data/sample_output.csv` row order. Rows with `subtaskID=1` hold the clarity label and rows with `subtaskID=2` the evasion label, for each of the 296 test rows (592 rows). The uploaded file is `submissions/vN/polero_vN.py`, built by `make_submission_py.py`.
- **Public / private split:** the visible leaderboard is a *partial* one, scored on a fixed subset of the 296 test rows. Full scores come only after the deadline, and only for the 2 finals. Treat it as noisy and don't tune against it. For two similar submissions, a gap under about ±5–7 points is noise (`ensemble/leaderboard_noise.md`, section 4, 50/50 columns).

## Deadline
- **Sun 22 Nov 2026, 09:00 Romania time** (EET, UTC+2).
- Internal hard stop on experiments: **Sun 22 Nov, 06:00**, moved from Sat 23:30. Finals get picked 06:00–08:00.

## Submission limit
- **30 submissions in total**, and we pick **2 final** ones. Rares makes every submission.

## External data (English QEvasion corpus)
- **Not allowed** (confirmed by the team, 21 Nov).
- So **task 2 is dropped.** No QEvasion, no EN+RO joint training (the paper's best fine-tuned evasion setup, .570), and no machine-translated English data (EN2RO).
- Pretrained model weights (XLM-R, RoBERT, Romanian BERT) are what run 1 already uses, and we treat them as allowed. Only extra *labelled data* is ruled out.
- That includes the public PolERo release (`github.com/gabriel-stefan/polero`). Its test split is most likely our test set (both have n=296), so it almost certainly holds the test gold labels. **Don't download or use its data files.**

## LLM APIs vs. open models
- **External LLM APIs are not allowed** (confirmed by the team, 21 Nov).
- So **task 3 as written (an API classifier) is dropped.**
- **Open-weight LLMs run by us:** first off (Rares, 21 Nov). **Reopened at ~17:55 the same day** for a train-only test (task 12). The model runs inside our own free Colab/Kaggle notebook, so it's neither an external API nor external data. For the record: in the paper, few-shot Qwen3.6-35B-A3B got the best Romanian evasion score (.682 vs .545 for the best encoder).

## Other rules
- **Team size is 3:** Rares (leader), TreiPaDus, and a third colleague who joins later.
- **No code or system description is required.**
- **Test data:** never read, inspect or hand-label `test_data.csv` answers. Test only goes through a finished model.
