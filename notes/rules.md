# Competition rules

**Status:** all questions answered by Rares in issue #2 (21 Nov). The competition page doesn't name the metric, so the formula below is our working assumption.

## Evaluation metric
- **Each subtask is worth 50 points.** The task text doesn't name the metric. We optimise **macro-F1 per subtask**, as in the PolERo paper (Section 3.1) and SemEval-2026 Task 6, and assume score = 50 × clarity macro-F1 + 50 × evasion macro-F1.
- **The leaderboard doesn't settle the formula.** It shows two identical score columns. v1 scored **41.93** (rank 20 of 35). The real top is about 54.2, because #1's exact 100 is almost certainly the leaked public labels.
- **41.93 looks like evasion only.** Under the 50/50 formula, OOF predicts v1 at 52.5 ± 3.5, and 41.93 would be a 1-in-1,700 result (`ensemble/leaderboard_noise.md`). It fits evasion only: 100 × evasion macro-F1 is expected at 38.9 (31.1–46.2).
- **Working rule (Rares):** evasion is at least half the score under either formula and has the most room to gain. So **rank models by evasion OOF macro-F1 first**, and accept a clarity drop of at most 0.01.
- **Submission format:** as `common.write_submission` / `evaluate.py` write it. One CSV with columns `subtaskID, datapointID, answer`, in `data/sample_output.csv` row order. Rows with `subtaskID=1` hold the clarity label and rows with `subtaskID=2` the evasion label, for each of the 296 test rows (592 rows). The uploaded file is `submissions/vN/polero_vN.py`, built by `make_submission_py.py`.
- **Public / private split:** not stated. Treat the leaderboard as noisy either way and don't tune against it. With 296 rows, a gap under about ±4–6 points is noise for two similar submissions (`ensemble/leaderboard_noise.md`).

## Deadline
- **Sun 22 Nov 2026, 09:00 Romania time** (EET, UTC+2).
- Internal hard stop on experiments: Sat 21 Nov, 23:30.

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
