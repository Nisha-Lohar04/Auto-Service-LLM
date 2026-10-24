# AutoService-LLM

A dataset pipeline for building and evaluating a question-answering system for automotive service and maintenance.

## Dataset

- `data/raw/` — Source passages, source catalog, and initial Q&A data.
- `data/batches/` — LLM-drafted Q&A grouped by automotive topics.
- `data/overrides.json` — Reviewer corrections applied during dataset preparation.
- `data/processed/` — Training, validation, and test splits.
- `data/review_sample.md` and `data/review_test.md` — Review materials.

## Pipeline

`src/build_dataset.py` merges the data, applies overrides, removes duplicates, creates group-based splits, and checks for data leakage.

Run the pipeline:

```powershell
python src/build_dataset.py