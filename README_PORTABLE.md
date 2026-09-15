# Portable V1 evaluation package

This branch contains the V1 movie-search code, reviewed evaluation artifacts, and `docs/ground_truth.docx`.

## Included evaluation artifacts

- `groundtruth/generated/qrels.tsv`: movie-level relevance labels for 174 kept survey queries.
- `groundtruth/generated/audit.csv`: all 183 queries and their user-review decision.
- `groundtruth/generated/split.csv`: participant-level dev/test split (42/132 kept queries).
- `groundtruth/generated/metrics_all.json`: metrics on all kept queries.
- `groundtruth/generated/metrics_supported.json`: supported-only sensitivity analysis.

## Run on another machine

```bash
cd SearchEngine
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-extraction.txt
export NLTK_DATA=.cache/nltk_data
docker compose up -d
python -m src.v1.db_builder
python src/v1/main.py
```

The first run needs network access to download the Hugging Face models if they are not already cached. `data/DataMovie/` and `data/movies_data_english_clean.json` are included so the vector database can be rebuilt locally. Existing frozen results are in `data/rebuild_v1/survey_runs_UNJUDGED.jsonl`.

## Reproduce evaluation reports

From the repository root:

```bash
python3 groundtruth/evaluate_draft.py --split groundtruth/generated/split.csv --output metrics_all.json --label-status user_reviewed_all_kept
python3 groundtruth/evaluate_draft.py --split groundtruth/generated/split.csv --support supported --output metrics_supported.json --label-status user_reviewed_supported_only
```

The labels are user-reviewed target labels; independent double adjudication would still be needed to claim independently adjudicated ground truth.
