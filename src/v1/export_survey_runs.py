"""Export V1 rankings for survey queries; it never creates relevance labels.

Run: .venv/bin/python -m src.v1.export_survey_runs
"""
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
QUERY_FILE = ROOT.parent / 'groundtruth/generated/queries_DRAFT.csv'
OUTPUT = ROOT / 'data/rebuild_v1/survey_runs_UNJUDGED.jsonl'

def main():
    from src.v1.search_engine import MovieSearchEngine

    with QUERY_FILE.open(encoding='utf-8-sig') as f:
        queries = list(csv.DictReader(f))
    if len(queries) != 183:
        raise ValueError(f'Expected 183 survey queries, found {len(queries)}')

    engine = MovieSearchEngine()
    methods = {
        'BM25': engine.search_bm25_only,
        'SBERT': engine.search_sbert_only,
        'CLIP_Text': engine.search_clip_text_only,
        'CLIP_Image': engine.search_image_only,
        'V1_PT2': engine.search,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open('w', encoding='utf-8') as output:
        for i, query in enumerate(queries, 1):
            for name, search in methods.items():
                ranking = search(query['query_raw'], top_n=10)
                output.write(json.dumps({
                    'query_id': query['query_id'],
                    'query_raw': query['query_raw'],
                    'prompt_group': query['prompt_group'],
                    'system': name,
                    'ranked_movies': ranking,
                    'judgment_status': 'unjudged',
                }, ensure_ascii=False) + '\n')
            output.flush()
            print(f'Exported {i}/{len(queries)} queries', flush=True)
    print(f'Wrote {len(queries) * len(methods)} unjudged runs to {OUTPUT}')

if __name__ == '__main__':
    main()
