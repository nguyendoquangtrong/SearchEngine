"""Run Drive import -> strict full-corpus V1 index -> survey candidate export.

This produces retrieval candidates, never automatic ground truth labels.
"""
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import argparse

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT/'data/rebuild_v1'

def state(stage, **details):
    REPORT.mkdir(parents=True,exist_ok=True)
    (REPORT/'pipeline_status.json').write_text(json.dumps(
        {'stage':stage,'time_unix':time.time(),**details},ensure_ascii=False,indent=2))
    print(stage,details,flush=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--existing-local-data', action='store_true',
                        help='Use data/DataMovie as-is; do not access Google Drive.')
    args = parser.parse_args()
    os.chdir(ROOT)
    # Keep model and tokenizer downloads within this project's writable cache.
    os.environ.setdefault('NLTK_DATA',str(ROOT/'.cache/nltk_data'))
    try:
        if args.existing_local_data:
            state('using_existing_local_datamovie')
        else:
            state('importing_existing_drive_data')
            subprocess.run([sys.executable,'-m','src.v1.import_drive','--reuse-list'],check=True)
        state('auditing_original_corpus')
        subprocess.run([sys.executable,'-m','src.v1.audit_corpus'],check=True)
        state('building_v1')
        subprocess.run([sys.executable,'-m','src.v1.db_builder'],check=True)
        state('exporting_survey_candidates')
        from src.v1.search_engine import MovieSearchEngine
        engine = MovieSearchEngine()
        query_file = ROOT.parent/'groundtruth/generated/queries_DRAFT.csv'
        with query_file.open(encoding='utf-8-sig') as f:
            queries = list(csv.DictReader(f))
        methods = {'BM25':engine.search_bm25_only,'SBERT':engine.search_sbert_only,
                   'CLIP_Text':engine.search_clip_text_only,'CLIP_Image':engine.search_image_only,
                   'V1_PT2':engine.search}
        with (REPORT/'survey_runs_UNJUDGED.jsonl').open('w',encoding='utf-8') as output:
            for i,q in enumerate(queries,1):
                for name,method in methods.items():
                    ranked = method(q['query_raw'],top_n=10)
                    output.write(json.dumps({'query_id':q['query_id'],'query_raw':q['query_raw'],
                        'system':name,'ranked_movies':ranked,'judgment_status':'unjudged'},ensure_ascii=False)+'\n')
                    output.flush()
                print(f'Survey candidates {i}/{len(queries)}',flush=True)
        state('candidates_ready_for_human_annotation',queries=len(queries),verified_labels=0)
    except Exception as exc:
        state('failed',error=str(exc))
        raise

if __name__ == '__main__': main()
