"""Create reproducible proposed QC evidence for survey queries.

This script does not create final qrels. Its labels are evidence-based suggestions
that must remain distinguishable from independently adjudicated human labels.
"""
import csv
import difflib
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parent.parent
ENGINE = ROOT / 'SearchEngine'
OUT = ROOT / 'groundtruth' / 'generated'
QUERY_FILE = OUT / 'queries_DRAFT.csv'
RUN_FILE = ENGINE / 'data/rebuild_v1/survey_runs_UNJUDGED.jsonl'
MOVIES_FILE = ENGINE / 'data/movies_data_english_clean.json'
MEDIA = ENGINE / 'data/DataMovie'

STOP = {'a','an','the','and','or','to','of','in','on','with','for','is','are','i','you','it','that','this','my','me'}

def movie_key(value):
    return re.sub(r'[^a-z0-9]', '', value.lower())

def tokens(value):
    return [x for x in re.findall(r"[a-z0-9]+", value.lower()) if x not in STOP]

def timestamp_text(value):
    return re.sub(r'^\[[^]]*\]\s*:?\s*', '', value).strip()

def f1(left, right):
    a, b = set(tokens(left)), set(tokens(right))
    if not a or not b: return 0.0
    shared = len(a & b)
    return 2 * shared / (len(a) + len(b))

def quote_evidence(query, lines):
    best = (0.0, '', '')
    clean_query = ' '.join(tokens(query))
    for start in range(len(lines)):
        for width in range(1, 5):
            excerpt = ' '.join(timestamp_text(line) for line in lines[start:start+width])
            normalized = ' '.join(tokens(excerpt))
            score = max(f1(query, excerpt), difflib.SequenceMatcher(None, clean_query, normalized).ratio())
            if score > best[0]: best = (score, excerpt, f'lines {start+1}-{start+width}')
    return best

def read_runs():
    ranks = {}
    for line in RUN_FILE.read_text(encoding='utf-8').splitlines():
        item = json.loads(line)
        ranks[(item['query_id'], item['system'])] = item['ranked_movies']
    return ranks

def rank_of(target, ranking):
    try: return ranking.index(target) + 1
    except ValueError: return 0

def wb_output(records, path):
    wb = Workbook()
    ws = wb.active
    ws.title = 'QC_proposals'
    headers = list(records[0])
    ws.append(headers)
    for row in records: ws.append([row.get(h, '') for h in headers])
    ws.freeze_panes = 'A2'
    ws.auto_filter.ref = ws.dimensions
    for cell in ws[1]:
        cell.font = Font(color='FFFFFF', bold=True)
        cell.fill = PatternFill('solid', fgColor='17365D')
    for col in ws.columns:
        title = str(col[0].value)
        ws.column_dimensions[col[0].column_letter].width = 58 if any(x in title for x in ['query','evidence','reason']) else 22
        for cell in col[1:]: cell.alignment = Alignment(vertical='top', wrap_text=True)
    readme = wb.create_sheet('README', 0)
    readme.append(['Item','Meaning'])
    readme.append(['Status','PROPOSED QC ONLY. Do not convert this file directly to final qrels.'])
    readme.append(['Target','self_reported_target is copied from the survey and has only corpus-presence validation here.'])
    readme.append(['Evidence','Quote support uses fuzzy transcript matching. Plot/visual scores use locally cached SBERT against summaries/captions.'])
    readme.append(['Review required','Required for all rows marked review_required, non_response_candidate, or title_leakage.'])
    readme.append(['Prediction ranks','These are diagnostic evidence only; they did not determine the proposed target.'])
    for cell in readme[1]:
        cell.font = Font(color='FFFFFF', bold=True)
        cell.fill = PatternFill('solid', fgColor='17365D')
    readme.column_dimensions['A'].width = 25
    readme.column_dimensions['B'].width = 110
    wb.save(path)

def main():
    queries = list(csv.DictReader(QUERY_FILE.open(encoding='utf-8-sig')))
    movies = json.loads(MOVIES_FILE.read_text())
    titles = [m['title'] for m in movies]
    folders = {movie_key(p.name): p for p in MEDIA.iterdir() if p.is_dir()}
    summaries = [m.get('content','') for m in movies]
    captions, caption_movies = [], []
    transcripts = {}
    for title in titles:
        folder = folders[movie_key(title)]
        scripts = list((folder/'script').glob('*.txt'))
        transcripts[title] = scripts[0].read_text(encoding='utf-8-sig').splitlines() if scripts else []
        caption_files = list(folder.glob('*_captions.txt'))
        for file in caption_files:
            for line in file.read_text(encoding='utf-8-sig', errors='replace').splitlines():
                if '|' in line:
                    captions.append(line.split('|',1)[1].strip())
                    caption_movies.append(title)
    model = SentenceTransformer('all-MiniLM-L6-v2', local_files_only=True)
    summary_vectors = np.asarray(model.encode(summaries, normalize_embeddings=True, show_progress_bar=True), dtype=np.float32)
    caption_vectors = np.asarray(model.encode(captions, normalize_embeddings=True, batch_size=64, show_progress_bar=True), dtype=np.float32)
    ranks = read_runs()
    records = []
    for index, q in enumerate(queries, 1):
        query = q['query_raw']; target = q['self_reported_title']; group = q['prompt_group']
        target_present = target in titles
        vector = np.asarray(model.encode(query, normalize_embeddings=True), dtype=np.float32)
        # einsum avoids an Apple/NumPy BLAS warning observed with 2-D @ 1-D
        # despite finite, normalized float32 input vectors.
        summary_scores = np.einsum('ij,j->i', summary_vectors, vector)
        summary_order = np.argsort(-summary_scores)
        summary_rank = int(np.where(summary_order == titles.index(target))[0][0]) + 1 if target_present else 0
        caption_scores = np.einsum('ij,j->i', caption_vectors, vector)
        caption_best = defaultdict(lambda: -1.0)
        for movie, score in zip(caption_movies, caption_scores): caption_best[movie] = max(caption_best[movie], float(score))
        caption_order = sorted(titles, key=lambda title: caption_best[title], reverse=True)
        caption_rank = caption_order.index(target) + 1 if target_present else 0
        quote_score, quote_text, quote_ref = quote_evidence(query, transcripts[target]) if target_present and group == '1' else (0.0,'','')
        flags = [x for x in q['screening_flags'].split(';') if x]
        low = query.casefold().strip()
        if low in {'i can’t remember.','i can\'t remember.',''}:
            decision, reason = 'non_response_candidate', 'Query does not provide retrievable content.'
        elif 'explicit_target_title' in flags:
            decision, reason = 'review_required', 'Query contains the target title and may inflate retrieval.'
        elif group == '1' and quote_score >= .78:
            decision, reason = 'provisionally_usable', 'Strong target-transcript lexical/fuzzy support.'
        elif group == '1':
            decision, reason = 'review_required', 'Quote did not obtain strong transcript support; may be paraphrased, inaccurate, or absent.'
        elif group == '2' and summary_rank <= 10:
            decision, reason = 'provisionally_usable', 'Target synopsis is among top-10 semantic matches; verify factual support.'
        elif group == '3' and caption_rank <= 10:
            decision, reason = 'provisionally_usable', 'Target captions are among top-10 semantic visual-description matches; verify frame evidence.'
        else:
            decision, reason = 'review_required', 'Automatic evidence is weak; preserve query but check target support manually.'
        record = {
            'query_id':q['query_id'], 'participant_id':q['participant_id'], 'prompt_group':group,
            'query_raw':query, 'self_reported_target':target, 'target_in_62_movie_corpus':target_present,
            'proposed_qc_status':decision, 'proposed_reason':reason, 'screening_flags':';'.join(flags),
            'quote_support_score':round(quote_score,3), 'quote_evidence_ref':quote_ref,
            'quote_evidence_excerpt':quote_text, 'target_summary_semantic_rank':summary_rank,
            'target_caption_semantic_rank':caption_rank,
        }
        for system in ['BM25','SBERT','CLIP_Text','CLIP_Image','V1_PT2']:
            record[f'{system}_target_rank_top10'] = rank_of(target, ranks[(q['query_id'],system)])
        records.append(record)
        print(f'QC {index}/{len(queries)}', flush=True)
    fields = list(records[0])
    with (OUT/'query_qc_PROPOSED.csv').open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(records)
    wb_output(records, OUT/'query_qc_PROPOSED.xlsx')
    stats=defaultdict(int)
    for record in records: stats[record['proposed_qc_status']]+=1
    (OUT/'query_qc_summary.json').write_text(json.dumps({'queries':len(records),'status_counts':stats,'note':'Proposed QC labels are not final ground truth.'},ensure_ascii=False,indent=2))
    print(json.dumps(dict(stats),ensure_ascii=False))

if __name__ == '__main__': main()
