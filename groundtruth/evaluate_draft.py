"""Evaluate frozen survey runs against draft target qrels with clustered bootstrap."""
import csv
import json
import random
import argparse
from collections import defaultdict
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent.parent
GEN=ROOT/'groundtruth/generated'
RUNS=ROOT/'SearchEngine/data/rebuild_v1/survey_runs_UNJUDGED.jsonl'
SYSTEMS=['BM25','SBERT','CLIP_Text','CLIP_Image','V1_PT2']
BOOTSTRAP=10000; SEED=4204

def values(ranking,target):
    rank=ranking.index(target)+1 if target in ranking else 0
    return {'success_at_1':float(rank==1),'success_at_5':float(0<rank<=5),'mrr_at_5':1/rank if 0<rank<=5 else 0.0}
def mean(items,metric): return float(np.mean([x[metric] for x in items]))
def ci_by_person(records, metric):
    by_person=defaultdict(list)
    for r in records: by_person[r['participant_id']].append(r)
    people=list(by_person); rng=random.Random(SEED); samples=[]
    for _ in range(BOOTSTRAP):
        chosen=[by_person[rng.choice(people)] for _ in people]
        rows=[x for group in chosen for x in group]
        samples.append(mean(rows,metric))
    return [round(float(np.quantile(samples,x)),4) for x in [.025,.975]]
def paired_delta_ci(records_a, records_b, metric):
    by_person_a=defaultdict(list); by_person_b=defaultdict(list)
    for r in records_a: by_person_a[r['participant_id']].append(r)
    for r in records_b: by_person_b[r['participant_id']].append(r)
    people=sorted(by_person_a); rng=random.Random(SEED); samples=[]
    for _ in range(BOOTSTRAP):
        chosen=[rng.choice(people) for _ in people]
        a=[x for person in chosen for x in by_person_a[person]]
        b=[x for person in chosen for x in by_person_b[person]]
        samples.append(mean(a,metric)-mean(b,metric))
    return [round(float(np.quantile(samples,x)),4) for x in [.025,.975]]
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--split', default='groundtruth_split_DRAFT.csv')
    parser.add_argument('--output', default='preliminary_metrics_DRAFT.json')
    parser.add_argument('--label-status', default='preliminary_draft_evaluation')
    parser.add_argument('--support', default='',
                        help='Optional comma-separated content_support labels, e.g. supported')
    args=parser.parse_args()
    rows=[r for r in csv.DictReader((GEN/args.split).open(encoding='utf-8-sig')) if r['decision']=='keep']
    if args.support:
        allowed={value.strip() for value in args.support.split(',') if value.strip()}
        rows=[r for r in rows if r['content_support'] in allowed]
    predictions={}
    for line in RUNS.read_text(encoding='utf-8').splitlines():
        r=json.loads(line); predictions[(r['query_id'],r['system'])]=r['ranked_movies']
    report={'status':args.label_status,'query_count':len(rows),'content_support_filter':args.support or 'all kept queries',
            'note':'Targets use the supplied user-review decisions; do not describe them as independently adjudicated without a second independent reviewer.'}
    for subset in ['dev','test','all']:
        selected=[r for r in rows if subset=='all' or r['split']==subset]
        section={'queries':len(selected),'participants':len({r['participant_id'] for r in selected}),'systems':{}}
        for system in SYSTEMS:
            scored=[{**r,**values(predictions[(r['query_id'],system)],r['self_reported_title'])} for r in selected]
            section['systems'][system]={m:round(mean(scored,m),4) for m in ['success_at_1','success_at_5','mrr_at_5']}
            if subset=='test': section['systems'][system]['mrr_at_5_ci95_clustered']=ci_by_person(scored,'mrr_at_5')
        report[subset]=section
    test_rows=[r for r in rows if r['split']=='test']
    all_scores={}
    for system in SYSTEMS:
        all_scores[system]=[{**r,**values(predictions[(r['query_id'],system)],r['self_reported_title'])} for r in test_rows]
    report['test_paired_mrr_at_5_delta_ci95_clustered']={
        'V1_PT2_minus_BM25':paired_delta_ci(all_scores['V1_PT2'],all_scores['BM25'],'mrr_at_5'),
        'V1_PT2_minus_SBERT':paired_delta_ci(all_scores['V1_PT2'],all_scores['SBERT'],'mrr_at_5')}
    # Group results are presented only for the draft test subset.
    groups={}
    for group in ['1','2','3']:
        selected=[r for r in rows if r['split']=='test' and r['prompt_group']==group]
        groups[group]={'queries':len(selected),'systems':{}}
        for system in SYSTEMS:
            scored=[values(predictions[(r['query_id'],system)],r['self_reported_title']) for r in selected]
            groups[group]['systems'][system]={m:round(mean(scored,m),4) for m in ['success_at_1','success_at_5','mrr_at_5']}
    report['test_by_prompt_group']=groups
    (GEN/args.output).write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(report['test'],ensure_ascii=False,indent=2))
if __name__=='__main__':main()
