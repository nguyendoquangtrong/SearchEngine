"""Create a reproducible participant-level dev/test split for draft qrels."""
import csv
import json
import random
import argparse
from collections import defaultdict
from pathlib import Path

ROOT=Path(__file__).resolve().parent.parent
GEN=ROOT/'groundtruth/generated'
SEED=4204
DEV_PARTICIPANTS=15

def find(parent, x):
    while parent[x]!=x:
        parent[x]=parent[parent[x]]; x=parent[x]
    return x
def union(parent,a,b):
    a,b=find(parent,a),find(parent,b)
    if a!=b: parent[b]=a

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--input', default='groundtruth_audit_DRAFT.csv')
    parser.add_argument('--output-prefix', default='groundtruth_split_DRAFT')
    args=parser.parse_args()
    rows=list(csv.DictReader((GEN/args.input).open(encoding='utf-8-sig')))
    included=[r for r in rows if r['decision']=='keep']
    people=sorted({r['participant_id'] for r in included})
    parent={p:p for p in people}
    clusters=defaultdict(set)
    for row in included:
        if row['duplicate_cluster']: clusters[row['duplicate_cluster']].add(row['participant_id'])
    for members in clusters.values():
        members=sorted(members)
        for person in members[1:]: union(parent,members[0],person)
    components=defaultdict(list)
    for person in people: components[find(parent,person)].append(person)
    components=list(components.values())
    rng=random.Random(SEED); rng.shuffle(components)
    # Select whole duplicate-linked components; exact 15 users is feasible here.
    chosen=[]; n=0
    for component in components:
        if n+len(component)<=DEV_PARTICIPANTS:
            chosen.extend(component); n+=len(component)
        if n==DEV_PARTICIPANTS: break
    if n!=DEV_PARTICIPANTS: raise ValueError(f'Could not form {DEV_PARTICIPANTS}-person dev split')
    dev=set(chosen)
    for row in rows:
        row['split']='dev' if row['participant_id'] in dev else 'test'
    with (GEN/f'{args.output_prefix}.csv').open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    summary={'status':'draft_split','seed':SEED,'unit':'participant',
        'duplicate_clusters_kept_together':{k:sorted(v) for k,v in clusters.items()},
        'dev_participants':sorted(dev),'test_participants':sorted(set(people)-dev),
        'dev_queries':sum(r['decision']=='keep' and r['split']=='dev' for r in rows),
        'test_queries':sum(r['decision']=='keep' and r['split']=='test' for r in rows),
        'excluded_queries':sum(r['decision']=='exclude' for r in rows)}
    (GEN/f'{args.output_prefix}_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(summary,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
